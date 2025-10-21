from typing import List, Optional, Type, Union
import warnings
import copy
from functools import partial
import torch as th
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from saferl.common.policies import ActorCriticWithCostPolicy
from saferl.common.trust_region_algorithm import SafeTrustRegionAlgorithm
from stable_baselines3.common.buffers import RolloutBufferSamples
from itertools import chain

class CPO(SafeTrustRegionAlgorithm):
    
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticWithCostPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 15,
        n_critic_updates: int = 10,
        failure_prediction_updates: int = 10,
        failure_prediction_horizon: int = 10,
        cost_bonus_weight: float = 0.0,
        kl_div_step_size: float = 0.01,
        eps: float = 1e-8,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        sub_sampling_factor: int = 1,
        cost_constraint: Union[float, List] = [2, None],
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        **kwargs,
    ):
        super(CPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            ent_coef=0.0,  # entropy bonus is not used by CPO
            vf_coef=0.0,  # value function is optimized separately
            max_grad_norm=0.0,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            cost_constraint=cost_constraint,
            state_safe_horizon=failure_prediction_horizon,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,  
            seed=seed,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            **kwargs,
        )

        # assert len(cost_constraint) == self.cost_dim[0], "cost_constraint must be same as the cost_dim of the environment"
        assert self.cost_dim == 1, "CPO currently only supports one dimensional costs"
        # for the 1 dimensional case an analytical solution exists, otherwise an inner optimization problem has to be solved
        # the code is prepared for this case, only the inner optimization problem has to be implemented if needed
        self.kl_div_step_size = kl_div_step_size
        self.eps = eps
        if batch_size > n_steps * self.env.num_envs:
            warnings.warn(
                f"batch_size ({batch_size}) is bigger than n_steps * n_envs ({n_steps * self.env.num_envs}). "
                "Setting batch_size to n_steps * n_envs = " f"{n_steps * self.env.num_envs}"
            )
            batch_size = n_steps * self.env.num_envs


        self.normalize_advantages = normalize_advantage
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            if normalize_advantage:
                assert buffer_size > 1, (
                    "`n_steps * n_envs` must be greater than 1. "
                    f"Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
                )
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )


        self.batch_size = batch_size
        self.n_critic_updates = n_critic_updates
        # Parameters for conjugate gradient
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        # Parameters for line search
        self.line_search_shrinking_factor = line_search_shrinking_factor
        self.line_search_max_iter = line_search_max_iter
        # Parameters for the constraint
        self.sub_sampling_factor = sub_sampling_factor
        # rpo perturbance
        self.use_rpo_perturb = False
        if "rpo_perturbance_alpha" in kwargs["policy_kwargs"]:
            self.use_rpo_perturb = kwargs["policy_kwargs"]["rpo_perturbance_alpha"] > 0

        self.problem_feasibility_tracker = {-1:0, 0:0, 1:0, 2:0, 3:0, 4:0}

        self.problem_feasibility = -1
        self.c = 0
        self.rollout_avg_ep_length = 1
        self.cost_objective = None
        self.c = None

        self.cost_bonus_weight = cost_bonus_weight
        if cost_bonus_weight > self.eps:
            print(f"Using failure prediction network. Cost bonus weight: {cost_bonus_weight}, Failure prediction horizon: {failure_prediction_horizon}")
            self.failure_prediction_updates = failure_prediction_updates
            self.failure_prediction_horizon = failure_prediction_horizon
            self.failure_prediction_network = FailurePredictionNetwork(np.sum(self.observation_space.shape[0]), 32).to(self.device)
            # adam optimizer for failure prediction network
            self.failure_prediction_network_optimizer = th.optim.Adam(self.failure_prediction_network.parameters(), lr=learning_rate)

            def cost_modification_function(**rollout_data):
                # get observation, convert to tensor and move to device
                with th.no_grad():
                    observation = th.tensor(rollout_data["observation"], dtype=th.float32, device=self.device)
                    new_cost = rollout_data["cost"] + self.cost_bonus_weight * self.failure_prediction_network(observation).cpu().numpy()
                return new_cost
            
            self.rollout_buffer.register_cost_modification_function("FailurePredictionAddition", cost_modification_function)

    def train(self) -> None:
        """
        Use current rollout data to update the policy with CPO
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        critic_losses = []
        cost_critic_losses = []
        failure_prediction_losses = []
        
        # check if rollout buffer has episode stats
        if "ep_returns" not in self.rollout_buffer.episode_stats:
            warnings.warn("Rollout buffer does not contain episode stats. Skipping update.")
            return
        
        # rollout_avg_ep_return = np.array(self.rollout_buffer.episode_stats["ep_returns"]).mean()

        # rollout_avg_ep_cost_return = np.array(self.rollout_buffer.episode_stats["ep_cost_returns_undiscounted"]).mean()
        rollout_avg_ep_cost_return = np.mean([np.sum(ep_info["cost"]) for ep_info in self.ep_info_buffer])
        # self.rollout_avg_ep_length = np.array(self.rollout_buffer.episode_stats["ep_lengths"]).mean()
        self.rollout_avg_ep_length = np.mean([ep_info["l"] for ep_info in self.ep_info_buffer])
        
        # get all of the samples from the rollout buffer
        for rollout_data in self.rollout_buffer.get(None):

            # if using rpo perturbance then old and new policy distributio need to be perturbed the same way because CPO linearizes the KL divergence
            with th.no_grad():
                old_policy_dist = copy.deepcopy(self.policy.get_distribution(rollout_data.observations, rpo_perturb=self.use_rpo_perturb, recalc_rpo_perturb=self.use_rpo_perturb))

            policy_dist = self.policy.get_distribution(rollout_data.observations, rpo_perturb=self.use_rpo_perturb)
            log_prob = policy_dist.log_prob(rollout_data.actions)

            # calculate mean kl divergence
            kl_div = kl_divergence(old_policy_dist, policy_dist).mean()

            advantages = rollout_data.advantages
            cost_advantages = rollout_data.cost_advantages
            # normalize advantages
            if self.normalize_advantages:
                advantages= (advantages - advantages.mean()) / (advantages.std() + self.eps)
                for i in range(self.cost_dim):
                    cost_advantages[..., i] = (cost_advantages[..., i] - cost_advantages[..., i].mean()) / (cost_advantages[..., i].std() + self.eps)

            # build the policy objective = expectation of advantage
            # policy loss = negative policy objective
            policy_loss = -self.get_policy_objective(advantages, log_prob, rollout_data.old_log_prob)
            self.cost_objective = self.get_cost_objective(cost_advantages, log_prob, rollout_data.old_log_prob)

            ########################################
            # Gathering gradients and collecting terms with the fisher information matrix
            ########################################

            # set grad to zero
            self.policy.optimizer.zero_grad()
            kl_div_gradient, policy_gradient_g, actor_params, grad_shape = self.calculate_kl_div_grad_and_policy_grad(kl_div, policy_loss)

            # calculate derivative of the policy objective, cost objective, and kl divergence
            # already rescaled in cost objective calculation
            cost_gradient_b = flat_grad(self.cost_objective, actor_params, retain_graph=True)
            # use conjugate gradient to solve for the hessian approximation
            cg_function = partial(self.matrix_vector_product_derivatives, kl_div_gradient, actor_params)      
            # round to 6 decimals
            cost_gradient_b = th.round(cost_gradient_b, decimals=6)
            policy_gradient_g = th.round(policy_gradient_g, decimals=6)
            inv_H_g = conjugate_gradient_solver(cg_function, policy_gradient_g, self.cg_max_steps, self.cg_damping)
            # recalculate g and b from hessian approximation
            g = self.matrix_vector_product_derivatives(kl_div_gradient, actor_params, inv_H_g)
            # calculate inv_H_b and s
            norm_b = th.sqrt(th.dot(cost_gradient_b, cost_gradient_b))
            # usind the normalized cost gradient as the direction for the conjugate gradient can be beneficial for following reasons:
            # 1. numerical stability
            # 2. balancing different scales in parameter space (if not a large scaled dimension can dominate the conjugate gradient)
            unit_b = cost_gradient_b / (norm_b + self.eps)
            # solve for inv_H_b with conjugate gradient using unit_b and rescaling afterwards with norm_b
            inv_H_b = norm_b * conjugate_gradient_solver(cg_function, unit_b, self.cg_max_steps, self.cg_damping)
            b = self.matrix_vector_product_derivatives(kl_div_gradient, actor_params, inv_H_b)
            
            # calculate q, r, s
            q = th.dot(g, inv_H_g)
            r = th.dot(g, inv_H_b)
            s = th.dot(inv_H_b, b)

            # calculate c
            # no rescaling for c required, as we already rescaled the cost gradient (in the cost objective calculation) which is equivalent in the following calculations
            self.c = (rollout_avg_ep_cost_return - self.cost_constraint[0])
            self.c = th.tensor(self.c, dtype=th.float32, device=self.device)

            # check geometry of problem
            self.problem_feasibility = -1
            # calculate A from cauchy-schwarz inequality
            A = q - (r**2 / (s + self.eps))
            if A < 0:
                warnings.warn("A is negative, problem is not convex. This should never happen.")
            # calculate B
            B = 2*self.kl_div_step_size - self.c**2 / s

            ########################################
            # Definition of feasibility cases
            # case -1: this should never happen
            # case 0: problem is infeasible, this is the recovery case
            # case 1: current point infeasible, trust region is partially contained in the linear constraint satisfying half space
            # case 2: current point feasible, trust region is partially contained in the linear constraint satisfying half space
            # case 3: problem is feasible and trust region is fully contained in the linear constraint satisfying half space
            # case 4: problem is feasible, but cost gradient is very small --> ignore constraint
            ########################################

            # if cost gradient is very small, ignore constraint
            if th.dot(cost_gradient_b, cost_gradient_b) < self.eps and self.c < 0:
                self.problem_feasibility = 4
            
            # check if the trust region is fully contained in the linear constraint satisfying half space
            elif B < 0 and self.c < 0:
                self.problem_feasibility = 3

            # check if the trust region is fully outside of the linear constraint satisfying half space
            # --> problem infeasible --> recovery case
            elif B < 0 and self.c >= 0:
                self.problem_feasibility = 0

            # check if the trust region is partially contained in the linear constraint satisfaction half space (intersection)
            else:
                # check if intersection of trust region and constraint satisfaction half space and current policy is feasible
                if B >= 0 and self.c < 0:
                    self.problem_feasibility = 2
                # check if intersection of trust region and constraint satisfaction half space and current policy is infeasible
                elif B >= 0 and self.c >= 0:
                    self.problem_feasibility = 1

            if self.problem_feasibility == -1:
                warnings.warn("Problem feasibility is -1. This should never happen.")

            self.problem_feasibility_tracker[self.problem_feasibility] += 1
            lambda_star = th.sqrt(q / (2*self.kl_div_step_size))
            nu_star = 0
            # calculate policy update proposal
            if self.problem_feasibility in [3,4]:
                # ignore linear constraint
                pass
            
            elif self.problem_feasibility in [1,2]:
                lambda_a = th.sqrt(A/B)
                lambda_b = th.sqrt(q/(2*self.kl_div_step_size))

                # make c numerically stable by adding epsilon
                c_eps = self.c + self.eps

                if c_eps < 0:
                    lambda_a_bounds = [0, r/c_eps]
                    lambda_b_bounds = [r/c_eps, th.inf]
                else:
                    lambda_a_bounds = [r/c_eps, th.inf]
                    lambda_b_bounds = [0, r/c_eps]
                # convert bounds to tensors
                lambda_a_bounds = th.tensor(lambda_a_bounds, dtype=th.float32, device=self.device)
                lambda_b_bounds = th.tensor(lambda_b_bounds, dtype=th.float32, device=self.device)

                # project lambda_a onto the bounds
                lambda_a = th.max(lambda_a_bounds[0], th.min(lambda_a_bounds[1], lambda_a))
                # project lambda_b onto the bounds
                lambda_b = th.max(lambda_b_bounds[0], th.min(lambda_b_bounds[1], lambda_b))

                # calculate f_a and f_b
                f_a = -0.5 * (A / (lambda_a + self.eps) + lambda_a * B) - r*c_eps / (s + self.eps)
                f_b = - 0.5 * (q / (lambda_b + self.eps) + lambda_b * 2 * self.kl_div_step_size)

                # calculate lambda_star and nu_star
                lambda_star = lambda_a if f_a > f_b else lambda_b
                nu_star = (lambda_star * c_eps - r) / (s + self.eps)
                nu_star = nu_star if nu_star > 0 else 0


            if self.problem_feasibility > 0:
                # calculate the policy update
                policy_update_direction = (inv_H_g - nu_star * inv_H_b) / (lambda_star + self.eps)
            else:
                # compute recovery policy update
                policy_update_direction = - th.sqrt(2*self.kl_div_step_size / (s + self.eps)) * inv_H_b

            self.update_with_line_search(actor_params, grad_shape, policy_update_direction, rollout_data, old_policy_dist, policy_loss, advantages, cost_advantages)

        # update critics
        for j in range(self.n_critic_updates):
            # Sample rollout data from buffer
            for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
                returns = rollout_data.returns
                cost_returns = rollout_data.cost_returns

                self.policy.optimizer.zero_grad()
                
                values_predicted = self.policy.predict_values(rollout_data.observations).squeeze()
                cost_values_predicted = self.policy.predict_cost_values(rollout_data.observations)

                critic_loss = nn.MSELoss()(values_predicted, returns)
                cost_critic_loss = nn.MSELoss()(cost_values_predicted, cost_returns)

                # critic norm
                # for param in self.policy.value_net.parameters() + self.policy.mlp_extractor.value_net.parameters():
                for param in chain(self.policy.value_net.parameters(), self.policy.mlp_extractor.value_net.parameters()):
                    critic_loss += 0.001 * th.sum(param ** 2)

                # cost critic norm
                # for param in self.policy.cost_net.parameters() + self.policy.mlp_extractor_cost_net.parameters():
                for param in chain(self.policy.cost_net.parameters(), self.policy.mlp_extractor_cost_net.parameters()):
                    cost_critic_loss += 0.001 * th.sum(param ** 2)

                critic_losses.append(critic_loss.item())
                cost_critic_losses.append(cost_critic_loss.item())

                (critic_loss + cost_critic_loss).backward()

                # grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), 40.0)

                # filter actor parameters
                for param in actor_params:
                        param.grad = None
                self.policy.optimizer.step()

        # update failure prediction network
        if self.cost_bonus_weight > 0 and self.state_safe_horizon > 0:
            for j in range(self.failure_prediction_updates):
                # Sample rollout data from buffer
                for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
                    # get observation, convert to tensor and move to device
                    observation = th.tensor(rollout_data.observations, dtype=th.float32, device=self.device)
                    # get target, convert to tensor and move to device
                    states_safe_in_horizon_count = rollout_data.states_safe_in_horizon
                    states_unsafe_in_horizon_count = self.state_safe_horizon - states_safe_in_horizon_count
                    prob_unsafe_state_within_horizon = th.tensor(states_unsafe_in_horizon_count / self.state_safe_horizon, dtype=th.float32, device=self.device)
                    # get prediction
                    prediction = self.failure_prediction_network(observation)
                    # calculate failure_prediction_loss
                    failure_prediction_loss = nn.MSELoss()(prediction, prob_unsafe_state_within_horizon)
                    # zero gradients
                    self.failure_prediction_network_optimizer.zero_grad()
                    # calculate gradients
                    failure_prediction_loss.backward()
                    # update network
                    self.failure_prediction_network_optimizer.step()

                    failure_prediction_losses.append(failure_prediction_loss.detach().cpu().item())

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        # log stats
        for i, log_std in enumerate(self.policy.log_std):
            self.logger.record(f"train/log_std_{i}", log_std.item())
        self.logger.record("train/cost_objective", np.mean(self.cost_objective.item()))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/cost_critic_loss", np.mean(cost_critic_losses))
        if self.cost_bonus_weight > 0:
            self.logger.record("train/failure_prediction_loss", np.mean(failure_prediction_losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/c_value", self.c.item())
        self.logger.record("train/cauchy_schwarz_larger0", q.item())
        self.logger.record("train/problem_geometry", B.item())
        self.logger.record("train/lagrange_lambda", lambda_star.cpu().item())
        self.logger.record("train/A", A.cpu().item())
        if isinstance(nu_star, th.Tensor):
            nu_star = nu_star.cpu().item()
        self.logger.record("train/lagrange_nu", nu_star)
        self.logger.record("train/problem_feasibility", self.problem_feasibility)
        for key, value in self.problem_feasibility_tracker.items():
            self.logger.record(f"train/problem_feasibility_{key}", value)

    def evaluate_line_search_iterate(self, rollout_data: RolloutBufferSamples, old_policy_dist: th.distributions.Distribution, policy_loss: th.Tensor, advantages: th.Tensor, cost_advantages: th.Tensor):
        """
        Evaluate the line search iterate
        """
        # calculate new kl divergence
        new_policy_dist = self.policy.get_distribution(rollout_data.observations, rpo_perturb=self.use_rpo_perturb)
        new_kl_div = kl_divergence(old_policy_dist, new_policy_dist).mean()

        # calculate new policy and cost objective
        new_log_prob = new_policy_dist.log_prob(rollout_data.actions)
        new_policy_loss = -self.get_policy_objective(advantages, new_log_prob, rollout_data.old_log_prob)
        new_cost_objective = self.get_cost_objective(cost_advantages, new_log_prob, rollout_data.old_log_prob)

        cost_threshold = max(0.0, -self.c) + self.cost_objective
        cost_constraint_satisfied = new_cost_objective <= cost_threshold
        policy_improved = new_policy_loss <= policy_loss if self.problem_feasibility > 1 else True

        # check if new policy satisfies the constraint
        valid = new_kl_div < self.kl_div_step_size and cost_constraint_satisfied and policy_improved

        return valid, new_policy_loss, new_kl_div
    
    def get_policy_objective(self, advantages: th.Tensor, log_prob: th.Tensor, old_log_prob : th.Tensor) -> th.Tensor:
        """
        Compute the policy objective
        """
        policy_objective = th.exp(log_prob - old_log_prob) * advantages
        return policy_objective.mean()
    
    def get_cost_objective(self, cost_advantages: th.Tensor, log_prob: th.Tensor, old_log_prob : th.Tensor) -> th.Tensor:
        """
        Compute the cost objective
        """
        expectation = th.exp(log_prob - old_log_prob) * cost_advantages.squeeze()
        # rescale so cost gradient correctly for trajectory expectation
        # the need for rescaling the cost gradient is because we want an expectation over trajectories and not (state,action) pairs
        return expectation.mean() * self.rollout_avg_ep_length

class FailurePredictionNetwork(th.nn.Module):
    """
    This networks predicts whether the agent will be in a safe state within the next n steps.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # merge all dimension except batch dimension
        x = th.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = th.relu(x)
        x = self.fc2(x)
        x = th.sigmoid(x)
        return x