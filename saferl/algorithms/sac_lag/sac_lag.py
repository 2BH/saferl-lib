from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutReturn, TrainFreq
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_wrapper
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from saferl.common.buffers import CostReplayBuffer
from saferl.common.policies import SACwithCostPolicy
from saferl.common.wrappers import ExtendedVecNormalize
from saferl.common.off_policy_algorithm import SafeOffPolicyAlgorithm

class SAC_LAG(SafeOffPolicyAlgorithm):
    """
    Soft Actor-Critic with Lagrangian constraints (SAC-Lag).
    
    This implementation extends the standard SAC algorithm to handle safety constraints
    through Lagrangian multipliers. The algorithm learns both the policy and the 
    Lagrangian multipliers simultaneously to satisfy cost constraints while maximizing
    the expected return.
    
    The key innovation is the use of a separate cost critic network that estimates
    the expected cumulative cost, which is then used in the Lagrangian penalty term
    to enforce safety constraints.
    
    Paper: "Constrained Policy Optimization" (CPO) and "Soft Actor-Critic" (SAC)
    
    Args:
        policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        env: The environment to learn from (if registered in Gym, can be str. Can be None for loading trained models)
        learning_rate: Learning rate for the optimizer, it can be a function of the current progress remaining (from 1 to 0)
        buffer_size: Size of the replay buffer
        learning_starts: How many steps of the model to collect transitions for before learning starts
        batch_size: Minibatch size for each gradient update
        tau: The soft update coefficient ("Polyak update", between 0 and 1)
        gamma: The discount factor (can be a list for separate reward and cost discount factors)
        train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
            like ``(5, "step")`` or ``(2, "episode")``.
        gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
            Set to ``-1`` means to do as many gradient steps as steps done in the environment
            during the rollout.
        action_noise: The action noise type (None by default), this can help
            for hard exploration problem. Cf common.noise for the different action noise type.
        action_noise_scaling_factor: Scaling factor for the action noise for scheduling strategies
        replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
            If ``None``, it will be automatically selected.
        replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
        optimize_memory_usage: Enable a memory efficient variant of the replay buffer
            at a cost of more complexity.
        ent_coef: Entropy regularization coefficient. It can be "auto" to automatically tune it,
            or a float (then fixed), or a callable (then schedule).
        target_update_interval: Update the target network every ``target_update_interval``
            environment steps.
        target_entropy: Target entropy when learning ``ent_coef``, if ``ent_coef`` is "auto".
            When "auto", the target entropy is set to -prod(env.action_space.shape).
        cost_constraint: List of cost constraints for each cost dimension. Each element can be
            a float (hard constraint) or None (no constraint for that dimension).
        use_sde: Whether to use State Dependent Exploration (SDE)
            instead of action noise exploration (default: False)
        sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
            during the warm up phase (before learning starts)
        tensorboard_log: The log location for tensorboard (if None, no logging)
        policy_kwargs: Additional arguments to be passed to the policy on creation
        verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
            debug messages
        seed: Seed for the pseudo random generators
        device: Device on which the code should run.
            By default, it will try to use a Cuda compatible device and fallback to cpu
            if it is not possible.
        _init_setup_model: Whether or not to call ``_setup_model()`` when initializing the model.
            This should be set to False when loading a model from disk.
    
    Example:
        >>> from saferl.algorithms.sac_lag import SAC_LAG
        >>> from saferl.common.utils import create_env
        >>> 
        >>> # Create environment
        >>> env = create_env(env_cfg, seed=42)
        >>> 
        >>> # Create and train model
        >>> model = SAC_LAG("MlpPolicy", env, cost_constraint=[5.0])
        >>> model.learn(total_timesteps=100000)
        >>> 
        >>> # Save model
        >>> model.save("safe_agent")
    """
    def __init__(
        self,
        policy: Union[str, Type[SACwithCostPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        action_noise_scaling_factor: Union[float, Schedule] = 1.0,
        replay_buffer_class: Optional[CostReplayBuffer] = CostReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        cost_constraint: Union[float, list] = [2, None],
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        **kwargs,
    ):
        if type(gamma) is float:
            self.gamma = gamma
            self.cost_gamma = gamma
        elif type(gamma) is list:
            self.gamma = gamma[0]
            self.cost_gamma = gamma[1]
        super(SAC_LAG, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            self.gamma,
            train_freq,
            gradient_steps,
            action_noise,
            action_noise_scaling_factor,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
            **kwargs
        )
        # assert isinstance(env, SafeEnv), "env must be an instance of SafeEnv"
        # TODO: fix that cost_dim is a list
        # self.cost_dim = self.env.venv.get_attr("cost_dim")[0]
        self.cost_dim = self.env.get_attr("cost_dim")[0]
        assert len(cost_constraint) == self.cost_dim, "cost_constraint must be same as the cost_dim of the environment"

        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {"cost_dim": self.cost_dim}
        else:
            replay_buffer_kwargs["cost_dim"] = self.cost_dim

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        self.cost_constraint = cost_constraint
        self._safe_episode_num = 0
        self._success_episodes_num = 0 

        if _init_setup_model:
            self._setup_model()
        # Overwrite the VecNormalize Env
        
        self._vec_normalize_env = unwrap_vec_wrapper(env, ExtendedVecNormalize)
    
    def _setup_model(self) -> None:
        """
        Create the networks and the optimizer.
        
        This method initializes the actor, critic, and cost critic networks,
        along with their target networks. It also sets up the Lagrangian
        multiplier (beta) and entropy coefficient optimizers.
        """
        super(SAC_LAG, self)._setup_model()
        self.soft_beta = th.zeros(1, device=self.device).requires_grad_(True)
        self.beta_optimizer = th.optim.Adam([self.soft_beta], lr=self.lr_schedule(1))
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self):
        """
        Create aliases for the policy components for easier access.
        
        This method creates direct references to the actor, critic, and cost critic
        networks and their target networks from the policy object.
        """
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.cost_critic = self.policy.cost_critic
        self.cost_critic_target = self.policy.cost_critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Update the policy, critics, and Lagrangian multipliers.
        
        This method performs the core training step of SAC-Lag, which includes:
        1. Updating the entropy coefficient (if learned)
        2. Updating the actor policy
        3. Updating the Lagrangian multiplier (beta)
        4. Updating the reward critic
        5. Updating the cost critic
        6. Updating target networks
        
        Args:
            gradient_steps: Number of gradient steps to perform
            batch_size: Size of the batch for training
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.train()

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, cost_critic_losses, beta_losses = [], [], [], []

        optimizers = [self.actor.optimizer, self.critic.optimizer, self.cost_critic.optimizer, self.beta_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)
        
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
 
            with th.no_grad():
                # Select action acc ording to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                # Compute the target Q_cost values
                # Compute the next Q-cost values: min over all critics targets
                next_qc_values = th.cat(self.cost_critic_target(replay_data.next_observations, next_actions), dim=1)
                next_qc_values, _ = th.max(next_qc_values, dim=1, keepdim=True)
                # next_qc_values, _ = th.max(next_qc_values, dim=1, keepdim=True)
                target_qc_values = replay_data.costs + (1 - replay_data.dones) * self.cost_gamma * next_qc_values
                
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            
            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            current_qc_values = self.cost_critic(replay_data.observations, replay_data.actions)                

            beta = F.softplus(self.soft_beta)
            beta_loss = beta * sum([th.mean(self.cost_constraint[0] - current_qc) for current_qc in current_qc_values])
            
            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            qc_values_pi = th.cat(self.cost_critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            min_qc_pi, _ = th.max(qc_values_pi, dim=1, keepdim=True)

            actor_loss = (ent_coef * log_prob - min_qf_pi + beta * min_qc_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            # Optimize the cost penalty factor
            self.beta_optimizer.zero_grad()
            beta_loss.backward(retain_graph=True)
            self.beta_optimizer.step()
            beta_losses.append(beta_loss.item())

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute cost critic loss
            cost_critic_loss = 0.5 * sum([F.mse_loss(current_qc, target_qc_values) for current_qc in current_qc_values])
            cost_critic_losses.append(cost_critic_loss.item())

            # Optimize the cost critic
            self.cost_critic.optimizer.zero_grad()
            cost_critic_loss.backward()
            self.cost_critic.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.cost_critic.parameters(), self.cost_critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/cost_critic_loss", np.mean(cost_critic_losses))
        self.logger.record("train/target_q_values", np.mean(target_q_values.cpu().numpy()))
        self.logger.record("train/target_qc_values", np.mean(target_qc_values.cpu().numpy()))
        self.logger.record("train/max_target_qc_values", np.max(target_qc_values.cpu().numpy()))
        self.logger.record("train/min_target_qc_values", np.min(target_qc_values.cpu().numpy()))
        if self.get_vec_normalize_env() is not None:
            self.logger.record("train/norm_cost_mean", np.mean(self.env.cost_rms.mean))
            self.logger.record("train/norm_cost_var", np.mean(self.env.cost_rms.var))
            self.logger.record("train/norm_ret_mean", np.mean(self.env.ret_rms.mean))
            self.logger.record("train/norm_ret_var", np.mean(self.env.ret_rms.var))

        self.logger.record("train/beta_loss", np.mean(beta_losses))
        self.logger.record("train/beta", beta.detach().cpu().numpy()[0])
        
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _excluded_save_params(self) -> List[str]:
        save_params = super(SAC_LAG, self)._excluded_save_params() + ["actor", "critic", \
            "critic_target", "cost_critic", "cost_critic_target", "soft_beta"]
        return save_params
        
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "cost_critic.optimizer", "beta_optimizer"]
        saved_pytorch_variables = ["log_ent_coef", "soft_beta"]

        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables
