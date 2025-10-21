from typing import List, Optional, Type, Union
import warnings
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance
from saferl.common.policies import ActorCriticWithCostPolicy
from saferl.common.on_policy_algorithm import SafeOnPolicyAlgorithm
from stable_baselines3.common.buffers import RolloutBufferSamples


class APPO(SafeOnPolicyAlgorithm):
    """
    APPO: Augmented Policy Optimization for constrained RL.
    """
    
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticWithCostPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_critic_updates: int = 1,
        n_policy_updates: int = 1,
        eps: float = 1e-8,
        clip_range: float = 0.2,
        target_kl: float = 0.01,
        # quadratic penalty parameters
        sigma: float = 0.1,
        rho: float = 0.1,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        cost_constraint: Union[float, List] = [25, None],
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        **kwargs,
    ):
        super(APPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            ent_coef=0.0,  # entropy bonus is not used by PPO-Lag
            vf_coef=0.0,  # value function is optimized separately
            max_grad_norm=0.0,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            cost_constraint=cost_constraint,
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

        assert self.cost_dim == 1, "PPO-Lag currently only supports one dimensional costs"
        
        self.eps = eps
        self.batch_size = batch_size
        self.n_critic_updates = n_critic_updates
        self.n_policy_updates = n_policy_updates
        self.target_kl = target_kl
        self.clip_range = clip_range
        self.normalize_advantages = normalize_advantage
        
        self.sigma = sigma
        self.rho = rho
        
        # Lagrangian multiplier parameters
        self.init_beta = 0.001
        
        # Initialize Lagrangian multiplier
        self.soft_beta = th.nn.Parameter(
            th.tensor([self.init_beta], device=self.device), 
            requires_grad=True
        )
        self.beta_optimizer = th.optim.Adam(
            [self.soft_beta], 
            lr=0.2
        )
        
    def train(self) -> None:
        """
        Use current rollout data to update the policy with PPO-Lag
        """
        self.policy.set_training_mode(True)

        critic_losses = []
        cost_critic_losses = []
        
        # Check if rollout buffer has episode stats
        if "ep_returns" not in self.rollout_buffer.episode_stats:
            warnings.warn("Rollout buffer does not contain episode stats. Skipping update.")
            return
        
        # Calculate average episode cost return
        rollout_avg_ep_cost_return = np.mean([np.sum(ep_info["cost"]) for ep_info in self.ep_info_buffer])

        # Update critics
        for j in range(self.n_critic_updates):
            # Sample rollout data from buffer
            for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):
                values_predicted = self.policy.predict_values(rollout_data.observations).squeeze()
                cost_values_predicted = self.policy.predict_cost_values(rollout_data.observations)

                returns = rollout_data.returns
                cost_returns = rollout_data.cost_returns

                critic_loss = nn.MSELoss()(values_predicted, returns)
                cost_critic_loss = nn.MSELoss()(cost_values_predicted, cost_returns)
                critic_losses.append(critic_loss.item())
                cost_critic_losses.append(cost_critic_loss.item())

                self.policy.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.policy.critic_optimizer.step()
                
                self.policy.cost_critic_optimizer.zero_grad()
                cost_critic_loss.backward()
                self.policy.cost_critic_optimizer.step()

        # Update Lagrangian multiplier based on constraint violation
        beta_loss = self._update_lagrangian_multiplier(rollout_avg_ep_cost_return)
        
        # Policy updates with minibatches and early stopping on KL
        for _ in range(self.n_policy_updates):
            approx_kl_mean = 0.0
            num_minibatches = 0
            for rollout_data in self.rollout_buffer.get(None):
                # Current policy log prob
                policy_dist = self.policy.get_distribution(
                    rollout_data.observations
                )
                new_log_prob = policy_dist.log_prob(rollout_data.actions)

                # Advantages
                advantages = rollout_data.advantages
                cost_advantages = rollout_data.cost_advantages

                if self.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
                    # Normalize each cost dimension independently
                    for i in range(self.cost_dim):
                        cost_advantages[..., i] = (
                            cost_advantages[..., i] - cost_advantages[..., i].mean()
                        ) / (cost_advantages[..., i].std() + self.eps)

                # Ratios
                ratio = th.exp(new_log_prob - rollout_data.old_log_prob)

                # Clipped surrogate for reward advantages
                clipped_ratio = th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                surrogate_reward = th.min(ratio * advantages, clipped_ratio * advantages).mean()

                # Unclipped surrogate for cost advantages (use cost sum if multi-dim)
                cost_advantages = cost_advantages.squeeze(-1) if self.cost_dim == 1 else cost_advantages.sum(dim=-1)
                surrogate_cost = th.max(ratio * cost_advantages, clipped_ratio * cost_advantages).mean()

                # Argumented quadratic penalty
                beta = F.softplus(self.soft_beta).detach()
                if rollout_avg_ep_cost_return - self.cost_constraint[0] + beta / self.sigma > 0:
                    factor = beta + self.sigma * (rollout_avg_ep_cost_return - self.cost_constraint[0])
                else:
                    factor = 0.0
                penalty = factor * surrogate_cost
                policy_loss = (-surrogate_reward + penalty) / (1 + factor)

                # Optimize actor only: filter to actor params by zeroing other grads implicitly ok
                self.policy.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.policy.actor_optimizer.step()

                # Approximate KL: E[old_logp - new_logp]
                with th.no_grad():
                    approx_kl = (rollout_data.old_log_prob - new_log_prob).mean().item()
                approx_kl_mean += approx_kl
                num_minibatches += 1

            approx_kl_mean = approx_kl_mean / max(num_minibatches, 1)

        # Log Lagrangian multiplier
        self.logger.record("train/beta", beta.item())
        self.logger.record("train/beta_loss", beta_loss)
        # Log critic losses
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/cost_critic_loss", np.mean(cost_critic_losses))

    def _update_lagrangian_multiplier(self, ep_cost_return_mean: float) -> None:
        """
        Update the Lagrangian multiplier based on constraint violation
        """
        self.beta_optimizer.zero_grad()
        # Maximize lambda when constraint is violated: minimize negative of it
        beta_loss = -self.soft_beta * (ep_cost_return_mean - self.cost_constraint[0])
        beta_loss.backward()
        self.beta_optimizer.step()
        return beta_loss.item()

    def _compute_policy_advantage(
        self, 
        advantages: th.Tensor, 
        log_prob: th.Tensor, 
        old_log_prob: th.Tensor
    ) -> th.Tensor:
        """
        Compute the policy advantage (reward advantage)
        """
        ratio = th.exp(log_prob - old_log_prob)
        policy_advantage = ratio * advantages
        return policy_advantage.mean()

    def _compute_cost_advantage(
        self, 
        cost_advantages: th.Tensor, 
        log_prob: th.Tensor, 
        old_log_prob: th.Tensor
    ) -> th.Tensor:
        """
        Compute the cost advantage (cost advantage)
        """
        ratio = th.exp(log_prob - old_log_prob)
        cost_advantage = ratio * cost_advantages.squeeze()
        return cost_advantage.mean()        

    def _excluded_save_params(self) -> List[str]:
        """Parameters to exclude from saving"""
        return super(APPO, self)._excluded_save_params() + [
            "beta_optimizer",
            "soft_beta",
            "policy"
        ]
    
    def _get_torch_save_params(self) -> List[str]:
        """Parameters to save"""
        state_dicts = ["policy", "beta_optimizer", "policy.actor_optimizer", "policy.critic_optimizer", "policy.cost_critic_optimizer"]
        saved_pytorch_variables = ["soft_beta"]

        return state_dicts, saved_pytorch_variables