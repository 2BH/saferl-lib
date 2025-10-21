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
from stable_baselines3.common.vec_env import VecEnv, unwrap_vec_normalize, unwrap_vec_wrapper
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from saferl.common.buffers import CostReplayBuffer
from saferl.common.policies import SACwithCostPolicy
from saferl.common.wrappers import ExtendedVecNormalize
from saferl.common.off_policy_algorithm import SafeOffPolicyAlgorithm


class WCSAC(SafeOffPolicyAlgorithm):
    """
    This code is based on the implementation from Stable-baseline3 and Safety-starter-agent (OpenAI).
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
        risk_level: float = 1,
        damp_scale: float = 0.1,
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
        
        assert policy_kwargs["n_cost_critics"] == 2, "WCSAC only supports 2 cost critics"
        super(WCSAC, self).__init__(
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
        self.risk_level = risk_level
        self.damp_scale = damp_scale
        self._safe_episode_num = 0
        self._success_episodes_num = 0 

        if _init_setup_model:
            self._setup_model()
        
        # Overwrite the VecNormalize Env        
        self._vec_normalize_env = unwrap_vec_wrapper(env, ExtendedVecNormalize)
    
    def _setup_model(self) -> None:
        super(WCSAC, self)._setup_model()
        self._create_aliases()

        self.log_beta = th.zeros(1, device=self.device).requires_grad_(True)
        self.beta_optimizer = th.optim.Adam([self.log_beta], lr=self.lr_schedule(1))

        normal = th.distributions.normal.Normal(th.tensor([0.0]), th.tensor([1.0]))
        self.pdf_cdf = normal.log_prob(normal.icdf(th.tensor(self.risk_level))).exp() / self.risk_level
        self.pdf_cdf = self.pdf_cdf.to(self.device)
        
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
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.cost_critic = self.policy.cost_critic
        self.cost_critic_target = self.policy.cost_critic_target
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
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

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            # cast added because of type mismatch between normalized observation (casted to float32) and action from replay_data
            # CHANGED: same cast as in sac_lb
            actions = replay_data.actions.float()
            current_q_values = self.critic(replay_data.observations, actions)
            current_qc_values, current_vc_values = self.cost_critic(replay_data.observations, actions)
            current_vc_values = F.softplus(current_vc_values)
            current_vc_values = th.clamp(current_vc_values, min=1e-8, max=1e8)
            with th.no_grad():
                # Select action acc ording to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)

                # Compute the target Q_cost values
                # Compute the next Q-cost values with CVar
                next_qc_values, next_vc_values = self.cost_critic_target(replay_data.next_observations, next_actions)
                next_vc_values = F.softplus(next_vc_values)
                next_vc_values = th.clamp(next_vc_values, min=1e-8, max=1e8)
                target_qc_values = replay_data.costs + (1 - replay_data.dones) * self.cost_gamma * next_qc_values
                target_vc_values = replay_data.costs**2 - \
                                   current_qc_values**2 + \
                                   2 * self.cost_gamma * replay_data.costs * next_qc_values + \
                                   self.cost_gamma**2 * next_vc_values + \
                                   self.cost_gamma**2 * next_qc_values
                target_vc_values = th.clamp(target_vc_values, min=1e-8, max=1e8)

                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            
            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            qc_values, vc_values = self.cost_critic.forward(replay_data.observations, actions_pi)
            vc_values = F.softplus(vc_values)
            vc_values = th.clamp(vc_values, min=1e-8, max=1e8)

            # Compute the CVar Cost with Damp Impact
            cvar = qc_values + self.pdf_cdf * th.sqrt(vc_values)
            damp = self.damp_scale * th.mean(self.cost_constraint[0] - cvar)

            beta = th.exp(self.log_beta)
            actor_loss = (ent_coef * log_prob - \
                min_qf_pi + (beta.detach() - damp) * (qc_values + self.pdf_cdf * th.sqrt(vc_values))).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            # Optimize the cost penalty factor
            beta_loss = th.mean(beta * (self.cost_constraint[0] - cvar).detach())
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
            cost_critic_loss = 0.5* (F.mse_loss(current_qc_values, target_qc_values).mean() + \
                (current_vc_values + target_vc_values - 2 * th.sqrt(current_vc_values * target_vc_values)).mean())
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
        save_params = super(WCSAC, self)._excluded_save_params() + ["actor", "critic", \
            "critic_target", "cost_critic", "cost_critic_target", \
            "log_beta"]
        return save_params
        
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer", "cost_critic.optimizer", "beta_optimizer"]
        saved_pytorch_variables = ["log_ent_coef", "log_beta"]

        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables
