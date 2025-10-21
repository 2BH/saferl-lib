from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.policies import ActorCriticPolicy, ContinuousCritic

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from torch import nn
from torch.distributions import Uniform
import numpy as np
from functools import partial
from copy import deepcopy

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    Distribution,
)
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs

class SACwithCostPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        n_cost_critics: int = 1,
        share_features_extractor: bool = False,
    ):
        super(SACwithCostPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
        self.n_cost_critics = n_cost_critics
        if self.share_features_extractor:
            self.cost_critic = self.make_critic(features_extractor=self.actor.features_extractor)
            cost_critic_parameters = [param for name, param in self.cost_critic.named_parameters() if "features_extractor" not in name]
        else:
            self.cost_critic = self.make_cost_critic(features_extractor=None)
            cost_critic_parameters = self.cost_critic.parameters()
        
        self.cost_critic_target = self.make_cost_critic(features_extractor=None)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        self.cost_critic.optimizer = self.optimizer_class(cost_critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)
    
    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)
    
    def make_cost_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        cost_critic_kwargs = self.critic_kwargs
        cost_critic_kwargs["n_critics"] = self.n_cost_critics
        cost_critic_kwargs = self._update_features_extractor(cost_critic_kwargs, features_extractor)
        return ContinuousCritic(**cost_critic_kwargs).to(self.device)


class ActorCriticWithCostPolicy(ActorCriticPolicy):
    """
    Policy class for ActorCritic with Cost critic
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        cost_dim: int = 1,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        rpo_perturbance_alpha: float = 0.0,
    ):
        # cost critic output dimension
        self.cost_dim = cost_dim
        # perturbance used in RPO paper: "ROBUST POLICY OPTIMIZATION IN DEEP REINFORCEMENT LEARNING" (2022) by Rahman et al.
        # https://arxiv.org/abs/2212.07536
        self.rpo_perturbance_dist = None
        self.rpo_perturbance = None
        if rpo_perturbance_alpha > 0:
            self.rpo_perturbance_dist = Uniform(-rpo_perturbance_alpha, rpo_perturbance_alpha)

        super(ActorCriticWithCostPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Original code from stable_baselines3.common.policies.ActorCriticPolicy._build modified for cost value function
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        # handle shared features extractor for cost value function
        if self.share_features_extractor:
            self.cf_features_extractor = self.features_extractor
        else:
            self.cf_features_extractor = self.make_features_extractor()

        self._build_mlp_extractor()

        # create mlp extractor cost value function in analogy to value function
        # self.mlp_extractor_cost_net = deepcopy(self.mlp_extractor.value_net)
        self.mlp_extractor_cost_net, self.latent_dim_cf = self.create_cost_critic()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # create cost_net in analogy to value_net
        self.cost_net = nn.Linear(self.latent_dim_cf, self.cost_dim)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.mlp_extractor_cost_net: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.cost_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
                module_gains[self.cf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        # self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.actor_optimizer = self.optimizer_class(self.action_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.critic_optimizer = self.optimizer_class(self.value_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.cost_critic_optimizer = self.optimizer_class(self.cost_net.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor, critic and cost critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, costs and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
            latent_cf = self.mlp_extractor_cost_net(features)
        else:
            pi_features, vf_features = features
            cf_features = super(ActorCriticPolicy, self).extract_features(obs, self.cf_features_extractor)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            latent_cf = self.mlp_extractor_cost_net(cf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        # Evaluate the costs for the given observations
        costs = self.cost_net(latent_cf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, costs, log_prob
    
    def get_distribution(self, obs: PyTorchObs, rpo_perturb = False, recalc_rpo_perturb = False) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)

        perturb = rpo_perturb and self.rpo_perturbance_dist is not None
        return self._get_action_dist_from_latent(latent_pi, rpo_perturb=perturb, recalc_rpo_perturb=recalc_rpo_perturb)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, rpo_perturb = False, recalc_rpo_perturb = False) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if rpo_perturb:
            # Add perturbance to the mean actions which is sampled from uniform distribution -self.rpo_perturbance_alpha to +self.rpo_perturbance_alpha
            if recalc_rpo_perturb or self.rpo_perturbance is None:
                self.rpo_perturbance = self.rpo_perturbance_dist.sample(mean_actions.shape).to(self.device)
            mean_actions = mean_actions + self.rpo_perturbance

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def predict_cost_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated cost values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super(ActorCriticPolicy, self).extract_features(obs, self.cf_features_extractor)
        latent_cf = self.mlp_extractor_cost_net(features)
        return self.cost_net(latent_cf)


    def create_cost_critic(self) -> None:
        """
        Create cost critic like value critic in the original code: stable_baselines3.common.policies.ActorCriticPolicy._build_mlp_extractor
        """
        cost_net: List[nn.Module] = []
        last_layer_dim_cf = self.features_dim

        # save dimensions of layers in policy and value nets
        if isinstance(self.net_arch, dict):
            # Note: if key is not specificed, assume linear network
            cf_layers_dims = self.net_arch.get("cf", [])  # Layer sizes of the value network
        else:
            cf_layers_dims = self.net_arch

        # Iterate through the value layers and build the value net
        for curr_layer_dim in cf_layers_dims:
            cost_net.append(nn.Linear(last_layer_dim_cf, curr_layer_dim))
            cost_net.append(self.activation_fn())
            last_layer_dim_cf = curr_layer_dim

        # Save dim, used to create the distributions
        latent_dim_cf = last_layer_dim_cf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        return nn.Sequential(*cost_net).to(self.device), latent_dim_cf 

