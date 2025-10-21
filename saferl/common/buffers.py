import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, NamedTuple
from collections import deque
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.buffers import BaseBuffer

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer, DictReplayBuffer, ReplayBuffer
try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
from itertools import groupby
from saferl.common.utils import compute_consecutive_cost_chains_stats
from saferl.common.safe_type_aliases import CostRolloutBufferSamples

# extended functionality from stable-baselines3/stable_baselines3/common/buffers.py RolloutBuffer to include cost
class CostRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer from sb3 (stable_baselines3/common/buffers.py) extended to include cost

    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gae_cost_lamda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator for cost
    :param gamma: Discount factor
    :param cost_gamma: Discount factor for cost
    :param n_envs: Number of parallel environments
    :param cost_dim: Cost dimension
    :param cost_constraint: Cost threshold, describes the maximum allowed cost for each cost dimension. If cost exceeds threshold, state is considered unsafe.
    :param state_safe_horizon: Number of steps to look ahead to determine if state fails in future
    :param infos: List of additional information about the transition.
    """

    costs: np.ndarray
    cost_advantages: np.ndarray
    cost_returns: np.ndarray
    cost_values: np.ndarray
    states_safe_in_horizon: np.ndarray
    infos: List[Dict[str, np.ndarray]]
    episode_stats: Dict[str, List]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        cost_gae_lambda: float = 1,
        gamma: float = 0.99,
        cost_gamma: float = 0.99,
        n_envs: int = 1,
        cost_dim: List = None,
        cost_constraint: List[int] = [],
        state_safe_horizon: int = 0,
    ):
        if cost_dim is None:
            raise ValueError("cost_dim must be specified. Should be given from environment.")
            
        # set cost dim before calling super().__init__() because it calls reset() which uses cost_dim
        if isinstance(cost_dim, int):
            self.cost_dim = cost_dim
        elif isinstance(cost_dim, list):
            self.cost_dim = cost_dim[0]
        else:
            raise ValueError("cost_dim must be int or list of int")

        self.cost_constraint = np.array(cost_constraint)
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, gae_lambda=gae_lambda, gamma=gamma)
        self.cost_gamma = cost_gamma
        self.cost_gae_lambda = cost_gae_lambda
        self.state_safe_horizon = state_safe_horizon
        self.episode_stats = {}

        # funtion for modifying cost when adding to buffer
        self.cost_modification_functions = {}

        # torch rng state after rollout (for loading)
        self.torch_rng_state = None
        self.torch_cuda_rng_state = None
    
    def register_cost_modification_function(self, function_name: str, function: callable) -> None:
        """
        Register a function to modify the cost when adding to the buffer
        """
        if function_name in self.cost_modification_functions:
            raise Warning(f"Overwriting function with name {function_name} (function name was already registered")
        self.cost_modification_functions[function_name] = function

    def reset(self) -> None:
        """
        Original code from reset() in stable_baselines3/common/buffers.py modified for cost
        """
        self.torch_rng_state = None
        self.torch_cuda_rng_state = None
        self.costs = np.zeros((self.buffer_size, self.n_envs, self.cost_dim), dtype=np.float32)
        self.cost_advantages = np.zeros((self.buffer_size, self.n_envs, self.cost_dim), dtype=np.float32)
        self.cost_returns = np.zeros((self.buffer_size, self.n_envs, self.cost_dim), dtype=np.float32)
        self.cost_values = np.zeros((self.buffer_size, self.n_envs, self.cost_dim), dtype=np.float32)
        self.states_safe_in_horizon = np.zeros((self.buffer_size, self.n_envs), dtype=np.int0)
        self.infos = np.array([{} for _ in range(self.buffer_size)], dtype=object)
        self.episode_stats = {}
        super().reset()

    def compute_cost_returns_and_advantage(self, last_cost_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Original code from compute_returns_and_advantage() in stable_baselines3/common/buffers.py modified for cost

        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state cost value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """

        # Convert to numpy
        last_cost_values = last_cost_values.clone().cpu().numpy()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_cost_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.cost_values[step + 1]

            next_non_terminal = next_non_terminal.reshape(-1, 1)

            delta = self.costs[step] + self.cost_gamma * next_values * next_non_terminal - self.cost_values[step]
            last_gae_lam = delta + self.cost_gamma * self.cost_gae_lambda * next_non_terminal * last_gae_lam
            self.cost_advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.cost_returns = self.cost_advantages + self.cost_values

    def compute_consecutive_cost_chains(self) -> None:
        """
        Compute the consecutive cost chains over all environments
        Do not consider chains over episode boundaries

        :return: the frequencies of each chain length from all environments and the total number of unsafe steps
        :return: the total number of unsafe steps
        :return: the maximum number of consecutive unsafe steps for each environment
        :return: the expected normalized maximum number of consecutive unsafe steps

        Note: 0 cost chains are excluded from consecutive_cost_chain_frequencies
        """
        info_per_env = np.zeros((self.n_envs, self.buffer_size), dtype=np.int32)
        for i, step_info in enumerate(self.infos):
            for env_index, info in enumerate(step_info):
                info_per_env[env_index, i] = info['state_safe']
        
        stats_safe_flags_all_envs = [np.where(info_per_env[env_index], 1, 0) for env_index in range(self.n_envs)]
        episode_start_indices_all_envs = [np.argwhere(self.episode_starts[:, env_index] == 1).flatten() for env_index in range(self.n_envs)]
        consecutive_unsafe_steps_freq, total_unsafe_steps, max_consecutive_unsafe_steps_per_env, normalized_max_consecutive_unsafe_steps = compute_consecutive_cost_chains_stats(stats_safe_flags_all_envs, episode_start_indices_all_envs)

        return consecutive_unsafe_steps_freq, total_unsafe_steps, max_consecutive_unsafe_steps_per_env, normalized_max_consecutive_unsafe_steps

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        cost: np.ndarray,
        cost_value: th.Tensor,
        infos: List[Dict[str, np.ndarray]] = [],
    ) -> None:
        """
        Original code from add() in stable_baselines3/common/buffers.py modified for cost

        :param obs: Observation
        :param action: Action
        :param reward: Reward
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param cost: Sum of cost per environment
        :param cost_value: estimated cost value of the current state
        """

        # add one dimension to cost (at end)
        cost = np.array(cost).copy().reshape(-1, self.cost_dim)
        # check if cost modification functions are registered and apply them
        if self.cost_modification_functions:
            for function_name, function in self.cost_modification_functions.items():
                cost = function(observation=obs, action=action, reward=reward, episode_start=episode_start, value=value, log_prob=log_prob, cost=cost, cost_value=cost_value)

        self.costs[self.pos] = cost

        self.cost_values[self.pos] = cost_value.clone().cpu().numpy()
        self.infos[self.pos] = infos

        # update states_safe_in_horizon given the current state for the previous state_safe_horizon states
        if self.state_safe_horizon > 0:
            # 1 is safe, 0 is unsafe
            current_state_safe = np.array([info['state_safe'] for info in infos])
            # tracks whether the last update was a new episode start, if true then stop updating states_safe_in_horizon for this env
            envs_break = [False for _ in range(self.n_envs)]
            # alter states_safe_in_horizon from previous states
            for i in range(1, self.state_safe_horizon + 1):
                for env_index in range(self.n_envs):
                    if current_state_safe[env_index] == 0:
                        continue
                    if envs_break[env_index] or self.pos - i < 0 or self.episode_starts[self.pos, env_index] == 1:
                        continue

                    if self.episode_starts[self.pos - i, env_index] == 1:
                        envs_break[env_index] = True

                    self.states_safe_in_horizon[self.pos - i, env_index] += 1
            
        super().add(obs, action, reward, episode_start, value, log_prob)

    def add_episode_stats(self, episode_stats: Dict[str, np.ndarray]) -> None:
        """
        Add episode stats to buffer
        """
        if not isinstance(episode_stats, dict):
            raise ValueError("episode_stats must be a dict")
        if len(episode_stats) == 0:
            print("Warning: episode_stats is empty")
            return
        
        for key, value in episode_stats.items():
            if key not in self.episode_stats:
                # print(f"Warning: episode_stats key {key} not in buffer. Adding key.")
                self.episode_stats[key] = []
            self.episode_stats[key].extend(value)

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        """
        Original code from get() in stable_baselines3/common/buffers.py modified for cost
        """

        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "cost_values",
                "cost_advantages",
                "cost_returns",
                "states_safe_in_horizon"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> CostRolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        """
        Original code from _get_samples() in stable_baselines3/common/buffers.py modified for cost
        """
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.cost_values[batch_inds],
            self.cost_advantages[batch_inds],
            self.cost_returns[batch_inds],
            self.states_safe_in_horizon[batch_inds],
        )
        return CostRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CostReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    costs: th.Tensor


class CostReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in off-policy algorithms with constraints.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        cost_dim: int = 1,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        track_transition_type_stats: bool = False,
        sample_types_by_shares: List[float] = None,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # assert n_envs == 1, "Replay buffer only support single environment for now"
        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)
        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
        self.cost_dim = cost_dim
        self.optimize_memory_usage = optimize_memory_usage
        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.costs = np.zeros((self.buffer_size, self.n_envs, self.cost_dim), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # variable for storing the state_safe of the last state (= some pos) before its overwritten in the buffer when full
        self.last_state_safe = np.ones((self.n_envs), dtype=bool)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.infos = np.array([{} for _ in range(self.buffer_size)], dtype=object)

        self.track_transition_type_stats = track_transition_type_stats
        self.sample_types_by_shares = sample_types_by_shares
        if not track_transition_type_stats and sample_types_by_shares is not None:
            warnings.warn("sample_types_by_shares is only used when track_transition_type_stats is True. Ignoring sample_types_by_shares.")
            self.sample_types_by_shares = None
        if self.sample_types_by_shares is not None:
            assert len(self.sample_types_by_shares) == 4, "sample_types_by_shares must have 4 elements"
            assert abs(sum(self.sample_types_by_shares) - 1) < 1e-9, "sample_types_by_shares must sum to 1"

        # store stats about type of transitions (safe->safe, safe->unsafe, unsafe->safe, unsafe->unsafe)
        self.safe2safe = 0
        self.safe2unsafe = 0
        self.unsafe2safe = 0
        self.unsafe2unsafe = 0

        self.single_env = n_envs == 1

        self.transition_type_indices = {
            'safe2safe': deque(),
            'safe2unsafe': deque(),
            'unsafe2safe': deque(),
            'unsafe2unsafe': deque()
        }


        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )
        
        self.consecutive_unsafe_steps_frequencies = []

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray, 
            reward: np.ndarray,
            done: np.ndarray,
            cost: np.ndarray,
            episode_starts: np.ndarray = None,
            infos: Dict[str, np.ndarray] = None) -> None:
        
        if self.track_transition_type_stats:
            previous_index = self.pos - 1 if self.pos > 0 else self.pos
            if self.full and self.pos == 0:
                previous_index = self.buffer_size - 1
            if self.full:
                # remove stats about type of transitions from the oldest transition which are replaced by the new ones
                current_pos_safe = self.last_state_safe
                next_pos_safe = np.array([info['state_safe'] for info in self.infos[self.pos]])
                episode_start_mask = ~np.array(self.episode_starts[self.pos], dtype=bool)
                self.safe2safe -= np.sum(current_pos_safe & next_pos_safe & episode_start_mask)
                self.safe2unsafe -= np.sum(current_pos_safe & ~next_pos_safe & episode_start_mask)
                self.unsafe2safe -= np.sum(~current_pos_safe & next_pos_safe & episode_start_mask)
                self.unsafe2unsafe -= np.sum(~current_pos_safe & ~next_pos_safe & episode_start_mask)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.costs[self.pos] = np.array(cost)
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        if episode_starts is not None:
            self.episode_starts[self.pos] = np.array(episode_starts).copy()
        
        if self.full:
            self.last_state_safe = np.array([info['state_safe'] for info in self.infos[self.pos]])
        if infos is not None:
            self.infos[self.pos] = infos

        if self.track_transition_type_stats:
            # update stats about type of transitions
            current_state_safe = np.array([info['state_safe'] for info in self.infos[previous_index]])
            next_state_safe = np.array([info['state_safe'] for info in infos])
            # remove current states from the arrays that start a new episode
            episode_start_mask = ~np.array(episode_starts, dtype=bool)
            safe2safe = current_state_safe & next_state_safe & episode_start_mask
            safe2unsafe = current_state_safe & ~next_state_safe & episode_start_mask
            unsafe2safe = ~current_state_safe & next_state_safe & episode_start_mask
            unsafe2unsafe = ~current_state_safe & ~next_state_safe & episode_start_mask

            # count transitions by summing the number of True values in the logical arrays
            self.safe2safe += np.sum(safe2safe)
            self.safe2unsafe += np.sum(safe2unsafe)
            self.unsafe2safe += np.sum(unsafe2safe)
            self.unsafe2unsafe += np.sum(unsafe2unsafe)

            # store indices of transitions for each type (respect envs)
            for transition_type, transitions in zip(["safe2safe", "safe2unsafe", "unsafe2safe", "unsafe2unsafe"], [safe2safe, safe2unsafe, unsafe2safe, unsafe2unsafe]):
                # remove the oldest transition of the same type if buffer is full
                # by choosing to only store if there are transitions of given type we save memory
                if self.full and len(self.transition_type_indices[transition_type]) > 0:
                    transition_index = self.transition_type_indices[transition_type][0]
                    if not self.single_env:
                        transition_index = transition_index[0]
                    if transition_index == previous_index:
                        self.transition_type_indices[transition_type].popleft()
                
                type_indices = np.where(transitions)[0]
                if len(type_indices) > 0:
                    transition_index = (previous_index, type_indices)
                    if self.single_env:
                        transition_index = previous_index
                    self.transition_type_indices[transition_type].append(transition_index)


        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get_transition_type_stats(self):
        return self.safe2safe, self.safe2unsafe, self.unsafe2safe, self.unsafe2unsafe

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if self.sample_types_by_shares is not None:
            return self._sample_with_type_shares(batch_size, env=env)

        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _sample_with_type_shares(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer with different types of transitions
        based on the shares defined in `sample_types_by_shares`.
        """
        if self.optimize_memory_usage:
            raise ValueError("Sampling with type shares is currently only supported without memory optimization.")
        if self.sample_types_by_shares is None:
            raise ValueError("sample_types_by_shares must be defined to use this function.")
        if not self.track_transition_type_stats:
            raise ValueError("track_transition_type_stats must be True to use this function.")
        
        # get the number (defined by self.sample_types_by_shares) of transition indices for each type
        transition_indices = np.array([], dtype=np.int64)
        transition_env_indices = np.array([], dtype=np.int64)
        for i, transition_type in enumerate(["safe2safe", "safe2unsafe", "unsafe2safe", "unsafe2unsafe"]):
            max_sample_size = int(batch_size * self.sample_types_by_shares[i])
            # maybe restricting the number of samples in case of not enough samples is helpful to not overfit to unlikely types?
            # if max_sample_size > len(self.transition_type_indices[transition_type]):
            #     max_sample_size = len(self.transition_type_indices[transition_type])

            if len(self.transition_type_indices[transition_type]) == 0 or max_sample_size == 0:
                continue

            if self.single_env:
                # sample indices for each type
                type_indices = np.random.randint(0, len(self.transition_type_indices[transition_type]), size=max_sample_size)
                # access queue directly, if this becomes a bottleneck try converting to list before accessing
                transition_indices = np.concatenate([transition_indices, np.array([self.transition_type_indices[transition_type][i] for i in type_indices], dtype=np.int64)])
            
            else:
                # store the indices and env indices in seperate lists
                seperated_previous_indices = []
                seperated_previous_env_indices = []
                for transition_type_previous_index, env_indices in self.transition_type_indices[transition_type]:
                    seperated_previous_indices.extend([transition_type_previous_index for _ in env_indices])
                    seperated_previous_env_indices.extend(env_indices)
                
                type_indices = np.random.randint(0, len(seperated_previous_indices), size=(max_sample_size, ))
                transition_indices = np.concatenate([transition_indices, np.array(seperated_previous_indices, dtype=np.int64)[type_indices]])
                transition_env_indices = np.concatenate([transition_env_indices, np.array(seperated_previous_env_indices, np.int64)[type_indices]])

        if self.single_env:
            return self._get_samples(transition_indices, env=env)
        else:
            return self._get_samples_with_env_indices(transition_indices, transition_env_indices, env=env)
        
        
    def _get_samples_with_env_indices(self, transition_indices: np.ndarray, transition_env_indices: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Method equivalent to _get_samples() but with env indices as parameter
        """
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(transition_indices + 1) % self.buffer_size, transition_env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[transition_indices, transition_env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[transition_indices, transition_env_indices, :], env),
            self.actions[transition_indices, transition_env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[transition_indices, transition_env_indices] * (1 - self.timeouts[transition_indices, transition_env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[transition_indices, transition_env_indices].reshape(-1, 1), env),
            self.costs[transition_indices, transition_env_indices]
        )
        return CostReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.costs[batch_inds, env_indices]
        )
        return CostReplayBufferSamples(*tuple(map(self.to_torch, data)))
    
    def compute_consecutive_cost_chains(self, replay_buffer_pos_before) -> None:
        """
        Compute the consecutive cost chains for each environment
        Do not consider chains over episode boundaries
        Data is considered from position replay_buffer_pos_before to self.pos
        :param replay_buffer_pos_before: position of the replay buffer from which to start the computation

        :return: consecutive_cost_chain_frequencies: frequencies of consecutive cost chains from all environments
        :return: total_cost_steps: total number of cost steps from all environments
        :return: max_consecutive_cost_steps_per_env: maximum number of consecutive cost steps for each environment
        :return: normalized_max_consecutive_unsafe_steps: normalized maximum number of consecutive cost steps

        Note: 0 cost chains are excluded from consecutive_cost_chain_frequencies
        """
        
        request_size = self.pos - replay_buffer_pos_before
        if request_size < 0:
            request_size = self.buffer_size - replay_buffer_pos_before + self.pos
            relevant_info = self.infos[replay_buffer_pos_before:self.buffer_size]
            relevant_info = np.concatenate((relevant_info, self.infos[0:self.pos]), axis=0)
        else:
            relevant_info = self.infos[replay_buffer_pos_before:self.pos]

        info_per_env = np.zeros((self.n_envs, request_size), dtype=np.int32)

        for i, step_info in enumerate(relevant_info):
            for env_index, info in enumerate(step_info):
                info_per_env[env_index, i] = info['state_safe']

        state_safe_flags_all_envs = [np.where(info_per_env[env_index], 1, 0) for env_index in range(self.n_envs)]
        episode_start_indices_all_envs = []
        for env_idx in range(self.n_envs):
            if  self.pos < replay_buffer_pos_before:
                episode_starts = np.argwhere(self.episode_starts[replay_buffer_pos_before:self.buffer_size, env_idx] == 1).flatten()
                episode_starts = np.concatenate((episode_starts, np.argwhere(self.episode_starts[0:self.pos, env_idx] == 1).flatten() + (self.buffer_size-replay_buffer_pos_before)), axis=0)
            else:
                episode_starts = np.argwhere(self.episode_starts[replay_buffer_pos_before:self.pos, env_idx] == 1).flatten()

            episode_start_indices_all_envs.append(episode_starts)
        consecutive_unsafe_steps_frequencies, total_unsafe_steps, max_consecutive_unsafe_steps_per_env, normalized_max_consecutive_unsafe_steps = compute_consecutive_cost_chains_stats(state_safe_flags_all_envs, episode_start_indices_all_envs)
        return consecutive_unsafe_steps_frequencies, total_unsafe_steps, max_consecutive_unsafe_steps_per_env, normalized_max_consecutive_unsafe_steps
    

class SaveRolloutsBuffer:
    """
    Class for storing rollout buffers into one replay buffer

    :param verbose: Verbosity level
    :param kwargs: Additional arguments for creation of the replay buffer
    """
    def __init__(self, verbose, **kwargs):
        self.source_n_envs = kwargs["n_envs"]
        # set n_envs to 1 
        kwargs["n_envs"] = 1
        self.replay_buffer = CostReplayBuffer(**kwargs)
        self.buffers_saved = 0
        self.overwrite_on_full = False
        self.verbose = verbose
        
    def add_buffer(self, buffer : CostRolloutBuffer, safe_add = False):
        """
        Add a rollout buffer to the replay buffer

        :param buffer: Rollout buffer to add
        :param safe_add: If True, add each transition individually. If False, add all transitions at once
        """
        assert isinstance(buffer, BaseBuffer), "Buffer is not valid"
        if self.replay_buffer.full and not self.overwrite_on_full:
            warnings.warn("Replay buffer is full and overwrite_on_full is False. Buffer not added.")
            return
        
        assert buffer.pos >= 1, f"Buffer requires at least two transitions, found {buffer.pos}"

        observations = buffer.swap_and_flatten(buffer.observations[:-1])
        next_observations = buffer.swap_and_flatten(buffer.observations[1:])
        actions = buffer.swap_and_flatten(buffer.actions[:-1])
        rewards = buffer.swap_and_flatten(buffer.rewards[:-1])
        costs = buffer.swap_and_flatten(buffer.costs[:-1])
        episode_starts = buffer.swap_and_flatten(buffer.episode_starts[:-1])
        dones = buffer.swap_and_flatten(buffer.episode_starts[1:])
        # add 1 dimension at index 1 as new num_env dimension
        observations = np.expand_dims(observations, axis=1)
        next_observations = np.expand_dims(next_observations, axis=1)
        actions = np.expand_dims(actions, axis=1)
        costs = np.expand_dims(costs, axis=1)
        # info is structured as list of dicts, where each dict contains the info for every env
        # convert info to list where all entries of an env are sequential
        info_per_env = [[] for _ in range(self.source_n_envs)]
        for step_pos in range(buffer.pos-1):
            for env_index, dict_info in enumerate(buffer.infos[step_pos]):
                info_per_env[env_index].append(dict_info)
        
        infos = info_per_env[0]
        # concatenate infos of all envs
        for i in range(1, len(info_per_env)):
            infos = np.concatenate((infos, info_per_env[i]), axis=0)

        number_of_transitions = buffer.pos * self.source_n_envs - 1
        if safe_add:
            # add each transition individually, by inversly replaying the buffer
            for i in range(number_of_transitions):
                if i + number_of_transitions + 1 >= self.replay_buffer.buffer_size:
                    break
                self.replay_buffer.add(
                    obs=observations[i],
                    next_obs=next_observations[i],
                    action=actions[i],
                    reward=rewards[i],
                    done=dones[i],
                    episode_starts=episode_starts[i],
                    cost=costs[i],
                    infos=infos[i]
                )
        else:
            # add all transitions at once, assumes compatibility of buffers
            new_pos = self.replay_buffer.pos + number_of_transitions
            if new_pos >= self.replay_buffer.buffer_size:
                number_of_transitions -= new_pos - self.replay_buffer.buffer_size + 1
                new_pos = self.replay_buffer.buffer_size
                self.replay_buffer.full = True
            self.replay_buffer.observations[self.replay_buffer.pos:new_pos] = observations[:number_of_transitions]
            self.replay_buffer.next_observations[self.replay_buffer.pos:new_pos] = next_observations[:number_of_transitions]
            self.replay_buffer.actions[self.replay_buffer.pos:new_pos] = actions[:number_of_transitions]
            self.replay_buffer.rewards[self.replay_buffer.pos:new_pos] = rewards[:number_of_transitions]
            self.replay_buffer.dones[self.replay_buffer.pos:new_pos] = dones[:number_of_transitions]
            self.replay_buffer.costs[self.replay_buffer.pos:new_pos] = costs[:number_of_transitions]
            self.replay_buffer.episode_starts[self.replay_buffer.pos:new_pos] = episode_starts[:number_of_transitions]
            self.replay_buffer.infos[self.replay_buffer.pos:new_pos] = infos[:number_of_transitions]
        
        self.buffers_saved += 1

    def save(self, save_path):
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(save_path, self.replay_buffer, self.verbose)