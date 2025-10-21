import pickle
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import utils
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

from typing import Optional
import gymnasium as gym
import numpy as np

class SafetyGymWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        cost_dim: int = 1,
        **env_kwargs,
    ):
        super().__init__(env)
        self.cost_dim = cost_dim
        self.incl_cost_in_reward = env_kwargs.get("incl_cost_in_reward", False)

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        # observation, reward, done, info = self.env.step(action)
        # info["cost"] = [1.0] if info["cost"] > 0 else [0.0]
        
        # info["cost"] = [info["cost"]]
        # info["state_safe"] = True if np.sum(info["cost"]) <= 0 else False
        if self.cost_dim == 1:
            cost = [cost]
        info["cost"] = cost
        info["state_safe"] = True if np.sum(cost) <= 0 else False
        if self.incl_cost_in_reward:
            reward -= np.sum(info["cost"])
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with :param:`**kwargs` and sets the number of steps elapsed to zero.
        Args:
            **kwargs: The kwargs to reset the environment with
        Returns:
            The reset environment
        """
        return self.env.reset(**kwargs)


class ExtendedVecNormalize(VecEnvWrapper):
    """
    """

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        norm_cost: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        clip_cost: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        VecEnvWrapper.__init__(self, venv)
        assert isinstance(
            self.observation_space, (gym.spaces.Box, gym.spaces.Dict)
        ), "ExtendedVecNormalize only support `gym.spaces.Box` and `gym.spaces.Dict` observation spaces"

        if isinstance(self.observation_space, gym.spaces.Dict):
            self.obs_keys = set(self.observation_space.spaces.keys())
            self.obs_spaces = self.observation_space.spaces
            self.obs_rms = {key: RunningMeanStd(shape=space.shape) for key, space in self.obs_spaces.items()}
        else:
            self.obs_keys, self.obs_spaces = None, None
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = np.array([])
        self.old_reward = np.array([])
        self.cost_dim = [venv.get_attr("cost_dim")[0]]
        # Check observation spaces,88455434
        self.norm_cost = norm_cost
        self.cost_rms = RunningMeanStd(shape=(self.cost_dim))
        self.clip_cost = clip_cost
        # Returns: discounted rewards
        self.cost_ret = np.zeros((self.num_envs, *self.cost_dim), dtype=np.float32)
        self.old_cost = np.array([])

    def __getstate__(self) -> Dict[str, Any]:
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["ret"]
        del state["cost_ret"]
        return state
     
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None

    def set_venv(self, venv: VecEnv) -> None:
        """
        Sets the vector environment to wrap to venv.
        Also sets attributes derived from this such as `num_env`.
        :param venv:
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        VecEnvWrapper.__init__(self, venv)

        # Check only that the observation_space match
        utils.check_for_correct_spaces(venv, self.observation_space, venv.action_space)
        self.ret = np.zeros(self.num_envs)
        self.cost_ret = np.zeros((self.num_envs, *self.cost_dim))

    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)
        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        costs = np.array([info["cost"] for info in infos], dtype=np.float32)
        self.old_obs = obs
        self.old_reward = rewards
        self.old_cost = costs

        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        if self.training:
            self._update_cost(costs)
        costs = self.normalize_cost(costs)
        # Normalize the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])

        self.ret[dones] = 0
        self.cost_ret[dones] = 0
        return obs, rewards, dones, infos

    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(self.ret)
    
    def _update_cost(self, cost: np.ndarray) -> None:
        self.cost_ret = self.cost_ret * self.gamma + cost
        self.cost_rms.update(self.cost_ret)

    def _normalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: normalized observation
        """
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def _unnormalize_obs(self, obs: np.ndarray, obs_rms: RunningMeanStd) -> np.ndarray:
        """
        Helper to unnormalize observation.
        :param obs:
        :param obs_rms: associated statistics
        :return: unnormalized observation
        """
        return (obs * np.sqrt(obs_rms.var + self.epsilon)) + obs_rms.mean

    def normalize_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        obs_ = deepcopy(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                # Only normalize the specified keys
                for key in self.norm_obs_keys:
                    obs_[key] = self._normalize_obs(obs[key], self.obs_rms[key]).astype(np.float32)
            else:
                obs_ = self._normalize_obs(obs, self.obs_rms).astype(np.float32)

        return obs_

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.norm_reward:
            reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return reward
    
    def normalize_cost(self, cost: np.ndarray) -> np.ndarray:
        if self.norm_cost:
            cost = np.clip(cost / np.sqrt(self.cost_rms.var + self.epsilon), -self.clip_cost, self.clip_cost)
        return cost
    
    def unnormalize_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        # Avoid modifying by reference the original object
        obs_ = deepcopy(obs)
        if self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    obs_[key] = self._unnormalize_obs(obs[key], self.obs_rms[key])
            else:
                obs_ = self._unnormalize_obs(obs, self.obs_rms)
        return obs_

    def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.norm_reward:
            return reward * np.sqrt(self.ret_rms.var + self.epsilon)
        return reward

    def unnormalize_cost(self, cost: np.ndarray) -> np.ndarray:
        if self.norm_cost:
            return cost * np.sqrt(self.cost_rms.var + self.epsilon)
        return cost

    def get_original_obs(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns an unnormalized version of the observations from the most recent
        step or reset.
        """
        return deepcopy(self.old_obs)

    def get_original_reward(self) -> np.ndarray:
        """
        Returns an unnormalized version of the rewards from the most recent step.
        """
        return self.old_reward.copy()
    
    def get_original_cost(self) -> np.ndarray:
        """
        Returns an unnormalized version of the costs from the most recent step.
        """
        return self.old_cost.copy()

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        :return: first observation of the episode
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.ret = np.zeros(self.num_envs)
        self.cost_ret = np.zeros((self.num_envs, *self.cost_dim))
        if self.training:
            self._update_reward(self.ret)
            self._update_cost(self.cost_ret)
        return self.normalize_obs(obs)

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> "ExtendedVecNormalize":
        """
        Loads a saved VecNormalize object.
        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            extended_vec_normalize = pickle.load(file_handler)
        extended_vec_normalize.set_venv(venv)
        return extended_vec_normalize

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)
        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)
