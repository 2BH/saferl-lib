import sys
import time
import pathlib
import io
import copy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import pickle
import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise

from saferl.common.policies import ActorCriticWithCostPolicy
from saferl.common.buffers import CostRolloutBuffer

from saferl.common.buffers import SaveRolloutsBuffer

import matplotlib.pyplot as plt

SelfSafeOnPolicyAlgorithm = TypeVar("SelfSafeOnPolicyAlgorithm", bound="SafeOnPolicyAlgorithm")


class SafeOnPolicyAlgorithm(BaseAlgorithm):
    """
    Modified stable_baselines3/common/on_policy_algorithm.py for safe RL

    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: CostRolloutBuffer
    policy: ActorCriticWithCostPolicy

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticWithCostPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.0,
        vf_coef: float = 0.0,
        max_grad_norm: float = 0.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        cost_constraint: List[int] = [0],
        state_safe_horizon: int = 0,
        reset_on_rollout_end : bool = False,
        save_rollout_buffers: bool = False,
        save_rollout_buffer_size: int = 20000,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
        action_noise: Optional[ActionNoise] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )
        if type(gamma) is float:
            self.gamma = gamma
            self.cost_gamma = gamma
        elif type(gamma) is list:
            self.gamma = gamma[0]
            self.cost_gamma = gamma[1]

        if type(gae_lambda) is float:
            self.gae_lambda = gae_lambda
            self.cost_gae_lambda = gae_lambda
        elif type(gae_lambda) is list:
            self.gae_lambda = gae_lambda[0]
            self.cost_gae_lambda = gae_lambda[1]

        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.cost_constraint = cost_constraint
        self.state_safe_horizon = state_safe_horizon
        
        self.cost_dim = self.env.get_attr("cost_dim")[0]
        self.reset_on_rollout_end = reset_on_rollout_end

        # check if cost_constraint specified
        if len(cost_constraint) != self.cost_dim:
            # if cost_constraint not specified, set to 0
            if len(cost_constraint) != 0:
                assert Warning(f"cost_constraint and cost_dim have different lengths: {len(cost_constraint)} and {self.cost_dim}")
            else:
                print(f"WARNING: cost_constraint not specified. Using 0 as cost constraint value for every cost dimension")
                # set cost_constraint to 0 for each cost dimension
                cost_constraint = [0 for _ in range(self.cost_dim)]

        self.save_buffer_container = None        
        if save_rollout_buffers:
            replay_buffer_kwargs = {
                "buffer_size": save_rollout_buffer_size,
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "device": self.device,
                "n_envs": self.n_envs,
                "cost_dim": self.cost_dim,
                "optimize_memory_usage": False,
                "handle_timeout_termination": True

            }
            self.save_buffer_container = SaveRolloutsBuffer(self.verbose, **replay_buffer_kwargs)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = CostRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            cost_gamma=self.cost_gamma,
            gae_lambda=self.gae_lambda,
            cost_gae_lambda=self.cost_gae_lambda,
            n_envs=self.n_envs,
            cost_dim=self.cost_dim,
            cost_constraint=self.cost_constraint,
            state_safe_horizon=self.state_safe_horizon
        )
        # pytype:disable=not-instantiable
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, cost_dim = self.cost_dim, use_sde=self.use_sde, **self.policy_kwargs
        )
        # pytype:enable=not-instantiable
        self.policy = self.policy.to(self.device)


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CostRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
        callback.on_rollout_start()

        ep_returns = np.zeros((env.num_envs, 1))
        ep_returns_undiscounted = np.zeros((env.num_envs, 1))
        ep_cost_returns = np.zeros((env.num_envs, self.cost_dim))
        ep_cost_returns_undiscounted = np.zeros((env.num_envs, self.cost_dim))
        ep_lengths = np.zeros((env.num_envs, 1))

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, cost_values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions

            # Rescale and perform action
            if isinstance(self.action_space, spaces.Box):               
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # make step in every environment
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # the costs are the sum of costs for each environment (costs from single cost types are available in infos)
            if self.cost_dim == 1:
                costs = np.array([info['cost'] for info in infos])
            else:
                cost_keys = list(infos[0].keys())
                # filter cost_sum, cost and all other keys not containing cost_
                cost_keys = [key for key in cost_keys if "cost_" in key and "sum" not in key]
                # filter cost_sum and cost and take all remaining cost dimensions
                costs = np.array([[info[key] for key in cost_keys] for info in infos])
            # states_safe = np.array([info['state_safe'] for info in infos])

            self.num_timesteps += env.num_envs
            n_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)


            # calculate returns and cost returns (undiscounted and discounted without bootstrapping)
            # currently this only works for cost sum and not seperate cost types
            ep_returns += self.gamma**ep_lengths * rewards.reshape(-1, 1)
            ep_returns_undiscounted += rewards.reshape(-1, 1)
            ep_cost_returns += self.cost_gamma**ep_lengths * costs
            ep_cost_returns_undiscounted += costs
            ep_lengths += 1
            # get indices of done environments
            done_indices = np.where(dones)[0]
            if len(done_indices) > 0:
                # add returns, cost returns and lengths of done envs to rollout
                self.rollout_buffer.add_episode_stats({
                    "ep_returns": ep_returns[done_indices].flatten(),
                    "ep_returns_undiscounted": ep_returns_undiscounted[done_indices].flatten(),
                    "ep_lengths": ep_lengths[done_indices].flatten(),
                    "ep_cost_returns": ep_cost_returns[done_indices],
                    "ep_cost_returns_undiscounted": ep_cost_returns_undiscounted[done_indices]
                })
                # reset returns, costs and lengths for done envs
                ep_returns[dones] = 0
                ep_returns_undiscounted[dones] = 0
                ep_cost_returns[dones] = 0
                ep_cost_returns_undiscounted[dones] = 0
                ep_lengths[dones] = 0

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                        terminal_cost_value = self.policy.predict_cost_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value
                    if self.cost_dim == 1:
                        costs[idx][0] += self.cost_gamma * terminal_cost_value
                    else:
                        # costs[env_index] = cost sum of env
                        costs[idx] = np.add(costs[idx], self.cost_gamma * terminal_cost_value.detach().cpu().numpy())


            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                costs,
                cost_values,
                infos
            )


            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
            cost_values = self.policy.predict_cost_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.compute_cost_returns_and_advantage(last_cost_values=cost_values, dones=dones)

        rollout_buffer.torch_rng_state = th.random.get_rng_state()
        if th.cuda.is_available():
            rollout_buffer.torch_cuda_rng_state = th.cuda.random.get_rng_state()
        if self.save_buffer_container is not None:
            self.save_buffer_container.add_buffer(copy.deepcopy(rollout_buffer))

        callback.on_rollout_end()

        if self.reset_on_rollout_end:
            for env_idx in range(self.n_envs):
                # only reset environments that are not done (and therefore already reset)
                if not dones[env_idx]:
                    self._last_obs[env_idx], _ = self.env.envs[env_idx].reset()
            # self._last_obs = self.env.reset()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: SelfSafeOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "SafeOnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSafeOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        # print all logs with levels larger or same as given level
        verbose_to_level = {
            1: (10, "DEBUG"),
            2: (20, "INFO"),
            3: (30, "WARN"),
            4: (40, "ERROR"),
            0: (50, "DISABLED"),
        }
        print("Setting log level to {}".format(verbose_to_level[self.verbose][1]))
        self.logger.set_level(verbose_to_level[self.verbose][0])

        if self.save_buffer_container is not None:
            self.logger.warn("Saving rollout buffers is enabled. This can use a lot of memory.")

        callback.on_training_start(locals(), globals())

        assert self.env is not None
        self._last_obs = self.env.reset()

        while self.num_timesteps < total_timesteps:
            num_timesteps_before = self.num_timesteps

            self.logger.info("Start rollout iteration {}".format(iteration))
            self.logger.debug("Collecting rollout")
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            self.logger.debug("Finished collecting rollout, total timesteps collected {}".format(self.num_timesteps - num_timesteps_before))
            
            iteration += 1
            if continue_training is False:
                break

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.write_logs(iteration, num_timesteps_before)

            self.logger.debug("Start training iteration {}".format(iteration))
            self.train()
            self.logger.dump(step=self.num_timesteps)
            self.logger.debug("Finished training iteration {}".format(iteration))
            self.logger.debug("---------------------------------")

        callback.on_training_end()
        self.logger.info("Training ended")
        self.logger.info("Total timesteps: {:}".format(self.num_timesteps))
        self.logger.info("Total time elapsed: {:.2f} s".format(max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)))
        
        return self
    
    def write_logs(self, iteration, num_timesteps_before) -> None:
        """
        Write logs to tensorboard
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/env_interactions_over_iterations", self.num_timesteps - num_timesteps_before)
        self.logger.record("time/iterations", iteration)#, exclude="tensorboard")

        # undiscounted
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_ret_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_cost_ret_mean", safe_mean([np.sum(ep_info["cost"]) for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed))#, exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps)#, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
    

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Use this method to enable a consistent option for saving rollout buffers of multiple iterations
    
        Normally this method is not used in on-policy algorithms as there exists no replay buffer.
        """
        
        if self.save_buffer_container is not None:
            self.logger.info("Appending rollout to replay buffer")
            self.save_buffer_container.save(path)