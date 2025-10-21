import numpy as np
import hydra
import os
from copy import deepcopy
from omegaconf import DictConfig
from itertools import groupby
from saferl.common.wrappers import ExtendedVecNormalize, SafetyGymWrapper
from saferl.common.monitor import CostMonitor
from saferl.common.policies import SACwithCostPolicy, ActorCriticWithCostPolicy
import stable_baselines3.common.noise
from stable_baselines3.common.utils import get_linear_fn, constant_fn
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.type_aliases import Schedule
import gymnasium as gym
from gymnasium.wrappers.rescale_action import RescaleAction
from gymnasium.wrappers.flatten_observation import FlattenObservation
import safety_gymnasium
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def evaluate(
    model,
    env, 
    save_video=False,
    save_path=None,
    num_episodes=1, 
    episode_len=1000, 
    deterministic=True, 
    render=False,
    seed=None,
    ):
    """
    Evaluate the model on the environment
    :param model: (stable_baselines3) the model to evaluate
    :param env: (gym.Env) the environment to evaluate on
    :param save_video: (bool) whether to save the video
    :param save_path: (str) the path to save the video
    :param num_episodes: (int) the number of episodes to evaluate
    :param episode_len: (int) the length of each episode
    :param deterministic: (bool) whether to use deterministic actions
    :param render: (bool) whether to render the environment
    :param seed: (int) the seed for the environment
    :return: (dict) the evaluation results
    """

    rets = []
    costs = []
    max_consecutive_cost_steps = []
    is_safe_episodes = []
    all_episode_len = []

    for episode_idx in range(num_episodes):
        if seed:
            env.seed(seed+episode_idx)
        if save_video:
            eval_env = VecVideoRecorder(
                env,
                video_folder = os.path.join(save_path, "videos"),
                record_video_trigger=lambda x: x == 0,
                video_length=episode_len,
                name_prefix="eval_video_{}".format(episode_idx))
        else:
            eval_env = env
        obs = eval_env.reset()
        done = False
        total_reward = []
        total_costs = []
        is_episode_safe = True
        for step_idx in range(episode_len):
            
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, infos = eval_env.step(action)
            total_reward.append(rewards) 
            total_costs.append([info["cost"] for info in infos])
            if render:
                eval_env.render()
            if any([not info["state_safe"] for info in infos]):
                is_episode_safe = False
            if done:
                break
        episode_ret = np.sum(total_reward)
        episode_cost = np.sum(total_costs)
        actual_episode_len = step_idx + 1
        rets.append(episode_ret)
        costs.append(episode_cost)
        is_safe_episodes.append(is_episode_safe)
        all_episode_len.append(actual_episode_len)

        total_costs_arr = np.array(total_costs)
        if any(total_costs_arr > 0):
            # convert to binary, transpose to get shape (n_cost_dims, n_steps), then find the maximum consecutive cost steps
            binary_costs = np.where(total_costs_arr > 0, 1, 0).T
            # find the maximum consecutive cost steps
            # groupby grouped consecutive elements and gives value as key, filter for keys with value 1 and get length of each group, then get max
            consecutive_cost_steps = [len(list(group)) for env_costs in binary_costs for cost_dim in env_costs for key, group in groupby(cost_dim) if key == 1]
            max_consecutive_cost_steps.append(max(consecutive_cost_steps))
        else:
            max_consecutive_cost_steps.append(0.0)

        if save_video:
            if eval_env.recording:
                eval_env._stop_recording() 

    return {"ret": rets, "cost": costs, "is_safe": is_safe_episodes, "len": all_episode_len, "max_consecutive_cost_steps": max_consecutive_cost_steps}

def evaluate_after_training(    
    model,
    env, 
    save_video=False,
    save_path=None,
    num_episodes=1, 
    deterministic=True, 
    render=False,
    seed=None,
    cvar_alphas=None):
    
    print("Evaluating after training")
    if hasattr(env, '_max_episode_steps'):
        episode_len = env._max_episode_steps
    else:
        try:
            episode_len = env.get_attr('_max_episode_steps')[0]
        except:
            episode_len = 1000
    result = evaluate(model, env, save_video, save_path, num_episodes, episode_len, deterministic, render, seed)
    print("Evaluation finished")
    
    costs = result["cost"]
    cvar_results = {}
    if cvar_alphas is not None:
        if isinstance(costs[0], list):
            costs = [cost for episode_costs in costs for cost in episode_costs]
        costs = np.array(costs)
        # check cvar_alpha data type
        if isinstance(cvar_alphas, float):
            cvar_alphas = [cvar_alphas]
        cvar_alphas = np.array(cvar_alphas)
        cvar_results = cvar_from_distribution(costs, cvar_alphas)

    norm_max_consecutive_cost_steps = np.array(result["max_consecutive_cost_steps"]) / np.array(result["len"])
    emcc_cvar_results = {}
    if cvar_alphas is not None:
        emcc_cvar_results = cvar_from_distribution(norm_max_consecutive_cost_steps, cvar_alphas)

    # write metric to log file
    with open(os.path.join(save_path, "eval_metrics_random_seeds.log"), "a") as f:
        f.write(f"Evaluation metrics from {num_episodes} episodes\n")
        # f.write(f"Seed: {seed}\n")
        f.write(f"Average Return: {np.mean(result['ret'])}\n")
        f.write(f"Average Cost: {np.mean(result['cost'])}\n")
        f.write(f"Average Length: {np.mean(result['len'])}\n")
        for key, value in cvar_results.items():
            f.write(f"CVaR at {key}: {value}\n")
        for key, value in emcc_cvar_results.items():
            f.write(f"EMCC CVaR at {key}: {value}\n")
        f.write("\n")


def sync_envs_normalization(env: GymEnv, eval_env: GymEnv) -> None:
    """
    Sync eval env and train env when using ExtendedVecNormalize

    :param env:
    :param eval_env:
    """
    env_tmp, eval_env_tmp = env, eval_env
    while isinstance(env_tmp, VecEnvWrapper):
        if isinstance(env_tmp, ExtendedVecNormalize):
            eval_env_tmp.obs_rms = deepcopy(env_tmp.obs_rms)
            # eval_env_tmp.ret_rms = deepcopy(env_tmp.ret_rms)
            # eval_env_tmp.cost_rms = deepcopy(env_tmp.cost_rms)
        env_tmp = env_tmp.venv
        eval_env_tmp = eval_env_tmp.venv


def interval_barrier(x, lb, rb, eps=1e-2, grad=None):
    x = (x - lb) / (rb - lb) * 2 - 1
    b = -((1 + x + eps) * (1 - x + eps) / np.log(1 + eps)**2)
    b_min, b_max = 0, -np.log(eps * (2 + eps) / (1 + eps)**2)
    if grad is None:
        grad = 2. / eps / (2 + eps)
    out = grad * (np.abs(x) - 1)
    return np.where((-1 < x) & (x < 1), b / b_max, 1 + out)

def create_env(
        env_cfg: DictConfig,
        seed: int = 0,
        monitor: bool = True,
        save_path: str = None,
        norm_obs: bool = True,
        norm_act: bool = True,
        norm_reward: bool = False,
        norm_cost: bool = False,
        num_env: int = 1,
        training: bool = True,
        env_kwargs: dict = {},
        use_multi_process: bool = False
        ):
    """
    Create the environment
    :param env_cfg: (dict) the config of the environment
    :param seed: (int) the seed of the environment
    :param monitor: (bool) whether to monitor the environment
    :param save_path: (str) the path to save the model
    :return: (GymEnv) the environment
    """
    if use_multi_process:
        assert num_env > 1, "use_multi_process is set to True but num_env is not greater than 1"
        env = SubprocVecEnv([lambda: instantiate_env(env_cfg, seed, norm_act=norm_act, monitor=monitor, save_path=save_path, env_kwargs=env_kwargs) for i in range(num_env)])
    else:
        env = DummyVecEnv([lambda: instantiate_env(env_cfg, seed, norm_act=norm_act, monitor=monitor, save_path=save_path, env_kwargs=env_kwargs) for i in range(num_env)])
    # env.seed(seed)
    # _ = env.reset()
    print("All envs created")
    if norm_obs or norm_reward or norm_cost:
        env = ExtendedVecNormalize(
            env,
            norm_reward=norm_reward,
            norm_obs=norm_obs,
            norm_cost=norm_cost,
            # gamma=env_cfg.gamma, # TODO: automatically infer this from model config
            training=training)

    return env

def instantiate_env(
        env_cfg: DictConfig,
        seed: int = 0,
        norm_act: bool = True,
        min_action: float = -1.0,
        max_action: float = 1.0,
        monitor: bool = True,
        save_path: str = None,
        env_kwargs: dict = {}):
    
    if "NumericalOptimization" in env_cfg.env_name:
        from saferl.common.envs.numerical_optimization_tasks import numerical_env_alias_dict, create_numerical_optimization_env
        assert env_kwargs["env_type"] in numerical_env_alias_dict.keys(), f"env_type {env_kwargs['env_type']} not in {numerical_env_alias_dict.keys()}"
        env = create_numerical_optimization_env(env_cfg, **env_kwargs)
        if norm_act:
            env = RescaleAction(env, min_action, max_action)
        if monitor:
            env = CostMonitor(env, filename=save_path)
        return env

    if env_cfg.env_name in gym.envs.registry.keys():
        env_kwargs = dict(env_kwargs)
        cost_dim = env_kwargs.pop("cost_dim", 1)
        safe_env = safety_gymnasium.make(env_cfg.env_name, **env_kwargs)
        env = SafetyGymWrapper(safe_env, cost_dim, **env_kwargs)
        env = FlattenObservation(env)
        print(f"Safety Gym env: {env_cfg.env_name} created")
    else:
        import saferl.common.envs
        env = safety_gymnasium.make(env_cfg.env_name, **env_kwargs)
        env = SafetyGymWrapper(env, 1, **env_kwargs)
        env = FlattenObservation(env)
    
    # env.seed(seed)
    if norm_act:
        env = RescaleAction(env, min_action, max_action)
    if monitor:
        env = CostMonitor(env, filename=save_path)
    return env

def create_training_model(model_cfg: DictConfig, env: GymEnv, **kwargs):
    if isinstance(env.action_space, gym.spaces.Box) > 0:
        n_actions = env.action_space.shape[-1]
    else:
        # Discrete case
        if "action_n" in model_cfg.model:
            n_actions = model_cfg.model.action_n
        else:
            n_actions = env.action_space.n

    if model_cfg.noise is not None:
        action_noise = getattr(stable_baselines3.common.noise, model_cfg.noise.noise_type)(
            mean=np.zeros(n_actions, dtype=np.float32),
            sigma=model_cfg.noise.sigma * np.ones(n_actions, dtype=np.float32))
    else:
        action_noise = None

    if model_cfg.policy_class not in ["MlpPolicy", "CnnPolicy"]:
        model = hydra.utils.instantiate(model_cfg.model, policy=eval(model_cfg.policy_class), action_noise=action_noise, env=env, **kwargs)
    else:
        model = hydra.utils.instantiate(model_cfg.model, policy=model_cfg.policy_class, action_noise=action_noise, env=env, **kwargs)
    
    return model

def create_on_step_callback(on_step_callback_cfg: DictConfig, eval_env: GymEnv, save_path: str):
    """
    Create a callback that will be called on each step of the training
    :param on_step_callback_cfg: (dict) the config of the experiment
    :param save_path: (str) the path to save the model
    :return: (function) the callback
    """
    if on_step_callback_cfg.get("log_dir", None) is None:
        on_step_callback_cfg["log_dir"] = save_path
    on_step_callback = hydra.utils.instantiate(on_step_callback_cfg, eval_env=eval_env)
    return on_step_callback

def load_env(path, venv):
    env = ExtendedVecNormalize.load(path, venv=venv)
    return env

def compute_consecutive_cost_chains_stats(state_safe_flags, ep_start_indices):
    """
    Compute stats about consecutive cost chains in the episode

    :param state_safe_flags: (np.ndarray) the state safe flags indicating if the state is safe or not (shape: (n_envs, n_steps))
    :param ep_start_indices: (np.ndarray) indices fpr state_safe_flags, where an episode started (shape: (n_envs, n_steps))
    
    
    :return: consecutive_unsafe_steps_freq: (np.ndarray) the frequency of consecutive unsafe steps 
                (e.g. [3, 5] means that there are 3 consecutive cost steps with length 1 and 5 consecutive cost steps with length 2 respectively)
    :return: total_unsafe_steps: (int) the total number of unsafe steps
    :return: max_consecutive_unsafe_steps_per_env: (list) the maximum length of consecutive unsafe steps per environment
    :return: normalized_max_consecutive_unsafe_steps: (float) the expected maximum length of consecutive unsafe steps normalized by the length of the respective trajectory
    """
    state_safe_flags = np.array(state_safe_flags)
    consecutive_unsafe_steps_frequencies_per_env = []
    max_consecutive_unsafe_steps_per_env = []
    total_unsafe_steps = 0
    max_consecutive_unsafe_steps_normalized = []
    
    for env_idx, episode_start_indices_env in enumerate(ep_start_indices):
        state_safe_flags_env = state_safe_flags[env_idx]
        total_unsafe_steps += state_safe_flags_env.size - np.sum(state_safe_flags_env)
        # append the last index + 1 for the theoretic start index of the next episode
        episode_start_indices_env = np.insert(episode_start_indices_env, episode_start_indices_env.shape[0], len(state_safe_flags[env_idx]))
        consecutive_unsafe_steps_count_per_env = []

        for i, episode_start in enumerate(episode_start_indices_env):
            # counts how long consecutive unsafe step trajectories are
            consecutive_unsafe_steps_count = []
            if i > len(episode_start_indices_env) - 2:
                break
            next_episode_start = episode_start_indices_env[i+1]
            state_safe_flags_episode = state_safe_flags_env[episode_start:next_episode_start]
            for k, g in groupby(state_safe_flags_episode):
                if k == 0:
                    lenght = len(list(g))
                    consecutive_unsafe_steps_count.append(lenght)
                    consecutive_unsafe_steps_count_per_env.append(lenght)
            # calc the max length of consecutive unsafe steps per trajectory divided by the length or the respective trajectory
            max_consecutive_unsafe_steps_normalized.append(max(consecutive_unsafe_steps_count, default=0.0) / len(state_safe_flags_episode))
            
        if len(consecutive_unsafe_steps_count_per_env) > 0:
            max_consecutive_unsafe_steps_per_env.append(max(consecutive_unsafe_steps_count_per_env, default=0))
            # create full histogram
            bincount = np.bincount(np.array(consecutive_unsafe_steps_count_per_env, dtype=np.int32))
            # [1:] to remove 0s
            consecutive_unsafe_steps_frequencies_per_env.append(bincount[1:])
        else:
            max_consecutive_unsafe_steps_per_env.append(0)
            max_consecutive_unsafe_steps_normalized.append(0.0)
            consecutive_unsafe_steps_frequencies_per_env.append(np.zeros(1, dtype=np.int32))

    # currently ignore multi dimensional cost
    # consecutive_unsafe_steps_costs = np.sum(consecutive_unsafe_steps_freq_per_env, axis=0)
            
    # sum over all environments (therefore pad with 0s)
    max_length = max([len(consecutive_unsafe_steps_freq) for consecutive_unsafe_steps_freq in consecutive_unsafe_steps_frequencies_per_env])
    consecutive_unsafe_steps_frequencies_per_env = np.array([np.pad(consecutive_unsafe_steps_freq, (0, max_length - len(consecutive_unsafe_steps_freq))) for consecutive_unsafe_steps_freq in consecutive_unsafe_steps_frequencies_per_env])
    consecutive_unsafe_steps_freq = np.sum(consecutive_unsafe_steps_frequencies_per_env, axis=0)
    normalized_max_consecutive_unsafe_steps = np.max(max_consecutive_unsafe_steps_normalized)

    return consecutive_unsafe_steps_freq, total_unsafe_steps, max_consecutive_unsafe_steps_per_env, normalized_max_consecutive_unsafe_steps

def store_heatmap(x, y = None, title = None, x_label = None, y_label = None, save_path = None, draw_full_problem_method = None, rectify_heatmap = True):
    """
    Create a heatmap from the data
    :param x: (np.ndarray, optional) the x data
    :param y: (np.ndarray, optional) the y data
    :param title: (str, optional) the title of the plot
    :param x_label: (str, optional) the label of the x axis
    :param y_label: (str, optional) the label of the y axis
    :param save_path: (str, optional) the path to save the plot
    :param draw_full_problem_method: (function, optional) a method to draw the full problem
    :param rectify_heatmap: (bool, optional) whether to rectify the heatmap
    """

    is_1D = False
    x = np.array(x).flatten()
    if y is None:
        is_1D = True
    else:
        y = np.array(y).flatten()

    fig = plt.figure(figsize=(12, 5))
    axs = None
    cbar_label = "Frequency"
    bins = 100
    
    # plot data
    if is_1D:
        ax = fig.add_subplot(111)
        axs = [ax]
        ax.hist(x, bins=bins)
        ax.grid(True)
        ax.set_xlabel(x_label)
        ax.set_ylabel(cbar_label)
    else:
        axs = fig.subplots(1, 2)
        
        if draw_full_problem_method is not None:
            x_range = [np.min(x), np.max(x)]
            y_range = [np.min(y), np.max(y)]
            draw_full_problem_method(x_range=x_range, y_range=y_range, color_infeasible_region=True, axs=axs)

        # set cmin to 0.001 to set 0 values to white
        hist2d = axs[0].hist2d(x, y, bins=bins, cmap=plt.cm.jet, cmin=0.001)
        axs[0].grid(True)
        axs[0].set_xlabel(f"{x_label} (Dimension 0)")
        axs[0].set_ylabel(f"{y_label} (Dimension 1)")

        # hist2d returns 4 values, the fourth is mappable for colorbar
        cbar_0 = fig.colorbar(hist2d[3], ax=axs[0], orientation='vertical')
        cbar_0.set_label(cbar_label)

        gridsize = 70
        hb = axs[1].hexbin(x, y, gridsize=gridsize, cmap=plt.cm.jet, bins=None, mincnt=1)
        axs[1].set_xlabel(f"{x_label} (Dimension 0)")
        axs[1].set_ylabel(f"{y_label} (Dimension 1)")
        cbar_1 = fig.colorbar(hb, ax=axs[1], orientation='vertical')
        cbar_1.set_label(cbar_label)

        if rectify_heatmap:
            axs[0].set_aspect('equal', 'box')
            axs[1].set_aspect('equal', 'box')
    
    fig.suptitle(title)
    axs[0].legend(bbox_to_anchor=(0, 1.01, 1, 0.2), loc="lower left", borderaxespad=0, ncol=2)
    axs[1].legend(bbox_to_anchor=(0, 1.01, 1, 0.2), loc="lower left", borderaxespad=0, ncol=2)
    fig.tight_layout()
    
    if save_path is not None:
        path = save_path
        if ".png" not in path:
            path = f"{path}_heatmap.png"
    else:
        path = "heatmap.png"
    
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

def create_consecutive_cost_plot(consecutive_cost_steps_freq_dataset, total_unsafe_steps, max_consecutive_unsafe_steps_length, total_rollout_timesteps, x_label = None, x_tick_labels = None, save_path = None):
    """
    Create a plot of the consecutive cost chains for relevant iterations given by the data

    :param consecutive_cost_steps_freq_dataset: (np.ndarray) the frequency of consecutive unsafe steps
    :param total_unsafe_steps: (int) the total number of unsafe steps
    :param max_consecutive_unsafe_steps_length: (int) the maximum length of consecutive unsafe steps per iteration
    :param total_rollout_timesteps: (int) the total number of rollout timesteps per relevant iteration
    :param x_label: (str, optional) the label of the x axis (Filling in the blank in "Iterations in steps of __")
    :param x_tick_labels: (list, optional) the labels of the ticks on the x axis (each tick should be the iteration where the data was collected)
    :param save_path: (str, optional) the path to save the plot
    """
    dataset_length = len(consecutive_cost_steps_freq_dataset)
    assert len(total_rollout_timesteps) == dataset_length, "total_rollout_timesteps and consecutive_cost_steps_freq_dataset must have the same length"
    assert len(total_unsafe_steps) == dataset_length, "total_unsafe_steps and consecutive_cost_steps_freq_dataset must have the same length"
    assert len(max_consecutive_unsafe_steps_length) == dataset_length, "max_consecutive_unsafe_steps_length and consecutive_cost_steps_freq_dataset must have the same length"
    if x_tick_labels is not None:
        assert len(x_tick_labels) == dataset_length, f"x_tick_labels and consecutive_cost_steps_freq_dataset must have the same length. Lenghts {len(x_tick_labels)} and {dataset_length} given."

    # create data for violinplot where each value is repeated as often as its frequency to correctly visualize the cost share of each cost chain length
    dataset = []
    # iterate over all relevant iterations (given in consecutive_cost_steps_freq_dataset)
    for consecutive_cost_steps_freq_data in consecutive_cost_steps_freq_dataset:
        # prepare data to correctly represent the cost shares from every consecutive cost chain length
        consecutive_cost_steps_iteration = []
        for value, frequency in enumerate(consecutive_cost_steps_freq_data):
            # combine all cost chains with length >= 10 to one bin for more precise boxplot in range 0-10
            if value < 9:
                consecutive_cost_steps_iteration.extend([value+1] * frequency * (value+1))
            else:
                consecutive_cost_steps_iteration.extend([10] * frequency * (value+1))
        if len(consecutive_cost_steps_iteration) == 0:
            dataset.append([0])
        else:
            dataset.append(consecutive_cost_steps_iteration)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharey=True)
    ax.violinplot(dataset, showmedians=False , showmeans=True, showextrema=True)
    
    # draw total costs in background with seperate y axis
    labels_x_pos = np.arange(1, len(dataset)+1)
    ax2 = ax.twinx()
    ax2.fill_between(labels_x_pos, 0, np.array(total_unsafe_steps) / total_rollout_timesteps, color="b", alpha=0.2)
    ax2.set_ylabel("Unsafe steps share of total rollout timestep")

    # write total number of unsafe steps for each iteration in plot above each violin
    for i in range(len(total_unsafe_steps)):
        if total_unsafe_steps[i] == 0:
            continue
        height = max_consecutive_unsafe_steps_length[i] if max_consecutive_unsafe_steps_length[i] < 10 else 10
        ax.text(i+1, height + 0.5, f"{total_unsafe_steps[i]}", ha='center', va='center', color='black')
    
    ax.set_xticks(np.arange(1, dataset_length+1), labels=x_tick_labels)
    if x_label is None:
        x_label = "Iterations"
    else:
        x_label = f"Iterations in steps of {x_label}"
    ax.set_xlabel(x_label)
    y_labels = np.arange(0, 11).astype(str)
    y_labels[-1] = "10+"
    ax.set_yticks(np.arange(0, 11), labels=y_labels)
    ax.set_ylim(0, 11)
    ax.set_title("Consecutive unsafe steps violinplot")
    ax.set_ylabel("Length of consecutive unsafe steps")
    ax.legend()
    fig.tight_layout()
    if save_path is None:
        save_path = f'cost_chain_info_plot.png'
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def get_schedule_from_name(schedule_name):
    """
    Get the schedule function from the name
    :param schedule_name: (str) the name of the schedule
    :return: (callable) the scheduler
    """
    if schedule_name is None:
        return None

    if schedule_name == "linear":
        return get_linear_fn
    elif schedule_name == "logistic":
        return get_logistic_fn
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")
    
def get_logistic_fn(start: float, end: float, end_fraction: float, middle_point: float = 0.5) -> Schedule:
    """
    Create a function that interpolates logsiticly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :params middle_point: point or progress where the logistic function evaluates to 0.5 (default: 0.5, allowed range: 0-1)
    :return: logistic schedule function.
    """
    assert 0 <= middle_point <= 1, "middle_point must be in [0, 1]"

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start * 1 / (1 + np.exp(10 * ((1-progress_remaining) - middle_point)))

    return func

def cvar_from_distribution(distribution, alphas):
    """
    Compute the Conditional Value at Risk (CVaR) of a distribution
    :param distribution: (np.ndarray) the distribution of the cost returns
    :param alpha: (float) the risk level
    :return: (dict) the CVaR at the given risk levels
    """

    # Sort the cost returns (ascending)
    sorted_costs = np.sort(distribution)
    print(f"Total number of eval episodes: {len(sorted_costs)}")
    print(f"Mean of costs: {np.mean(sorted_costs)}")
    print(f"Unsafest episode cost: {sorted_costs[-1]}")
    cvar_results = {}
    for alpha in alphas:
        assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
        # Compute the index of the alpha percentile
        alpha_index = int((1-alpha) * len(sorted_costs))
        # Compute the CVaR by only respecting values above the alpha percentile
        cvar = np.mean(sorted_costs[alpha_index:])
        cvar_results[alpha] = cvar
    return cvar_results

