import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv
import saferl.common.utils as utils
from stable_baselines3.common.utils import safe_mean
from typing import List, Tuple
import matplotlib.pyplot as plt
from gymnasium import spaces
from saferl.common.buffers import CostRolloutBuffer, CostReplayBuffer
import pickle
from saferl.common.utils import store_heatmap, create_consecutive_cost_plot, cvar_from_distribution

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``eval_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param eval_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
    It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, eval_freq: int, save_freq: int, log_dir: str, eval_env: GymEnv,
        save_video: bool = False, save_video_freq : int = None, num_eval_episodes: int = 1, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_video = save_video
        self.save_video_freq = save_video_freq if save_video_freq is not None else eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.log_dir = log_dir
        self.best_model_save_path = os.path.join(log_dir, 'best_model')
        self.save_path = os.path.join(log_dir, 'eval')
        self.best_train_ret = - np.inf
        self.best_train_cost = np.inf
        self.best_eval_ret = - np.inf
        self.best_eval_cost = np.inf
        if hasattr(eval_env, '_max_episode_steps'):
            self.episode_len = eval_env._max_episode_steps
        else:
            try:
                self.episode_len = eval_env.get_attr('_max_episode_steps')[0]
            except:
                self.episode_len = 1000
        self.eval_env = eval_env
        self.cumulative_cost = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # TODO: set dones to False due to unlimited task for PPO
        if self.num_timesteps % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    utils.sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )
            
            save_video = self.save_video
            if self.num_timesteps % self.save_video_freq != 0:
                save_video = False
            
            res = utils.evaluate(
                self.model, 
                self.eval_env,
                save_video=save_video,
                save_path=os.path.join(self.save_path, "{}".format(self.num_timesteps)),
                num_episodes=self.num_eval_episodes,
                episode_len=self.episode_len, 
                deterministic=True,
                seed=self.num_timesteps)
            
            self.logger.record('eval/episode_ret_mean', np.mean(res["ret"]))
            self.logger.record('eval/episode_ret_std', np.std(res["ret"]))
            self.logger.record('eval/episode_cost', np.mean(res["cost"]))
            self.logger.record('eval/episode_cost_std', np.std(res["cost"]))
            self.logger.record('eval/num_safe_episodes', np.sum(res["is_safe"]))
            self.logger.record('eval/avg_episode_len', np.mean(res["len"]))
            self.logger.record('eval/max_consecutive_cost_steps', np.max(res["max_consecutive_cost_steps"]))
            # self.logger.record('eval/max_consecutive_cost_steps_mean', np.mean(res["max_consecutive_cost_steps"]))
            self.logger.dump(step=self.num_timesteps)

            self.cumulative_cost += np.mean(res["cost"])
            self.logger.record('eval/cumulative_cost', self.cumulative_cost)

            # Saving for the best models
            if np.mean(res["ret"]) > self.best_eval_ret:
                self.best_eval_ret = np.mean(res["ret"])
                self.model.save(os.path.join(self.best_model_save_path, "best_ret_eval_model.zip"))
            if np.mean(res["cost"]) < self.best_eval_cost:
                self.best_eval_cost = np.mean(res["cost"])
                self.model.save(os.path.join(self.best_model_save_path, "best_cost_eval_model.zip"))

        if self.num_timesteps % self.save_freq == 0:
            self.model.save(os.path.join(self.save_path, "{}".format(self.num_timesteps), "model.zip"))
            # only try saving if training env has save method
            if hasattr(self.training_env, "save"):
                self.training_env.save(os.path.join(self.save_path, "{}".format(self.num_timesteps), "env.zip"))
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) <= 0:
            return None

        if np.mean(self.model.ep_info_buffer[-1]['r']) > self.best_train_ret:
            self.best_train_ret = np.mean(self.model.ep_info_buffer[-1]['r'])
            self.model.save(os.path.join(self.best_model_save_path, "best_ret_eval_model.zip"))
        if np.mean(self.model.ep_info_buffer[-1]["cost"]) < self.best_train_cost:
            self.best_train_cost = np.mean(self.model.ep_info_buffer[-1]["cost"])
            self.model.save(os.path.join(self.best_model_save_path, "best_cost_eval_model.zip"))

class HeatmapCallback(BaseCallback):
    def __init__(self, heatmap_freq: int, 
                 log_dir: str, 
                 rectify_heatmap = True, 
                 heatmap_type : str = "State", 
                 heatmap_freq_change_info : List[Tuple] = None, 
                 mode : str = "both", 
                 combine_pre_learning_starts_data=False,
                 always_calc_stats=False,
                 emcc_split_percentage=0.33,
                 eval_env : GymEnv = None, 
                 verbose=1
                 ):
        """
        Callback for creating heatmaps of the state or action space during training.

        :param heatmap_freq: (int) Frequency of creating heatmaps
        :param log_dir: (str) Path to the folder where the heatmaps will be saved.
        :param heatmap_type: (str) Type of heatmap. Valid types are "State" or "Action"
        :param heatmap_freq_change_info: (List[Tuple]) List of tuples with the first element being the new frequency and the second element being the iteration at which the frequency should be changed
        :param mode: (str) Mode of the callback. Valid modes are "data_only", "plot_only" or "both"
            data_only: Only store data for iterations defined by heatmap_freq and do not create plots
            plot_only: Only create plots for iterations defined by heatmap_freq and do not store data for the heatmap
            both: Store data and create plots for iterations defined by heatmap_freq
        :param combine_pre_learning_starts_data: (bool) Whether to combine data before learning starts (only for off-policy algorithms)
        :param always_calc_stats: (bool) Whether to always calculate stats of consecutive costs for logging (even if not plotting or storing data)
        :param eval_env: (GymEnv) Evaluation environment (only for consistency with other callbacks, not used in this callback)
        :param verbose: (int)

        """
        
        super(HeatmapCallback, self).__init__(verbose)

        # only for consistency with other callbacks
        self.eval_env = eval_env

        # defines whether the heatmap axis should be equally scaled (True) or not (False)
        self.rectify_heatmap = rectify_heatmap

        # keep track of current iteration as local() is too slow
        self.iteration = 0

        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'heatmaps')
        # heatmap type with option "State" or "Action"
        self.heatmap_type : str = heatmap_type
        if self.heatmap_type not in ["State", "Action"]:
            raise ValueError("Invalid heatmap type. Valid types are 'State' or 'Action'")
    
        self.heatmap = None
        self.heatmap_freq = heatmap_freq
        self.active_heatmap_freq = heatmap_freq
        self.heatmap_freq_change_info = heatmap_freq_change_info
        self.iteration_offset = 0
        self.mode = mode
        assert self.mode in ["data_only", "plot_only", "both"], "Invalid mode. Valid modes are 'data_only', 'plot_only' or 'both'"
        
        # for off policy algorithms to show all data collected before learning started
        self.combine_pre_learning_starts_data = combine_pre_learning_starts_data
        # always calculate stats for consecutive costs for logging
        self.always_calc_stats = always_calc_stats

        # heatmap data over iterations
        # labels for the x axis
        self.labels = []
        # total unsafe steps at each iteration relevant for heatmap
        self.total_unsafe_steps = []
        # maximum length of consecutive unsafe steps at each iteration relevant for heatmap
        self.max_consecutive_unsafe_steps_length = []
        # frequency of consecutive unsafe steps at each iteration relevant for heatmap
        self.consecutive_unsafe_steps_freq = []
        # total rollout timesteps at each iteration relevant for heatmap
        self.total_rollout_timesteps = []
        # track the number of done episodes per rollout
        self.episodes_done = 0
        # number of episodes done total
        self.total_episodes_done = 0
        # total costs from all training rollouts
        self.total_costs_full_training = 0

        # expected max consecutive cost steps
        self.emcc = []
        self.emcc_split_percentage = emcc_split_percentage
        # self.emcc_split_iteration = None
        # offset from the last emcc split
        self.training_progress_offset = 0
        # store cvar of emcc for different shares of training progress
        self.emcc_alpha_per_share = {}

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_start(self) -> None:
        self.episodes_done = 0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        self.episodes_done += np.sum(dones)
        self.total_costs_full_training += np.sum(self.locals["costs"])
        return True

    def on_rollout_end(self) -> None:
        self.total_episodes_done += self.episodes_done

        # change to new heatmap frequency if iteration is reached
        if self.heatmap_freq_change_info is not None:
            for new_freq, freq_change_iteration in self.heatmap_freq_change_info:
                if freq_change_iteration == self.iteration:
                    self.active_heatmap_freq = new_freq
                    self.iteration_offset = self.iteration
                    break
        
        iteration_reached = ((self.iteration-self.iteration_offset) % self.active_heatmap_freq == 0) or self.combine_pre_learning_starts_data
        last_iteration = self.locals["self"].num_timesteps >= self.locals["total_timesteps"]

        # log total costs from all training until this timestep divided by the total number of env interactions
        self.logger.record("rollout/rho_c", self.total_costs_full_training / self.locals["self"].num_timesteps)

        # only do heatmaps for relevant iterations (or last iteration)
        no_heatmap_creation = not iteration_reached and not last_iteration
        # in case of no_heatmap_creation here we could return to save computation time, therefore only calculate stats if always_calc_stats is True
        if not self.always_calc_stats and no_heatmap_creation:
            self.iteration += 1
            return None            
        
        mean_ep_cost_return = 0
        mean_ep_length = 0
        
        title = ""
        # check if buffer is rollout or replay buffer
        # on-policy data
        if "rollout_buffer" in self.locals:
            buffer = self.locals["rollout_buffer"]
            assert isinstance(buffer, CostRolloutBuffer), "Rollout buffer is not of type CostRolloutBuffer"
            rollout_timesteps = self.locals["n_steps"] * self.locals["env"].num_envs
            consecutive_unsafe_steps_freq_at_iteration, total_unsafe_steps_at_iteration, max_consecutive_unsafe_steps_per_env, normalized_max_consecutive_unsafe_steps = buffer.compute_consecutive_cost_chains()
            start_pos = 0
            if len(buffer.episode_stats) > 0:
                mean_ep_cost_return = np.round(np.mean(buffer.episode_stats["ep_cost_returns_undiscounted"]), 2)
                mean_ep_length = np.round(np.mean(buffer.episode_stats["ep_lengths"]), 2)

            # ensure that combine_pre_learning_starts_data is false. because it is only used for off-policy data
            self.combine_pre_learning_starts_data = False


        # off-policy data
        elif "replay_buffer" in self.locals:
            buffer = self.locals["replay_buffer"]
            assert isinstance(buffer, CostReplayBuffer), "Replay buffer is not of type CostReplayBuffer"
            rollout_timesteps = self.locals["num_collected_steps"]*self.locals["env"].num_envs
            start_pos = buffer.pos - rollout_timesteps
            if start_pos < 0:
                start_pos = buffer.buffer_size + start_pos

            if self.combine_pre_learning_starts_data:
                assert self.locals['learning_starts'] >= 0, "learning_starts not found in locals"
                if self.locals["self"].num_timesteps >= self.locals["learning_starts"]:
                    self.combine_pre_learning_starts_data = False
                    start_pos = 0
                    title += "Pre-learning data (stats only for recent rollout data)\n"
                # enables calculating stats per rollout before learning starts
                elif self.always_calc_stats:
                    no_heatmap_creation = True
                else:
                    self.iteration += 1
                    return None


            consecutive_unsafe_steps_freq_at_iteration, total_unsafe_steps_at_iteration, max_consecutive_unsafe_steps_per_env, normalized_max_consecutive_unsafe_steps = buffer.compute_consecutive_cost_chains(start_pos)            
            ep_info_buffer = self.locals["self"].ep_info_buffer
            if len(ep_info_buffer) > 0:
                total_episodes = self.episodes_done
                # only get the episodes from this rollout (assume that less episodes than size (100) of ep_info_buffer have been collected, which is reasonable for off-policy data)
                total_episodes = total_episodes if total_episodes < ep_info_buffer.maxlen else ep_info_buffer.maxlen
                rollout_episodes = list(ep_info_buffer)[-total_episodes:]
                mean_ep_cost_return = np.round(np.mean([np.sum(episode_info["cost"]) for episode_info in rollout_episodes]), 2)
                mean_ep_length = np.round(np.mean([episode_info["l"] for episode_info in rollout_episodes]), 2)

        else:
            raise ValueError("No valid buffer found. Callback only works with CostRolloutBuffer or CostReplayBuffer")
        
        self.logger.record("rollout/max_consecutive_cost_steps", len(consecutive_unsafe_steps_freq_at_iteration))
        # self.logger.record("rollout/avg_maxima_consecutive_cost_steps", safe_mean(max_consecutive_unsafe_steps_per_env))
        # self.logger.record("rollout/expected_max_consecutive_cost_steps_normalized", expected_normalized_max_consecutive_unsafe_steps)

        self.emcc.append(normalized_max_consecutive_unsafe_steps)
        current_training_progress = self.locals["self"].num_timesteps / self.locals["total_timesteps"]
        # if len(self.emcc) % self.emcc_split_iteration == 0:
        if current_training_progress - self.training_progress_offset > self.emcc_split_percentage:
            self.logger.record(f"rollout/emcc_splitted_{self.emcc_split_percentage}", safe_mean(self.emcc))
            # log cvar of emcc
            alphas = [0.1, 0.5, 0.9]
            cvar_results = cvar_from_distribution(self.emcc, alphas)
            for alpha, cvar in cvar_results.items():
                self.logger.record(f"rollout/emcc_splitted_{self.emcc_split_percentage}_cvar_{alpha}", cvar)
            self.emcc = []
            self.training_progress_offset = current_training_progress
            self.emcc_alpha_per_share[self.training_progress_offset] = cvar_results

        if no_heatmap_creation:
            self.iteration += 1
            return None

        self.total_unsafe_steps.append(total_unsafe_steps_at_iteration)
        self.max_consecutive_unsafe_steps_length.append(len(consecutive_unsafe_steps_freq_at_iteration))
        self.consecutive_unsafe_steps_freq.append(consecutive_unsafe_steps_freq_at_iteration)       
        self.total_rollout_timesteps.append(rollout_timesteps)
        self.labels.append(self.iteration)

        if last_iteration:
            # write log file with all important training metrics:
            # - total costs from all training until this timestep divided by the total number of env interactions (rho_c)
            # - average episodic length of full training
            # - CVar_EMCC_0.1, CVar_EMCC_0.5, CVar_EMCC_0.9 respectively for defined share of training progress

            path = os.path.join(self.log_dir, "training_metrics.txt")
            with open(path, "w") as f:
                f.write("Training metrics\n")
                f.write(f"Cost rate rho_c: {self.total_costs_full_training / self.locals['self'].num_timesteps}\n")
                if self.total_episodes_done > 0:
                    f.write(f"Average episodic length of full training: {self.locals['self'].num_timesteps / self.total_episodes_done}\n")
                for training_progress, cvar_results in self.emcc_alpha_per_share.items():
                    f.write(f"Training progress: {training_progress}\n")
                    for alpha, cvar in cvar_results.items():
                        f.write(f"CVar_EMCC_{alpha}: {cvar}\n")
                    f.write("\n")

        if self.mode not in ["plot_only"]:
            # store data as pickle file
            with open(f"consecutive_costs_per_iteration.pkl", "wb") as f:
                pickle.dump([self.labels, self.consecutive_unsafe_steps_freq, self.total_unsafe_steps, self.total_rollout_timesteps], f)
        
        # only plot if plotting wanted
        if self.mode not in ["data_only"]:
            # calc information for title
            cost_steps_percentage = np.round(total_unsafe_steps_at_iteration / rollout_timesteps * 100, 2)
            title += "Rollout heatmap - iteration {} (after {} env interactions)\n {}/{} states unsafe ({}%) \n ~{} states unsafe per episode (length ~{})".format(self.iteration, self.num_timesteps, total_unsafe_steps_at_iteration, rollout_timesteps, cost_steps_percentage, mean_ep_cost_return, mean_ep_length)
            self.store_rollout_heatmap(buffer, title, start_pos=start_pos, end_pos=buffer.pos)
            # update plot
            if iteration_reached:
                save_path = os.path.join(self.log_dir, f'cost_chain_info_plot.png')
                create_consecutive_cost_plot(self.consecutive_unsafe_steps_freq, 
                                             self.total_unsafe_steps, 
                                             self.max_consecutive_unsafe_steps_length, 
                                             self.total_rollout_timesteps,
                                             x_label=self.heatmap_freq,
                                             x_tick_labels=self.labels, 
                                             save_path=save_path, 
                                             )
    
        self.iteration += 1
    
    def store_rollout_heatmap(self, buffer, title : str, start_pos : int = 0, end_pos : int = 1) -> None:
        """
        Store heatmap of last rollout
        :param buffer: buffer with data of last rollout
        :param start_pos: start position relevant data in buffer (needed for handling off-policy data)
        :param end_pos: end position relevant data in buffer (needed for handling off-policy data)

        Note: The first two dimensions are used for the heatmap
        """

        final_end_pos = end_pos
        if end_pos < start_pos:
            end_pos = buffer.size
        
        y = None
        heatmap_data = buffer.observations if not self.heatmap_type == "Action" else buffer.actions
        heatmap_data_type = "Observation" if not self.heatmap_type == "Action" else "Action"
        x = heatmap_data[start_pos:end_pos,:,0]
        if end_pos != final_end_pos:
            x = np.concatenate((x, heatmap_data[0:final_end_pos, :, 0]), axis=0)

        if heatmap_data.shape[-1] != 1:
            y = heatmap_data[start_pos:end_pos, :, 1]
            if end_pos != final_end_pos:
                y = np.concatenate((y, heatmap_data[0:final_end_pos, :, 1]), axis=0)

        # get function for drawing the full problem if available
        draw_full_problem = None
        if hasattr(self.eval_env.envs[0], "draw_full_problem"):
            draw_full_problem = self.eval_env.envs[0].draw_full_problem

        path = os.path.join(self.save_path, f'Rollout_heatmap_iteration_{self.iteration}.pdf')
        store_heatmap(x, y, title, x_label=heatmap_data_type, y_label=heatmap_data_type, save_path=path, draw_full_problem_method=draw_full_problem, rectify_heatmap=self.rectify_heatmap)


class DummyCallback(BaseCallback):
    def __init__(self, verbose=0, log_dir: str = None, eval_env: GymEnv = None):
        super(DummyCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        return None