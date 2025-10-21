import os
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import set_random_seed
from saferl.common.utils import create_env, create_training_model, create_on_step_callback, evaluate_after_training
from stable_baselines3.common.callbacks import CallbackList
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    print("STARTED")
    # set random seed for reproducibility (cpu and gpu)
    set_random_seed(cfg.seed)
    save_path = os.getcwd()
    display = None
    # if "save_video" in cfg.callback.on_step_callback and cfg.callback.on_step_callback.save_video:
        # Create virtual display for video recording
        # from pyvirtualdisplay import Display
        # display = Display(visible=0)
        # display.start()

    # if cfg.use_wandb:
    #     import wandb
    #     from wandb.integration.sb3 import WandbCallback
    #     run = wandb.init(
    #         project="SafeRL",  # Specify your project
    #         config=OmegaConf.to_container(config, resolve=True),
    #         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #         monitor_gym=True,  # auto-upload sb3's videos
    #     )

    # Initialize the environment
    env = create_env(cfg.env.train_env, cfg.seed,
        norm_obs=cfg.norm_obs, norm_act=cfg.norm_act, norm_reward=cfg.norm_reward, norm_cost=cfg.norm_cost, 
        monitor=True, save_path=save_path, env_kwargs=cfg.env.train_env.env_kwargs, num_env=cfg.env.train_env.num_env, use_multi_process = cfg.use_multi_process)
    
    print(f"{env.num_envs} training environments created")
    
    eval_env = create_env(cfg.env.eval_env, cfg.seed,
        norm_obs=cfg.norm_obs, norm_act=cfg.norm_act, norm_reward=cfg.norm_reward, norm_cost=cfg.norm_cost,
        monitor=False, training=False, env_kwargs=cfg.env.eval_env.env_kwargs, use_multi_process = False)
    print(f"{eval_env.num_envs} evaluation environments created")
    
    # find all the available callbacks by searching for substring "callback" in the config
    callback_list = []
    for key in cfg:
        if "callback" in key:
            callback_list.append(cfg[key])
    on_step_callback = CallbackList(
        [create_on_step_callback(callback.on_step_callback, eval_env=eval_env, save_path=save_path) for callback in callback_list])

    # Create the policy
    model = create_training_model(cfg.algorithm, env, tensorboard_log=save_path)

    # load model if specified
    if "load_model" in cfg and cfg.load_model is not None:
        model.set_parameters(cfg.load_model, device=cfg.device)

    print("Model created")
    model.learn(total_timesteps=cfg.env.total_timesteps, log_interval=1, tb_log_name=cfg.algorithm.algorithm_name, callback=on_step_callback)
    print("Training finished")
    model.save(os.path.join(save_path, "model"))
    
    if hasattr(env, "save") and isinstance(env, VecNormalize):
        env.save(os.path.join(save_path, "env.zip"))

    if "eval_after_training_num_episodes" in cfg and cfg.eval_after_training_num_episodes:
        print(f"Evaluating the model for {cfg.eval_after_training_num_episodes} episodes")
        evaluate_after_training(model, eval_env, num_episodes=cfg.eval_after_training_num_episodes, cvar_alphas=cvar_alphas, save_path=save_path)


    if "save_video" in cfg.callback.on_step_callback and cfg.callback.on_step_callback.save_video and display:
        display.stop()

if __name__ == "__main__":
    main()
