from safety_gymnasium import __register_helper

EGG_MANIPULATION_EPISODE_LENGTH = 400

__register_helper(
    env_id='SafeInvertedPendulumSwing-v0',
    entry_point='saferl.common.envs.inverted_pendulum:SafeInvertedPendulumSwingEnv',
    max_episode_steps=1000,
    disable_env_checker=True,
)
__register_helper(
    env_id='SafeInvertedPendulumMove-v0',
    entry_point='saferl.common.envs.inverted_pendulum:SafeInvertedPendulumMoveEnv',
    max_episode_steps=1000,
    disable_env_checker=True,
)

__register_helper(
    env_id='SafePendulumUpright-v0',
    entry_point='saferl.common.envs.classic_pendulum:SafePendulumUprightEnv',
    max_episode_steps=200,
)

__register_helper(
    env_id='SafePendulumTilt-v0',
    entry_point='saferl.common.envs.classic_pendulum:SafePendulumTiltEnv',
    max_episode_steps=200,
    disable_env_checker=True,
)

__register_helper(
    env_id='SafeEggManipulationFullDense-v0',
    entry_point='saferl.common.envs.manipulation:SafeEggManipulationTask',
    max_episode_steps=EGG_MANIPULATION_EPISODE_LENGTH,
    spec_kwargs={
        'target_position': 'random',
        'target_rotation': 'xyz',
        'touch_get_obs': 'sensordata',
        "reward_type": "dense",
    },
    disable_env_checker=True,
)
__register_helper(
    env_id='SafeEggManipulationFullSparse-v0',
    entry_point='saferl.common.envs.manipulation:SafeEggManipulationTask',
    max_episode_steps=EGG_MANIPULATION_EPISODE_LENGTH,
    spec_kwargs={
        'target_position': 'random',
        'target_rotation': 'xyz',
        'touch_get_obs': 'sensordata',
        "reward_type": "sparse",
    },
    disable_env_checker=True,
)

__register_helper(
    env_id='SafeEggManipulationRotateSparse-v0',
    entry_point='saferl.common.envs.manipulation:SafeEggManipulationTask',
    max_episode_steps=EGG_MANIPULATION_EPISODE_LENGTH,
    spec_kwargs={
        'target_position': 'ignore',
        'target_rotation': 'xyz',
        'touch_get_obs': 'sensordata',
        "reward_type": "sparse",
    },
    disable_env_checker=True,
)

__register_helper(
    env_id='SafeEggManipulationRotateDense-v0',
    entry_point='saferl.common.envs.manipulation:SafeEggManipulationTask',
    max_episode_steps=EGG_MANIPULATION_EPISODE_LENGTH,
    spec_kwargs={
        'target_position': 'ignore',
        'target_rotation': 'xyz',
        'touch_get_obs': 'sensordata',
        "reward_type": "dense",
    },
    disable_env_checker=True,
)

