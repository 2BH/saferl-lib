# SafeRL-Lib Configuration System

This directory contains the configuration files for SafeRL-Lib experiments. The configuration system is built on top of Hydra and provides a flexible way to manage experiments, hyperparameters, and environment settings.

## Directory Structure

```
configs/
├── experiments/           # Unified experiment configurations
│   ├── SafetyAntVelocity_sac_lag.yaml
│   ├── SafetyCarCircle1_trpo_lag.yaml
│   └── SafetyHalfCheetahVelocity_csac_lb.yaml
├── algorithm/            # Algorithm-specific configurations (legacy)
├── env/                  # Environment-specific configurations (legacy)
├── callback/             # Callback configurations
├── main.yaml            # Main configuration file
└── README.md            # This file
```

## New Unified Configuration System

The new system combines environment, algorithm, and training configurations into single files per experiment. This approach provides several benefits:

1. **Self-contained experiments**: Each experiment file contains all necessary parameters
2. **Easy parameter sweeps**: Override any parameter without modifying files
3. **Better organization**: Clear separation between different experiment types
4. **Reduced complexity**: No need to manage multiple configuration files

### Experiment Configuration Structure

Each experiment configuration file follows this structure:

```yaml
# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================
seed: 0
device: "cuda"
verbose: 1
# ... other global settings

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
env:
  total_timesteps: 3000000
  episode_len: 1000
  train_env:
    env_name: SafetyAntVelocity-v1
    env_kwargs: 
      camera_id: 0
    num_env: ${num_env}
  eval_env:
    env_name: SafetyAntVelocity-v1
    env_kwargs: 
      render_mode: rgb_array
      camera_id: 0

# =============================================================================
# ALGORITHM CONFIGURATION
# =============================================================================
algorithm:
  model:
    _target_: saferl.algorithms.sac_lag.SAC_LAG
    _convert_: partial
    # ... algorithm parameters
  algorithm_name: sac_lag
  policy_class: SACwithCostPolicy
  noise: null

# =============================================================================
# CALLBACK CONFIGURATION
# =============================================================================
callback:
  on_step_callback:
    _target_: saferl.common.callbacks.EvalCallback
    # ... callback parameters

# =============================================================================
# HYDRA CONFIGURATION
# =============================================================================
hydra:
  mode: RUN
  # ... hydra settings
```

## Usage Examples

### Running a Specific Experiment

```bash
# Run SafetyAntVelocity with SAC-Lag
python -m saferl.examples.run_experiment experiment=SafetyAntVelocity_sac_lag

# Run SafetyCarCircle1 with TRPO-Lag
python -m saferl.examples.run_experiment experiment=SafetyCarCircle1_trpo_lag
```

### Parameter Overrides

```bash
# Change cost constraint
python -m saferl.examples.run_experiment experiment=SafetyAntVelocity_sac_lag algorithm.model.cost_constraint=[10.0]

# Change learning rate
python -m saferl.examples.run_experiment experiment=SafetyAntVelocity_sac_lag algorithm.model.learning_rate=1e-4

# Change environment parameters
python -m saferl.examples.run_experiment experiment=SafetyAntVelocity_sac_lag env.train_env.env_kwargs.camera_id=1

# Change training settings
python -m saferl.examples.run_experiment experiment=SafetyAntVelocity_sac_lag env.total_timesteps=1000000 seed=42
```

### Parameter Sweeps

```bash
# Sweep over multiple seeds
python -m saferl.examples.run_experiment --multirun experiment=SafetyAntVelocity_sac_lag seed=0,1,2,3,4

# Sweep over cost constraints
python -m saferl.examples.run_experiment --multirun experiment=SafetyAntVelocity_sac_lag algorithm.model.cost_constraint=[5.0,10.0,15.0]

# Sweep over learning rates
python -m saferl.examples.run_experiment --multirun experiment=SafetyAntVelocity_sac_lag algorithm.model.learning_rate=1e-4,3e-4,1e-3
```

### Custom Experiment Configurations

You can create custom experiment configurations by:

1. **Copying an existing configuration**:
   ```bash
   cp experiments/SafetyAntVelocity_sac_lag.yaml experiments/MyCustomExperiment.yaml
   ```

2. **Modifying the parameters** in the new file

3. **Running the experiment**:
   ```bash
   python -m saferl.examples.run_experiment experiment=MyCustomExperiment
   ```

## Available Experiments

### SafetyAntVelocity
- **SAC-Lag**: `SafetyAntVelocity_sac_lag.yaml`
- **TRPO-Lag**: `SafetyAntVelocity_trpo_lag.yaml` (create as needed)
- **CSAC-LB**: `SafetyAntVelocity_csac_lb.yaml` (create as needed)

### SafetyCarCircle1
- **TRPO-Lag**: `SafetyCarCircle1_trpo_lag.yaml`
- **SAC-Lag**: `SafetyCarCircle1_sac_lag.yaml` (create as needed)

### SafetyHalfCheetahVelocity
- **CSAC-LB**: `SafetyHalfCheetahVelocity_csac_lb.yaml`
- **SAC-Lag**: `SafetyHalfCheetahVelocity_sac_lag.yaml` (create as needed)

## Configuration Parameters

### Global Settings
- `seed`: Random seed for reproducibility
- `device`: Device to use ("cuda" or "cpu")
- `verbose`: Verbosity level (0-2)
- `save_freq`: Frequency to save model checkpoints
- `eval_freq`: Frequency to evaluate the model
- `save_video_freq`: Frequency to save videos
- `save_video`: Whether to save videos
- `num_eval_episodes`: Number of episodes for evaluation
- `norm_obs`: Whether to normalize observations
- `norm_act`: Whether to normalize actions
- `norm_reward`: Whether to normalize rewards
- `norm_cost`: Whether to normalize costs
- `num_env`: Number of parallel environments
- `use_multi_process`: Whether to use multiprocessing

### Environment Settings
- `env.total_timesteps`: Total training timesteps
- `env.episode_len`: Maximum episode length
- `env.train_env.env_name`: Training environment name
- `env.train_env.env_kwargs`: Training environment arguments
- `env.train_env.num_env`: Number of training environments
- `env.eval_env.env_name`: Evaluation environment name
- `env.eval_env.env_kwargs`: Evaluation environment arguments

### Algorithm Settings
- `algorithm.model._target_`: Algorithm class to instantiate
- `algorithm.model.cost_constraint`: Cost constraint values
- `algorithm.model.learning_rate`: Learning rate
- `algorithm.model.batch_size`: Batch size
- `algorithm.model.buffer_size`: Replay buffer size
- `algorithm.model.policy_kwargs`: Policy network architecture
- `algorithm.algorithm_name`: Algorithm name for logging
- `algorithm.policy_class`: Policy class to use

## Migration from Legacy System

If you're migrating from the old configuration system:

1. **Old way**:
   ```bash
   python -m saferl.examples.main algorithm=sac_lag env=SafetyAntVelocity
   ```

2. **New way**:
   ```bash
   python -m saferl.examples.run_experiment experiment=SafetyAntVelocity_sac_lag
   ```

The new system provides the same functionality with better organization and easier parameter management.

## Best Practices

1. **Use descriptive experiment names**: Include environment and algorithm in the filename
2. **Document parameter choices**: Add comments explaining why specific values were chosen
3. **Use parameter overrides**: Don't modify experiment files for simple parameter changes
4. **Organize by environment**: Group related experiments together
5. **Version control**: Keep experiment configurations in version control for reproducibility

## Troubleshooting

### Common Issues

1. **Configuration not found**: Make sure the experiment name matches the filename (without .yaml)
2. **Parameter override not working**: Check the parameter path in the configuration hierarchy
3. **Import errors**: Ensure all algorithm classes are properly imported in the codebase

### Getting Help

- Check the main README.md for general usage
- Look at existing experiment configurations for examples
- Use `--help` flag for command-line help
- Check the Hydra documentation for advanced configuration features
