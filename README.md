# SafeRL-Lib

A comprehensive toolkit for Constrained Reinforcement Learning and Safe Reinforcement Learning research, built on top of [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium).

## Features

- **Multiple Safe RL Algorithms**: Implementation of state-of-the-art safe RL algorithms including:
  - **CSAC-LB** (Constrained Soft Actor-Critic with Log Barrier) - [TMLR 2025](https://openreview.net/forum?id=Amh95oURaE) ⭐
  - SAC-Lag (Soft Actor-Critic with Lagrangian constraints) - [RSS 2020](https://arxiv.org/abs/2002.08550)
  - CPO (Constrained Policy Optimization) - [ICML 2017](https://arxiv.org/abs/1705.10528)
  - WCSAC (Worst-Case Soft Actor-Critic) - [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17272)
  - APPO (Augmented Proximal Policy Optimization) - [AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25888)
  - SAC (Soft Actor-Critic) with Reward Shaping - Modified from [ICML 2018](https://arxiv.org/abs/1801.01290)


## Installation

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (recommended)


### Step 1: Clone the Repository
```bash
git clone git@github.com:2BH/saferl-lib.git
cd saferl-lib
```

### Step 2: Create Conda Environment (Optional)
```bash
conda env create -n saferl python=3.10
conda activate saferl-lib
```

### Step 3: Install Safety-Gymnasium and other dependencies
```bash
pip install stable_baselines3==2.7.0
pip install sb3_contrib==2.7.0
cd ~/
wget https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip
unzip main.zip
cd safety-gymnasium-main
pip install -e .
pip install gymnasium==0.29.1
pip install hydra-core==1.3.2
pip install tensorboard==2.20.0
```

## 🌟 Featured Algorithm: CSAC-LB

**Constrained Soft Actor-Critic with Log Barrier** - [TMLR 2025](https://openreview.net/forum?id=Amh95oURaE)

CSAC-LB is our algorithm that addresses key challenges in constrained reinforcement learning:

- **Numerical Stability**: Uses a linear smoothed log barrier function that provides non-vanishing gradients
- **Quick Recovery**: Enables agents to quickly recover from unsafe states during training
- **Enhanced Safety**: Employs a pessimistic double-critic architecture to mitigate constraint violation underestimation



**Key Innovation**: The integration of a smoothed log barrier function into the actor's objective provides a numerically stable alternative to traditional interior-point methods, making it particularly suitable for safety-critical applications.

![image](assets/smoothed_log_barrier.png)

```python
# Quick CSAC-LB example
from saferl import CSAC_LB, create_env

env = create_env(env_cfg, seed=42)
model = CSAC_LB("MlpPolicy", env, cost_constraint=[5.0], lower_bound=0.1)
model.learn(total_timesteps=100000)
```

## Quick Start

### Basic Usage
```bash
# Activate environment
conda activate saferl

# Run CSAC-LB experiment (recommended)
python -m saferl.examples.main algorithm=csac_lb
```

### Reproducing the Results
```bash
conda activate saferl
# For Crabs environments, env=CrabsMove/CrabsSwing/CrabsTilt/CrabsUpright
python -m saferl.examples.main env=CrabsMove norm_obs=False eval_freq=1000
# For other environments, i.e. env=SafetyAntVelocity/SafetyHumanoidVelocity/SafetyWalker2DVelocity/SafetyHalfCheetahVelocity/SafetyHopperVelocity/SafetyCarircle1
python -m saferl.examples.main env=SafetyAntVelocity norm_obs=True eval_freq=100000
```

## Configuration System

The library uses Hydra for configuration management. Configurations are organized as follows:

```
saferl/examples/configs/
├── algorithm/          # Algorithm-specific configurations
├── env/               # Environment-specific configurations
├── callback/          # Callback configurations
└── main.yaml         # Main configuration file
```

### Customizing Experiments

You can override any configuration parameter:

```bash
# Change learning rate
python -m saferl.examples.main algorithm=sac_lag algorithm.model.learning_rate=1e-4

# Change environment parameters
python -m saferl.examples.main env=SafetyAntVelocity env.train_env.env_kwargs.camera_id=1

# Run with different seeds
python -m saferl.examples.main seed=42
```

## Examples

### Training with CSAC-LB (Recommended)
```python
from saferl import CSAC_LB, create_env

# Create environment
env = create_env(env_cfg, seed=42)

# Create and train CSAC-LB model
model = CSAC_LB("MlpPolicy", env, cost_constraint=[5.0], lower_bound=0.1)
model.learn(total_timesteps=100000)

# Save model
model.save("csac_lb_agent")
```

### Training with SAC-Lag
```python
from saferl import SAC_LAG, create_env

# Create environment
env = create_env(env_cfg, seed=42)

# Create and train SAC-Lag model
model = SAC_LAG("MlpPolicy", env, cost_constraint=[5.0])
model.learn(total_timesteps=100000)

# Save model
model.save("sac_lag_agent")
```

### Evaluation
```python
from saferl.common.utils import evaluate

# Evaluate the trained model
results = evaluate(model, env, num_episodes=10)
print(f"Average return: {results['ret']}")
print(f"Average cost: {results['cost']}")
print(f"Safety rate: {results['is_safe']}")
```

## Documentation

For detailed API documentation and examples, please refer to the individual algorithm modules and the `saferl.common` utilities.

## Citation

If you use this library in your research or use our algorithm, please cite our work:

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Environment support from [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- Configuration management with [Hydra](https://github.com/facebookresearch/hydra)

![image](assets/UFR-vorlage-designsystem-typo-farben-V1.92-1536x1086.png)