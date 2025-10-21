"""
SafeRL-Lib: A Comprehensive Toolkit for Safe Reinforcement Learning

This package provides implementations of state-of-the-art safe reinforcement learning
algorithms, built on top of Stable-Baselines3 and Safety-Gymnasium.

Key Features:
- Multiple safe RL algorithms (SAC-Lag, TRPO-Lag, CPO, CSAC-LB, WCSAC, APPO)
- Comprehensive environment support (Safety-Gymnasium, custom tasks, Isaac Gym)
- Flexible configuration system with Hydra
- Advanced monitoring and evaluation tools

Example:
    >>> from saferl.algorithms.sac_lag import SAC_LAG
    >>> from saferl.common.utils import create_env
    >>> 
    >>> # Create environment
    >>> env = create_env(env_cfg, seed=42)
    >>> 
    >>> # Create and train model
    >>> model = SAC_LAG("MlpPolicy", env, cost_constraint=[5.0])
    >>> model.learn(total_timesteps=100000)
"""

__version__ = "0.1.0"
__author__ = "SafeRL-Lib Team"
__email__ = "saferl@example.com"

# Import main algorithm classes for easy access
from saferl.algorithms.sac_lag import SAC_LAG
from saferl.algorithms.trpo_lag import TRPO_LAG
from saferl.algorithms.cpo import CPO
from saferl.algorithms.csac_lb import CSAC_LB
from saferl.algorithms.wcsac import WCSAC
from saferl.algorithms.appo import APPO

# Import common utilities
from saferl.common.utils import create_env, evaluate, evaluate_after_training

# Define what gets imported with "from saferl import *"
__all__ = [
    # Algorithm classes
    "SAC_LAG",
    "TRPO_LAG", 
    "CPO",
    "CSAC_LB",
    "WCSAC",
    "APPO",
    # Utility functions
    "create_env",
    "evaluate",
    "evaluate_after_training",
]
