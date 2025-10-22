"""
SafeRL-Lib Algorithms Package

This package contains implementations of safe reinforcement learning algorithms.
All algorithms are built on top of Stable-Baselines3 and extend the base classes
to handle safety constraints.

Available Algorithms:
- SAC_LAG: Soft Actor-Critic with Lagrangian constraints
- TRPO_LAG: Trust Region Policy Optimization with Lagrangian constraints  
- CPO: Constrained Policy Optimization
- CSAC_LB: Constrained Soft Actor-Critic with Lower Bound
- WCSAC: Worst-Case Soft Actor-Critic
- APPO: Asynchronous Proximal Policy Optimization
- PPO_LAG: Proximal Policy Optimization with Lagrangian constraints
- SAC: Standard Soft Actor-Critic (baseline)

Example:
    >>> from saferl.algorithms import SAC_LAG
    >>> model = SAC_LAG("MlpPolicy", env, cost_constraint=[5.0])
"""

from saferl.algorithms.csac_lb import CSAC_LB
from saferl.algorithms.wcsac import WCSAC
from saferl.algorithms.cpo import CPO
from saferl.algorithms.sac_lag import SAC_LAG   
from saferl.algorithms.sac import SAC
from saferl.algorithms.appo import APPO

__all__ = [
    "WCSAC",
    "CPO",
    "SAC_LAG",
    "SAC",
    "CSAC_LB",
    "APPO",
]
