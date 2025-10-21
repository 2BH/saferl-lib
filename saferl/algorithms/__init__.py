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
from saferl.algorithms.ppo_lag.ppo_lag import PPO_LAG
from saferl.algorithms.wcsac import WCSAC
from saferl.algorithms.cpo import CPO
from saferl.algorithms.sac_lag import SAC_LAG   
from saferl.algorithms.sac import SAC
from saferl.algorithms.appo import APPO

# Try to import TRPO_LAG if it exists
try:
    from saferl.algorithms.trpo_lag import TRPO_LAG
    _has_trpo_lag = True
except ImportError:
    _has_trpo_lag = False

__all__ = [
    "WCSAC",
    "CPO",
    "SAC_LAG",
    "SAC",
    "CSAC_LB",
    "PPO_LAG",
    "APPO",
]

# Add TRPO_LAG to __all__ if available
if _has_trpo_lag:
    __all__.append("TRPO_LAG")

