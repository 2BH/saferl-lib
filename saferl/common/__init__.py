"""
SafeRL-Lib Common Package

This package contains common utilities, base classes, and shared components
used across different safe reinforcement learning algorithms.

Key Components:
- Base algorithm classes (SafeOffPolicyAlgorithm, SafeOnPolicyAlgorithm)
- Policy implementations with cost awareness
- Environment wrappers and utilities
- Monitoring and evaluation tools
- Buffer implementations for safe RL

Example:
    >>> from saferl.common.utils import create_env, evaluate
    >>> from saferl.common.policies import SACwithCostPolicy
"""

# Import commonly used utilities
from saferl.common.utils import (
    create_env,
    evaluate,
    evaluate_after_training,
    create_training_model,
    create_on_step_callback,
)

# Import base algorithm classes
from saferl.common.off_policy_algorithm import SafeOffPolicyAlgorithm
from saferl.common.on_policy_algorithm import SafeOnPolicyAlgorithm
from saferl.common.trust_region_algorithm import SafeTrustRegionAlgorithm

# Import policy classes
from saferl.common.policies import (
    SACwithCostPolicy,
    ActorCriticWithCostPolicy,
)

# Import buffer classes
from saferl.common.buffers import CostReplayBuffer

# Import wrapper classes
from saferl.common.wrappers import (
    ExtendedVecNormalize,
    SafetyGymWrapper,
)

# Import monitoring tools
from saferl.common.monitor import CostMonitor

__all__ = [
    # Utility functions
    "create_env",
    "evaluate", 
    "evaluate_after_training",
    "create_training_model",
    "create_on_step_callback",
    # Base algorithm classes
    "SafeOffPolicyAlgorithm",
    "SafeOnPolicyAlgorithm", 
    "SafeTrustRegionAlgorithm",
    # Policy classes
    "SACwithCostPolicy",
    "ActorCriticWithCostPolicy",
    # Buffer classes
    "CostReplayBuffer",
    # Wrapper classes
    "ExtendedVecNormalize",
    "SafetyGymWrapper",
    # Monitoring tools
    "CostMonitor",
]
