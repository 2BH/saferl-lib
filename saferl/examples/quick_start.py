#!/usr/bin/env python3
"""
SafeRL-Lib Quick Start Example

This script demonstrates the basic usage of SafeRL-Lib with the new unified
configuration system. It shows how to:

1. Run a simple experiment
2. Override parameters
3. Evaluate trained models
4. Use different algorithms and environments

Usage:
    python quick_start.py
"""

import os
import sys
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the parent directory to the path to import saferl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from saferl.algorithms import CSAC_LB, SAC_LAG
from saferl.common.utils import create_env, evaluate

def run_quick_experiment():
    """Run a quick experiment to demonstrate SafeRL-Lib usage."""
    
    print("=" * 80)
    print("SafeRL-Lib Quick Start Example")
    print("=" * 80)
    
    # Create a simple environment configuration
    env_cfg = {
        'env_name': 'SafetyAntVelocity-v1',
        'env_kwargs': {'camera_id': 0},
        'num_env': 1
    }
    
    print("1. Creating Environment...")
    env = create_env(env_cfg, seed=42, monitor=True)
    print(f"   Environment: {env_cfg['env_name']}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    print("\n2. Creating CSAC-LB Model...")
    model = CSAC_LB(
        policy="MlpPolicy",
        env=env,
        cost_constraint=[5.0],
        lower_bound=0.1,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=64,
        verbose=1,
        device="cpu"  # Use CPU for quick demo
    )
    print(f"   Algorithm: CSAC-LB (TMLR 2025)")
    print(f"   Cost constraint: {model.cost_constraint}")
    print(f"   Lower bound: {model.lower_bound}")
    print(f"   Learning rate: {model.learning_rate}")
    
    print("\n3. Training Model...")
    print("   Training for 1000 timesteps (this is just a demo)...")
    model.learn(total_timesteps=1000, log_interval=100)
    print("   Training completed!")
    
    print("\n4. Evaluating Model...")
    eval_results = evaluate(
        model=model,
        env=env,
        num_episodes=3,
        deterministic=True
    )
    
    print(f"   Average return: {np.mean(eval_results['ret']):.2f}")
    print(f"   Average cost: {np.mean(eval_results['cost']):.2f}")
    print(f"   Safety rate: {np.mean(eval_results['is_safe']):.2f}")
    print(f"   Average episode length: {np.mean(eval_results['len']):.2f}")
    
    print("\n5. Saving Model...")
    model.save("quick_start_model")
    print("   Model saved as 'quick_start_model'")
    
    print("\n" + "=" * 80)
    print("Quick Start Example Completed Successfully!")
    print("=" * 80)
    
    return model, eval_results

def demonstrate_parameter_override():
    """Demonstrate how to override parameters."""
    
    print("\n" + "=" * 80)
    print("Parameter Override Example")
    print("=" * 80)
    
    # Create environment with different parameters
    env_cfg = {
        'env_name': 'SafetyAntVelocity-v1',
        'env_kwargs': {'camera_id': 1},  # Different camera angle
        'num_env': 2  # More parallel environments
    }
    
    print("Creating model with overridden parameters...")
    model = SAC_LAG(
        policy="MlpPolicy",
        env=create_env(env_cfg, seed=123),
        cost_constraint=[10.0],  # Different cost constraint
        learning_rate=1e-3,      # Different learning rate
        batch_size=128,          # Different batch size
        verbose=0,
        device="cpu"
    )
    
    print(f"   Cost constraint: {model.cost_constraint}")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   Batch size: {model.batch_size}")
    print("   Parameter override successful!")

def demonstrate_different_algorithms():
    """Demonstrate different algorithms (if available)."""
    
    print("\n" + "=" * 80)
    print("Different Algorithms Example")
    print("=" * 80)
    
    env_cfg = {
        'env_name': 'SafetyAntVelocity-v1',
        'env_kwargs': {'camera_id': 0},
        'num_env': 1
    }
    
    env = create_env(env_cfg, seed=42)
    
    # Try different algorithms
    algorithms = [
        ("CSAC-LB", CSAC_LB),
        ("SAC-Lag", SAC_LAG),
    ]
    
    for name, algorithm_class in algorithms:
        try:
            print(f"\nTesting {name}...")
            if name == "CSAC-LB":
                model = algorithm_class(
                    policy="MlpPolicy",
                    env=env,
                    cost_constraint=[5.0],
                    lower_bound=0.1,
                    learning_rate=3e-4,
                    verbose=0,
                    device="cpu"
                )
            else:
                model = algorithm_class(
                    policy="MlpPolicy",
                    env=env,
                    cost_constraint=[5.0],
                    learning_rate=3e-4,
                    verbose=0,
                    device="cpu"
                )
            print(f"   {name} created successfully!")
            
            # Quick training
            model.learn(total_timesteps=500, log_interval=100)
            
            # Quick evaluation
            eval_results = evaluate(model, env, num_episodes=1, deterministic=True)
            print(f"   Return: {np.mean(eval_results['ret']):.2f}")
            print(f"   Cost: {np.mean(eval_results['cost']):.2f}")
            
        except Exception as e:
            print(f"   {name} not available: {e}")

if __name__ == "__main__":
    try:
        # Run the main example
        model, results = run_quick_experiment()
        
        # Demonstrate parameter overrides
        demonstrate_parameter_override()
        
        # Demonstrate different algorithms
        demonstrate_different_algorithms()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Try running with different environments")
        print("2. Experiment with different cost constraints")
        print("3. Use the new configuration system: python -m saferl.examples.run_experiment")
        print("4. Check out the documentation in saferl/examples/configs/README.md")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)
