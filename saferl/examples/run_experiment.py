#!/usr/bin/env python3
"""
SafeRL-Lib Experiment Runner

This script demonstrates how to run experiments using the new unified configuration system.
It supports both individual experiments and parameter sweeps.

Usage:
    # Run a specific experiment
    python run_experiment.py experiment=SafetyAntVelocity_sac_lag
    
    # Run with parameter overrides
    python run_experiment.py experiment=SafetyAntVelocity_sac_lag algorithm.model.cost_constraint=[10.0] seed=42
    
    # Run a parameter sweep
    python run_experiment.py --multirun experiment=SafetyAntVelocity_sac_lag seed=0,1,2,3,4
    
    # Run with different algorithms on the same environment
    python run_experiment.py experiment=SafetyAntVelocity_sac_lag algorithm.model._target_=saferl.algorithms.trpo_lag.TRPO_LAG
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from saferl.examples.main import main as train_main

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def run_experiment(cfg: DictConfig) -> None:
    """
    Run a SafeRL experiment with the specified configuration.
    
    Args:
        cfg: Hydra configuration object containing all experiment parameters
    """
    print("=" * 80)
    print("SafeRL-Lib Experiment Runner")
    print("=" * 80)
    print(f"Experiment: {cfg.get('experiment', 'default')}")
    print(f"Algorithm: {cfg.algorithm.algorithm_name}")
    print(f"Environment: {cfg.env.train_env.env_name}")
    print(f"Cost Constraint: {cfg.algorithm.model.cost_constraint}")
    print(f"Seed: {cfg.seed}")
    print(f"Device: {cfg.device}")
    print("=" * 80)
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Total Timesteps: {cfg.env.total_timesteps:,}")
    print(f"  Episode Length: {cfg.env.episode_len}")
    print(f"  Number of Environments: {cfg.num_env}")
    print(f"  Learning Rate: {cfg.algorithm.model.learning_rate}")
    print(f"  Batch Size: {cfg.algorithm.model.batch_size}")
    print(f"  Buffer Size: {cfg.algorithm.model.buffer_size}")
    print(f"  Policy Architecture: {cfg.algorithm.model.policy_kwargs.net_arch}")
    
    # Run the training
    train_main(cfg)

if __name__ == "__main__":
    run_experiment()
