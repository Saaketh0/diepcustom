"""Resolve Tune experiment paths for training resume."""

from __future__ import annotations

from pathlib import Path

from league_initialization.paths import RLLIB_CHECKPOINT_DIR
from rewards import training_env_config

TUNE_RUN_NAME = "rl_run"
CHECKPOINT_FREQUENCY = 5
KEEP_PER_TRIAL = 50
TRAINING_ITERATIONS = 2000

DIEP_ENV_CONFIG = training_env_config()


def find_experiment_path() -> str:
    """Return the Tune experiment directory for ``Tuner.restore``."""
    path = RLLIB_CHECKPOINT_DIR / TUNE_RUN_NAME
    if not path.is_dir():
        raise FileNotFoundError(
            f"No prior Tune experiment at {path}. "
            "Start a fresh run first, or pass --resume-path explicitly."
        )
    return str(path)


def resolve_experiment_path(explicit: str | None = None) -> str:
    """Use an explicit experiment path or discover the default ``rl_run`` dir."""
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(f"Resume path does not exist: {path}")
        return str(path)
    return find_experiment_path()


def main() -> None:
    """Resume a prior Tune PPO run. Start Redis + league SSD from the same run first."""
    import argparse

    import ray
    from ray import tune
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env

    from RL_training import DiepCustomParallelEnv

    parser = argparse.ArgumentParser(description="Resume Diep RLlib PPO training")
    parser.add_argument(
        "--resume-path",
        default=None,
        help="Optional explicit Tune experiment directory for Tuner.restore",
    )
    args = parser.parse_args()

    ray.init()
    register_env(
        "diepcustom_headless",
        lambda cfg: ParallelPettingZooEnv(DiepCustomParallelEnv(**cfg)),
    )
    tuner = tune.Tuner.restore(
        resolve_experiment_path(args.resume_path),
        trainable="PPO",
    )
    tuner.fit()


if __name__ == "__main__":
    main()
