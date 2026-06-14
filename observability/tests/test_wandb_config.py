"""Tests for W&B offline-first observability defaults."""

from __future__ import annotations

from observability.config import ObservabilityConfig
from observability.logging.wandb_tune import wandb_logger_kwargs


# Verifies W&B logging defaults are local/offline and checkpoint-safe.
def test_wandb_logger_kwargs_default_offline():
    config = ObservabilityConfig(run_id="test-run")
    kwargs = wandb_logger_kwargs(config)
    assert kwargs["project"] == "diepcustom-headless-rl"
    assert kwargs["group"] == "ppo-training"
    assert kwargs["mode"] == "offline"
    assert kwargs["log_config"] is True
    assert kwargs["upload_checkpoints"] is False


# Verifies environment overrides used by training are honored.
def test_observability_config_from_env(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("DIEP_VIDEO_INTERVAL", "1")
    monkeypatch.setenv("DIEP_VIDEO_FPS", "12")
    config = ObservabilityConfig.from_env(run_id="env-run")
    assert config.wandb_mode == "disabled"
    assert config.video_interval_iterations == 1
    assert config.video_fps == 12


def test_default_observability_paths_live_under_training_data_wandb():
    config = ObservabilityConfig(run_id="path-test")
    assert config.runs_root == config.runs_root.parents[0] / "W&B"
    assert config.runs_root.name == "W&B"
    assert config.runs_root.parent.name == "training_data"
    assert config.run_dir == config.runs_root / "path-test"
    assert config.eval_iteration_dir(500) == config.run_dir / "eval" / "500"


def test_wandb_logger_kwargs_store_offline_runs_under_wandb_root():
    config = ObservabilityConfig(run_id="wandb-dir-test")
    kwargs = wandb_logger_kwargs(config)
    assert kwargs["name"] == "wandb-dir-test"
    assert kwargs["dir"] == str(config.runs_root)
    assert config.upload_checkpoints is False
