"""Tiny Redis + safetensors model weight store for RLlib experiments.

This is intentionally boring: save PyTorch state_dicts to Redis, keep a rolling
history per class, optionally export safetensors files to SSD, and provide a
small RLlib checkpoint helper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import redis
import torch
from safetensors.torch import load, save, save_file


DEFAULT_CLASSES = ("A", "B", "C", "D")

# .../diepcustom/RL_testing/model_store.py -> .../diepcustom/training_data/redis
DEFAULT_SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "training_data" / "redis"


class MissingWeights(KeyError):
    """Raised when a requested Redis weight key is missing."""


def _cpu_state_dict(state_dict: dict) -> dict:
    return {name: _to_cpu_tensor(value) for name, value in state_dict.items()}


def _to_cpu_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return torch.as_tensor(value).cpu()
    except ImportError:
        pass
    return value


class RedisModelStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        *,
        classes: Iterable[str] = DEFAULT_CLASSES,
        window_size: int = 50,
        snapshot_every: int = 10,
        key_prefix: str = "policy",
        snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
        client=None,
    ):
        self.redis = client or redis.Redis(host=host, port=port)
        self.classes = tuple(classes)
        self.window_size = int(window_size)
        self.snapshot_every = int(snapshot_every)
        self.key_prefix = key_prefix
        self.snapshot_dir = Path(snapshot_dir)

    def key(self, char_class: str, iteration: int) -> str:
        return f"{self.key_prefix}:{char_class}:{int(iteration)}"

    def save_class(self, char_class: str, state_dict: dict, iteration: int) -> str:
        key = self.key(char_class, iteration)
        self.redis.set(key, save(_cpu_state_dict(state_dict)))
        self._drop_old(char_class, iteration)
        if self.snapshot_every > 0 and int(iteration) % self.snapshot_every == 0:
            self.export_class(char_class, iteration)
        return key

    def save_all(self, state_dicts_by_class: dict[str, dict], iteration: int) -> list[str]:
        return [
            self.save_class(char_class, state_dict, iteration)
            for char_class, state_dict in state_dicts_by_class.items()
        ]

    def load_state_dict(self, char_class: str, iteration: int) -> dict:
        key = self.key(char_class, iteration)
        raw = self.redis.get(key)
        if raw is None:
            raise MissingWeights(f"missing weights: {key}")
        return load(raw)

    def load_model(self, char_class: str, iteration: int, model):
        return model.load_state_dict(self.load_state_dict(char_class, iteration))

    def export_class(self, char_class: str, iteration: int, export_path: str | Path | None = None) -> Path:
        weights = self.load_state_dict(char_class, iteration)
        path = Path(export_path) if export_path else self.snapshot_dir / char_class / f"iter_{int(iteration)}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(_cpu_state_dict(weights), path)
        return path

    def export_all(self, iteration: int) -> list[Path]:
        return [
            self.export_class(char_class, iteration)
            for char_class in self.classes
            if self.redis.get(self.key(char_class, iteration)) is not None
        ]

    def latest_by_class(self) -> dict[str, int]:
        latest = {char_class: -1 for char_class in self.classes}
        for raw_key in self._iter_keys(f"{self.key_prefix}:*:*"):
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
            parts = key.split(":")
            if len(parts) != 3 or parts[0] != self.key_prefix:
                continue
            char_class, value = parts[1], parts[2]
            try:
                iteration = int(value)
            except ValueError:
                continue
            latest[char_class] = max(latest.get(char_class, -1), iteration)
        return latest

    def latest_iteration(self) -> int:
        latest = self.latest_by_class().values()
        return max(latest, default=-1)

    def has_class_keys(self, char_class: str) -> bool:
        return self.latest_by_class().get(char_class, -1) >= 0

    def has_league_keys(self) -> bool:
        return all(self.has_class_keys(char_class) for char_class in self.classes)

    def list_class_keys(self, char_class: str) -> list[str]:
        pattern = f"{self.key_prefix}:{char_class}:*"
        keys: list[str] = []
        for raw_key in self._iter_keys(pattern):
            key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
            keys.append(key)
        return sorted(keys, key=lambda value: int(value.rsplit(":", 1)[-1]))

    def next_iteration(self) -> int:
        return self.latest_iteration() + 1

    def save_rllib_checkpoint(self, algorithm, iteration: int, checkpoint_dir: str | Path = "RL_testing/rllib_checkpoints") -> Path:
        path = Path(checkpoint_dir) / f"iter_{int(iteration)}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return Path(algorithm.save_to_path(str(path)))

    def _drop_old(self, char_class: str, iteration: int) -> None:
        old_iteration = int(iteration) - self.window_size
        if old_iteration < 0:
            return
        old_key = self.key(char_class, old_iteration)
        unlink = getattr(self.redis, "unlink", None)
        if callable(unlink):
            unlink(old_key)
        else:
            self.redis.delete(old_key)

    def _iter_keys(self, pattern: str):
        scan_iter = getattr(self.redis, "scan_iter", None)
        if callable(scan_iter):
            yield from scan_iter(match=pattern)
        else:
            yield from self.redis.keys(pattern)
