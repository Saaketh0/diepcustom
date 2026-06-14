"""Tests for league_loop per-iteration helpers and disk_store hydration."""

from __future__ import annotations

import fnmatch

import torch
from safetensors.torch import load, save_file

from league_initialization.constants import CHAR_CLASSES, ghost_policy_id, main_policy_id
from league_initialization.disk_store import export_league_to_disk, hydrate_redis_from_disk
from league_initialization.league_loop import (
    save_mains_and_refresh_ghosts,
    save_mains_to_redis,
)
from model_store import RedisModelStore


class FakeRedis:
    def __init__(self):
        self.data: dict[str, bytes] = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        self.data.pop(key, None)

    unlink = delete

    def keys(self, pattern):
        return [key.encode("utf-8") for key in self.data if fnmatch.fnmatch(key, pattern)]

    def scan_iter(self, match=None):
        pattern = match or "*"
        for key in list(self.data):
            if fnmatch.fnmatch(key, pattern):
                yield key.encode("utf-8")


class FakeModule:
    def __init__(self, value: float = 0.0):
        self._state = {"weight": torch.tensor([value])}

    def get_state(self):
        return {name: value.clone() for name, value in self._state.items()}

    def set_state(self, state_dict):
        self._state = {name: value.clone() for name, value in state_dict.items()}


class FakeAlgorithm:
    def __init__(self):
        self.modules = {}
        for index, char_class in enumerate(CHAR_CLASSES):
            self.modules[main_policy_id(char_class)] = FakeModule(float(index + 1))
            for slot in range(4):
                self.modules[ghost_policy_id(char_class, slot)] = FakeModule()
        self.sync_calls = 0

    def get_module(self, module_id):
        return self.modules[module_id]

    class _Workers:
        def __init__(self, algorithm):
            self.algorithm = algorithm

        def sync_weights(self):
            self.algorithm.sync_calls += 1

    @property
    def workers(self):
        return self._Workers(self)


def _store(tmp_path) -> RedisModelStore:
    return RedisModelStore(client=FakeRedis(), snapshot_every=0, snapshot_dir=tmp_path)


def test_save_mains_to_redis_writes_and_exports(tmp_path):
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    iteration = save_mains_to_redis(algorithm, store)

    assert iteration == 0
    assert store.latest_by_class() == {char_class: 0 for char_class in CHAR_CLASSES}
    for index, char_class in enumerate(CHAR_CLASSES):
        export = tmp_path / char_class / "iter_0.safetensors"
        assert export.exists()
        assert load(store.redis.get(store.key(char_class, 0)))["weight"].item() == float(index + 1)


def test_save_mains_and_refresh_ghosts_syncs_once(tmp_path):
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)

    result = save_mains_and_refresh_ghosts(algorithm, store)

    assert result["iteration"] == 0
    # One sync for the whole ghost refresh, not one per class.
    assert algorithm.sync_calls == 1
    for index, char_class in enumerate(CHAR_CLASSES):
        ghost = algorithm.modules[ghost_policy_id(char_class, 0)]
        assert ghost.get_state()["weight"].item() == float(index + 1)


def test_hydrate_redis_from_disk_repopulates_empty_redis(tmp_path):
    # Write safetensors exports directly to disk, leave Redis empty.
    for char_class in CHAR_CLASSES:
        class_dir = tmp_path / char_class
        class_dir.mkdir(parents=True)
        save_file({"weight": torch.tensor([3.0])}, class_dir / "iter_0.safetensors")
        save_file({"weight": torch.tensor([4.0])}, class_dir / "iter_1.safetensors")

    store = _store(tmp_path)
    assert store.has_league_keys() is False

    written = hydrate_redis_from_disk(store)

    assert written == len(CHAR_CLASSES) * 2
    assert store.has_league_keys() is True
    assert load(store.redis.get(store.key("A", 1)))["weight"].item() == 4.0


def test_hydrate_is_idempotent(tmp_path):
    for char_class in CHAR_CLASSES:
        class_dir = tmp_path / char_class
        class_dir.mkdir(parents=True)
        save_file({"weight": torch.tensor([3.0])}, class_dir / "iter_0.safetensors")

    store = _store(tmp_path)
    assert hydrate_redis_from_disk(store) == len(CHAR_CLASSES)
    assert hydrate_redis_from_disk(store) == 0


def test_export_league_to_disk_writes_all_keys(tmp_path):
    algorithm = FakeAlgorithm()
    store = _store(tmp_path)
    save_mains_to_redis(algorithm, store)

    exported = export_league_to_disk(store)

    assert len(exported) == len(CHAR_CLASSES)
    assert all(path.exists() for path in exported)
