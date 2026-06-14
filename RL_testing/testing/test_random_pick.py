"""Tests for random_pick league sampling."""

from __future__ import annotations

import fnmatch

import torch
from safetensors.torch import load, save

from league_initialization.constants import ghost_policy_id, main_policy_id
from model_store import RedisModelStore
from random_pick import load_random_league, weighted_recent_sample


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
    def __init__(self):
        self._state = {"weight": torch.tensor([0.0])}

    def get_state(self):
        return {name: value.clone() for name, value in self._state.items()}

    def set_state(self, state_dict):
        self._state = {name: value.clone() for name, value in state_dict.items()}


class FakeAlgorithm:
    def __init__(self):
        self.modules = {
            main_policy_id("A"): FakeModule(),
            ghost_policy_id("A", 0): FakeModule(),
            ghost_policy_id("A", 1): FakeModule(),
            ghost_policy_id("A", 2): FakeModule(),
            ghost_policy_id("A", 3): FakeModule(),
        }
        self.synced = False

    def get_module(self, module_id):
        return self.modules[module_id]

    class _Workers:
        def __init__(self, algorithm):
            self.algorithm = algorithm

        def sync_weights(self):
            self.algorithm.synced = True

    @property
    def workers(self):
        return self._Workers(self)


def test_weighted_recent_sample_empty_pool():
    assert weighted_recent_sample([], 4) == []


def test_weighted_recent_sample_partial_pool():
    keys = ["policy:A:1", "policy:A:3"]
    picked = weighted_recent_sample(keys, 4)
    assert len(picked) == 2
    assert set(picked) == set(keys)


def test_load_random_league_empty_redis():
    algorithm = FakeAlgorithm()
    store = RedisModelStore(client=FakeRedis(), snapshot_every=0)

    result = load_random_league(algorithm, "A", store=store)

    assert result["source"] == "none"
    assert result["loaded"] == 0
    assert algorithm.synced is False


def test_load_random_league_after_bootstrap():
    algorithm = FakeAlgorithm()
    store = RedisModelStore(client=FakeRedis(), snapshot_every=0)
    state = {"weight": torch.tensor([7.0])}
    for iteration in range(50):
        store.save_class("A", state, iteration)

    result = load_random_league(algorithm, "A", store=store)

    assert result["source"] == "redis"
    assert result["loaded"] == 4
    assert algorithm.synced is True
    assert algorithm.modules[ghost_policy_id("A", 0)].get_state()["weight"].item() == 7.0
