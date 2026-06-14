"""RLlib callback: hydrate the league on init, then save mains + refresh ghosts each iteration."""

from __future__ import annotations

import logging

from ray.rllib.algorithms.callbacks import DefaultCallbacks

from model_store import RedisModelStore
from random_pick import load_random_league

from .constants import CHAR_CLASSES
from .disk_store import hydrate_redis_from_disk
from .league_loop import save_mains_and_refresh_ghosts

logger = logging.getLogger(__name__)

_SEED_HINT = "League empty — run: python -m league_initialization.seed_league_cache"


class LeagueBootstrapCallback(DefaultCallbacks):
    """Load ghosts from the pre-seeded Redis/SSD league, then maintain it each iteration."""

    def __init__(self):
        super().__init__()
        self._store: RedisModelStore | None = None

    def on_algorithm_init(self, *, algorithm, **kwargs):
        self._store = RedisModelStore()

        if not self._store.has_league_keys():
            hydrate_redis_from_disk(self._store)

        if not self._store.has_league_keys():
            raise RuntimeError(_SEED_HINT)

        for char_class in CHAR_CLASSES:
            load_random_league(algorithm, char_class, store=self._store)

    def on_train_result(self, *, algorithm, result=None, **kwargs):
        if self._store is None:
            self._store = RedisModelStore()
        league_result = save_mains_and_refresh_ghosts(algorithm, self._store)
        if isinstance(result, dict):
            result["league_iteration"] = league_result["iteration"]
