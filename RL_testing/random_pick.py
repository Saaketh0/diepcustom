"""Sample ghost policy weights from the Redis league."""

from __future__ import annotations

import logging
import random

from safetensors.torch import load

from league_initialization.constants import GHOST_SLOTS, ghost_policy_id
from league_initialization.module_state import set_module_state, sync_module_weights
from model_store import RedisModelStore

logger = logging.getLogger(__name__)


def weighted_recent_sample(keys, k, decay=0.90):
    """Pick up to k distinct keys, favoring more recent iteration numbers."""
    if not keys:
        return []

    latest = max(int(key.rsplit(":", 1)[-1]) for key in keys)
    pool = list(keys)
    picked = []
    for _ in range(min(k, len(pool))):
        weights = [decay ** (latest - int(key.rsplit(":", 1)[-1])) for key in pool]
        choice = random.choices(pool, weights=weights, k=1)[0]
        picked.append(choice)
        pool.remove(choice)
    return picked


def load_random_league(
    algorithm,
    char_class: str,
    *,
    store: RedisModelStore | None = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    ghost_slots: int = GHOST_SLOTS,
    sync: bool = True,
) -> dict:
    """Load sampled ghost weights for one class from Redis into RLModules."""
    league_store = store or RedisModelStore(host=redis_host, port=redis_port)
    all_keys = league_store.list_class_keys(char_class)

    if not all_keys:
        logger.error(
            "No Redis league keys for class %s; bootstrap should run before ghost load",
            char_class,
        )
        return {
            "char_class": char_class,
            "source": "none",
            "loaded": 0,
            "keys": [],
        }

    sampled_keys = weighted_recent_sample(all_keys, ghost_slots)
    loaded = 0

    for idx, redis_key in enumerate(sampled_keys):
        safetensor_bytes = league_store.redis.get(redis_key)
        if safetensor_bytes is None:
            logger.warning("Missing Redis weight key %s", redis_key)
            continue

        weights_dict = load(safetensor_bytes)
        set_module_state(algorithm, ghost_policy_id(char_class, idx), weights_dict)
        loaded += 1

    if loaded and sync:
        sync_module_weights(algorithm)

    return {
        "char_class": char_class,
        "source": "redis",
        "loaded": loaded,
        "keys": sampled_keys,
    }
