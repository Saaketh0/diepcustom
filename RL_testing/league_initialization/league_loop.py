"""Per-iteration league loop: save mains to Redis+SSD, refresh ghosts."""

from __future__ import annotations

import logging

from model_store import RedisModelStore
from random_pick import load_random_league

from .constants import CHAR_CLASSES, main_policy_id
from .module_state import get_module_state, sync_module_weights

logger = logging.getLogger(__name__)


def collect_main_states(algorithm) -> dict[str, dict]:
    """Snapshot each main_class_{X} RLModule state."""
    return {
        char_class: get_module_state(algorithm, main_policy_id(char_class))
        for char_class in CHAR_CLASSES
    }


def save_mains_to_redis(algorithm, store: RedisModelStore) -> int:
    """Save all main weights at the next iteration index. Returns that index."""
    iteration = store.next_iteration()
    states = collect_main_states(algorithm)
    store.save_all(states, iteration)
    # Ensure this iteration always lands on SSD, independent of snapshot_every.
    for char_class in states:
        store.export_class(char_class, iteration)
    return iteration


def refresh_all_ghosts(algorithm, store: RedisModelStore) -> list[dict]:
    """Resample ghost weights for every class, syncing env runners once."""
    results = [
        load_random_league(algorithm, char_class, store=store, sync=False)
        for char_class in CHAR_CLASSES
    ]
    sync_module_weights(algorithm)
    return results


def save_mains_and_refresh_ghosts(algorithm, store: RedisModelStore) -> dict:
    """Save mains first (so this iteration enters the pool), then refresh ghosts."""
    iteration = save_mains_to_redis(algorithm, store)
    ghost_results = refresh_all_ghosts(algorithm, store)
    logger.info("League iteration %d: saved mains, refreshed ghosts", iteration)
    return {"iteration": iteration, "ghosts": ghost_results}
