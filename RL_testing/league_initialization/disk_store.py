"""Persist the Redis league to/from SSD safetensors files.

The export directory is the single source of truth for cold starts: if Redis is
empty (fresh container, flushed memory), ``hydrate_redis_from_disk`` repopulates
it from ``diepcustom/training_data/redis/{class}/iter_{N}.safetensors``.
"""

from __future__ import annotations

import logging
import re

from model_store import RedisModelStore

from .paths import LEAGUE_EXPORT_DIR

logger = logging.getLogger(__name__)

_ITER_RE = re.compile(r"iter_(\d+)\.safetensors$")


def hydrate_redis_from_disk(store: RedisModelStore) -> int:
    """Load safetensors exports from SSD into Redis. Returns keys written.

    Idempotent: keys already present in Redis are left untouched. The on-disk
    safetensors byte format is identical to what Redis stores, so file bytes are
    written directly without a torch round-trip.
    """
    written = 0
    for char_class in store.classes:
        class_dir = store.snapshot_dir / char_class
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("iter_*.safetensors")):
            match = _ITER_RE.search(path.name)
            if match is None:
                continue
            iteration = int(match.group(1))
            key = store.key(char_class, iteration)
            if store.redis.get(key) is not None:
                continue
            store.redis.set(key, path.read_bytes())
            written += 1

    if written:
        logger.info("Hydrated Redis league from disk: %d keys from %s", written, LEAGUE_EXPORT_DIR)
    return written


def export_league_to_disk(store: RedisModelStore) -> list:
    """Export every league weight currently in Redis to SSD safetensors files."""
    exported = []
    for char_class in store.classes:
        for redis_key in store.list_class_keys(char_class):
            iteration = int(redis_key.rsplit(":", 1)[-1])
            exported.append(store.export_class(char_class, iteration))
    logger.info("Exported %d league weights to %s", len(exported), LEAGUE_EXPORT_DIR)
    return exported
