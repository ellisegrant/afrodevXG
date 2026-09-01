"""Tiny JSON disk cache shared by the data-layer clients.

Both upstream APIs are rate limited (football-data.org ~10 req/min,
The Odds API 500 credits/month), so every network response is parked on disk
under `cache/` and re-read until it goes stale.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _path(key: str) -> Path:
    return CACHE_DIR / (_UNSAFE.sub("_", key) + ".json")


def load(key: str, max_age_seconds: float) -> Optional[Any]:
    """Return the cached payload for `key`, or None if missing/stale."""
    path = _path(key)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    age = time.time() - raw.get("_cached_at", 0)
    if max_age_seconds >= 0 and age > max_age_seconds:
        return None

    log.debug("cache hit %s (age %.0fs)", key, age)
    return raw.get("payload")


def save(key: str, payload: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _path(key)
    try:
        path.write_text(json.dumps({"_cached_at": time.time(), "payload": payload}))
    except OSError as exc:  # a cache failure must never break a request
        log.warning("could not write cache %s: %s", key, exc)
