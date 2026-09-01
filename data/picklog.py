"""Persistent record of every value pick the scanner has made.

Historical bookmaker odds need a paid plan, so the model cannot be scored
against the market retrospectively. Recording picks as they are made solves the
same problem forwards: once a fixture has been played, the pick can be settled,
and before it is played the market's drift toward or away from the model is
itself informative (closing line value).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

PICKS_DIR = Path(__file__).resolve().parent.parent / "picks"
PICKS_FILE = PICKS_DIR / "picks.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(pick: dict[str, Any]) -> tuple[str, str, str]:
    """Identity of a pick: the same selection on the same fixture."""
    return (pick.get("match", ""), pick.get("outcome", ""), pick.get("kickoff", "")[:10])


def load_all() -> list[dict[str, Any]]:
    if not PICKS_FILE.exists():
        return []
    records = []
    for line in PICKS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("skipping corrupt pick log line")
    return records


def _write_all(records: list[dict[str, Any]]) -> None:
    PICKS_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_FILE.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def record(competition: str, picks: list[dict[str, Any]]) -> dict[str, int]:
    """Append new picks, ignoring ones already logged.

    The first price seen is kept: that is the price that was actually available
    when the model flagged the selection.
    """
    existing = load_all()
    seen = {_key(record) for record in existing}

    added = 0
    for pick in picks:
        if _key(pick) in seen:
            continue
        existing.append(
            {
                "id": uuid.uuid4().hex[:12],
                "competition": competition,
                "recorded_at": _now(),
                "status": "open",
                "match": pick["match"],
                "kickoff": pick["kickoff"],
                "selection": pick["selection"],
                "outcome": pick["outcome"],
                "model_prob": pick["model_prob"],
                "fair_prob_at_pick": pick["fair_prob"],
                "edge_at_pick": pick["edge"],
                "price_taken": pick["price"],
                "price_book": pick["price_book"],
                "best_price_at_pick": pick["best_price"],
                "closing_fair_prob": None,
                "closing_best_price": None,
                "result": None,
                "profit_units": None,
            }
        )
        seen.add(_key(pick))
        added += 1

    _write_all(existing)
    log.info("pick log: %d added, %d total", added, len(existing))
    return {"added": added, "skipped": len(picks) - added, "total": len(existing)}


def update(pick_id: str, **fields: Any) -> None:
    records = load_all()
    for record in records:
        if record["id"] == pick_id:
            record.update(fields)
            break
    _write_all(records)


def update_many(updates: dict[str, dict[str, Any]]) -> None:
    """Apply {pick_id: {field: value}} in a single rewrite."""
    if not updates:
        return
    records = load_all()
    for record in records:
        if record["id"] in updates:
            record.update(updates[record["id"]])
    _write_all(records)
