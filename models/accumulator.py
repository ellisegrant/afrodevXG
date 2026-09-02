"""Build a multi-leg bet that lands near a target price.

Given every selection the model has a view on, find combinations whose decimal
odds multiply to roughly what the punter asked for - "give me a 4.0" - and rank
them by how likely the model thinks they all are to come in.

Two rules are enforced rather than left to the caller:

* one leg per fixture, because two selections from the same match are not
  independent and multiplying their probabilities would be plain wrong;
* the search runs in log space, since combining odds is multiplication and
  floating-point products across many legs drift.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

log = logging.getLogger(__name__)

# Beam width: how many partial combinations to carry forward per fixture.
DEFAULT_BEAM = 400


def _score(state: dict[str, Any], objective: str) -> float:
    """Higher is better. Probability maximises the chance every leg lands;
    value maximises expected return, which tolerates longer prices."""
    if objective == "value":
        return state["log_prob"] + state["log_odds"]
    return state["log_prob"]


def build(
    candidates: list[dict[str, Any]],
    target_odds: float,
    tolerance_pct: float = 10.0,
    min_legs: int = 2,
    max_legs: int = 6,
    objective: str = "probability",
    max_results: int = 5,
    beam_width: int = DEFAULT_BEAM,
    per_fixture: int = 6,
) -> list[dict[str, Any]]:
    """Combinations whose combined odds sit within tolerance of `target_odds`.

    Each candidate needs `fixture`, `price` and `model_prob`; everything else is
    carried through untouched for display.
    """
    if target_odds <= 1.0:
        raise ValueError("target odds must be greater than 1.0")
    if min_legs < 1 or max_legs < min_legs:
        raise ValueError("leg bounds are inconsistent")

    lower = target_odds * (1.0 - tolerance_pct / 100.0)
    upper = target_odds * (1.0 + tolerance_pct / 100.0)
    log_upper = math.log(upper)

    # Group by fixture and keep only the strongest few selections from each, so
    # the search does not drown in near-duplicates of the same match.
    by_fixture: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if candidate["price"] <= 1.0 or not (0.0 < candidate["model_prob"] < 1.0):
            continue
        by_fixture.setdefault(candidate["fixture"], []).append(candidate)

    for fixture, options in by_fixture.items():
        options.sort(
            key=lambda option: option["model_prob"] * option["price"], reverse=True
        )
        by_fixture[fixture] = options[:per_fixture]

    states: list[dict[str, Any]] = [{"legs": [], "log_odds": 0.0, "log_prob": 0.0}]
    complete: list[dict[str, Any]] = []

    for options in by_fixture.values():
        next_states = list(states)  # skipping this fixture is always allowed

        for state in states:
            if len(state["legs"]) >= max_legs:
                continue
            for option in options:
                log_odds = state["log_odds"] + math.log(option["price"])
                if log_odds > log_upper:
                    continue  # already past the target, adding legs only overshoots
                next_states.append(
                    {
                        "legs": state["legs"] + [option],
                        "log_odds": log_odds,
                        "log_prob": state["log_prob"] + math.log(option["model_prob"]),
                    }
                )

        next_states.sort(key=lambda state: _score(state, objective), reverse=True)
        states = next_states[:beam_width]

        for state in states:
            if min_legs <= len(state["legs"]) <= max_legs:
                combined = math.exp(state["log_odds"])
                if lower <= combined <= upper:
                    complete.append(state)

    # De-duplicate: the same combination can be completed on different passes.
    seen: set[tuple] = set()
    unique = []
    for state in complete:
        key = tuple(sorted(f"{leg['fixture']}|{leg['selection']}" for leg in state["legs"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(state)

    unique.sort(key=lambda state: _score(state, objective), reverse=True)

    accumulators = []
    for state in unique[:max_results]:
        combined_odds = math.exp(state["log_odds"])
        model_success = math.exp(state["log_prob"])
        market_success = 1.0
        for leg in state["legs"]:
            market_success *= leg.get("fair_prob", leg["model_prob"])
        accumulators.append(
            {
                "legs": state["legs"],
                "combined_odds": combined_odds,
                "model_success": model_success,
                "market_success": market_success,
                # What one unit returns on average, if the model is right.
                "expected_value_pct": (model_success * combined_odds - 1.0) * 100.0,
                "fair_odds": 1.0 / model_success if model_success > 0 else float("inf"),
            }
        )
    return accumulators


def margin_compounding(legs: int, per_leg_margin: float = 0.05) -> float:
    """How much bookmaker margin a multi-leg bet carries, as a percentage.

    Each leg is priced with its own cut, and those cuts multiply. This is why an
    accumulator of fairly-priced-looking legs is usually a worse bet than any of
    them alone.
    """
    return ((1.0 + per_leg_margin) ** legs - 1.0) * 100.0
