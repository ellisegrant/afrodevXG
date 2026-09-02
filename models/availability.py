"""Adjust a fitted model for players who will not be playing.

The fit knows what a team has done historically, which implicitly includes its
best players. When one of them is out, the team's expected goals should fall -
but not by the player's full output, because someone replaces them and penalty
duty transfers.

The adjustment is deliberately simple and visible:

    attack multiplier = 1 - goal_share * (1 - replacement_level)

with `goal_share` the fraction of the team's open-play goals that player scored,
and `replacement_level` how much of that a stand-in is assumed to provide.

This is an extrapolation. The model was never fitted on lineups, and the free
data tier carries no historical team-sheets, so unlike every other choice in
this project it cannot be backtested. Treat the output as a considered
adjustment, not a measured one.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import numpy as np

log = logging.getLogger(__name__)

# Share of an absent player's output that a replacement is assumed to provide.
DEFAULT_REPLACEMENT_LEVEL = 0.5

# However many stars are missing, refuse to scale a team's attack below this.
MIN_ATTACK_MULTIPLIER = 0.55


def attack_multiplier(
    goal_shares: list[float], replacement_level: float = DEFAULT_REPLACEMENT_LEVEL
) -> float:
    """Combined scoring multiplier for a set of absences."""
    multiplier = 1.0
    for share in goal_shares:
        multiplier *= 1.0 - max(share, 0.0) * (1.0 - replacement_level)
    return max(multiplier, MIN_ATTACK_MULTIPLIER)


def _team_index(model, team: str) -> Optional[int]:
    teams = [str(name) for name in getattr(model, "teams", [])]
    return teams.index(team) if team in teams else None


@contextmanager
def adjusted(
    model,
    attack: Optional[dict[str, float]] = None,
    defence: Optional[dict[str, float]] = None,
) -> Iterator[Any]:
    """Temporarily scale team strengths, restoring the model on the way out.

    Both coefficients are on a log scale, so scaling expected goals by `m` means
    adding log(m). Attack is a team's own scoring; defence is how much the
    opposition is expected to score against them, so a weakened defence means a
    multiplier above 1.
    """
    attack, defence = attack or {}, defence or {}
    original = np.asarray(model._params, dtype=float).copy()
    teams = [str(name) for name in getattr(model, "teams", [])]
    count = len(teams)

    try:
        params = np.asarray(model._params, dtype=float).copy()
        for team, multiplier in attack.items():
            index = _team_index(model, team)
            if index is None or multiplier <= 0:
                log.warning("cannot adjust attack for unknown team %r", team)
                continue
            params[index] += math.log(multiplier)
        for team, multiplier in defence.items():
            index = _team_index(model, team)
            if index is None or multiplier <= 0:
                log.warning("cannot adjust defence for unknown team %r", team)
                continue
            params[count + index] += math.log(multiplier)
        model._params = params
        yield model
    finally:
        model._params = original
