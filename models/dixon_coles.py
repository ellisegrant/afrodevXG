"""Dixon-Coles goal model - the brain of the server.

Fitting is slow (seconds) so a fitted model is cached per league; prediction off
a fitted model is fast. This module knows maths, not HTTP or MCP.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import penaltyblog as pb

log = logging.getLogger(__name__)

# Dixon-Coles time decay. 0.0018/day halves a match's weight after ~1 year.
DEFAULT_XI = 0.0018

# Refit at most once a day per league.
MODEL_TTL_SECONDS = 24 * 3600.0

# competition code -> {"model", "fitted_at", "matches", "teams"}
_MODEL_CACHE: dict[str, dict[str, Any]] = {}


class ModelError(RuntimeError):
    """Raised when the model cannot be fitted or a team is unknown to it."""


def _time_weights(dates: list[str], xi: float) -> Optional[np.ndarray]:
    """Exponential decay weights, newest match ~1.0. None if dates are missing."""
    if not dates or any(not date for date in dates):
        return None
    try:
        parsed = [datetime.fromisoformat(date).replace(tzinfo=timezone.utc) for date in dates]
    except ValueError:
        log.warning("unparseable match dates, fitting without time decay")
        return None

    latest = max(parsed)
    days = np.array([(latest - date).days for date in parsed], dtype=float)
    return np.exp(-xi * days)


def fit_model(results: list[dict], xi: float = DEFAULT_XI):
    """Fit a Dixon-Coles model on `results` from `data.fixtures.get_recent_results`."""
    if not results:
        raise ModelError("no historical results to fit on")
    if len(results) < 40:
        log.warning("fitting on only %d matches - estimates will be shaky", len(results))

    goals_home = [int(match["home_goals"]) for match in results]
    goals_away = [int(match["away_goals"]) for match in results]
    teams_home = [match["home"] for match in results]
    teams_away = [match["away"] for match in results]
    weights = _time_weights([match.get("date", "") for match in results], xi)

    started = time.monotonic()
    try:
        if weights is not None:
            model = pb.models.DixonColesGoalModel(
                goals_home, goals_away, teams_home, teams_away, weights
            )
        else:
            model = pb.models.DixonColesGoalModel(
                goals_home, goals_away, teams_home, teams_away
            )
        model.fit()
    except Exception as exc:  # penaltyblog raises plain exceptions on bad input
        raise ModelError(f"Dixon-Coles fit failed: {exc}") from exc

    log.info("fitted Dixon-Coles on %d matches in %.1fs", len(results), time.monotonic() - started)
    return model


def get_model(competition_code: str, results: list[dict], xi: float = DEFAULT_XI):
    """Fitted model for a league, refitted at most once per `MODEL_TTL_SECONDS`."""
    key = f"{competition_code.upper()}:{xi}"
    entry = _MODEL_CACHE.get(key)
    if (
        entry
        and time.time() - entry["fitted_at"] < MODEL_TTL_SECONDS
        and entry["matches"] == len(results)
    ):
        log.info("using cached model for %s", key)
        return entry["model"]

    model = fit_model(results, xi=xi)
    _MODEL_CACHE[key] = {
        "model": model,
        "fitted_at": time.time(),
        "matches": len(results),
        "teams": model_teams(model),
    }
    return model


def model_teams(model) -> list[str]:
    """Team names the fitted model knows about."""
    for attribute in ("teams", "team_names"):
        teams = getattr(model, attribute, None)
        if teams is not None:
            return sorted(str(team) for team in teams)
    return []


def _grid_matrix(grid) -> np.ndarray:
    """The score-probability matrix behind a penaltyblog probability grid."""
    for attribute in ("goal_matrix", "grid", "probability_grid", "matrix", "_grid"):
        candidate = getattr(grid, attribute, None)
        if candidate is not None:
            return np.asarray(candidate, dtype=float)
    return np.asarray(grid, dtype=float)


def _goal_expectations(grid) -> tuple[float, float]:
    """Expected goals per side, read off the grid if not exposed directly."""
    home = getattr(grid, "home_goal_expectation", None)
    away = getattr(grid, "away_goal_expectation", None)
    if home is not None and away is not None:
        return float(home), float(away)

    matrix = _grid_matrix(grid)
    home_goals = np.arange(matrix.shape[0])
    away_goals = np.arange(matrix.shape[1])
    total = matrix.sum()
    return (
        float((matrix.sum(axis=1) * home_goals).sum() / total),
        float((matrix.sum(axis=0) * away_goals).sum() / total),
    )


def _totals(grid, line: float = 2.5) -> tuple[float, float]:
    """(over, under) probabilities for a goals line."""
    try:
        over = float(grid.total_goals("over", line))
        under = float(grid.total_goals("under", line))
        if over > 0 or under > 0:
            return over, under
    except Exception:  # noqa: BLE001 - fall back to the raw grid
        pass

    matrix = _grid_matrix(grid)
    totals = np.add.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    over = float(matrix[totals > line].sum())
    return over, float(matrix.sum() - over)


def _btts(grid) -> float:
    for attribute in ("both_teams_to_score", "btts_yes", "btts"):
        value = getattr(grid, attribute, None)
        if value is not None:
            return float(value)
    matrix = _grid_matrix(grid)
    return float(matrix[1:, 1:].sum())


def predict_match(model, home: str, away: str) -> dict[str, float]:
    """1X2, totals, BTTS and goal expectations for one fixture."""
    known = model_teams(model)
    if known:
        for team, role in ((home, "home"), (away, "away")):
            if team not in known:
                raise ModelError(
                    f"unknown {role} team {team!r}. Known teams: {', '.join(known)}"
                )

    try:
        grid = model.predict(home, away)
    except Exception as exc:
        raise ModelError(f"prediction failed for {home} v {away}: {exc}") from exc

    over_2_5, under_2_5 = _totals(grid, 2.5)
    home_xg, away_xg = _goal_expectations(grid)

    return {
        "home_win": float(grid.home_win),
        "draw": float(grid.draw),
        "away_win": float(grid.away_win),
        "over_2_5": over_2_5,
        "under_2_5": under_2_5,
        "btts_yes": _btts(grid),
        "home_xg": home_xg,
        "away_xg": away_xg,
    }
