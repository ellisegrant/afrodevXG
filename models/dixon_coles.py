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

# Matches of history a team needs before its own record outweighs the league
# average. A promoted side with two games otherwise gets a wild, unconstrained
# strength estimate that shows up as a huge fake edge.
SHRINKAGE_K = 8.0

# Above this many matches a team's own record is taken at face value. Without
# it, n/(n+k) still pulls a side with a hundred matches several percent toward
# the mean, which measurably costs accuracy on exactly the fixtures we know
# most about.
SHRINKAGE_FULL_CONFIDENCE = 30

# Every penaltyblog goal model takes the same constructor arguments, so they are
# interchangeable here and can be ranked against each other by backtest.
MODEL_CLASSES = {
    "dixon_coles": "DixonColesGoalModel",
    "poisson": "PoissonGoalsModel",
    "bivariate_poisson": "BivariatePoissonGoalModel",
    "negative_binomial": "NegativeBinomialGoalModel",
    "zero_inflated_poisson": "ZeroInflatedPoissonGoalsModel",
    "weibull_copula": "WeibullCopulaGoalsModel",
}

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


def fit_model(
    results: list[dict],
    xi: float = DEFAULT_XI,
    shrinkage_k: float = SHRINKAGE_K,
    model_name: str = "dixon_coles",
):
    """Fit a goal model on `results` from `data.fixtures.get_recent_results`."""
    if model_name not in MODEL_CLASSES:
        raise ModelError(
            f"unknown model {model_name!r}. Available: {', '.join(MODEL_CLASSES)}"
        )
    model_class = getattr(pb.models, MODEL_CLASSES[model_name])
    if not results:
        raise ModelError("no historical results to fit on")
    if len(results) < 40:
        log.warning("fitting on only %d matches - estimates will be shaky", len(results))

    goals_home = [int(match["home_goals"]) for match in results]
    goals_away = [int(match["away_goals"]) for match in results]
    teams_home = [match["home"] for match in results]
    teams_away = [match["away"] for match in results]
    weights = _time_weights([match.get("date", "") for match in results], xi)
    # A match can carry its own weight - second-division games count for less.
    explicit = np.array([float(match.get("weight", 1.0)) for match in results])
    if weights is None:
        weights = explicit if not np.allclose(explicit, 1.0) else None
    else:
        weights = weights * explicit

    started = time.monotonic()
    try:
        if weights is not None:
            model = model_class(goals_home, goals_away, teams_home, teams_away, weights)
        else:
            model = model_class(goals_home, goals_away, teams_home, teams_away)
        model.fit()
    except Exception as exc:  # penaltyblog raises plain exceptions on bad input
        raise ModelError(f"{model_name} fit failed: {exc}") from exc

    if shrinkage_k > 0:
        _shrink_team_strengths(model, team_match_counts(results), shrinkage_k)

    log.info(
        "fitted %s on %d matches in %.1fs",
        model_name, len(results), time.monotonic() - started,
    )
    return model



def _shrink_team_strengths(model, counts: dict[str, int], k: float) -> None:
    """Pull thinly-observed teams toward the league average, in place.

    Each team's attack and defence coefficient is blended with the league mean
    using weight n/(n+k): a team with k matches sits halfway, a team with many
    matches keeps essentially its own estimate.
    """
    teams = [str(team) for team in getattr(model, "teams", [])]
    params = getattr(model, "_params", None)
    if not teams or params is None or k <= 0:
        return

    params = np.asarray(params, dtype=float).copy()
    n = len(teams)
    if params.size < 2 * n:
        log.warning("unexpected parameter layout, skipping shrinkage")
        return

    attack, defence = params[:n], params[n : 2 * n]
    mean_attack, mean_defence = float(attack.mean()), float(defence.mean())

    for index, team in enumerate(teams):
        played = counts.get(team, 0)
        if played >= SHRINKAGE_FULL_CONFIDENCE:
            continue
        weight = played / (played + k)
        params[index] = mean_attack + weight * (attack[index] - mean_attack)
        params[n + index] = mean_defence + weight * (defence[index] - mean_defence)

    model._params = params


def get_model(
    competition_code: str,
    results: list[dict],
    xi: float = DEFAULT_XI,
    model_name: str = "dixon_coles",
):
    """Fitted model for a league, refitted at most once per `MODEL_TTL_SECONDS`."""
    key = f"{competition_code.upper()}:{xi}:{model_name}"
    entry = _MODEL_CACHE.get(key)
    if (
        entry
        and time.time() - entry["fitted_at"] < MODEL_TTL_SECONDS
        and entry["matches"] == len(results)
    ):
        log.info("using cached model for %s", key)
        return entry["model"]

    model = fit_model(results, xi=xi, model_name=model_name)
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


def team_match_counts(results: list[dict]) -> dict[str, int]:
    """How many matches of history each team has - thin samples give wild estimates."""
    counts: dict[str, int] = {}
    for match in results:
        counts[match["home"]] = counts.get(match["home"], 0) + 1
        counts[match["away"]] = counts.get(match["away"], 0) + 1
    return counts


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



def _safe(callable_or_value, default: float = 0.0) -> float:
    """penaltyblog raises on some lines; a missing market is not a crash."""
    try:
        value = callable_or_value() if callable(callable_or_value) else callable_or_value
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def market_sheet(
    model, home: str, away: str, top_scores: int = 8
) -> dict[str, Any]:
    """Every market the score grid can price, whether or not a book quotes it.

    The free odds feed only carries 1X2, totals and handicaps, but the same
    fitted grid prices double chance, draw no bet, correct score, clean sheets,
    win to nil and team totals. Those come out as model probabilities for you to
    compare by eye against whatever book you actually use.
    """
    known = model_teams(model)
    for team, role in ((home, "home"), (away, "away")):
        if known and team not in known:
            raise ModelError(f"unknown {role} team {team!r}")

    try:
        grid = model.predict(home, away)
    except Exception as exc:
        raise ModelError(f"prediction failed for {home} v {away}: {exc}") from exc

    matrix = _grid_matrix(grid)
    home_goals = matrix.sum(axis=1)
    away_goals = matrix.sum(axis=0)
    home_xg, away_xg = _goal_expectations(grid)

    goal_lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    totals = {}
    for line in goal_lines:
        over, under = _totals(grid, line)
        totals[f"over_{line}"] = over
        totals[f"under_{line}"] = under

    handicaps = {}
    for strike in (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0):
        handicaps[f"home_{strike:+g}"] = _safe(lambda s=strike: grid.asian_handicap("home", s))
        handicaps[f"away_{strike:+g}"] = _safe(lambda s=strike: grid.asian_handicap("away", s))

    scores = []
    for home_score in range(min(6, matrix.shape[0])):
        for away_score in range(min(6, matrix.shape[1])):
            scores.append(
                {
                    "score": f"{home_score}-{away_score}",
                    "probability": float(matrix[home_score, away_score]),
                }
            )
    scores.sort(key=lambda entry: -entry["probability"])

    return {
        "home_team": home,
        "away_team": away,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "result": {
            "home_win": float(grid.home_win),
            "draw": float(grid.draw),
            "away_win": float(grid.away_win),
        },
        "double_chance": {
            "home_or_draw": _safe(getattr(grid, "double_chance_1x", 0.0)),
            "home_or_away": _safe(getattr(grid, "double_chance_12", 0.0)),
            "draw_or_away": _safe(getattr(grid, "double_chance_x2", 0.0)),
        },
        "draw_no_bet": {
            "home": _safe(getattr(grid, "draw_no_bet_home", 0.0)),
            "away": _safe(getattr(grid, "draw_no_bet_away", 0.0)),
        },
        "totals": totals,
        "btts": {"yes": _btts(grid), "no": 1.0 - _btts(grid)},
        "clean_sheet": {
            "home": float(away_goals[0]),
            "away": float(home_goals[0]),
        },
        "win_to_nil": {
            "home": _safe(getattr(grid, "win_to_nil_home", 0.0)),
            "away": _safe(getattr(grid, "win_to_nil_away", 0.0)),
        },
        "team_totals": {
            "home_over_0.5": float(1.0 - home_goals[0]),
            "home_over_1.5": float(home_goals[2:].sum()),
            "home_over_2.5": float(home_goals[3:].sum()),
            "away_over_0.5": float(1.0 - away_goals[0]),
            "away_over_1.5": float(away_goals[2:].sum()),
            "away_over_2.5": float(away_goals[3:].sum()),
        },
        "handicaps": handicaps,
        "correct_score": scores[:top_scores],
        "expected_points": {
            "home": _safe(getattr(grid, "expected_points_home", 0.0)),
            "away": _safe(getattr(grid, "expected_points_away", 0.0)),
        },
    }


def predict_handicap(
    model, home: str, away: str, line: float
) -> tuple[float, float]:
    """(home covers, away covers) for an Asian handicap quoted on the home side."""
    try:
        grid = model.predict(home, away)
    except Exception as exc:
        raise ModelError(f"prediction failed for {home} v {away}: {exc}") from exc
    return (
        _safe(lambda: grid.asian_handicap("home", line)),
        _safe(lambda: grid.asian_handicap("away", -line)),
    )


def predict_totals(model, home: str, away: str, line: float) -> tuple[float, float]:
    """(over, under) probabilities for an arbitrary goals line."""
    try:
        grid = model.predict(home, away)
    except Exception as exc:
        raise ModelError(f"prediction failed for {home} v {away}: {exc}") from exc
    return _totals(grid, line)


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
