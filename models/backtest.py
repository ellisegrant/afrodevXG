"""Walk-forward backtesting for the Dixon-Coles model.

Answers the only question that matters: are these probabilities any good?
Scoring is by Ranked Probability Score (RPS), the standard metric for ordered
football outcomes - lower is better, and it punishes confident wrong calls
harder than a plain log loss does.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from . import dixon_coles

log = logging.getLogger(__name__)

# Outcome ordering matters for RPS: home > draw > away is the natural ranking.
_OUTCOMES = ("home", "draw", "away")


def outcome_of(match: dict) -> str:
    if match["home_goals"] > match["away_goals"]:
        return "home"
    if match["home_goals"] < match["away_goals"]:
        return "away"
    return "draw"


def rps(home: float, draw: float, away: float, outcome: str) -> float:
    """Ranked Probability Score for one match. 0 is perfect, 1 is worst."""
    predicted = np.cumsum([home, draw, away])[:-1]
    actual = np.cumsum([1.0 if _OUTCOMES[i] == outcome else 0.0 for i in range(3)])[:-1]
    return float(np.sum((predicted - actual) ** 2) / (len(_OUTCOMES) - 1))


def base_rates(results: list[dict]) -> dict[str, float]:
    """Empirical home/draw/away rates - the baseline any model must beat."""
    counts = {"home": 0, "draw": 0, "away": 0}
    for match in results:
        counts[outcome_of(match)] += 1
    total = max(len(results), 1)
    return {outcome: count / total for outcome, count in counts.items()}


def _calibration(records: list[dict], bins: int = 5) -> list[dict[str, Any]]:
    """Do things predicted at 30% actually happen 30% of the time?

    Pools all three outcomes of every scored match into probability buckets.
    """
    predicted, happened = [], []
    for record in records:
        for outcome in _OUTCOMES:
            predicted.append(record["probs"][outcome])
            happened.append(1.0 if record["outcome"] == outcome else 0.0)

    predicted_array = np.asarray(predicted)
    happened_array = np.asarray(happened)
    edges = np.linspace(0.0, 1.0, bins + 1)

    table = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (predicted_array >= low) & (predicted_array < high)
        if not mask.any():
            continue
        table.append(
            {
                "bucket": f"{low:.0%}-{high:.0%}",
                "predicted_mean": float(predicted_array[mask].mean()),
                "observed_rate": float(happened_array[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return table


def _brier(probabilities: list[float], outcomes: list[float]) -> float:
    """Mean squared error of a two-way probability. Lower is better."""
    return float(
        np.mean([(p - o) ** 2 for p, o in zip(probabilities, outcomes)])
    )


def _score_goals_markets(
    records: list[dict[str, Any]], training: list[dict]
) -> dict[str, Any]:
    """Score over/under 2.5 and both-teams-to-score, and check for goal bias.

    Half the scanner's picks are goals markets, and until now nothing measured
    them. A model can rank match results well and still be systematically wrong
    about how many goals a game will contain.
    """
    over_rate = (
        sum(1 for match in training if match["home_goals"] + match["away_goals"] > 2.5)
        / max(len(training), 1)
    )
    btts_rate = (
        sum(1 for match in training if match["home_goals"] > 0 and match["away_goals"] > 0)
        / max(len(training), 1)
    )
    training_goals = (
        sum(match["home_goals"] + match["away_goals"] for match in training)
        / max(len(training), 1)
    )

    over_outcomes = [record["over_happened"] for record in records]
    btts_outcomes = [record["btts_happened"] for record in records]

    return {
        "totals_brier": _brier([r["p_over"] for r in records], over_outcomes),
        "totals_baseline_brier": _brier([over_rate] * len(records), over_outcomes),
        "predicted_over_rate": float(np.mean([r["p_over"] for r in records])),
        "actual_over_rate": float(np.mean(over_outcomes)),
        "btts_brier": _brier([r["p_btts"] for r in records], btts_outcomes),
        "btts_baseline_brier": _brier([btts_rate] * len(records), btts_outcomes),
        "predicted_btts_rate": float(np.mean([r["p_btts"] for r in records])),
        "actual_btts_rate": float(np.mean(btts_outcomes)),
        "predicted_goals_mean": float(np.mean([r["predicted_goals"] for r in records])),
        "actual_goals_mean": float(np.mean([r["actual_goals"] for r in records])),
        "training_goals_mean": training_goals,
    }


def walk_forward(
    results: list[dict],
    test_matches: int = 100,
    xi: float = dixon_coles.DEFAULT_XI,
    refit_every: int = 10,
    model_name: str = "dixon_coles",
    shrinkage_k: float = dixon_coles.SHRINKAGE_K,
) -> dict[str, Any]:
    """Predict the last `test_matches` using only matches played before each one.

    The model is refitted every `refit_every` matches rather than every match -
    a full refit is cheap but not free, and a handful of extra matches barely
    moves the parameters.
    """
    if len(results) <= test_matches + 50:
        raise dixon_coles.ModelError(
            f"need more than {test_matches + 50} matches to backtest "
            f"{test_matches} of them; got {len(results)}"
        )

    ordered = sorted(results, key=lambda match: match.get("date", ""))
    split = len(ordered) - test_matches

    records: list[dict[str, Any]] = []
    skipped = 0
    model = None

    for index in range(split, len(ordered)):
        match = ordered[index]
        if model is None or (index - split) % refit_every == 0:
            model = dixon_coles.fit_model(
                ordered[:index], xi=xi, model_name=model_name, shrinkage_k=shrinkage_k
            )
            known = set(dixon_coles.model_teams(model))

        if match["home"] not in known or match["away"] not in known:
            skipped += 1  # promoted side with no history yet
            continue

        try:
            prediction = dixon_coles.predict_match(model, match["home"], match["away"])
        except dixon_coles.ModelError as exc:
            log.warning("skipping %s v %s: %s", match["home"], match["away"], exc)
            skipped += 1
            continue

        probs = {
            "home": prediction["home_win"],
            "draw": prediction["draw"],
            "away": prediction["away_win"],
        }
        total_goals = match["home_goals"] + match["away_goals"]
        records.append(
            {
                "date": match.get("date", ""),
                "home": match["home"],
                "away": match["away"],
                "probs": probs,
                "outcome": outcome_of(match),
                "rps": rps(probs["home"], probs["draw"], probs["away"], outcome_of(match)),
                # Goals markets, scored separately: half the scanner's picks are
                # over/under, and nothing was measuring them.
                "p_over": prediction["over_2_5"],
                "over_happened": 1.0 if total_goals > 2.5 else 0.0,
                "p_btts": prediction["btts_yes"],
                "btts_happened": 1.0 if (match["home_goals"] > 0 and match["away_goals"] > 0) else 0.0,
                "predicted_goals": prediction["home_xg"] + prediction["away_xg"],
                "actual_goals": total_goals,
            }
        )

    if not records:
        raise dixon_coles.ModelError("no test matches could be scored")

    return {
        "xi": xi,
        "model_name": model_name,
        "shrinkage_k": shrinkage_k,
        "matches_trained_on": split,
        "matches_skipped": skipped,
        **summarise(records, ordered[:split]),
    }


def summarise(records: list[dict[str, Any]], training: list[dict]) -> dict[str, Any]:
    """Turn scored predictions into the numbers a report needs."""
    rates = base_rates(training)
    baseline = float(
        np.mean(
            [rps(rates["home"], rates["draw"], rates["away"], record["outcome"])
             for record in records]
        )
    )
    model_rps = float(np.mean([record["rps"] for record in records]))
    hits = sum(
        1 for record in records
        if max(record["probs"], key=record["probs"].get) == record["outcome"]
    )

    return {
        "matches_scored": len(records),
        "model_rps": model_rps,
        "baseline_rps": baseline,
        "rps_improvement_pct": (baseline - model_rps) / baseline * 100.0 if baseline else 0.0,
        "baseline_rates": rates,
        "hit_rate": hits / len(records),
        "calibration": _calibration(records),
        "worst_calls": sorted(records, key=lambda r: -r["rps"])[:5],
        **_score_goals_markets(records, training),
    }


def _predict_record(model, match: dict) -> Optional[dict[str, Any]]:
    """Score one match against a fitted model, or None if it cannot be priced."""
    try:
        prediction = dixon_coles.predict_match(model, match["home"], match["away"])
    except dixon_coles.ModelError:
        return None

    probs = {
        "home": prediction["home_win"],
        "draw": prediction["draw"],
        "away": prediction["away_win"],
    }
    total_goals = match["home_goals"] + match["away_goals"]
    return {
        "date": match.get("date", ""),
        "home": match["home"],
        "away": match["away"],
        "probs": probs,
        "outcome": outcome_of(match),
        "rps": rps(probs["home"], probs["draw"], probs["away"], outcome_of(match)),
        "p_over": prediction["over_2_5"],
        "over_happened": 1.0 if total_goals > 2.5 else 0.0,
        "p_btts": prediction["btts_yes"],
        "btts_happened": 1.0 if (match["home_goals"] > 0 and match["away_goals"] > 0) else 0.0,
        "predicted_goals": prediction["home_xg"] + prediction["away_xg"],
        "actual_goals": total_goals,
    }


def holdout(
    train: list[dict],
    test: list[dict],
    xi: float = dixon_coles.DEFAULT_XI,
    model_name: str = "dixon_coles",
    shrinkage_k: float = dixon_coles.SHRINKAGE_K,
) -> dict[str, Any]:
    """Fit on one block of matches and predict another, untouched, block.

    Harder than walk-forward and closer to real use at the start of a season:
    the model gets no information at all from the period it is predicting, so
    transfers, new managers and summer form are invisible to it. Teams absent
    from the training block cannot be priced and are reported as skipped.
    """
    if not train or not test:
        raise dixon_coles.ModelError("both a training and a test block are needed")

    model = dixon_coles.fit_model(
        train, xi=xi, model_name=model_name, shrinkage_k=shrinkage_k
    )
    known = set(dixon_coles.model_teams(model))

    records, skipped, unknown_teams = [], 0, set()
    for match in sorted(test, key=lambda match: match.get("date", "")):
        missing = [team for team in (match["home"], match["away"]) if team not in known]
        if missing:
            unknown_teams.update(missing)
            skipped += 1
            continue
        record = _predict_record(model, match)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    if not records:
        raise dixon_coles.ModelError(
            "no test matches could be scored - none of these teams appear in the "
            "training block"
        )

    return {
        "xi": xi,
        "model_name": model_name,
        "shrinkage_k": shrinkage_k,
        "matches_trained_on": len(train),
        "matches_skipped": skipped,
        "unknown_teams": sorted(unknown_teams),
        "first_test_date": records[0]["date"],
        "last_test_date": records[-1]["date"],
        **summarise(records, train),
    }


def tune_xi(
    results: list[dict],
    candidates: Optional[list[float]] = None,
    test_matches: int = 100,
    refit_every: int = 10,
) -> dict[str, Any]:
    """Grid-search the time-decay constant by backtest RPS.

    xi=0 is no decay at all (every match weighted equally); bigger values forget
    the past faster.
    """
    candidates = candidates or [0.0, 0.001, 0.0018, 0.003, 0.005, 0.01]

    trials = []
    baseline = 0.0
    for candidate in candidates:
        result = walk_forward(
            results, test_matches=test_matches, xi=candidate, refit_every=refit_every
        )
        baseline = result["baseline_rps"]  # identical across xi, capture once
        trials.append(
            {
                "xi": candidate,
                "model_rps": result["model_rps"],
                "hit_rate": result["hit_rate"],
                "matches_scored": result["matches_scored"],
            }
        )
        log.info("xi=%.4f -> RPS %.5f", candidate, result["model_rps"])

    best = min(trials, key=lambda trial: trial["model_rps"])
    return {
        "trials": trials,
        "best_xi": best["xi"],
        "best_rps": best["model_rps"],
        "current_default": dixon_coles.DEFAULT_XI,
        "baseline_rps": baseline,
    }


def compare_models(
    results: list[dict],
    model_names: Optional[list[str]] = None,
    test_matches: int = 100,
    refit_every: int = 10,
) -> dict[str, Any]:
    """Rank the available goal models on the same held-out matches.

    Dixon-Coles is the default because it is the standard, not because it has
    been shown to be the best fit for any particular league.
    """
    model_names = model_names or list(dixon_coles.MODEL_CLASSES)

    trials, baseline = [], 0.0
    for name in model_names:
        started = time.monotonic()
        try:
            result = walk_forward(
                results, test_matches=test_matches, refit_every=refit_every, model_name=name
            )
        except Exception as exc:  # a model that will not fit is a result too
            log.warning("%s failed: %s", name, exc)
            trials.append({"model": name, "error": str(exc)[:200]})
            continue
        baseline = result["baseline_rps"]
        trials.append(
            {
                "model": name,
                "model_rps": result["model_rps"],
                "hit_rate": result["hit_rate"],
                "matches_scored": result["matches_scored"],
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        log.info("%s -> RPS %.5f", name, result["model_rps"])

    scored = [trial for trial in trials if "model_rps" in trial]
    best = min(scored, key=lambda trial: trial["model_rps"]) if scored else None
    return {
        "trials": trials,
        "best_model": best["model"] if best else "none",
        "best_rps": best["model_rps"] if best else 0.0,
        "baseline_rps": baseline,
    }
