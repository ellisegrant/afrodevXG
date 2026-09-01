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
        records.append(
            {
                "date": match.get("date", ""),
                "home": match["home"],
                "away": match["away"],
                "probs": probs,
                "outcome": outcome_of(match),
                "rps": rps(probs["home"], probs["draw"], probs["away"], outcome_of(match)),
            }
        )

    if not records:
        raise dixon_coles.ModelError("no test matches could be scored")

    rates = base_rates(ordered[:split])
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
        "xi": xi,
        "model_name": model_name,
        "shrinkage_k": shrinkage_k,
        "matches_trained_on": split,
        "matches_scored": len(records),
        "matches_skipped": skipped,
        "model_rps": model_rps,
        "baseline_rps": baseline,
        "rps_improvement_pct": (baseline - model_rps) / baseline * 100.0 if baseline else 0.0,
        "baseline_rates": rates,
        "hit_rate": hits / len(records),
        "calibration": _calibration(records),
        "worst_calls": sorted(records, key=lambda r: -r["rps"])[:5],
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
