"""Scoring rules and the walk-forward harness."""

import pytest

from models import backtest


def test_rps_bounds():
    assert backtest.rps(1.0, 0.0, 0.0, "home") == 0.0
    assert backtest.rps(0.0, 0.0, 1.0, "home") == 1.0


def test_rps_punishes_confident_errors_more_than_hedged_ones():
    confident_wrong = backtest.rps(0.9, 0.05, 0.05, "away")
    hedged_wrong = backtest.rps(0.4, 0.3, 0.3, "away")
    assert confident_wrong > hedged_wrong


def test_rps_respects_outcome_ordering():
    """Predicting home when it was a draw beats predicting home when it was away."""
    near_miss = backtest.rps(0.8, 0.1, 0.1, "draw")
    far_miss = backtest.rps(0.8, 0.1, 0.1, "away")
    assert near_miss < far_miss


def test_outcome_of():
    assert backtest.outcome_of({"home_goals": 2, "away_goals": 1}) == "home"
    assert backtest.outcome_of({"home_goals": 1, "away_goals": 1}) == "draw"
    assert backtest.outcome_of({"home_goals": 0, "away_goals": 3}) == "away"


def test_base_rates_sum_to_one(synthetic_results):
    rates = backtest.base_rates(synthetic_results)
    assert sum(rates.values()) == pytest.approx(1.0)


def test_walk_forward_beats_the_baseline(synthetic_results):
    report = backtest.walk_forward(synthetic_results, test_matches=60, refit_every=20)
    assert report["matches_scored"] > 0
    assert report["model_rps"] < report["baseline_rps"]
    assert 0.0 <= report["hit_rate"] <= 1.0


def test_walk_forward_refuses_an_impossible_split(synthetic_results):
    from models.dixon_coles import ModelError

    with pytest.raises(ModelError):
        backtest.walk_forward(synthetic_results, test_matches=len(synthetic_results))
