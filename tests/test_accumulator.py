"""The multi-leg bet builder."""

import math

import pytest

from models import accumulator


def _candidate(fixture, selection, price, prob, fair=None):
    return {
        "fixture": fixture,
        "match": fixture,
        "kickoff": "2030-01-01T15:00:00Z",
        "selection": selection,
        "price": price,
        "price_book": "Betway",
        "model_prob": prob,
        "fair_prob": fair if fair is not None else 1 / price,
        "edge": 0.0,
    }


@pytest.fixture
def candidates():
    return [
        _candidate("A v B", "A win", 2.0, 0.55),
        _candidate("A v B", "Over 2.5", 1.9, 0.54),
        _candidate("C v D", "C win", 2.0, 0.52),
        _candidate("C v D", "Under 2.5", 2.1, 0.50),
        _candidate("E v F", "Draw", 3.5, 0.30),
        _candidate("G v H", "H win", 1.5, 0.70),
    ]


def test_lands_within_tolerance(candidates):
    built = accumulator.build(candidates, target_odds=4.0, tolerance_pct=10.0)
    assert built
    for entry in built:
        assert 3.6 <= entry["combined_odds"] <= 4.4


def test_never_two_legs_from_one_fixture(candidates):
    built = accumulator.build(candidates, target_odds=4.0, tolerance_pct=25.0, max_results=20)
    for entry in built:
        fixtures = [leg["fixture"] for leg in entry["legs"]]
        assert len(fixtures) == len(set(fixtures))


def test_combined_odds_are_the_product_of_legs(candidates):
    entry = accumulator.build(candidates, target_odds=4.0, tolerance_pct=15.0)[0]
    product = math.prod(leg["price"] for leg in entry["legs"])
    assert entry["combined_odds"] == pytest.approx(product)


def test_success_probability_is_the_product_of_legs(candidates):
    entry = accumulator.build(candidates, target_odds=4.0, tolerance_pct=15.0)[0]
    product = math.prod(leg["model_prob"] for leg in entry["legs"])
    assert entry["model_success"] == pytest.approx(product)


def test_fair_odds_invert_the_success_probability(candidates):
    entry = accumulator.build(candidates, target_odds=4.0, tolerance_pct=15.0)[0]
    assert entry["fair_odds"] == pytest.approx(1.0 / entry["model_success"])


def test_expected_value_matches_odds_and_probability(candidates):
    entry = accumulator.build(candidates, target_odds=4.0, tolerance_pct=15.0)[0]
    expected = (entry["model_success"] * entry["combined_odds"] - 1.0) * 100.0
    assert entry["expected_value_pct"] == pytest.approx(expected)


def test_leg_bounds_are_respected(candidates):
    built = accumulator.build(
        candidates, target_odds=6.0, tolerance_pct=40.0, min_legs=3, max_legs=3, max_results=20
    )
    assert built
    assert all(len(entry["legs"]) == 3 for entry in built)


def test_unreachable_target_returns_nothing(candidates):
    assert accumulator.build(candidates, target_odds=500.0, tolerance_pct=1.0) == []


def test_probability_objective_beats_value_objective_on_likelihood(candidates):
    likely = accumulator.build(candidates, target_odds=4.0, tolerance_pct=20.0, objective="probability")
    valuable = accumulator.build(candidates, target_odds=4.0, tolerance_pct=20.0, objective="value")
    assert likely[0]["model_success"] >= valuable[0]["model_success"]


def test_rejects_impossible_target():
    with pytest.raises(ValueError):
        accumulator.build([], target_odds=1.0)


def test_margin_compounds_with_leg_count():
    assert accumulator.margin_compounding(1) == pytest.approx(5.0)
    assert accumulator.margin_compounding(4) > accumulator.margin_compounding(2)
    assert accumulator.margin_compounding(4) == pytest.approx(21.55, abs=0.01)
