"""Fitting, shrinkage and prediction on a synthetic league."""

import pytest

from models import dixon_coles


def test_fit_learns_the_right_ordering(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    strong = dixon_coles.predict_match(model, "Team J", "Team A")
    weak = dixon_coles.predict_match(model, "Team A", "Team J")
    assert strong["home_win"] > strong["away_win"]
    assert weak["away_win"] > weak["home_win"]


def test_probabilities_are_coherent(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    prediction = dixon_coles.predict_match(model, "Team C", "Team G")
    assert prediction["home_win"] + prediction["draw"] + prediction["away_win"] == pytest.approx(1.0, abs=0.02)
    assert prediction["over_2_5"] + prediction["under_2_5"] == pytest.approx(1.0, abs=0.02)
    assert 0.0 < prediction["btts_yes"] < 1.0
    assert 0.0 < prediction["home_xg"] < 6.0


def test_unknown_team_is_rejected(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    with pytest.raises(dixon_coles.ModelError):
        dixon_coles.predict_match(model, "Team A", "Nonexistent FC")


def test_empty_results_rejected():
    with pytest.raises(dixon_coles.ModelError):
        dixon_coles.fit_model([])


def test_team_match_counts(synthetic_results):
    counts = dixon_coles.team_match_counts(synthetic_results)
    assert len(counts) == 10
    assert all(count > 0 for count in counts.values())


def test_shrinkage_pulls_a_thin_team_toward_average(synthetic_results):
    """A team with two matches should not be treated as confidently as one with many."""
    thin = [match for match in synthetic_results if "Team K" not in (match["home"], match["away"])]
    thin += [
        {"home": "Team K", "away": "Team A", "home_goals": 5, "away_goals": 0, "date": "2024-09-01"},
        {"home": "Team K", "away": "Team B", "home_goals": 4, "away_goals": 0, "date": "2024-09-08"},
    ]

    unshrunk = dixon_coles.predict_match(
        dixon_coles.fit_model(thin, shrinkage_k=0.0), "Team K", "Team J"
    )
    shrunk = dixon_coles.predict_match(
        dixon_coles.fit_model(thin, shrinkage_k=8.0), "Team K", "Team J"
    )
    # Two thrashings should not make Team K a near-certainty against the best side.
    assert shrunk["home_win"] < unshrunk["home_win"]
    assert shrunk["home_xg"] < unshrunk["home_xg"]


def test_totals_can_be_priced_at_any_line(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    over_25, under_25 = dixon_coles.predict_totals(model, "Team C", "Team G", 2.5)
    over_35, _ = dixon_coles.predict_totals(model, "Team C", "Team G", 3.5)
    assert over_25 + under_25 == pytest.approx(1.0, abs=0.02)
    assert over_35 < over_25  # more goals is always less likely


def test_model_cache_returns_the_same_object(synthetic_results):
    first = dixon_coles.get_model("TEST_CACHE", synthetic_results)
    second = dixon_coles.get_model("TEST_CACHE", synthetic_results)
    assert first is second
