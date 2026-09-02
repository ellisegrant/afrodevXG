"""Absence adjustments: the multiplier maths and the model patching."""

import pytest

from models import availability, dixon_coles


def test_no_absences_leaves_scoring_untouched():
    assert availability.attack_multiplier([]) == 1.0


def test_a_bigger_share_costs_more():
    small = availability.attack_multiplier([0.10])
    large = availability.attack_multiplier([0.30])
    assert large < small < 1.0


def test_replacement_level_softens_the_hit():
    no_replacement = availability.attack_multiplier([0.30], replacement_level=0.0)
    half_replaced = availability.attack_multiplier([0.30], replacement_level=0.5)
    fully_replaced = availability.attack_multiplier([0.30], replacement_level=1.0)
    assert no_replacement == pytest.approx(0.70)
    assert half_replaced == pytest.approx(0.85)
    assert fully_replaced == pytest.approx(1.0)


def test_multiple_absences_compound():
    combined = availability.attack_multiplier([0.30, 0.20])
    assert combined == pytest.approx(0.85 * 0.90)


def test_multiplier_has_a_floor():
    """Losing everyone should not drive expected goals to nothing."""
    assert availability.attack_multiplier([0.9, 0.9, 0.9]) == availability.MIN_ATTACK_MULTIPLIER


def test_adjustment_lowers_scoring_then_restores_the_model(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    before = dixon_coles.predict_match(model, "Team J", "Team A")

    with availability.adjusted(model, attack={"Team J": 0.7}):
        weakened = dixon_coles.predict_match(model, "Team J", "Team A")

    after = dixon_coles.predict_match(model, "Team J", "Team A")

    assert weakened["home_xg"] < before["home_xg"]
    assert weakened["home_win"] < before["home_win"]
    assert after["home_xg"] == pytest.approx(before["home_xg"])
    assert after["home_win"] == pytest.approx(before["home_win"])


def test_attack_multiplier_scales_expected_goals_proportionally(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    before = dixon_coles.predict_match(model, "Team J", "Team A")
    with availability.adjusted(model, attack={"Team J": 0.5}):
        halved = dixon_coles.predict_match(model, "Team J", "Team A")
    assert halved["home_xg"] == pytest.approx(before["home_xg"] * 0.5, rel=0.02)


def test_weakening_a_defence_helps_the_opponent(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    before = dixon_coles.predict_match(model, "Team J", "Team A")
    with availability.adjusted(model, defence={"Team J": 1.4}):
        leaky = dixon_coles.predict_match(model, "Team J", "Team A")
    assert leaky["away_xg"] > before["away_xg"]


def test_model_is_restored_even_if_prediction_raises(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    before = dixon_coles.predict_match(model, "Team J", "Team A")
    with pytest.raises(RuntimeError):
        with availability.adjusted(model, attack={"Team J": 0.5}):
            raise RuntimeError("boom")
    after = dixon_coles.predict_match(model, "Team J", "Team A")
    assert after["home_xg"] == pytest.approx(before["home_xg"])


def test_unknown_team_is_ignored_not_fatal(synthetic_results):
    model = dixon_coles.fit_model(synthetic_results)
    with availability.adjusted(model, attack={"Nonexistent FC": 0.5}):
        prediction = dixon_coles.predict_match(model, "Team J", "Team A")
    assert prediction["home_win"] > 0
