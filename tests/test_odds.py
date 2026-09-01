"""De-vigging, name matching and market assembly."""

import pytest

from data import odds


def test_devig_sums_to_one_and_keeps_order():
    fair = odds.devig({"home": 2.0, "draw": 3.5, "away": 4.0})
    assert sum(fair.values()) == pytest.approx(1.0)
    assert fair["home"] > fair["draw"] > fair["away"]


def test_devig_removes_the_margin():
    # These prices imply 105% before de-vigging.
    raw = sum(1 / price for price in (2.0, 3.5, 4.0))
    assert raw > 1.0
    assert sum(odds.devig({"h": 2.0, "d": 3.5, "a": 4.0}).values()) == pytest.approx(1.0)


def test_devig_rejects_impossible_prices():
    with pytest.raises(ValueError):
        odds.devig({"home": 1.0, "draw": 3.0, "away": 4.0})
    with pytest.raises(ValueError):
        odds.devig({"home": 0.0, "draw": 3.0, "away": 4.0})


@pytest.mark.parametrize(
    "left,right",
    [
        ("Manchester United", "Man Utd"),
        ("Tottenham Hotspur", "Spurs"),
        ("Wolverhampton Wanderers", "Wolves"),
        ("Paris Saint-Germain", "PSG"),
        ("Brighton & Hove Albion", "Brighton"),
        ("Atlético Madrid", "Atletico Madrid"),
        ("Arsenal FC", "Arsenal"),
    ],
)
def test_team_names_normalize_to_the_same_token(left, right):
    assert odds.normalize_team(left) == odds.normalize_team(right)


def test_different_clubs_do_not_collide():
    assert odds.normalize_team("Manchester United") != odds.normalize_team("Manchester City")


def test_market_prefers_the_exchange_as_baseline(odds_event):
    market = odds.market_from_event(odds_event, target_book="betway")
    assert market["sharp_book"] == "Betfair"
    assert market["target_book"] == "Betway"
    assert market["target_odds"]["home"] == 1.90
    assert sum(market["fair"].values()) == pytest.approx(1.0)


def test_market_reports_the_best_price_across_books(odds_event):
    market = odds.market_from_event(odds_event, target_book="betway")
    assert market["best_odds"]["home"]["odds"] == 2.00
    assert market["best_odds"]["home"]["book"] == "Betfair"


def test_totals_line_is_chosen_by_consensus(odds_event):
    market = odds.market_from_event(odds_event, target_book="betway")
    assert market["totals"]["line"] == 2.5
    # Betway quotes no totals here, so there is no target price for that market.
    assert market["totals"]["target_odds"] is None
    assert market["totals"]["best_odds"]["under"]["odds"] == 2.05


def test_event_matching_tolerates_name_variants(odds_event):
    assert odds._match_event([odds_event], "Arsenal FC", "Chelsea FC") is odds_event
    assert odds._match_event([odds_event], "Liverpool", "Everton") is None
