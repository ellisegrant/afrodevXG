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


def test_overround_measures_the_margin():
    assert odds.overround({"home": 2.0, "away": 2.0}) == pytest.approx(1.0)
    assert odds.overround({"home": 1.9, "away": 1.9}) == pytest.approx(1.0526, abs=1e-3)


def _illiquid_event():
    """A real shape from the feed: an exchange with no liquidity quoting nonsense.

    Betfair returned 1.27 / 1.10 / 1.09 on Atalanta v Cagliari - an implied 261%.
    De-vigged blindly that made a heavy favourite look like a 30% shot and
    manufactured a 31-point edge.
    """
    return {
        "home_team": "Atalanta BC",
        "away_team": "Cagliari",
        "commence_time": "2030-01-01T15:00:00Z",
        "bookmakers": [
            {"key": "betfair_ex_uk", "title": "Betfair", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Atalanta BC", "price": 1.27},
                    {"name": "Cagliari", "price": 1.10},
                    {"name": "Draw", "price": 1.09},
                ]}]},
            {"key": "betway", "title": "Betway", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "Atalanta BC", "price": 1.50},
                    {"name": "Cagliari", "price": 6.50},
                    {"name": "Draw", "price": 4.20},
                ]}]},
        ],
    }


def test_illiquid_exchange_prices_are_rejected():
    market = odds.market_from_event(_illiquid_event(), target_book="betway")
    assert market["sharp_book"] != "Betfair"
    # The favourite must come back a favourite, not a 30% shot.
    assert market["fair"]["home"] > 0.55
    assert market["sharp_overround_pct"] < 12.0


def test_a_sane_exchange_is_still_preferred(odds_event):
    market = odds.market_from_event(odds_event, target_book="betway")
    assert market["sharp_book"] == "Betfair"


def test_event_with_no_usable_prices_returns_nothing():
    broken = {
        "home_team": "A", "away_team": "B", "commence_time": "2030-01-01T15:00:00Z",
        "bookmakers": [
            {"key": "betway", "title": "Betway", "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": "A", "price": 1.01},
                    {"name": "B", "price": 1.01},
                    {"name": "Draw", "price": 1.01},
                ]}]},
        ],
    }
    assert odds.market_from_event(broken, target_book="betway") is None
