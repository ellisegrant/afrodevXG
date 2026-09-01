"""Shared fixtures. Every test here runs offline - no API keys, no network."""

import datetime
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def synthetic_results():
    """A fake league where team strength is known, so the model can be checked."""
    random.seed(7)
    teams = [f"Team {letter}" for letter in "ABCDEFGHIJ"]
    strength = {team: 0.8 + 0.15 * index for index, team in enumerate(teams)}
    start = datetime.date(2024, 8, 1)

    results, day = [], 0
    for _round in range(8):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                day += 1
                lambda_home = strength[home] * 1.35 / strength[away]
                lambda_away = strength[away] * 1.0 / strength[home]
                results.append(
                    {
                        "home": home,
                        "away": away,
                        "home_goals": sum(random.random() < lambda_home / 4 for _ in range(4)),
                        "away_goals": sum(random.random() < lambda_away / 4 for _ in range(4)),
                        "date": (start + datetime.timedelta(days=day // 10)).isoformat(),
                    }
                )
    return results


@pytest.fixture
def odds_event():
    """One Odds API event with 1X2 from three books and a totals line."""
    return {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "commence_time": "2030-01-01T15:00:00Z",
        "bookmakers": [
            {
                "key": "betway",
                "title": "Betway",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Arsenal", "price": 1.90},
                        {"name": "Chelsea", "price": 4.00},
                        {"name": "Draw", "price": 3.50},
                    ]},
                ],
            },
            {
                "key": "betfair_ex_uk",
                "title": "Betfair",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Arsenal", "price": 2.00},
                        {"name": "Chelsea", "price": 4.40},
                        {"name": "Draw", "price": 3.70},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": 1.95, "point": 2.5},
                        {"name": "Under", "price": 1.95, "point": 2.5},
                    ]},
                ],
            },
            {
                "key": "williamhill",
                "title": "William Hill",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Arsenal", "price": 1.85},
                        {"name": "Chelsea", "price": 3.90},
                        {"name": "Draw", "price": 3.40},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "price": 1.80, "point": 2.5},
                        {"name": "Under", "price": 2.05, "point": 2.5},
                    ]},
                ],
            },
        ],
    }
