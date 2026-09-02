"""Player lookup by loose name."""

from data import players


def _shares():
    return {
        "Erling Haaland": {
            "player": "Erling Haaland", "team": "Man City", "goal_share": 0.29,
        },
        "Bukayo Saka": {"player": "Bukayo Saka", "team": "Arsenal", "goal_share": 0.14},
        "Phil Foden": {"player": "Phil Foden", "team": "Man City", "goal_share": 0.09},
    }


def test_surname_alone_finds_the_player():
    assert players.find_player(_shares(), "Haaland")["player"] == "Erling Haaland"


def test_full_name_finds_the_player():
    assert players.find_player(_shares(), "Erling Haaland")["player"] == "Erling Haaland"


def test_team_restricts_the_search():
    assert players.find_player(_shares(), "Saka", team="Man City") is None
    assert players.find_player(_shares(), "Saka", team="Arsenal")["player"] == "Bukayo Saka"


def test_unknown_name_returns_nothing():
    assert players.find_player(_shares(), "Nonexistent Person") is None
