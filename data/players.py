"""Player scoring records from football-data.org.

Used to work out how much of a team's goal output a given player accounts for,
so that a known absence can be reflected in the model. The free tier carries top
scorers per season but no injury feed and no lineups, so who is missing has to
be supplied by the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from . import _cache
from .fixtures import (
    _CURRENT_SEASON_TTL,
    _FINISHED_SEASON_TTL,
    FootballDataError,
    _get,
    current_season_start,
)

log = logging.getLogger(__name__)

# The API caps this; 100 covers every player who matters to a goal share.
SCORERS_LIMIT = 100


def get_scorers(competition_code: str, season: int) -> list[dict[str, Any]]:
    """Top scorers for one season, as {player, team, goals, penalties, played}."""
    competition_code = competition_code.upper().strip()
    ttl = (
        _CURRENT_SEASON_TTL
        if season == current_season_start()
        else _FINISHED_SEASON_TTL
    )
    payload = _get(
        f"/competitions/{competition_code}/scorers",
        {"limit": SCORERS_LIMIT, "season": season},
        cache_key=f"fd_scorers_{competition_code}_{season}",
        ttl=ttl,
    )

    scorers = []
    for entry in payload.get("scorers", []):
        player = entry.get("player") or {}
        team = entry.get("team") or {}
        scorers.append(
            {
                "player": player.get("name", "Unknown"),
                "player_id": player.get("id"),
                "position": player.get("position"),
                "team": team.get("shortName") or team.get("name") or "Unknown",
                "goals": int(entry.get("goals") or 0),
                "penalties": int(entry.get("penalties") or 0),
                "played": int(entry.get("playedMatches") or 0),
            }
        )
    return scorers


def goal_shares(
    competition_code: str, results: list[dict], seasons_back: int = 3
) -> dict[str, dict[str, Any]]:
    """How much of each team's scoring each player accounts for.

    Penalties are excluded: penalty duty transfers to whoever is on the pitch, so
    a missing player does not take those goals with them. Shares are computed
    over the same seasons the model was fitted on.
    """
    latest = current_season_start()
    seasons = [latest - offset for offset in range(seasons_back)]

    totals: dict[str, dict[str, Any]] = {}
    for season in sorted(seasons):
        try:
            scorers = get_scorers(competition_code, season)
        except FootballDataError as exc:
            log.warning("no scorers for %s %s: %s", competition_code, season, exc)
            continue
        for entry in scorers:
            key = entry["player"]
            record = totals.setdefault(
                key,
                {"player": key, "team": entry["team"], "goals": 0,
                 "penalties": 0, "played": 0, "position": entry["position"]},
            )
            record["goals"] += entry["goals"]
            record["penalties"] += entry["penalties"]
            record["played"] += entry["played"]
            record["team"] = entry["team"]  # most recent club wins

    team_goals: dict[str, int] = {}
    for match in results:
        team_goals[match["home"]] = team_goals.get(match["home"], 0) + match["home_goals"]
        team_goals[match["away"]] = team_goals.get(match["away"], 0) + match["away_goals"]

    for record in totals.values():
        open_play = max(record["goals"] - record["penalties"], 0)
        scored_by_team = team_goals.get(record["team"], 0)
        record["open_play_goals"] = open_play
        record["team_goals"] = scored_by_team
        record["goal_share"] = open_play / scored_by_team if scored_by_team else 0.0

    return totals


def team_key_players(
    competition_code: str, results: list[dict], team: str, seasons_back: int = 3, top: int = 6
) -> list[dict[str, Any]]:
    """The players carrying a team's scoring, biggest share first."""
    shares = goal_shares(competition_code, results, seasons_back=seasons_back)
    players = [record for record in shares.values() if record["team"] == team]
    players.sort(key=lambda record: -record["goal_share"])
    return players[:top]


def find_player(
    shares: dict[str, dict[str, Any]], name: str, team: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Look a player up by loose name, optionally restricted to one team."""
    from .odds import normalize_team  # same normalization works for people

    target = normalize_team(name)
    best, best_score = None, 0.0
    for record in shares.values():
        if team and record["team"] != team:
            continue
        candidate = normalize_team(record["player"])
        if candidate == target:
            return record
        # Surname-only lookups are the common case ("Haaland").
        score = 1.0 if target and target in candidate.split() else 0.0
        if score > best_score:
            best, best_score = record, score
    return best
