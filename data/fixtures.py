"""football-data.org client: results and fixtures.

Knows nothing about probability or MCP - it fetches JSON and hands back clean
dicts. Free tier is ~10 requests/minute across 12 major competitions, so every
call is rate limited and disk cached.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from . import _cache

log = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# Free-tier competitions, for reference / validation of the code the caller passes.
COMPETITIONS = {
    "PL": "Premier League (England)",
    "PD": "La Liga (Spain)",
    "BL1": "Bundesliga (Germany)",
    "SA": "Serie A (Italy)",
    "FL1": "Ligue 1 (France)",
    "DED": "Eredivisie (Netherlands)",
    "PPL": "Primeira Liga (Portugal)",
    "ELC": "Championship (England)",
    "BSA": "Serie A (Brazil)",
    "CL": "UEFA Champions League",
    "EC": "European Championship",
    "WC": "FIFA World Cup",
}

# Completed seasons never change; the current season does.
_FINISHED_SEASON_TTL = 30 * 24 * 3600.0
_CURRENT_SEASON_TTL = 6 * 3600.0
_FIXTURES_TTL = 3 * 3600.0

# Free tier allows 10 requests per minute.
_RATE_LIMIT = 10
_RATE_WINDOW = 60.0
_request_times: list[float] = []


class FootballDataError(RuntimeError):
    """Raised when football-data.org cannot be reached or refuses a request."""


def _api_key() -> str:
    key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not key:
        raise FootballDataError(
            "FOOTBALL_DATA_API_KEY is not set. Copy .env.example to .env and add "
            "a key from https://www.football-data.org/client/register"
        )
    return key


def _throttle() -> None:
    """Block until another request fits inside the 10-per-minute allowance."""
    now = time.monotonic()
    _request_times[:] = [t for t in _request_times if now - t < _RATE_WINDOW]
    if len(_request_times) >= _RATE_LIMIT:
        wait = _RATE_WINDOW - (now - _request_times[0]) + 0.5
        log.info("football-data rate limit reached, sleeping %.1fs", wait)
        time.sleep(max(wait, 0.0))
        now = time.monotonic()
        _request_times[:] = [t for t in _request_times if now - t < _RATE_WINDOW]
    _request_times.append(time.monotonic())


def _get(path: str, params: dict[str, Any], cache_key: str, ttl: float) -> dict:
    """GET a football-data.org endpoint, with disk cache and rate limiting."""
    cached = _cache.load(cache_key, ttl)
    if cached is not None:
        return cached

    _throttle()
    log.info("football-data GET %s %s", path, params)
    try:
        response = httpx.get(
            f"{BASE_URL}{path}",
            params=params,
            headers={"X-Auth-Token": _api_key()},
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise FootballDataError(f"request to football-data.org failed: {exc}") from exc

    if response.status_code == 429:
        # Ran into the limit anyway (shared key, other clients): back off once.
        log.warning("football-data returned 429, backing off 60s")
        time.sleep(60)
        return _get(path, params, cache_key, ttl)
    if response.status_code in (403, 404):
        raise FootballDataError(
            f"football-data.org refused {path} {params} "
            f"({response.status_code}): {response.text[:200]}"
        )
    response.raise_for_status()

    payload = response.json()
    _cache.save(cache_key, payload)
    return payload


def current_season_start(today: Optional[datetime] = None) -> int:
    """European seasons are labelled by their starting year (2024/25 -> 2024)."""
    today = today or datetime.now(timezone.utc)
    return today.year if today.month >= 7 else today.year - 1


def _team_name(team: dict) -> str:
    return team.get("shortName") or team.get("name") or team.get("tla") or "Unknown"


def _parse_matches(payload: dict) -> list[dict]:
    results = []
    for match in payload.get("matches", []):
        full_time = (match.get("score") or {}).get("fullTime") or {}
        home_goals, away_goals = full_time.get("home"), full_time.get("away")
        if home_goals is None or away_goals is None:
            continue
        results.append(
            {
                "home": _team_name(match.get("homeTeam") or {}),
                "away": _team_name(match.get("awayTeam") or {}),
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                # Not part of the minimum contract, but the Dixon-Coles time
                # decay needs a date per match.
                "date": (match.get("utcDate") or "")[:10],
            }
        )
    return results


def get_recent_results(competition_code: str, seasons_back: int = 3) -> list[dict]:
    """Finished matches for a competition, most recent `seasons_back` seasons.

    Returns dicts of {home, away, home_goals, away_goals, date}, oldest first.
    Seasons the free tier will not serve are skipped with a warning rather than
    failing the whole call.
    """
    competition_code = competition_code.upper().strip()
    latest = current_season_start()
    seasons = [latest - offset for offset in range(seasons_back)]

    results: list[dict] = []
    errors: list[str] = []
    for season in sorted(seasons):
        ttl = _CURRENT_SEASON_TTL if season == latest else _FINISHED_SEASON_TTL
        try:
            payload = _get(
                f"/competitions/{competition_code}/matches",
                {"status": "FINISHED", "season": season},
                cache_key=f"fd_results_{competition_code}_{season}",
                ttl=ttl,
            )
        except FootballDataError as exc:
            log.warning("skipping %s season %s: %s", competition_code, season, exc)
            errors.append(f"{season}: {exc}")
            continue
        season_results = _parse_matches(payload)
        log.info("%s %s: %d finished matches", competition_code, season, len(season_results))
        results.extend(season_results)

    if not results:
        raise FootballDataError(
            f"no finished matches available for {competition_code}. "
            + (" | ".join(errors) if errors else "Check the competition code.")
        )

    results.sort(key=lambda match: match["date"])
    return results



def get_season_results(competition_code: str, season: int) -> list[dict]:
    """Finished matches for one specific season."""
    competition_code = competition_code.upper().strip()
    ttl = _CURRENT_SEASON_TTL if season == current_season_start() else _FINISHED_SEASON_TTL
    payload = _get(
        f"/competitions/{competition_code}/matches",
        {"status": "FINISHED", "season": season},
        cache_key=f"fd_results_{competition_code}_{season}",
        ttl=ttl,
    )
    results = _parse_matches(payload)
    results.sort(key=lambda match: match["date"])
    return results


def get_upcoming_fixtures(
    competition_code: str, days_ahead: int = 14
) -> list[dict]:
    """Scheduled matches for a competition over the next `days_ahead` days."""
    competition_code = competition_code.upper().strip()
    today = datetime.now(timezone.utc).date()
    date_to = today + timedelta(days=days_ahead)

    payload = _get(
        f"/competitions/{competition_code}/matches",
        {
            "status": "SCHEDULED",
            "dateFrom": today.isoformat(),
            "dateTo": date_to.isoformat(),
        },
        cache_key=f"fd_fixtures_{competition_code}_{today.isoformat()}_{days_ahead}",
        ttl=_FIXTURES_TTL,
    )

    fixtures = []
    for match in payload.get("matches", []):
        fixtures.append(
            {
                "home": _team_name(match.get("homeTeam") or {}),
                "away": _team_name(match.get("awayTeam") or {}),
                "utc_date": match.get("utcDate", ""),
                "matchday": match.get("matchday"),
                "competition": competition_code,
            }
        )
    fixtures.sort(key=lambda fixture: fixture["utc_date"])
    return fixtures


def list_teams(competition_code: str) -> list[str]:
    """Team names as this API spells them - useful for name matching."""
    results = get_recent_results(competition_code, seasons_back=1)
    return sorted({match["home"] for match in results} | {match["away"] for match in results})
