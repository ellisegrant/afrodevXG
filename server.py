"""Football analytics MCP server.

The MCP front door: it wires the data layer and the model layer together and
returns Pydantic-shaped results. Claude only ever talks to this file.

Data flow is one-way:  Claude -> server.py -> data/ -> models/ -> schemas.py
"""

from __future__ import annotations

import difflib
import logging
import sys
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from data import fixtures as fixtures_api
from data import odds as odds_api
from models import dixon_coles
from schemas import Fixture, MatchProbabilities

load_dotenv()

# stdio transport owns stdout - every log line must go to stderr.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("football-mcp")

mcp = FastMCP("football-analytics")


def _resolve_team(name: str, known: list[str]) -> Optional[str]:
    """Map a loosely typed team name onto the spelling the model was fitted on."""
    if not known:
        return name
    if name in known:
        return name

    target = odds_api.normalize_team(name)
    normalized = {team: odds_api.normalize_team(team) for team in known}

    for team, norm in normalized.items():
        if norm == target:
            return team

    best, best_score = None, 0.0
    for team, norm in normalized.items():
        score = difflib.SequenceMatcher(None, target, norm).ratio()
        if target and (target in norm or norm in target):
            score = max(score, 0.95)
        if score > best_score:
            best, best_score = team, score
    return best if best_score >= 0.7 else None


@mcp.tool()
def get_match_probabilities(
    home_team: str,
    away_team: str,
    competition_code: str = "PL",
    seasons_back: int = 3,
    include_odds: bool = True,
) -> MatchProbabilities:
    """Estimate match probabilities with a Dixon-Coles model and compare to the market.

    Args:
        home_team: Home side, e.g. "Arsenal".
        away_team: Away side, e.g. "Chelsea".
        competition_code: football-data.org code (PL, PD, BL1, SA, FL1, ...).
        seasons_back: How many seasons of results to fit on (3 is a good default).
        include_odds: Also fetch bookmaker odds and compute the value edge.

    Returns 1X2, over/under 2.5, BTTS and expected goals, plus the de-vigged
    market probabilities and model-minus-market edge when odds are available.
    """
    competition_code = competition_code.upper().strip()
    notes: list[str] = []

    results = fixtures_api.get_recent_results(competition_code, seasons_back=seasons_back)
    model = dixon_coles.get_model(competition_code, results)

    known = dixon_coles.model_teams(model)
    home_resolved = _resolve_team(home_team, known)
    away_resolved = _resolve_team(away_team, known)
    if home_resolved is None or away_resolved is None:
        unknown = home_team if home_resolved is None else away_team
        raise ValueError(
            f"{unknown!r} is not a team in {competition_code}. "
            f"Known teams: {', '.join(known)}"
        )
    if home_resolved != home_team:
        notes.append(f"matched {home_team!r} to {home_resolved!r}")
    if away_resolved != away_team:
        notes.append(f"matched {away_team!r} to {away_resolved!r}")

    prediction = dixon_coles.predict_match(model, home_resolved, away_resolved)

    probabilities = MatchProbabilities(
        home_team=home_resolved,
        away_team=away_resolved,
        competition=competition_code,
        matches_used=len(results),
        notes=notes,
        **prediction,
    )

    if include_odds:
        try:
            market = odds_api.fetch_odds(home_resolved, away_resolved, competition_code)
        except odds_api.OddsAPIError as exc:
            market = None
            notes.append(f"odds unavailable: {exc}")

        if market:
            fair = odds_api.devig(market)
            probabilities.market_home_odds = market["home"]
            probabilities.market_draw_odds = market["draw"]
            probabilities.market_away_odds = market["away"]
            probabilities.market_home_fair = fair["home"]
            probabilities.market_draw_fair = fair["draw"]
            probabilities.market_away_fair = fair["away"]
            probabilities.value_edge_home = probabilities.home_win - fair["home"]
            probabilities.value_edge_draw = probabilities.draw - fair["draw"]
            probabilities.value_edge_away = probabilities.away_win - fair["away"]
        else:
            notes.append("no bookmaker odds matched this fixture")

    probabilities.notes = notes
    return probabilities


@mcp.tool()
def list_competitions() -> dict[str, str]:
    """Competition codes this server understands, mapped to their names."""
    return fixtures_api.COMPETITIONS


@mcp.tool()
def list_teams(competition_code: str = "PL") -> list[str]:
    """Team names as the results feed spells them - use these for exact lookups."""
    return fixtures_api.list_teams(competition_code)


@mcp.tool()
def get_upcoming_fixtures(
    competition_code: str = "PL", days_ahead: int = 14
) -> list[Fixture]:
    """Scheduled matches in a competition over the next `days_ahead` days."""
    return [
        Fixture(
            home_team=fixture["home"],
            away_team=fixture["away"],
            competition=fixture["competition"],
            utc_date=fixture["utc_date"],
            matchday=fixture["matchday"],
        )
        for fixture in fixtures_api.get_upcoming_fixtures(
            competition_code, days_ahead=days_ahead
        )
    ]


if __name__ == "__main__":
    mcp.run()
