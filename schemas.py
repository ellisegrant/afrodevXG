"""Return shapes for the football analytics MCP server.

This module is pure data definition: it knows nothing about HTTP, MCP or
probability maths. Every tool in `server.py` returns one of these models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MatchProbabilities(BaseModel):
    """Model probabilities for a single fixture, plus optional market comparison."""

    home_team: str
    away_team: str
    competition: str

    # 1X2
    home_win: float = Field(..., ge=0.0, le=1.0)
    draw: float = Field(..., ge=0.0, le=1.0)
    away_win: float = Field(..., ge=0.0, le=1.0)

    # Goals markets
    over_2_5: float = Field(..., ge=0.0, le=1.0)
    under_2_5: float = Field(..., ge=0.0, le=1.0)
    btts_yes: float = Field(..., ge=0.0, le=1.0)

    # Expected goals implied by the fitted attack/defence strengths
    home_xg: float = Field(..., ge=0.0)
    away_xg: float = Field(..., ge=0.0)

    # Provenance
    matches_used: int = Field(
        0, description="Number of historical matches the model was fitted on."
    )

    # Market comparison (only populated when bookmaker odds were available)
    market_home_odds: Optional[float] = None
    market_draw_odds: Optional[float] = None
    market_away_odds: Optional[float] = None
    market_home_fair: Optional[float] = Field(
        None, description="De-vigged bookmaker probability of a home win."
    )
    market_draw_fair: Optional[float] = None
    market_away_fair: Optional[float] = None
    value_edge_home: Optional[float] = Field(
        None, description="model home_win - de-vigged market home probability."
    )
    value_edge_draw: Optional[float] = None
    value_edge_away: Optional[float] = None

    notes: list[str] = Field(default_factory=list)


class Fixture(BaseModel):
    """An upcoming scheduled match."""

    home_team: str
    away_team: str
    competition: str
    utc_date: str
    matchday: Optional[int] = None
