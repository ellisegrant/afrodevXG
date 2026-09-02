"""Bring second-division form into a top-flight model.

A newly promoted side has no top-flight history, so the model cannot price it -
roughly a quarter of an opening-weekend card. The fix is to fit both divisions
at once rather than to patch the promoted teams afterwards.

That works because the divisions are not disjoint: over a few seasons, teams go
up and come down, and those shared teams put both leagues on one scale. A club
that beat Championship opponents by the same margin that a relegated
Premier League side did lands in the same place. Nothing has to be assumed about
how big the gap between divisions is - the overlap measures it.

Patching a promoted team's coefficients directly cannot work, because a team
absent from the fit has no coefficients to patch.

**Measured result: this does not work, and is off by default.** Fitting
2023/24 and 2024/25 of both divisions and predicting all of 2025/26 makes every
fixture priceable - 380 of 380 against 306 - but the 74 newly priceable matches
score 0.269 RPS against a 0.214 base-rate baseline. Predictions for promoted
teams are worse than guessing, at every second-tier weight tried. Fixtures
involving already-known teams are unaffected either way.

The likely reason is that promotion is a genuine discontinuity: a squad that has
just gone up is rebuilt over the summer, and its Championship results describe a
different team. The code stays because it is the right shape if more divisions
or more seasons become available, but the honest conclusion today is the one the
scanner already implements - leave promoted teams alone until they have played.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# The only tier pairing football-data.org serves on the free plan.
SECOND_TIER = {"PL": "ELC"}

# Second-division matches carry less information about top-flight ability, so
# they count for less in the fit. 1.0 would treat the divisions as equivalent.
DEFAULT_SECOND_TIER_WEIGHT = 0.7


def merge_divisions(
    top_results: list[dict],
    second_results: list[dict],
    second_weight: float = DEFAULT_SECOND_TIER_WEIGHT,
) -> list[dict]:
    """One training set spanning both divisions, oldest match first."""
    merged = [dict(match, tier=1) for match in top_results]
    merged += [
        dict(match, tier=2, weight=second_weight) for match in second_results
    ]
    merged.sort(key=lambda match: match.get("date", ""))
    return merged


def bridge_teams(top_results: list[dict], second_results: list[dict]) -> set[str]:
    """Teams appearing in both divisions - the ones that set the exchange rate."""
    def teams(results: list[dict]) -> set[str]:
        return {match["home"] for match in results} | {match["away"] for match in results}

    return teams(top_results) & teams(second_results)


def coverage(
    top_results: list[dict], second_results: list[dict], fixtures_teams: set[str]
) -> dict[str, Any]:
    """How much of an upcoming card each division accounts for."""
    def teams(results: list[dict]) -> set[str]:
        return {match["home"] for match in results} | {match["away"] for match in results}

    top_teams, second_teams = teams(top_results), teams(second_results)
    return {
        "priceable_from_top_flight": sorted(fixtures_teams & top_teams),
        "priceable_only_via_second_tier": sorted(
            (fixtures_teams & second_teams) - top_teams
        ),
        "unpriceable": sorted(fixtures_teams - top_teams - second_teams),
        "bridge_teams": sorted(bridge_teams(top_results, second_results)),
    }
