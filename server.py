"""Football analytics MCP server.

The MCP front door: it wires the data layer and the model layer together and
returns Pydantic-shaped results. Claude only ever talks to this file.

Data flow is one-way:  Claude -> server.py -> data/ -> models/ -> schemas.py
"""

from __future__ import annotations

import difflib
import logging
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from data import fixtures as fixtures_api
from data import odds as odds_api
from models import backtest as backtest_api
from models import dixon_coles
from schemas import (
    BacktestResult,
    Fixture,
    MatchProbabilities,
    ValuePick,
    ValueScan,
    XiTuningResult,
)

# Claude Desktop launches this server from an arbitrary working directory,
# so point dotenv at the .env sitting next to this file.
load_dotenv(Path(__file__).resolve().parent / ".env")

# stdio transport owns stdout - every log line must go to stderr.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# httpx logs full request URLs at INFO, and The Odds API takes its key as a
# query parameter - that would write the key into every log file.
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    target_book: str = odds_api.DEFAULT_TARGET_BOOK,
) -> MatchProbabilities:
    """Estimate match probabilities with a Dixon-Coles model and compare to the market.

    Args:
        home_team: Home side, e.g. "Arsenal".
        away_team: Away side, e.g. "Chelsea".
        competition_code: football-data.org code (PL, PD, BL1, SA, FL1, ...).
        seasons_back: How many seasons of results to fit on (3 is a good default).
        include_odds: Also fetch bookmaker odds and compute the value edge.
        target_book: Bookmaker whose price you could actually take (default betway).

    Returns 1X2, over/under 2.5, BTTS and expected goals. The market comparison
    uses a betting exchange as the honest baseline (exchanges carry no margin)
    and reports the target book's price separately as the one you could take.
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
            market = odds_api.fetch_market(
                home_resolved, away_resolved, competition_code, target_book=target_book
            )
        except odds_api.OddsAPIError as exc:
            market = None
            notes.append(f"odds unavailable: {exc}")

        if market:
            fair = market["fair"]
            probabilities.sharp_book = market["sharp_book"]
            probabilities.sharp_overround_pct = market["sharp_overround_pct"]
            probabilities.market_home_odds = market["sharp_odds"]["home"]
            probabilities.market_draw_odds = market["sharp_odds"]["draw"]
            probabilities.market_away_odds = market["sharp_odds"]["away"]
            probabilities.market_home_fair = fair["home"]
            probabilities.market_draw_fair = fair["draw"]
            probabilities.market_away_fair = fair["away"]
            probabilities.value_edge_home = probabilities.home_win - fair["home"]
            probabilities.value_edge_draw = probabilities.draw - fair["draw"]
            probabilities.value_edge_away = probabilities.away_win - fair["away"]

            if market["target_odds"]:
                probabilities.target_book = market["target_book"]
                probabilities.target_home_odds = market["target_odds"]["home"]
                probabilities.target_draw_odds = market["target_odds"]["draw"]
                probabilities.target_away_odds = market["target_odds"]["away"]
            else:
                notes.append(f"{target_book} is not quoting this fixture")
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


@mcp.tool()
def backtest_model(
    competition_code: str = "PL",
    test_matches: int = 100,
    seasons_back: int = 3,
    xi: float = dixon_coles.DEFAULT_XI,
) -> BacktestResult:
    """Score the model on past matches it was not trained on.

    Walk-forward: each test match is predicted using only matches played before
    it. Reports Ranked Probability Score (lower is better) against a base-rate
    baseline, plus a calibration table.

    Args:
        competition_code: football-data.org code (PL, PD, BL1, SA, FL1, ...).
        test_matches: How many of the most recent matches to score.
        seasons_back: How many seasons of history to draw on.
        xi: Time-decay constant. 0 weights every match equally.
    """
    competition_code = competition_code.upper().strip()
    results = fixtures_api.get_recent_results(competition_code, seasons_back=seasons_back)
    report = backtest_api.walk_forward(results, test_matches=test_matches, xi=xi)

    notes = [
        "RPS is Ranked Probability Score: lower is better, 0 is perfect.",
        "Baseline is the empirical home/draw/away rate over the training window.",
        "Bookmaker comparison is not included: historical odds need a paid "
        "Odds API plan, so only live fixtures can be compared to the market.",
    ]
    if report["matches_skipped"]:
        notes.append(
            f"{report['matches_skipped']} matches skipped - a team had no prior history."
        )

    return BacktestResult(
        competition=competition_code,
        notes=notes,
        **{key: report[key] for key in (
            "xi", "matches_trained_on", "matches_scored", "matches_skipped",
            "model_rps", "baseline_rps", "rps_improvement_pct", "hit_rate",
            "baseline_rates", "calibration",
        )},
    )


@mcp.tool()
def tune_time_decay(
    competition_code: str = "PL",
    test_matches: int = 100,
    seasons_back: int = 3,
) -> XiTuningResult:
    """Grid-search the time-decay constant and report which value scores best.

    Runs a full backtest per candidate, so this takes appreciably longer than a
    single prediction.
    """
    competition_code = competition_code.upper().strip()
    results = fixtures_api.get_recent_results(competition_code, seasons_back=seasons_back)
    report = backtest_api.tune_xi(results, test_matches=test_matches)

    return XiTuningResult(
        competition=competition_code,
        notes=[
            "Lower RPS is better. A win of less than ~0.002 RPS is noise at this "
            "sample size, not a reason to change the default.",
        ],
        **report,
    )


_SELECTION_LABEL = {"home": "{home} win", "draw": "Draw", "away": "{away} win"}


@mcp.tool()
def scan_value(
    competition_code: str = "PL",
    min_edge: float = 0.03,
    seasons_back: int = 3,
    target_book: str = odds_api.DEFAULT_TARGET_BOOK,
    bankroll: Optional[float] = None,
) -> ValueScan:
    """Check every upcoming fixture in a competition and list the model's disagreements.

    For each match the model probability is compared to the de-vigged price from
    a betting exchange (the honest baseline), and the price you could actually
    take is read off the target book.

    Args:
        competition_code: football-data.org code (PL, PD, BL1, SA, FL1, ...).
        min_edge: Minimum model-minus-market gap to report, in probability
            points (0.03 = three points).
        seasons_back: Seasons of history to fit the model on.
        target_book: Bookmaker whose price you would take.
        bankroll: If given, adds a quarter-Kelly stake per pick in the same units.
    """
    competition_code = competition_code.upper().strip()

    results = fixtures_api.get_recent_results(competition_code, seasons_back=seasons_back)
    model = dixon_coles.get_model(competition_code, results)
    known = dixon_coles.model_teams(model)

    events = odds_api.fetch_events(competition_code)
    picks: list[ValuePick] = []
    sharp_books: set[str] = set()
    checked = 0

    for event in events:
        market = odds_api.market_from_event(event, target_book=target_book)
        if market is None or not market["target_odds"]:
            continue

        home = _resolve_team(market["home_team"], known)
        away = _resolve_team(market["away_team"], known)
        if home is None or away is None:
            continue

        try:
            prediction = dixon_coles.predict_match(model, home, away)
        except dixon_coles.ModelError:
            continue

        checked += 1
        sharp_books.add(market["sharp_book"])
        model_probs = {
            "home": prediction["home_win"],
            "draw": prediction["draw"],
            "away": prediction["away_win"],
        }

        for outcome, model_prob in model_probs.items():
            edge = model_prob - market["fair"][outcome]
            if edge < min_edge:
                continue

            price = market["target_odds"][outcome]
            best = market["best_odds"][outcome]
            # Expected return per unit staked, if the model probability is right.
            expected_value = model_prob * price - 1.0
            # Full Kelly: edge over the price, divided by the price's net return.
            net_return = price - 1.0
            kelly = (model_prob * price - 1.0) / net_return if net_return > 0 else 0.0

            picks.append(
                ValuePick(
                    match=f"{market['home_team']} v {market['away_team']}",
                    kickoff=market["commence_time"],
                    selection=_SELECTION_LABEL[outcome].format(
                        home=market["home_team"], away=market["away_team"]
                    ),
                    outcome=outcome,
                    model_prob=model_prob,
                    fair_prob=market["fair"][outcome],
                    edge=edge,
                    price=price,
                    price_book=market["target_book"],
                    best_price=best["odds"],
                    best_price_book=best["book"],
                    expected_value_pct=expected_value * 100.0,
                    kelly_fraction=max(kelly, 0.0),
                    stake=round(bankroll * max(kelly, 0.0) / 4.0, 2) if bankroll else None,
                )
            )

    picks.sort(key=lambda pick: pick.edge, reverse=True)

    notes = [
        "Edge is model probability minus the exchange's de-vigged probability.",
        "Backtesting shows this model beats a base-rate baseline but not the "
        "market, so treat any edge as unproven rather than as a signal.",
    ]
    if bankroll:
        notes.append(
            "Stakes are quarter-Kelly and assume the model probability is correct; "
            "if the model is overconfident, Kelly overstakes badly."
        )
    if not events:
        notes.append("The odds feed returned no upcoming fixtures for this competition.")

    return ValueScan(
        competition=competition_code,
        sharp_book=", ".join(sorted(sharp_books)) or "none",
        target_book=target_book,
        fixtures_checked=checked,
        picks_found=len(picks),
        min_edge=min_edge,
        picks=picks,
        notes=notes,
    )


if __name__ == "__main__":
    mcp.run()
