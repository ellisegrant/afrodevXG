"""Football analytics MCP server.

The MCP front door: it wires the data layer and the model layer together and
returns Pydantic-shaped results. Claude only ever talks to this file.

Data flow is one-way:  Claude -> server.py -> data/ -> models/ -> schemas.py
"""

from __future__ import annotations

import difflib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from data import fixtures as fixtures_api
from data import odds as odds_api
from data import picklog
from models import backtest as backtest_api
from models import dixon_coles
from schemas import (
    BacktestResult,
    Fixture,
    LoggedPick,
    MatchProbabilities,
    ModelComparison,
    PickReview,
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

    counts = dixon_coles.team_match_counts(results)
    for team in (home_resolved, away_resolved):
        if counts.get(team, 0) < 10:
            notes.append(
                f"{team} has only {counts.get(team, 0)} matches of history. Its "
                "strength is shrunk toward the league average, so this estimate "
                "reflects the league more than the team."
            )

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


_SELECTION_LABEL = {
    "home": "{home} win",
    "draw": "Draw",
    "away": "{away} win",
    "over": "Over {line} goals",
    "under": "Under {line} goals",
    "yes": "Both teams to score",
    "no": "Both teams not to score",
}


@mcp.tool()
def scan_value(
    competition_code: str = "PL",
    min_edge: float = 0.03,
    seasons_back: int = 3,
    target_book: str = odds_api.DEFAULT_TARGET_BOOK,
    bankroll: Optional[float] = None,
    min_team_matches: int = 4,
    markets: str = odds_api.DEFAULT_MARKETS,
    log_picks: bool = True,
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
        min_team_matches: Skip fixtures where either side has fewer matches of
            history than this. Shrinkage already pulls thin teams toward the
            league average, so this only excludes the very sparsest.
        markets: Odds API markets to pull. Each one costs a credit per request,
            so the default is "h2h,totals"; add "btts" for both-teams-to-score.
        log_picks: Record the picks so review_picks can score them later.
    """
    competition_code = competition_code.upper().strip()

    results = fixtures_api.get_recent_results(competition_code, seasons_back=seasons_back)
    model = dixon_coles.get_model(competition_code, results)
    known = dixon_coles.model_teams(model)

    counts = dixon_coles.team_match_counts(results)

    events = odds_api.fetch_events(competition_code, markets=markets)
    picks: list[ValuePick] = []
    sharp_books: set[str] = set()
    checked = 0
    thin: set[str] = set()

    for event in events:
        market = odds_api.market_from_event(event, target_book=target_book)
        if market is None or not market["target_odds"]:
            continue

        home = _resolve_team(market["home_team"], known)
        away = _resolve_team(market["away_team"], known)
        if home is None or away is None:
            continue

        # A side with almost no history gets a barely-constrained estimate, which
        # shows up as an enormous fake edge. Leave those fixtures out.
        thin_sides = [t for t in (home, away) if counts.get(t, 0) < min_team_matches]
        if thin_sides:
            thin.update(thin_sides)
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

        # Each market contributes its own selections off the same fit.
        market_blocks = [
            (
                model_probs,
                {
                    "fair": market["fair"],
                    "target_odds": market["target_odds"],
                    "best_odds": market["best_odds"],
                },
            )
        ]
        if market.get("totals"):
            line = market["totals"]["line"]
            if line == 2.5:
                over, under = prediction["over_2_5"], prediction["under_2_5"]
            else:
                over, under = dixon_coles.predict_totals(model, home, away, line)
            market_blocks.append(({"over": over, "under": under}, market["totals"]))
        if market.get("btts"):
            market_blocks.append(
                (
                    {"yes": prediction["btts_yes"], "no": 1.0 - prediction["btts_yes"]},
                    market["btts"],
                )
            )

        for probabilities_by_outcome, block in market_blocks:
            if not block:
                continue
            # Betway does not quote every market. Rather than drop those
            # selections, price them at the best book that does quote them.
            prices = block.get("target_odds") or {
                outcome: best["odds"] for outcome, best in block["best_odds"].items()
            }
            if not prices:
                continue

            for outcome, model_prob in probabilities_by_outcome.items():
                if outcome not in block["fair"] or outcome not in prices:
                    continue

                edge = model_prob - block["fair"][outcome]
                if edge < min_edge:
                    continue

                price = prices[outcome]
                best = block["best_odds"][outcome]
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
                            home=market["home_team"],
                            away=market["away_team"],
                            line=block.get("line", ""),
                        ),
                        outcome=outcome,
                        model_prob=model_prob,
                        fair_prob=block["fair"][outcome],
                        edge=edge,
                        price=price,
                        price_book=(
                            market["target_book"]
                            if block.get("target_odds")
                            else best["book"]
                        ),
                        best_price=best["odds"],
                        best_price_book=best["book"],
                        expected_value_pct=expected_value * 100.0,
                        kelly_fraction=max(kelly, 0.0),
                        stake=round(bankroll * max(kelly, 0.0) / 4.0, 2) if bankroll else None,
                    )
                )

    picks.sort(key=lambda pick: pick.expected_value_pct, reverse=True)

    notes = [
        "Edge is model probability minus the exchange's de-vigged probability.",
        "Expected value is what the target book's price actually returns. A pick can show a positive edge and negative expected value when that book prices it worse than the exchange - those are not bets, they are near-misses.",
        "Backtesting shows this model beats a base-rate baseline but not the "
        "market, so treat any edge as unproven rather than as a signal.",
    ]
    if bankroll:
        notes.append(
            "Stakes are quarter-Kelly and assume the model probability is correct; "
            "if the model is overconfident, Kelly overstakes badly."
        )
    if thin:
        notes.append(
            "Skipped fixtures involving "
            + ", ".join(sorted(thin))
            + f" - fewer than {min_team_matches} matches of history."
        )
    if not events:
        notes.append("The odds feed returned no upcoming fixtures for this competition.")

    if log_picks and picks:
        recorded = picklog.record(
            competition_code, [pick.model_dump() for pick in picks]
        )
        notes.append(
            f"Logged {recorded['added']} new picks "
            f"({recorded['skipped']} already recorded). Score them with review_picks."
        )

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


def _settle_outcome(match: dict) -> str:
    if match["home_goals"] > match["away_goals"]:
        return "home"
    if match["home_goals"] < match["away_goals"]:
        return "away"
    return "draw"


@mcp.tool()
def review_picks(competition_code: Optional[str] = None, refresh_odds: bool = True) -> PickReview:
    """Score every pick the scanner has logged.

    Settles fixtures that have been played, and for those still to come records
    how the market has moved since the pick was made. That movement - closing
    line value - is the fastest honest read on whether the model finds real
    value, because it needs weeks rather than the thousands of bets that
    profit-and-loss requires.

    Args:
        competition_code: Limit to one competition. Default reviews all.
        refresh_odds: Update the closing price on fixtures not yet played.
            Costs one Odds API credit per competition involved.
    """
    records = picklog.load_all()
    if competition_code:
        competition_code = competition_code.upper().strip()
        records = [r for r in records if r["competition"] == competition_code]

    now = datetime.now(timezone.utc).isoformat()
    competitions = sorted({record["competition"] for record in records})
    updates: dict[str, dict] = {}

    # Settle anything that has been played.
    results_by_competition: dict[str, dict] = {}
    for competition in competitions:
        try:
            played = fixtures_api.get_recent_results(competition, seasons_back=1)
        except fixtures_api.FootballDataError as exc:
            log.warning("cannot settle %s: %s", competition, exc)
            continue
        results_by_competition[competition] = {
            (odds_api.normalize_team(match["home"]), odds_api.normalize_team(match["away"])): match
            for match in played
        }

    # Refresh the market on fixtures still to come.
    live_events: dict[str, list] = {}
    if refresh_odds:
        for competition in competitions:
            if any(r["status"] == "open" and r["kickoff"] > now for r in records
                   if r["competition"] == competition):
                try:
                    live_events[competition] = odds_api.fetch_events(competition)
                except odds_api.OddsAPIError as exc:
                    log.warning("cannot refresh odds for %s: %s", competition, exc)

    for record in records:
        if record["status"] == "settled":
            continue

        home_name, _, away_name = record["match"].partition(" v ")
        key = (odds_api.normalize_team(home_name), odds_api.normalize_team(away_name))
        played = results_by_competition.get(record["competition"], {}).get(key)

        if played is not None:
            actual = _settle_outcome(played)
            won = actual == record["outcome"]
            updates[record["id"]] = {
                "status": "settled",
                "result": f"{played['home_goals']}-{played['away_goals']} ({actual})",
                "profit_units": round(record["price_taken"] - 1.0, 4) if won else -1.0,
            }
            continue

        events = live_events.get(record["competition"], [])
        if events and record["kickoff"] > now:
            event = odds_api._match_event(events, home_name, away_name)
            market = odds_api.market_from_event(event) if event else None
            if market:
                updates[record["id"]] = {
                    "closing_fair_prob": market["fair"][record["outcome"]],
                    "closing_best_price": market["best_odds"][record["outcome"]]["odds"],
                }

    picklog.update_many(updates)
    records = picklog.load_all()
    if competition_code:
        records = [r for r in records if r["competition"] == competition_code]

    settled = [r for r in records if r["status"] == "settled"]
    upcoming = [r for r in records if r["status"] != "settled" and r["kickoff"] > now]
    stale = [r for r in records if r["status"] != "settled" and r["kickoff"] <= now]

    review = PickReview(
        total_picks=len(records),
        settled=len(settled),
        awaiting_kickoff=len(upcoming),
        unsettled_past_kickoff=len(stale),
    )

    if settled:
        review.wins = sum(1 for r in settled if (r["profit_units"] or 0) > 0)
        review.win_rate = review.wins / len(settled)
        review.profit_units = round(sum(r["profit_units"] or 0.0 for r in settled), 3)
        review.roi_pct = review.profit_units / len(settled) * 100.0
        # Brier on the single selection: (probability - what happened) squared.
        review.model_brier = sum(
            (r["model_prob"] - (1.0 if (r["profit_units"] or 0) > 0 else 0.0)) ** 2
            for r in settled
        ) / len(settled)
        review.market_brier = sum(
            (r["fair_prob_at_pick"] - (1.0 if (r["profit_units"] or 0) > 0 else 0.0)) ** 2
            for r in settled
        ) / len(settled)

    with_clv = [r for r in records if r.get("closing_fair_prob") is not None]
    if with_clv:
        movements = [r["closing_fair_prob"] - r["fair_prob_at_pick"] for r in with_clv]
        review.avg_closing_line_value = sum(movements) / len(movements)
        review.clv_positive_rate = sum(1 for m in movements if m > 0) / len(movements)

    review.picks = [
        LoggedPick(
            **{key: record[key] for key in (
                "id", "competition", "recorded_at", "status", "match", "kickoff",
                "selection", "model_prob", "fair_prob_at_pick", "edge_at_pick",
                "price_taken", "price_book", "closing_fair_prob", "result",
                "profit_units",
            )},
            closing_line_value=(
                record["closing_fair_prob"] - record["fair_prob_at_pick"]
                if record.get("closing_fair_prob") is not None else None
            ),
        )
        for record in sorted(records, key=lambda r: r["kickoff"])
    ]

    review.notes = [
        "Brier score: lower is better. If the market's Brier beats the model's on "
        "your own picks, the model is not finding value.",
        "Closing line value is the honest early signal - positive means the market "
        "moved toward your number after you picked.",
    ]
    if len(settled) < 50:
        review.notes.append(
            f"Only {len(settled)} settled picks: far too few to draw any conclusion "
            "about profitability. Hundreds are needed."
        )
    if stale:
        review.notes.append(
            f"{len(stale)} picks are past kickoff but unsettled - the result feed may "
            "not have caught up, or the fixture names did not match."
        )
    return review


@mcp.tool()
def compare_models(
    competition_code: str = "PL",
    test_matches: int = 100,
    seasons_back: int = 3,
    models: Optional[list[str]] = None,
) -> ModelComparison:
    """Backtest every available goal model on the same matches and rank them.

    Dixon-Coles is the default because it is the standard choice, not because it
    has been shown to suit any particular league. This checks that assumption.

    Args:
        competition_code: football-data.org code (PL, PD, BL1, SA, FL1, ...).
        test_matches: Held-out matches to score each model on.
        seasons_back: Seasons of history to draw on.
        models: Subset to try. Default runs all of them, which is slow.
    """
    competition_code = competition_code.upper().strip()
    results = fixtures_api.get_recent_results(competition_code, seasons_back=seasons_back)
    report = backtest_api.compare_models(results, model_names=models, test_matches=test_matches)

    return ModelComparison(
        competition=competition_code,
        notes=[
            "Lower RPS is better. Differences under ~0.002 are noise at this sample "
            "size - prefer the simpler model when the gap is that small.",
            "A model that errors out is reported rather than hidden; some need more "
            "data than a single league season provides.",
        ],
        **report,
    )


if __name__ == "__main__":
    mcp.run()
