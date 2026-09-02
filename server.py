"""Football analytics MCP server.

The MCP front door: it wires the data layer and the model layer together and
returns Pydantic-shaped results. Claude only ever talks to this file.

Data flow is one-way:  Claude -> server.py -> data/ -> models/ -> schemas.py
"""

from __future__ import annotations

import difflib
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from data import fixtures as fixtures_api
from data import odds as odds_api
from data import picklog
from data import players as players_api
from models import accumulator as accumulator_api
from models import availability
from models import backtest as backtest_api
from models import dixon_coles
from schemas import (
    AbsenceImpact,
    Accumulator,
    AccumulatorLeg,
    AccumulatorPlan,
    BacktestResult,
    Fixture,
    KeyPlayer,
    LoggedPick,
    MarketSheet,
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



def _absence_impact(
    competition_code: str,
    results: list[dict],
    team: str,
    names: list[str],
    seasons_back: int,
    replacement_level: float,
) -> Optional[AbsenceImpact]:
    """Turn a list of missing players into a scoring multiplier for their team."""
    if not names:
        return None

    shares = players_api.goal_shares(competition_code, results, seasons_back=seasons_back)
    found: list[KeyPlayer] = []
    unmatched: list[str] = []

    for name in names:
        record = players_api.find_player(shares, name, team=team)
        if record is None:
            unmatched.append(name)
            continue
        found.append(KeyPlayer(**{key: record[key] for key in (
            "player", "team", "position", "goals", "penalties",
            "open_play_goals", "team_goals", "goal_share", "played",
        )}))

    multiplier = availability.attack_multiplier(
        [player.goal_share for player in found], replacement_level=replacement_level
    )
    return AbsenceImpact(
        team=team,
        absentees=found,
        unmatched=unmatched,
        attack_multiplier=multiplier,
        replacement_level=replacement_level,
    )


@mcp.tool()
def get_match_probabilities(
    home_team: str,
    away_team: str,
    competition_code: str = "PL",
    seasons_back: int = 3,
    include_odds: bool = True,
    target_book: str = odds_api.DEFAULT_TARGET_BOOK,
    home_absentees: Optional[list[str]] = None,
    away_absentees: Optional[list[str]] = None,
    replacement_level: float = availability.DEFAULT_REPLACEMENT_LEVEL,
) -> MatchProbabilities:
    """Estimate match probabilities with a Dixon-Coles model and compare to the market.

    Args:
        home_team: Home side, e.g. "Arsenal".
        away_team: Away side, e.g. "Chelsea".
        competition_code: football-data.org code (PL, PD, BL1, SA, FL1, ...).
        seasons_back: How many seasons of results to fit on (3 is a good default).
        include_odds: Also fetch bookmaker odds and compute the value edge.
        target_book: Bookmaker whose price you could actually take (default betway).
        home_absentees: Players missing for the home side, e.g. ["Haaland"].
            Their share of the team's open-play goals is used to scale its
            expected scoring down.
        away_absentees: The same for the away side.
        replacement_level: How much of an absent player's output a stand-in is
            assumed to provide. 0.5 means half.

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

    home_impact = _absence_impact(
        competition_code, results, home_resolved, home_absentees or [],
        seasons_back, replacement_level,
    )
    away_impact = _absence_impact(
        competition_code, results, away_resolved, away_absentees or [],
        seasons_back, replacement_level,
    )

    attack_scaling = {}
    if home_impact:
        attack_scaling[home_resolved] = home_impact.attack_multiplier
    if away_impact:
        attack_scaling[away_resolved] = away_impact.attack_multiplier

    if attack_scaling:
        with availability.adjusted(model, attack=attack_scaling):
            prediction = dixon_coles.predict_match(model, home_resolved, away_resolved)
        for impact in (home_impact, away_impact):
            if impact is None:
                continue
            if impact.absentees:
                missing = ", ".join(
                    f"{p.player} ({p.goal_share:.0%} of goals)" for p in impact.absentees
                )
                notes.append(
                    f"{impact.team} without {missing}: scoring scaled to "
                    f"{impact.attack_multiplier:.2f} of normal."
                )
            if impact.unmatched:
                notes.append(
                    f"Not found in {impact.team}'s scoring data, so ignored: "
                    + ", ".join(impact.unmatched)
                )
        notes.append(
            "Absence adjustments cannot be backtested: the free data tier carries "
            "no historical lineups. Treat them as a considered adjustment, not a "
            "measured one."
        )
    else:
        prediction = dixon_coles.predict_match(model, home_resolved, away_resolved)

    probabilities = MatchProbabilities(
        home_team=home_resolved,
        away_team=away_resolved,
        competition=competition_code,
        matches_used=len(results),
        notes=notes,
        home_absence_impact=home_impact,
        away_absence_impact=away_impact,
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


def _parse_competitions(competition_code: str) -> list[str]:
    """Accept one code or several: "PL" or "PL,PD,SA,BL1"."""
    codes = [code.strip().upper() for code in competition_code.split(",") if code.strip()]
    if not codes:
        raise ValueError("no competition code given")
    return codes


def _collect_many(
    competitions: list[str],
    min_edge: float,
    seasons_back: int,
    target_book: str,
    bankroll: Optional[float],
    min_team_matches: int,
    markets: str,
) -> tuple[list[ValuePick], dict]:
    """Selections from several competitions, merged.

    Each competition costs its own odds request, and a cold run also refetches
    results, so a four-league scan is appreciably slower and dearer than one.
    """
    picks: list[ValuePick] = []
    meta = {"fixtures_checked": 0, "sharp_books": set(), "thin": set(), "events": 0}
    failures: list[str] = []

    for competition in competitions:
        try:
            found, one = _collect_selections(
                competition, min_edge, seasons_back, target_book, bankroll,
                min_team_matches, markets,
            )
        except (fixtures_api.FootballDataError, dixon_coles.ModelError) as exc:
            # One league being unavailable should not sink the whole scan.
            log.warning("skipping %s: %s", competition, exc)
            failures.append(f"{competition}: {exc}")
            continue
        for pick in found:
            pick.competition = competition
        picks.extend(found)
        meta["fixtures_checked"] += one["fixtures_checked"]
        meta["sharp_books"] |= one["sharp_books"]
        meta["thin"] |= one["thin"]
        meta["events"] += one["events"]

    meta["failures"] = failures
    return picks, meta


def _collect_selections(
    competition_code: str,
    min_edge: float,
    seasons_back: int,
    target_book: str,
    bankroll: Optional[float],
    min_team_matches: int,
    markets: str,
) -> tuple[list[ValuePick], dict]:
    """Every selection the model has a view on, with the price you could take.

    Shared by scan_value and build_accumulator: one fit, one odds pull, one set
    of candidate selections.
    """
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
        if market.get("spreads"):
            line = market["spreads"]["line"]
            home_cover, away_cover = dixon_coles.predict_handicap(model, home, away, line)
            block = dict(market["spreads"])
            # Relabel the two legs so they cannot be confused with 1X2.
            for field in ("fair", "target_odds", "best_odds"):
                if block.get(field):
                    block[field] = {
                        f"{side}_hcp": value for side, value in block[field].items()
                    }
            market_blocks.append(
                ({"home_hcp": home_cover, "away_hcp": away_cover}, block)
            )

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
                            line=block.get("line") if block.get("line") is not None else "",
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

    return picks, {
        "fixtures_checked": checked,
        "sharp_books": sharp_books,
        "thin": thin,
        "events": len(events),
    }


_SELECTION_LABEL = {
    "home": "{home} win",
    "draw": "Draw",
    "away": "{away} win",
    "over": "Over {line} goals",
    "under": "Under {line} goals",
    "home_hcp": "{home} {line:+g} handicap",
    "away_hcp": "{away} {line:+g} handicap",
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
        competition_code: One code or several, comma separated: "PL" or
            "PL,PD,SA,BL1". Each extra competition costs its own odds request.
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
    competitions = _parse_competitions(competition_code)
    picks, meta = _collect_many(
        competitions, min_edge, seasons_back, target_book, bankroll,
        min_team_matches, markets,
    )
    checked, sharp_books, thin, events = (
        meta["fixtures_checked"], meta["sharp_books"], meta["thin"], meta["events"]
    )
    competition_code = ",".join(competitions)

    picks.sort(key=lambda pick: pick.expected_value_pct, reverse=True)

    notes = [
        "Edge is model probability minus the exchange's de-vigged probability.",
        "Expected value is what the target book's price actually returns. A pick can show a positive edge and negative expected value when that book prices it worse than the exchange - those are not bets, they are near-misses.",
        "Backtesting shows this model beats a base-rate baseline but not the "
        "market, so treat any edge as unproven rather than as a signal.",
    ]
    if any(pick.outcome.endswith("_hcp") and float(pick.selection.split()[-2]) % 1 == 0
           for pick in picks):
        notes.append(
            "Whole-number handicaps can push - if the match lands exactly on the "
            "line your stake is returned. The expected value shown treats a push "
            "as a loss, so it understates those picks."
        )
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
    if meta.get("failures"):
        notes.append("Skipped competitions - " + " | ".join(meta["failures"]))
    if len(competitions) > 1:
        notes.append(
            f"Scanned {len(competitions)} competitions. Each one costs its own odds "
            "request, so a wide scan spends credits faster than a single league."
        )
    if not events:
        notes.append("The odds feed returned no upcoming fixtures for this competition.")

    if log_picks and picks:
        added = skipped = 0
        for competition in competitions:
            batch = [
                pick.model_dump() for pick in picks
                if getattr(pick, "competition", competition) == competition
            ]
            if not batch:
                continue
            outcome = picklog.record(competition, batch)
            added += outcome["added"]
            skipped += outcome["skipped"]
        recorded = {"added": added, "skipped": skipped}
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


@mcp.tool()
def get_match_markets(
    home_team: str,
    away_team: str,
    competition_code: str = "PL",
    seasons_back: int = 3,
) -> MarketSheet:
    """Model probabilities for every market the score grid can price.

    Beyond 1X2 and over/under this covers double chance, draw no bet, correct
    score, clean sheets, win to nil, team totals and Asian handicaps. The free
    odds feed carries prices for only three of those, so the rest come back as
    probabilities to compare by eye against your own bookmaker's odds.

    Args:
        home_team: Home side.
        away_team: Away side.
        competition_code: football-data.org code (PL, PD, BL1, SA, FL1, ...).
        seasons_back: Seasons of history to fit on.
    """
    competition_code = competition_code.upper().strip()
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

    sheet = dixon_coles.market_sheet(model, home_resolved, away_resolved)
    notes = [
        "Only 1X2, totals and handicaps can be checked against the market on the "
        "free Odds API plan. Everything else here is the model's view alone.",
        "Compare a probability to a price by dividing: a 40% chance is fair at "
        "decimal odds of 2.50 (1 / 0.40).",
    ]
    counts = dixon_coles.team_match_counts(results)
    for team in (home_resolved, away_resolved):
        if counts.get(team, 0) < 10:
            notes.append(
                f"{team} has only {counts.get(team, 0)} matches of history, so its "
                "strength is shrunk toward the league average."
            )

    return MarketSheet(
        competition=competition_code,
        matches_used=len(results),
        notes=notes,
        **sheet,
    )


@mcp.tool()
def build_accumulator(
    target_odds: float = 4.0,
    competition_code: str = "PL",
    days_ahead: float = 3.0,
    tolerance_pct: float = 10.0,
    min_legs: int = 2,
    max_legs: int = 6,
    objective: str = "probability",
    max_results: int = 5,
    seasons_back: int = 3,
    target_book: str = odds_api.DEFAULT_TARGET_BOOK,
    markets: str = odds_api.DEFAULT_MARKETS,
) -> AccumulatorPlan:
    """Build multi-leg bets whose combined odds land near a target price.

    Answers "give me a 4.0 on this weekend's games" by searching every selection
    the model has a view on - match result, over/under, handicap - for
    combinations that multiply out to roughly the requested price, then ranking
    them by how likely the model thinks they all are.

    Only one leg per fixture is ever used: two selections from the same match are
    correlated, and multiplying their probabilities would overstate the chance of
    the bet landing.

    Args:
        target_odds: Combined decimal odds you want, e.g. 4.0.
        competition_code: One code or several, comma separated: "PL" or
            "PL,PD,SA,BL1". More leagues means more fixtures to choose from and
            combinations that land closer to the target.
        days_ahead: Only include fixtures kicking off within this many days.
        tolerance_pct: How far from the target is acceptable, in percent.
        min_legs / max_legs: Bounds on the number of selections.
        objective: "probability" picks the combination most likely to land;
            "value" picks the one with the best expected return.
        max_results: How many alternative accumulators to return.
        seasons_back: Seasons of history to fit the model on.
        target_book: Bookmaker whose price you would take.
        markets: Odds API markets to pull.
    """
    competitions = _parse_competitions(competition_code)
    if objective not in ("probability", "value"):
        raise ValueError('objective must be "probability" or "value"')

    picks, meta = _collect_many(
        competitions,
        min_edge=-1.0,  # every selection is a candidate, not just the value ones
        seasons_back=seasons_back,
        target_book=target_book,
        bankroll=None,
        min_team_matches=4,
        markets=markets,
    )

    cutoff = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    candidates = [
        {
            "fixture": pick.match,
            "match": pick.match,
            "kickoff": pick.kickoff,
            "selection": pick.selection,
            "price": pick.price,
            "price_book": pick.price_book,
            "model_prob": pick.model_prob,
            "fair_prob": pick.fair_prob,
            "edge": pick.edge,
        }
        for pick in picks
        if now <= pick.kickoff <= cutoff
    ]

    built = accumulator_api.build(
        candidates,
        target_odds=target_odds,
        tolerance_pct=tolerance_pct,
        min_legs=min_legs,
        max_legs=max_legs,
        objective=objective,
        max_results=max_results,
    )

    accumulators = [
        Accumulator(
            legs=[
                AccumulatorLeg(
                    match=leg["match"],
                    kickoff=leg["kickoff"],
                    selection=leg["selection"],
                    price=leg["price"],
                    price_book=leg["price_book"],
                    model_prob=leg["model_prob"],
                    fair_prob=leg["fair_prob"],
                    edge=leg["edge"],
                )
                for leg in entry["legs"]
            ],
            leg_count=len(entry["legs"]),
            combined_odds=round(entry["combined_odds"], 3),
            model_success_pct=entry["model_success"] * 100.0,
            market_success_pct=entry["market_success"] * 100.0,
            fair_odds=round(entry["fair_odds"], 3),
            expected_value_pct=entry["expected_value_pct"],
        )
        for entry in built
    ]

    notes = []
    if meta.get("failures"):
        notes.append("Skipped competitions - " + " | ".join(meta["failures"]))
    notes += [
        "Model success is the chance every leg lands, on the model's numbers. It "
        "assumes the legs are independent - they are from different matches, but "
        "conditions common to a matchday still correlate them a little.",
        "Compare model success to market success: if the market rates the "
        "combination higher than the model does, the model is not finding value "
        "here, it is just finding long odds.",
    ]
    if accumulators:
        typical_legs = accumulators[0].leg_count
        margin = accumulator_api.margin_compounding(typical_legs)
        notes.append(
            f"A {typical_legs}-leg bet carries roughly {margin:.0f}% of bookmaker "
            "margin, because each leg's cut multiplies. This is why accumulators "
            "are usually worse value than the same selections bet singly."
        )
    else:
        notes.append(
            "No combination landed within tolerance. Widen tolerance_pct, raise "
            "max_legs, or extend days_ahead."
        )

    return AccumulatorPlan(
        competition=",".join(competitions),
        target_odds=target_odds,
        tolerance_pct=tolerance_pct,
        objective=objective,
        fixtures_available=meta["fixtures_checked"],
        selections_available=len(candidates),
        accumulators=accumulators,
        margin_warning_pct=(
            accumulator_api.margin_compounding(accumulators[0].leg_count)
            if accumulators else None
        ),
        notes=notes,
    )


@mcp.tool()
def list_key_players(
    team: str, competition_code: str = "PL", seasons_back: int = 3, top: int = 6
) -> list[KeyPlayer]:
    """Which players carry a team's scoring, and by how much.

    Goal share excludes penalties, since penalty duty transfers to whoever is on
    the pitch. Use these names with get_match_probabilities' absentee arguments.
    """
    competition_code = competition_code.upper().strip()
    results = fixtures_api.get_recent_results(competition_code, seasons_back=seasons_back)
    model = dixon_coles.get_model(competition_code, results)

    resolved = _resolve_team(team, dixon_coles.model_teams(model))
    if resolved is None:
        raise ValueError(f"{team!r} is not a team in {competition_code}.")

    return [
        KeyPlayer(**{key: record[key] for key in (
            "player", "team", "position", "goals", "penalties",
            "open_play_goals", "team_goals", "goal_share", "played",
        )})
        for record in players_api.team_key_players(
            competition_code, results, resolved, seasons_back=seasons_back, top=top
        )
    ]


if __name__ == "__main__":
    mcp.run()
