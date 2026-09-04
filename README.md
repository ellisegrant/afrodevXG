# afrodevXG

Football match probabilities as an MCP server. A personal, local tool that fits a
Dixon-Coles model to recent results and compares its estimates to real bookmaker
odds to spot value. Built for use from Claude Desktop.

Personal, non-commercial probability analysis. No scraping — all data comes
from documented APIs.

## Architecture

One-way data flow, each layer with a single job:

```
Claude → server.py → data/ (fetch) → models/ (compute) → schemas.py (shape) → Claude
```

| File | Job |
| --- | --- |
| `server.py` | MCP front door. Thin: defines `@mcp.tool()` functions and wires layers. |
| `data/fixtures.py` | football-data.org results/fixtures. Knows nothing about probability. |
| `data/odds.py` | The Odds API 1X2 odds + `devig()`. |
| `data/_cache.py` | JSON disk cache under `cache/` (both APIs are rate limited). |
| `models/dixon_coles.py` | The brain: fit (slow, cached per league) and predict (fast). |
| `schemas.py` | `MatchProbabilities` / `Fixture` — the return shapes. |

Lower layers never call upward. API keys always come from environment
variables, never hard-coded.

## Setup

Anyone can run this — it's a local server, so you bring your own free API
keys and nothing is shared between users.

```bash
git clone https://github.com/ellisegrant/afrodevXG.git
cd afrodevXG
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                # then paste your own keys
```

Keys:

- `FOOTBALL_DATA_API_KEY` — https://www.football-data.org/client/register
  (free: ~10 req/min, 12 major competitions)
- `ODDS_API_KEY` — https://the-odds-api.com/ (free: 500 credits/month)
- `ODDS_REGIONS` — optional, defaults to `uk`. Each extra region costs another
  credit per request.

Test locally with the MCP Inspector:

```bash
mcp dev server.py
```

## Tools

- `get_match_probabilities(home_team, away_team, competition_code="PL", seasons_back=3, include_odds=True, home_absentees=[], away_absentees=[])`
  — 1X2, over/under 2.5, BTTS, expected goals, plus de-vigged market
  probabilities and the model-minus-market edge.
- `list_competitions()` — competition codes (PL, PD, BL1, SA, FL1, …).
- `list_teams(competition_code)` — team names as the results feed spells them.
- `get_upcoming_fixtures(competition_code, days_ahead=14)` — scheduled matches.
- `build_accumulator(target_odds=4.0, competition_code, days_ahead=3, objective)`
  — "give me a 4.0 on this weekend's games": searches every selection across
  every market for combinations that multiply out to the requested price,
  ranked by the model's chance that all legs land. One leg per fixture.
- `list_key_players(team, competition_code)` — who carries a team's scoring,
  as a share of its open-play goals.
- `get_match_markets(home_team, away_team, competition_code)` — the full model
  probability sheet: 1X2, double chance, draw no bet, over/under from 0.5 to
  4.5, BTTS, clean sheets, win to nil, team totals, Asian handicaps, correct
  score and expected points.
- `scan_value(competition_code, min_edge=0.03, target_book="betway", bankroll=None)`
  — `competition_code` takes one league or several: `"PL"` or `"PL,PD,SA,BL1"`.
  — every upcoming fixture where the model disagrees with the exchange, with
  the target book's price, the best price anywhere, expected value and an
  optional quarter-Kelly stake.
- `backtest_season_start(competition_code, train_season, test_season)` — fit on
  one whole season and predict the next, which is how the tool is used in
  August: no current-season data at all.
- `backtest_model(competition_code, test_matches=100, seasons_back=3, xi=0.0018)`
  — walk-forward scoring on matches the model never saw: Ranked Probability
  Score against a base-rate baseline, plus a calibration table.
- `tune_time_decay(competition_code, test_matches=100, seasons_back=3)` —
  grid-searches the time-decay constant ξ by backtest RPS.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite runs entirely offline - no API keys, no network - so it is safe to run
anywhere and cannot flake on a rate limit. GitHub Actions runs it on every push
against Python 3.11, 3.12 and 3.13.

## Claude Desktop

Add to `claude_desktop_config.json`
(`~/Library/Application Support/Claude/` on macOS), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "football-analytics": {
      "command": "/Users/ellisegrantboamah/Desktop/afrodevXG/venv/bin/python",
      "args": ["/Users/ellisegrantboamah/Desktop/afrodevXG/server.py"],
      "env": {
        "FOOTBALL_DATA_API_KEY": "your-key",
        "ODDS_API_KEY": "your-key"
      }
    }
  }
}
```

## How the value number works

Raw implied probability is `1 / decimal_odds`; across 1X2 those sum to more
than 1 (the bookmaker's overround). `devig()` divides each by the sum so they
total 1.0, giving fair market probabilities. The edge is then
`model_probability − fair_market_probability`.

## Missing players

The fit already reflects a team's best players, because they played in the
matches it learned from. Naming an absentee scales that team's expected scoring
down by their share of its open-play goals:

```
attack multiplier = 1 - goal_share x (1 - replacement_level)
```

Penalties are excluded from the share, since penalty duty transfers to whoever
is on the pitch. `replacement_level` (default 0.5) is how much of the absent
player's output a stand-in is assumed to provide.

Man City without Haaland, who has 29% of their open-play goals: scoring scaled
to 0.85, win probability 61.9% to 55.4%, over 2.5 goals 50.2% to 43.2%.

**This is the one part of the project that cannot be backtested.** The free tier
carries no historical lineups, so there is no way to check the adjustment
against matches that were actually played without a key player. Every other
choice here was measured; this one is reasoned. There is also no injury feed on
the free tier, so who is missing has to be supplied by you.

## Market coverage

The free Odds API plan serves only three markets through the endpoint this uses:
**1X2, over/under totals and Asian handicaps**. Those are the ones `scan_value`
can compute an edge for, because an edge needs a price to compare against.

`btts`, `double_chance`, `draw_no_bet` and `team_totals` are rejected outright
(`INVALID_MARKET`) — they live on the per-event endpoint, which needs a paid
plan.

That does not limit the model. `get_match_markets` prices every market the score
grid supports — double chance, draw no bet, clean sheets, win to nil, team
totals, correct score, handicaps at eight lines, expected points — and returns
them as probabilities. Read those against whatever odds your own bookmaker
shows: a 40% chance is fair at decimal odds of 2.50, so anything above that is
value.

Out of reach entirely, because neither data source carries them: corners, cards,
shots, player goalscorers, and anything in-play.

## Sanity checks on prices

A bookmaker's prices should sum to a little over 100% once converted to
probabilities — that surplus is its margin. An exchange sits near 102%, a soft
book near 106%.

Illiquid exchange markets return prices that sum to far more. Betfair quoted
1.27 / 1.10 / 1.09 on Atalanta v Cagliari — an implied 261%. De-vigged blindly,
that turned a 1.50 favourite into a 30% shot and produced the largest "edge" the
scanner has ever reported. Prices outside a plausible band are now rejected, and
the next sane book is used instead.

The lesson generalises: the biggest edge in a list is far more often a data
fault than an opportunity.

## Caveats

- Historical bookmaker odds need a paid Odds API plan, so `backtest_model`
  scores against a base-rate baseline, not against the market. Only live
  fixtures can be compared to real odds.
- Dixon-Coles gains over plain Poisson are real but small. De-vigged sharp odds
  are a hard baseline — treat outputs as probabilities, not certainties.
- Team-name spellings differ between the two APIs; `data/odds.py` normalizes
  names and fuzzy-matches, with an alias map for the awkward cases.
- Models are cached per league and refit at most once a day. API responses are
  cached on disk under `cache/`.

## What the backtest has found so far

Walk-forward RPS over recent seasons (lower is better):

| League | Model | Base-rate baseline | Gain |
| --- | --- | --- | --- |
| Serie A (249 matches) | **0.1941** | 0.2393 | 18.9% |
| Bundesliga (247 matches) | **0.1956** | 0.2318 | 15.6% |
| Premier League (248 matches) | 0.2063 | 0.2258 | 8.6% |
| La Liga (247 matches) | 0.2076 | 0.2226 | 6.7% |

**The model is far better in Serie A and the Bundesliga than in England or
Spain.** At 0.194 and 0.196 those two are in the range a bookmaker operates in;
the Premier League and La Liga are not close. Part of that gap is that the
Italian and German baselines are worse, so there is more room to improve on
them — but the absolute scores are genuinely lower too. Weight scanning
accordingly, and treat a large edge in La Liga, the weakest fit, with the most
suspicion.

**Second-division form does not price promoted teams.** Fitting the Championship
alongside the Premier League - two seasons of each, so that relegated and
promoted clubs link the divisions onto one scale - makes every fixture priceable
(380 of 380 against 306). But the 74 newly priceable matches score **0.269 RPS
against a 0.214 base-rate baseline**, at every second-tier weight from 0.3 to
1.0. Fixtures between established teams are unchanged.

So the coverage is real and the accuracy is not: those predictions are worse
than guessing. Promotion looks like a genuine discontinuity - a squad that just
went up is rebuilt over the summer, and its Championship results describe a
different team. `include_second_tier` exists and defaults to off. The right
behaviour is what the scanner already did: leave promoted teams alone until they
have played.

Note the trap in the first version of this test: within a single season no club
plays in both divisions, so the two leagues are disconnected and the promoted
teams' ratings float on an arbitrary scale. That run scored 0.27 to 0.36. The
bridge only exists across seasons.

**Predicting a new season from the last one works, on a small sample.** Fitted
on 2025/26 alone and asked to price the opening two weeks of 2026/27 - no
current-season data at all, so transfers and summer form are invisible:

| League | Matches | Model | Baseline | Skipped |
| --- | --- | --- | --- | --- |
| Premier League | 15 | 0.1881 | 0.2320 | 5 |
| Serie A | 14 | 0.1546 | 0.2595 | 6 |
| Bundesliga | 6 | 0.1335 | 0.1979 | 3 |
| La Liga | 22 | 0.2008 | 0.2349 | 8 |
| **Pooled** | **57** | **0.1790** | **0.2363** | **22** |

24% better than the base rate, and nominally better than the walk-forward runs.
Do not take that at face value: 57 matches gives a standard error around 0.02,
so the gap is under three standard errors, and the per-league samples of 6 to 22
are far too small to rank leagues by. Run it again in December.

The structural finding is the skipped column: **22 of 79 fixtures could not be
priced at all**, because promoted teams have no top-flight history. At the start
of a season roughly a quarter of the card is invisible to the model.

**The goals markets are not a source of edge, and are off by default.**
Over/under 2.5 and both-teams-to-score were scored by Brier across four leagues
(250 walk-forward matches each). The model lost to the league base rate on
over/under in all four, and on BTTS in three of four:

| League | Over 2.5 Brier | Base rate | Predicted over | Actual over |
| --- | --- | --- | --- | --- |
| Premier League | 0.2564 | 0.2490 | 52.6% | 54.0% |
| Serie A | 0.2590 | 0.2491 | 44.4% | 47.0% |
| Bundesliga | 0.2313 | 0.2308 | 59.1% | 64.4% |
| La Liga | 0.2520 | 0.2515 | 47.8% | 53.0% |

It also under-predicts goals in every league — which is exactly why an early
scan returned eleven unders and two overs. That was bias, not opportunity.
`scan_value` and `build_accumulator` now exclude these markets unless you pass
`include_goals_markets=True`.

**The model family barely matters.** All six penaltyblog goal models were
backtested on the same 98 PL matches and landed within 0.0012 RPS of each other
(Weibull copula 0.2043, Dixon-Coles 0.2052, plain Poisson 0.2052). Weibull
copula nominally wins while taking 107 seconds to fit against Dixon-Coles' one.
Notably, plain Poisson scores the same as Dixon-Coles: the low-score correction
Dixon-Coles is known for is worth nothing measurable here.

**Shrinkage costs a little average accuracy and buys tail protection.** Pulling
thin teams toward the league average makes RPS slightly worse (0.20633 to
0.20682 over 250 matches — inside the noise), because it adds bias to fixtures
the model already understands. It is kept because the failure it prevents is
severe rather than frequent: Coventry City, with two matches of history, was
priced at 0.3% to win at Chelsea with an away xG of 0.03. Teams above 30
matches are exempt, so established fixtures are untouched.

**Time decay is a dead end.** ξ was grid-searched on both leagues at two sample
sizes. On 98 PL matches ξ=0.005 looked best; on 248 PL matches the winner moved
to ξ=0.003, and on La Liga *no decay at all* (ξ=0.0) scored best, with RPS
rising monotonically as ξ grew. The spread across the whole grid is under 0.002
RPS — inside the noise. The default stays at 0.0018; effort belongs elsewhere.

Note that hit rate and RPS disagree (ξ=0.01 picks more winners in both leagues
while scoring worse). Hit rate ignores confidence — trust RPS.

## Roadmap

- xG-based λ via Understat (penaltyblog scrapers).
- Live/in-play tool (score + minute-adjusted Poisson).
- Other sports.
