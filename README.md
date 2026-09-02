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

- `get_match_probabilities(home_team, away_team, competition_code="PL", seasons_back=3, include_odds=True)`
  — 1X2, over/under 2.5, BTTS, expected goals, plus de-vigged market
  probabilities and the model-minus-market edge.
- `list_competitions()` — competition codes (PL, PD, BL1, SA, FL1, …).
- `list_teams(competition_code)` — team names as the results feed spells them.
- `get_upcoming_fixtures(competition_code, days_ahead=14)` — scheduled matches.
- `build_accumulator(target_odds=4.0, competition_code, days_ahead=3, objective)`
  — "give me a 4.0 on this weekend's games": searches every selection across
  every market for combinations that multiply out to the requested price,
  ranked by the model's chance that all legs land. One leg per fixture.
- `get_match_markets(home_team, away_team, competition_code)` — the full model
  probability sheet: 1X2, double chance, draw no bet, over/under from 0.5 to
  4.5, BTTS, clean sheets, win to nil, team totals, Asian handicaps, correct
  score and expected points.
- `scan_value(competition_code, min_edge=0.03, target_book="betway", bankroll=None)`
  — every upcoming fixture where the model disagrees with the exchange, with
  the target book's price, the best price anywhere, expected value and an
  optional quarter-Kelly stake.
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

| League | Model | Base-rate baseline |
| --- | --- | --- |
| Premier League (248 matches) | 0.2063 | 0.2258 |
| La Liga (247 matches) | 0.2095 | 0.2226 |

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
