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
- `backtest_model(competition_code, test_matches=100, seasons_back=3, xi=0.0018)`
  — walk-forward scoring on matches the model never saw: Ranked Probability
  Score against a base-rate baseline, plus a calibration table.
- `tune_time_decay(competition_code, test_matches=100, seasons_back=3)` —
  grid-searches the time-decay constant ξ by backtest RPS.

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

## Roadmap

- Time-decay (ξ) tuning by ranked probability score.
- xG-based λ via Understat (penaltyblog scrapers).
- Live/in-play tool (score + minute-adjusted Poisson).
- Other sports.
