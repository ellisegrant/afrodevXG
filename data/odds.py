"""The Odds API client plus the de-vigging helper.

Free tier is 500 credits/month and a request costs one credit per region per
market, so responses are cached on disk for a few minutes and we default to a
single region.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import unicodedata
from typing import Any, Optional

import httpx

from . import _cache

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

_ODDS_TTL = 15 * 60.0

# football-data.org competition code -> The Odds API sport key.
SPORT_KEYS = {
    "PL": "soccer_epl",
    "ELC": "soccer_efl_champ",
    "PD": "soccer_spain_la_liga",
    "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a",
    "FL1": "soccer_france_ligue_one",
    "DED": "soccer_netherlands_eredivisie",
    "PPL": "soccer_portugal_primeira_liga",
    "BSA": "soccer_brazil_campeonato",
    "CL": "soccer_uefa_champs_league",
}

# Betting exchanges quote what people will actually trade at, with no margin
# baked into the price, so they are the honest yardstick for "is this likely".
# Ordered by preference; the first one quoting a match wins.
SHARP_BOOKS = ["betfair_ex_uk", "smarkets", "matchbook", "pinnacle", "betfair_sb_uk"]

# A book's prices should imply a little over 100% once summed - that surplus is
# its margin. An exchange sits near 1.02; a soft book near 1.06. Anything outside
# this band is bad data, not a bargain: illiquid exchange markets in particular
# come back with prices that imply 200%+ and would read as an enormous edge.
MIN_OVERROUND = 0.98
MAX_SHARP_OVERROUND = 1.12
MAX_TWO_WAY_OVERROUND = 1.10

# The book you would actually place the bet with - the price you can really get.
DEFAULT_TARGET_BOOK = "betway"

# Each market costs one credit per region per request, so keep the default lean.
DEFAULT_MARKETS = "h2h,totals,spreads"

# Preferred goals line when books quote more than one; the model can price any.
TOTALS_LINE = 2.5

# Markets the free plan's /odds endpoint will not serve. They exist only on the
# per-event endpoint, which needs a paid plan: btts, double_chance,
# draw_no_bet, team_totals. get_match_markets prices them from the model instead.
UNSUPPORTED_FREE_MARKETS = ("btts", "double_chance", "draw_no_bet", "team_totals")


# Searched in order when the caller does not say which competition it is.
_DEFAULT_SPORT_SEARCH = [
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
]

# The two feeds spell some clubs differently; map both sides onto one token.
NAME_ALIASES = {
    "manchester united": "man united",
    "manchester utd": "man united",
    "man utd": "man united",
    "man u": "man united",
    "manchester city": "man city",
    "man city": "man city",
    "manchester city": "man city",
    "tottenham hotspur": "tottenham",
    "spurs": "tottenham",
    "wolverhampton wanderers": "wolves",
    "wolverhampton": "wolves",
    "leicester city": "leicester",
    "west bromwich albion": "west brom",
    "brighton and hove albion": "brighton",
    "brighton hove albion": "brighton",
    "west ham united": "west ham",
    "newcastle united": "newcastle",
    "leeds united": "leeds",
    "nottingham forest": "nottm forest",
    "sheffield united": "sheffield utd",
    "paris saint germain": "psg",
    "paris saint-germain": "psg",
    "bayern munchen": "bayern munich",
    "fc bayern munchen": "bayern munich",
    "borussia monchengladbach": "gladbach",
    "borussia dortmund": "dortmund",
    "bayer 04 leverkusen": "leverkusen",
    "rb leipzig": "leipzig",
    "atletico madrid": "atletico",
    "club atletico de madrid": "atletico",
    "athletic club": "athletic bilbao",
    "real sociedad de futbol": "real sociedad",
    "internazionale": "inter",
    "inter milan": "inter",
    "ac milan": "milan",
    "as roma": "roma",
    "ss lazio": "lazio",
    "olympique de marseille": "marseille",
    "olympique lyonnais": "lyon",
}

_NOISE_WORDS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "rc", "cd", "ud",
    "club", "calcio", "de", "the",
}


class OddsAPIError(RuntimeError):
    """Raised when The Odds API cannot be reached or refuses a request."""


def normalize_team(name: str) -> str:
    """Lowercase, strip accents, drop club-suffix noise, then apply aliases."""
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in NAME_ALIASES:
        return NAME_ALIASES[text]

    stripped = " ".join(word for word in text.split() if word not in _NOISE_WORDS)
    stripped = stripped or text
    return NAME_ALIASES.get(stripped, stripped)


def _similarity(left: str, right: str) -> float:
    left, right = normalize_team(left), normalize_team(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.95
    return difflib.SequenceMatcher(None, left, right).ratio()


def devig(odds: dict[str, float]) -> dict[str, float]:
    """Strip the bookmaker margin by multiplicative normalization.

    Raw implied probability is 1/decimal_odds; those sum to more than 1 (the
    overround), so each is divided by the sum to give fair probabilities.
    """
    raw = {}
    for outcome, price in odds.items():
        if not price or price <= 1.0:
            raise ValueError(f"invalid decimal odds for {outcome}: {price!r}")
        raw[outcome] = 1.0 / float(price)

    total = sum(raw.values())
    if total <= 0:
        raise ValueError("odds produced a non-positive overround")
    return {outcome: value / total for outcome, value in raw.items()}


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        raise OddsAPIError(
            "ODDS_API_KEY is not set. Copy .env.example to .env and add a key "
            "from https://the-odds-api.com/"
        )
    return key


def _regions() -> str:
    # Each extra region costs another credit per request.
    return os.environ.get("ODDS_REGIONS", "uk").strip() or "uk"


def _fetch_sport(sport_key: str, markets: str = DEFAULT_MARKETS) -> list[dict[str, Any]]:
    """Upcoming markets for one sport key (one credit per market per region)."""
    cache_key = f"odds_{sport_key}_{_regions()}_{markets}"
    cached = _cache.load(cache_key, _ODDS_TTL)
    if cached is not None:
        return cached

    log.info("odds-api GET %s", sport_key)
    try:
        response = httpx.get(
            f"{BASE_URL}/sports/{sport_key}/odds",
            params={
                "apiKey": _api_key(),
                "regions": _regions(),
                "markets": markets,
                "oddsFormat": "decimal",
            },
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        raise OddsAPIError(f"request to the-odds-api.com failed: {exc}") from exc

    if response.status_code == 401:
        raise OddsAPIError("The Odds API rejected the key (401). Check ODDS_API_KEY.")
    if response.status_code == 429:
        raise OddsAPIError("The Odds API quota exhausted (429). Credits reset monthly.")
    if response.status_code == 422:
        log.warning("unknown sport key %s", sport_key)
        return []
    response.raise_for_status()

    remaining = response.headers.get("x-requests-remaining")
    if remaining is not None:
        log.info("odds-api credits remaining: %s", remaining)

    events = response.json()
    _cache.save(cache_key, events)
    return events


def _match_event(
    events: list[dict[str, Any]], home_team: str, away_team: str, threshold: float = 0.72
) -> Optional[dict[str, Any]]:
    """Best fuzzy name match for the fixture, or None if nothing is close."""
    best, best_score = None, 0.0
    for event in events:
        score = min(
            _similarity(home_team, event.get("home_team", "")),
            _similarity(away_team, event.get("away_team", "")),
        )
        if score > best_score:
            best, best_score = event, score
    if best is None or best_score < threshold:
        return None
    log.info(
        "matched %s v %s -> %s v %s (%.2f)",
        home_team, away_team, best.get("home_team"), best.get("away_team"), best_score,
    )
    return best


def _average_h2h(event: dict[str, Any]) -> Optional[dict[str, float]]:
    """Mean decimal odds across every bookmaker quoting the 1X2 market."""
    home_name = event.get("home_team", "")
    away_name = event.get("away_team", "")
    buckets: dict[str, list[float]] = {"home": [], "draw": [], "away": []}

    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name, price = outcome.get("name", ""), outcome.get("price")
                if not price:
                    continue
                if name.lower() == "draw":
                    buckets["draw"].append(float(price))
                elif _similarity(name, home_name) >= 0.9:
                    buckets["home"].append(float(price))
                elif _similarity(name, away_name) >= 0.9:
                    buckets["away"].append(float(price))

    if not all(buckets.values()):
        return None
    return {key: sum(prices) / len(prices) for key, prices in buckets.items()}


def fetch_odds(
    home_team: str, away_team: str, competition_code: Optional[str] = None
) -> Optional[dict[str, float]]:
    """Average bookmaker 1X2 decimal odds for a fixture, or None if unlisted.

    Pass `competition_code` (e.g. "PL") to look in a single sport and spend one
    credit instead of searching several.
    """
    if competition_code:
        sport_keys = [SPORT_KEYS.get(competition_code.upper().strip())]
        sport_keys = [key for key in sport_keys if key]
        if not sport_keys:
            log.warning("no Odds API sport key mapped for %s", competition_code)
            return None
    else:
        sport_keys = _DEFAULT_SPORT_SEARCH

    for sport_key in sport_keys:
        try:
            events = _fetch_sport(sport_key)
        except OddsAPIError as exc:
            log.warning("odds lookup failed for %s: %s", sport_key, exc)
            return None
        event = _match_event(events, home_team, away_team)
        if event is None:
            continue
        prices = _average_h2h(event)
        if prices:
            return prices
        log.info("event found in %s but no usable h2h market", sport_key)

    log.info("no bookmaker odds found for %s v %s", home_team, away_team)
    return None


def _book_prices(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-bookmaker prices for one event, keyed by bookmaker key.

    Each entry carries whichever of the 1X2, over/under and both-teams-to-score
    markets that book is quoting.
    """
    home_name = event.get("home_team", "")
    away_name = event.get("away_team", "")

    books: dict[str, dict[str, Any]] = {}
    for bookmaker in event.get("bookmakers", []):
        entry: dict[str, Any] = {
            "title": bookmaker.get("title", bookmaker.get("key", "")),
            "odds": None,
            "totals": None,
            "spreads": None,
            "btts": None,
        }

        for market in bookmaker.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])

            if key == "h2h":
                prices: dict[str, float] = {}
                for outcome in outcomes:
                    name, price = outcome.get("name", ""), outcome.get("price")
                    if not price:
                        continue
                    if name.lower() == "draw":
                        prices["draw"] = float(price)
                    elif _similarity(name, home_name) >= 0.9:
                        prices["home"] = float(price)
                    elif _similarity(name, away_name) >= 0.9:
                        prices["away"] = float(price)
                if len(prices) == 3:
                    entry["odds"] = prices

            elif key == "totals":
                by_line: dict[float, dict[str, float]] = {}
                for outcome in outcomes:
                    price, point = outcome.get("price"), outcome.get("point")
                    side = outcome.get("name", "").lower()
                    if not price or point is None or side not in ("over", "under"):
                        continue
                    by_line.setdefault(float(point), {})[side] = float(price)
                complete = {
                    line: prices for line, prices in by_line.items() if len(prices) == 2
                }
                if complete:
                    entry["totals"] = complete

            elif key == "spreads":
                by_line: dict[float, dict[str, float]] = {}
                for outcome in outcomes:
                    price, point = outcome.get("price"), outcome.get("point")
                    name = outcome.get("name", "")
                    if not price or point is None:
                        continue
                    if _similarity(name, home_name) >= 0.9:
                        side = "home"
                    elif _similarity(name, away_name) >= 0.9:
                        side = "away"
                    else:
                        continue
                    # A handicap is quoted from each side's own perspective; index
                    # both legs under the home team's line so they pair up.
                    line = float(point) if side == "home" else -float(point)
                    by_line.setdefault(line, {})[side] = float(price)
                complete = {
                    line: prices for line, prices in by_line.items() if len(prices) == 2
                }
                if complete:
                    entry["spreads"] = complete

            elif key == "btts":
                prices = {}
                for outcome in outcomes:
                    price = outcome.get("price")
                    side = outcome.get("name", "").lower()
                    if price and side in ("yes", "no"):
                        prices[side] = float(price)
                if len(prices) == 2:
                    entry["btts"] = prices

        if any(entry[market] for market in ("odds", "totals", "spreads", "btts")):
            books[bookmaker.get("key", "")] = entry
    return books


def overround(prices: dict[str, float]) -> float:
    """Sum of implied probabilities. 1.05 means a 5% margin."""
    try:
        return sum(1.0 / float(price) for price in prices.values())
    except (TypeError, ZeroDivisionError):
        return float("inf")


def _usable(prices: dict[str, float], maximum: float) -> bool:
    """Reject prices that cannot be a real market."""
    if not prices or any(not price or price <= 1.0 for price in prices.values()):
        return False
    return MIN_OVERROUND <= overround(prices) <= maximum


def _choose_sharp(
    books: dict[str, dict[str, Any]], market: str, maximum: float
) -> Optional[str]:
    """First preferred exchange whose prices are sane, else the tightest book.

    Preference order alone is not enough: an exchange with no liquidity still
    returns prices, and they are nonsense.
    """
    for key in SHARP_BOOKS:
        book = books.get(key)
        if book and _usable(book.get(market) or {}, maximum):
            return key

    valid = {
        key: overround(book[market])
        for key, book in books.items()
        if _usable(book.get(market) or {}, maximum)
    }
    return min(valid, key=valid.get) if valid else None


def _best_prices(
    books: dict[str, dict[str, Any]], market: str = "odds"
) -> dict[str, dict[str, Any]]:
    """Highest price available on each outcome of one market, and who offers it."""
    best: dict[str, dict[str, Any]] = {}
    for key, book in books.items():
        prices = book.get(market)
        if not prices:
            continue
        for outcome, price in prices.items():
            if outcome not in best or price > best[outcome]["odds"]:
                best[outcome] = {"odds": price, "book": book["title"], "key": key}
    return best


def _pick_line(
    books: dict[str, dict[str, Any]], market: str, preferred: Optional[float] = None
) -> Optional[float]:
    """The line the most books are quoting, breaking ties toward `preferred`."""
    counts: dict[float, int] = {}
    for book in books.values():
        for line in (book.get(market) or {}):
            counts[line] = counts.get(line, 0) + 1
    if not counts:
        return None
    most = max(counts.values())
    candidates = [line for line, count in counts.items() if count == most]
    if preferred is not None and preferred in candidates:
        return preferred
    return min(candidates, key=abs)


def _flatten_line(
    books: dict[str, dict[str, Any]], market: str, line: float
) -> dict[str, dict[str, Any]]:
    """Books quoting one specific line, shaped like any other two-way market."""
    flattened = {}
    for key, book in books.items():
        prices = (book.get(market) or {}).get(line)
        if prices:
            flattened[key] = {"title": book["title"], market: prices}
    return flattened


def _side_market(
    books: dict[str, dict[str, Any]], market: str, target_book: str
) -> Optional[dict[str, Any]]:
    """Sharp baseline, target price and best price for a two-way market."""
    quoting = {key: book for key, book in books.items() if book.get(market)}
    if not quoting:
        return None

    sharp_key = _choose_sharp(quoting, market, MAX_TWO_WAY_OVERROUND)
    if sharp_key is None:
        sane = [
            book for book in quoting.values()
            if _usable(book.get(market) or {}, 1.20)
        ]
        if not sane:
            return None
        outcomes = sane[0][market].keys()
        averaged = {
            outcome: sum(book[market][outcome] for book in sane) / len(sane)
            for outcome in outcomes
        }
        sharp = {"title": f"average of {len(sane)} books", market: averaged}
    else:
        sharp = quoting[sharp_key]

    target = quoting.get(target_book)
    return {
        "sharp_book": sharp["title"],
        "sharp_odds": sharp[market],
        "fair": devig(sharp[market]),
        "target_odds": target[market] if target else None,
        "best_odds": _best_prices(quoting, market),
        "books_seen": len(quoting),
    }


def _line_market(
    books: dict[str, dict[str, Any]],
    market: str,
    line: Optional[float],
    target_book: str,
) -> Optional[dict[str, Any]]:
    """One line of a lined market, assembled like any other two-way market."""
    if line is None:
        return None
    assembled = _side_market(_flatten_line(books, market, line), market, target_book)
    return {**assembled, "line": line} if assembled else None


def market_from_event(
    event: dict[str, Any], target_book: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Split one event's prices into a sharp baseline and a bettable price.

    The sharp book (an exchange where possible) is de-vigged to give the fair
    market probability; the target book is simply the price you could take.
    """
    all_books = _book_prices(event)
    books = {key: book for key, book in all_books.items() if book.get("odds")}
    if not books:
        return None

    target_book = (target_book or DEFAULT_TARGET_BOOK).lower()

    sharp_key = _choose_sharp(books, "odds", MAX_SHARP_OVERROUND)
    if sharp_key is None:
        # Nothing quoting sane prices: average the books that at least look like
        # a market. This carries every book's margin and is a weaker baseline.
        sane = [
            book for book in books.values()
            if _usable(book.get("odds") or {}, 1.25)
        ]
        if not sane:
            log.warning(
                "no usable 1X2 prices for %s v %s",
                event.get("home_team"), event.get("away_team"),
            )
            return None
        averaged = {
            outcome: sum(book["odds"][outcome] for book in sane) / len(sane)
            for outcome in ("home", "draw", "away")
        }
        sharp = {"title": f"average of {len(sane)} books", "odds": averaged}
    else:
        sharp = books[sharp_key]

    raw_total = sum(1.0 / price for price in sharp["odds"].values())
    totals_line = _pick_line(all_books, "totals", TOTALS_LINE)
    spread_line = _pick_line(all_books, "spreads")

    return {
        "home_team": event.get("home_team", ""),
        "away_team": event.get("away_team", ""),
        "commence_time": event.get("commence_time", ""),
        "sharp_book": sharp["title"],
        "sharp_odds": sharp["odds"],
        "fair": devig(sharp["odds"]),
        "sharp_overround_pct": (raw_total - 1.0) * 100.0,
        "target_book": books[target_book]["title"] if target_book in books else None,
        "target_odds": books[target_book]["odds"] if target_book in books else None,
        "best_odds": _best_prices(books),
        "books_seen": len(books),
        "totals_line": totals_line,
        "totals": _line_market(all_books, "totals", totals_line, target_book),
        "spread_line": spread_line,
        "spreads": _line_market(all_books, "spreads", spread_line, target_book),
        "btts": _side_market(all_books, "btts", target_book),
    }


def fetch_events(
    competition_code: str, markets: str = DEFAULT_MARKETS
) -> list[dict[str, Any]]:
    """Every upcoming event with odds for a competition (one credit, then cached)."""
    sport_key = SPORT_KEYS.get(competition_code.upper().strip())
    if not sport_key:
        log.warning("no Odds API sport key mapped for %s", competition_code)
        return []
    return _fetch_sport(sport_key, markets=markets)


def fetch_market(
    home_team: str,
    away_team: str,
    competition_code: Optional[str] = None,
    target_book: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Sharp baseline + bettable price for one fixture, or None if unlisted."""
    sport_keys = (
        [SPORT_KEYS[competition_code.upper().strip()]]
        if competition_code and competition_code.upper().strip() in SPORT_KEYS
        else _DEFAULT_SPORT_SEARCH
    )

    for sport_key in sport_keys:
        try:
            events = _fetch_sport(sport_key)
        except OddsAPIError as exc:
            log.warning("odds lookup failed for %s: %s", sport_key, exc)
            return None
        event = _match_event(events, home_team, away_team)
        if event is not None:
            return market_from_event(event, target_book=target_book)

    log.info("no bookmaker odds found for %s v %s", home_team, away_team)
    return None
