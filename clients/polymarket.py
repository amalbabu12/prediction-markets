"""
Polymarket API clients — fully public, no authentication required for read access.

Three separate APIs:

1. Gamma Markets API  (https://gamma-api.polymarket.com)
   Market and event metadata, categories, discovery.
   Rate limits: ~50 req/s on /events, ~30 req/s on /markets.

2. CLOB API           (https://clob.polymarket.com)
   Central Limit Order Book — orderbooks, pricing, price history, trades.
   Rate limits: ~150 req/s for single lookups, ~50 req/s for batch endpoints.

3. Data API           (https://data-api.polymarket.com)
   Trade history, positions, open interest, holder data.
   Rate limits: ~20 req/s on /trades, ~15 req/s on /positions.

Key ID types:
  condition_id — one per market (used in Gamma + Data APIs)
  token_id     — one per outcome (Yes token / No token); used in CLOB API for
                 orderbooks and price history. Each market has two token_ids.

Prices on Polymarket are decimal probabilities in [0.0, 1.0].
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional

import requests

logger = logging.getLogger(__name__)


class _RateLimiter:
    def __init__(self, calls_per_second: float) -> None:
        self._interval = 1.0 / calls_per_second
        self._last: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.monotonic()


class _BaseClient:
    """Shared HTTP logic for all three Polymarket API clients."""

    BASE_URL: str = ""

    def __init__(self, rate_limit: float) -> None:
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._rl = _RateLimiter(rate_limit)

    def _get(
        self,
        path: str,
        params: Optional[dict] = None,
        retries: int = 4,
    ) -> Any:
        self._rl.wait()
        url = self.BASE_URL + path
        for attempt in range(retries):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    backoff = 2 ** (attempt + 1)
                    logger.warning("%s 429 rate limit. Sleeping %ds.", self.BASE_URL, backoff)
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    logger.error("Request failed (%s%s): %s", self.BASE_URL, path, exc)
                    raise
                backoff = 2 ** attempt
                logger.warning("Error attempt %d/%d (%s): %s. Retry in %ds.", attempt + 1, retries, path, exc, backoff)
                time.sleep(backoff)
        return {}

    def _post(
        self,
        path: str,
        body: Any,
        retries: int = 4,
    ) -> Any:
        self._rl.wait()
        url = self.BASE_URL + path
        for attempt in range(retries):
            try:
                resp = self._session.post(url, json=body, timeout=30)
                if resp.status_code == 429:
                    backoff = 2 ** (attempt + 1)
                    logger.warning("%s 429 rate limit. Sleeping %ds.", self.BASE_URL, backoff)
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    logger.error("POST failed (%s%s): %s", self.BASE_URL, path, exc)
                    raise
                backoff = 2 ** attempt
                time.sleep(backoff)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# Gamma Markets API
# ══════════════════════════════════════════════════════════════════════════════

class PolymarketGammaClient(_BaseClient):
    """
    Polymarket Gamma Markets API client.

    Provides market/event metadata, categories, and discovery.
    Fully public — no authentication needed.

    Usage:
        gamma = PolymarketGammaClient()
        events = gamma.get_all_events(active=True, closed=False)
        markets = gamma.get_all_markets()
    """

    BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, rate_limit: float = 30.0) -> None:
        super().__init__(rate_limit)

    # ── Tags & Categories ─────────────────────────────────────────────────────

    def get_tags(self) -> list[dict]:
        """All category tags (Politics, Sports, Crypto, etc.) with rank and market counts."""
        result = self._get("/tags")
        return result if isinstance(result, list) else []

    def get_series(self) -> list[dict]:
        """Grouped event series collections."""
        result = self._get("/series")
        return result if isinstance(result, list) else []

    # ── Events ────────────────────────────────────────────────────────────────

    def iter_events(
        self,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        archived: Optional[bool] = None,
        tag_id: Optional[str] = None,
        order: str = "volume",
        ascending: bool = False,
        limit: int = 100,
    ) -> Iterator[dict]:
        """
        Paginate through all Polymarket events.

        Args:
            active: True = currently tradable; False = inactive
            closed: True = include closed events; False = exclude
            archived: Include/exclude archived events
            tag_id: Filter by category tag ID (get IDs from get_tags())
            order: Sort field — "volume", "volume_24hr", "liquidity",
                   "start_date", "end_date", "competitive", "closed_time"
            ascending: Sort direction (default: descending)
            limit: Page size (max 100 recommended)

        Each event contains: id, slug, title, description, active, closed,
        archived, volume, liquidity, startDate, endDate, markets (list), tags.
        """
        offset = 0
        page = 0
        while True:
            params: dict[str, Any] = {
                "limit": limit,
                "offset": offset,
                "order": order,
                "ascending": str(ascending).lower(),
            }
            if active is not None:
                params["active"] = str(active).lower()
            if closed is not None:
                params["closed"] = str(closed).lower()
            if archived is not None:
                params["archived"] = str(archived).lower()
            if tag_id:
                params["tag_id"] = tag_id

            data = self._get("/events", params)
            events: list[dict] = data if isinstance(data, list) else data.get("events", [])
            page += 1
            logger.debug("Gamma events page %d (offset=%d): %d results.", page, offset, len(events))
            yield from events

            if len(events) < limit:
                break
            offset += limit

    def get_all_events(self, **kwargs: Any) -> list[dict]:
        """Collect all events matching criteria."""
        return list(self.iter_events(**kwargs))

    def get_event(self, event_id: str) -> dict:
        """Single event by ID."""
        result = self._get(f"/events/{event_id}")
        return result if isinstance(result, dict) else {}

    def get_event_by_slug(self, slug: str) -> dict:
        """Single event by URL slug."""
        result = self._get("/events", {"slug": slug})
        if isinstance(result, list) and result:
            return result[0]
        return result if isinstance(result, dict) else {}

    # ── Markets ───────────────────────────────────────────────────────────────

    def iter_markets(
        self,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        archived: Optional[bool] = None,
        tag_id: Optional[str] = None,
        order: str = "volume",
        ascending: bool = False,
        limit: int = 100,
    ) -> Iterator[dict]:
        """
        Paginate through all Gamma markets.

        Each market contains: id, question, conditionId, slug, resolutionSource,
        endDate, liquidity, volume, active, closed, archived, outcomePrices,
        outcomes, clobTokenIds, groupItemTitle, groupItemThreshold.

        Note: conditionId links to the CLOB API; clobTokenIds are the Yes/No token IDs.
        """
        offset = 0
        page = 0
        while True:
            params: dict[str, Any] = {
                "limit": limit,
                "offset": offset,
                "order": order,
                "ascending": str(ascending).lower(),
            }
            if active is not None:
                params["active"] = str(active).lower()
            if closed is not None:
                params["closed"] = str(closed).lower()
            if archived is not None:
                params["archived"] = str(archived).lower()
            if tag_id:
                params["tag_id"] = tag_id

            data = self._get("/markets", params)
            markets: list[dict] = data if isinstance(data, list) else data.get("markets", [])
            page += 1
            logger.debug("Gamma markets page %d (offset=%d): %d results.", page, offset, len(markets))
            yield from markets

            if len(markets) < limit:
                break
            offset += limit

    def get_all_markets(self, **kwargs: Any) -> list[dict]:
        """Collect all markets matching criteria."""
        return list(self.iter_markets(**kwargs))

    def get_market(self, market_id: str) -> dict:
        """Single market by Gamma market ID."""
        result = self._get(f"/markets/{market_id}")
        return result if isinstance(result, dict) else {}

    def get_market_by_slug(self, slug: str) -> dict:
        """Single market by URL slug."""
        result = self._get("/markets", {"slug": slug})
        if isinstance(result, list) and result:
            return result[0]
        return result if isinstance(result, dict) else {}

    def search(self, query: str) -> dict:
        """Cross-search events, markets, and profiles."""
        result = self._get("/public-search", {"q": query})
        return result if isinstance(result, dict) else {}


# ══════════════════════════════════════════════════════════════════════════════
# CLOB API
# ══════════════════════════════════════════════════════════════════════════════

class PolymarketCLOBClient(_BaseClient):
    """
    Polymarket CLOB (Central Limit Order Book) API client.

    Public endpoints (L1) — no auth needed:
      - Market listing
      - Orderbooks (single + batch)
      - Pricing: best price, midpoint, spread, last trade
      - Price history (time series)
      - Trades

    All prices are decimal probabilities: 0.0 = 0%, 1.0 = 100%.
    token_id is the ERC1155 token ID for a specific outcome (Yes or No).

    Usage:
        clob = PolymarketCLOBClient()
        book = clob.get_orderbook("TOKEN_ID_FOR_YES")
        history = clob.get_price_history("TOKEN_ID", interval="max", fidelity=60)
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, rate_limit: float = 80.0) -> None:
        super().__init__(rate_limit)

    # ── Markets ───────────────────────────────────────────────────────────────

    def iter_markets(self, next_cursor: str = "") -> Iterator[dict]:
        """
        Paginate through all CLOB markets.

        Each market: condition_id, question_id, tokens (list of {token_id, outcome}),
        rewards, minimum_order_size, minimum_tick_size, description, category,
        end_date_iso, game_start_time, question, market_slug, accepting_orders.
        """
        cursor = next_cursor
        page = 0
        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["next_cursor"] = cursor

            data = self._get("/markets", params)
            markets: list[dict] = data.get("data", [])
            page += 1
            logger.debug("CLOB markets page %d: %d results.", page, len(markets))
            yield from markets

            cursor = data.get("next_cursor", "")
            # "LTE=" is the terminal cursor sentinel
            if not cursor or cursor == "LTE=" or not markets:
                break

    def get_all_clob_markets(self) -> list[dict]:
        """Collect all CLOB markets."""
        return list(self.iter_markets())

    def get_market(self, condition_id: str) -> dict:
        """Single CLOB market by condition ID."""
        result = self._get(f"/markets/{condition_id}")
        return result if isinstance(result, dict) else {}

    def get_simplified_markets(self) -> list[dict]:
        """Lighter-weight market list (faster, less detail)."""
        result = self._get("/simplified-markets")
        return result.get("data", []) if isinstance(result, dict) else []

    # ── Orderbook ─────────────────────────────────────────────────────────────

    def get_orderbook(self, token_id: str) -> dict:
        """
        Current limit order book for a single outcome token.

        Args:
            token_id: ERC1155 token ID for a specific outcome (Yes or No)

        Returns:
            {
              "bids": [{"price": "0.65", "size": "100"}, ...],
              "asks": [{"price": "0.66", "size": "200"}, ...],
              "market": "condition_id",
              "asset_id": "token_id",
            }
        """
        return self._get("/book", {"token_id": token_id}) or {}

    def get_orderbooks_batch(self, token_ids: list[str]) -> list[dict]:
        """
        Fetch multiple orderbooks in a single POST request (more efficient).
        Returns a list of orderbook dicts in the same order as token_ids.
        """
        body = [{"token_id": tid} for tid in token_ids]
        result = self._post("/books", body)
        return result if isinstance(result, list) else []

    # ── Pricing ───────────────────────────────────────────────────────────────

    def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Best available price for a given side ('BUY' or 'SELL')."""
        data = self._get("/price", {"token_id": token_id, "side": side})
        raw = data.get("price") if isinstance(data, dict) else None
        return float(raw) if raw is not None else None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """(best_bid + best_ask) / 2 for a token."""
        data = self._get("/midpoint", {"token_id": token_id})
        raw = data.get("mid") if isinstance(data, dict) else None
        return float(raw) if raw is not None else None

    def get_spread(self, token_id: str) -> Optional[float]:
        """best_ask - best_bid for a token."""
        data = self._get("/spread", {"token_id": token_id})
        raw = data.get("spread") if isinstance(data, dict) else None
        return float(raw) if raw is not None else None

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Most recent execution price for a token."""
        data = self._get("/last-trade-price", {"token_id": token_id})
        raw = data.get("price") if isinstance(data, dict) else None
        return float(raw) if raw is not None else None

    def get_midpoints_batch(self, token_ids: list[str]) -> dict[str, float]:
        """
        Batch midpoint prices. More efficient than individual calls.
        Returns {token_id: midpoint_price}.
        """
        body = [{"token_id": tid} for tid in token_ids]
        result = self._post("/midpoints", body)
        if isinstance(result, dict):
            return {k: float(v) for k, v in result.items() if v is not None}
        return {}

    def get_prices_batch(self, token_ids: list[str], side: str = "BUY") -> dict[str, float]:
        """
        Batch best prices for a side. Returns {token_id: price}.
        """
        body = [{"token_id": tid, "side": side} for tid in token_ids]
        result = self._post("/prices", body)
        if isinstance(result, dict):
            return {k: float(v) for k, v in result.items() if v is not None}
        return {}

    # ── Price History (Time Series) ───────────────────────────────────────────

    def get_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: Optional[str] = None,
        fidelity: int = 60,
    ) -> list[dict]:
        """
        Historical price time series for a token.

        Args:
            token_id: ERC1155 token ID (Yes or No outcome)
            start_ts / end_ts: Unix timestamp range (mutually exclusive with interval)
            interval: Relative window — "1m", "1h", "6h", "1d", "1w", "max", "all"
                      "max" / "all" returns the full history at the given fidelity.
            fidelity: Resolution in minutes (e.g., 1 = 1-minute bars, 60 = hourly)

        Returns:
            [{"t": unix_timestamp, "p": price_float}, ...]

        Notes:
            - Pass interval="max" with fidelity=1 for the highest-resolution full history.
            - Pass interval="max" with fidelity=60 for hourly full history (smaller payload).
            - Polymarket prices are in [0.0, 1.0].
        """
        params: dict[str, Any] = {"market": token_id, "fidelity": fidelity}
        if interval:
            params["interval"] = interval
        else:
            if start_ts is not None:
                params["startTs"] = start_ts
            if end_ts is not None:
                params["endTs"] = end_ts

        data = self._get("/prices-history", params)
        return (data or {}).get("history", [])

    def get_full_price_history(
        self,
        token_id: str,
        fidelity: int = 60,
    ) -> list[dict]:
        """Convenience: fetch the complete price history at the given fidelity."""
        return self.get_price_history(token_id, interval="max", fidelity=fidelity)

    # ── Trades ────────────────────────────────────────────────────────────────

    def get_trades(self, condition_id: str, limit: int = 500) -> list[dict]:
        """
        Recent trades for a market.

        Args:
            condition_id: Market condition ID
            limit: Max results (capped at 500)

        Each trade: taker_side, size, price, fee_rate_bps, transaction_hash,
        bucket_index, timestamp (ISO).
        """
        params = {"market": condition_id, "limit": min(limit, 500)}
        result = self._get("/trades", params)
        return result if isinstance(result, list) else []

    # ── Market Metadata ───────────────────────────────────────────────────────

    def get_tick_size(self, token_id: str) -> Optional[float]:
        """Minimum price increment for a token."""
        data = self._get("/tick-size", {"token_id": token_id})
        raw = data.get("minimum_tick_size") if isinstance(data, dict) else None
        return float(raw) if raw is not None else None

    def get_fee_rate_bps(self, token_id: str) -> Optional[int]:
        """Fee rate in basis points (e.g., 200 = 2%)."""
        data = self._get("/fee-rate-bps", {"token_id": token_id})
        raw = data.get("fee_rate_bps") if isinstance(data, dict) else None
        return int(raw) if raw is not None else None

    def is_neg_risk(self, token_id: str) -> Optional[bool]:
        """Whether a market uses the negative-risk (multi-outcome) mechanism."""
        data = self._get("/neg-risk", {"token_id": token_id})
        raw = data.get("neg_risk") if isinstance(data, dict) else None
        return bool(raw) if raw is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# Data API
# ══════════════════════════════════════════════════════════════════════════════

class PolymarketDataClient(_BaseClient):
    """
    Polymarket Data API client.

    Provides trade history, positions, open interest, and holder data.
    Fully public — no authentication needed.

    Usage:
        data = PolymarketDataClient()
        trades = data.get_trades(market="CONDITION_ID")
        oi = data.get_open_interest("CONDITION_ID")
    """

    BASE_URL = "https://data-api.polymarket.com"

    def __init__(self, rate_limit: float = 15.0) -> None:
        super().__init__(rate_limit)

    def get_trades(
        self,
        market: Optional[str] = None,
        user: Optional[str] = None,
        limit: int = 500,
        taker_only: bool = False,
        side: Optional[str] = None,
    ) -> list[dict]:
        """
        Trade history from the Data API.

        Args:
            market: condition_id — filter to a specific market
            user: Polygon wallet address — filter to a specific trader
            limit: Max results (capped at 500 per call)
            taker_only: Return only taker-side trades
            side: "BUY" or "SELL"

        Each trade: id, taker_order_id, market (conditionId), asset_id (tokenId),
        side, size, fee_rate_bps, price, status, outcome, timestamp, transaction_hash.
        """
        params: dict[str, Any] = {"limit": min(limit, 500)}
        if market:
            params["market"] = market
        if user:
            params["user"] = user
        if taker_only:
            params["takerOnly"] = "true"
        if side:
            params["side"] = side
        result = self._get("/trades", params)
        return result if isinstance(result, list) else []

    def iter_trades(
        self,
        market: Optional[str] = None,
        user: Optional[str] = None,
        taker_only: bool = False,
    ) -> Iterator[dict]:
        """
        Paginate through all trades for a market (500 per page).
        Note: Data API doesn't expose a cursor; use start/end timestamps
        for deeper pagination if needed.
        """
        result = self.get_trades(market=market, user=user, limit=500, taker_only=taker_only)
        yield from result

    def get_open_interest(self, market: str) -> dict:
        """
        Open interest for a market.
        Returns: {condition_id, outstanding_shares, value}.
        """
        result = self._get("/oi", {"market": market})
        return result if isinstance(result, dict) else {}

    def get_holders(
        self,
        market: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        Top token holders for a market.
        Each entry: account, position (YES/NO), size, value.
        """
        params: dict[str, Any] = {"market": market, "limit": limit}
        result = self._get("/holders", params)
        return result if isinstance(result, list) else []

    def get_activity(
        self,
        user: Optional[str] = None,
        market: Optional[str] = None,
        activity_type: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        On-chain activity for a user or market.

        Args:
            activity_type: "TRADE", "SPLIT", "MERGE", "REDEEM", "REWARD", "CONVERSION"
            start / end: Unix timestamp bounds

        Each entry: type, timestamp, market, outcome, amount, price, shares, tx_hash.
        """
        params: dict[str, Any] = {"limit": min(limit, 500)}
        if user:
            params["user"] = user
        if market:
            params["market"] = market
        if activity_type:
            params["type"] = activity_type
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        result = self._get("/activity", params)
        return result if isinstance(result, list) else []

    def get_positions(self, user: str, market: Optional[str] = None) -> list[dict]:
        """Open positions for a wallet address."""
        params: dict[str, Any] = {"user": user}
        if market:
            params["market"] = market
        result = self._get("/positions", params)
        return result if isinstance(result, list) else []
