"""
Kalshi Trade API v2 client.

Public endpoints (no auth required):
  - Exchange status
  - Series list
  - Events (all statuses, paginated)
  - Markets (all statuses, paginated)
  - Market candlesticks (OHLCV) at 1m / 60m / 1440m resolution
  - Public trade history

Authenticated endpoints (requires api_key_id + RSA private key):
  - Orderbook snapshots
  - Portfolio / order management (not implemented here — read-only focus)

Auth uses RSA-PSS (SHA-256) over a timestamp + method + path string.
API keys are created at: https://kalshi.com/account/api
Rate limits: Basic tier = 20 req/s reads; we default to 10/s to be safe.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Any, Iterator, Optional

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)


class _RateLimiter:
    """Simple leaky-bucket rate limiter."""

    def __init__(self, calls_per_second: float) -> None:
        self._interval = 1.0 / calls_per_second
        self._last: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.monotonic()


class KalshiClient:
    """
    Kalshi Trade API v2 — read-focused client.

    Usage (public endpoints only):
        client = KalshiClient()
        markets = client.get_all_markets()

    Usage (with auth for orderbook):
        client = KalshiClient(
            api_key_id="your-uuid",
            private_key_pem=open("kalshi_private.pem").read(),
        )
        book = client.get_orderbook("INXW-23DEC29-B4800")
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_pem: Optional[str | bytes] = None,
        rate_limit: float = 10.0,
    ) -> None:
        self.api_key_id = api_key_id
        self._private_key = None

        if private_key_pem:
            raw = private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem
            self._private_key = serialization.load_pem_private_key(
                raw, password=None, backend=default_backend()
            )

        self._session = requests.Session()
        self._session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )
        self._rl = _RateLimiter(rate_limit)

    @property
    def is_authenticated(self) -> bool:
        return bool(self.api_key_id and self._private_key)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """RSA-PSS signed headers for authenticated requests."""
        if not self.is_authenticated:
            return {}
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode()
        sig = self._private_key.sign(  # type: ignore[union-attr]
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,  # type: ignore[dict-item]
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def _get(
        self,
        path: str,
        params: Optional[dict] = None,
        authenticated: bool = False,
        retries: int = 4,
    ) -> Any:
        self._rl.wait()
        url = self.BASE_URL + path
        headers = self._auth_headers("GET", path) if authenticated else {}

        for attempt in range(retries):
            try:
                resp = self._session.get(url, params=params, headers=headers, timeout=30)

                if resp.status_code == 429:
                    backoff = 2 ** (attempt + 1)
                    logger.warning("Kalshi 429 rate limit. Sleeping %ds.", backoff)
                    time.sleep(backoff)
                    continue

                # Some endpoints silently require auth — retry with credentials.
                if resp.status_code == 401 and not authenticated and self.is_authenticated:
                    logger.debug("401 on %s — retrying with auth headers.", path)
                    headers = self._auth_headers("GET", path)
                    authenticated = True
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as exc:
                if attempt == retries - 1:
                    logger.error("Kalshi request failed (%s): %s", path, exc)
                    raise
                backoff = 2 ** attempt
                logger.warning("Kalshi error attempt %d/%d: %s. Retry in %ds.", attempt + 1, retries, exc, backoff)
                time.sleep(backoff)

        return {}

    # ── Exchange ─────────────────────────────────────────────────────────────

    def get_exchange_status(self) -> dict:
        """Current exchange trading status (trading_active, exchange_active)."""
        return self._get("/exchange/status")

    # ── Series ───────────────────────────────────────────────────────────────

    def get_series_list(
        self,
        category: Optional[str] = None,
        tags: Optional[str] = None,
        include_volume: bool = True,
    ) -> list[dict]:
        """
        All series (market groups like INXW, KXBTC, US-PRES, etc.).
        Each series contains: ticker, title, category, frequency, tags.
        """
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        if tags:
            params["tags"] = tags
        if include_volume:
            params["include_volume"] = "true"
        data = self._get("/series", params)
        return data.get("series", [])

    def get_series(self, series_ticker: str) -> dict:
        """Details for a single series."""
        return self._get(f"/series/{series_ticker}")

    # ── Events ───────────────────────────────────────────────────────────────

    def iter_events(
        self,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        with_nested_markets: bool = False,
        min_close_ts: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Paginate through events.

        Args:
            status: "open" | "closed" | "settled" — omit for all
            series_ticker: Filter to a specific series
            with_nested_markets: Embed Market objects in each Event (bigger payload)
            min_close_ts: Unix timestamp — only return events closing after this time
        """
        cursor: Optional[str] = None
        page = 0
        while True:
            params: dict[str, Any] = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            if status:
                params["status"] = status
            if series_ticker:
                params["series_ticker"] = series_ticker
            if with_nested_markets:
                params["with_nested_markets"] = "true"
            if min_close_ts:
                params["min_close_ts"] = min_close_ts

            data = self._get("/events", params)
            events: list[dict] = data.get("events", [])
            page += 1
            logger.debug("Kalshi events page %d: %d results (status=%s).", page, len(events), status)
            yield from events

            cursor = data.get("cursor") or None
            if not cursor or not events:
                break

    def get_all_events(
        self,
        statuses: tuple[str, ...] = ("open", "closed", "settled"),
        **kwargs: Any,
    ) -> list[dict]:
        """Collect events for all specified statuses."""
        results: list[dict] = []
        for status in statuses:
            results.extend(self.iter_events(status=status, **kwargs))
        return results

    def get_event(self, event_ticker: str, with_nested_markets: bool = True) -> dict:
        """Single event by ticker."""
        params: dict[str, Any] = {}
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        return self._get(f"/events/{event_ticker}", params)

    # ── Markets ───────────────────────────────────────────────────────────────

    def iter_markets(
        self,
        status: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        max_close_ts: Optional[int] = None,
        min_settled_ts: Optional[int] = None,
        max_settled_ts: Optional[int] = None,
        min_created_ts: Optional[int] = None,
        max_created_ts: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Paginate through markets.

        Args:
            status: "unopened" | "open" | "paused" | "closed" | "settled"
                    Omit to get all (but must loop per status — API doesn't support multi-status).

        Each market contains:
            ticker, event_ticker, series_ticker, title, subtitle, status,
            market_type (binary/scalar), yes/no bid/ask (cents), volume,
            open_interest, close_time, result, settlement_value, rules, etc.
        """
        cursor: Optional[str] = None
        page = 0
        while True:
            params: dict[str, Any] = {"limit": 1000}
            if cursor:
                params["cursor"] = cursor
            if status:
                params["status"] = status
            if event_ticker:
                params["event_ticker"] = event_ticker
            if series_ticker:
                params["series_ticker"] = series_ticker
            if min_close_ts:
                params["min_close_ts"] = min_close_ts
            if max_close_ts:
                params["max_close_ts"] = max_close_ts
            if min_settled_ts:
                params["min_settled_ts"] = min_settled_ts
            if max_settled_ts:
                params["max_settled_ts"] = max_settled_ts
            if min_created_ts:
                params["min_created_ts"] = min_created_ts
            if max_created_ts:
                params["max_created_ts"] = max_created_ts

            data = self._get("/markets", params)
            markets: list[dict] = data.get("markets", [])
            page += 1
            logger.debug("Kalshi markets page %d: %d results (status=%s).", page, len(markets), status)
            yield from markets

            cursor = data.get("cursor") or None
            if not cursor or not markets:
                break

    def get_all_markets(
        self,
        statuses: tuple[str, ...] = ("unopened", "open", "paused", "closed", "settled"),
        **kwargs: Any,
    ) -> list[dict]:
        """Collect markets for all specified statuses (deduped by ticker)."""
        seen: set[str] = set()
        results: list[dict] = []
        for status in statuses:
            for m in self.iter_markets(status=status, **kwargs):
                t = m.get("ticker", "")
                if t not in seen:
                    seen.add(t)
                    results.append(m)
        return results

    def get_market(self, ticker: str) -> dict:
        """Single market by ticker."""
        data = self._get(f"/markets/{ticker}")
        return data.get("market", data)

    # ── Orderbook ─────────────────────────────────────────────────────────────

    def get_orderbook(self, ticker: str, depth: int = 0) -> dict:
        """
        Current limit order book for a market.

        Args:
            ticker: Market ticker (e.g. "INXW-23DEC29-B4800")
            depth: 0 = all price levels; 1-100 = limit to N levels per side

        Returns:
            {
              "yes": [[price_cents, quantity], ...],  # bids for YES
              "no":  [[price_cents, quantity], ...],  # bids for NO
            }
        Note: Requires authentication on some Kalshi accounts. Client will
        automatically retry with auth headers if a 401 is received.
        """
        params: dict[str, Any] = {}
        if depth > 0:
            params["depth"] = depth
        data = self._get(
            f"/markets/{ticker}/orderbook",
            params,
            authenticated=self.is_authenticated,
        )
        return data.get("orderbook", data)

    # ── Trades ────────────────────────────────────────────────────────────────

    def iter_trades(
        self,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Paginate through public trade history.

        Each trade: trade_id, ticker, yes_price (cents), no_price (cents),
        count (contracts), taker_side ("yes"/"no"), created_time (ISO 8601).

        Args:
            ticker: Filter to a specific market
            min_ts / max_ts: Unix timestamp bounds (seconds)
        """
        cursor: Optional[str] = None
        page = 0
        while True:
            params: dict[str, Any] = {"limit": 1000}
            if cursor:
                params["cursor"] = cursor
            if ticker:
                params["ticker"] = ticker
            if min_ts:
                params["min_ts"] = min_ts
            if max_ts:
                params["max_ts"] = max_ts

            data = self._get("/markets/trades", params)
            trades: list[dict] = data.get("trades", [])
            page += 1
            logger.debug("Kalshi trades page %d: %d results.", page, len(trades))
            yield from trades

            cursor = data.get("cursor") or None
            if not cursor or not trades:
                break

    def get_trades(self, **kwargs: Any) -> list[dict]:
        return list(self.iter_trades(**kwargs))

    # ── Candlesticks ──────────────────────────────────────────────────────────

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[dict]:
        """
        OHLCV candlestick data for a market.

        Args:
            series_ticker: Parent series ticker (required in the URL path)
            ticker: Market ticker
            start_ts: Range start (Unix seconds)
            end_ts: Range end (Unix seconds)
            period_interval: Candle width in minutes. Must be 1, 60, or 1440.

        Each candle:
            end_period_ts (int), volume (int), open_interest (int),
            yes_bid: {open, high, low, close} (cents),
            yes_ask: {open, high, low, close} (cents),
            price:   {open, high, low, close} (cents — midpoint)
        """
        if period_interval not in (1, 60, 1440):
            raise ValueError(f"period_interval must be 1, 60, or 1440 — got {period_interval}")
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        data = self._get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params,
        )
        return data.get("candlesticks", [])

    def get_full_candlestick_history(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
    ) -> dict[int, list[dict]]:
        """
        Fetch all three resolutions of candles for a market.
        Returns {1: [...], 60: [...], 1440: [...]}.
        """
        result: dict[int, list[dict]] = {}
        for interval in (1, 60, 1440):
            try:
                candles = self.get_candlesticks(series_ticker, ticker, start_ts, end_ts, interval)
                result[interval] = candles
                logger.debug(
                    "Kalshi %dm candles for %s: %d bars.", interval, ticker, len(candles)
                )
            except Exception as exc:
                logger.warning("Failed %dm candles for %s: %s", interval, ticker, exc)
                result[interval] = []
        return result

    def get_event_candlesticks(
        self,
        event_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[dict]:
        """Event-level candlesticks (aggregate across all markets in the event)."""
        if period_interval not in (1, 60, 1440):
            raise ValueError(f"period_interval must be 1, 60, or 1440 — got {period_interval}")
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        data = self._get(f"/events/{event_ticker}/candlesticks", params)
        return data.get("candlesticks", [])
