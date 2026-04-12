"""
Kalshi WebSocket client — real-time ticker price feed.

Connects to the Kalshi Trade API v2 WebSocket, authenticates with RSA-PSS,
and subscribes to the `ticker` channel for specific market tickers.

Price callback receives (ticker: str, mid_price: float) where mid_price is
the midpoint of yes_ask and yes_bid converted from cents to [0, 1].

Dynamic subscriptions are thread-safe: call subscribe() from any thread
and new tickers will be sent to the server on the next loop iteration.
Existing subscriptions are re-sent automatically on reconnect.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Callable, Optional

import websockets

log = logging.getLogger(__name__)

WS_URL = "wss://trading-api.kalshi.com/trade-api/ws/v2"


class KalshiWSClient:
    """
    WebSocket client for Kalshi live ticker prices.

    Args:
        auth_headers_fn: Callable[[], dict[str, str]] — returns fresh RSA-PSS
            signed headers for a GET /trade-api/ws/v2 request.
            Typically: lambda: kalshi_client._auth_headers("GET", "/trade-api/ws/v2")
        on_price: Callable[[str, str, float], None] — called with
            (ticker, side, ask_price) where side is "yes" or "no" and
            ask_price is the ask in [0, 1]. Fired twice per ticker update
            (once for YES, once for NO). May be called from the WS thread.
    """

    def __init__(
        self,
        auth_headers_fn: Callable[[], dict[str, str]],
        on_price: Callable[[str, str, float], None],
    ) -> None:
        self._auth_headers_fn = auth_headers_fn
        self._on_price = on_price
        self._subscribed: set[str] = set()
        self._pending: list[list[str]] = []     # buffered before loop starts
        self._subscribe_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the WebSocket loop in a daemon thread."""
        self._loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_loop, daemon=True, name="kalshi-ws")
        t.start()

    def subscribe(self, tickers: list[str]) -> None:
        """Thread-safe: request subscription to additional tickers."""
        new = [t for t in tickers if t not in self._subscribed]
        if not new:
            return
        if self._loop is None or self._subscribe_queue is None:
            self._pending.append(new)
            return
        asyncio.run_coroutine_threadsafe(
            self._subscribe_queue.put(new), self._loop
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop.run_until_complete(self._run())

    async def _run(self) -> None:
        self._subscribe_queue = asyncio.Queue()
        # Drain any pre-start subscribe() calls
        for batch in self._pending:
            await self._subscribe_queue.put(batch)
        self._pending.clear()

        while True:
            try:
                headers = self._auth_headers_fn()
                async with websockets.connect(WS_URL, additional_headers=headers) as ws:
                    log.info("Kalshi WS connected")

                    # Re-subscribe everything after a reconnect
                    if self._subscribed:
                        await ws.send(json.dumps({
                            "id": 1, "cmd": "subscribe",
                            "params": {
                                "channels": ["ticker"],
                                "market_tickers": list(self._subscribed),
                            },
                        }))

                    recv_task = asyncio.create_task(self._recv_loop(ws))
                    send_task = asyncio.create_task(self._send_loop(ws))
                    done, pending = await asyncio.wait(
                        [recv_task, send_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    for t in done:
                        exc = t.exception()
                        if exc:
                            log.warning("Kalshi WS task error: %s", exc)

            except Exception as exc:
                log.warning("Kalshi WS error: %s — reconnecting in 5s", exc)

            await asyncio.sleep(5)

    async def _send_loop(self, ws) -> None:
        """Process subscription queue and send periodic pings."""
        msg_id = 2  # 1 is used for the initial re-subscribe on connect
        while True:
            try:
                batch = await asyncio.wait_for(self._subscribe_queue.get(), timeout=20)
                new = [t for t in batch if t not in self._subscribed]
                if new:
                    self._subscribed.update(new)
                    msg_id += 1
                    await ws.send(json.dumps({
                        "id": msg_id,
                        "cmd": "subscribe",
                        "params": {"channels": ["ticker"], "market_tickers": new},
                    }))
                    log.debug("Kalshi WS subscribed to %d new tickers (total=%d)",
                              len(new), len(self._subscribed))
            except asyncio.TimeoutError:
                msg_id += 1
                await ws.send(json.dumps({"id": msg_id, "cmd": "ping"}))

    async def _recv_loop(self, ws) -> None:
        """Parse incoming messages and fire the price callback."""
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")
            if msg_type != "ticker":
                continue

            data = msg.get("msg", {})
            ticker = data.get("market_ticker")
            yes_ask = data.get("yes_ask")
            no_ask = data.get("no_ask")
            if ticker is None:
                continue

            try:
                if yes_ask is not None:
                    self._on_price(ticker, "yes", yes_ask / 100.0)
                if no_ask is not None:
                    self._on_price(ticker, "no", no_ask / 100.0)
            except Exception as exc:
                log.debug("on_price callback error: %s", exc)
