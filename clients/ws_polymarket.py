"""
Polymarket CLOB WebSocket client — real-time market price feed.

Connects to the Polymarket CLOB WebSocket and subscribes to market updates
for specific asset IDs (ERC-1155 token IDs for YES outcomes).

The server sends three relevant event types:
  - "price_change"      : a trade or quote moved the price; has "price" field
  - "last_trade_price"  : price of the most recent trade; has "price" field
  - "book"              : full order book snapshot; has "bids" / "asks" arrays

Price callback receives (condition_id: str, price: float). The token_id →
condition_id mapping is maintained internally and updated via subscribe().

Dynamic subscriptions are thread-safe: call subscribe() from any thread.
Existing subscriptions are re-sent automatically on reconnect.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Callable, Optional

import websockets
import websockets.exceptions

log = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


def _book_top_price(levels: list) -> Optional[float]:
    """
    Return the price of the top level. Polymarket book entries may be either
    {"price": "...", "size": "..."} dicts or [price, size] arrays.
    """
    if not levels:
        return None
    top = levels[0]
    if isinstance(top, dict):
        raw = top.get("price")
    else:
        raw = top[0]
    return float(raw) if raw is not None else None


class PolymarketWSClient:
    """
    WebSocket client for Polymarket CLOB live market prices.

    Subscribes to both YES and NO token IDs for each market so that actual
    ask prices for both sides are available for arbitrage calculations.

    Args:
        on_price: Callable[[str, str, float], None] — called with
            (condition_id, side, price) where side is "yes" or "no".
            May be called from the WS thread.
    """

    def __init__(self, on_price: Callable[[str, str, float], None]) -> None:
        self._on_price = on_price
        # token_id → (condition_id, "yes"|"no")
        self._token_to_market: dict[str, tuple[str, str]] = {}
        self._subscribed: set[str] = set()        # token_ids currently subscribed
        self._pending: list[tuple[list[str], dict[str, tuple[str, str]]]] = []
        self._subscribe_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the WebSocket loop in a daemon thread."""
        self._loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_loop, daemon=True, name="poly-ws")
        t.start()

    def subscribe(
        self,
        token_ids: list[str],
        token_to_market: dict[str, tuple[str, str]],
    ) -> None:
        """
        Thread-safe: subscribe to additional token IDs and register the
        token_id → (condition_id, side) mapping used in price callbacks.

        token_to_market maps each token_id to (condition_id, "yes"|"no").
        """
        self._token_to_market.update(token_to_market)
        new = [t for t in token_ids if t not in self._subscribed]
        if not new:
            return
        if self._loop is None or self._subscribe_queue is None:
            self._pending.append((new, token_to_market))
            return
        asyncio.run_coroutine_threadsafe(
            self._subscribe_queue.put((new, token_to_market)), self._loop
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop.run_until_complete(self._run())

    async def _run(self) -> None:
        self._subscribe_queue = asyncio.Queue()
        for batch, mapping in self._pending:
            await self._subscribe_queue.put((batch, mapping))
        self._pending.clear()

        while True:
            try:
                async with websockets.connect(WS_URL) as ws:
                    log.info("Polymarket WS connected")

                    # Re-subscribe on reconnect
                    if self._subscribed:
                        await ws.send(json.dumps({
                            "assets_ids": list(self._subscribed),
                            "type": "market",
                        }))

                    recv_task = asyncio.create_task(self._recv_loop(ws), name="recv")
                    send_task = asyncio.create_task(self._send_loop(ws), name="send")
                    done, pending = await asyncio.wait(
                        [recv_task, send_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                    for t in done:
                        exc = t.exception()
                        if exc is None:
                            continue
                        if isinstance(exc, websockets.exceptions.ConnectionClosedOK):
                            log.info("Polymarket WS %s closed cleanly", t.get_name())
                        else:
                            log.warning(
                                "Polymarket WS %s task error: %s: %r",
                                t.get_name(), type(exc).__name__, exc,
                            )

            except Exception as exc:
                log.warning("Polymarket WS error: %s — reconnecting in 5s", exc)

            await asyncio.sleep(5)

    async def _send_loop(self, ws) -> None:
        """Process subscription queue and send periodic pings."""
        while True:
            try:
                batch, mapping = await asyncio.wait_for(
                    self._subscribe_queue.get(), timeout=10
                )
                self._token_to_market.update(mapping)
                new = [t for t in batch if t not in self._subscribed]
                if new:
                    self._subscribed.update(new)
                    await ws.send(json.dumps({
                        "assets_ids": new,
                        "type": "market",
                    }))
                    log.debug("Polymarket WS subscribed to %d new assets (total=%d)",
                              len(new), len(self._subscribed))
            except asyncio.TimeoutError:
                await ws.ping()

    def _handle_msg(self, msg: dict) -> None:
        """Process a single WS message dict."""
        event_type = msg.get("event_type", "")
        asset_id = msg.get("asset_id", "")

        price: Optional[float] = None

        if event_type in ("price_change", "last_trade_price"):
            raw_price = msg.get("price")
            if raw_price is not None:
                try:
                    price = float(raw_price)
                except (ValueError, TypeError):
                    pass

        elif event_type == "book":
            bids = msg.get("bids") or []
            asks = msg.get("asks") or []
            try:
                best_bid = _book_top_price(bids)
                best_ask = _book_top_price(asks)
                if best_bid is not None and best_ask is not None:
                    price = (best_bid + best_ask) / 2.0
            except (IndexError, KeyError, ValueError, TypeError):
                pass

        if price is None or not asset_id:
            return

        mapping = self._token_to_market.get(asset_id)
        if mapping:
            cid, side = mapping
        else:
            cid = msg.get("market") or asset_id
            side = "yes"
        try:
            self._on_price(cid, side, price)
        except Exception as exc:
            log.debug("on_price callback error: %s", exc)

    async def _recv_loop(self, ws) -> None:
        """Parse incoming messages and fire the price callback."""
        async for raw in ws:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Server may send a single dict or a list of dicts
            msgs = data if isinstance(data, list) else [data]
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                try:
                    self._handle_msg(msg)
                except Exception as exc:
                    log.warning(
                        "Polymarket WS handler error: %s: %r (msg keys=%s)",
                        type(exc).__name__, exc, list(msg.keys()),
                    )
