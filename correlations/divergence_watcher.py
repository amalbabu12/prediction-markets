"""WebSocket-driven divergence watcher.

Subscribes to live YES prices for every market in `correlated_pairs` and, on
every price tick, recomputes the spread (price_b_yes − price_a_yes) and its
z-score against the baseline (mean_spread, std_spread) captured by the
correlator. When |z| ≥ DIVERGENCE_Z_THRESHOLD it logs a row in
price_divergences.

We only watch YES prices — for correlation purposes the NO side is just 1−YES,
and halving the subscription set makes Polymarket's WS quieter.

Mirrors detect_arbitrage.ArbitrageWatcher's threading model: WS threads push
ticks to `_on_price`, the main thread re-syncs subscriptions every
`sync_interval` seconds.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Optional

from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

import config
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketCLOBClient
from clients.ws_kalshi import KalshiWSClient
from clients.ws_polymarket import PolymarketWSClient
from db.models import init_db
from correlations import models  # noqa: F401  — register tables with Base
from correlations.config import DIVERGENCE_Z_THRESHOLD
from correlations.models import PriceDivergence

log = logging.getLogger("correlations.divergence_watcher")

SYNC_INTERVAL_SEC = 60


class CorrelationWatcher:
    def __init__(self, sf, kalshi_client: KalshiClient) -> None:
        self._sf = sf

        # Latest YES price keyed by market_id (Kalshi tickers and Polymarket condition_ids
        # do not collide because they use different alphabets, but we still namespace
        # internally to be safe).
        self._prices: dict[tuple[str, str], float] = {}    # (platform, market_id) -> price_yes
        self._prices_lock = threading.Lock()

        # Pair registry
        self._pairs: dict[tuple[str, str, str, str], dict] = {}  # (pa, ida, pb, idb) -> row
        self._market_to_pairs: dict[tuple[str, str], list[tuple[str, str, str, str]]] = {}
        self._pairs_lock = threading.Lock()

        # Polymarket token → condition_id map (so the WS callback can reverse-lookup)
        self._poly_token_to_cid: dict[str, str] = {}

        self._subscribed_kalshi: set[str] = set()
        self._subscribed_poly_tokens: set[str] = set()

        self._kalshi_ws = KalshiWSClient(
            auth_headers_fn=lambda: kalshi_client._auth_headers("GET", "/trade-api/ws/v2"),
            on_price=self._on_kalshi_price,
        )
        self._poly_ws = PolymarketWSClient(on_price=self._on_poly_price)

    # ── Price callbacks (called from WS threads) ─────────────────────────────

    def _on_kalshi_price(self, ticker: str, side: str, price: float) -> None:
        if side != "yes":
            return
        self._record_price("kalshi", ticker, price)

    def _on_poly_price(self, condition_id: str, side: str, price: float) -> None:
        if side != "yes":
            return
        self._record_price("polymarket", condition_id, price)

    def _record_price(self, platform: str, market_id: str, price: float) -> None:
        key = (platform, market_id)
        with self._prices_lock:
            self._prices[key] = price

        with self._pairs_lock:
            pair_keys = list(self._market_to_pairs.get(key, []))

        for pair_key in pair_keys:
            with self._pairs_lock:
                pair = self._pairs.get(pair_key)
            if pair is None:
                continue
            self._check_divergence(pair)

    def _check_divergence(self, pair: dict) -> None:
        with self._prices_lock:
            yes_a = self._prices.get((pair["platform_a"], pair["id_a"]))
            yes_b = self._prices.get((pair["platform_b"], pair["id_b"]))
        if yes_a is None or yes_b is None:
            return

        std = pair["std_spread"]
        if std is None or std <= 0:
            return

        spread = yes_b - yes_a
        z = (spread - pair["mean_spread"]) / std
        if abs(z) < DIVERGENCE_Z_THRESHOLD:
            return

        try:
            with self._sf() as session:
                stmt = sqlite_insert(PriceDivergence).values(
                    pair_id=pair["id"],
                    price_a=yes_a,
                    price_b=yes_b,
                    spread=spread,
                    z_score=z,
                )
                session.execute(stmt)
                session.commit()
            log.info("divergence pair_id=%d  spread=%+.4f  z=%+.2f  (a=%.4f b=%.4f)",
                     pair["id"], spread, z, yes_a, yes_b)
        except Exception as exc:
            log.warning("failed to insert divergence for pair %d: %s", pair["id"], exc)

    # ── Sync (subscribe to new pairs) ────────────────────────────────────────

    def _sync(self) -> None:
        with self._sf() as session:
            rows = session.execute(text("""
                SELECT cp.id, cp.id_a, cp.platform_a, cp.id_b, cp.platform_b,
                       cp.mean_spread, cp.std_spread,
                       pa.token_id_yes AS token_a,
                       pb.token_id_yes AS token_b
                FROM correlated_pairs cp
                LEFT JOIN polymarket_markets pa
                       ON cp.platform_a = 'polymarket' AND pa.condition_id = cp.id_a
                LEFT JOIN polymarket_markets pb
                       ON cp.platform_b = 'polymarket' AND pb.condition_id = cp.id_b
            """)).fetchall()

        new_kalshi: list[str] = []
        new_poly: list[str] = []
        new_token_to_market: dict[str, tuple[str, str]] = {}

        with self._pairs_lock:
            for r in rows:
                pair_key = (r.platform_a, r.id_a, r.platform_b, r.id_b)
                if pair_key in self._pairs:
                    continue

                pair = {
                    "id": r.id,
                    "id_a": r.id_a, "platform_a": r.platform_a,
                    "id_b": r.id_b, "platform_b": r.platform_b,
                    "mean_spread": r.mean_spread or 0.0,
                    "std_spread": r.std_spread,
                }
                self._pairs[pair_key] = pair

                for platform, mid in [(r.platform_a, r.id_a), (r.platform_b, r.id_b)]:
                    self._market_to_pairs.setdefault((platform, mid), []).append(pair_key)

                # Kalshi subscriptions
                for platform, mid in [(r.platform_a, r.id_a), (r.platform_b, r.id_b)]:
                    if platform == "kalshi" and mid not in self._subscribed_kalshi:
                        new_kalshi.append(mid)
                        self._subscribed_kalshi.add(mid)

                # Polymarket subscriptions — YES token only
                for platform, cid, token in [
                    (r.platform_a, r.id_a, r.token_a),
                    (r.platform_b, r.id_b, r.token_b),
                ]:
                    if platform != "polymarket":
                        continue
                    if not token:
                        continue
                    if token in self._subscribed_poly_tokens:
                        continue
                    new_poly.append(token)
                    self._subscribed_poly_tokens.add(token)
                    new_token_to_market[token] = (cid, "yes")
                    self._poly_token_to_cid[token] = cid

        if new_kalshi:
            self._kalshi_ws.subscribe(new_kalshi)
            log.info("subscribed to %d new Kalshi tickers", len(new_kalshi))
        if new_poly:
            self._poly_ws.subscribe(new_poly, new_token_to_market)
            log.info("subscribed to %d new Polymarket tokens", len(new_poly))

    def run(self) -> None:
        self._kalshi_ws.start()
        self._poly_ws.start()
        while True:
            try:
                self._sync()
            except Exception as exc:
                log.warning("sync error: %s", exc)
            time.sleep(SYNC_INTERVAL_SEC)


def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket-driven correlation divergence watcher.")
    parser.add_argument("--db", default="./data/markets.db")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sf = init_db(args.db)

    kalshi_client = KalshiClient(
        api_key_id=config.KALSHI_API_KEY_ID,
        private_key_pem=config.KALSHI_PRIVATE_KEY,
        rate_limit=config.KALSHI_RATE_LIMIT,
    )

    log.info("divergence_watcher started — z_threshold=%.2f  sync=%ds",
             DIVERGENCE_Z_THRESHOLD, SYNC_INTERVAL_SEC)

    watcher = CorrelationWatcher(sf, kalshi_client)
    watcher.run()


if __name__ == "__main__":
    main()
