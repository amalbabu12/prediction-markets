"""
Streaming arbitrage detector.

On startup, bootstraps a faiss index from all markets already in the DB.
Then runs Kalshi and Polymarket pollers in background threads. For each new
market that arrives, it:
  1. Embeds the question with all-MiniLM-L6-v2
  2. Searches the faiss index for K nearest cross-platform neighbors
  3. Runs NLI (cross-encoder/nli-deberta-v3-large) on each candidate pair
  4. Prints an alert if is_same_outcome=True and spread > threshold

Usage:
    python detect_arbitrage.py [--db PATH] [--interval SECONDS] [--duration HOURS]
                               [--spread THRESHOLD] [--lookback SECONDS] [--k K]
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Optional

import numpy as np

import config
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketGammaClient
from db.models import init_db
from forecasting.embedder import embed_questions
from forecasting.loader import load_markets
from forecasting.nli import NLIClassifier
from stream_markets import poll_kalshi, poll_polymarket, _ts

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

ENTRY_SPREAD_THRESHOLD = 0.04
K_NEIGHBORS = 10


# ── Price extraction from raw API dicts ───────────────────────────────────────

def _get_price(market: dict, platform: str) -> Optional[float]:
    if platform == "kalshi":
        v = market.get("yes_ask")
        return v / 100.0 if v is not None else None
    # polymarket
    prices = market.get("outcomePrices") or []
    try:
        return float(prices[0]) if prices else None
    except (ValueError, TypeError):
        return None


def _get_question(market: dict, platform: str) -> str:
    if platform == "kalshi":
        q = market.get("title", "")
        if market.get("subtitle"):
            q = f"{q} — {market['subtitle']}"
        return q.strip()
    return (market.get("question") or "").strip()


def _get_id(market: dict, platform: str) -> str:
    if platform == "kalshi":
        return market.get("ticker", "")
    return market.get("conditionId", "")


# ── Alert formatting ──────────────────────────────────────────────────────────

def _print_opportunity(
    new_id: str,
    new_platform: str,
    new_question: str,
    new_price: float,
    neighbor: dict,
    n_price: float,
    spread: float,
    confidence: float,
) -> None:
    if new_price <= n_price:
        buy_platform, buy_q, buy_price = new_platform, new_question, new_price
        sell_platform, sell_q, sell_price = neighbor["platform"], neighbor["question"], n_price
    else:
        buy_platform, buy_q, buy_price = neighbor["platform"], neighbor["question"], n_price
        sell_platform, sell_q, sell_price = new_platform, new_question, new_price

    print(
        f"\n[{_ts()}] *** ARBITRAGE  spread={spread:.1%}  conf={confidence:.2f} ***\n"
        f"  BUY YES   {buy_platform.upper():<12} @ {buy_price:.3f}  {buy_q[:80]}\n"
        f"  SELL YES  {sell_platform.upper():<12} @ {sell_price:.3f}  {sell_q[:80]}\n",
        flush=True,
    )


# ── Detector ──────────────────────────────────────────────────────────────────

class StreamingDetector:
    """
    Thread-safe streaming arbitrage detector backed by a faiss index.

    The faiss index and metadata list are protected by a single lock so
    concurrent calls from the Kalshi and Polymarket poller threads are safe.
    """

    def __init__(self, spread_threshold: float = ENTRY_SPREAD_THRESHOLD, k: int = K_NEIGHBORS):
        self._lock = threading.Lock()
        self._nli = NLIClassifier()
        self._index = None
        self._meta: list[dict] = []   # parallel to faiss positions
        self._model = None            # SentenceTransformer, loaded at bootstrap
        self._spread_threshold = spread_threshold
        self._k = k

    def bootstrap(self, db_path: str) -> None:
        """Load all existing markets, embed them, build the faiss index."""
        import faiss
        from sentence_transformers import SentenceTransformer

        print(f"[{_ts()}] Bootstrapping index from {db_path} ...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

        sf = init_db(db_path)
        df = load_markets(sf)

        d = 384  # all-MiniLM-L6-v2 output dimension
        self._index = faiss.IndexFlatIP(d)

        if df.empty:
            print(f"[{_ts()}] DB empty — starting with empty index")
            self._meta = []
            return

        embeddings = embed_questions(df, model=self._model)
        self._index.add(embeddings.astype(np.float32))
        self._meta = df.to_dict("records")
        print(f"[{_ts()}] Index ready — {self._index.ntotal:,} vectors ({len(df):,} markets)")

    def process(self, market: dict, platform: str) -> None:
        """Called for each new market. Embeds, searches, classifies, alerts."""
        question = _get_question(market, platform)
        market_id = _get_id(market, platform)
        price = _get_price(market, platform)

        if not question or not market_id:
            return

        emb = self._model.encode([question], normalize_embeddings=True).astype(np.float32)

        with self._lock:
            n_total = self._index.ntotal

            if n_total > 0:
                k = min(self._k, n_total)
                _, indices = self._index.search(emb, k)

                for idx in indices[0]:
                    if idx < 0 or idx >= len(self._meta):
                        continue
                    neighbor = self._meta[idx]
                    if neighbor["platform"] == platform:
                        continue

                    is_same, conf = self._nli.classify(question, neighbor["question"])
                    if not is_same:
                        continue

                    n_price = neighbor.get("price_yes")
                    if price is None or n_price is None:
                        continue

                    spread = abs(price - n_price)
                    if spread >= self._spread_threshold:
                        _print_opportunity(
                            market_id, platform, question, price,
                            neighbor, n_price, spread, conf,
                        )

            # Add new market to the index
            self._index.add(emb)
            self._meta.append({
                "id": market_id,
                "platform": platform,
                "question": question,
                "price_yes": price,
            })


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming arbitrage detector")
    parser.add_argument("--db", default="./data/markets.db",
                        help="SQLite DB path (default: ./data/markets.db)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds (default: 30)")
    parser.add_argument("--duration", type=float, default=0,
                        help="Stop after N hours (default: 0 = run forever)")
    parser.add_argument("--spread", type=float, default=ENTRY_SPREAD_THRESHOLD,
                        help=f"Min spread to alert (default: {ENTRY_SPREAD_THRESHOLD})")
    parser.add_argument("--lookback", type=int, default=300,
                        help="Seconds back to look for new markets per poll (default: 300)")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS,
                        help=f"Nearest neighbors to check per new market (default: {K_NEIGHBORS})")
    args = parser.parse_args()

    detector = StreamingDetector(spread_threshold=args.spread, k=args.k)
    detector.bootstrap(args.db)

    deadline = (
        time.monotonic() + args.duration * 3600
        if args.duration > 0
        else float("inf")
    )

    kalshi_client = KalshiClient(
        api_key_id=config.KALSHI_API_KEY_ID,
        private_key_pem=config.KALSHI_PRIVATE_KEY,
        rate_limit=config.KALSHI_RATE_LIMIT,
    )
    gamma_client = PolymarketGammaClient(rate_limit=config.POLYMARKET_RATE_LIMIT)

    threads = [
        threading.Thread(
            target=poll_kalshi,
            kwargs=dict(
                client=kalshi_client,
                interval=args.interval,
                deadline=deadline,
                lookback=args.lookback,
                on_new_market=lambda m: detector.process(m, "kalshi"),
            ),
            daemon=True,
            name="kalshi-poller",
        ),
        threading.Thread(
            target=poll_polymarket,
            kwargs=dict(
                client=gamma_client,
                interval=args.interval,
                deadline=deadline,
                lookback=args.lookback,
                on_new_market=lambda m: detector.process(m, "polymarket"),
            ),
            daemon=True,
            name="polymarket-poller",
        ),
    ]

    print(
        f"Streaming arbitrage detector  |  spread>={args.spread:.0%}"
        f"  |  k={args.k}  |  poll={args.interval}s  |  Ctrl+C to stop\n"
    )

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
        print("\nDone.")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
