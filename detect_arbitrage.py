"""
Streaming arbitrage detector.

On startup, bootstraps a faiss index from all markets already in the DB.
Then runs Kalshi and Polymarket pollers in background threads. For each new
market that arrives, it:
  1. Embeds the question with all-MiniLM-L6-v2
  2. Searches the faiss index for K nearest neighbors (any platform)
  3. Passes the new market + neighbors to an LLM in a single call
  4. Filters LLM output for cross-platform same-outcome pairs with spread > threshold
  5. Writes confirmed pairs to the arbitrage_pairs table and prints an alert

Usage:
    python detect_arbitrage.py [--db PATH] [--interval SECONDS] [--duration HOURS]
                               [--spread THRESHOLD] [--lookback SECONDS] [--k K]

LLM is configured via env vars (or .env):
    LLM_MODEL     model name (default: gemini-2.0-flash)
    LLM_API_KEY   API key
    LLM_BASE_URL  base URL for OpenAI-compatible endpoint (optional)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time
from typing import Optional

import pandas as pd
import numpy as np

import config
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketGammaClient
from db.models import init_db, ArbitragePair
from forecasting.embedder import embed_questions
from forecasting.loader import load_markets
from forecasting.llm import OpenAICompatibleBackend
from forecasting.relationships import _discover_pairs_in_group
from stream_markets import poll_kalshi, poll_polymarket, _ts

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

ENTRY_SPREAD_THRESHOLD = 0.04
K_NEIGHBORS = 10
MIN_CONFIDENCE = 0.5


# ── Price / question / id extraction from raw API dicts ──────────────────────

def _get_price(market: dict, platform: str) -> Optional[float]:
    if platform == "kalshi":
        v = market.get("yes_ask")
        return v / 100.0 if v is not None else None
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


# ── DB write ──────────────────────────────────────────────────────────────────

def _upsert_pair(sf, row_a: dict, row_b: dict, spread: float, conf: float,
                 category: str, rationale: str) -> None:
    pair = ArbitragePair(
        id_a=row_a["id"],          platform_a=row_a["platform"],
        question_a=row_a["question"], price_a=row_a.get("price_yes"),
        id_b=row_b["id"],          platform_b=row_b["platform"],
        question_b=row_b["question"], price_b=row_b.get("price_yes"),
        spread=spread,
        confidence_score=conf,
        category=category,
        rationale=rationale,
    )
    with sf() as session:
        session.merge(pair)
        session.commit()


# ── Alert formatting ──────────────────────────────────────────────────────────

def _print_opportunity(row_a: dict, row_b: dict, spread: float, conf: float) -> None:
    if (row_a.get("price_yes") or 1) <= (row_b.get("price_yes") or 0):
        buy, sell = row_a, row_b
    else:
        buy, sell = row_b, row_a

    print(
        f"\n[{_ts()}] *** ARBITRAGE  spread={spread:.1%}  conf={conf:.2f} ***\n"
        f"  BUY YES   {buy['platform'].upper():<12} @ {buy.get('price_yes', '?'):.3f}"
        f"  {buy['question'][:80]}\n"
        f"  SELL YES  {sell['platform'].upper():<12} @ {sell.get('price_yes', '?'):.3f}"
        f"  {sell['question'][:80]}\n",
        flush=True,
    )


# ── Detector ──────────────────────────────────────────────────────────────────

class StreamingDetector:
    """
    Thread-safe streaming arbitrage detector backed by a faiss index.

    For each new market, builds a group of the new market + K nearest neighbors,
    sends the group to an LLM in one call, then filters the response for
    cross-platform same-outcome pairs with sufficient spread.

    AsyncOpenAI uses httpx.AsyncClient internally, which is NOT safe to share
    across threads with separate event loops. A thread-local backend is created
    per poller thread so each gets its own httpx connection pool.
    """

    def __init__(
        self,
        backend_factory,
        sf,
        spread_threshold: float = ENTRY_SPREAD_THRESHOLD,
        k: int = K_NEIGHBORS,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        # backend_factory() must return a fresh LLMBackend — called once per thread
        self._backend_factory = backend_factory
        self._thread_local = threading.local()
        self._sf = sf
        self._spread_threshold = spread_threshold
        self._k = k
        self._min_confidence = min_confidence
        self._lock = threading.Lock()
        # Serialise LLM calls across all poller threads — prevents both threads
        # from firing simultaneously and blowing the shared Groq TPM limit.
        self._llm_sem = threading.Semaphore(1)
        self._index = None
        self._meta: list[dict] = []
        self._model = None

    def _get_backend(self):
        """Return a thread-local LLMBackend instance (one per poller thread)."""
        if not hasattr(self._thread_local, "backend"):
            self._thread_local.backend = self._backend_factory()
        return self._thread_local.backend

    def bootstrap(self, db_path: str) -> None:
        """Load all existing markets, embed them, build the faiss index."""
        import faiss
        from sentence_transformers import SentenceTransformer

        device = os.getenv("EMBEDDER_DEVICE", "cpu")
        print(f"[{_ts()}] Bootstrapping index from {db_path} (device={device}) ...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        sf = init_db(db_path)
        df = load_markets(sf, active_only=True)

        d = 384  # all-MiniLM-L6-v2 output dimension
        self._index = faiss.IndexFlatIP(d)

        if df.empty:
            print(f"[{_ts()}] DB empty — starting with empty index")
            self._meta = []
            return

        embeddings = embed_questions(df, model=self._model)
        self._index.add(embeddings.astype(np.float32))
        self._meta = df.to_dict("records")
        print(f"[{_ts()}] Index ready — {self._index.ntotal:,} vectors")

    def process(self, market: dict, platform: str) -> None:
        """Called for each new market from the poller threads."""
        question = _get_question(market, platform)
        market_id = _get_id(market, platform)
        price = _get_price(market, platform)

        if not question or not market_id:
            return

        emb = self._model.encode([question], normalize_embeddings=True).astype(np.float32)

        # Build group and update index under lock (fast ops only)
        with self._lock:
            new_row = {"id": market_id, "platform": platform,
                       "question": question, "price_yes": price}
            group = [new_row]

            n_total = self._index.ntotal
            if n_total > 0:
                k = min(self._k, n_total)
                _, indices = self._index.search(emb, k)
                for idx in indices[0]:
                    if 0 <= idx < len(self._meta):
                        group.append(self._meta[idx])

            self._index.add(emb)
            self._meta.append(new_row)

        if len(group) < 2:
            return

        # LLM call outside the index lock — serialised via semaphore so both
        # poller threads never hit Groq simultaneously (shared TPM budget).
        group_df = pd.DataFrame(group)
        with self._llm_sem:
            try:
                category, raw_pairs = asyncio.run(
                    _discover_pairs_in_group(self._get_backend(), group_df)
                )
            except Exception as exc:
                logging.getLogger(__name__).warning("LLM call failed: %s", exc)
                return

            q_to_row = {r["question"]: r for r in group}

            for pair in raw_pairs:
                if not pair.get("is_same_outcome"):
                    continue
                conf = float(pair.get("confidence_score", 0))
                if conf < self._min_confidence:
                    continue

                row_a = q_to_row.get(pair.get("question_a", ""))
                row_b = q_to_row.get(pair.get("question_b", ""))
                if row_a is None or row_b is None:
                    continue
                if row_a["platform"] == row_b["platform"]:
                    continue  # same platform — not actionable

                price_a = row_a.get("price_yes")
                price_b = row_b.get("price_yes")
                if price_a is None or price_b is None:
                    continue

                spread = abs(price_a - price_b)
                if spread < self._spread_threshold:
                    continue

                _upsert_pair(self._sf, row_a, row_b, spread, conf,
                             category, pair.get("rationale", ""))
                _print_opportunity(row_a, row_b, spread, conf)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming arbitrage detector")
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--duration", type=float, default=0,
                        help="Stop after N hours (0 = run forever)")
    parser.add_argument("--spread", type=float, default=ENTRY_SPREAD_THRESHOLD,
                        help=f"Min spread to alert (default: {ENTRY_SPREAD_THRESHOLD})")
    parser.add_argument("--lookback", type=int, default=300,
                        help="Seconds back to look for new markets per poll (default: 300)")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS,
                        help=f"Nearest neighbors per new market (default: {K_NEIGHBORS})")
    args = parser.parse_args()

    def backend_factory():
        return OpenAICompatibleBackend(
            model=config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            rpm_limit=30,
        )

    sf = init_db(args.db)
    detector = StreamingDetector(backend_factory, sf, spread_threshold=args.spread, k=args.k)
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
                SessionFactory=sf,
                lookback=args.lookback,
                on_new_market=lambda m: detector.process(m, "kalshi"),
                seed_first_pass=True,
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
                SessionFactory=sf,
                lookback=args.lookback,
                on_new_market=lambda m: detector.process(m, "polymarket"),
                seed_first_pass=True,
            ),
            daemon=True,
            name="polymarket-poller",
        ),
    ]

    print(
        f"Streaming arbitrage detector  |  model={config.LLM_MODEL} (Groq)"
        f"  |  spread>={args.spread:.0%}  |  k={args.k}"
        f"  |  poll={args.interval}s  |  Ctrl+C to stop\n"
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
