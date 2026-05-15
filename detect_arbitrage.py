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
import math
import os
import queue
import threading
import time
from typing import Optional

import pandas as pd
import numpy as np

import config
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketGammaClient
from clients.ws_kalshi import KalshiWSClient
from clients.ws_polymarket import PolymarketWSClient
from db.models import init_db, ArbitragePair, WatchedPair
from forecasting.embedder import embed_questions
from forecasting.loader import load_markets
from forecasting.llm import OpenAICompatibleBackend
from forecasting.relationships import _discover_pairs_in_group, _discover_pairs_in_batch
from stream_markets import poll_kalshi, poll_polymarket, _ts
from correlations.tagger import KALSHI_SKIP_PREFIXES

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

ENTRY_SPREAD_THRESHOLD = 0.04
K_NEIGHBORS = 10
MIN_CONFIDENCE = 0.8
# Minimum cosine similarity for a cross-platform neighbor to trigger an LLM call.
# IndexFlatIP on L2-normalised vectors returns dot product = cosine similarity.
MIN_CROSS_PLATFORM_SIM = 0.55
# Number of new-market anchors batched into a single LLM call. Larger = fewer
# calls but bigger prompts and higher risk of the model dropping a group from
# its response.
LLM_BATCH_SIZE = 5


# ── Price / question / id extraction from raw API dicts ──────────────────────

def _get_prices(market: dict, platform: str) -> tuple[Optional[float], Optional[float]]:
    """Return (yes_ask, no_ask) as decimals in [0, 1]."""
    if platform == "kalshi":
        yes = market.get("yes_ask")
        no = market.get("no_ask")
        return (
            yes / 100.0 if yes is not None else None,
            no / 100.0 if no is not None else None,
        )
    prices = market.get("outcomePrices") or []
    try:
        yes = float(prices[0]) if len(prices) > 0 else None
        no = float(prices[1]) if len(prices) > 1 else None
        if yes is not None and math.isnan(yes):
            yes = None
        if no is not None and math.isnan(no):
            no = None
        return (yes, no)
    except (ValueError, TypeError):
        return (None, None)


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


# ── Arbitrage math ────────────────────────────────────────────────────────────

def compute_arb(
    yes_a: float, no_a: float,
    yes_b: float, no_b: float,
    is_same_outcome: bool,
) -> tuple[float, str]:
    """
    Compute the risk-free arbitrage profit using actual ask prices.

    Returns (profit, strategy) where profit > 0 means a real arb exists.

    Entailment (is_same_outcome=True):
        Both markets resolve the same way (A=YES ↔ B=YES).
        Strategy: buy YES on the cheap side, buy NO on the expensive side.
        Guaranteed payout = $1. Profit = $1 - cost.

    Contradiction (is_same_outcome=False):
        Markets resolve opposite (A=YES ↔ B=NO).
        Strategy: buy the same side on both (YES+YES or NO+NO).
        Guaranteed payout = $1. Profit = $1 - cost.
    """
    if is_same_outcome:
        # buy YES A + NO B  vs  buy YES B + NO A
        profit_ab = 1.0 - yes_a - no_b
        profit_ba = 1.0 - yes_b - no_a
        if profit_ab >= profit_ba:
            return (profit_ab,
                    f"BUY YES {{}}.A @ {yes_a:.3f} + BUY NO {{}}.B @ {no_b:.3f}")
        return (profit_ba,
                f"BUY YES {{}}.B @ {yes_b:.3f} + BUY NO {{}}.A @ {no_a:.3f}")
    else:
        # buy YES on both  vs  buy NO on both
        profit_yes = 1.0 - yes_a - yes_b
        profit_no = 1.0 - no_a - no_b
        if profit_yes >= profit_no:
            return (profit_yes,
                    f"BUY YES {{}}.A @ {yes_a:.3f} + BUY YES {{}}.B @ {yes_b:.3f}")
        return (profit_no,
                f"BUY NO {{}}.A @ {no_a:.3f} + BUY NO {{}}.B @ {no_b:.3f}")


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


def _upsert_watched_pair(
    sf,
    row_a: dict,
    row_b: dict,
    is_same_outcome: bool,
    conf: float,
    category: str,
    rationale: str,
) -> None:
    """Register a semantically related cross-platform pair for continuous price monitoring."""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert

    yes_a = row_a.get("price_yes")
    no_a = row_a.get("price_no")
    yes_b = row_b.get("price_yes")
    no_b = row_b.get("price_no")

    if all(v is not None for v in (yes_a, no_a, yes_b, no_b)):
        spread, _ = compute_arb(yes_a, no_a, yes_b, no_b, is_same_outcome)
    else:
        spread = None

    vals = dict(
        is_same_outcome=is_same_outcome,
        id_a=row_a["id"],            platform_a=row_a["platform"],
        question_a=row_a["question"],
        token_id_a=row_a.get("token_id_yes"),
        token_id_no_a=row_a.get("token_id_no"),
        price_yes_a=yes_a,           price_no_a=no_a,
        id_b=row_b["id"],            platform_b=row_b["platform"],
        question_b=row_b["question"],
        token_id_b=row_b.get("token_id_yes"),
        token_id_no_b=row_b.get("token_id_no"),
        price_yes_b=yes_b,           price_no_b=no_b,
        spread=spread, confidence_score=conf,
        category=category, rationale=rationale,
    )
    stmt = (
        sqlite_insert(WatchedPair)
        .values(**vals)
        .on_conflict_do_update(
            index_elements=["id_a", "id_b"],
            set_={
                "is_same_outcome": is_same_outcome,
                "token_id_a": vals["token_id_a"],
                "token_id_no_a": vals["token_id_no_a"],
                "token_id_b": vals["token_id_b"],
                "token_id_no_b": vals["token_id_no_b"],
                "price_yes_a": yes_a, "price_no_a": no_a,
                "price_yes_b": yes_b, "price_no_b": no_b,
                "spread": spread,
                "confidence_score": conf,
            },
        )
    )
    with sf() as session:
        session.execute(stmt)
        session.commit()


# ── Alert formatting ──────────────────────────────────────────────────────────

def _print_opportunity(
    row_a: dict, row_b: dict,
    profit: float, conf: float, strategy: str,
) -> None:
    plat_a = row_a["platform"].upper()
    plat_b = row_b["platform"].upper()
    # Fill platform names into the strategy template
    strategy_str = strategy.format(plat_a, plat_b)

    print(
        f"\n[{_ts()}] *** ARBITRAGE  profit={profit:.1%}  conf={conf:.2f} ***\n"
        f"  {strategy_str}\n"
        f"  A: {plat_a:<12} yes_ask={row_a.get('price_yes', 0):.3f}"
        f"  no_ask={row_a.get('price_no', 0):.3f}"
        f"  {row_a['question'][:70]}\n"
        f"  B: {plat_b:<12} yes_ask={row_b.get('price_yes', 0):.3f}"
        f"  no_ask={row_b.get('price_no', 0):.3f}"
        f"  {row_b['question'][:70]}\n",
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
        self._index = None
        self._meta: list[dict] = []
        self._model = None
        # LLM calls run in a single background worker thread so pollers never block.
        self._llm_queue: queue.Queue = queue.Queue()
        self._llm_worker = threading.Thread(target=self._llm_loop, daemon=True, name="llm-worker")
        self._llm_worker.start()

    def _get_backend(self):
        """Return a thread-local LLMBackend instance (one per worker thread)."""
        if not hasattr(self._thread_local, "backend"):
            self._thread_local.backend = self._backend_factory()
        return self._thread_local.backend

    def _llm_loop(self) -> None:
        """Background worker: drain the queue in chunks of LLM_BATCH_SIZE anchors
        and evaluate them in one LLM call per chunk."""
        while True:
            # Block until at least one item is available, then drain up to
            # LLM_BATCH_SIZE-1 more without blocking.
            first = self._llm_queue.get()
            if first is None:  # shutdown sentinel
                break
            batch = [first]
            while len(batch) < LLM_BATCH_SIZE:
                try:
                    item = self._llm_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    # Re-queue the shutdown sentinel so the outer loop sees it
                    # after this batch is processed.
                    self._llm_queue.put(None)
                    break
                batch.append(item)

            self._process_batch(batch)

    def _process_batch(self, batch: list) -> None:
        groups = [grp for (_df, grp) in batch]
        try:
            _t0 = time.time()
            print(
                f"[{_ts()}] LLM     batch_call anchors={len(groups)} "
                f"total_questions={sum(len(g) for g in groups)} "
                f"queue_remaining={self._llm_queue.qsize()}",
                flush=True,
            )
            results = asyncio.run(
                _discover_pairs_in_batch(self._get_backend(), groups)
            )
            total_pairs = sum(len(p) for _c, p in results)
            print(
                f"[{_ts()}] LLM     batch_done in {time.time()-_t0:.1f}s  "
                f"pairs={total_pairs}",
                flush=True,
            )
        except Exception as exc:
            logging.getLogger(__name__).warning("LLM batch call failed: %s", exc)
            return

        for group, (category, raw_pairs) in zip(groups, results):
            q_to_row = {r["question"]: r for r in group}
            anchor_q = group[0]["question"]

            for pair in raw_pairs:
                is_same = pair.get("is_same_outcome")
                if is_same is None:
                    continue
                conf = float(pair.get("confidence_score", 0))
                if conf < self._min_confidence:
                    continue

                q_a = pair.get("question_a", "")
                q_b = pair.get("question_b", "")
                # Enforce anchor-involvement: skip any pair the LLM returned
                # that doesn't involve this group's anchor.
                if anchor_q not in (q_a, q_b):
                    continue

                row_a = q_to_row.get(q_a)
                row_b = q_to_row.get(q_b)
                if row_a is None or row_b is None:
                    continue
                # Allow both cross-platform AND same-platform pairs:
                # same-platform contradictions (e.g. R wins / D wins of the
                # same race, both on Polymarket) yield real arbitrage when
                # yes_a + yes_b < $1 (contradiction) or via NO-NO when same.

                yes_a, no_a = row_a.get("price_yes"), row_a.get("price_no")
                yes_b, no_b = row_b.get("price_yes"), row_b.get("price_no")
                if any(v is None for v in (yes_a, no_a, yes_b, no_b)):
                    continue
                if any(math.isnan(v) for v in (yes_a, no_a, yes_b, no_b)):
                    continue

                rationale = pair.get("rationale", "")
                profit, strategy = compute_arb(yes_a, no_a, yes_b, no_b, is_same)

                # Always register for continuous price monitoring.
                _upsert_watched_pair(self._sf, row_a, row_b, is_same, conf, category, rationale)

                if profit >= self._spread_threshold:
                    _upsert_pair(self._sf, row_a, row_b, profit, conf, category, rationale)
                    _print_opportunity(row_a, row_b, profit, conf, strategy)

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
        price_yes, price_no = _get_prices(market, platform)

        if not question or not market_id:
            return
        # Skip Kalshi prefixes with no plausible Polymarket counterpart (parlays,
        # sports player props, etc.) — see correlations.tagger.KALSHI_SKIP_PREFIXES.
        if platform == "kalshi" and market_id.startswith(KALSHI_SKIP_PREFIXES):
            return

        emb = self._model.encode([question], normalize_embeddings=True).astype(np.float32)

        # Build group and update index under lock (fast ops only)
        clob_tokens = market.get("clobTokenIds") or []
        token_id_yes = clob_tokens[0] if platform == "polymarket" and clob_tokens else None
        token_id_no = clob_tokens[1] if platform == "polymarket" and len(clob_tokens) > 1 else None

        with self._lock:
            new_row = {"id": market_id, "platform": platform,
                       "question": question,
                       "price_yes": price_yes, "price_no": price_no,
                       "token_id_yes": token_id_yes, "token_id_no": token_id_no}
            group = [new_row]

            n_total = self._index.ntotal
            if n_total > 0:
                # Oversample then keep top-(K/2) per platform. Same-platform
                # clusters (esp. Polymarket sports) used to dominate top-K and
                # crowd out the cross-platform Kalshi neighbors — searching
                # wider and bucketing by platform guarantees a balanced group.
                k_per = max(1, self._k // 2)
                k_search = min(self._k * 3, n_total)
                scores, indices = self._index.search(emb, k_search)
                per_platform_count: dict[str, int] = {}
                for score, idx in zip(scores[0], indices[0]):
                    if not (0 <= idx < len(self._meta)):
                        continue
                    if score < MIN_CROSS_PLATFORM_SIM:
                        continue
                    neighbor = self._meta[idx]
                    plat = neighbor["platform"]
                    if per_platform_count.get(plat, 0) >= k_per:
                        continue
                    group.append(neighbor)
                    per_platform_count[plat] = per_platform_count.get(plat, 0) + 1

            self._index.add(emb)
            self._meta.append(new_row)

        if len(group) < 2:
            return

        # Same-platform contradictions (e.g. "Republican wins X" vs "Democrat
        # wins X" both on Polymarket) ARE valid arb pairs and we want them.
        # No cross-platform precondition here — defer to the LLM to find pairs.
        self._llm_queue.put((pd.DataFrame(group), group))


# ── Arbitrage watcher (WebSocket-driven) ──────────────────────────────────────

class ArbitrageWatcher:
    """
    Subscribes to live WebSocket price feeds for all watched pairs and fires
    arbitrage alerts the moment a spread crosses the threshold.

    Architecture:
      - KalshiWSClient   runs in its own daemon thread / event loop
      - PolymarketWSClient runs in its own daemon thread / event loop
      - A sync loop (run()) polls the DB every `sync_interval` seconds for
        newly registered pairs and adds WS subscriptions for new markets.
      - on_price() is called from WS threads; all shared state is protected
        by locks so the main thread and sync loop are never blocked.

    Price updates in the DB (price_a, price_b, spread, last_checked_at) are
    written on every callback so the watchlist is always fresh.
    """

    def __init__(
        self,
        sf,
        kalshi_client: KalshiClient,
        spread_threshold: float,
        sync_interval: int = 60,
    ) -> None:
        self._sf = sf
        self._spread_threshold = spread_threshold
        self._sync_interval = sync_interval
        self._log = logging.getLogger("arb-watcher")

        # Price cache: market_id → {"yes": float, "no": float}
        self._prices: dict[str, dict[str, float]] = {}
        self._prices_lock = threading.Lock()

        # Pair registry: (id_a, id_b) → pair dict; market_id → list of pair keys
        self._pairs: dict[tuple, dict] = {}
        self._market_to_pairs: dict[str, list[tuple]] = {}
        self._pairs_lock = threading.Lock()

        # WS clients
        self._kalshi_ws = KalshiWSClient(
            auth_headers_fn=lambda: kalshi_client._auth_headers("GET", "/trade-api/ws/v2"),
            on_price=self._on_price,
        )
        self._poly_ws = PolymarketWSClient(on_price=self._on_price)

        # Track which market IDs are already subscribed
        self._subscribed_kalshi: set[str] = set()
        self._subscribed_poly: set[str] = set()

    # ── Price callback (called from WS threads) ───────────────────────────────

    def _on_price(self, market_id: str, side: str, price: float) -> None:
        with self._prices_lock:
            self._prices.setdefault(market_id, {})[side] = price

        with self._pairs_lock:
            pair_keys = list(self._market_to_pairs.get(market_id, []))

        for key in pair_keys:
            with self._pairs_lock:
                pair = self._pairs.get(key)
            if pair is None:
                continue

            with self._prices_lock:
                prices_a = self._prices.get(pair["id_a"], {})
                prices_b = self._prices.get(pair["id_b"], {})
                yes_a, no_a = prices_a.get("yes"), prices_a.get("no")
                yes_b, no_b = prices_b.get("yes"), prices_b.get("no")

            if any(v is None for v in (yes_a, no_a, yes_b, no_b)):
                continue

            is_same = pair["is_same_outcome"]
            profit, strategy = compute_arb(yes_a, no_a, yes_b, no_b, is_same)
            self._update_pair_prices(pair, yes_a, no_a, yes_b, no_b, profit)

            if profit >= self._spread_threshold:
                conf = pair["confidence_score"] or 0.0
                row_a = {"id": pair["id_a"], "platform": pair["platform_a"],
                         "question": pair["question_a"],
                         "price_yes": yes_a, "price_no": no_a}
                row_b = {"id": pair["id_b"], "platform": pair["platform_b"],
                         "question": pair["question_b"],
                         "price_yes": yes_b, "price_no": no_b}
                _upsert_pair(self._sf, row_a, row_b, profit, conf,
                             pair["category"], pair["rationale"])
                _print_opportunity(row_a, row_b, profit, conf, strategy)

    def _update_pair_prices(
        self, pair: dict,
        yes_a: float, no_a: float,
        yes_b: float, no_b: float,
        profit: float,
    ) -> None:
        from datetime import datetime, timezone
        try:
            with self._sf() as session:
                row = session.get(WatchedPair, pair["db_id"])
                if row is not None:
                    row.price_yes_a = yes_a
                    row.price_no_a = no_a
                    row.price_yes_b = yes_b
                    row.price_no_b = no_b
                    row.spread = profit
                    row.last_checked_at = datetime.now(timezone.utc)
                    session.commit()
        except Exception as exc:
            self._log.debug("DB update error for pair %s/%s: %s",
                            pair["id_a"], pair["id_b"], exc)

    # ── Subscription sync ─────────────────────────────────────────────────────

    def _sync(self) -> None:
        """Load watched pairs from DB, subscribe to any markets not yet subscribed."""
        with self._sf() as session:
            rows = session.query(WatchedPair).all()
            db_pairs = [
                {
                    "db_id": r.id,
                    "is_same_outcome": r.is_same_outcome,
                    "id_a": r.id_a, "platform_a": r.platform_a,
                    "question_a": r.question_a,
                    "token_id_a": r.token_id_a, "token_id_no_a": r.token_id_no_a,
                    "id_b": r.id_b, "platform_b": r.platform_b,
                    "question_b": r.question_b,
                    "token_id_b": r.token_id_b, "token_id_no_b": r.token_id_no_b,
                    "confidence_score": r.confidence_score,
                    "category": r.category or "", "rationale": r.rationale or "",
                }
                for r in rows
            ]

        new_kalshi: list[str] = []
        new_poly_tokens: list[str] = []
        token_to_market: dict[str, tuple[str, str]] = {}

        with self._pairs_lock:
            for p in db_pairs:
                key = (p["id_a"], p["id_b"])
                if key not in self._pairs:
                    self._pairs[key] = p
                    for mid in (p["id_a"], p["id_b"]):
                        self._market_to_pairs.setdefault(mid, []).append(key)

                # Kalshi subscriptions (WS delivers both yes_ask and no_ask per ticker)
                for platform, mid in [(p["platform_a"], p["id_a"]),
                                       (p["platform_b"], p["id_b"])]:
                    if platform == "kalshi" and mid not in self._subscribed_kalshi:
                        new_kalshi.append(mid)
                        self._subscribed_kalshi.add(mid)

                # Polymarket subscriptions — subscribe both YES and NO tokens
                for platform, cid, tid_yes, tid_no in [
                    (p["platform_a"], p["id_a"], p["token_id_a"], p["token_id_no_a"]),
                    (p["platform_b"], p["id_b"], p["token_id_b"], p["token_id_no_b"]),
                ]:
                    if platform != "polymarket":
                        continue
                    if tid_yes and tid_yes not in self._subscribed_poly:
                        new_poly_tokens.append(tid_yes)
                        self._subscribed_poly.add(tid_yes)
                        token_to_market[tid_yes] = (cid, "yes")
                    if tid_no and tid_no not in self._subscribed_poly:
                        new_poly_tokens.append(tid_no)
                        self._subscribed_poly.add(tid_no)
                        token_to_market[tid_no] = (cid, "no")

        if new_kalshi:
            self._kalshi_ws.subscribe(new_kalshi)
            self._log.info("Subscribed to %d Kalshi tickers: %s",
                           len(new_kalshi), new_kalshi[:5])
        if new_poly_tokens:
            self._poly_ws.subscribe(new_poly_tokens, token_to_market)
            self._log.info("Subscribed to %d Polymarket tokens", len(new_poly_tokens))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, deadline: float) -> None:
        self._kalshi_ws.start()
        self._poly_ws.start()

        while time.monotonic() < deadline:
            try:
                self._sync()
            except Exception as exc:
                self._log.warning("sync error: %s", exc)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self._sync_interval, remaining))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming arbitrage detector")
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--duration", type=float, default=0,
                        help="Stop after N hours (0 = run forever)")
    parser.add_argument("--spread", type=float, default=ENTRY_SPREAD_THRESHOLD,
                        help=f"Min spread to alert (default: {ENTRY_SPREAD_THRESHOLD})")
    parser.add_argument("--lookback", type=int, default=None,
                        help="Seconds back to look for new markets per poll (default: same as --interval to avoid gaps)")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS,
                        help=f"Nearest neighbors per new market (default: {K_NEIGHBORS})")
    parser.add_argument("--monitor-interval", type=int, default=None,
                        help="How often (seconds) to sync DB for new watched pairs to subscribe (default: same as --interval)")
    args = parser.parse_args()
    if args.monitor_interval is None:
        args.monitor_interval = args.interval
    if args.lookback is None:
        args.lookback = args.interval

    def backend_factory():
        return OpenAICompatibleBackend(
            model=config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            rpm_limit=5,
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

    watcher = ArbitrageWatcher(
        sf=sf,
        kalshi_client=kalshi_client,
        spread_threshold=args.spread,
        sync_interval=args.monitor_interval,
    )

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
        threading.Thread(
            target=watcher.run,
            args=(deadline,),
            daemon=True,
            name="arb-watcher",
        ),
    ]

    print(
        f"Streaming arbitrage detector  |  model={config.LLM_MODEL} (Groq)"
        f"  |  spread>={args.spread:.0%}  |  k={args.k}"
        f"  |  poll={args.interval}s  |  ws-sync={args.monitor_interval}s  |  Ctrl+C to stop\n"
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
