"""
One-shot KNN cross-platform pair discovery over the existing market pool.

Loads every active non-parlay market via the same bootstrap path as the live
detector, then for each market queries the FAISS index for k nearest neighbors
on the other platform. Cross-platform candidates with similarity above
MIN_CROSS_PLATFORM_SIM are queued for LLM verification, mirroring the live
`process()` path so any confirmed pairs land in `watched_pairs` exactly as the
streaming pipeline would have written them.

Run this with the live detector stopped so they don't fight for LLM budget.

Usage:
    python backfill_pairs.py [--db PATH] [--k K] [--limit N]
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

import config
from db.models import init_db
from detect_arbitrage import (
    ENTRY_SPREAD_THRESHOLD,
    K_NEIGHBORS,
    MIN_CONFIDENCE,
    MIN_CROSS_PLATFORM_SIM,
    StreamingDetector,
    _ts,
)
from forecasting.llm import OpenAICompatibleBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill cross-platform watched pairs over the existing market pool."
    )
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--k", type=int, default=K_NEIGHBORS)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max markets to scan (0 = all).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=30,
        help="Max unique markets per LLM call. Larger = fewer calls but heavier prompts.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sf = init_db(args.db)

    def backend_factory():
        return OpenAICompatibleBackend(
            model=config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            rpm_limit=5,
        )

    detector = StreamingDetector(
        backend_factory=backend_factory,
        sf=sf,
        spread_threshold=ENTRY_SPREAD_THRESHOLD,
        k=args.k,
        min_confidence=MIN_CONFIDENCE,
    )
    detector.bootstrap(args.db)

    meta = list(detector._meta)
    if args.limit:
        meta = meta[: args.limit]
    n = len(meta)
    print(
        f"[{_ts()}] backfill: scanning {n:,} markets "
        f"(k={args.k}, sim>={MIN_CROSS_PLATFORM_SIM})",
        flush=True,
    )

    queued = 0
    skipped_no_price = 0
    current_chunk: list[dict] = []
    chunk_ids: set[str] = set()

    def flush_chunk() -> None:
        nonlocal queued, current_chunk, chunk_ids
        if not current_chunk:
            return
        platforms = {r["platform"] for r in current_chunk}
        if len(platforms) >= 2:
            detector._llm_queue.put((pd.DataFrame(current_chunk), current_chunk))
            queued += 1
        current_chunk = []
        chunk_ids = set()

    for i, row in enumerate(meta):
        if row.get("price_yes") is None or row.get("price_no") is None:
            skipped_no_price += 1
            continue

        vec = detector._index.reconstruct(i).reshape(1, -1).astype(np.float32)
        scores, indices = detector._index.search(vec, args.k + 1)

        block: list[dict] = [row]
        seen_in_block: set[str] = {row["id"]}
        for score, idx in zip(scores[0], indices[0]):
            if idx == i or idx < 0:
                continue
            if score < MIN_CROSS_PLATFORM_SIM:
                break
            cand = detector._meta[idx]
            if cand["platform"] == row["platform"]:
                continue
            if cand.get("price_yes") is None or cand.get("price_no") is None:
                continue
            if cand["id"] in seen_in_block:
                continue
            block.append(cand)
            seen_in_block.add(cand["id"])

        if len(block) < 2:
            continue

        if len(current_chunk) + len(block) > args.chunk_size:
            flush_chunk()

        for r in block:
            if r["id"] not in chunk_ids:
                current_chunk.append(r)
                chunk_ids.add(r["id"])

        if (i + 1) % 5000 == 0:
            print(
                f"[{_ts()}] backfill: scanned {i+1:,}/{n:,}  "
                f"chunks_queued={queued}  llm_queue={detector._llm_queue.qsize()}",
                flush=True,
            )

    flush_chunk()

    print(
        f"[{_ts()}] backfill: scan complete. chunks_queued={queued}; "
        f"skipped(no_price)={skipped_no_price}",
        flush=True,
    )
    print(
        f"[{_ts()}] draining LLM queue (size={detector._llm_queue.qsize()})...",
        flush=True,
    )

    last_print = time.time()
    while detector._llm_queue.qsize() > 0:
        time.sleep(10)
        if time.time() - last_print >= 60:
            print(
                f"[{_ts()}] LLM queue: {detector._llm_queue.qsize()} remaining",
                flush=True,
            )
            last_print = time.time()

    # Grace period for the last in-flight LLM call to commit.
    time.sleep(20)
    print(f"[{_ts()}] backfill done.", flush=True)


if __name__ == "__main__":
    main()
