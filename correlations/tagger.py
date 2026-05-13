"""LLM-driven causal tagging worker.

Polls the markets tables for untagged rows, asks the LLM to assign 3-6 causal
driver tags per market, and writes results to market_tags + tag_dictionary.

The LLM is given the current top-N most-used tags (rolling vocabulary) and
instructed to reuse them where possible, only minting new ones if no existing
tag captures the concept. New tags are persisted with the LLM-provided
description so they show up in future prompts.

Process model: this module's main() is intended to run in its own process.
No interaction with the arb detector beyond shared SQLite reads.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Optional

from sqlalchemy import text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

import config
from db.models import KalshiMarket, PolymarketMarket, init_db
from forecasting.llm import OpenAICompatibleBackend, extract_json
from correlations import models  # noqa: F401  — register tables with Base
from correlations.models import MarketTag, TagDictionary
from correlations.config import (
    TAG_DICT_CONTEXT_TOP_N,
    TAGGER_BATCH_SIZE,
    TAGGER_POLL_INTERVAL_SEC,
)

log = logging.getLogger("correlations.tagger")

# Sentinel inserted when LLM returns no tags for a market. Prevents re-processing.
# Pair-builder must filter this out.
NO_TAGS_SENTINEL = "_no_tags_"

# Kalshi ticker prefixes we skip entirely — these are sports player-prop, parlay,
# and esports markets with no plausible Polymarket counterpart. Tagging them is
# pure LLM waste. KXMV is multi-game parlays; the rest are league-specific prop
# markets (player rebounds, first goal scorer, etc.).
KALSHI_SKIP_PREFIXES = (
    "KXMV",        # multi-game parlays (esports, cross-category, NBA single-game, ...)
    "KXMLB",       # MLB player props & game lines
    "KXNBA",       # NBA player props & team totals
    "KXNHL",       # NHL goals, points, first goal
    "KXATP", "KXWTA", "KXITF",  # tennis matches
    "KXNCAABB",    # college basketball games
    "KXPGATOUR",   # PGA tour
    "KXCS2GAME",   # Counter-Strike 2
    "KXEPL",       # English Premier League (first goalscorer etc.)
    "KXNFL",       # NFL player/game props
    "KXWNBA",      # WNBA
    "KXBOXING",
    "KXUFC",
)


SYSTEM_PROMPT = """\
You are tagging prediction-market questions with their CAUSAL DRIVER tags —
the underlying real-world events or policies whose state determines how each
market resolves. The goal is to identify other markets that would tend to
resolve the same way because they share a real-world cause.

GOOD tags name a specific real-world cause:
  trade_policy, fed_rate_decision, election_2026_us_president,
  geopolitics_china_taiwan, opec_supply_decision, fda_approval_drug_x,
  btc_price_movement, recession_us_2026, ai_model_release_openai

BAD tags — DO NOT USE these or anything similar:
  market_sentiment, trading_volume, market_events, volatility,
  market_outlook, market_dynamics, price_action, investor_psychology
These describe properties of markets themselves, not causes that decide
outcomes. Every market has these — they create false matches.

Avoid broad topic words alone (politics, sports, economy, technology, crypto).
Add specificity: politics_us_2026, nba_finals_2026, ai_regulation_eu.

Tag rules:
- snake_case, lowercase, no spaces
- 3-6 tags per market (fewer is fine for narrow markets)
- A causal driver should plausibly affect MULTIPLE markets, not just this one
- Strongly prefer existing tags from the dictionary; mint a new one ONLY when
  no existing tag captures a real cause
- Avoid near-duplicates — if "election_2026_us_president" exists, do not
  invent "us_president_election_2026"
- New tags must come with a one-line description of the cause

Inputs are batched — you'll receive multiple questions, each with an index.
Return ONLY a JSON object of the form:
{
  "results": [
    {"index": 0, "tags": [
      {"tag": "trade_policy",   "confidence": 0.9, "is_new": false},
      {"tag": "supply_chain",   "confidence": 0.7, "is_new": true,
       "description": "Logistics or supply-chain disruption events"}
    ]},
    {"index": 1, "tags": [...]}
  ]
}
Every input question must appear in the results array, with its original index.
"""


def _format_tag_dict(rows: list[tuple[str, Optional[str], int]]) -> str:
    """Compact name-only listing. Descriptions are intentionally omitted from
    the prompt to save tokens — the LLM only needs the names to reuse them."""
    if not rows:
        return "(no tags yet — every tag you propose will be new)"
    return ", ".join(tag for tag, _, _ in rows)


def _load_top_tags(session: Session, limit: int) -> list[tuple[str, Optional[str], int]]:
    rows = (
        session.query(TagDictionary.tag, TagDictionary.description, TagDictionary.market_count)
        .order_by(TagDictionary.market_count.desc())
        .limit(limit)
        .all()
    )
    return [(r[0], r[1], r[2] or 0) for r in rows]


def _untagged_kalshi(session: Session, batch: int, max_age_minutes: int) -> list[dict]:
    age_clause = ""
    params: dict = {"batch": batch}
    if max_age_minutes > 0:
        age_clause = " AND km.fetched_at > datetime('now', :cutoff)"
        params["cutoff"] = f"-{max_age_minutes} minutes"
    skip_clause = " AND " + " AND ".join(
        f"km.ticker NOT LIKE '{p}%'" for p in KALSHI_SKIP_PREFIXES
    )
    sql = f"""
    SELECT km.ticker, km.title, km.subtitle
    FROM kalshi_markets km
    WHERE km.status = 'active'
      {skip_clause}
      AND NOT EXISTS (
        SELECT 1 FROM market_tags mt
        WHERE mt.platform = 'kalshi' AND mt.market_id = km.ticker
      )
      {age_clause}
    LIMIT :batch
    """
    rows = session.execute(text(sql), params).fetchall()
    out = []
    for ticker, title, subtitle in rows:
        q = (title or "").strip()
        if subtitle:
            q = f"{q} — {subtitle.strip()}"
        if not q:
            continue
        out.append({"market_id": ticker, "platform": "kalshi", "question": q})
    return out


def _untagged_polymarket(session: Session, batch: int, max_age_minutes: int) -> list[dict]:
    age_clause = ""
    params: dict = {"batch": batch}
    if max_age_minutes > 0:
        age_clause = " AND pm.fetched_at > datetime('now', :cutoff)"
        params["cutoff"] = f"-{max_age_minutes} minutes"
    sql = f"""
    SELECT pm.condition_id, pm.question
    FROM polymarket_markets pm
    WHERE pm.closed = 0
      AND NOT EXISTS (
        SELECT 1 FROM market_tags mt
        WHERE mt.platform = 'polymarket' AND mt.market_id = pm.condition_id
      )
      {age_clause}
    LIMIT :batch
    """
    rows = session.execute(text(sql), params).fetchall()
    out = []
    for cid, question in rows:
        q = (question or "").strip()
        if not q:
            continue
        out.append({"market_id": cid, "platform": "polymarket", "question": q})
    return out


class LLMCallFailed(Exception):
    """Raised when the LLM returned no usable response (rate-limit exhaustion, etc).
    Distinct from 'LLM responded with zero tags', which is treated as a real result.
    """


def _build_batch_user_prompt(
    markets: list[dict],
    tag_dict_rows: list[tuple[str, Optional[str], int]],
) -> str:
    questions_block = "\n".join(
        f"  [{i}] ({m['platform']}) {m['question']}"
        for i, m in enumerate(markets)
    )
    return (
        f"Existing tag dictionary (reuse strongly preferred):\n"
        f"{_format_tag_dict(tag_dict_rows)}\n\n"
        f"Tag each of the following {len(markets)} market questions with their "
        f"causal driver tags. Return one entry per question in the results array.\n\n"
        f"{questions_block}"
    )


def _parse_tag_obj(t: dict) -> Optional[dict]:
    if not isinstance(t, dict):
        return None
    tag = (t.get("tag") or "").strip().lower()
    if not tag or " " in tag:
        return None
    return {
        "tag": tag,
        "confidence": float(t.get("confidence") or 0.0),
        "is_new": bool(t.get("is_new") or False),
        "description": (t.get("description") or "").strip() or None,
    }


async def _tag_markets_batch(
    backend: OpenAICompatibleBackend,
    markets: list[dict],
    tag_dict_rows: list,
) -> dict[int, list[dict]]:
    """Returns {market_index: [tag_dict, ...]} for each market index that came
    back in the LLM response. Markets not present in the result are simply
    omitted (they'll be retried next cycle)."""
    user = _build_batch_user_prompt(markets, tag_dict_rows)
    # Output budget scales with batch size; ~150 tokens per market is generous.
    raw = await backend.generate(
        user_prompt=user,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=150 * len(markets) + 256,
    )
    if not raw:
        raise LLMCallFailed("empty response from LLM (likely retries exhausted)")
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return {}

    results = parsed.get("results") or []
    out: dict[int, list[dict]] = {}
    for entry in results:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(markets):
            continue
        tags_raw = entry.get("tags") or []
        tags: list[dict] = []
        for t in tags_raw:
            parsed_t = _parse_tag_obj(t)
            if parsed_t is not None:
                tags.append(parsed_t)
        out[idx] = tags
    return out


def _persist_tags(session: Session, market: dict, tags: list[dict]) -> None:
    if not tags:
        return

    for t in tags:
        stmt = sqlite_insert(TagDictionary).values(
            tag=t["tag"],
            description=t["description"],
            market_count=1,
        )
        # On conflict: bump count; keep the older description (stable vocab)
        stmt = stmt.on_conflict_do_update(
            index_elements=[TagDictionary.tag],
            set_={"market_count": TagDictionary.market_count + 1},
        )
        session.execute(stmt)

    for t in tags:
        stmt = sqlite_insert(MarketTag).values(
            market_id=market["market_id"],
            platform=market["platform"],
            tag=t["tag"],
            confidence=t["confidence"],
        )
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["market_id", "platform", "tag"]
        )
        session.execute(stmt)

    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal-tag new markets via LLM.")
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--model", default="llama-3.1-8b-instant",
                        help="LLM model name. Tagging is simple enough for an 8B model; "
                             "Groq's free tier allows much higher RPM on 8B than 70B. "
                             "Default: llama-3.1-8b-instant.")
    parser.add_argument("--rpm", type=int, default=30,
                        help="LLM requests-per-minute cap (default 30 — fine for Groq 8B free tier).")
    parser.add_argument("--max-markets", type=int, default=0,
                        help="Exit after tagging N markets (0 = run forever).")
    parser.add_argument("--max-age-minutes", type=int, default=0,
                        help="Only tag markets with fetched_at within the last N minutes "
                             "(0 = no filter, processes the whole untagged backlog).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sf = init_db(args.db)

    backend = OpenAICompatibleBackend(
        model=args.model,
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        rpm_limit=args.rpm,
    )

    log.info("tagger started — model=%s  rpm=%d  poll_interval=%ds  batch=%d  "
             "max_markets=%d  max_age_minutes=%d",
             args.model, args.rpm, TAGGER_POLL_INTERVAL_SEC, TAGGER_BATCH_SIZE,
             args.max_markets, args.max_age_minutes)

    tagged_count = 0
    half = TAGGER_BATCH_SIZE // 2
    while True:
        with sf() as session:
            tag_dict_rows = _load_top_tags(session, TAG_DICT_CONTEXT_TOP_N)
            kalshi_pending = _untagged_kalshi(session, half, args.max_age_minutes)
            poly_pending = _untagged_polymarket(
                session, TAGGER_BATCH_SIZE - half, args.max_age_minutes
            )

        pending = kalshi_pending + poly_pending
        if not pending:
            time.sleep(TAGGER_POLL_INTERVAL_SEC)
            continue

        try:
            results = asyncio.run(_tag_markets_batch(backend, pending, tag_dict_rows))
        except Exception as exc:
            log.warning("LLM batch tag failed (%d markets): %s", len(pending), exc)
            continue

        for idx, market in enumerate(pending):
            tags = results.get(idx, [])
            if not tags:
                log.info("no tags returned for %s/%s — marking with sentinel",
                         market["platform"], market["market_id"])
                tags = [{"tag": NO_TAGS_SENTINEL, "confidence": 0.0,
                         "is_new": True, "description": "LLM returned no tags"}]

            with sf() as session:
                _persist_tags(session, market, tags)
            log.info("tagged %s/%s with %d tags: %s",
                     market["platform"], market["market_id"], len(tags),
                     ",".join(t["tag"] for t in tags))

            tagged_count += 1
            if args.max_markets and tagged_count >= args.max_markets:
                log.info("reached --max-markets=%d, exiting", args.max_markets)
                return


if __name__ == "__main__":
    main()
