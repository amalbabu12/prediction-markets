"""Build tag-overlap candidate pairs from market_tags.

For every market tagged within the last cycle, find every other tagged market
that shares at least one tag (excluding the _no_tags_ sentinel), compute the
Jaccard similarity of their tag sets, and write the pair to candidate_pairs.

The (id_a, platform_a) entry is lexicographically ordered before (id_b, platform_b)
so that re-processing the same pair from either side hits the same primary-key
collision and is a no-op via INSERT OR IGNORE.

Process model: standalone, idempotent, polls every PAIR_BUILDER_POLL_INTERVAL_SEC.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from db.models import init_db
from correlations import models  # noqa: F401  — register tables with Base
from correlations.config import PAIR_MIN_SHARED_TAGS
from correlations.models import CandidatePair
from correlations.tagger import NO_TAGS_SENTINEL

log = logging.getLogger("correlations.pair_builder")

PAIR_BUILDER_POLL_INTERVAL_SEC = 600   # 10 min — pair_builder is cheap, no need to be aggressive
RECENTLY_TAGGED_WINDOW_MIN = 60        # process markets tagged within the last hour each cycle


def _recently_tagged_markets(session: Session, window_min: int) -> list[tuple[str, str]]:
    sql = """
    SELECT DISTINCT market_id, platform
    FROM market_tags
    WHERE tag != :sentinel
      AND tagged_at > datetime('now', :cutoff)
    """
    rows = session.execute(
        text(sql),
        {"sentinel": NO_TAGS_SENTINEL, "cutoff": f"-{window_min} minutes"},
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _tags_for_market(session: Session, market_id: str, platform: str) -> set[str]:
    sql = """
    SELECT tag FROM market_tags
    WHERE market_id = :mid AND platform = :pf AND tag != :sentinel
    """
    rows = session.execute(
        text(sql),
        {"mid": market_id, "pf": platform, "sentinel": NO_TAGS_SENTINEL},
    ).fetchall()
    return {r[0] for r in rows}


def _markets_sharing_tags(
    session: Session,
    tags: set[str],
    exclude_id: str,
    exclude_platform: str,
) -> dict[tuple[str, str], set[str]]:
    """Return {(market_id, platform): set_of_shared_tags} for every other
    market that has at least one tag in common with `tags`."""
    if not tags:
        return {}
    stmt = text("""
        SELECT market_id, platform, tag FROM market_tags
        WHERE tag IN :tags
          AND tag != :sentinel
          AND NOT (market_id = :exid AND platform = :expf)
    """).bindparams(bindparam("tags", expanding=True))
    rows = session.execute(stmt, {
        "tags": list(tags),
        "sentinel": NO_TAGS_SENTINEL,
        "exid": exclude_id,
        "expf": exclude_platform,
    }).fetchall()

    out: dict[tuple[str, str], set[str]] = {}
    for market_id, platform, tag in rows:
        out.setdefault((market_id, platform), set()).add(tag)
    return out


def _ordered(a: tuple[str, str], b: tuple[str, str]) -> tuple[tuple[str, str], tuple[str, str]]:
    """Lexicographic order so (A, B) and (B, A) collide on the unique constraint."""
    return (a, b) if (a[1], a[0]) <= (b[1], b[0]) else (b, a)


def _build_pairs_for_market(
    session: Session,
    market_id: str,
    platform: str,
) -> int:
    my_tags = _tags_for_market(session, market_id, platform)
    if not my_tags:
        return 0

    peers = _markets_sharing_tags(session, my_tags, market_id, platform)
    inserted = 0

    for (peer_id, peer_pf), shared_tags in peers.items():
        if len(shared_tags) < PAIR_MIN_SHARED_TAGS:
            continue
        peer_tags = _tags_for_market(session, peer_id, peer_pf)
        if not peer_tags:
            continue

        union = my_tags | peer_tags
        jaccard = len(shared_tags) / len(union)

        side_a, side_b = _ordered((market_id, platform), (peer_id, peer_pf))

        stmt = sqlite_insert(CandidatePair).values(
            id_a=side_a[0], platform_a=side_a[1],
            id_b=side_b[0], platform_b=side_b[1],
            jaccard=jaccard,
            shared_tags=json.dumps(sorted(shared_tags)),
        )
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["id_a", "id_b"]
        )
        result = session.execute(stmt)
        if result.rowcount:
            inserted += 1

    if inserted:
        session.commit()
    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tag-overlap candidate pairs.")
    parser.add_argument("--db", default="./data/markets.db")
    parser.add_argument("--window-min", type=int, default=RECENTLY_TAGGED_WINDOW_MIN,
                        help="Process markets tagged within the last N minutes (default: %(default)s).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sf = init_db(args.db)
    log.info("pair_builder started — poll=%ds  window=%dm  min_shared_tags=%d",
             PAIR_BUILDER_POLL_INTERVAL_SEC, args.window_min, PAIR_MIN_SHARED_TAGS)

    while True:
        with sf() as session:
            markets = _recently_tagged_markets(session, args.window_min)

        if not markets:
            time.sleep(PAIR_BUILDER_POLL_INTERVAL_SEC)
            continue

        total = 0
        for market_id, platform in markets:
            with sf() as session:
                total += _build_pairs_for_market(session, market_id, platform)

        log.info("processed %d tagged markets, inserted %d new candidate pairs",
                 len(markets), total)
        time.sleep(PAIR_BUILDER_POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
