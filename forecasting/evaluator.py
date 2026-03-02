"""
Accuracy evaluation for discovered market pair relationships.

Implements the two metrics from the paper:
  - Overall accuracy:  pooled fraction correct across all evaluable pairs
  - Cluster accuracy:  mean of per-cluster accuracy (each cluster weighted equally)

Also breaks down accuracy by topic category and by platform-pair type
(kalshi-kalshi, kalshi-polymarket, polymarket-polymarket).

Only pairs where BOTH markets have a known outcome are included in evaluation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from forecasting.relationships import DiscoveredPair

logger = logging.getLogger(__name__)


@dataclass
class AccuracyReport:
    total_pairs: int          # all pairs returned by relationship discovery
    evaluable_pairs: int      # pairs where both outcomes are known
    overall_accuracy: float   # pooled fraction correct
    cluster_accuracy: float   # mean of per-cluster accuracy (equal cluster weight)
    by_category: dict[str, float]       # accuracy per topic category
    by_platform_pair: dict[str, float]  # accuracy per platform combo
    confusion: dict[str, int]           # TP, FP, TN, FN


def evaluate(pairs: list[DiscoveredPair]) -> AccuracyReport:
    """
    Compute accuracy metrics against ground-truth resolved outcomes.

    Ground truth: ground_truth(i,j) = 1 if outcome_i == outcome_j
    Prediction:   is_same_outcome predicted by the LLM

    Args:
        pairs: Output from discover_relationships()

    Returns:
        AccuracyReport with overall, cluster-level, and breakdown metrics
    """
    rows = []
    for p in pairs:
        if p.outcome_a is None or p.outcome_b is None:
            continue  # unresolved — can't evaluate yet

        ground_truth = (p.outcome_a == p.outcome_b)
        predicted = p.is_same_outcome
        correct = (predicted == ground_truth)

        # Normalize platform pair to alphabetical order so
        # "polymarket-kalshi" and "kalshi-polymarket" are the same bucket
        platforms = tuple(sorted([p.platform_a, p.platform_b]))
        platform_pair = f"{platforms[0]}-{platforms[1]}"

        rows.append({
            "anchor_id": p.anchor_id,
            "category": p.group_category,
            "platform_pair": platform_pair,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": correct,
            "confidence": p.confidence_score,
        })

    if not rows:
        logger.warning(
            "No evaluable pairs (both outcomes must be resolved). "
            "Run after more markets have settled."
        )
        return AccuracyReport(
            total_pairs=len(pairs),
            evaluable_pairs=0,
            overall_accuracy=0.0,
            cluster_accuracy=0.0,
            by_category={},
            by_platform_pair={},
            confusion={"TP": 0, "FP": 0, "TN": 0, "FN": 0},
        )

    df = pd.DataFrame(rows)

    overall_acc = float(df["correct"].mean())
    cluster_acc = float(df.groupby("anchor_id")["correct"].mean().mean())

    by_cat = {
        cat: float(grp["correct"].mean())
        for cat, grp in df.groupby("category")
    }
    by_plat = {
        pair: float(grp["correct"].mean())
        for pair, grp in df.groupby("platform_pair")
    }

    tp = int(((df["predicted"] == True) & (df["ground_truth"] == True)).sum())
    fp = int(((df["predicted"] == True) & (df["ground_truth"] == False)).sum())
    tn = int(((df["predicted"] == False) & (df["ground_truth"] == False)).sum())
    fn = int(((df["predicted"] == False) & (df["ground_truth"] == True)).sum())

    report = AccuracyReport(
        total_pairs=len(pairs),
        evaluable_pairs=len(df),
        overall_accuracy=overall_acc,
        cluster_accuracy=cluster_acc,
        by_category=by_cat,
        by_platform_pair=by_plat,
        confusion={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    )
    _log_report(report)
    return report


def _log_report(r: AccuracyReport) -> None:
    logger.info("=== Accuracy Report ===")
    logger.info("Total pairs: %d  |  Evaluable: %d", r.total_pairs, r.evaluable_pairs)
    logger.info("Overall accuracy:  %.1f%%", r.overall_accuracy * 100)
    logger.info("Cluster accuracy:  %.1f%%", r.cluster_accuracy * 100)
    logger.info(
        "Confusion — TP=%d  FP=%d  TN=%d  FN=%d",
        r.confusion["TP"], r.confusion["FP"], r.confusion["TN"], r.confusion["FN"],
    )
    if r.by_category:
        logger.info("By category:")
        for cat, acc in sorted(r.by_category.items(), key=lambda x: -x[1]):
            logger.info("  %-15s %.1f%%", cat, acc * 100)
    if r.by_platform_pair:
        logger.info("By platform pair:")
        for pair, acc in sorted(r.by_platform_pair.items(), key=lambda x: -x[1]):
            logger.info("  %-30s %.1f%%", pair, acc * 100)
