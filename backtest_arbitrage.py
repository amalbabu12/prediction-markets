#!/usr/bin/env python3
"""
Backtest of the cross-platform prediction market arbitrage strategy.

Strategy overview
-----------------
The pipeline discovers semantically equivalent markets across Kalshi and
Polymarket (is_same_outcome=True pairs).  Whenever two equivalent markets
price the same event differently, a risk-free arbitrage exists:

  Position: Buy YES on the cheaper platform (A)  +  Buy NO on the expensive
            platform (B)  [i.e. pay price_a and pay (1-price_b)]
  Total cost = price_a + 1 - price_b  =  1 - spread
              where spread = price_b - price_a  > 0

  Payout at resolution (per $1 notional):
    A=YES, B=YES → receive 1+0 = 1  →  P&L = spread   (expected)
    A=NO,  B=NO  → receive 0+1 = 1  →  P&L = spread   (expected)
    A=YES, B=NO  → receive 1+1 = 2  →  P&L = 1+spread (windfall, misclassified pair)
    A=NO,  B=YES → receive 0+0 = 0  →  P&L = -(1-spread) (loss, misclassified pair)

So the strategy is profitable as long as the LLM correctly identified the
pair as same-outcome AND the discovered spread exceeds transaction costs.

Run
---
    python backtest_arbitrage.py [--mock]

    Without --mock: uses HuggingFace typeform/distilbert-base-uncased-mnli
                    for real NLI-based pair classification.
    With --mock:    uses the deterministic MockLLMBackend (fast, no model download).

Writes results to: backtest_results.json
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import pandas as pd

# ── HuggingFace / torch environment setup ─────────────────────────────────────
# torch requires libcudart.so.12 which is installed alongside the nvidia-*
# pip packages. Load it explicitly so the dynamic linker can find it before
# torch's own __init__.py attempts ctypes.CDLL("libtorch_global_deps.so").
_CUDART_PATH = (
    "/home/agent/.local/lib/python3.11/site-packages/"
    "nvidia/cuda_runtime/lib/libcudart.so.12"
)
if os.path.exists(_CUDART_PATH):
    try:
        ctypes.CDLL(_CUDART_PATH)
    except OSError:
        pass  # GPU not available; torch CPU path will still work

# Make site-packages visible (needed when running directly, not via pip install)
_LOCAL_SITE = "/home/agent/.local/lib/python3.11/site-packages"
if _LOCAL_SITE not in sys.path:
    sys.path.insert(0, _LOCAL_SITE)

sys.path.insert(0, "/workspace")

from forecasting.relationships import DiscoveredPair, discover_relationships
from forecasting.evaluator import evaluate, AccuracyReport
from forecasting.neighbors import find_neighbors
from forecasting.llm import LLMBackend


# ── Configuration ─────────────────────────────────────────────────────────────

ENTRY_SPREAD_THRESHOLD = 0.04   # min spread (4 pp) required to enter a trade
TRANSACTION_COST       = 0.02   # round-trip transaction cost as fraction of $1
N_DAYS                 = 30     # simulated trading days before resolution
RNG_SEED               = 42


# ── Mock LLM backend (same logic as forecasting/test_e2e.py) ──────────────────

class MockLLMBackend(LLMBackend):
    """
    Deterministic responses without any model call.

    - Category calls → assign topic by keyword scan
    - Pair calls     → pair consecutive questions, is_same_outcome=True, conf=0.85
    """

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
    ) -> str:
        # _PAIRS_USER contains "pairs" in its JSON schema; _LABEL_USER does not.
        if '"pairs"' in user_prompt:
            return self._pairs(user_prompt)
        return self._label(user_prompt)

    def _label(self, prompt: str) -> str:
        p = prompt.lower()
        if any(w in p for w in ("btc", "bitcoin", "crypto", "eth")):
            return '{"category": "crypto"}'
        if any(w in p for w in ("election", "president", "vote", "senate")):
            return '{"category": "elections"}'
        if any(w in p for w in ("fed", "rate", "inflation", "gdp", "economy",
                                 "unemployment", "recession", "cpi")):
            return '{"category": "economy"}'
        return '{"category": "other"}'

    def _pairs(self, prompt: str) -> str:
        questions = re.findall(r"^\d+\.\s+(.+)$", prompt, re.MULTILINE)
        if len(questions) < 2:
            return '{"pairs": []}'
        pairs = []
        for i in range(0, len(questions) - 1, 2):
            pairs.append({
                "question_a": questions[i],
                "question_b": questions[i + 1],
                "is_same_outcome": True,
                "confidence_score": 0.85,
                "rationale": "mock: consecutive questions cover the same event",
            })
        return json.dumps({"pairs": pairs})


# ── Real HuggingFace NLI backend ───────────────────────────────────────────────

class HuggingFaceNLIBackend(LLMBackend):
    """
    Pair-classification backend using a real HuggingFace NLI model.

    Model: typeform/distilbert-base-uncased-mnli  (66 M params, ~264 MB)
    ─────────────────────────────────────────────────────────────────────
    A DistilBERT model fine-tuned on Multi-Genre NLI, available freely on
    the HuggingFace Hub.  It outputs ENTAILMENT / NEUTRAL / CONTRADICTION
    probabilities for a (premise, hypothesis) pair.

    Pair classification strategy
    ────────────────────────────
    For each (question_a, question_b) candidate pair:

      premise    = 'If the answer to "<question_a>" is YES, then'
      hypothesis = 'the answer to "<question_b>" is also YES'

    If P(ENTAILMENT) > entailment_threshold  → is_same_outcome = True
    If P(CONTRADICTION) > contradiction_threshold → is_same_outcome = False
    Otherwise the pair is not included in the output (below min_confidence).

    Category labelling is handled by keyword matching (same heuristic as the
    mock backend) since categorisation is a secondary concern; the key
    improvement over the mock is the data-driven pair classification.

    Args:
        entailment_threshold:    Min entailment score to emit is_same_outcome=True.
        contradiction_threshold: Min contradiction score to emit is_same_outcome=False.
        device:                  'cpu' or 'cuda'.  Defaults to 'cpu' for portability.
    """

    _MODEL_ID = "typeform/distilbert-base-uncased-mnli"

    def __init__(
        self,
        entailment_threshold: float = 0.25,
        contradiction_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self._ent_thresh  = entailment_threshold
        self._con_thresh  = contradiction_threshold
        self._device      = device
        self._pipeline    = None   # lazy-loaded on first call

    def _load(self) -> None:
        """Lazy-load the NLI pipeline (downloads model on first use)."""
        if self._pipeline is not None:
            return
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        from transformers import pipeline as hf_pipeline
        print(f"    [HF] Loading {self._MODEL_ID} on {self._device} ...")
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self._MODEL_ID,
            device=self._device,
            top_k=None,          # return all label scores
        )
        print(f"    [HF] Model ready.")

    def _nli_score(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return {label: score} dict for a single (premise, hypothesis) pair."""
        results = self._pipeline({"text": premise, "text_pair": hypothesis})
        return {r["label"].upper(): r["score"] for r in results}

    def _label_category(self, prompt: str) -> str:
        """Keyword-based category labelling (fast, no model call needed)."""
        p = prompt.lower()
        if any(w in p for w in ("btc", "bitcoin", "crypto", "eth")):
            return '{"category": "crypto"}'
        if any(w in p for w in ("election", "president", "vote", "senate")):
            return '{"category": "elections"}'
        if any(w in p for w in ("fed", "rate", "inflation", "gdp", "economy",
                                 "unemployment", "recession", "cpi")):
            return '{"category": "economy"}'
        return '{"category": "other"}'

    def _discover_pairs(self, prompt: str) -> str:
        """NLI-based pair discovery for a group of questions."""
        self._load()

        # Extract category via keywords
        p = prompt.lower()
        if any(w in p for w in ("btc", "bitcoin", "crypto", "eth")):
            category = "crypto"
        elif any(w in p for w in ("election", "president", "vote", "senate")):
            category = "elections"
        elif any(w in p for w in ("fed", "rate", "inflation", "gdp", "economy",
                                  "unemployment", "recession", "cpi")):
            category = "economy"
        else:
            category = "other"

        questions = re.findall(r"^\d+\.\s+(.+)$", prompt, re.MULTILINE)
        if len(questions) < 2:
            return json.dumps({"category": category, "pairs": []})

        pairs: list[dict] = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                q_a = questions[i]
                q_b = questions[j]

                premise    = f'If the answer to "{q_a}" is YES, then'
                hypothesis = f'the answer to "{q_b}" is also YES'
                scores     = self._nli_score(premise, hypothesis)

                ent = scores.get("ENTAILMENT", 0.0)
                con = scores.get("CONTRADICTION", 0.0)

                if ent >= self._ent_thresh:
                    pairs.append({
                        "question_a":      q_a,
                        "question_b":      q_b,
                        "is_same_outcome": True,
                        "confidence_score": round(float(ent), 4),
                        "rationale": f"NLI entailment={ent:.2f}",
                    })
                elif con >= self._con_thresh:
                    pairs.append({
                        "question_a":      q_a,
                        "question_b":      q_b,
                        "is_same_outcome": False,
                        "confidence_score": round(float(con), 4),
                        "rationale": f"NLI contradiction={con:.2f}",
                    })

        return json.dumps({"category": category, "pairs": pairs})

    async def generate(
        self,
        user_prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
    ) -> str:
        # Dispatch: pairs prompt vs. category-only prompt
        if '"pairs"' in user_prompt:
            return await asyncio.to_thread(self._discover_pairs, user_prompt)
        return self._label_category(user_prompt)


# ── Mock market data with price time-series ───────────────────────────────────

# Each entry: (kalshi_id, poly_id, kalshi_question, poly_question, outcome, true_prob)
_MARKET_SPECS = [
    ("k1",  "p1",
     "Will BTC close above $50,000 by end of March?",
     "Bitcoin above $50k on March 31?",
     "YES", 0.80),
    ("k2",  "p2",
     "Will BTC close above $55,000 by end of March?",
     "Bitcoin above $60k on March 31?",
     "NO",  0.15),
    ("k3",  "p3",
     "Will ETH close above $3,000 by end of March?",
     "Ethereum price above $3000 on March 31?",
     "YES", 0.70),
    ("k4",  "p4",
     "Will crypto market cap exceed $2T in Q1?",
     "Total crypto market cap over $2 trillion in Q1?",
     "YES", 0.75),
    ("k5",  "p5",
     "Will BTC ETF inflows exceed $1B in March?",
     "Bitcoin ETF sees over $1B net inflows in March?",
     "NO",  0.25),
    ("k6",  "p6",
     "Will Democrats win the 2026 Senate majority?",
     "Democratic Party controls Senate after 2026 midterms?",
     "NO",  0.35),
    ("k7",  "p7",
     "Will voter turnout exceed 50% in 2026 midterms?",
     "2026 midterm turnout above 50%?",
     "YES", 0.65),
    ("k8",  "p8",
     "Will Republicans gain seats in the House in 2026?",
     "GOP gains House seats in 2026 midterms?",
     "YES", 0.60),
    ("k9",  "p9",
     "Will there be a third-party candidate in 2028?",
     "Independent candidate runs for president in 2028?",
     "YES", 0.55),
    ("k10", "p10",
     "Will the 2026 midterms see record spending?",
     "Record campaign spending in 2026 midterms?",
     "NO",  0.40),
    ("k11", "p11",
     "Will the Fed cut rates in Q1 2026?",
     "Federal Reserve cuts interest rates in Q1 2026?",
     "NO",  0.20),
    ("k12", "p12",
     "Will US CPI exceed 3% in February 2026?",
     "US inflation above 3% in February 2026?",
     "YES", 0.70),
    ("k13", "p13",
     "Will US GDP growth exceed 2% in 2026?",
     "US GDP grows more than 2% in 2026?",
     "YES", 0.68),
    ("k14", "p14",
     "Will the unemployment rate fall below 4% in 2026?",
     "US unemployment rate drops below 4% in 2026?",
     "NO",  0.30),
    ("k15", "p15",
     "Will the US enter recession in 2026?",
     "United States recession in 2026?",
     "NO",  0.18),
]


def simulate_price_pair(
    rng: np.random.Generator,
    true_prob: float,
    n_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated YES-price time-series for two equivalent markets.

    Prices start near 0.5 (maximum uncertainty) and drift toward `true_prob`
    as resolution approaches.  A decaying divergence term creates temporary
    spread opportunities — widest early, converging to zero by day N.

    Returns:
        (prices_kalshi, prices_polymarket) — each shape (n_days,)
    """
    t = np.linspace(0, 1, n_days)

    # Shared drift from 0.5 → true_prob
    drift = 0.5 + (true_prob - 0.5) * t

    # Shared macro shock (both platforms react together)
    raw_shock = rng.standard_normal(n_days) * 0.03
    shared_noise = np.cumsum(raw_shock) * 0.05

    # Platform-specific micro noise
    noise_k = rng.standard_normal(n_days) * 0.015
    noise_p = rng.standard_normal(n_days) * 0.015

    # Divergence: large early, exponentially decaying to zero at resolution
    amplitude  = rng.uniform(0.06, 0.15)
    decay      = np.exp(-6 * t)
    raw_div    = rng.standard_normal(n_days)
    divergence = amplitude * decay * raw_div

    prices_k = np.clip(drift + shared_noise + noise_k + divergence * 0.5,  0.01, 0.99)
    prices_p = np.clip(drift + shared_noise + noise_p - divergence * 0.5,  0.01, 0.99)

    return prices_k, prices_p


def build_market_data(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Build the markets DataFrame (for the pipeline) and a price-series dict.

    Returns:
        df          — DataFrame with columns id, platform, question, outcome, resolved_at
        price_series — dict mapping market id → np.ndarray of shape (N_DAYS,)
    """
    rows: list[dict] = []
    price_series: dict[str, np.ndarray] = {}

    for k_id, p_id, k_q, p_q, outcome, true_prob in _MARKET_SPECS:
        prices_k, prices_p = simulate_price_pair(rng, true_prob, N_DAYS)
        price_series[k_id] = prices_k
        price_series[p_id] = prices_p

        rows.append({"id": k_id, "platform": "kalshi",     "question": k_q,
                     "outcome": outcome, "resolved_at": "2026-03-31T00:00:00Z"})
        rows.append({"id": p_id, "platform": "polymarket", "question": p_q,
                     "outcome": outcome, "resolved_at": "2026-03-31T00:00:00Z"})

    return pd.DataFrame(rows), price_series


def make_mock_embeddings(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Synthetic L2-normalised embeddings that cluster by topic.

    Every 10 rows share a base vector; small noise ensures distinct positions.
    """
    n, d = len(df), 64
    embeddings = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        group = i // 10            # 0=crypto, 1=elections, 2=economy
        base  = rng.standard_normal(d) * 0.01
        base[group * (d // 3): (group + 1) * (d // 3)] += 1.0
        embeddings[i] = base + rng.standard_normal(d) * 0.08
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# ── Arbitrage signal and trade dataclasses ────────────────────────────────────

@dataclass
class ArbitrageSignal:
    day:         int
    pair_key:    str    # "id_a<>id_b"
    id_a:        str
    id_b:        str
    platform_a:  str
    platform_b:  str
    price_a:     float
    price_b:     float
    spread:      float  # price_b - price_a  (always positive, b is more expensive)
    category:    str


@dataclass
class TradeResult:
    signal:         ArbitrageSignal
    outcome_a:      str             # "YES" or "NO"
    outcome_b:      str
    outcomes_match: bool            # did the pair truly resolve identically?
    gross_pnl:      float           # P&L before costs (per $1 notional)
    tx_cost:        float
    net_pnl:        float
    profitable:     bool


# ── Arbitrage strategy ─────────────────────────────────────────────────────────

def scan_for_opportunities(
    pairs: list[DiscoveredPair],
    price_series: dict[str, np.ndarray],
    day: int,
    threshold: float = ENTRY_SPREAD_THRESHOLD,
) -> list[ArbitrageSignal]:
    """
    For each discovered same-outcome pair, check whether the YES-price spread
    on `day` exceeds `threshold`.  Only one signal per pair per day.
    """
    signals: list[ArbitrageSignal] = []

    for pair in pairs:
        if not pair.is_same_outcome:
            continue

        prices_a = price_series.get(pair.id_a)
        prices_b = price_series.get(pair.id_b)
        if prices_a is None or prices_b is None:
            continue
        if day >= len(prices_a) or day >= len(prices_b):
            continue

        pa = float(prices_a[day])
        pb = float(prices_b[day])

        # Always put the cheaper side as 'a', more expensive as 'b'
        if pa > pb:
            pa, pb = pb, pa
            id_a, id_b = pair.id_b, pair.id_a
            plat_a, plat_b = pair.platform_b, pair.platform_a
        else:
            id_a, id_b = pair.id_a, pair.id_b
            plat_a, plat_b = pair.platform_a, pair.platform_b

        spread = pb - pa
        if spread < threshold:
            continue

        signals.append(ArbitrageSignal(
            day=day,
            pair_key=f"{id_a}<>{id_b}",
            id_a=id_a,
            id_b=id_b,
            platform_a=plat_a,
            platform_b=plat_b,
            price_a=pa,
            price_b=pb,
            spread=spread,
            category=pair.group_category,
        ))

    signals.sort(key=lambda s: -s.spread)
    return signals


def resolve_trade(
    signal: ArbitrageSignal,
    df: pd.DataFrame,
    tx_cost: float = TRANSACTION_COST,
) -> TradeResult:
    """
    Calculate realised P&L for a trade entered at `signal`.

    P&L formula for Long YES-A / Long NO-B:
      cost   = price_a + (1 - price_b) = 1 - spread
      payout depends on (outcome_a, outcome_b):
        A=YES, B=YES → receive 1   → P&L = spread
        A=NO,  B=NO  → receive 1   → P&L = spread
        A=YES, B=NO  → receive 2   → P&L = 1 + spread  (windfall)
        A=NO,  B=YES → receive 0   → P&L = -(1-spread) (loss)
    """
    id_to_outcome = dict(zip(df["id"], df["outcome"]))
    out_a = id_to_outcome.get(signal.id_a, "UNKNOWN")
    out_b = id_to_outcome.get(signal.id_b, "UNKNOWN")
    matches = (out_a == out_b)

    if out_a == "YES" and out_b == "YES":
        gross = signal.spread
    elif out_a == "NO" and out_b == "NO":
        gross = signal.spread
    elif out_a == "YES" and out_b == "NO":
        gross = 1.0 + signal.spread      # windfall: both sides pay out
    else:  # A=NO, B=YES
        gross = -(1.0 - signal.spread)   # loss: neither side pays out

    net = gross - tx_cost
    return TradeResult(
        signal=signal,
        outcome_a=out_a,
        outcome_b=out_b,
        outcomes_match=matches,
        gross_pnl=gross,
        tx_cost=tx_cost,
        net_pnl=net,
        profitable=net > 0,
    )


# ── Backtester ────────────────────────────────────────────────────────────────

def run_backtest(
    pairs: list[DiscoveredPair],
    df: pd.DataFrame,
    price_series: dict[str, np.ndarray],
    entry_threshold: float = ENTRY_SPREAD_THRESHOLD,
    tx_cost: float = TRANSACTION_COST,
    n_days: int = N_DAYS,
) -> list[TradeResult]:
    """
    Walk forward through n_days of price data.

    Rules:
      - Each pair may generate at most one trade (first signal wins).
      - Trades are held to resolution — P&L is realised at end-of-period.
    """
    entered: set[str] = set()   # pair keys already traded
    all_trades: list[TradeResult] = []

    for day in range(n_days):
        signals = scan_for_opportunities(pairs, price_series, day, entry_threshold)
        for sig in signals:
            if sig.pair_key in entered:
                continue
            trade = resolve_trade(sig, df, tx_cost)
            all_trades.append(trade)
            entered.add(sig.pair_key)

    return all_trades


# ── Results formatting ─────────────────────────────────────────────────────────

def summarise_results(
    trades: list[TradeResult],
    acc_report: AccuracyReport,
    n_pairs_total: int,
    n_pairs_discovered: int,
) -> dict:
    """Return a JSON-serialisable results dict."""
    if not trades:
        return {"error": "no trades executed"}

    net_pnls   = [t.net_pnl for t in trades]
    gross_pnls = [t.gross_pnl for t in trades]
    spreads    = [t.signal.spread for t in trades]

    n         = len(trades)
    n_profit  = sum(1 for t in trades if t.profitable)
    n_match   = sum(1 for t in trades if t.outcomes_match)

    mean_net  = float(np.mean(net_pnls))
    total_net = float(np.sum(net_pnls))
    std_net   = float(np.std(net_pnls)) if n > 1 else 0.0
    sharpe    = (mean_net / std_net) if std_net > 0 else float("nan")

    # Per-category breakdown
    by_cat: dict[str, dict] = {}
    for t in trades:
        cat = t.signal.category
        if cat not in by_cat:
            by_cat[cat] = {"n": 0, "profitable": 0, "total_net_pnl": 0.0}
        by_cat[cat]["n"] += 1
        by_cat[cat]["profitable"] += int(t.profitable)
        by_cat[cat]["total_net_pnl"] = round(
            by_cat[cat]["total_net_pnl"] + t.net_pnl, 6
        )

    # Per-platform-pair breakdown
    by_plat: dict[str, dict] = {}
    for t in trades:
        plats = tuple(sorted([t.signal.platform_a, t.signal.platform_b]))
        key = f"{plats[0]}-{plats[1]}"
        if key not in by_plat:
            by_plat[key] = {"n": 0, "profitable": 0, "total_net_pnl": 0.0}
        by_plat[key]["n"] += 1
        by_plat[key]["profitable"] += int(t.profitable)
        by_plat[key]["total_net_pnl"] = round(
            by_plat[key]["total_net_pnl"] + t.net_pnl, 6
        )

    # Individual trade details
    trade_details = []
    for t in trades:
        trade_details.append({
            "pair":            t.signal.pair_key,
            "platform_a":     t.signal.platform_a,
            "platform_b":     t.signal.platform_b,
            "category":       t.signal.category,
            "entry_day":      t.signal.day,
            "price_a":        round(t.signal.price_a, 4),
            "price_b":        round(t.signal.price_b, 4),
            "spread":         round(t.signal.spread, 4),
            "outcome_a":      t.outcome_a,
            "outcome_b":      t.outcome_b,
            "outcomes_match": t.outcomes_match,
            "gross_pnl":      round(t.gross_pnl, 4),
            "tx_cost":        round(t.tx_cost, 4),
            "net_pnl":        round(t.net_pnl, 4),
            "profitable":     t.profitable,
        })

    return {
        "config": {
            "entry_spread_threshold": ENTRY_SPREAD_THRESHOLD,
            "transaction_cost":       TRANSACTION_COST,
            "n_simulated_days":       N_DAYS,
            "rng_seed":               RNG_SEED,
        },
        "pipeline": {
            "markets_total":       n_pairs_total * 2,
            "pairs_total":         n_pairs_total,
            "pairs_discovered":    n_pairs_discovered,
            "relationship_eval": {
                "total_pairs":     acc_report.total_pairs,
                "evaluable_pairs": acc_report.evaluable_pairs,
                "overall_accuracy": round(acc_report.overall_accuracy, 4),
                "cluster_accuracy": round(acc_report.cluster_accuracy, 4),
                "confusion":        acc_report.confusion,
                "by_category":      {k: round(v, 4) for k, v in acc_report.by_category.items()},
                "by_platform_pair": {k: round(v, 4) for k, v in acc_report.by_platform_pair.items()},
            },
        },
        "backtest": {
            "n_opportunities_scanned": n_pairs_discovered,
            "n_trades_executed":       n,
            "n_profitable":            n_profit,
            "win_rate":                round(n_profit / n, 4) if n else 0,
            "n_outcomes_matched":      n_match,
            "pair_match_rate":         round(n_match / n, 4) if n else 0,
            "mean_entry_spread":       round(float(np.mean(spreads)), 4),
            "mean_gross_pnl_per_trade": round(float(np.mean(gross_pnls)), 4),
            "mean_net_pnl_per_trade":   round(mean_net, 4),
            "total_net_pnl":            round(total_net, 4),
            "std_net_pnl":              round(std_net, 4),
            "sharpe_ratio":             round(sharpe, 4) if not np.isnan(sharpe) else None,
            "by_category":              by_cat,
            "by_platform_pair":         by_plat,
        },
        "trades": trade_details,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> dict:
    use_mock = "--mock" in sys.argv

    print("=" * 65)
    if use_mock:
        print("Arbitrage Strategy Backtest — Mock Prediction Market Data")
    else:
        print("Arbitrage Strategy Backtest — Real HuggingFace NLI Model")
    print("=" * 65)

    rng = np.random.default_rng(RNG_SEED)

    # ── 1. Build mock market data with synthetic price histories ──────────────
    print(f"\n[1] Building mock markets ({len(_MARKET_SPECS)} pairs, "
          f"{len(_MARKET_SPECS) * 2} markets total, {N_DAYS} trading days) ...")
    df, price_series = build_market_data(rng)
    print(f"    Platforms: {df['platform'].value_counts().to_dict()}")
    print(f"    Outcomes:  {df['outcome'].value_counts().to_dict()}")

    # ── 2. Synthetic embeddings ───────────────────────────────────────────────
    print("\n[2] Generating synthetic topic-clustered embeddings ...")
    embeddings = make_mock_embeddings(df, rng)
    print(f"    Shape: {embeddings.shape}  |  L2-norms ≈ {np.linalg.norm(embeddings, axis=1).mean():.3f}")

    # ── 3. KNN neighbor search ────────────────────────────────────────────────
    print("\n[3] Running KNN neighbor search (k=4) ...")
    neighbor_pairs = find_neighbors(df, embeddings, k=4)
    print(f"    Found {len(neighbor_pairs)} unique neighbor pairs")

    # ── 4. LLM relationship discovery ─────────────────────────────────────────
    if use_mock:
        print("\n[4] Discovering relationships via mock LLM ...")
        backend: LLMBackend = MockLLMBackend()
    else:
        print("\n[4] Discovering relationships via HuggingFace NLI model ...")
        print(f"    Model: {HuggingFaceNLIBackend._MODEL_ID}")
        backend = HuggingFaceNLIBackend(
            entailment_threshold=0.25,
            contradiction_threshold=0.25,
            device="cpu",
        )
    pairs = await discover_relationships(
        df,
        neighbor_pairs=neighbor_pairs,
        backend=backend,
        min_confidence=0.5,
        concurrency=4,
    )
    cross_platform = [p for p in pairs if p.platform_a != p.platform_b]
    same_outcome   = [p for p in pairs if p.is_same_outcome]
    print(f"    Total discovered pairs:   {len(pairs)}")
    print(f"    Cross-platform pairs:     {len(cross_platform)}")
    print(f"    Same-outcome pairs:       {len(same_outcome)}")

    if pairs:
        print("    Sample pairs:")
        for p in pairs[:3]:
            print(f"      [{p.group_category}] {p.id_a}({p.platform_a}) ↔ "
                  f"{p.id_b}({p.platform_b})  "
                  f"same={p.is_same_outcome}  conf={p.confidence_score:.2f}")

    # ── 5. Evaluate relationship accuracy ─────────────────────────────────────
    print("\n[5] Evaluating relationship accuracy vs ground-truth outcomes ...")
    acc_report = evaluate(pairs)
    print(f"    Evaluable pairs:    {acc_report.evaluable_pairs} / {acc_report.total_pairs}")
    print(f"    Overall accuracy:   {acc_report.overall_accuracy:.1%}")
    print(f"    Cluster accuracy:   {acc_report.cluster_accuracy:.1%}")
    print(f"    Confusion:          {acc_report.confusion}")
    if acc_report.by_category:
        print("    By category:")
        for cat, acc in sorted(acc_report.by_category.items(), key=lambda x: -x[1]):
            print(f"      {cat:<15} {acc:.1%}")

    # ── 6. Run arbitrage backtest ─────────────────────────────────────────────
    print(f"\n[6] Running arbitrage backtest over {N_DAYS} days ...")
    print(f"    Entry threshold: {ENTRY_SPREAD_THRESHOLD:.0%}  |  "
          f"Transaction cost: {TRANSACTION_COST:.0%}")

    trades = run_backtest(
        pairs=same_outcome,
        df=df,
        price_series=price_series,
        entry_threshold=ENTRY_SPREAD_THRESHOLD,
        tx_cost=TRANSACTION_COST,
        n_days=N_DAYS,
    )

    # ── 7. Print and return results ───────────────────────────────────────────
    n = len(trades)
    if n == 0:
        print("    No arbitrage opportunities detected above threshold.")
    else:
        n_profit = sum(1 for t in trades if t.profitable)
        n_match  = sum(1 for t in trades if t.outcomes_match)
        net_pnls = [t.net_pnl for t in trades]

        print(f"\n{'─' * 65}")
        print("  BACKTEST RESULTS")
        print(f"{'─' * 65}")
        print(f"  Trades executed:        {n}")
        print(f"  Profitable trades:      {n_profit}  ({n_profit/n:.1%})")
        print(f"  Pair match rate:        {n_match}/{n}  ({n_match/n:.1%})")
        print(f"  Mean entry spread:      {np.mean([t.signal.spread for t in trades]):.2%}")
        print(f"  Mean gross P&L/trade:   {np.mean([t.gross_pnl for t in trades]):.4f}")
        print(f"  Mean net P&L/trade:     {np.mean(net_pnls):.4f}")
        print(f"  Total net P&L:          {sum(net_pnls):.4f}")
        std = np.std(net_pnls)
        if std > 0:
            print(f"  Sharpe ratio:           {np.mean(net_pnls)/std:.3f}")
        print(f"{'─' * 65}")

        print("\n  Top 5 trades by net P&L:")
        for t in sorted(trades, key=lambda x: -x.net_pnl)[:5]:
            print(f"    {t.signal.pair_key:<20}  "
                  f"spread={t.signal.spread:.3f}  "
                  f"net_pnl={t.net_pnl:+.4f}  "
                  f"[{t.outcome_a}/{t.outcome_b}]  "
                  f"{'✓' if t.profitable else '✗'}")

        print("\n  Worst 3 trades:")
        for t in sorted(trades, key=lambda x: x.net_pnl)[:3]:
            print(f"    {t.signal.pair_key:<20}  "
                  f"spread={t.signal.spread:.3f}  "
                  f"net_pnl={t.net_pnl:+.4f}  "
                  f"[{t.outcome_a}/{t.outcome_b}]  "
                  f"{'✓' if t.profitable else '✗'}")

    results = summarise_results(
        trades, acc_report,
        n_pairs_total=len(_MARKET_SPECS),
        n_pairs_discovered=len(pairs),
    )
    print("\n[OK] Backtest complete.")
    return results


if __name__ == "__main__":
    results = asyncio.run(main())
    use_mock = "--mock" in sys.argv
    out_path = "/workspace/backtest_results.json" if use_mock else "/workspace/backtest_results_hf.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {out_path}")
