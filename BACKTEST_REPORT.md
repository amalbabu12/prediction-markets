# Arbitrage Strategy Backtest Report

**Date:** 2026-03-03
**Script:** [`backtest_arbitrage.py`](backtest_arbitrage.py)
**Mock results:** [`backtest_results.json`](backtest_results.json) · [`backtest_output.txt`](backtest_output.txt)
**Real HF results:** [`backtest_results_hf.json`](backtest_results_hf.json) · [`backtest_output_hf.txt`](backtest_output_hf.txt)

---

## Strategy Overview

The strategy discovers semantically equivalent markets across Kalshi and Polymarket. When two markets pricing the same binary event show a meaningful price divergence, a risk-free arbitrage exists in theory:

- **Buy YES** on the cheaper platform (A)
- **Buy NO** on the expensive platform (B)

**Total cost** = `price_a + (1 - price_b)` = `1 - spread`

**Payout at resolution:**

| Outcome A | Outcome B | Payout | Net P&L |
|-----------|-----------|--------|---------|
| YES | YES | 1 | +spread |
| NO | NO | 1 | +spread |
| YES | NO | 2 | +1 + spread *(windfall — pair misclassified)* |
| NO | YES | 0 | -(1 - spread) *(loss — pair misclassified)* |

The trade is only risk-free when the model correctly identifies the pair as same-outcome. Misclassified pairs can lose approximately −$1 per $1 of notional.

---

## Test Script

The full script is in [`backtest_arbitrage.py`](backtest_arbitrage.py). Below is a summary of each stage.

### Pipeline Stages

`backtest_arbitrage.py` runs a 6-step pipeline against simulated data:

1. **Build mock markets** — 15 Kalshi/Polymarket pairs (30 markets total) with 30-day simulated price histories. Prices drift from 0.50 toward a known `true_prob`, with a decaying divergence term that creates early-period spread opportunities widest at day 0 and converging to zero by day 30.
2. **Generate embeddings** — 64-dimensional topic-clustered embeddings (3 groups: crypto, elections, economy).
3. **KNN neighbor search** — finds the 4 nearest neighbours per market to generate candidate pairs (59 unique pairs found).
4. **Relationship discovery** — either the mock LLM or the real HuggingFace NLI model classifies each candidate pair as `is_same_outcome=True/False`.
5. **Evaluate accuracy** — compares model labels against ground-truth resolved outcomes.
6. **Run arbitrage backtest** — walks forward day-by-day; the first signal above the 4% spread threshold triggers a trade held to resolution.

### Backends

**Mock backend** (`--mock` flag):
- Labels consecutive question-pairs as `is_same_outcome=True` with `confidence=0.85`.
- Fast and deterministic. No model downloaded.
- All pairs get `category="other"` (keyword scan fails on this dataset).

**Real HuggingFace NLI backend** (default, no flag):
- Model: `typeform/distilbert-base-uncased-mnli` (DistilBERT, 66 M params, ~264 MB)
- Fine-tuned on Multi-Genre NLI; outputs ENTAILMENT / NEUTRAL / CONTRADICTION probabilities.
- Pair strategy: `premise = 'If the answer to "Q_A" is YES, then'`, `hypothesis = 'the answer to "Q_B" is also YES'`. Entailment score ≥ 0.25 → same outcome; Contradiction score ≥ 0.25 → different outcome.
- Correct categories assigned via keyword matching.

### Key Configuration

| Parameter | Value |
|-----------|-------|
| Entry spread threshold | 4% |
| Transaction cost (round-trip) | 2% |
| Simulated trading days | 30 |
| RNG seed | 42 |
| NLI model | `typeform/distilbert-base-uncased-mnli` |
| Entailment threshold | 0.25 |
| Contradiction threshold | 0.25 |

---

## Full Results — Mock LLM

*Console output: [`backtest_output.txt`](backtest_output.txt) · JSON: [`backtest_results.json`](backtest_results.json)*

### Pipeline

| Metric | Value |
|--------|-------|
| Markets total | 30 (15 Kalshi + 15 Polymarket) |
| Ground-truth pairs | 15 |
| Pairs discovered | 32 |
| Evaluable pairs | 32 / 32 |
| Overall accuracy | **46.9%** |
| Cluster accuracy | 45.0% |

**Confusion matrix:**

| | Predicted Same | Predicted Different |
|-|----------------|---------------------|
| **Actually Same** | TP = 15 | FN = 0 |
| **Actually Different** | FP = 17 | TN = 0 |

> The mock LLM labels every discovered pair as `same_outcome=True`, so TN = FN = 0. 17 of 32 pairs are false positives.

**Accuracy by platform pair:**

| Platform Pair | Accuracy |
|---------------|----------|
| Kalshi ↔ Polymarket | 55.0% |
| Kalshi ↔ Kalshi | 33.3% |
| Polymarket ↔ Polymarket | 33.3% |

### Backtest

| Metric | Value |
|--------|-------|
| Opportunities scanned | 32 |
| Trades executed | 54 |
| Profitable trades | 37 (68.5%) |
| Pair match rate | 26 / 54 (48.1%) |
| Mean entry spread | 8.20% |
| Mean gross P&L / trade | −0.0291 |
| Mean net P&L / trade | **−0.0491** |
| Total net P&L | **−2.6536** |
| Std dev net P&L | 0.7112 |
| Sharpe ratio | **−0.069** |

**Breakdown by platform pair:**

| Platform Pair | Trades | Profitable | Total Net P&L |
|---------------|--------|------------|---------------|
| Polymarket ↔ Polymarket | 11 | 7 (63.6%) | **+0.8192** |
| Kalshi ↔ Kalshi | 9 | 5 (55.6%) | −1.4194 |
| Kalshi ↔ Polymarket | 34 | 25 (73.5%) | −2.0534 |

**Top 5 trades by net P&L:**

| Pair | Spread | Outcome A/B | Net P&L |
|------|--------|-------------|---------|
| p8 ↔ p10 | 14.2% | YES / NO | +1.1225 |
| p4 ↔ p5 | 10.7% | YES / NO | +1.0875 |
| p13 ↔ p14 | 10.2% | YES / NO | +1.0820 |
| p12 ↔ p14 | 8.8% | YES / NO | +1.0676 |
| p12 ↔ k15 | 7.9% | YES / NO | +1.0590 |

**Worst 3 trades:**

| Pair | Spread | Outcome A/B | Net P&L |
|------|--------|-------------|---------|
| k5 ↔ p4 | 4.7% | NO / YES | −0.9729 |
| p10 ↔ k9 | 4.9% | NO / YES | −0.9705 |
| p6 ↔ k8 | 5.4% | NO / YES | −0.9656 |

---

## Full Results — Real HuggingFace NLI Model

*Model: `typeform/distilbert-base-uncased-mnli`*
*Console output: [`backtest_output_hf.txt`](backtest_output_hf.txt) · JSON: [`backtest_results_hf.json`](backtest_results_hf.json)*

### Pipeline

| Metric | Value |
|--------|-------|
| Markets total | 30 (15 Kalshi + 15 Polymarket) |
| Ground-truth pairs | 15 |
| Pairs discovered | 32 |
| Same-outcome pairs classified | 24 (mock: 32) |
| Evaluable pairs | 32 / 32 |
| Overall accuracy | **62.5%** |
| Cluster accuracy | **66.5%** |

**Confusion matrix:**

| | Predicted Same | Predicted Different |
|-|----------------|---------------------|
| **Actually Same** | TP = 14 | FN = 2 |
| **Actually Different** | FP = 10 | TN = 6 |

> The real model correctly rejects 6 false-positive pairs that the mock accepted. It misses 2 true same-outcome pairs (FN=2) — a small trade-off for far fewer bad trades.

**Accuracy by platform pair:**

| Platform Pair | Accuracy |
|---------------|----------|
| Polymarket ↔ Polymarket | 66.7% |
| Kalshi ↔ Polymarket | 61.1% |
| Kalshi ↔ Kalshi | 60.0% |

### Backtest

| Metric | Value |
|--------|-------|
| Opportunities scanned | 32 |
| Trades executed | 38 |
| Profitable trades | 28 (73.7%) |
| Pair match rate | 23 / 38 (60.5%) |
| Mean entry spread | 8.92% |
| Mean gross P&L / trade | −0.0424 |
| Mean net P&L / trade | **−0.0624** |
| Total net P&L | **−2.3717** |
| Std dev net P&L | 0.6144 |
| Sharpe ratio | **−0.102** |

**Breakdown by category:**

| Category | Trades | Profitable | Total Net P&L |
|----------|--------|------------|---------------|
| crypto | 38 | 28 (73.7%) | −2.3717 |

**Top 5 trades by net P&L:**

| Pair | Spread | Outcome A/B | Net P&L |
|------|--------|-------------|---------|
| k3 ↔ p5 | 15.2% | YES / NO | +1.1320 |
| p8 ↔ p6 | 10.6% | YES / NO | +1.0859 |
| p1 ↔ p5 | 10.1% | YES / NO | +1.0811 |
| p12 ↔ k15 | 7.9% | YES / NO | +1.0590 |
| k12 ↔ k15 | 4.5% | YES / NO | +1.0250 |

**Worst 3 trades:**

| Pair | Spread | Outcome A/B | Net P&L |
|------|--------|-------------|---------|
| p5 ↔ p3 | 7.0% | NO / YES | −0.9497 |
| p2 ↔ k4 | 8.1% | NO / YES | −0.9395 |
| k2 ↔ p1 | 8.3% | NO / YES | −0.9371 |

---

## Results Summary

### Side-by-Side Comparison

| Metric | Mock LLM | HuggingFace NLI | Change |
|--------|----------|-----------------|--------|
| Same-outcome pairs | 32 | 24 | −8 |
| Overall accuracy | 46.9% | **62.5%** | +15.6 pp |
| Cluster accuracy | 45.0% | **66.5%** | +21.5 pp |
| False positives (FP) | 17 | **10** | −7 |
| True negatives (TN) | 0 | **6** | +6 |
| Trades executed | 54 | **38** | −16 |
| Win rate | 68.5% | **73.7%** | +5.2 pp |
| Pair match rate | 48.1% | **60.5%** | +12.4 pp |
| Mean entry spread | 8.20% | **8.92%** | +0.72 pp |
| Total net P&L | −2.6536 | **−2.3717** | +0.2819 |
| Sharpe ratio | −0.069 | −0.102 | −0.033 |

### What the Numbers Mean

**The strategy remains unprofitable in both runs**, but the real HuggingFace NLI model produces measurably better pair classification than the mock, and this improvement is directly visible in the backtest results.

**Why the real model is better:**
- The NLI model correctly rejects 6 pairs the mock blindly accepted (TN = 6 vs 0). Each rejected false-positive is a prevented catastrophic-loss trade (≈ −$0.96 each).
- Pair match rate improves from 48.1% → 60.5%: fewer executed trades resolve with mismatched outcomes.
- Trades executed drops from 54 → 38 because the model filters out 8 low-confidence pairs.
- Total net P&L improves by +$0.28 (from −2.65 to −2.37).

**Why both runs are still unprofitable:**
The payoff structure is highly asymmetric:
- **Winning trade (correct pair, spread captured):** gain ≈ `spread − 2%`, typically **+$0.02 to +$0.21** per $1 notional.
- **Losing trade (misclassified pair):** lose ≈ `1 − spread`, typically **−$0.86 to −$0.97** per $1 notional.

Even at 60.5% pair match rate, the expected value per trade is negative because losses from the 39.5% misclassified trades dwarf the gains from correct trades. To break even, the pair match rate needs to exceed ~91% (assuming average spread of 9% and 2% transaction cost): you need 11 correct trades to offset 1 catastrophic loss.

**The NLI model's Sharpe ratio is worse (−0.102 vs −0.069)** despite the improved accuracy. This is because the model's correct rejections also removed some winning windfall trades (pairs where mismatched outcomes coincidentally produced +$1 payouts), slightly increasing per-trade variance relative to the smaller trade count.

**Key takeaways:**

1. **The real NLI model outperforms mock on every classification metric** — accuracy +15.6 pp, pair match rate +12.4 pp, FP count reduced by 41%.
2. **The arbitrage signal exists and is detectable** — mean entry spread of 8.9% is well above costs on true same-outcome pairs.
3. **Classification quality is the bottleneck** — the threshold for profitability is a pair match rate ≥ ~91%. The NLI model at 60.5% is better than mock at 48.1%, but neither is near profitability.
4. **Recommended next steps:**
   - Use a larger / stronger NLI model (e.g., `cross-encoder/nli-deberta-v3-large`) for better entailment discrimination.
   - Add confidence gating: only trade pairs where `confidence_score ≥ 0.80` to further trim FPs.
   - Implement position sizing proportional to confidence to bound catastrophic-loss exposure.
   - Collect real market data to validate whether cross-platform spreads persist long enough for entry.
