# Prediction Markets Arbitrage

Cross-platform arbitrage discovery between **Kalshi** and **Polymarket**. Collects market data, finds semantic and tag-overlap pairs, and watches live WebSocket prices for spread-crossing alerts.

---

## Structure

```
clients/        REST + WebSocket clients for Kalshi and Polymarket
collectors/     Bulk collectors + close_checker (refreshes stale rows)
db/             SQLAlchemy schema (kalshi_*, polymarket_*, arbitrage_pairs)
forecasting/    MiniLM embeddings, faiss KNN, LLM same-outcome classifier
correlations/   Tag-based pipeline: tagger → pair_builder → correlator → watcher
detect_arbitrage.py    Streaming detector
backtest_arbitrage.py  Walk-forward backtest (see BACKTEST_REPORT.md)
main.py                Data-collection CLI
```

---

## Data Sources

| Platform | Endpoints | Auth |
|---|---|---|
| Kalshi | `/series`, `/events`, `/markets`, `/trades`, `/candlesticks`, `/orderbook`, WS | RSA-PSS signed headers |
| Polymarket Gamma | `/events`, `/markets`, `/tags` | none |
| Polymarket CLOB | `/markets`, `/prices-history`, `/books`, `/trades`, WS | none |
| Polymarket Data | `/trades`, `/oi` | none |

Kalshi prices in cents (1–99); Polymarket prices in [0, 1].

Raw JSON is stored on every row.

---

## Setup

```bash
cp .env.example .env  # add KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, GROQ_API_KEY
pip install -r requirements.txt

python main.py snapshot                          # initial market metadata
python main.py history                           # full backfill (run overnight)
python main.py continuous --interval 300         # keep fresh
```

Or via Docker: `docker compose run --rm collector`.

---

## Pipelines

**Streaming detector** — embeds each new market, KNN-searches a faiss index, sends batched anchor-groups to an LLM that flags cross-platform same-outcome pairs, then watches WS prices for spread alerts.

```bash
python detect_arbitrage.py --interval 60
```

**Tag-based correlation** — four standalone workers sharing the DB:

```bash
python -m correlations.tagger             # LLM tags markets with causal drivers
python -m correlations.pair_builder       # Jaccard tag overlap → candidates
python -m correlations.correlator         # Pearson r over price history
python -m correlations.divergence_watcher # z-score alerts on live spread
```

**Status refresher** — bulk collectors miss settled markets; this walks oldest-checked rows and refreshes them:

```bash
python -m collectors.close_checker --batch 1000 --interval 600
```

---

## Detected Arbitrage Examples

Real pairs flagged by the streaming detector:

| # | Spread | Q1 | Q2 |
|---|---|---|---|
| 1 | 0.530 | polymarket — Set 1 Winner: Jodar vs Norrie | kalshi — Will Rafael Jodar win set 1 in the Rafael Jodar vs Cameron Norrie match |
| 2 | 0.490 | kalshi — Will Mattia Bellucci win set 1 in the Sebastian Korda vs Mattia Bellucci match | polymarket — Set 1 Winner: Korda vs Bellucci |
| 3 | 0.375 | kalshi — Will Omega win map 1 in the Omega vs. Acend match? | polymarket — Counter-Strike: Acend vs Omega - Map 1 Winner |
| 5 | 0.235 | kalshi — Maccabi Tel-Aviv vs AS Monaco Winner? | polymarket — Monaco vs. Maccabi Tel Aviv |
| 6 | 0.220 | polymarket — Will Taylor Swift release a new song in 2026? | kalshi — Will Taylor Swift release a new song 2026? |
| 7 | 0.175 | polymarket — KHL: Ak Bars Kazan vs. Lada Togliatti | kalshi — Lada Togliatti vs Ak Bars Kazan Winner? |
| 9 | 0.066 | polymarket — Will Trump say "Crazy Bernie" during the 2026 State of the Union address? | kalshi — Will Trump say "Crazy Bernie" before Apr 1, 2026? |
| 10 | 0.065 | polymarket — Will Boston Red Sox win the 2026 American League Championship Series? | kalshi — Will Boston win the 2026 Pro Baseball American League Championship? |
| 11 | 0.065 | polymarket — Will Donald Trump announce a presidential run before 2027? | kalshi — Will Donald Trump announce a run for President of the United States before Jan 1, 2027? |

See `BACKTEST_REPORT.md` for accuracy and P&L analysis.

---

## Rate Limits

| Platform | Tier | Default in client |
|---|---|---|
| Kalshi | 20 req/s | 10 |
| Polymarket Gamma | ~50 req/s | 30 |
| Polymarket CLOB | ~150 req/s | 80 |
| Polymarket Data | ~20 req/s | 15 |

Leaky-bucket limiter + exponential backoff on 429s, fail-fast on 404s.
