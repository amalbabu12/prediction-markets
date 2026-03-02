# Prediction Markets Arbitrage — Data Collector

Fetches market data from **Kalshi** and **Polymarket** and stores it in a local SQLite database for backtesting arbitrage strategies.

---

## Project Structure

```
PredictionMarkets/
├── clients/
│   ├── kalshi.py            # Kalshi Trade API v2 client
│   └── polymarket.py        # Polymarket Gamma + CLOB + Data API clients
├── collectors/
│   ├── kalshi_collector.py  # Fetches + persists Kalshi data
│   └── polymarket_collector.py  # Fetches + persists Polymarket data
├── db/
│   └── models.py            # SQLAlchemy models + DB init
├── main.py                  # CLI entry point
├── config.py                # Loads env vars
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## What Data Is Collected

### Kalshi (`https://api.elections.kalshi.com/trade-api/v2`)

| Data | Endpoint | Stored In |
|---|---|---|
| Series / categories | `GET /series` | `kalshi_series` |
| Events (all statuses) | `GET /events` | `kalshi_events` |
| Markets (all statuses) | `GET /markets` | `kalshi_markets` |
| Trade history | `GET /markets/trades` | `kalshi_trades` |
| OHLCV candlesticks (1m / 60m / 1440m) | `GET /series/{s}/markets/{t}/candlesticks` | `kalshi_candlesticks` |
| Order book snapshots | `GET /markets/{ticker}/orderbook` | `kalshi_orderbook_snaps` |

Kalshi **requires authentication** (RSA-PSS signed headers) even for market data endpoints. Create an API key at [kalshi.com/account/api](https://kalshi.com/account/api) and set `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH` in your `.env`.

Prices are in **cents (1–99)**.

### Polymarket

Polymarket exposes three APIs, all **fully public** (no auth needed for read access).

#### Gamma API (`https://gamma-api.polymarket.com`)

| Data | Endpoint | Stored In |
|---|---|---|
| Category tags | `GET /tags` | *(not persisted — used for filtering)* |
| Events | `GET /events` | `polymarket_events` |
| Markets (metadata) | `GET /markets` | `polymarket_markets` |

#### CLOB API (`https://clob.polymarket.com`)

| Data | Endpoint | Stored In |
|---|---|---|
| Market list (with token IDs) | `GET /markets` | `polymarket_markets` |
| Price history (time series) | `GET /prices-history` | `polymarket_price_hist` |
| Order book snapshots | `POST /books` (batch) | `polymarket_ob_snaps` |
| Trades | `GET /trades` | `polymarket_trades` |

#### Data API (`https://data-api.polymarket.com`)

| Data | Endpoint | Stored In |
|---|---|---|
| Trade history | `GET /trades` | `polymarket_trades` |
| Open interest | `GET /oi` | *(fetched on demand)* |

Prices are **decimal probabilities (0.0–1.0)**.

Each market has two **token IDs** (ERC1155 on Polygon): one for the YES outcome and one for NO. Price history is stored per-token.

---

## Database Schema

SQLite at `./data/markets.db` (WAL mode for concurrency).

**Kalshi tables**

| Table | Key Columns |
|---|---|
| `kalshi_series` | `ticker`, `title`, `category`, `frequency` |
| `kalshi_events` | `event_ticker`, `series_ticker`, `status`, `close_time` |
| `kalshi_markets` | `ticker`, `event_ticker`, `status`, `yes_bid`, `yes_ask`, `volume`, `result` |
| `kalshi_trades` | `trade_id`, `ticker`, `yes_price`, `count`, `taker_side`, `created_time` |
| `kalshi_candlesticks` | `ticker`, `period_interval`, `end_period_ts`, OHLC prices, `volume` |
| `kalshi_orderbook_snaps` | `ticker`, `snapshot_time`, `yes_bids` (JSON), `no_bids` (JSON) |

**Polymarket tables**

| Table | Key Columns |
|---|---|
| `polymarket_events` | `id`, `slug`, `title`, `active`, `closed`, `volume`, `end_date` |
| `polymarket_markets` | `condition_id`, `token_id_yes`, `token_id_no`, `price_yes`, `price_no`, `volume` |
| `polymarket_price_hist` | `token_id`, `fidelity`, `ts`, `price` |
| `polymarket_trades` | `condition_id`, `token_id`, `side`, `size`, `price`, `trade_time` |
| `polymarket_ob_snaps` | `token_id`, `outcome`, `snapshot_time`, `bids` (JSON), `asks` (JSON) |

Every row also stores `raw_json` with the complete API response so no data is lost.

---

## Setup

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env with your Kalshi API credentials (Polymarket needs none)
```

**.env fields:**
```
KALSHI_API_KEY_ID=your-uuid-from-kalshi-dashboard
KALSHI_PRIVATE_KEY_PATH=./kalshi_private.pem
DB_PATH=./data/markets.db
```

### 2a. Run with Python directly

```bash
pip install -r requirements.txt

# Quick snapshot (markets + orderbooks, ~5 min)
python main.py snapshot

# Full historical data (candlesticks + all trades — can take hours)
python main.py history --price-fidelity 60 --candle-intervals 60 1440

# Continuous snapshots every 5 minutes
python main.py continuous --interval 300

# Skip one platform
python main.py snapshot --no-kalshi
python main.py snapshot --no-polymarket
```

### 2b. Run with Docker

```bash
# Build image
docker compose build

# Single snapshot
docker compose run --rm collector

# Full history (run once)
docker compose --profile history run --rm history

# Continuous (background, auto-restarts)
docker compose --profile continuous up -d continuous
```

The SQLite database is mounted at `./data/markets.db` on the host so data persists across container runs.

---

## CLI Reference

```
python main.py <mode> [options]

Modes:
  snapshot     Refresh market metadata and orderbooks (fast, ~5–10 min)
  history      Full historical OHLCV and trade data (slow, run once)
  continuous   Repeat snapshot on a schedule (default: every 300s)

Options:
  --db PATH               SQLite file path (default: ./data/markets.db)
  --no-kalshi             Skip Kalshi collection
  --no-polymarket         Skip Polymarket collection
  --interval N            Seconds between continuous snapshots (default: 300)
  --start-ts UNIX         History start timestamp (default: 2023-01-01)
  --end-ts UNIX           History end timestamp (default: now)
  --price-fidelity MIN    Polymarket bar size in minutes: 1,5,10,30,60,1440 (default: 60)
  --candle-intervals MIN  Kalshi candle widths, space-separated: 1 60 1440 (default: 60 1440)
  -v / --verbose          Debug logging
```

---

## Recommended Collection Workflow

```bash
# 1. Initial metadata snapshot (fast — establishes all market rows in DB)
python main.py snapshot

# 2. Backfill all historical data (run overnight)
python main.py history --price-fidelity 60 --candle-intervals 60 1440

# 3. Keep data fresh with continuous snapshots
python main.py continuous --interval 300
```

---

## API Clients

The clients in `clients/` are usable independently of the collectors:

```python
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketGammaClient, PolymarketCLOBClient, PolymarketDataClient

# Kalshi — needs API key for all endpoints
k = KalshiClient(api_key_id="...", private_key_pem=open("key.pem").read())
markets = k.get_all_markets()                              # all statuses
candles = k.get_candlesticks("INXW", "INXW-25JAN-B4800",
                              start_ts=..., end_ts=..., period_interval=60)
trades  = list(k.iter_trades(ticker="INXW-25JAN-B4800"))

# Polymarket Gamma — fully public
gamma  = PolymarketGammaClient()
events = gamma.get_all_events(active=True, closed=False)

# Polymarket CLOB — fully public
clob    = PolymarketCLOBClient()
book    = clob.get_orderbook("TOKEN_ID")
history = clob.get_full_price_history("TOKEN_ID", fidelity=60)
books   = clob.get_orderbooks_batch(["TOKEN_A", "TOKEN_B", ...])  # batch

# Polymarket Data — fully public
data   = PolymarketDataClient()
trades = data.get_trades(market="CONDITION_ID")
```

---

## Rate Limits

| Platform | Tier | Limit | Default in client |
|---|---|---|---|
| Kalshi | Basic | 20 req/s | 10 req/s |
| Polymarket Gamma | — | ~50 req/s | 30 req/s |
| Polymarket CLOB | — | ~150 req/s | 80 req/s |
| Polymarket Data | — | ~20 req/s | 15 req/s |

All clients use a leaky-bucket rate limiter and exponential backoff on 429s.

---

## Next Steps

- [ ] Arbitrage signal detection (compare Kalshi YES prices vs Polymarket YES probabilities for equivalent events)
- [ ] WebSocket feed for real-time orderbook updates
- [ ] Backtesting framework over stored candlestick / price history data
- [ ] Scheduled collection via cron or a proper task queue
