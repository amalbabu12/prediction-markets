FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Data directory for the SQLite DB (mount a volume here)
RUN mkdir -p /data

ENV DB_PATH=/data/markets.db \
    KALSHI_RATE_LIMIT=10 \
    POLYMARKET_RATE_LIMIT=30

# Default: run a snapshot. Override CMD to run history or continuous.
CMD ["python", "main.py", "snapshot"]
