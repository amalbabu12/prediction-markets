"""Load configuration from .env and environment variables."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _read_key_file(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p.read_text()
    return None


KALSHI_API_KEY_ID: str | None = os.getenv("KALSHI_API_KEY_ID") or None
KALSHI_PRIVATE_KEY: str | None = _read_key_file(os.getenv("KALSHI_PRIVATE_KEY_PATH"))

DB_PATH: str = os.getenv("DB_PATH", "./data/markets.db")

KALSHI_RATE_LIMIT: float = float(os.getenv("KALSHI_RATE_LIMIT", "10"))
POLYMARKET_RATE_LIMIT: float = float(os.getenv("POLYMARKET_RATE_LIMIT", "30"))

