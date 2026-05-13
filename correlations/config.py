"""Tunable thresholds for the correlation pipeline."""
from __future__ import annotations

# ── Tagging ──────────────────────────────────────────────────────────────────
TAGGER_POLL_INTERVAL_SEC = 60          # how often to look for newly-fetched markets
TAGGER_BATCH_SIZE = 10                 # markets per LLM call — amortizes prompt overhead
TAG_DICT_CONTEXT_TOP_N = 100           # top-N most-used tags shown to the LLM as the rolling vocab

# ── Pair builder ─────────────────────────────────────────────────────────────
PAIR_MIN_SHARED_TAGS = 1               # confirmed: even 1 tag is enough to create a candidate

# ── Correlator ───────────────────────────────────────────────────────────────
CORRELATOR_POLL_INTERVAL_SEC = 600     # 10 min
CORRELATOR_AGE_DAYS = 3                # candidate must be ≥ 3 days old before we look back
CORRELATOR_WINDOW_DAYS = 3             # look at last 3 days of price history
CORRELATOR_MIN_SAMPLES = 50            # need at least N joint price points for Pearson
CORRELATOR_PEARSON_MIN = 0.6           # cutoff for "high correlation" → start watching

# ── Divergence watcher ───────────────────────────────────────────────────────
DIVERGENCE_Z_THRESHOLD = 2.0           # |z| > 2 logs a row in price_divergences
