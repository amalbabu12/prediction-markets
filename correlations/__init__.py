"""Correlation pipeline: tag markets, find tag-overlap pairs, validate via
historical price correlation, and watch high-correlation pairs over WS for
significant price divergences. Operates on the same DB as detect_arbitrage but
in its own tables and own process so the two pipelines never block each other.
"""
