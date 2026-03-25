"""
End-to-end tests for stream_markets.py

Hit the real Kalshi and Polymarket REST APIs (no auth required for market listing).
Markets are fetched once per session via module-scoped fixtures to keep the suite fast.

Run with:
    pytest test_stream_markets_e2e.py -v -s
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketGammaClient
import stream_markets


# ── Fixtures — fetch once per session ────────────────────────────────────────

@pytest.fixture(scope="module")
def kalshi_client():
    return KalshiClient(rate_limit=5.0)


@pytest.fixture(scope="module")
def gamma_client():
    return PolymarketGammaClient(rate_limit=10.0)


@pytest.fixture(scope="module")
def kalshi_markets(kalshi_client):
    """Fetch the first page of open Kalshi markets once for the whole session."""
    # Use the internal _get directly to avoid full pagination — one page is enough.
    data = kalshi_client._get("/markets", {"status": "open", "limit": 100})
    markets = data.get("markets", [])
    assert markets, "Kalshi returned no open markets — is the exchange up?"
    return markets


@pytest.fixture(scope="module")
def poly_markets(gamma_client):
    """Fetch one page of active Polymarket markets once for the whole session."""
    data = gamma_client._get("/markets", {
        "active": "true", "closed": "false", "limit": 100,
        "order": "volume", "ascending": "false",
    })
    markets = data if isinstance(data, list) else data.get("markets", [])
    assert markets, "Polymarket returned no active markets — is the API up?"
    return markets


# ── Kalshi live data ──────────────────────────────────────────────────────────

class TestKalshiLive:
    def test_returns_at_least_one_open_market(self, kalshi_markets):
        assert len(kalshi_markets) > 0

    def test_markets_have_ticker(self, kalshi_markets):
        missing = [m for m in kalshi_markets if not m.get("ticker")]
        assert not missing, f"{len(missing)} markets missing ticker"

    def test_markets_have_title(self, kalshi_markets):
        missing = [m for m in kalshi_markets if not m.get("title")]
        assert not missing, f"{len(missing)} markets missing title"

    def test_print_kalshi_handles_live_shape(self, capsys, kalshi_markets):
        """print_kalshi must not raise on any real market dict."""
        for m in kalshi_markets[:20]:
            stream_markets.print_kalshi(m)
        out = capsys.readouterr().out
        assert out.count("KALSHI") == min(20, len(kalshi_markets))

    def test_poll_seeds_existing_markets_silently(self, capsys, kalshi_markets):
        """
        First poll pass: all current markets are loaded into seen without being printed.
        We mock iter_markets to return our pre-fetched page so no network call is made.
        """
        client = KalshiClient(rate_limit=5.0)
        client.iter_markets = lambda **kw: iter(kalshi_markets)

        with patch("stream_markets.time.sleep", side_effect=StopIteration):
            with pytest.raises(StopIteration):
                stream_markets.poll_kalshi(client, interval=30, deadline=float("inf"))

        out = capsys.readouterr().out
        assert "seeded" in out
        seeded_count = int(out.split("seeded")[1].split()[0])
        assert seeded_count == len(kalshi_markets)
        # No individual tickers in stdout before the seeded line
        before_seeded = out.split("seeded")[0]
        for m in kalshi_markets[:5]:
            assert m["ticker"] not in before_seeded

    def test_poll_detects_new_market_after_seed(self, capsys, kalshi_markets):
        """
        Second poll pass: a market not in the seed batch is printed.
        """
        fake = {
            "ticker": "__FAKE_KALSHI_TICKER__",
            "title": "Injected test market",
            "close_time": "2099-01-01T00:00:00Z",
            "yes_ask": 55,
        }
        call_no = 0
        def _iter(**kw):
            nonlocal call_no
            call_no += 1
            return iter(kalshi_markets) if call_no == 1 else iter(kalshi_markets + [fake])

        client = KalshiClient(rate_limit=5.0)
        client.iter_markets = _iter

        sleep_calls = 0
        def _stop(n):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise StopIteration

        with patch("stream_markets.time.sleep", side_effect=_stop):
            with pytest.raises(StopIteration):
                stream_markets.poll_kalshi(client, interval=1, deadline=float("inf"))

        out = capsys.readouterr().out
        assert "__FAKE_KALSHI_TICKER__" in out

        # Existing markets must NOT appear after the seeded line
        after_seed = out.split("seeded", 1)[-1]
        for m in kalshi_markets[:10]:
            assert m["ticker"] not in after_seed, (
                f"Existing market {m['ticker']} was unexpectedly re-printed"
            )


# ── Polymarket live data ──────────────────────────────────────────────────────

class TestPolymarketLive:
    def test_returns_at_least_one_active_market(self, poly_markets):
        assert len(poly_markets) > 0

    def test_markets_have_condition_id(self, poly_markets):
        missing = [m for m in poly_markets if not m.get("conditionId")]
        assert not missing, f"{len(missing)} markets missing conditionId"

    def test_markets_have_question(self, poly_markets):
        missing = [m for m in poly_markets if not m.get("question")]
        assert not missing, f"{len(missing)} markets missing question"

    def test_print_polymarket_handles_live_shape(self, capsys, poly_markets):
        """print_polymarket must not raise on any real market dict."""
        for m in poly_markets[:20]:
            stream_markets.print_polymarket(m)
        out = capsys.readouterr().out
        assert out.count("POLY") == min(20, len(poly_markets))

    def test_poll_seeds_existing_markets_silently(self, capsys, poly_markets):
        """First poll pass: all current markets are loaded silently."""
        client = PolymarketGammaClient(rate_limit=10.0)
        client.iter_markets = lambda **kw: iter(poly_markets)

        with patch("stream_markets.time.sleep", side_effect=StopIteration):
            with pytest.raises(StopIteration):
                stream_markets.poll_polymarket(client, interval=30, deadline=float("inf"))

        out = capsys.readouterr().out
        assert "seeded" in out
        seeded_count = int(out.split("seeded")[1].split()[0])
        assert seeded_count == len(poly_markets)
        before_seeded = out.split("seeded")[0]
        for m in poly_markets[:5]:
            assert m["conditionId"] not in before_seeded

    def test_poll_detects_new_market_after_seed(self, capsys, poly_markets):
        """Second poll pass: a market not in the seed batch is printed."""
        fake = {
            "conditionId": "0x__FAKE_CONDITION__",
            "question": "Injected Polymarket question?",
            "endDate": "2099-01-01T00:00:00Z",
            "volume": 0,
        }
        call_no = 0
        def _iter(**kw):
            nonlocal call_no
            call_no += 1
            # fake must come first (newest-first order) so it's seen before early-stop
            return iter(poly_markets) if call_no == 1 else iter([fake] + poly_markets)

        client = PolymarketGammaClient(rate_limit=10.0)
        client.iter_markets = _iter

        sleep_calls = 0
        def _stop(n):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise StopIteration

        with patch("stream_markets.time.sleep", side_effect=_stop):
            with pytest.raises(StopIteration):
                stream_markets.poll_polymarket(client, interval=1, deadline=float("inf"))

        out = capsys.readouterr().out
        assert "0x__FAKE_" in out

        after_seed = out.split("seeded", 1)[-1]
        for m in poly_markets[:10]:
            assert m["conditionId"] not in after_seed, (
                f"Existing market {m['conditionId']} was unexpectedly re-printed"
            )
