"""
End-to-end tests for arbitrage detection using actual no prices
(not 1 - yes_price).

Tests the full flow: price extraction → compute_arb → watcher callbacks → DB writes.
Includes a pipeline integration test that exercises:
  detector.process() → LLM (mocked) → _upsert_watched_pair → watcher._on_price → alert
"""
from __future__ import annotations

import math
import threading
from unittest.mock import MagicMock, patch

import pytest

from detect_arbitrage import (
    ArbitrageWatcher,
    StreamingDetector,
    _get_prices,
    _upsert_watched_pair,
    compute_arb,
)


# ── compute_arb ──────────────────────────────────────────────────────────────


class TestComputeArb:
    """Verify compute_arb uses actual no prices, not 1 - yes."""

    def test_entailment_no_arb_when_market_is_fair(self):
        # Fair market: yes=0.60, no=0.42 on both → cost > $1
        profit, _ = compute_arb(0.60, 0.42, 0.60, 0.42, is_same_outcome=True)
        assert profit < 0

    def test_entailment_arb_with_actual_no_prices(self):
        # A: yes=0.55, no=0.50  B: yes=0.60, no=0.35
        # Best: buy YES A (0.55) + NO B (0.35) = 0.90, profit = 0.10
        profit, strategy = compute_arb(0.55, 0.50, 0.60, 0.35, is_same_outcome=True)
        assert profit == pytest.approx(0.10, abs=1e-6)
        assert "BUY YES" in strategy and "BUY NO" in strategy

    def test_entailment_differs_from_naive_1_minus_yes(self):
        # If no_ask != 1 - yes_ask, the arb profit changes.
        # A: yes=0.55, no=0.50 (sum=1.05, overround)
        # B: yes=0.60, no=0.35 (sum=0.95, underround!)
        # Naive (1-yes): no_a=0.45, no_b=0.40 → profit = 1-0.55-0.40 = 0.05
        # Actual: no_b=0.35 → profit = 1-0.55-0.35 = 0.10
        naive_profit, _ = compute_arb(0.55, 1 - 0.55, 0.60, 1 - 0.60, is_same_outcome=True)
        actual_profit, _ = compute_arb(0.55, 0.50, 0.60, 0.35, is_same_outcome=True)
        assert actual_profit != pytest.approx(naive_profit, abs=1e-6)
        assert actual_profit > naive_profit

    def test_contradiction_arb(self):
        # Contradiction: A=YES ↔ B=NO
        # Buy YES both: 0.30 + 0.40 = 0.70, profit = 0.30
        profit, strategy = compute_arb(0.30, 0.75, 0.40, 0.65, is_same_outcome=False)
        assert profit == pytest.approx(0.30, abs=1e-6)
        assert "BUY YES" in strategy

    def test_contradiction_buy_no_both(self):
        # Buy NO on both is cheaper: no_a=0.20, no_b=0.25
        profit, strategy = compute_arb(0.80, 0.20, 0.80, 0.25, is_same_outcome=False)
        assert profit == pytest.approx(0.55, abs=1e-6)
        assert "BUY NO" in strategy


# ── _get_prices ──────────────────────────────────────────────────────────────


class TestGetPrices:
    def test_kalshi_returns_both_asks(self):
        market = {"yes_ask": 65, "no_ask": 38}
        yes, no = _get_prices(market, "kalshi")
        assert yes == pytest.approx(0.65)
        assert no == pytest.approx(0.38)
        # Verify no != 1 - yes (the whole point)
        assert no != pytest.approx(1 - yes)

    def test_kalshi_missing_no_ask(self):
        market = {"yes_ask": 65}
        yes, no = _get_prices(market, "kalshi")
        assert yes == pytest.approx(0.65)
        assert no is None

    def test_polymarket_returns_both_outcomes(self):
        market = {"outcomePrices": ["0.62", "0.40"]}
        yes, no = _get_prices(market, "polymarket")
        assert yes == pytest.approx(0.62)
        assert no == pytest.approx(0.40)

    def test_polymarket_nan_handling(self):
        market = {"outcomePrices": ["nan", "0.40"]}
        yes, no = _get_prices(market, "polymarket")
        assert yes is None
        assert no == pytest.approx(0.40)


# ── ArbitrageWatcher._on_price (integration) ────────────────────────────────


class TestWatcherOnPrice:
    """Test that the watcher uses actual yes/no prices and compute_arb."""

    def _make_watcher(self, sf):
        """Create a watcher with mocked WS clients."""
        kalshi_client = MagicMock()
        kalshi_client._auth_headers = MagicMock(return_value={})
        with patch("detect_arbitrage.KalshiWSClient"), \
             patch("detect_arbitrage.PolymarketWSClient"):
            watcher = ArbitrageWatcher(
                sf=sf, kalshi_client=kalshi_client,
                spread_threshold=0.04, sync_interval=9999,
            )
        return watcher

    def test_on_price_tracks_yes_and_no_separately(self):
        sf = MagicMock()
        watcher = self._make_watcher(sf)

        # Register a pair manually
        pair = {
            "db_id": 1, "is_same_outcome": True,
            "id_a": "TICKER-A", "platform_a": "kalshi", "question_a": "Q A?",
            "id_b": "CID-B", "platform_b": "polymarket", "question_b": "Q B?",
            "confidence_score": 0.95, "category": "test", "rationale": "test",
        }
        key = ("TICKER-A", "CID-B")
        watcher._pairs[key] = pair
        watcher._market_to_pairs["TICKER-A"] = [key]
        watcher._market_to_pairs["CID-B"] = [key]

        # Feed prices one at a time — should not fire until all 4 sides arrive
        watcher._on_price("TICKER-A", "yes", 0.55)
        watcher._on_price("TICKER-A", "no", 0.50)
        watcher._on_price("CID-B", "yes", 0.60)

        # Still missing CID-B no — no DB update yet
        assert watcher._prices["TICKER-A"] == {"yes": 0.55, "no": 0.50}
        assert watcher._prices["CID-B"] == {"yes": 0.60}

    def test_on_price_uses_compute_arb_with_actual_no(self):
        sf = MagicMock()
        watcher = self._make_watcher(sf)

        pair = {
            "db_id": 1, "is_same_outcome": True,
            "id_a": "TICKER-A", "platform_a": "kalshi", "question_a": "Q A?",
            "id_b": "CID-B", "platform_b": "polymarket", "question_b": "Q B?",
            "confidence_score": 0.95, "category": "test", "rationale": "test",
        }
        key = ("TICKER-A", "CID-B")
        watcher._pairs[key] = pair
        watcher._market_to_pairs["TICKER-A"] = [key]
        watcher._market_to_pairs["CID-B"] = [key]

        captured = {}
        original_print_opportunity = __import__("detect_arbitrage")._print_opportunity

        def mock_print(row_a, row_b, profit, conf, strategy):
            captured["profit"] = profit
            captured["strategy"] = strategy
            captured["row_a"] = row_a
            captured["row_b"] = row_b

        with patch("detect_arbitrage._print_opportunity", side_effect=mock_print), \
             patch("detect_arbitrage._upsert_pair"):
            # A: yes=0.55, no=0.35 → B: yes=0.60, no=0.35
            # Best: buy YES A (0.55) + NO B (0.35) = 0.90, profit = 0.10
            watcher._on_price("TICKER-A", "yes", 0.55)
            watcher._on_price("TICKER-A", "no", 0.35)
            watcher._on_price("CID-B", "yes", 0.60)
            watcher._on_price("CID-B", "no", 0.35)

        assert "profit" in captured
        assert captured["profit"] == pytest.approx(0.10, abs=1e-6)
        assert captured["row_a"]["price_no"] == 0.35
        assert captured["row_b"]["price_no"] == 0.35

    def test_on_price_no_alert_when_below_threshold(self):
        sf = MagicMock()
        watcher = self._make_watcher(sf)

        pair = {
            "db_id": 1, "is_same_outcome": True,
            "id_a": "T-A", "platform_a": "kalshi", "question_a": "Q?",
            "id_b": "C-B", "platform_b": "polymarket", "question_b": "Q?",
            "confidence_score": 0.95, "category": "test", "rationale": "test",
        }
        key = ("T-A", "C-B")
        watcher._pairs[key] = pair
        watcher._market_to_pairs["T-A"] = [key]
        watcher._market_to_pairs["C-B"] = [key]

        with patch("detect_arbitrage._print_opportunity") as mock_print, \
             patch("detect_arbitrage._upsert_pair") as mock_upsert:
            # Fair prices → no arb → no alert
            watcher._on_price("T-A", "yes", 0.55)
            watcher._on_price("T-A", "no", 0.47)
            watcher._on_price("C-B", "yes", 0.56)
            watcher._on_price("C-B", "no", 0.46)

        mock_print.assert_not_called()
        mock_upsert.assert_not_called()


# ── _upsert_watched_pair (DB integration) ────────────────────────────────────


class TestUpsertWatchedPair:
    def test_stores_actual_no_prices(self, tmp_path):
        from db.models import WatchedPair, init_db

        db_path = str(tmp_path / "test.db")
        sf = init_db(db_path)

        row_a = {
            "id": "TICKER-A", "platform": "kalshi", "question": "Q A?",
            "price_yes": 0.55, "price_no": 0.38,
            "token_id_yes": None, "token_id_no": None,
        }
        row_b = {
            "id": "CID-B", "platform": "polymarket", "question": "Q B?",
            "price_yes": 0.60, "price_no": 0.35,
            "token_id_yes": "tok-yes", "token_id_no": "tok-no",
        }

        _upsert_watched_pair(sf, row_a, row_b, True, 0.9, "politics", "same market")

        with sf() as session:
            pair = session.query(WatchedPair).one()
            assert pair.price_yes_a == pytest.approx(0.55)
            assert pair.price_no_a == pytest.approx(0.38)
            assert pair.price_yes_b == pytest.approx(0.60)
            assert pair.price_no_b == pytest.approx(0.35)
            assert pair.token_id_no_b == "tok-no"
            # Spread should use compute_arb, not abs(yes_a - yes_b)
            # Entailment best: buy YES A (0.55) + NO B (0.35) = 0.90, profit=0.10
            assert pair.spread == pytest.approx(0.10, abs=1e-6)
            assert pair.is_same_outcome is True


# ── Full pipeline: detector → LLM (mock) → watched pair → watcher → alert ───


class TestFullPipeline:
    """
    Integration test exercising the entire path:
      1. Seed DB with a Kalshi + Polymarket market on the same topic
      2. StreamingDetector.process() embeds and finds the cross-platform neighbor
      3. LLM (mocked) returns an entailment pair
      4. _upsert_watched_pair stores it with actual no prices
      5. ArbitrageWatcher._on_price receives WS-style callbacks → fires arb alert
    """

    def test_detector_to_watcher_pipeline(self, tmp_path):
        """
        Full pipeline: seed DB → bootstrap → process() → LLM (mock) →
        watched pair in DB → watcher._on_price → arb alert.
        """
        import time
        from db.models import WatchedPair, KalshiMarket, PolymarketMarket, init_db

        db_path = str(tmp_path / "pipeline.db")
        sf = init_db(db_path)

        # ── Step 1: Seed the DB with two related markets ─────────────────
        with sf() as session:
            session.add(KalshiMarket(
                ticker="KXBTC-50K", event_ticker="EV1", series_ticker="S1",
                title="Bitcoin above $50k by end of 2026?", status="active",
                yes_ask=62, no_ask=41, yes_bid=60, no_bid=39,
            ))
            session.add(PolymarketMarket(
                condition_id="0xBTC50K", question="Will Bitcoin exceed $50,000 by Dec 2026?",
                active=True, closed=False,
                price_yes=0.58, price_no=0.38,
                token_id_yes="tok-btc-yes", token_id_no="tok-btc-no",
            ))
            session.commit()

        # ── Step 2: Bootstrap detector with the seeded DB ────────────────
        mock_llm_result = [
            {
                "question_a": "Bitcoin above $50k by end of 2026?",
                "question_b": "Will Bitcoin exceed $50,000 by Dec 2026?",
                "is_same_outcome": True,
                "confidence_score": 0.95,
                "rationale": "Both ask about BTC > $50k by end of 2026",
            }
        ]

        async def fake_discover(backend, group_df):
            return ("crypto", mock_llm_result)

        backend_factory = MagicMock

        with patch("detect_arbitrage._discover_pairs_in_group", side_effect=fake_discover):
            detector = StreamingDetector(
                backend_factory=backend_factory, sf=sf,
                spread_threshold=0.02, k=10, min_confidence=0.8,
            )
            detector.bootstrap(db_path)

            assert detector._index.ntotal == 2

            # ── Step 3: Feed a *new* Kalshi market that's similar ────────
            # Use a slightly different ticker so it's treated as new, but same
            # question text so the LLM mock can match it.
            kalshi_market = {
                "ticker": "KXBTC-50K-V2",
                "title": "Bitcoin above $50k by end of 2026?",
                "yes_ask": 62, "no_ask": 41,
            }
            detector.process(kalshi_market, "kalshi")

            # Wait for the LLM worker thread to drain the queue
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with sf() as session:
                    count = session.query(WatchedPair).count()
                if count > 0:
                    break
                time.sleep(0.1)

        # ── Step 4: Verify watched pair was stored with actual no prices ──
        with sf() as session:
            pairs = session.query(WatchedPair).all()

        assert len(pairs) >= 1
        wp = pairs[0]
        assert wp.is_same_outcome is True
        assert wp.price_yes_a is not None
        assert wp.price_no_a is not None
        assert wp.price_yes_b is not None
        assert wp.price_no_b is not None

        # The actual no prices (not 1 - yes):
        # Kalshi: no_ask=41 → 0.41 (not 1-0.62=0.38)
        # Polymarket: price_no=0.38 (not 1-0.58=0.42)
        all_no_prices = {wp.price_no_a, wp.price_no_b}
        assert 0.41 in {round(p, 2) for p in all_no_prices if p is not None} or \
               0.38 in {round(p, 2) for p in all_no_prices if p is not None}

        # Spread uses compute_arb with actual no prices
        assert wp.spread is not None

        # ── Step 5: Feed prices into ArbitrageWatcher to trigger alert ───
        kalshi_client = MagicMock()
        kalshi_client._auth_headers = MagicMock(return_value={})

        with patch("detect_arbitrage.KalshiWSClient"), \
             patch("detect_arbitrage.PolymarketWSClient"):
            watcher = ArbitrageWatcher(
                sf=sf, kalshi_client=kalshi_client,
                spread_threshold=0.02, sync_interval=9999,
            )

        # Manually register the pair (normally _sync() does this)
        pair_dict = {
            "db_id": wp.id, "is_same_outcome": wp.is_same_outcome,
            "id_a": wp.id_a, "platform_a": wp.platform_a, "question_a": wp.question_a,
            "id_b": wp.id_b, "platform_b": wp.platform_b, "question_b": wp.question_b,
            "confidence_score": wp.confidence_score,
            "category": wp.category or "", "rationale": wp.rationale or "",
        }
        key = (wp.id_a, wp.id_b)
        watcher._pairs[key] = pair_dict
        watcher._market_to_pairs[wp.id_a] = [key]
        watcher._market_to_pairs[wp.id_b] = [key]

        alerts = []

        def capture_alert(row_a, row_b, profit, conf, strategy):
            alerts.append({
                "profit": profit, "strategy": strategy,
                "yes_a": row_a["price_yes"], "no_a": row_a["price_no"],
                "yes_b": row_b["price_yes"], "no_b": row_b["price_no"],
            })

        with patch("detect_arbitrage._print_opportunity", side_effect=capture_alert), \
             patch("detect_arbitrage._upsert_pair"):
            # Simulate WS delivering prices with a wider spread
            # Kalshi: yes=0.55, no=0.35 → Polymarket: yes=0.60, no=0.33
            # Best entailment: buy YES Kalshi (0.55) + NO Poly (0.33) = 0.88
            # Profit = 1.0 - 0.88 = 0.12
            watcher._on_price(wp.id_a, "yes", 0.55)
            watcher._on_price(wp.id_a, "no", 0.35)
            watcher._on_price(wp.id_b, "yes", 0.60)
            watcher._on_price(wp.id_b, "no", 0.33)

        assert len(alerts) == 1
        assert alerts[0]["profit"] == pytest.approx(0.12, abs=1e-6)
        assert alerts[0]["no_a"] == 0.35
        assert alerts[0]["no_b"] == 0.33
        assert "BUY YES" in alerts[0]["strategy"]
        assert "BUY NO" in alerts[0]["strategy"]
