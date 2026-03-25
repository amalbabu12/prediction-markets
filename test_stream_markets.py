"""
Tests for stream_markets.py

Strategy: the poll_* functions are infinite loops, so we mock time.sleep to
raise StopIteration after N calls, letting us run exactly N iterations.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

import stream_markets


# ── print_kalshi ──────────────────────────────────────────────────────────────

class TestPrintKalshi:
    def test_contains_platform_label(self, capsys):
        stream_markets.print_kalshi({})
        assert "KALSHI" in capsys.readouterr().out

    def test_formats_ticker_and_title(self, capsys):
        stream_markets.print_kalshi({
            "ticker": "KXBTC-25DEC",
            "title": "Bitcoin above $50k?",
            "close_time": "2025-12-31T00:00:00Z",
            "yes_ask": 65,
        })
        out = capsys.readouterr().out
        assert "KXBTC-25DEC" in out
        assert "yes_ask=" in out and "65" in out
        assert "2025-12-31" in out
        assert "Bitcoin above $50k?" in out

    def test_missing_fields_do_not_crash(self, capsys):
        stream_markets.print_kalshi({})
        out = capsys.readouterr().out
        assert "KALSHI" in out

    def test_title_truncated_at_60_chars(self, capsys):
        long_title = "A" * 100
        stream_markets.print_kalshi({"ticker": "T", "title": long_title})
        out = capsys.readouterr().out
        # At most 60 chars of the title should appear (the format string slices [:60])
        assert "A" * 61 not in out


# ── print_polymarket ──────────────────────────────────────────────────────────

class TestPrintPolymarket:
    def test_contains_platform_label(self, capsys):
        stream_markets.print_polymarket({})
        assert "POLY" in capsys.readouterr().out

    def test_formats_condition_id_and_question(self, capsys):
        stream_markets.print_polymarket({
            "conditionId": "0xabc123def456789",
            "question": "Will X happen?",
            "endDate": "2025-06-01T00:00:00Z",
            "volume": 12345.67,
        })
        out = capsys.readouterr().out
        assert "0xabc123" in out          # first 12 chars of conditionId
        assert "Will X happen?" in out
        assert "2025-06-01" in out
        assert "$12,346" in out           # formatted volume

    def test_large_volume_formatting(self, capsys):
        stream_markets.print_polymarket({"conditionId": "abc", "volume": 1_000_000})
        assert "$1,000,000" in capsys.readouterr().out

    def test_zero_volume(self, capsys):
        stream_markets.print_polymarket({"conditionId": "abc", "volume": 0})
        assert "$0" in capsys.readouterr().out

    def test_non_numeric_volume_does_not_crash(self, capsys):
        stream_markets.print_polymarket({"conditionId": "abc", "volume": "N/A"})
        assert "POLY" in capsys.readouterr().out

    def test_missing_fields_do_not_crash(self, capsys):
        stream_markets.print_polymarket({})
        assert "POLY" in capsys.readouterr().out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stop_after(n: int):
    """Returns a time.sleep side_effect that raises StopIteration after n calls."""
    count = 0
    def _side_effect(interval):
        nonlocal count
        count += 1
        if count >= n:
            raise StopIteration
    return _side_effect


# ── poll_kalshi ───────────────────────────────────────────────────────────────

class TestPollKalshi:
    def _make_client(self, *batches):
        """Client whose iter_markets returns each batch in turn, then repeats last."""
        client = MagicMock()
        call_no = 0
        def _iter_markets(**kwargs):
            nonlocal call_no
            idx = min(call_no, len(batches) - 1)
            call_no += 1
            return list(batches[idx])
        client.iter_markets.side_effect = _iter_markets
        return client

    def test_new_market_printed_immediately(self, capsys):
        """Markets returned on the first poll are printed right away (no seed pass)."""
        client = self._make_client([
            {"ticker": "MKT-A", "title": "A", "close_time": "", "yes_ask": 50},
            {"ticker": "MKT-B", "title": "B", "close_time": "", "yes_ask": 60},
        ])
        with patch("stream_markets.time.sleep", side_effect=_stop_after(1)):
            with pytest.raises(StopIteration):
                stream_markets.poll_kalshi(client, interval=30, deadline=float("inf"))

        out = capsys.readouterr().out
        assert "MKT-A" in out
        assert "MKT-B" in out

    def test_duplicate_market_printed_only_once(self, capsys):
        """Same market appearing in multiple passes is not printed twice."""
        batch = [{"ticker": "MKT-A", "title": "A", "close_time": "", "yes_ask": 50}]
        client = self._make_client(batch, batch)

        with patch("stream_markets.time.sleep", side_effect=_stop_after(2)):
            with pytest.raises(StopIteration):
                stream_markets.poll_kalshi(client, interval=30, deadline=float("inf"))

        out = capsys.readouterr().out
        assert out.count("MKT-A") == 1

    def test_market_without_ticker_is_skipped(self, capsys):
        """Markets missing a ticker key are not printed."""
        client = self._make_client([{"title": "No ticker here"}])
        with patch("stream_markets.time.sleep", side_effect=_stop_after(1)):
            with pytest.raises(StopIteration):
                stream_markets.poll_kalshi(client, interval=30, deadline=float("inf"))
        assert "KALSHI" not in capsys.readouterr().out

    def test_uses_min_created_ts_filter(self):
        """iter_markets is called with min_created_ts on every poll."""
        client = MagicMock()
        client.iter_markets.return_value = []

        with patch("stream_markets.time.sleep", side_effect=_stop_after(1)):
            with pytest.raises(StopIteration):
                stream_markets.poll_kalshi(client, interval=30, deadline=float("inf"), lookback=300)

        call_kwargs = client.iter_markets.call_args[1]
        assert "min_created_ts" in call_kwargs

    def test_poll_error_logged_to_stderr_and_loop_continues(self, capsys):
        """An exception during iter_markets is caught; the loop keeps going."""
        client = MagicMock()
        call_no = 0
        def _fail_then_succeed(**kwargs):
            nonlocal call_no
            call_no += 1
            if call_no == 1:
                raise ConnectionError("timeout")
            return []
        client.iter_markets.side_effect = _fail_then_succeed

        with patch("stream_markets.time.sleep", side_effect=_stop_after(2)):
            with pytest.raises(StopIteration):
                stream_markets.poll_kalshi(client, interval=30, deadline=float("inf"))

        err = capsys.readouterr().err
        assert "poll error" in err
        assert "timeout" in err

    def test_sleep_called_with_given_interval(self):
        """time.sleep is called with the interval argument."""
        client = MagicMock()
        client.iter_markets.return_value = []
        sleep_calls = []

        def capture_sleep(n):
            sleep_calls.append(n)
            raise StopIteration

        with patch("stream_markets.time.sleep", side_effect=capture_sleep):
            with pytest.raises(StopIteration):
                stream_markets.poll_kalshi(client, interval=42, deadline=float("inf"))

        assert sleep_calls == [42]


# ── poll_polymarket ───────────────────────────────────────────────────────────

class TestPollPolymarket:
    def _make_client(self, *batches):
        """Client whose iter_markets returns each batch in turn."""
        client = MagicMock()
        call_no = 0
        def _iter_markets(**kwargs):
            nonlocal call_no
            idx = min(call_no, len(batches) - 1)
            call_no += 1
            return iter(batches[idx])
        client.iter_markets.side_effect = _iter_markets
        return client

    def test_new_market_printed_immediately(self, capsys):
        """Markets returned on the first poll are printed right away."""
        client = self._make_client([
            {"conditionId": "0xAAA", "question": "Q1", "endDate": "", "volume": 0},
            {"conditionId": "0xBBB", "question": "Q2", "endDate": "", "volume": 0},
        ])
        with patch("stream_markets.time.sleep", side_effect=_stop_after(1)):
            with pytest.raises(StopIteration):
                stream_markets.poll_polymarket(client, interval=30, deadline=float("inf"))

        out = capsys.readouterr().out
        assert "0xAAA" in out
        assert "0xBBB" in out

    def test_duplicate_market_not_printed_again(self, capsys):
        client = self._make_client(
            [{"conditionId": "0xAAA", "question": "Q", "endDate": "", "volume": 0}],
            [{"conditionId": "0xAAA", "question": "Q", "endDate": "", "volume": 0}],
        )
        with patch("stream_markets.time.sleep", side_effect=_stop_after(2)):
            with pytest.raises(StopIteration):
                stream_markets.poll_polymarket(client, interval=30, deadline=float("inf"))

        out = capsys.readouterr().out
        assert out.count("0xAAA") == 1

    def test_uses_start_date_min_filter(self):
        """iter_markets is called with start_date_min on every poll."""
        client = MagicMock()
        client.iter_markets.return_value = iter([])

        with patch("stream_markets.time.sleep", side_effect=_stop_after(1)):
            with pytest.raises(StopIteration):
                stream_markets.poll_polymarket(client, interval=30, deadline=float("inf"), lookback=300)

        call_kwargs = client.iter_markets.call_args[1]
        assert "start_date_min" in call_kwargs

    def test_market_without_condition_id_is_skipped(self, capsys):
        client = self._make_client([{"question": "No conditionId"}])
        with patch("stream_markets.time.sleep", side_effect=_stop_after(1)):
            with pytest.raises(StopIteration):
                stream_markets.poll_polymarket(client, interval=30, deadline=float("inf"))
        assert "POLY" not in capsys.readouterr().out

    def test_poll_error_logged_to_stderr_and_loop_continues(self, capsys):
        client = MagicMock()
        call_no = 0
        def _fail_then_succeed(**kwargs):
            nonlocal call_no
            call_no += 1
            if call_no == 1:
                raise RuntimeError("network error")
            return iter([])
        client.iter_markets.side_effect = _fail_then_succeed

        with patch("stream_markets.time.sleep", side_effect=_stop_after(2)):
            with pytest.raises(StopIteration):
                stream_markets.poll_polymarket(client, interval=30, deadline=float("inf"))

        err = capsys.readouterr().err
        assert "poll error" in err
        assert "network error" in err

    def test_sleep_called_with_given_interval(self):
        client = MagicMock()
        client.iter_markets.return_value = iter([])
        sleep_calls = []

        def capture_sleep(n):
            sleep_calls.append(n)
            raise StopIteration

        with patch("stream_markets.time.sleep", side_effect=capture_sleep):
            with pytest.raises(StopIteration):
                stream_markets.poll_polymarket(client, interval=15, deadline=float("inf"))

        assert sleep_calls == [15]
