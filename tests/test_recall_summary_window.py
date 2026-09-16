import pytest

from tools.recall_summary_window import parse_time, summarize


def test_window_counts_only_successful_reuse_and_exposes_incomplete_time():
    def line(at, event, saved=0):
        return f"prefix recall_summary_cache at={at:.6f} event={event} key={'a'*64} saved_calls={saved}"
    rows = [line(10, "request"), line(11, "provider_start"), line(12, "computed"),
            line(13, "request"), line(14, "memory_hit", 1),
            line(15, "request"), line(16, "passthrough_deferred"),
            line(17, "failure"), line(18, "cancelled"), line(20, "memory_hit", 1)]
    result = summarize(rows, 10, 20, now=19)
    assert result["eligible_requests"] == 3
    assert result["cache_hits"] == result["saved_logical_calls"] == result["provider_invocations"] == 1
    assert result["cache_hit_rate"] == 1/3
    assert not result["window_elapsed"]
    assert summarize([], 10, 20, now=21)["cache_hit_rate"] is None
    with pytest.raises(ValueError):
        parse_time("2026-01-01T00:00:00")
