import json
from datetime import date, datetime, timezone

import pytest

import conversation_activity


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_summary_counts_only_real_user_chat_days_without_text(tmp_path):
    messages = tmp_path / "messages"
    _write_jsonl(
        messages / "room__main.jsonl",
        [
            {
                "sender": "user",
                "eventType": "chat_message",
                "recordedAt": "2026-07-30T23:30:00+08:00",
                "text": "must-not-leave-vps",
            },
            {
                "sender": "claude",
                "eventType": "chat_message",
                "recordedAt": "2026-07-30T23:31:00+08:00",
                "text": "ignored",
            },
            {
                "sender": "user",
                "eventType": "permission_request",
                "recordedAt": "2026-07-30T23:32:00+08:00",
                "text": "ignored",
            },
            # Legacy chat_message: absent eventType, full ISO ts.
            {
                "sender": "user",
                "ts": "2026-07-31T00:30:00Z",
                "text": "also-private",
            },
            # Time-only legacy rows cannot be assigned to a natural day.
            {"sender": "user", "ts": "15:00", "text": "skip-safely"},
        ],
    )
    _write_jsonl(
        messages / "room__heartbeat.jsonl",
        [{
            "sender": "user",
            "eventType": "chat_message",
            "recordedAt": "2026-07-30T10:00:00+08:00",
        }],
    )

    summary = conversation_activity.summarize_message_logs(
        messages,
        start_date=date(2026, 7, 30),
        generated_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
    )

    assert summary["daily_user_messages"] == {
        "2026-07-30": 1,
        "2026-07-31": 1,
    }
    assert summary["source"]["logs"] == ["room__main.jsonl"]
    assert summary["source"]["timestampless_user_messages"] == 1
    assert summary["source"]["contains_message_text"] is False
    encoded = json.dumps(summary, ensure_ascii=False)
    assert "must-not-leave-vps" not in encoded
    assert "also-private" not in encoded


def test_activity_summary_validation_fails_closed():
    payload = {
        "schema": conversation_activity.SCHEMA,
        "timezone": conversation_activity.TIMEZONE_NAME,
        "start_date": "2026-07-30",
        "daily_user_messages": {"2026-07-30": 1},
    }
    assert conversation_activity.validate_activity_summary(payload)[
        "daily_user_messages"
    ] == {"2026-07-30": 1}

    with pytest.raises(ValueError, match="timezone"):
        conversation_activity.validate_activity_summary({
            **payload,
            "timezone": "UTC",
        })
    with pytest.raises(ValueError, match="non-negative"):
        conversation_activity.validate_activity_summary({
            **payload,
            "daily_user_messages": {"2026-07-30": -1},
        })
