"""Identify auxiliary messages without rewriting or deleting their raw evidence.

This is a provenance check, not a topic filter. Real conversation about a
system notice, an audit, or a past promise stays conversation. Legacy generated
messages are recognized only by their envelope or known automation IDs.
"""
from __future__ import annotations

from collections.abc import Mapping


POLICY_VERSION = "memory-origin-v1"
_AUXILIARY_PURPOSES = frozenset({
    "auxiliary", "system", "tool", "tool_result", "heartbeat", "checkin",
    "compaction", "compaction_letter", "continuity", "continuity_capsule",
    "handoff_capsule", "recall_injection", "memory_maintenance",
    "task_notification", "task_stall_nudge", "memory_backup", "model_probe",
})
_AUTOMATION_IDS = ("task-ntf_", "ntf_")


def auxiliary_reason(record: Mapping[str, object]) -> str:
    """Return a bounded reason for non-dialogue input, or an empty string.

    Missing origin metadata remains eligible for legacy conversations. An
    explicit dialogue marker cannot override an existing automation envelope.
    Never inspect free-form conversation text for words like "system".
    """
    nested = record.get("payload")
    envelopes = (record, nested) if isinstance(nested, Mapping) else (record,)
    for row in envelopes:
        roles = {str(row.get(key) or "").strip().casefold() for key in ("sender", "role")}
        if roles & {"system", "tool", "developer"}:
            return "system_sender"
        if row.get("_clear") or row.get("isMeta") is True:
            return "control_record"
        automation = row.get("automation")
        if automation not in (None, False, "", 0, "false", "0"):
            return "automation"
        if row.get("auxiliary") is True:
            return "auxiliary"
        for key in ("memoryOrigin", "memory_origin", "purpose", "memoryPurpose",
                    "memory_purpose", "origin", "eventType", "event_type", "trigger"):
            value = row.get(key)
            if isinstance(value, str) and value.strip().casefold() in _AUXILIARY_PURPOSES:
                return "auxiliary_purpose"
        for key in ("id", "sourceEventId", "runId", "parentMessageId", "notificationId"):
            value = row.get(key)
            if isinstance(value, str) and value.strip().casefold().startswith(_AUTOMATION_IDS):
                return "automation_id"
    return ""


def memory_origin(record: Mapping[str, object]) -> str:
    return "auxiliary" if auxiliary_reason(record) else "dialogue"
