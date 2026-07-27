"""Strict X-axis source-provenance contracts.

Ombre already carries timeline coordinates through ``event_at`` and its
date/world/domain metadata.  This module does not replace that model.  It
adds a small, creation-time evidence chain for derived or imported buckets,
so a later update cannot silently mint or rewrite their origin.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping

from utils import normalize_event_at


X_SCHEMA_VERSION = 1

_SOURCE_KINDS = frozenset({
    "conversation",
    "episode",
    "external",
    "import",
    "saga",
})
_INPUT_FIELDS = frozenset({
    "source_kind",
    "source_buckets",
    "episode_buckets",
    "span_start",
    "span_end",
    "source_session",
    "source_thread",
    "source_event_ids",
    "source_digest",
    "source_chunk_ordinal",
})
X_METADATA_FIELDS = frozenset({
    "x_schema_version",
    *_INPUT_FIELDS,
})
_IMMUTABLE_UPDATE_FIELDS = X_METADATA_FIELDS - {"episode_buckets"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_FIELDS_BY_KIND = {
    "episode": frozenset({"source_buckets", "span_start", "span_end"}),
    "saga": frozenset({"episode_buckets"}),
    "import": frozenset({
        "source_digest",
        "source_chunk_ordinal",
        "span_start",
        "span_end",
        "source_session",
        "source_thread",
        "source_event_ids",
    }),
    "conversation": frozenset({
        "source_digest",
        "span_start",
        "span_end",
        "source_session",
        "source_thread",
        "source_event_ids",
    }),
    "external": frozenset({
        "source_digest",
        "span_start",
        "span_end",
        "source_thread",
        "source_event_ids",
    }),
}


def _clean_text(value: Any, *, field: str, max_length: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned or cleaned != value or len(cleaned) > max_length:
        raise ValueError(f"{field} must be a non-empty canonical string")
    return cleaned


def _clean_id_list(
    value: Any,
    *,
    field: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        suffix = "an array" if allow_empty else "a non-empty array"
        raise ValueError(f"{field} must be {suffix}")
    if len(value) > 4096:
        raise ValueError(f"{field} is too large")
    cleaned = [
        _clean_text(item, field=f"{field}[]")
        for item in value
    ]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field} must not contain duplicates")
    return cleaned


def _clean_span(start: Any, end: Any) -> tuple[str, str]:
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError("span_start and span_end must be ISO strings")
    if start.strip() != start or end.strip() != end:
        raise ValueError("span_start and span_end must be canonical strings")
    normalized_start, _ = normalize_event_at(start)
    normalized_end, _ = normalize_event_at(end)
    start_dt = datetime.fromisoformat(normalized_start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(normalized_end.replace("Z", "+00:00"))
    if (start_dt.tzinfo is None) != (end_dt.tzinfo is None):
        raise ValueError("span_start and span_end must use compatible timezones")
    if start_dt > end_dt:
        raise ValueError("span_start must not be after span_end")
    return normalized_start, normalized_end


def normalize_x_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and flatten a creation-time X provenance declaration.

    Unknown fields, mixed source kinds, fabricated empty identifiers, and
    partial time spans are rejected.  The returned mapping is safe to merge
    directly into bucket frontmatter before the bucket's first write.
    """

    if not isinstance(value, Mapping):
        raise ValueError("x_provenance must be an object")
    supplied = set(value)
    unknown = supplied - _INPUT_FIELDS
    if unknown:
        raise ValueError(f"unknown x_provenance fields: {sorted(unknown)}")

    source_kind = _clean_text(value.get("source_kind"), field="source_kind")
    if source_kind not in _SOURCE_KINDS:
        raise ValueError(f"unsupported source_kind: {source_kind}")

    kind_fields = supplied - {"source_kind"}
    disallowed = kind_fields - _FIELDS_BY_KIND[source_kind]
    if disallowed:
        raise ValueError(
            f"{source_kind} provenance does not allow: {sorted(disallowed)}"
        )

    result: dict[str, Any] = {
        "x_schema_version": X_SCHEMA_VERSION,
        "source_kind": source_kind,
    }

    for field in ("source_buckets", "episode_buckets", "source_event_ids"):
        if field in value:
            result[field] = _clean_id_list(value[field], field=field)

    for field in ("source_session", "source_thread"):
        if field in value:
            result[field] = _clean_text(value[field], field=field)

    has_start = "span_start" in value
    has_end = "span_end" in value
    if has_start != has_end:
        raise ValueError("span_start and span_end must be supplied together")
    if has_start:
        result["span_start"], result["span_end"] = _clean_span(
            value["span_start"], value["span_end"]
        )

    if "source_digest" in value:
        digest = _clean_text(
            value["source_digest"],
            field="source_digest",
            max_length=64,
        ).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("source_digest must be a SHA-256 hex digest")
        result["source_digest"] = digest

    if "source_chunk_ordinal" in value:
        ordinal = value["source_chunk_ordinal"]
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("source_chunk_ordinal must be a non-negative integer")
        result["source_chunk_ordinal"] = ordinal

    if source_kind == "episode":
        required = {"source_buckets", "span_start", "span_end"}
    elif source_kind == "saga":
        required = {"episode_buckets"}
    elif source_kind == "import":
        required = {"source_digest", "source_chunk_ordinal"}
    elif source_kind == "external":
        required = {"source_digest"}
    else:
        required = set()
        if not any(
            field in result
            for field in ("source_session", "source_thread", "source_event_ids")
        ):
            raise ValueError(
                "conversation provenance needs a real session, thread, or event id"
            )

    missing = required - set(result)
    if missing:
        raise ValueError(
            f"{source_kind} provenance is missing: {sorted(missing)}"
        )
    return result


def validate_x_provenance_update(
    metadata: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> None:
    """Reject provenance rewrites before a bucket update can touch activity.

    New provenance can only be declared during ``create``.  A saga's evidence
    chain is the sole exception: an existing list may grow by preserving its
    complete old prefix and adding unique episode ids.
    """

    if "x_provenance" in updates:
        raise ValueError("x_provenance is create-only")

    touched = set(updates) & X_METADATA_FIELDS
    immutable = touched & _IMMUTABLE_UPDATE_FIELDS
    if immutable:
        raise ValueError(
            f"X provenance is immutable after creation: {sorted(immutable)}"
        )
    if "episode_buckets" not in touched:
        return

    if metadata.get("type") != "saga":
        raise ValueError("episode_buckets may only be appended on saga buckets")
    if (
        "x_schema_version" in metadata
        and metadata.get("x_schema_version") != X_SCHEMA_VERSION
    ):
        raise ValueError("unsupported X provenance schema version")
    if (
        metadata.get("x_schema_version") == X_SCHEMA_VERSION
        and metadata.get("source_kind") != "saga"
    ):
        raise ValueError("episode_buckets require saga provenance")
    if "episode_buckets" not in metadata:
        raise ValueError("episode_buckets cannot be added after creation")

    old_ids = _clean_id_list(
        metadata.get("episode_buckets"),
        field="existing episode_buckets",
        allow_empty=True,
    )
    new_ids = _clean_id_list(
        updates.get("episode_buckets"),
        field="episode_buckets",
    )
    if len(new_ids) <= len(old_ids) or new_ids[: len(old_ids)] != old_ids:
        raise ValueError(
            "episode_buckets must preserve the old list as a strict prefix"
        )
