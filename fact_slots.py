"""Deterministic Z-axis fact-slot helpers.

Fact slots are optional metadata on ordinary Markdown buckets.  This module is
pure and read-only: it validates configured slot keys, proposes conservative
migration candidates, and filters historical versions for exact-fact recall.
It never writes frontmatter, resolves buckets, or supersedes memories.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping


FACT_STATUS_CURRENT = "current"
FACT_STATUS_HISTORICAL = "historical"
FACT_STATUS_CONTESTED = "contested"
FACT_STATUSES = frozenset({
    FACT_STATUS_CURRENT,
    FACT_STATUS_HISTORICAL,
    FACT_STATUS_CONTESTED,
})

STATE_VIEW_CURRENT = "current"
STATE_VIEW_HISTORICAL = "historical"
STATE_VIEW_TRANSITION = "transition"
STATE_VIEW_NEUTRAL = "neutral"
STATE_VIEWS = frozenset({
    STATE_VIEW_CURRENT,
    STATE_VIEW_HISTORICAL,
    STATE_VIEW_TRANSITION,
    STATE_VIEW_NEUTRAL,
})

_HISTORICAL_STATE_HINTS = (
    "以前", "过去", "上次", "历史", "当时", "之前", "曾经", "原来", "最初",
    "old", "previous", "historical", "before", "back then", "used to",
)
_CURRENT_STATE_HINTS = (
    "现在", "当前", "目前", "如今", "最新", "现用", "现行", "此刻",
    "now", "current", "currently", "latest", "at present",
)
_TRANSITION_STATE_HINTS = (
    "变化", "变成", "改成", "换成", "更新为", "取代", "替代", "搬到",
    "不再", "而是", "怎么变", "如何变", "演变", "change", "changed",
    "evolve", "evolved", "update", "updated", "replace", "replaced",
    "no longer",
)
_NEGATION_HINTS = ("不再", "不是", "并非", "没有", "never", "no longer", "not")
_SCOPE_HINTS = ("只在", "仅在", "除非", "例外", "except", "only when", "under")

# Emotional/narrative memory is not a mutable fact table.
PROTECTED_FACT_DOMAINS = frozenset({"恋爱", "纪念日", "约定", "家庭", "自省", "feel"})
NARRATIVE_FACT_TYPES = frozenset({"feel", "episode", "saga"})

# Canonical keys must be namespaced ASCII identifiers.  Human-language labels
# belong in the configured aliases, not in persisted fact_key values.
_FACT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_STRUCTURED_FACT_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?([A-Za-z0-9_.\-\u4e00-\u9fff]{1,80})\s*[:：=]\s*([^\n#;；]+?)\s*$"
)
_FACT_QUERY_CUES = (
    "现在", "当前", "目前", "如今", "最新", "现用", "现行",
    "以前", "过去", "上次", "历史", "当时", "之前", "曾经", "那次",
    "是什么", "是多少", "哪个", "哪一个", "哪种", "什么", "多少",
    "哪里", "哪儿", "是谁", "吗", "呢", "?", "？",
    "old", "previous", "historical", "before", "back then",
    "current", "currently", "latest", "what", "which",
)
_NARRATIVE_QUERY_CUES = (
    "回顾", "复盘", "时间线", "过程", "故事", "怎么变化", "如何变化",
    "recap", "summary", "timeline",
)


def _metadata(bucket: dict) -> dict:
    meta = bucket.get("metadata", {}) if isinstance(bucket, dict) else {}
    return meta if isinstance(meta, dict) else {}


def _metadata_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value]
    return []


def normalize_fact_slot_registry(registry: Mapping | None) -> dict[str, frozenset[str]]:
    """Return valid canonical keys mapped to normalized labels and aliases.

    Supported values are either a list of aliases or a mapping containing an
    ``aliases`` list.  Invalid/un-namespaced canonical keys are ignored.
    """
    normalized: dict[str, frozenset[str]] = {}
    if not isinstance(registry, Mapping):
        return normalized

    for raw_key, spec in registry.items():
        key = str(raw_key or "").strip().lower()
        if not _FACT_KEY_RE.fullmatch(key):
            continue
        if isinstance(spec, Mapping):
            aliases = spec.get("aliases", [])
        else:
            aliases = spec
        labels = {key, key.replace(".", "_")}
        labels.update(
            str(alias).strip().lower()
            for alias in _metadata_list(aliases)
            if str(alias).strip()
        )
        normalized[key] = frozenset(labels)
    return normalized


def registered_fact_query_matches(
    query: str,
    registry: Mapping | None,
) -> frozenset[str]:
    """Return registered slots explicitly targeted by a natural fact query.

    Merely mentioning an alias inside a narrative/recap request is not enough
    to turn the whole request into an exact-current fact lookup.  This keeps
    the Z gate conservative while still recognizing ordinary questions such
    as ``现在主色是什么``.  Ambiguous aliases fail open.
    """
    normalized_query = " ".join(str(query or "").strip().lower().split())
    if not normalized_query:
        return frozenset()

    matched_keys, matched_labels = _registered_fact_slot_mentions(
        normalized_query,
        registry,
    )

    if not matched_keys:
        return frozenset()

    exact_alias_query = normalized_query in matched_labels
    has_fact_cue = any(cue in normalized_query for cue in _FACT_QUERY_CUES)
    has_narrative_cue = any(cue in normalized_query for cue in _NARRATIVE_QUERY_CUES)
    if not exact_alias_query and (not has_fact_cue or has_narrative_cue):
        return frozenset()
    return frozenset(matched_keys)


def registered_fact_slot_mentions(
    query: str,
    registry: Mapping | None,
) -> frozenset[str]:
    """Return unambiguous registered slots mentioned by any query shape.

    Unlike :func:`registered_fact_query_matches`, this does not require an
    exact-fact question.  It is used by the state overlay for historical and
    transition questions such as ``主色从以前到现在怎么变了``.
    """
    normalized_query = " ".join(str(query or "").strip().lower().split())
    if not normalized_query:
        return frozenset()
    matched_keys, _ = _registered_fact_slot_mentions(normalized_query, registry)
    return frozenset(matched_keys)


def _registered_fact_slot_mentions(
    normalized_query: str,
    registry: Mapping | None,
) -> tuple[set[str], set[str]]:
    slots = normalize_fact_slot_registry(registry)
    if not slots:
        return set(), set()

    label_to_keys: defaultdict[str, set[str]] = defaultdict(set)
    for key, labels in slots.items():
        for label in labels:
            normalized_label = " ".join(str(label or "").strip().lower().split())
            if normalized_label:
                label_to_keys[normalized_label].add(key)

    matched_keys: set[str] = set()
    matched_labels: set[str] = set()
    for label, keys in label_to_keys.items():
        if len(keys) != 1 or not _query_contains_label(normalized_query, label):
            continue
        matched_labels.add(label)
        matched_keys.update(keys)
    return matched_keys, matched_labels


def profile_fact_state_query(
    query: str,
    registry: Mapping | None,
) -> dict:
    """Infer the requested state view with deterministic lexical rules.

    This is the lightweight A-TMA query-profiler shape: it never calls a
    model, and it activates only when the query unambiguously mentions a
    registered fact slot.  Plain exact-fact questions keep Ombre's established
    current-state default; narrative mentions without a state cue stay neutral.
    """
    normalized = " ".join(str(query or "").strip().lower().split())
    fact_keys = registered_fact_slot_mentions(normalized, registry)
    historical = tuple(hint for hint in _HISTORICAL_STATE_HINTS if hint in normalized)
    current = tuple(hint for hint in _CURRENT_STATE_HINTS if hint in normalized)
    transition = tuple(hint for hint in _TRANSITION_STATE_HINTS if hint in normalized)
    from_to = bool(
        re.search(r"从.{1,120}(?:到|变成|改成|换成)", normalized)
        or re.search(r"\bfrom\b.{1,160}\b(?:to|into)\b", normalized)
    )
    transition_count = len(transition) + int(from_to)

    view = STATE_VIEW_NEUTRAL
    if fact_keys:
        if transition_count and (
            transition_count >= max(len(historical), len(current))
            or (historical and current)
        ):
            view = STATE_VIEW_TRANSITION
        elif historical and current:
            view = STATE_VIEW_TRANSITION
        elif len(historical) > len(current):
            view = STATE_VIEW_HISTORICAL
        elif len(current) > len(historical):
            view = STATE_VIEW_CURRENT
        elif registered_fact_query_matches(normalized, registry):
            view = STATE_VIEW_CURRENT

    return {
        "view": view,
        "fact_keys": tuple(sorted(fact_keys)),
        "historical_hints": historical,
        "current_hints": current,
        "transition_hints": transition + (("from_to",) if from_to else ()),
        "negation": any(hint in normalized for hint in _NEGATION_HINTS),
        "scope": any(hint in normalized for hint in _SCOPE_HINTS),
        "update_wording": bool(transition_count),
    }


def _query_contains_label(query: str, label: str) -> bool:
    """Match CJK aliases by substring and ASCII aliases on token boundaries."""
    if not label:
        return False
    if re.search(r"[\u4e00-\u9fff]", label):
        return label in query
    return re.search(
        rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])",
        query,
    ) is not None


def is_fact_slot_exempt(bucket: dict) -> bool:
    """Return whether a bucket is outside automatic Z-axis semantics."""
    meta = _metadata(bucket)
    bucket_type = str(meta.get("type") or "").strip().lower()
    domains = {value.strip().lower() for value in _metadata_list(meta.get("domain"))}
    protected_domains = {value.lower() for value in PROTECTED_FACT_DOMAINS}
    return bool(
        meta.get("pinned")
        or meta.get("protected")
        or bucket_type in NARRATIVE_FACT_TYPES
        or domains & protected_domains
    )


def registered_fact_key(raw_key, registry: Mapping | None) -> str | None:
    """Accept only an exact canonical key present in the configured registry."""
    key = str(raw_key or "").strip().lower()
    slots = normalize_fact_slot_registry(registry)
    return key if key in slots else None


def fact_slot_applies_to_bucket(
    raw_key,
    bucket: dict,
    registry: Mapping | None,
) -> bool:
    """Validate one registered slot against its deterministic bucket context."""
    canonical = registered_fact_key(raw_key, registry)
    if canonical is None or is_fact_slot_exempt(bucket):
        return False
    spec = registry.get(canonical, {}) if isinstance(registry, Mapping) else {}
    return _slot_context_matches(bucket, spec)


def _slot_context_matches(bucket: dict | None, spec) -> bool:
    """Apply optional deterministic bucket constraints from one registry entry."""
    if not isinstance(spec, Mapping):
        return True
    constraint_keys = {"domains", "types", "tags_any", "name_contains"}
    if not constraint_keys.intersection(spec):
        return True
    if bucket is None:
        return False

    meta = _metadata(bucket)
    domains = {value.strip().lower() for value in _metadata_list(meta.get("domain"))}
    tags = {value.strip().lower() for value in _metadata_list(meta.get("tags"))}
    bucket_type = str(meta.get("type") or "").strip().lower()
    name = str(meta.get("name") or "").strip().lower()

    required_domains = {value.strip().lower() for value in _metadata_list(spec.get("domains"))}
    if required_domains and not domains.intersection(required_domains):
        return False
    required_types = {value.strip().lower() for value in _metadata_list(spec.get("types"))}
    if required_types and bucket_type not in required_types:
        return False
    required_tags = {value.strip().lower() for value in _metadata_list(spec.get("tags_any"))}
    if required_tags and not tags.intersection(required_tags):
        return False
    name_parts = [value.strip().lower() for value in _metadata_list(spec.get("name_contains")) if value.strip()]
    if name_parts and not any(part in name for part in name_parts):
        return False
    return True


def extract_registered_facts(
    content: str,
    registry: Mapping | None,
    *,
    bucket: dict | None = None,
) -> dict[str, list[str]]:
    """Extract configured structured labels from content without guessing.

    This is used only to propose migration candidates.  A label must exactly
    match one configured canonical key or alias.
    """
    slots = normalize_fact_slot_registry(registry)
    label_to_keys: defaultdict[str, set[str]] = defaultdict(set)
    for key, labels in slots.items():
        spec = registry.get(key, {}) if isinstance(registry, Mapping) else {}
        if not _slot_context_matches(bucket, spec):
            continue
        for label in labels:
            label_to_keys[label].add(key)

    found: defaultdict[str, list[str]] = defaultdict(list)
    for match in _STRUCTURED_FACT_RE.finditer(content or ""):
        label = match.group(1).strip().lower()
        value = match.group(2).strip()
        canonical_keys = label_to_keys.get(label, set())
        if len(canonical_keys) != 1 or not value:
            continue
        key = next(iter(canonical_keys))
        if value not in found[key]:
            found[key].append(value[:240])
    return dict(found)


def audit_fact_slots(buckets: Iterable[dict], registry: Mapping | None) -> dict:
    """Build a read-only migration and consistency report for fact slots."""
    slots = normalize_fact_slot_registry(registry)
    current_by_key: defaultdict[str, list[str]] = defaultdict(list)
    migration_candidates: list[dict] = []
    ambiguous_candidates: list[dict] = []
    invalid_fact_keys: list[dict] = []
    invalid_fact_statuses: list[dict] = []
    legacy_active_fact: list[dict] = []
    exempt_fact_metadata: list[dict] = []

    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        meta = _metadata(bucket)
        bucket_id = str(bucket.get("id") or meta.get("id") or "")
        raw_key = str(meta.get("fact_key") or "").strip()
        has_fact_metadata = bool(raw_key or "fact_status" in meta or "active_fact" in meta)

        if is_fact_slot_exempt(bucket):
            if has_fact_metadata:
                exempt_fact_metadata.append({"id": bucket_id, "fact_key": raw_key})
            continue

        if "active_fact" in meta:
            legacy_active_fact.append({"id": bucket_id, "value": meta.get("active_fact")})

        if raw_key:
            canonical = registered_fact_key(raw_key, slots)
            if canonical is None:
                invalid_fact_keys.append({"id": bucket_id, "fact_key": raw_key})
                continue
            status = str(meta.get("fact_status") or FACT_STATUS_CURRENT).strip().lower()
            if status not in FACT_STATUSES:
                invalid_fact_statuses.append({"id": bucket_id, "fact_key": canonical, "status": status})
                continue
            if status == FACT_STATUS_CURRENT:
                current_by_key[canonical].append(bucket_id)
            continue

        extracted = extract_registered_facts(
            str(bucket.get("content") or ""),
            registry,
            bucket=bucket,
        )
        if len(extracted) == 1:
            key, values = next(iter(extracted.items()))
            migration_candidates.append({"id": bucket_id, "fact_key": key, "values": values})
        elif len(extracted) > 1:
            ambiguous_candidates.append({"id": bucket_id, "fact_keys": sorted(extracted)})

    fact_conflicts = {
        key: bucket_ids
        for key, bucket_ids in sorted(current_by_key.items())
        if len(bucket_ids) > 1
    }
    return {
        "fact_conflicts": fact_conflicts,
        "migration_candidates": migration_candidates,
        "ambiguous_candidates": ambiguous_candidates,
        "invalid_fact_keys": invalid_fact_keys,
        "invalid_fact_statuses": invalid_fact_statuses,
        "legacy_active_fact": legacy_active_fact,
        "exempt_fact_metadata": exempt_fact_metadata,
    }


def filter_fact_slot_candidates(
    buckets: Iterable[dict],
    *,
    intent: str,
    registry: Mapping | None,
    fact_keys: Iterable[str] | None = None,
) -> list[dict]:
    """Hide registered historical facts only for exact-fact recall.

    Unknown keys and invalid statuses fail open so a bad migration cannot make
    memories disappear.  Protected and narrative buckets are always retained.
    ``resolved`` is deliberately ignored: it controls attention, not truth age.
    """
    candidates = list(buckets)
    if intent != "fact" or not normalize_fact_slot_registry(registry):
        return candidates

    requested_keys = {
        key
        for raw_key in (fact_keys or [])
        if (key := registered_fact_key(raw_key, registry)) is not None
    }
    kept: list[dict] = []
    for bucket in candidates:
        if is_fact_slot_exempt(bucket):
            kept.append(bucket)
            continue
        meta = _metadata(bucket)
        canonical = registered_fact_key(meta.get("fact_key"), registry)
        applies = fact_slot_applies_to_bucket(canonical, bucket, registry)
        status = str(meta.get("fact_status") or FACT_STATUS_CURRENT).strip().lower()
        slot_is_requested = not requested_keys or canonical in requested_keys
        if applies and slot_is_requested and status == FACT_STATUS_HISTORICAL:
            continue
        kept.append(bucket)
    return kept


def fact_state_label(bucket: dict, registry: Mapping | None) -> str:
    """Return one explicit, registered state label or an empty string."""
    if is_fact_slot_exempt(bucket):
        return ""
    meta = _metadata(bucket)
    canonical = registered_fact_key(meta.get("fact_key"), registry)
    if not fact_slot_applies_to_bucket(canonical, bucket, registry):
        return ""
    status = str(meta.get("fact_status") or "").strip().lower()
    return status if status in FACT_STATUSES else ""


def align_fact_state_candidates(
    buckets: Iterable[dict],
    *,
    profile: Mapping | None,
    registry: Mapping | None,
) -> list[dict]:
    """Stable-partition candidates so the requested fact state is foregrounded.

    Generic semantic order is preserved inside every partition.  Only buckets
    belonging to an explicitly mentioned registered slot can move.
    """
    candidates = list(buckets)
    profile = profile if isinstance(profile, Mapping) else {}
    view = str(profile.get("view") or STATE_VIEW_NEUTRAL)
    requested = {
        key
        for raw_key in _metadata_list(profile.get("fact_keys"))
        if (key := registered_fact_key(raw_key, registry)) is not None
    }
    if not requested or view == STATE_VIEW_NEUTRAL:
        return candidates

    def partition(bucket: dict) -> int:
        meta = _metadata(bucket)
        canonical = registered_fact_key(meta.get("fact_key"), registry)
        if canonical not in requested or not fact_slot_applies_to_bucket(
            canonical,
            bucket,
            registry,
        ):
            return 3
        status = fact_state_label(bucket, registry)
        if view == STATE_VIEW_CURRENT:
            return {
                FACT_STATUS_CURRENT: 0,
                FACT_STATUS_CONTESTED: 1,
                FACT_STATUS_HISTORICAL: 2,
            }.get(status, 3)
        if view == STATE_VIEW_HISTORICAL:
            return {
                FACT_STATUS_HISTORICAL: 0,
                FACT_STATUS_CURRENT: 1,
                FACT_STATUS_CONTESTED: 2,
            }.get(status, 3)
        if view == STATE_VIEW_TRANSITION:
            return 0 if status in {
                FACT_STATUS_CURRENT,
                FACT_STATUS_HISTORICAL,
            } else (1 if status == FACT_STATUS_CONTESTED else 3)
        return 3

    return sorted(candidates, key=partition)


def state_link_target_ids(
    bucket: dict,
    *,
    view: str,
    registry: Mapping | None,
) -> tuple[str, ...]:
    """Return explicit Z lifecycle counterparts relevant to one state view."""
    status = fact_state_label(bucket, registry)
    meta = _metadata(bucket)
    values: list[str] = []
    if status == FACT_STATUS_HISTORICAL and view in {
        STATE_VIEW_CURRENT,
        STATE_VIEW_TRANSITION,
    }:
        values.extend(_metadata_list(meta.get("superseded_by_bucket_id")))
    if status == FACT_STATUS_CURRENT and view in {
        STATE_VIEW_HISTORICAL,
        STATE_VIEW_TRANSITION,
    }:
        values.extend(_metadata_list(meta.get("supersedes_bucket_ids")))
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
