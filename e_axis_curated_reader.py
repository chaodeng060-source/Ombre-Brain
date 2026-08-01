"""Strict, read-only E0 source adapter for curated Markdown memories.

Only the five authoritative bucket roots are scanned.  The adapter does not
instantiate ``BucketManager`` because its constructor prepares runtime
directories and audit state; this reader must not create, migrate, touch, or
rewrite anything in the memory corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import errno
import stat
from collections import Counter
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from e_axis_trigger import (
    EAxisSourceError,
    EAxisSourceScan,
    EAxisSubject,
    decide_e_axis_trigger,
)
from lmc5_proposer import CANDIDATE_TYPES


CURATED_BUCKET_ROOTS = ("permanent", "dynamic", "archive", "feel", "涩涩")
_INPUT_SCHEMA = "ombre.e-axis-curated-input/v1"
_MAX_BUCKET_BYTES = 16 * 1024 * 1024
_MACHINE_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)


class EAxisCuratedError(EAxisSourceError):
    """A curated source violated the fail-closed read contract."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) is not str or not key:
            raise EAxisCuratedError("curated.frontmatter_key_invalid")
        if key in result:
            raise EAxisCuratedError("curated.frontmatter_duplicate_key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _required_text(
    value: object,
    code: str,
    *,
    maximum: int,
) -> str:
    if type(value) is not str:
        raise EAxisCuratedError(code)
    normalized = value.strip()
    if not normalized or normalized != value or len(normalized) > maximum:
        raise EAxisCuratedError(code)
    return normalized


def _canonical_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        # YAML silently turns unquoted YYYY-MM-DD into a date.  A date has no
        # timezone, so it cannot establish a strict oldest-first order.
        raise EAxisCuratedError("curated.created_at_timezone_missing")
    elif type(value) is str:
        raw = value.strip()
        if not raw or raw != value:
            raise EAxisCuratedError("curated.created_at_invalid")
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EAxisCuratedError("curated.created_at_invalid") from exc
    else:
        raise EAxisCuratedError("curated.created_at_invalid")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EAxisCuratedError("curated.created_at_timezone_missing")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _created_at(metadata: Mapping[str, Any]) -> str:
    for field in ("created", "recorded_at", "event_at"):
        if field in metadata:
            return _canonical_timestamp(metadata[field])
    raise EAxisCuratedError("curated.created_at_missing")


def _memory_type(metadata: Mapping[str, Any]) -> str:
    for field in ("semantic_type", "memory_type"):
        if field not in metadata:
            continue
        value = _required_text(
            metadata[field],
            f"curated.{field}_invalid",
            maximum=80,
        ).lower()
        if value not in CANDIDATE_TYPES:
            raise EAxisCuratedError(f"curated.{field}_unsupported")
        return value

    tags = metadata.get("tags", [])
    if type(tags) is not list:
        raise EAxisCuratedError("curated.tags_invalid")
    official_tags: list[str] = []
    for item in tags:
        tag = _required_text(item, "curated.tags_invalid", maximum=160).lower()
        if tag in CANDIDATE_TYPES and tag not in official_tags:
            official_tags.append(tag)
    if len(official_tags) > 1:
        raise EAxisCuratedError("curated.tags_ambiguous_type")
    if official_tags:
        return official_tags[0]

    return _required_text(
        metadata.get("type"),
        "curated.type_invalid",
        maximum=80,
    ).lower()


def _relation_type(item: object, *, field: str) -> str:
    if type(item) is str:
        return _required_text(
            item,
            f"curated.{field}_invalid",
            maximum=128,
        ).lower()
    if not isinstance(item, Mapping):
        raise EAxisCuratedError(f"curated.{field}_invalid")
    declared = [
        item[key]
        for key in ("relation_type", "type")
        if key in item
    ]
    if not declared:
        raise EAxisCuratedError(f"curated.{field}_invalid")
    normalized = [
        _required_text(
            value,
            f"curated.{field}_invalid",
            maximum=128,
        ).lower()
        for value in declared
    ]
    if len(set(normalized)) != 1:
        raise EAxisCuratedError(f"curated.{field}_conflict")
    return normalized[0]


def _relation_hints(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract only the gate-relevant relation type, never target or notes."""

    emotional_link = False
    for field in ("relation_hints", "relations"):
        if field not in metadata:
            continue
        rows = metadata[field]
        if type(rows) is not list:
            raise EAxisCuratedError(f"curated.{field}_invalid")
        for item in rows:
            if _relation_type(item, field=field) == "emotional_link":
                emotional_link = True
    return ("emotional_link",) if emotional_link else ()


def _strict_platform_flags() -> tuple[int, int]:
    """Return Linux/POSIX flags required for metadata-neutral safe reads."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    noatime = getattr(os, "O_NOATIME", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not noatime or not directory:
        raise EAxisCuratedError("curated.platform_unsupported")
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise EAxisCuratedError("curated.platform_unsupported")
    common = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
    return common, noatime


def _open_directory(
    name: str | os.PathLike[str],
    *,
    parent_fd: int | None,
    noatime: bool,
    expected: os.stat_result | None = None,
) -> int:
    common, noatime_flag = _strict_platform_flags()
    flags = common | getattr(os, "O_DIRECTORY")
    if noatime:
        flags |= noatime_flag
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if noatime and exc.errno in {
            errno.EPERM,
            errno.EACCES,
            errno.EINVAL,
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }:
            raise EAxisCuratedError("curated.noatime_unavailable") from exc
        raise EAxisCuratedError("curated.ancestor_unsafe") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise EAxisCuratedError("curated.ancestor_unsafe")
        if expected is not None and (
            opened.st_dev,
            opened.st_ino,
        ) != (expected.st_dev, expected.st_ino):
            raise EAxisCuratedError("curated.directory_changed_during_open")
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _open_root_directory(path: Path) -> int:
    """Open every absolute component without following any symlink."""

    absolute = Path(os.path.abspath(path))
    descriptor = _open_directory(
        absolute.anchor or os.sep,
        parent_fd=None,
        noatime=False,
    )
    try:
        parts = absolute.parts[1:]
        if not parts:
            raise EAxisCuratedError("curated.ancestor_unsafe")
        for index, component in enumerate(parts):
            try:
                expected = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise EAxisCuratedError("curated.ancestor_unsafe") from exc
            if stat.S_ISLNK(expected.st_mode) or not stat.S_ISDIR(
                expected.st_mode
            ):
                raise EAxisCuratedError("curated.ancestor_unsafe")
            child = _open_directory(
                component,
                parent_fd=descriptor,
                noatime=index == len(parts) - 1,
                expected=expected,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _entry_stat(directory_fd: int, name: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise EAxisCuratedError("curated.entry_unavailable") from exc


def _verify_root_reachable(path: Path, opened_fd: int) -> None:
    """Fail if the configured absolute root stopped naming this directory."""

    opened = os.fstat(opened_fd)
    verifier = _open_root_directory(path)
    try:
        current = os.fstat(verifier)
        if (opened.st_dev, opened.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            raise EAxisCuratedError("curated.root_changed_during_scan")
    finally:
        os.close(verifier)


def _read_unchanged(directory_fd: int, name: str) -> bytes:
    before = _entry_stat(directory_fd, name)
    if stat.S_ISLNK(before.st_mode):
        raise EAxisCuratedError("curated.symlink_unsafe")
    if not stat.S_ISREG(before.st_mode):
        raise EAxisCuratedError("curated.file_unsafe")
    if before.st_nlink != 1:
        raise EAxisCuratedError("curated.hardlink_unsafe")
    if before.st_size > _MAX_BUCKET_BYTES:
        raise EAxisCuratedError("curated.file_too_large")

    common, noatime = _strict_platform_flags()
    try:
        descriptor = os.open(
            name,
            common | noatime,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        if exc.errno in {
            errno.EPERM,
            errno.EACCES,
            errno.EINVAL,
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }:
            raise EAxisCuratedError("curated.noatime_unavailable") from exc
        raise EAxisCuratedError("curated.file_open_failed") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise EAxisCuratedError("curated.file_changed_during_open")
        chunks: list[bytes] = []
        remaining = _MAX_BUCKET_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_BUCKET_BYTES:
            raise EAxisCuratedError("curated.file_too_large")
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ):
            raise EAxisCuratedError("curated.file_changed_during_read")
        return raw
    finally:
        os.close(descriptor)


def _iter_markdown_bytes(directory_fd: int) -> Iterator[bytes]:
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as exc:
        raise EAxisCuratedError("curated.scan_failed") from exc
    for name in names:
        info = _entry_stat(directory_fd, name)
        if stat.S_ISLNK(info.st_mode):
            raise EAxisCuratedError("curated.symlink_unsafe")
        if stat.S_ISDIR(info.st_mode):
            child_fd = _open_directory(
                name,
                parent_fd=directory_fd,
                noatime=True,
                expected=info,
            )
            try:
                yield from _iter_markdown_bytes(child_fd)
            finally:
                os.close(child_fd)
            continue
        if Path(name).suffix.lower() != ".md":
            continue
        yield _read_unchanged(directory_fd, name)


def _parse_markdown(raw: bytes) -> tuple[dict[str, Any], str]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EAxisCuratedError("curated.utf8_invalid") from exc
    if not text.startswith("---\n"):
        raise EAxisCuratedError("curated.frontmatter_missing")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise EAxisCuratedError("curated.frontmatter_unterminated")
    header = text[4:end]
    content = text[end + 5:]
    if not content.strip():
        raise EAxisCuratedError("curated.content_empty")
    try:
        metadata = yaml.load(header, Loader=_UniqueKeyLoader)
    except EAxisCuratedError:
        raise
    except yaml.YAMLError as exc:
        raise EAxisCuratedError("curated.frontmatter_invalid") from exc
    if type(metadata) is not dict:
        raise EAxisCuratedError("curated.frontmatter_invalid")
    return metadata, content


def _bucket_id(metadata: Mapping[str, Any]) -> str:
    value = _required_text(
        metadata.get("id"),
        "curated.id_invalid",
        maximum=128,
    )
    if value[0] not in _MACHINE_ID_CHARS or any(
        character not in _MACHINE_ID_CHARS for character in value
    ):
        raise EAxisCuratedError("curated.id_invalid")
    return value


def _title(metadata: Mapping[str, Any], bucket_id: str) -> str:
    for field in ("name", "title"):
        if field in metadata:
            return _required_text(
                metadata[field],
                f"curated.{field}_invalid",
                maximum=512,
            )
    return bucket_id


def _subject_from_file(
    raw: bytes,
) -> tuple[EAxisSubject | None, str, str]:
    metadata, content = _parse_markdown(raw)
    bucket_id = _bucket_id(metadata)
    title = _title(metadata, bucket_id)
    memory_type = _memory_type(metadata)
    relation_hints = _relation_hints(metadata)
    created_at = _created_at(metadata)
    decision = decide_e_axis_trigger(
        memory_type=memory_type,
        title=title,
        content=content,
        relation_hints=relation_hints,
    )
    if not decision.included:
        return None, decision.reason, bucket_id

    canonical_input = {
        "bucket_id": bucket_id,
        "content": content,
        "memory_type": memory_type,
        "relation_hints": list(relation_hints),
        "schema": _INPUT_SCHEMA,
        "title": title,
        "trigger_reason": decision.reason,
    }
    encoded = json.dumps(
        canonical_input,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source_run_id = "curated:" + hashlib.sha256(
        bucket_id.encode("utf-8")
    ).hexdigest()[:32]
    return EAxisSubject(
        source_id="bucket:" + bucket_id,
        source_kind="curated_memory",
        source_digest=hashlib.sha256(encoded).hexdigest(),
        source_run_id=source_run_id,
        memory_type=memory_type,
        title=title,
        content=content,
        relation_hints=relation_hints,
        created_at=created_at,
        trigger_reason=decision.reason,
    ), "", bucket_id


def iter_curated_subjects(
    buckets_dir: str | os.PathLike[str],
) -> EAxisSourceScan:
    """Return eligible curated subjects and explicit gate coverage counts."""

    root = Path(os.path.abspath(os.fspath(buckets_dir)))
    root_fd = _open_root_directory(root)
    subjects: list[EAxisSubject] = []
    scanned = 0
    skipped = 0
    skip_reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()

    try:
        try:
            root_entries = set(os.listdir(root_fd))
        except OSError as exc:
            raise EAxisCuratedError("curated.scan_failed") from exc
        for root_name in CURATED_BUCKET_ROOTS:
            if root_name not in root_entries:
                continue
            info = _entry_stat(root_fd, root_name)
            if stat.S_ISLNK(info.st_mode):
                raise EAxisCuratedError("curated.symlink_unsafe")
            if not stat.S_ISDIR(info.st_mode):
                raise EAxisCuratedError("curated.bucket_root_unsafe")
            source_fd = _open_directory(
                root_name,
                parent_fd=root_fd,
                noatime=True,
                expected=info,
            )
            try:
                for raw in _iter_markdown_bytes(source_fd):
                    subject, skip_reason, bucket_id = _subject_from_file(raw)
                    if bucket_id in seen_ids:
                        raise EAxisCuratedError("curated.duplicate_id")
                    seen_ids.add(bucket_id)
                    scanned += 1
                    if subject is None:
                        skipped += 1
                        skip_reasons[skip_reason] += 1
                    else:
                        subjects.append(subject)
            finally:
                os.close(source_fd)
        _verify_root_reachable(root, root_fd)
    finally:
        os.close(root_fd)

    subjects.sort(key=lambda item: (item.created_at, item.source_id))
    return EAxisSourceScan(
        subjects=tuple(subjects),
        scanned=scanned,
        skipped=skipped,
        skip_reasons=tuple(sorted(skip_reasons.items())),
    )


__all__ = [
    "CURATED_BUCKET_ROOTS",
    "EAxisCuratedError",
    "iter_curated_subjects",
]
