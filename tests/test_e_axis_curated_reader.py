from __future__ import annotations

import os
import errno
from pathlib import Path

import pytest
import e_axis_curated_reader as curated_reader

from e_axis_curated_reader import (
    CURATED_BUCKET_ROOTS,
    EAxisCuratedError,
    iter_curated_subjects,
)


def _write_bucket(
    path: Path,
    *,
    bucket_id: str,
    content: str = "plain memory",
    created: str = "2026-07-31T00:00:00+00:00",
    bucket_type: str = "dynamic",
    tags: str = "[]",
    extra: str = "",
) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"id: {bucket_id}\n"
        f"name: memory-{bucket_id}\n"
        f"type: {bucket_type}\n"
        f"created: '{created}'\n"
        f"tags: {tags}\n"
        f"{extra}"
    )
    raw = f"---\n{header}---\n{content}".encode("utf-8")
    path.write_bytes(raw)
    return raw


def _subjects(root: Path, *, legacy_naive_timestamps_utc: bool = False):
    scan = iter_curated_subjects(
        root,
        legacy_naive_timestamps_utc=legacy_naive_timestamps_utc,
    )
    return (
        scan.subjects,
        scan.scanned,
        scan.skipped,
        scan.skip_reason_counts(),
    )


def test_scans_only_five_authoritative_roots(tmp_path):
    for ordinal, root_name in enumerate(CURATED_BUCKET_ROOTS, start=1):
        _write_bucket(
            tmp_path / root_name / "nested" / f"{ordinal}.md",
            bucket_id=f"bucket-{ordinal}",
            extra="semantic_type: preference\n",
        )
    _write_bucket(
        tmp_path / "snapshots" / "ignored.md",
        bucket_id="ignored",
        extra="semantic_type: preference\n",
    )

    subjects, scanned, skipped, reasons = _subjects(tmp_path)

    assert scanned == 5
    assert skipped == 0
    assert reasons == {}
    assert {item.source_id for item in subjects} == {
        f"bucket:bucket-{ordinal}" for ordinal in range(1, 6)
    }
    assert all(item.source_kind == "curated_memory" for item in subjects)


def test_type_priority_official_gate_relations_and_skip_reasons(tmp_path):
    root = tmp_path / "dynamic"
    _write_bucket(
        root / "semantic.md",
        bucket_id="semantic",
        extra=(
            "semantic_type: preference\n"
            "memory_type: fact\n"
        ),
        tags="[event]",
    )
    _write_bucket(
        root / "memory-type.md",
        bucket_id="memory-type",
        extra="memory_type: risk_boundary\n",
    )
    _write_bucket(
        root / "tag.md",
        bucket_id="tag",
        tags="[relationship_moment]",
    )
    _write_bucket(
        root / "event-skip.md",
        bucket_id="event-skip",
        tags="[event]",
    )
    _write_bucket(
        root / "fact-skip.md",
        bucket_id="fact-skip",
        bucket_type="fact",
    )
    _write_bucket(
        root / "keyword.md",
        bucket_id="keyword",
        content="这件事让我很焦虑。",
    )
    _write_bucket(
        root / "relation.md",
        bucket_id="relation",
        extra=(
            "relations:\n"
            "  - type: emotional_link\n"
            "    target: another-bucket\n"
        ),
    )

    subjects, scanned, skipped, reasons = _subjects(tmp_path)
    by_id = {item.source_id: item for item in subjects}

    assert scanned == 7
    assert skipped == 2
    assert reasons == {
        "gate.no_signal": 1,
        "type.fact.no_emotion": 1,
    }
    assert by_id["bucket:semantic"].memory_type == "preference"
    assert by_id["bucket:semantic"].trigger_reason == "type.preference"
    assert by_id["bucket:memory-type"].trigger_reason == "type.risk_boundary"
    assert by_id["bucket:tag"].trigger_reason == "type.relationship_moment"
    assert by_id["bucket:keyword"].trigger_reason == "keyword.emotion"
    assert by_id["bucket:relation"].relation_hints == ("emotional_link",)
    assert by_id["bucket:relation"].trigger_reason == "relation.emotional_link"


def test_oldest_first_uses_aware_timestamp_in_utc(tmp_path):
    root = tmp_path / "feel"
    _write_bucket(
        root / "later.md",
        bucket_id="later",
        created="2026-07-31T08:00:00+08:00",
        extra="semantic_type: preference\n",
    )
    _write_bucket(
        root / "earlier.md",
        bucket_id="earlier",
        created="2026-07-30T23:30:00+00:00",
        extra="semantic_type: preference\n",
    )

    subjects, *_ = _subjects(tmp_path)

    assert [item.source_id for item in subjects] == [
        "bucket:earlier",
        "bucket:later",
    ]
    assert all(item.created_at.endswith("+00:00") for item in subjects)


@pytest.mark.parametrize("explicit_false", [False, True])
def test_naive_timestamp_is_rejected_without_enabled_compatibility(
    tmp_path,
    explicit_false,
):
    _write_bucket(
        tmp_path / "feel" / "naive.md",
        bucket_id="naive",
        created="2026-07-31T00:00:00",
        extra="semantic_type: preference\n",
    )

    with pytest.raises(
        EAxisCuratedError,
        match="curated.created_at_timezone_missing",
    ):
        if explicit_false:
            _subjects(tmp_path, legacy_naive_timestamps_utc=False)
        else:
            _subjects(tmp_path)


def test_legacy_naive_utc_accepts_full_seconds_sorts_and_stays_read_only(
    tmp_path,
):
    root = tmp_path / "feel"
    quoted = root / "quoted.md"
    decoded = root / "decoded.md"
    _write_bucket(
        quoted,
        bucket_id="quoted",
        created="2026-07-31T00:00:00.123456",
        extra="semantic_type: preference\n",
    )
    decoded.parent.mkdir(parents=True, exist_ok=True)
    decoded.write_text(
        "---\n"
        "id: decoded\n"
        "name: memory-decoded\n"
        "type: dynamic\n"
        "created: 2026-07-30T23:59:59\n"
        "tags: []\n"
        "semantic_type: preference\n"
        "---\n"
        "plain memory",
        encoding="utf-8",
    )
    os.chmod(quoted, 0o640)
    os.chmod(decoded, 0o640)
    fixed_ns = 1_700_000_000_000_000_000
    os.utime(quoted, ns=(fixed_ns, fixed_ns))
    os.utime(decoded, ns=(fixed_ns, fixed_ns))
    before = {
        path: (
            path.read_bytes(),
            path.stat().st_mode,
            path.stat().st_mtime_ns,
            path.stat().st_atime_ns,
        )
        for path in (quoted, decoded)
    }
    before_entries = tuple(
        sorted(str(item.relative_to(tmp_path)) for item in tmp_path.rglob("*"))
    )

    subjects, scanned, skipped, reasons = _subjects(
        tmp_path,
        legacy_naive_timestamps_utc=True,
    )

    assert [item.source_id for item in subjects] == [
        "bucket:decoded",
        "bucket:quoted",
    ]
    assert [item.created_at for item in subjects] == [
        "2026-07-30T23:59:59.000000+00:00",
        "2026-07-31T00:00:00.123456+00:00",
    ]
    assert (scanned, skipped, reasons) == (2, 0, {})
    after_entries = tuple(
        sorted(str(item.relative_to(tmp_path)) for item in tmp_path.rglob("*"))
    )
    for path, expected in before.items():
        after = path.stat()
        assert (
            path.read_bytes(),
            after.st_mode,
            after.st_mtime_ns,
            after.st_atime_ns,
        ) == expected
    assert after_entries == before_entries
    assert not (tmp_path / ".axis").exists()


@pytest.mark.parametrize(
    "created_line",
    [
        "created: 2026-07-31\n",
        "created: '2026-07-31'\n",
        "created: '2026-07-31T00:00'\n",
    ],
)
def test_legacy_naive_utc_still_rejects_date_or_missing_seconds(
    tmp_path,
    created_line,
):
    path = tmp_path / "dynamic" / "bad.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        "id: bad-time\n"
        "name: bad-time\n"
        "type: dynamic\n"
        f"{created_line}"
        "tags: []\n"
        "semantic_type: preference\n"
        "---\n"
        "plain memory",
        encoding="utf-8",
    )

    with pytest.raises(
        EAxisCuratedError,
        match="curated.created_at_timezone_missing",
    ):
        _subjects(tmp_path, legacy_naive_timestamps_utc=True)


def test_legacy_naive_utc_still_rejects_whitespace_pollution(tmp_path):
    path = tmp_path / "dynamic" / "bad.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        "id: bad-time\n"
        "name: bad-time\n"
        "type: dynamic\n"
        "created: ' 2026-07-31T00:00:00'\n"
        "tags: []\n"
        "semantic_type: preference\n"
        "---\n"
        "plain memory",
        encoding="utf-8",
    )

    with pytest.raises(
        EAxisCuratedError,
        match="curated.created_at_invalid",
    ):
        _subjects(tmp_path, legacy_naive_timestamps_utc=True)


@pytest.mark.parametrize(
    "created",
    [
        "2026-07-31T00:00+00:00",
        "2026-07-31 00:00:00+00:00",
        "2026-07-31X00:00:00+00:00",
    ],
)
def test_timestamp_strings_require_strict_t_and_seconds_even_when_aware(
    tmp_path,
    created,
):
    _write_bucket(
        tmp_path / "feel" / "bad-shape.md",
        bucket_id="bad-shape",
        created=created,
        extra="semantic_type: preference\n",
    )

    with pytest.raises(EAxisCuratedError):
        _subjects(tmp_path, legacy_naive_timestamps_utc=True)


def test_digest_binds_only_canonical_e_input_and_run_id_is_stable(tmp_path):
    path = tmp_path / "permanent" / "memory.md"
    _write_bucket(
        path,
        bucket_id="stable",
        content="I prefer concise answers.",
        extra="semantic_type: preference\nimportance: 3\n",
    )
    first = _subjects(tmp_path)[0][0]

    _write_bucket(
        path,
        bucket_id="stable",
        content="I prefer concise answers.",
        extra="semantic_type: preference\nimportance: 9\n",
    )
    metadata_only = _subjects(tmp_path)[0][0]

    _write_bucket(
        path,
        bucket_id="stable",
        content="I prefer detailed answers.",
        extra="semantic_type: preference\nimportance: 9\n",
    )
    changed_input = _subjects(tmp_path)[0][0]

    assert first.source_digest == metadata_only.source_digest
    assert changed_input.source_digest != first.source_digest
    assert first.source_run_id == metadata_only.source_run_id
    assert changed_input.source_run_id == first.source_run_id


def test_reader_does_not_change_file_bits_or_timestamps_or_create_sidecars(tmp_path):
    path = tmp_path / "dynamic" / "memory.md"
    expected = _write_bucket(
        path,
        bucket_id="unchanged",
        content="I prefer direct answers.",
        extra="semantic_type: preference\n",
    )
    fixed_ns = 1_700_000_000_000_000_000
    os.utime(path, ns=(fixed_ns, fixed_ns))
    before = path.stat()
    before_entries = tuple(sorted(str(item.relative_to(tmp_path)) for item in tmp_path.rglob("*")))

    subjects, scanned, skipped, reasons = _subjects(tmp_path)

    after = path.stat()
    after_entries = tuple(sorted(str(item.relative_to(tmp_path)) for item in tmp_path.rglob("*")))
    assert len(subjects) == 1 and scanned == 1 and skipped == 0 and reasons == {}
    assert (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_atime_ns,
    ) == (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_atime_ns,
    )
    assert path.read_bytes() == expected
    assert after_entries == before_entries
    assert not (tmp_path / ".axis").exists()


def test_empty_existing_root_is_not_populated(tmp_path):
    before = tuple(tmp_path.iterdir())

    assert _subjects(tmp_path) == ((), 0, 0, {})

    assert tuple(tmp_path.iterdir()) == before
    assert all(not (tmp_path / name).exists() for name in CURATED_BUCKET_ROOTS)


def test_missing_root_is_rejected_without_creation(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(EAxisCuratedError, match="curated.ancestor_unsafe"):
        _subjects(missing)

    assert not missing.exists()


def test_rejects_symlink_file(tmp_path):
    target = tmp_path / "target.md"
    _write_bucket(
        target,
        bucket_id="target",
        extra="semantic_type: preference\n",
    )
    source_root = tmp_path / "dynamic"
    source_root.mkdir()
    (source_root / "linked.md").symlink_to(target)

    with pytest.raises(EAxisCuratedError, match="curated.symlink_unsafe"):
        _subjects(tmp_path)


def test_rejects_symlink_directory_ancestor(tmp_path):
    actual = tmp_path / "actual"
    _write_bucket(
        actual / "memory.md",
        bucket_id="target",
        extra="semantic_type: preference\n",
    )
    source_root = tmp_path / "dynamic"
    source_root.mkdir()
    (source_root / "linked-dir").symlink_to(actual, target_is_directory=True)

    with pytest.raises(EAxisCuratedError, match="curated.symlink_unsafe"):
        _subjects(tmp_path)


def test_rejects_hardlinked_markdown(tmp_path):
    original = tmp_path / "original.md"
    _write_bucket(
        original,
        bucket_id="hardlinked",
        extra="semantic_type: preference\n",
    )
    source_root = tmp_path / "dynamic"
    source_root.mkdir()
    os.link(original, source_root / "hardlinked.md")

    with pytest.raises(EAxisCuratedError, match="curated.hardlink_unsafe"):
        _subjects(tmp_path)


def test_rejects_symlinked_buckets_root(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(EAxisCuratedError, match="curated.ancestor_unsafe"):
        _subjects(alias)


def test_noatime_failure_is_explicit_and_never_falls_back(monkeypatch):
    real_open = curated_reader.os.open

    def deny_noatime(path, flags, mode=0o777, *, dir_fd=None):
        if flags & os.O_NOATIME:
            raise OSError(errno.EPERM, "no O_NOATIME permission")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        curated_reader,
        "_strict_platform_flags",
        lambda: (
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            os.O_NOATIME,
        ),
    )
    monkeypatch.setattr(curated_reader.os, "open", deny_noatime)

    with pytest.raises(
        EAxisCuratedError,
        match="curated.noatime_unavailable",
    ):
        curated_reader._open_directory(
            ".",
            parent_fd=None,
            noatime=True,
        )


def test_directory_swap_to_symlink_during_scan_fails_closed(
    tmp_path,
    monkeypatch,
):
    nested = tmp_path / "dynamic" / "nested"
    _write_bucket(
        nested / "inside.md",
        bucket_id="inside",
        extra="semantic_type: preference\n",
    )
    external = tmp_path / "external"
    _write_bucket(
        external / "outside.md",
        bucket_id="outside",
        extra="semantic_type: preference\n",
    )
    parked = tmp_path / "dynamic" / "parked"
    real_open_directory = curated_reader._open_directory
    swapped = False

    def swap_before_open(name, *, parent_fd, noatime, expected=None):
        nonlocal swapped
        if name == "nested" and not swapped:
            swapped = True
            nested.rename(parked)
            nested.symlink_to(external, target_is_directory=True)
        return real_open_directory(
            name,
            parent_fd=parent_fd,
            noatime=noatime,
            expected=expected,
        )

    monkeypatch.setattr(
        curated_reader,
        "_open_directory",
        swap_before_open,
    )

    with pytest.raises(EAxisCuratedError, match="curated.ancestor_unsafe"):
        _subjects(tmp_path)
    assert swapped


def test_real_root_ancestor_swap_during_open_fails_closed(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "owned"
    buckets = parent / "buckets"
    _write_bucket(
        buckets / "dynamic" / "inside.md",
        bucket_id="inside",
        extra="semantic_type: preference\n",
    )
    replacement = parent / "replacement"
    _write_bucket(
        replacement / "dynamic" / "outside.md",
        bucket_id="outside",
        extra="semantic_type: preference\n",
    )
    parked = parent / "parked"
    real_open_directory = curated_reader._open_directory
    swapped = False

    def swap_after_stat(name, *, parent_fd, noatime, expected=None):
        nonlocal swapped
        if name == "buckets" and expected is not None and not swapped:
            swapped = True
            buckets.rename(parked)
            replacement.rename(buckets)
        return real_open_directory(
            name,
            parent_fd=parent_fd,
            noatime=noatime,
            expected=expected,
        )

    monkeypatch.setattr(
        curated_reader,
        "_open_directory",
        swap_after_stat,
    )

    with pytest.raises(
        EAxisCuratedError,
        match="curated.directory_changed_during_open",
    ):
        _subjects(buckets)
    assert swapped


def test_configured_root_replacement_during_scan_fails_closed(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "owned"
    buckets = parent / "buckets"
    _write_bucket(
        buckets / "dynamic" / "inside.md",
        bucket_id="inside",
        extra="semantic_type: preference\n",
    )
    replacement = parent / "replacement"
    _write_bucket(
        replacement / "dynamic" / "outside.md",
        bucket_id="outside",
        extra="semantic_type: preference\n",
    )
    parked = parent / "parked"
    real_iter = curated_reader._iter_markdown_bytes
    swapped = False

    def swap_after_first_read(directory_fd):
        nonlocal swapped
        for raw in real_iter(directory_fd):
            yield raw
            if not swapped:
                swapped = True
                buckets.rename(parked)
                replacement.rename(buckets)

    monkeypatch.setattr(
        curated_reader,
        "_iter_markdown_bytes",
        swap_after_first_read,
    )

    with pytest.raises(
        EAxisCuratedError,
        match="curated.root_changed_during_scan",
    ):
        _subjects(buckets)
    assert swapped


@pytest.mark.parametrize(
    "header,body",
    [
        (
            "id: duplicate\nid: duplicate\nname: bad\ntype: dynamic\n"
            "created: '2026-07-31T00:00:00+00:00'\ntags: []\n",
            "焦虑",
        ),
        (
            "id: naive\nname: bad\ntype: dynamic\n"
            "created: '2026-07-31T00:00:00'\ntags: []\n",
            "焦虑",
        ),
        (
            "id: scalar-tags\nname: bad\ntype: dynamic\n"
            "created: '2026-07-31T00:00:00+00:00'\ntags: event\n",
            "焦虑",
        ),
        (
            "id: bad-relations\nname: bad\ntype: dynamic\n"
            "created: '2026-07-31T00:00:00+00:00'\ntags: []\n"
            "relations: [42]\n",
            "焦虑",
        ),
    ],
)
def test_rejects_dirty_frontmatter(tmp_path, header, body):
    path = tmp_path / "dynamic" / "bad.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"---\n{header}---\n{body}", encoding="utf-8")

    with pytest.raises(EAxisCuratedError):
        _subjects(tmp_path)


def test_rejects_duplicate_bucket_ids(tmp_path):
    _write_bucket(
        tmp_path / "dynamic" / "one.md",
        bucket_id="duplicate",
        extra="semantic_type: preference\n",
    )
    _write_bucket(
        tmp_path / "archive" / "two.md",
        bucket_id="duplicate",
        extra="semantic_type: preference\n",
    )

    with pytest.raises(EAxisCuratedError, match="curated.duplicate_id"):
        _subjects(tmp_path)
