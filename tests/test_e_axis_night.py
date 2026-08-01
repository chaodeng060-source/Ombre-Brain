from __future__ import annotations

import copy
import hashlib
import json
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from e_axis_night import (
    EAxisNightError,
    EAxisRunJournal,
    StrictEAxisScorer,
    _distribution,
    _result_is_healthy,
    _scorer_lineage_name,
    build_e_axis_runtime,
    run_e_axis_shadow,
)
import e_axis_night as e_axis_night_module
from e_axis_storage import EAxisStorageBusy
from e_axis_shadow import EAxisShadowStore, build_shadow_annotation


def _score(**changes):
    value = {
        "valence": -0.6,
        "arousal": 0.7,
        "tension": 0.8,
        "confidence": 0.9,
        "response_tendency": "comfort",
        "growth_delta": "growth",
    }
    value.update(changes)
    return value


def _envelope(score=None, *, finish_reason="stop"):
    return {
        "choices": [{
            "finish_reason": finish_reason,
            "message": {"content": json.dumps(score or _score())},
        }]
    }


def _candidate(
    candidate_id: int,
    *,
    memory_type: str,
    title: str,
    content: str,
    status: str = "deferred",
    created_at: str = "2026-07-29T00:00:00+00:00",
    hints=None,
):
    base_digest = hashlib.sha256(
        f"{candidate_id}:{memory_type}:{title}:{content}".encode()
    ).hexdigest()
    payload = {
        "axis": "X",
        "base_digest": base_digest,
        "candidate_ordinal": 0,
        "draft": {
            "type": memory_type,
            "title": title,
            "content": content,
            "importance": 7,
            "thread_hint": "",
            "relation_hints": list(hints or []),
            "source_chunk_ids": ["chunk-1"],
            "evidence": content,
            "risk": "normal",
        },
        "origin_run_id": "lmc5-night-20260729",
        "proposer": {
            "model": "source-model",
            "provider": "source-provider",
        },
        "schema": "ombre.lmc5-axis-candidate/v1",
        "source": {
            "created_at": created_at,
            "source_digest": "a" * 64,
        },
        "x_write_key": "lmc5-x:v1:" + base_digest,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return SimpleNamespace(
        candidate_id=candidate_id,
        axis="X",
        status=status,
        payload=raw,
        payload_digest=hashlib.sha256(raw).hexdigest(),
    )


class _Ledger:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)

    def list_candidates(self, status, *, limit, after=None):
        after = after or 0
        return tuple(
            row
            for row in self.rows
            if row.status == status and row.candidate_id > after
        )[:limit]


class _Provider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


class _FailingAttemptJournal(EAxisRunJournal):
    def append_attempt(self, row):
        raise OSError("disk unavailable")


class _FailNthAttemptJournal(EAxisRunJournal):
    def __init__(self, *args, fail_on: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_on = fail_on
        self.calls = 0

    def append_attempt(self, row):
        self.calls += 1
        if self.calls == self.fail_on:
            raise OSError("disk unavailable")
        return super().append_attempt(row)


def _scorer(provider):
    return StrictEAxisScorer(
        provider,
        provider_name="fake-provider",
        model="fake-model",
        scorer_name="fake-scorer",
        rubric_version="fake-rubric-v1",
    )


def _stored_success(
    ordinal: int,
    *,
    scored_at: str,
    provider: str = "fake-provider",
    scorer: str = "fake-scorer",
    model: str = "fake-model",
    rubric_version: str = "fake-rubric-v1",
    source_kind: str = "lmc5_candidate",
):
    row, error = build_shadow_annotation(
        bucket_id=f"candidate:{ordinal}",
        source_digest=hashlib.sha256(f"payload:{ordinal}".encode()).hexdigest(),
        source_kind=source_kind,
        source_run_id=f"source-run-{ordinal}",
        provider=provider,
        scorer=scorer,
        model=model,
        rubric_version=rubric_version,
        run_id=f"e-run-{ordinal}",
        trigger_reason=(
            "manual.api" if source_kind == "manual_bucket" else "type.preference"
        ),
        score=_score(valence=((ordinal % 11) - 5) / 5),
        scored_at=scored_at,
    )
    assert error is None
    return row


def _paths(tmp_path: Path):
    store = EAxisShadowStore(
        tmp_path / ".axis" / "e-shadow.jsonl",
        maintenance_root=tmp_path,
    )
    journal = EAxisRunJournal(
        tmp_path / ".axis",
        maintenance_root=tmp_path,
    )
    return store, journal


class EAxisNightTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_curated_markdown_is_backfilled_read_only(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "feel" / "memory.md"
            source.parent.mkdir()
            original = (
                "---\n"
                "id: curated-one\n"
                "name: 重要时刻\n"
                "type: feel\n"
                "semantic_type: relationship_moment\n"
                "created: '2026-07-01T00:00:00+00:00'\n"
                "tags: []\n"
                "---\n"
                "我们把误会说开了。api_key=sk-testSECRET123456789"
            ).encode("utf-8")
            source.write_bytes(original)
            fixed_ns = 1_700_000_000_000_000_000
            source.touch()
            import os
            os.utime(source, ns=(fixed_ns, fixed_ns))
            before = source.stat()
            provider = _Provider([_envelope()])
            store, journal = _paths(root)

            result = await run_e_axis_shadow(
                ledger=_Ledger([]),
                curated_buckets_dir=root,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-curated",
            )

            after = source.stat()
            self.assertEqual(result.scanned, 1)
            self.assertEqual(result.eligible, 1)
            self.assertEqual(result.added, 1)
            self.assertEqual(store.load()[0]["source_kind"], "curated_memory")
            self.assertNotIn("testSECRET123456789", provider.calls[0])
            self.assertIn("[REDACTED]", provider.calls[0])
            self.assertIn("我们把误会说开了", provider.calls[0])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                (after.st_mtime_ns, after.st_atime_ns),
                (before.st_mtime_ns, before.st_atime_ns),
            )

    async def test_gate_backlog_scores_only_eligible_and_persists_report(self):
        rows = [
            _candidate(
                1,
                memory_type="preference",
                title="偏好",
                content="我更喜欢安静的回答。",
            ),
            _candidate(
                2,
                memory_type="fact",
                title="端口",
                content="服务监听 8000 端口。",
            ),
            _candidate(
                3,
                memory_type="engineering_decision",
                title="事故",
                content="这次故障让我很焦虑，但决定保留回滚。",
            ),
        ]
        ledger = _Ledger(rows)
        before = copy.deepcopy(ledger.rows)
        provider = _Provider([
            _envelope(_score(valence=0.3, response_tendency="engage")),
            _envelope(_score(valence=-0.8, growth_delta="setback")),
        ])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            store, journal = _paths(Path(raw))
            result = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-1",
                max_per_run=10,
                clock=lambda: datetime(
                    2026, 7, 31, 12, 0, tzinfo=timezone.utc
                ),
            )

            self.assertEqual(result.scanned, 3)
            self.assertEqual(result.eligible, 2)
            self.assertEqual(result.skipped, 1)
            self.assertEqual(result.added, 2)
            self.assertEqual(result.failed, 0)
            self.assertFalse(result.promotion_eligible)
            self.assertEqual(ledger.rows, before)
            self.assertEqual(len(provider.calls), 2)
            self.assertNotIn("服务监听 8000", "\n".join(provider.calls))

            scores = store.load()
            self.assertEqual(len(scores), 2)
            self.assertTrue(all(row["provider"] == "fake-provider" for row in scores))
            self.assertTrue(all(row["run_id"] == "e-run-1" for row in scores))
            self.assertTrue(all(row["affects_ranking"] is False for row in scores))
            self.assertEqual(
                result.distribution["numeric"]["valence"]["min"],
                -0.8,
            )
            self.assertEqual(result.observed_natural_days, 1)

            attempts = journal.attempts_path.read_text(encoding="utf-8")
            report = journal.reports_path.read_text(encoding="utf-8")
            for source in rows:
                base_digest = json.loads(source.payload)["base_digest"]
                self.assertNotIn(base_digest, report)
            self.assertNotIn("我更喜欢安静", attempts)
            self.assertNotIn("故障让我很焦虑", report)
            parsed_report = json.loads(report)
            self.assertFalse(parsed_report["promotion_eligible"])
            self.assertEqual(
                parsed_report["promotion_guards"]["minimum_natural_days"],
                30,
            )
            self.assertEqual(parsed_report["coverage"]["eligible"], 2)
            self.assertEqual(parsed_report["coverage"]["scored"], 2)
            self.assertEqual(parsed_report["coverage"]["score_rate"], 1.0)
            self.assertFalse(
                parsed_report["coverage"]["denominator_zero"]
            )

    async def test_success_is_terminal_and_does_not_call_provider_twice(self):
        ledger = _Ledger([_candidate(
            1,
            memory_type="relationship_moment",
            title="和好",
            content="我们把误会说开了。",
        )])
        provider = _Provider([_envelope()])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            store, journal = _paths(Path(raw))
            first = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-1",
            )
            second = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-2",
            )
            self.assertEqual(first.added, 1)
            self.assertEqual(second.attempted, 0)
            self.assertEqual(second.existing_success, 1)
            self.assertEqual(len(provider.calls), 1)

    async def test_retryable_failure_is_recorded_and_retried_next_run(self):
        ledger = _Ledger([_candidate(
            1,
            memory_type="risk_boundary",
            title="边界",
            content="这条边界让我害怕。",
        )])
        provider = _Provider([TimeoutError(), _envelope()])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            store, journal = _paths(Path(raw))
            first = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-1",
            )
            second = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-2",
            )
            self.assertEqual(first.failed_retryable, 1)
            self.assertEqual(first.remaining, 1)
            self.assertEqual(second.added, 1)
            self.assertEqual(len(provider.calls), 2)
            rows = store.load()
            self.assertEqual([row["status"] for row in rows], ["failed", "success"])
            self.assertTrue(rows[0]["retryable"])
            attempts = [
                json.loads(line)
                for line in journal.attempts_path.read_text().splitlines()
            ]
            self.assertEqual(
                [row["error_code"] for row in attempts],
                ["provider.timeout", None],
            )

    async def test_two_retryable_runs_are_both_reconciled_after_attempt_gap(self):
        ledger = _Ledger([_candidate(
            1,
            memory_type="risk_boundary",
            title="边界",
            content="这条边界让我害怕。",
        )])
        provider = _Provider([TimeoutError(), TimeoutError(), _envelope()])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store, journal = _paths(root)
            first = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-retry-1",
            )
            self.assertEqual(first.failed_retryable, 1)

            fail_second_attempt = _FailNthAttemptJournal(
                root / ".axis",
                maintenance_root=root,
                fail_on=2,
            )
            with self.assertRaisesRegex(
                EAxisNightError,
                "journal.unavailable",
            ):
                await run_e_axis_shadow(
                    ledger=ledger,
                    store=store,
                    journal=fail_second_attempt,
                    scorer=_scorer(provider),
                    run_id="e-retry-2",
                )

            failed_rows = [
                row for row in store.load() if row["status"] == "failed"
            ]
            self.assertEqual(
                [row["run_id"] for row in failed_rows],
                ["e-retry-1", "e-retry-2"],
            )

            recovered = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=EAxisRunJournal(
                    root / ".axis",
                    maintenance_root=root,
                ),
                scorer=_scorer(provider),
                run_id="e-retry-3",
            )
            self.assertEqual(recovered.added, 1)
            attempts = [
                json.loads(line)
                for line in (
                    root / ".axis" / "e-shadow-attempts.jsonl"
                ).read_text().splitlines()
            ]
            failures = [row for row in attempts if row["status"] == "failed"]
            self.assertEqual(
                [row["run_id"] for row in failures],
                ["e-retry-1", "e-retry-2"],
            )
            self.assertTrue(all(
                row["error_code"] == "provider.timeout"
                and row["retryable"] is True
                for row in failures
            ))
            self.assertEqual(len(provider.calls), 3)

    async def test_terminal_schema_failure_is_not_retried_same_cohort(self):
        ledger = _Ledger([_candidate(
            1,
            memory_type="preference",
            title="偏好",
            content="我喜欢直接回答。",
        )])
        provider = _Provider([_envelope(_score(confidence=0.1))])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            store, journal = _paths(Path(raw))
            first = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-1",
            )
            second = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-2",
            )
            self.assertEqual(first.failed_terminal, 1)
            self.assertEqual(second.attempted, 0)
            self.assertEqual(second.existing_terminal, 1)
            self.assertEqual(len(provider.calls), 1)
            self.assertFalse(_result_is_healthy(second))

    async def test_oldest_backlog_is_bounded_first(self):
        ledger = _Ledger([
            _candidate(
                2,
                memory_type="preference",
                title="new",
                content="我喜欢新内容。",
                created_at="2026-07-30T00:00:00+00:00",
            ),
            _candidate(
                1,
                memory_type="preference",
                title="old",
                content="我喜欢旧内容。",
                created_at="2026-07-01T00:00:00+00:00",
            ),
        ])
        provider = _Provider([_envelope()])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            store, journal = _paths(Path(raw))
            result = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(provider),
                run_id="e-run-1",
                max_per_run=1,
            )
            self.assertEqual(result.attempted, 1)
            self.assertEqual(result.remaining, 1)
            self.assertIn('"title":"old"', provider.calls[0])

    async def test_empty_source_is_explicit_failure_not_green(self):
        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            store, journal = _paths(Path(raw))
            with self.assertRaisesRegex(EAxisNightError, "source.empty"):
                await run_e_axis_shadow(
                    ledger=_Ledger([]),
                    store=store,
                    journal=journal,
                    scorer=_scorer(_Provider([])),
                    run_id="e-run-empty",
                )

    async def test_zero_eligible_cohort_is_explicit_in_coverage(self):
        ledger = _Ledger([_candidate(
            1,
            memory_type="fact",
            title="端口",
            content="服务监听 8000 端口。",
        )])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            store, journal = _paths(Path(raw))
            result = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(_Provider([])),
                run_id="e-run-no-eligible",
            )
            self.assertEqual(result.scanned, 1)
            self.assertEqual(result.eligible, 0)
            report = json.loads(journal.reports_path.read_text())
            self.assertTrue(report["coverage"]["denominator_zero"])
            self.assertIsNone(report["coverage"]["score_rate"])

    async def test_report_breaks_down_failures_and_skip_reasons(self):
        ledger = _Ledger([
            _candidate(
                1,
                memory_type="fact",
                title="端口",
                content="服务监听 8000 端口。",
            ),
            _candidate(
                2,
                memory_type="risk_boundary",
                title="边界",
                content="这条边界让我害怕。",
            ),
        ])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            store, journal = _paths(Path(raw))
            result = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(_Provider([TimeoutError()])),
                run_id="e-run-breakdown",
            )

            expected_skips = {
                "lmc5_candidate:type.fact.no_emotion": 1,
            }
            expected_failures = {
                "count": 1,
                "by_code": {"provider.timeout": 1},
                "by_retryability": {"retryable": 1, "terminal": 0},
            }
            self.assertEqual(result.skip_reasons, expected_skips)
            self.assertEqual(
                result.distribution["failures"],
                expected_failures,
            )

            report = json.loads(journal.reports_path.read_text())
            self.assertEqual(report["skip_reasons"], expected_skips)
            self.assertEqual(
                report["distribution"]["failures"],
                expected_failures,
            )

    async def test_journal_failure_is_fatal_without_fake_score_failure(self):
        ledger = _Ledger([_candidate(
            1,
            memory_type="preference",
            title="偏好",
            content="我喜欢清楚的回答。",
        )])

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store = EAxisShadowStore(
                root / ".axis" / "e-shadow.jsonl",
                maintenance_root=root,
            )
            journal = _FailingAttemptJournal(
                root / ".axis",
                maintenance_root=root,
            )
            provider = _Provider([_envelope()])
            with self.assertRaisesRegex(
                EAxisNightError,
                "journal.unavailable",
            ):
                await run_e_axis_shadow(
                    ledger=ledger,
                    store=store,
                    journal=journal,
                    scorer=_scorer(provider),
                    run_id="e-run-journal-failure",
                )

            rows = store.load()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "success")
            self.assertFalse(any(
                row.get("category") == "journal.unavailable"
                for row in rows
            ))

            repaired = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=EAxisRunJournal(
                    root / ".axis",
                    maintenance_root=root,
                ),
                scorer=_scorer(provider),
                run_id="e-run-recovery",
            )
            self.assertEqual(repaired.attempted, 0)
            self.assertEqual(repaired.existing_success, 1)
            self.assertEqual(len(provider.calls), 1)
            attempts = [
                json.loads(line)
                for line in (
                    root / ".axis" / "e-shadow-attempts.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["run_id"], "e-run-journal-failure")

    async def test_thirty_one_days_never_auto_promotes_e0(self):
        ledger = _Ledger([_candidate(
            999,
            memory_type="preference",
            title="偏好",
            content="我喜欢清楚的回答。",
        )])
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)

        import tempfile
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            store, journal = _paths(root)
            for ordinal in range(31):
                scored_at = (start + timedelta(days=ordinal)).isoformat()
                self.assertTrue(store.append(_stored_success(
                    ordinal,
                    scored_at=scored_at,
                )))

            result = await run_e_axis_shadow(
                ledger=ledger,
                store=store,
                journal=journal,
                scorer=_scorer(_Provider([_envelope()])),
                run_id="e-run-day-31",
                clock=lambda: datetime(
                    2026, 7, 1, 12, 0, tzinfo=timezone.utc
                ),
            )
            self.assertEqual(result.observed_natural_days, 31)
            self.assertFalse(result.promotion_eligible)

            report = json.loads(journal.reports_path.read_text())
            guards = report["promotion_guards"]
            self.assertEqual(guards["minimum_natural_days"], 30)
            self.assertTrue(all(
                value is False
                for key, value in guards.items()
                if key != "minimum_natural_days"
            ))


class EAxisDistributionTests(unittest.TestCase):
    def test_same_lineage_different_provider_is_strictly_isolated(self):
        scorer = _scorer(_Provider([]))
        rows = [
            _stored_success(
                1,
                scored_at="2026-07-01T00:00:00+00:00",
            ),
            _stored_success(
                2,
                scored_at="2026-07-02T00:00:00+00:00",
            ),
            _stored_success(
                3,
                scored_at="2026-07-03T00:00:00+00:00",
                provider="other-provider",
            ),
            _stored_success(
                4,
                scored_at="2026-07-04T00:00:00+00:00",
                provider="other-provider",
            ),
        ]

        distribution, observed_days = _distribution(rows, scorer)
        self.assertEqual(distribution["numeric"]["valence"]["count"], 2)
        self.assertEqual(observed_days, 2)

    def test_manual_bucket_is_not_part_of_formal_cohort(self):
        scorer = _scorer(_Provider([]))
        rows = [
            _stored_success(
                1,
                scored_at="2026-07-01T00:00:00+00:00",
            ),
            _stored_success(
                2,
                scored_at="2026-07-02T00:00:00+00:00",
                source_kind="manual_bucket",
            ),
        ]

        distribution, observed_days = _distribution(rows, scorer)
        self.assertEqual(distribution["numeric"]["valence"]["count"], 1)
        self.assertEqual(observed_days, 1)


def test_main_returns_75_when_secure_run_lock_is_busy(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(e_axis_night_module, "load_config", lambda: {})
    monkeypatch.setattr(
        e_axis_night_module,
        "build_e_axis_runtime",
        lambda _config: (None, None, None, None, 1, tmp_path),
    )

    @contextmanager
    def busy_lock(*_args, **_kwargs):
        raise EAxisStorageBusy("busy")
        yield

    monkeypatch.setattr(
        e_axis_night_module,
        "secure_e_axis_lock",
        busy_lock,
    )

    assert e_axis_night_module.main() == 75
    assert json.loads(capsys.readouterr().out) == {
        "ok": False,
        "code": "run.busy",
    }


class EAxisRuntimeConfigTests(unittest.TestCase):
    def test_e0_is_disabled_unless_explicitly_enabled(self):
        with self.assertRaisesRegex(EAxisNightError, "config.disabled"):
            build_e_axis_runtime({
                "buckets_dir": "/does/not/matter",
                "dehydration": {},
            })

    def test_provider_name_is_required(self):
        with self.assertRaisesRegex(
            EAxisNightError,
            "config.provider_name_invalid",
        ):
            build_e_axis_runtime({
                "buckets_dir": "/does/not/matter",
                "dehydration": {
                    "api_key": "test-key",
                    "base_url": "https://example.invalid/v1",
                    "model": "test-model",
                },
                "e_axis_shadow": {"enabled": True},
            })

    def test_scorer_identity_changes_with_sampling_configuration(self):
        base = {
            "provider_name": "test-provider",
            "base_url": "https://example.invalid/v1",
            "model": "test-model",
            "rubric_version": "rubric-v1",
            "max_tokens": 512,
            "max_content_chars": 12_000,
            "min_confidence": 0.3,
            "temperature": 0.0,
        }
        first = _scorer_lineage_name(**base)
        second = _scorer_lineage_name(**base)
        changed = _scorer_lineage_name(**{**base, "temperature": 0.1})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
