from __future__ import annotations

import copy
import hashlib
import json
import unittest

from e_axis_night import (
    EAxisNightError,
    StrictEAxisScorer,
    _eligible_lmc5_bucket,
    run_e_axis_shadow,
)
from e_axis_shadow import EAxisShadowStore


def _bucket(bucket_id: str = "bucket-1", content: str = "我们终于把事情做完了。"):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "name": "一次真实完成",
            "created": "2026-07-29T10:00:00+00:00",
            "tags": ["lmc5", "night", "event"],
            "curated_write_key": "lmc5-x:v1:" + "b" * 64,
            "curated_payload_sha256": "c" * 64,
            "vector_policy": "required",
            "lmc5_recall_state": "ready_vector",
            "x_provenance": {
                "source_kind": "conversation",
                "source_session": "dm:claude",
                "source_event_ids": ["event-1"],
                "source_digest": "a" * 64,
            },
        },
    }


def _envelope(score=None, *, finish_reason="stop"):
    score = score or {
        "valence": 0.7,
        "arousal": 0.5,
        "tension": 0.2,
        "confidence": 0.9,
        "response_tendency": "engage",
        "growth_delta": "growth",
    }
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": json.dumps(score)},
            }
        ]
    }


class _Buckets:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows)

    async def list_all(self, **kwargs):
        assert kwargs == {"include_archive": False, "include_nsfw": False}
        return copy.deepcopy(self.rows)


class _Provider:
    def __init__(self, envelope):
        self.envelope = envelope
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        assert "Do not infer hidden motives" in prompt
        return self.envelope


def _scorer(provider):
    return StrictEAxisScorer(
        provider,
        model="test-model",
        scorer_name="test-scorer",
        rubric_version="test-rubric-v1",
    )


class EAxisNightTests(unittest.IsolatedAsyncioTestCase):
    def test_only_lmc5_night_x_buckets_are_eligible(self):
        self.assertTrue(_eligible_lmc5_bucket(_bucket()))

        ordinary = _bucket()
        ordinary["metadata"]["tags"] = ["personal"]
        self.assertFalse(_eligible_lmc5_bucket(ordinary))

        missing_provenance = _bucket()
        missing_provenance["metadata"].pop("x_provenance")
        self.assertFalse(_eligible_lmc5_bucket(missing_provenance))

        missing_curated_receipt = _bucket()
        missing_curated_receipt["metadata"].pop("curated_write_key")
        self.assertFalse(_eligible_lmc5_bucket(missing_curated_receipt))

    def test_strict_scorer_rejects_incomplete_duplicate_and_extra_fields(self):
        with self.assertRaisesRegex(EAxisNightError, "provider.incomplete"):
            _scorer(_Provider(_envelope(finish_reason="length"))).score(
                title="t",
                content="c",
            )

        duplicate = (
            '{"valence":0,"valence":1,"arousal":0,"tension":0,'
            '"confidence":1,"response_tendency":"engage",'
            '"growth_delta":"stable"}'
        )
        with self.assertRaisesRegex(EAxisNightError, "provider.invalid_json"):
            _scorer(
                _Provider(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": duplicate},
                            }
                        ]
                    }
                )
            ).score(title="t", content="c")

        score = json.loads(_envelope()["choices"][0]["message"]["content"])
        score["explanation"] = "not allowed"
        with self.assertRaisesRegex(EAxisNightError, "schema.fields"):
            _scorer(_Provider(_envelope(score))).score(
                title="t",
                content="c",
            )

    async def test_shadow_run_writes_once_without_mutating_bucket(self):
        with self.subTest("first write and exact replay"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as raw:
                tmp_path = Path(raw)
                bucket = _bucket()
                manager = _Buckets([bucket])
                provider = _Provider(_envelope())
                store = EAxisShadowStore(
                    tmp_path / ".axis" / "e-shadow.jsonl",
                    maintenance_root=tmp_path,
                )
                scorer = _scorer(provider)

                first = await run_e_axis_shadow(
                    bucket_manager=manager,
                    store=store,
                    scorer=scorer,
                )
                self.assertEqual(first.added, 1)
                self.assertEqual(first.failed, 0)
                self.assertEqual(manager.rows, [bucket])
                rows = store.load()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["bucket_id"], bucket["id"])
                self.assertEqual(
                    rows[0]["source_digest"],
                    hashlib.sha256(bucket["content"].encode()).hexdigest(),
                )
                self.assertIs(rows[0]["shadow_only"], True)
                self.assertIs(rows[0]["affects_ranking"], False)

                second = await run_e_axis_shadow(
                    bucket_manager=manager,
                    store=store,
                    scorer=scorer,
                )
                self.assertEqual(second.attempted, 0)
                self.assertEqual(second.existing, 1)
                self.assertEqual(provider.calls, 1)

    async def test_shadow_run_is_bounded_and_never_sends_ordinary_bucket(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            ordinary = _bucket("private", "ordinary private memory")
            ordinary["metadata"]["tags"] = ["ordinary"]
            rows = [
                _bucket(f"lmc-{index}", f"event {index}")
                for index in range(3)
            ]
            manager = _Buckets([ordinary, *rows])
            provider = _Provider(_envelope())
            store = EAxisShadowStore(
                tmp_path / ".axis" / "e-shadow.jsonl",
                maintenance_root=tmp_path,
            )

            result = await run_e_axis_shadow(
                bucket_manager=manager,
                store=store,
                scorer=_scorer(provider),
                max_per_run=2,
            )

            self.assertEqual(result.eligible, 3)
            self.assertEqual(result.attempted, 2)
            self.assertEqual(result.added, 2)
            self.assertEqual(result.remaining, 1)
            self.assertEqual(provider.calls, 2)
            self.assertTrue(
                all(row["bucket_id"] != "private" for row in store.load())
            )

    async def test_nonretryable_failure_is_categorized_and_idempotent(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            low_confidence = {
                "valence": 0.0,
                "arousal": 0.0,
                "tension": 0.0,
                "confidence": 0.1,
                "response_tendency": "engage",
                "growth_delta": "stable",
            }
            manager = _Buckets([_bucket()])
            provider = _Provider(_envelope(low_confidence))
            store = EAxisShadowStore(
                tmp_path / ".axis" / "e-shadow.jsonl",
                maintenance_root=tmp_path,
            )
            scorer = _scorer(provider)

            first = await run_e_axis_shadow(
                bucket_manager=manager,
                store=store,
                scorer=scorer,
            )
            self.assertEqual(first.failed, 1)
            self.assertEqual(store.load()[0]["category"], "confidence.low")

            second = await run_e_axis_shadow(
                bucket_manager=manager,
                store=store,
                scorer=scorer,
            )
            self.assertEqual(second.attempted, 0)
            self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
