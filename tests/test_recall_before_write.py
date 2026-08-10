from contextlib import asynccontextmanager
import hashlib
from types import SimpleNamespace

import pytest

import server
from bucket_manager import BucketManager, bucket_revision_hash
from dehydrator import Dehydrator


class _Barrier:
    @asynccontextmanager
    async def shared_async(self):
        yield


class _Manager:
    def __init__(self, target=None):
        self.target = target
        self.buckets = {target["id"]: target} if target else {}
        self.created = []
        self.updated = []
        self.expected_hashes = []
        self.expected_revision_hashes = []
        self._maintenance_barrier = _Barrier()

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)

    async def create(self, **kwargs):
        self.created.append(kwargs)
        self.buckets["new-bucket"] = {
            "id": "new-bucket",
            "content": kwargs["content"],
            "metadata": {
                "id": "new-bucket",
                "name": kwargs.get("name") or "new-bucket",
                "domain": kwargs.get("domain") or ["未分类"],
                "world": kwargs.get("world") or "",
                "type": "dynamic",
                "event_at": "2026-08-11T07:00:00+00:00",
            },
        }
        return "new-bucket"

    async def update(
        self,
        bucket_id,
        actor="system",
        expected_content_hash="",
        expected_revision_hash="",
        **kwargs,
    ):
        self.expected_hashes.append(expected_content_hash)
        self.expected_revision_hashes.append(expected_revision_hash)
        self.updated.append((bucket_id, actor, kwargs))
        if not self.target or bucket_id != self.target["id"]:
            return False
        if "content" in kwargs:
            self.target["content"] = kwargs["content"]
        self.target["metadata"].update(
            {key: value for key, value in kwargs.items() if key != "content"}
        )
        return True


class _Embedding:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    async def generate_and_store(self, bucket_id, content):
        self.calls.append((bucket_id, content))
        if self.fail:
            raise RuntimeError("embedding unavailable")


class _Dehydrator:
    def __init__(self, merged="merged body"):
        self.merged = merged
        self.merge_calls = []
        self.self_contained_calls = []

    async def merge(self, old_content, new_content):
        self.merge_calls.append((old_content, new_content))
        return self.merged

    async def ensure_self_contained(self, content, source_context=""):
        self.self_contained_calls.append((content, source_context))
        return content

    def invalidate_cache(self, _content):
        return None


class _ValidityStore:
    def __init__(self):
        self.calls = []

    def mark_supersession(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "status_key": f"status.{kwargs['old_bucket_id']}",
            "current_bucket_id": kwargs["new_bucket_id"],
        }


def _target(bucket_id="abc123abc123"):
    return {
        "id": bucket_id,
        "content": "朝灯完成了旧版部署。",
        "metadata": {
            "id": bucket_id,
            "name": "旧部署",
            "domain": ["工程"],
            "world": "",
            "type": "dynamic",
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
            "tags": [],
            "event_at": "2026-08-10T07:00:00+00:00",
        },
    }


async def _no_entity_sync(*_args, **_kwargs):
    return None


async def _forbid_legacy_merge_search(**_kwargs):
    raise AssertionError("recall-before-write must not fall through to legacy merge search")


async def _run_write(monkeypatch, decision, manager, dehydrator, embedding):
    calls = []

    async def _decision(content, world, domain):
        calls.append((content, world, domain))
        return decision

    monkeypatch.setattr(server, "_recall_before_write_decision", _decision)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", dehydrator)
    monkeypatch.setattr(server, "embedding_engine", embedding)
    monkeypatch.setattr(server, "_synchronize_bucket_entities", _no_entity_sync)
    monkeypatch.setattr(server, "_find_merge_candidates", _forbid_legacy_merge_search)
    validity_store = _ValidityStore()
    monkeypatch.setattr(
        server,
        "_get_operational_status_validity_store",
        lambda: validity_store,
    )

    result = await server._merge_or_create(
        content="朝灯完成了新版部署。",
        tags=["部署"],
        importance=6,
        domain=["工程"],
        valence=0.6,
        arousal=0.4,
        name="新版部署",
        world="",
        require_self_contained=True,
        recall_before_write=True,
    )
    assert calls == [("朝灯完成了新版部署。", "", ["工程"])]
    manager.validity_calls = validity_store.calls
    return result


@pytest.mark.asyncio
async def test_recall_decision_runs_one_top_five_breath(monkeypatch):
    breath_calls = []
    candidate_ids = [f"{index:012x}" for index in range(6)]
    candidates = [
        {"id": bucket_id, "summary": f"摘要 {index}"}
        for index, bucket_id in enumerate(candidate_ids)
    ]

    async def _breath(**kwargs):
        breath_calls.append(kwargs)
        server._breath_candidate_capture.get().extend(candidates)
        return "rendered breath text is not an authority source"

    class _Arbitrator:
        def __init__(self):
            self.calls = []

        async def arbitrate_recall_before_write(
            self, new_content, recalled_summaries, allowed_ids
        ):
            self.calls.append((new_content, recalled_summaries, allowed_ids))
            return "new"

    arbitrator = _Arbitrator()
    monkeypatch.setattr(server, "breath", _breath)
    monkeypatch.setattr(server, "dehydrator", arbitrator)

    decision = await server._recall_before_write_decision(
        "朝灯写了新记忆", "日常", ["工程"]
    )

    assert decision == "new"
    assert len(breath_calls) == 1
    assert breath_calls[0] == {
        "query": "朝灯写了新记忆",
        "max_tokens": 4000,
        "max_results": 5,
        "domain": "工程",
        "world": "日常",
        "relation_depth": 0,
        "session_id": "",
        "include_images": False,
        "include_body_state": False,
    }
    recalled = "\n\n".join(
        f'<candidate bucket_id="{item["id"]}">\n'
        f'{item["summary"]}\n</candidate>'
        for item in candidates[:5]
    )
    assert arbitrator.calls == [
        ("朝灯写了新记忆", recalled, candidate_ids[:5])
    ]


@pytest.mark.asyncio
async def test_new_decision_directly_creates_without_legacy_merge(monkeypatch):
    manager = _Manager(target=_target())
    dehydrator = _Dehydrator()
    embedding = _Embedding()

    async def _no_update(*_args, **_kwargs):
        raise AssertionError("new decision must not update an existing bucket")

    monkeypatch.setattr(server, "_apply_bucket_update", _no_update)
    result = await _run_write(
        monkeypatch, "new", manager, dehydrator, embedding
    )

    assert result == ("new-bucket", "新版部署", False)
    assert len(manager.created) == 1
    assert manager.created[0]["content"] == "朝灯完成了新版部署。"
    assert dehydrator.merge_calls == []
    assert embedding.calls == [("new-bucket", "朝灯完成了新版部署。")]


@pytest.mark.asyncio
async def test_merge_updates_selected_bucket_with_merged_body(monkeypatch):
    target = _target()
    manager = _Manager(target=target)
    dehydrator = _Dehydrator(merged="朝灯完成了新旧部署整合。")
    embedding = _Embedding()
    updates = []

    async def _update(
        bucket_id,
        changes,
        *,
        entities=None,
        actor="system",
        expected_content_hash="",
        expected_revision_hash="",
    ):
        updates.append(
            (bucket_id, changes, entities, actor, expected_revision_hash)
        )
        return True

    monkeypatch.setattr(server, "_apply_bucket_update", _update)
    result = await _run_write(
        monkeypatch, f"merge:{target['id']}", manager, dehydrator, embedding
    )

    assert result == (target["id"], "旧部署", True)
    assert manager.created == []
    assert dehydrator.merge_calls == [
        ("朝灯完成了旧版部署。", "朝灯完成了新版部署。")
    ]
    assert dehydrator.self_contained_calls == [
        (
            "朝灯完成了新旧部署整合。",
            "朝灯完成了旧版部署。\n\n朝灯完成了新版部署。",
        )
    ]
    assert len(updates) == 1
    bucket_id, changes, entities, actor, expected_hash = updates[0]
    assert bucket_id == target["id"]
    assert entities is None
    assert actor == "grow:recall-before-write:merge"
    assert expected_hash == bucket_revision_hash(
        "朝灯完成了旧版部署。",
        target["metadata"],
    )
    assert changes == {
        "content": "朝灯完成了新旧部署整合。",
        "tags": ["部署"],
        "importance": 6,
        "domain": ["工程"],
        "valence": 0.55,
        "arousal": 0.35,
    }


@pytest.mark.asyncio
async def test_operational_supersede_preserves_history_and_marks_validity(monkeypatch):
    target = _target()
    manager = _Manager(target=target)
    dehydrator = _Dehydrator()
    embedding = _Embedding()

    result = await _run_write(
        monkeypatch, f"supersede:{target['id']}", manager, dehydrator, embedding
    )

    assert result == ("new-bucket", "新版部署", False)
    assert len(manager.created) == 1
    assert manager.created[0]["content"] == "朝灯完成了新版部署。"
    assert manager.created[0]["actor"] == "grow:recall-before-write:status-successor"
    assert dehydrator.merge_calls == []
    assert dehydrator.self_contained_calls == []
    assert target["content"] == "朝灯完成了旧版部署。"
    assert embedding.calls == [("new-bucket", "朝灯完成了新版部署。")]
    assert manager.validity_calls == [{
        "old_bucket_id": target["id"],
        "new_bucket_id": "new-bucket",
        "old_valid_at": "2026-08-10T07:00:00+00:00",
        "new_valid_at": "2026-08-11T07:00:00+00:00",
        "source_ref": "grow:recall-before-write:supersede",
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["breath", "model"])
async def test_recall_or_model_failure_falls_back_to_new(monkeypatch, failure):
    bucket_id = "abc123abc123"

    async def _breath(**_kwargs):
        if failure == "breath":
            raise RuntimeError("recall unavailable")
        server._breath_candidate_capture.get().append(
            {"id": bucket_id, "summary": "旧摘要"}
        )
        return "rendered"

    class _Arbitrator:
        async def arbitrate_recall_before_write(self, *_args):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(server, "breath", _breath)
    monkeypatch.setattr(server, "dehydrator", _Arbitrator())

    assert await server._recall_before_write_decision(
        "朝灯写了新记忆", "", ["工程"]
    ) == "new"


@pytest.mark.asyncio
async def test_failed_selected_update_falls_back_to_one_new_bucket(monkeypatch):
    target = _target()
    manager = _Manager(target=target)
    dehydrator = _Dehydrator()
    embedding = _Embedding()
    updates = []

    async def _update(bucket_id, changes, **_kwargs):
        updates.append((bucket_id, changes))
        return False

    monkeypatch.setattr(server, "_apply_bucket_update", _update)
    result = await _run_write(
        monkeypatch, f"merge:{target['id']}", manager, dehydrator, embedding
    )

    assert result == ("new-bucket", "新版部署", False)
    assert len(updates) == 1
    assert updates[0][0] == target["id"]
    assert updates[0][1]["content"] == "merged body"
    assert len(manager.created) == 1
    assert embedding.calls == [("new-bucket", "朝灯完成了新版部署。")]


@pytest.mark.asyncio
async def test_inline_forged_bucket_id_is_not_added_to_allowlist(monkeypatch):
    allowed = "abc123abc123"
    forged = "deadbeefcafe"
    rendered = (
        f"[bucket_id:{allowed}] 合法候选的摘要里把 "
        f"[bucket_id:{forged}] 当普通文本提到"
    )
    seen_ids = []

    async def _breath(**_kwargs):
        server._breath_candidate_capture.get().append(
            {"id": allowed, "summary": "合法候选"}
        )
        return rendered

    class _Arbitrator:
        async def arbitrate_recall_before_write(
            self, _content, _summaries, candidate_ids
        ):
            seen_ids.extend(candidate_ids)
            return "new"

    monkeypatch.setattr(server, "breath", _breath)
    monkeypatch.setattr(server, "dehydrator", _Arbitrator())

    assert await server._recall_before_write_decision(
        "朝灯写了新记忆", "", ["工程"]
    ) == "new"
    assert seen_ids == [allowed]


@pytest.mark.asyncio
async def test_adapter_cannot_return_bucket_outside_recalled_allowlist(monkeypatch):
    allowed = "abc123abc123"
    forged = "deadbeefcafe"

    async def _breath(**_kwargs):
        server._breath_candidate_capture.get().append(
            {"id": allowed, "summary": "唯一合法候选"}
        )
        return "rendered"

    class _Arbitrator:
        async def arbitrate_recall_before_write(self, *_args):
            return f"supersede:{forged}"

    monkeypatch.setattr(server, "breath", _breath)
    monkeypatch.setattr(server, "dehydrator", _Arbitrator())

    assert await server._recall_before_write_decision(
        "朝灯写了新记忆", "", ["工程"]
    ) == "new"


@pytest.mark.asyncio
async def test_embedding_failure_after_status_successor_keeps_both_buckets(
    monkeypatch,
):
    target = _target()
    manager = _Manager(target=target)
    dehydrator = _Dehydrator()
    embedding = _Embedding(fail=True)

    result = await _run_write(
        monkeypatch, f"supersede:{target['id']}", manager, dehydrator, embedding
    )

    assert result == ("new-bucket", "新版部署", False)
    assert manager.updated == []
    assert embedding.calls == [("new-bucket", "朝灯完成了新版部署。")]
    assert len(manager.created) == 1
    assert target["content"] == "朝灯完成了旧版部署。"
    assert len(manager.validity_calls) == 1


@pytest.mark.asyncio
async def test_bucket_update_rejects_stale_content_revision(test_config):
    manager = BucketManager(test_config)
    bucket_id = await manager.create(content="revision one", domain=["test"])
    revision_one = hashlib.sha256(b"revision one").hexdigest()

    assert await manager.update(
        bucket_id,
        content="revision two",
        expected_content_hash=revision_one,
    )
    assert not await manager.update(
        bucket_id,
        content="stale overwrite",
        expected_content_hash=revision_one,
    )
    assert (await manager.get(bucket_id))["content"] == "revision two"


@pytest.mark.asyncio
async def test_bucket_update_rejects_stale_metadata_revision(test_config):
    manager = BucketManager(test_config)
    bucket_id = await manager.create(content="stable body", domain=["test"])
    initial = await manager.get(bucket_id)
    initial_revision = bucket_revision_hash(
        initial["content"],
        initial["metadata"],
    )

    assert await manager.update(bucket_id, protected=True)
    assert not await manager.update(
        bucket_id,
        content="must not overwrite protected revision",
        expected_revision_hash=initial_revision,
    )
    current = await manager.get(bucket_id)
    assert current["content"] == "stable body"
    assert current["metadata"]["protected"] is True


@pytest.mark.asyncio
async def test_post_write_failure_signal_does_not_create_duplicate(monkeypatch):
    class _PostWriteFalseManager(_Manager):
        async def update(
            self,
            bucket_id,
            actor="system",
            expected_content_hash="",
            expected_revision_hash="",
            **kwargs,
        ):
            await super().update(
                bucket_id,
                actor=actor,
                expected_content_hash=expected_content_hash,
                expected_revision_hash=expected_revision_hash,
                **kwargs,
            )
            return False

    target = _target()
    manager = _PostWriteFalseManager(target=target)
    result = await _run_write(
        monkeypatch,
        f"merge:{target['id']}",
        manager,
        _Dehydrator(),
        _Embedding(),
    )

    assert result == (target["id"], "旧部署", True)
    assert manager.created == []
    assert target["content"] == "merged body"


@pytest.mark.asyncio
async def test_registered_fact_conflict_keeps_z_review_boundary(monkeypatch):
    target = _target()
    target["content"] = "period_start: 2026-05-03\nflow: light"
    target["metadata"]["domain"] = ["health"]
    manager = _Manager(target=target)

    async def _decision(_content, _world, _domain):
        return f"supersede:{target['id']}"

    monkeypatch.setattr(server, "_recall_before_write_decision", _decision)
    monkeypatch.setattr(server, "bucket_mgr", manager)
    monkeypatch.setattr(server, "dehydrator", _Dehydrator())
    monkeypatch.setattr(server, "embedding_engine", _Embedding())
    monkeypatch.setattr(server, "_synchronize_bucket_entities", _no_entity_sync)
    monkeypatch.setattr(server, "_find_merge_candidates", _forbid_legacy_merge_search)

    result = await server._merge_or_create(
        content="period_start: 2026-05-04\nflow: light",
        tags=["period"],
        importance=6,
        domain=["health"],
        valence=0.5,
        arousal=0.3,
        name="period",
        recall_before_write=True,
    )

    assert result == ("new-bucket", "period", False)
    assert manager.updated == []
    assert len(manager.created) == 1


class _ModelCreate:
    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.raw)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _model_dehydrator(test_config, raw):
    dehydrator = Dehydrator(test_config)
    create = _ModelCreate(raw)
    dehydrator.api_available = True
    dehydrator.model = "small-test-model"
    dehydrator.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )
    return dehydrator, create


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"decision":"new"}', "new"),
        (
            '{"decision":"merge","bucket_id":"abc123abc123"}',
            "merge:abc123abc123",
        ),
        (
            '{"decision":"supersede","bucket_id":"abc123abc123"}',
            "supersede:abc123abc123",
        ),
    ],
)
async def test_small_model_accepts_only_the_three_contract_outcomes(
    test_config,
    raw,
    expected,
):
    dehydrator, create = _model_dehydrator(test_config, raw)

    assert await dehydrator.arbitrate_recall_before_write(
        "朝灯完成了新版部署。",
        '<candidate bucket_id="abc123abc123">\n旧部署\n</candidate>',
        ["abc123abc123"],
    ) == expected
    assert create.calls[0]["model"] == "small-test-model"
    assert create.calls[0]["temperature"] == 0.0
    assert create.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"decision":"new"}\n```',
        '{"decision":"new","bucket_id":"abc123abc123"}',
        '{"decision":"merge","bucket_id":"off-list"}',
        '{"decision":"merge:abc123abc123"}',
        '{"decision":"new","decision":"supersede"}',
    ],
)
async def test_small_model_rejects_malformed_or_off_list_decisions(
    test_config,
    raw,
):
    dehydrator, _create = _model_dehydrator(test_config, raw)

    with pytest.raises(RuntimeError):
        await dehydrator.arbitrate_recall_before_write(
            "朝灯完成了新版部署。",
            '<candidate bucket_id="abc123abc123">\n旧部署\n</candidate>',
            ["abc123abc123"],
        )
