import json
from types import SimpleNamespace

import pytest

from import_memory import ImportEngine


MEMORY_ITEM = {
    "name": "长期偏好",
    "content": "朝灯明确说过自己更喜欢有来源、可回放且不会重复写入的长期记忆。",
    "domain": ["AI"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["导入", "完整性"],
    "importance": 7,
    "preserve_raw": False,
    "is_pattern": False,
}
SOURCE = "User: 我更喜欢有来源、可回放且不会重复写入的长期记忆。"


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **_kwargs):
        self.calls += 1
        if not self.outcomes:
            raise AssertionError("provider was called again after extraction was durable")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome == "NO_CHOICES":
            return SimpleNamespace(choices=[])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=outcome)
                )
            ]
        )


class _Dehydrator:
    api_available = True
    model = "test"

    def __init__(self, outcomes):
        self.completions = _Completions(outcomes)
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions)
        )
        self.merge_calls = 0

    async def merge(self, old, new):
        self.merge_calls += 1
        return f"{old}\n{new}"


class _BucketManager:
    def __init__(self, existing=None):
        self.buckets = {}
        self.create_calls = 0
        self.update_calls = 0
        self.fail_create_before = False
        self.fail_create_after = False
        self.fail_update_after = False
        self.search_result = []
        if existing:
            self.buckets[existing["id"]] = existing
            self.search_result = [existing]

    async def list_all(self, include_archive=False):
        del include_archive
        return list(self.buckets.values())

    async def search(self, *_args, **_kwargs):
        return self.search_result

    async def create(self, content, **kwargs):
        self.create_calls += 1
        if self.fail_create_before:
            raise OSError("bucket storage unavailable")
        bucket_id = f"bucket-{self.create_calls}"
        self.buckets[bucket_id] = {
            "id": bucket_id,
            "content": content,
            "metadata": {
                "tags": list(kwargs.get("tags", [])),
                "domain": list(kwargs.get("domain", ["未分类"])),
                "importance": kwargs.get("importance", 5),
                "valence": kwargs.get("valence", 0.5),
                "arousal": kwargs.get("arousal", 0.3),
            },
        }
        if self.fail_create_after:
            raise OSError("worker died after durable create")
        return bucket_id

    async def update(self, bucket_id, **kwargs):
        self.update_calls += 1
        bucket = self.buckets[bucket_id]
        if "content" in kwargs:
            bucket["content"] = kwargs["content"]
        bucket["metadata"].update(
            {key: value for key, value in kwargs.items() if key != "content"}
        )
        if self.fail_update_after:
            raise OSError("worker died after durable update")
        return True


class _EmbeddingEngine:
    enabled = True

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def generate_and_store(self, bucket_id, content):
        self.calls.append((bucket_id, content))
        return self.outcomes.pop(0)


def _payload(item=MEMORY_ITEM):
    return json.dumps([item], ensure_ascii=False)


def _engine(test_config, manager, dehydrator, embedding=None, content_sync=None):
    engine = ImportEngine(
        test_config,
        bucket_mgr=manager,
        dehydrator=dehydrator,
        embedding_engine=embedding,
    )
    engine.content_sync = content_sync
    return engine


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider_outcome",
    [
        TimeoutError("provider timeout"),
        "NO_CHOICES",
        "",
        "{not-json",
        '{"content":"not-an-array"}',
    ],
)
async def test_provider_failures_never_ack_chunk_completed(
    test_config,
    provider_outcome,
):
    manager = _BucketManager()
    dehydrator = _Dehydrator([provider_outcome])

    result = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
    )

    assert result["status"] == "error"
    assert result["processed"] == 0
    assert result["chunks"][0]["status"] == "error"
    assert result["chunks"][0]["extraction_status"] == "pending"
    assert result["chunks"][0]["zero_candidates"] is False
    assert result["errors"]
    assert manager.create_calls == 0


@pytest.mark.anyio
async def test_valid_empty_array_is_a_completed_zero_candidate_chunk(test_config):
    manager = _BucketManager()
    dehydrator = _Dehydrator(["[]"])

    result = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
    )

    assert result["status"] == "completed"
    assert result["processed"] == 1
    assert result["chunks"][0]["status"] == "complete"
    assert result["chunks"][0]["extraction_status"] == "complete"
    assert result["chunks"][0]["zero_candidates"] is True
    assert result["chunks"][0]["outputs"] == []
    assert manager.create_calls == 0


@pytest.mark.anyio
async def test_resume_of_completed_batch_is_a_noop(test_config):
    manager = _BucketManager()
    dehydrator = _Dehydrator(["[]"])

    first = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
    )
    resumed = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
        resume=True,
    )

    assert first["status"] == "completed"
    assert resumed["status"] == "completed"
    assert dehydrator.completions.calls == 1


@pytest.mark.anyio
async def test_storage_failure_keeps_output_and_retry_skips_provider(
    test_config,
):
    manager = _BucketManager()
    manager.fail_create_before = True
    dehydrator = _Dehydrator([_payload()])

    first = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
    )

    assert first["status"] == "error"
    assert first["processed"] == 0
    chunk = first["chunks"][0]
    assert chunk["extraction_status"] == "complete"
    assert chunk["status"] == "error"
    assert chunk["outputs"][0]["status"] == "error"
    assert "item" not in chunk["outputs"][0]
    stable_output_id = chunk["outputs"][0]["output_id"]
    assert dehydrator.completions.calls == 1

    manager.fail_create_before = False
    resumed = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
        resume=True,
    )

    assert resumed["status"] == "completed"
    assert resumed["processed"] == 1
    assert resumed["chunks"][0]["outputs"][0]["output_id"] == stable_output_id
    assert dehydrator.completions.calls == 1
    assert manager.create_calls == 2
    assert len(manager.buckets) == 1


@pytest.mark.anyio
async def test_crash_after_create_is_reconciled_without_duplicate_bucket(
    test_config,
):
    manager = _BucketManager()
    manager.fail_create_after = True
    dehydrator = _Dehydrator([_payload()])

    first = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
    )
    assert first["status"] == "error"
    assert len(manager.buckets) == 1

    manager.fail_create_after = False
    resumed = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
        resume=True,
    )

    assert resumed["status"] == "completed"
    assert resumed["memories_created"] == 1
    assert manager.create_calls == 1
    assert len(manager.buckets) == 1
    assert dehydrator.completions.calls == 1
    state_text = (
        test_config["buckets_dir"] + "/import_state.json"
    )
    with open(state_text, "r", encoding="utf-8") as handle:
        durable_state = handle.read()
    assert MEMORY_ITEM["content"] not in durable_state


@pytest.mark.anyio
async def test_embedding_failure_is_error_then_reuses_created_bucket(
    test_config,
):
    manager = _BucketManager()
    dehydrator = _Dehydrator([_payload()])
    embeddings = _EmbeddingEngine([False, True])

    first = await _engine(
        test_config,
        manager,
        dehydrator,
        embeddings,
    ).start(SOURCE, filename="history.md")
    assert first["status"] == "error"
    assert first["processed"] == 0
    assert len(manager.buckets) == 1

    resumed = await _engine(
        test_config,
        manager,
        dehydrator,
        embeddings,
    ).start(SOURCE, filename="history.md", resume=True)

    assert resumed["status"] == "completed"
    assert manager.create_calls == 1
    assert len(embeddings.calls) == 2


@pytest.mark.anyio
async def test_crash_after_merge_marker_does_not_merge_twice(test_config):
    existing = {
        "id": "existing",
        "content": "旧内容",
        "score": 100,
        "metadata": {
            "tags": [],
            "domain": ["AI"],
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
        },
    }
    manager = _BucketManager(existing=existing)
    manager.fail_update_after = True
    dehydrator = _Dehydrator([_payload()])

    first = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
    )
    assert first["status"] == "error"
    assert manager.update_calls == 1
    merged_once = manager.buckets["existing"]["content"]

    manager.fail_update_after = False
    resumed = await _engine(test_config, manager, dehydrator).start(
        SOURCE,
        filename="history.md",
        resume=True,
    )

    assert resumed["status"] == "completed"
    assert resumed["memories_merged"] == 1
    assert manager.update_calls == 1
    assert manager.buckets["existing"]["content"] == merged_once
    assert dehydrator.merge_calls == 1
    assert dehydrator.completions.calls == 1


@pytest.mark.anyio
async def test_successful_import_merge_notifies_content_sidecars(test_config):
    existing = {
        "id": "existing",
        "content": "旧内容",
        "score": 100,
        "metadata": {
            "tags": [],
            "domain": ["AI"],
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
        },
    }
    manager = _BucketManager(existing=existing)
    dehydrator = _Dehydrator([_payload()])
    synced = []

    async def content_sync(bucket_id, content):
        synced.append((bucket_id, content))

    result = await _engine(
        test_config,
        manager,
        dehydrator,
        content_sync=content_sync,
    ).start(SOURCE, filename="history.md")

    assert result["status"] == "completed"
    assert synced == [("existing", manager.buckets["existing"]["content"])]


@pytest.mark.anyio
async def test_corrupt_existing_ledger_fails_closed_instead_of_reextracting(
    test_config,
):
    state_path = test_config["buckets_dir"] + "/import_state.json"
    with open(state_path, "w", encoding="utf-8") as handle:
        handle.write("{broken")

    manager = _BucketManager()
    dehydrator = _Dehydrator([_payload()])

    with pytest.raises(
        RuntimeError,
        match="ledger exists but cannot be read safely",
    ):
        await _engine(test_config, manager, dehydrator).start(
            SOURCE,
            filename="history.md",
            resume=True,
        )

    assert dehydrator.completions.calls == 0
    assert manager.create_calls == 0
