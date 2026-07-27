import asyncio
import hashlib
from datetime import datetime
from types import SimpleNamespace

import pytest

from bucket_manager import BucketManager
from episode_engine import EpisodeEngine
from import_memory import ImportEngine, build_import_x_provenance
from redact import redact_embedding_input
from saga_engine import SagaEngine
from x_provenance import normalize_x_provenance


def test_strict_normalizer_accepts_canonical_episode_provenance():
    normalized = normalize_x_provenance({
        "source_kind": "episode",
        "source_buckets": ["event-a", "event-b"],
        "span_start": "2026-07-01",
        "span_end": "2026-07-02T03:04:05",
    })

    assert normalized == {
        "x_schema_version": 1,
        "source_kind": "episode",
        "source_buckets": ["event-a", "event-b"],
        "span_start": "2026-07-01T00:00:00",
        "span_end": "2026-07-02T03:04:05",
    }


@pytest.mark.parametrize(
    "value",
    [
        {
            "source_kind": "episode",
            "source_buckets": ["a"],
            "span_start": "2026-07-02",
            "span_end": "2026-07-01",
        },
        {
            "source_kind": "episode",
            "source_buckets": ["a", "a"],
            "span_start": "2026-07-01",
            "span_end": "2026-07-02",
        },
        {
            "source_kind": "import",
            "source_digest": "a" * 64,
            "source_chunk_ordinal": True,
        },
        {
            "source_kind": "import",
            "source_digest": "a" * 64,
            "source_chunk_ordinal": 0,
            "span_start": "2026-07-01",
        },
        {
            "source_kind": "saga",
            "episode_buckets": ["ep-1"],
            "source_digest": "a" * 64,
        },
        {
            "source_kind": "external",
            "source_digest": "not-a-digest",
        },
        {
            "source_kind": "episode",
            "source_buckets": ["a"],
            "span_start": "2026-07-01",
            "span_end": "2026-07-02",
            "unknown": "no",
        },
    ],
)
def test_strict_normalizer_rejects_ambiguous_or_fabricated_provenance(value):
    with pytest.raises(ValueError):
        normalize_x_provenance(value)


@pytest.mark.asyncio
async def test_bucket_create_persists_x_provenance_in_first_write(test_config):
    manager = BucketManager(test_config)
    bucket_id = await manager.create(
        content="derived",
        bucket_type="episode",
        x_provenance={
            "source_kind": "episode",
            "source_buckets": ["event-a", "event-b"],
            "span_start": "2026-07-01T10:00:00",
            "span_end": "2026-07-01T11:00:00",
        },
    )

    meta = (await manager.get(bucket_id))["metadata"]
    assert meta["x_schema_version"] == 1
    assert meta["source_kind"] == "episode"
    assert meta["source_buckets"] == ["event-a", "event-b"]
    assert meta["span_start"] == "2026-07-01T10:00:00"
    assert meta["span_end"] == "2026-07-01T11:00:00"


@pytest.mark.asyncio
async def test_update_rejects_provenance_rewrite_before_last_active_changes(
    test_config,
):
    manager = BucketManager(test_config)
    saga_id = await manager.create(
        content="story",
        bucket_type="saga",
        x_provenance={
            "source_kind": "saga",
            "episode_buckets": ["ep-1"],
        },
    )
    before = (await manager.get(saga_id))["metadata"]

    assert not await manager.update(saga_id, source_kind="external")
    after_rewrite = (await manager.get(saga_id))["metadata"]
    assert after_rewrite["source_kind"] == "saga"
    assert after_rewrite["last_active"] == before["last_active"]

    assert await manager.update(
        saga_id,
        episode_buckets=["ep-1", "ep-2"],
    )
    appended = (await manager.get(saga_id))["metadata"]
    assert appended["episode_buckets"] == ["ep-1", "ep-2"]

    last_active = appended["last_active"]
    assert not await manager.update(
        saga_id,
        episode_buckets=["ep-2", "ep-1", "ep-3"],
    )
    assert not await manager.update(
        saga_id,
        episode_buckets=["ep-1", "ep-2", "ep-2"],
    )
    rejected = (await manager.get(saga_id))["metadata"]
    assert rejected["episode_buckets"] == ["ep-1", "ep-2"]
    assert rejected["last_active"] == last_active


class _RecordingBucketManager:
    def __init__(self):
        self.created = []
        self.updates = []
        self.buckets = {}

    async def create(self, content, bucket_type="dynamic", **kwargs):
        bucket_id = f"{bucket_type}-{len(self.created) + 1}"
        x_provenance = kwargs.get("x_provenance")
        metadata = {
            "id": bucket_id,
            "name": kwargs.get("name"),
            "type": bucket_type,
            "domain": kwargs.get("domain") or ["未分类"],
            "importance": kwargs.get("importance", 5),
            "valence": kwargs.get("valence", 0.5),
            "arousal": kwargs.get("arousal", 0.3),
        }
        if x_provenance is not None:
            metadata.update(normalize_x_provenance(x_provenance))
        self.created.append({
            "bucket_id": bucket_id,
            "content": content,
            "bucket_type": bucket_type,
            **kwargs,
        })
        self.buckets[bucket_id] = {
            "id": bucket_id,
            "content": content,
            "metadata": metadata,
        }
        return bucket_id

    async def update(self, bucket_id, **kwargs):
        self.updates.append((bucket_id, kwargs))
        self.buckets[bucket_id]["metadata"].update(kwargs)
        return True

    async def get(self, bucket_id):
        return self.buckets.get(bucket_id)


class _JsonCompletions:
    def __init__(self, payload):
        self.payload = payload

    async def create(self, **_kwargs):
        message = SimpleNamespace(content=self.payload)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _dehydrator(payload):
    completions = _JsonCompletions(payload)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return SimpleNamespace(
        api_available=True,
        client=client,
        model="test",
    )


def test_episode_creation_writes_evidence_atomically_without_followup_update():
    manager = _RecordingBucketManager()
    engine = EpisodeEngine(
        {"narrative": {}},
        manager,
        embedding_engine=None,
        dehydrator=_dehydrator('{"name":"一天","summary":"完整的一段经历"}'),
    )
    cluster = [
        {
            "id": "event-a",
            "_dt": datetime.fromisoformat("2026-07-01T10:00:00"),
            "content": "开始",
            "metadata": {
                "name": "开始",
                "event_at": "2026-07-01T10:00:00",
                "importance": 5,
                "valence": 0.5,
                "arousal": 0.4,
                "domain": ["工程"],
            },
        },
        {
            "id": "event-b",
            "_dt": datetime.fromisoformat("2026-07-01T11:00:00"),
            "content": "结束",
            "metadata": {
                "name": "结束",
                "event_at": "2026-07-01T11:00:00",
                "importance": 6,
                "valence": 0.6,
                "arousal": 0.5,
                "domain": ["工程"],
            },
        },
    ]

    episode_id = asyncio.run(engine.extract_episode(cluster))

    assert episode_id == "episode-1"
    assert manager.updates == []
    meta = manager.buckets[episode_id]["metadata"]
    assert meta["source_buckets"] == ["event-a", "event-b"]
    assert meta["span_start"] == "2026-07-01T10:00:00"
    assert meta["span_end"] == "2026-07-01T11:00:00"


def test_saga_creation_writes_first_episode_atomically_without_update():
    manager = _RecordingBucketManager()
    engine = SagaEngine(
        {"narrative": {}},
        manager,
        dehydrator=_dehydrator(
            '{"title":"长期主线","description":"这是一条长期主线"}'
        ),
    )
    episode = {
        "id": "episode-a",
        "content": "一段情节",
        "metadata": {
            "name": "情节",
            "importance": 6,
            "valence": 0.6,
            "arousal": 0.4,
            "domain": ["工程"],
        },
    }

    saga = asyncio.run(engine._create_saga(episode))

    assert saga["id"] == "saga-1"
    assert manager.updates == []
    assert saga["metadata"]["episode_buckets"] == ["episode-a"]


def test_import_provenance_hashes_exact_chunk_and_uses_only_real_time():
    chunk = {
        "content": "[用户] 真正送入提取器的原文",
        "timestamp_start": "2026-07-01T10:00",
        "timestamp_end": "2026-07-01T11:00",
    }
    provenance = build_import_x_provenance(chunk, 3)

    assert provenance["source_digest"] == hashlib.sha256(
        redact_embedding_input(chunk["content"])[:12000].encode("utf-8")
    ).hexdigest()
    assert provenance["source_chunk_ordinal"] == 3
    assert provenance["span_start"] == "2026-07-01T10:00:00"
    assert provenance["span_end"] == "2026-07-01T11:00:00"

    invalid_time = build_import_x_provenance({
        "content": "same",
        "timestamp_start": "someday",
        "timestamp_end": "",
    }, 0)
    assert "span_start" not in invalid_time
    assert "span_end" not in invalid_time

    secret_and_tail = "api_key=sk-leaksecret123456789 " + ("x" * 13000)
    bounded = build_import_x_provenance({
        "content": secret_and_tail,
        "timestamp_start": "",
        "timestamp_end": "",
    }, 1)
    exact_model_input = redact_embedding_input(secret_and_tail)[:12000]
    assert bounded["source_digest"] == hashlib.sha256(
        exact_model_input.encode("utf-8")
    ).hexdigest()
    assert bounded["source_digest"] != hashlib.sha256(
        secret_and_tail.encode("utf-8")
    ).hexdigest()


class _ImportBucketManager:
    def __init__(self, existing=None):
        self.existing = existing or []
        self.creates = []
        self.updates = []

    async def search(self, *_args, **_kwargs):
        return self.existing

    async def create(self, **kwargs):
        self.creates.append(kwargs)
        return "new-bucket"

    async def update(self, bucket_id, **kwargs):
        self.updates.append((bucket_id, kwargs))
        return True


class _ImportDehydrator:
    api_available = True

    async def merge(self, _old, new):
        return f"merged:{new}"


def _import_engine(test_config, manager):
    return ImportEngine(
        test_config,
        bucket_mgr=manager,
        dehydrator=_ImportDehydrator(),
    )


@pytest.mark.asyncio
async def test_import_new_create_gets_provenance_but_merge_does_not(test_config):
    provenance = {
        "source_kind": "import",
        "source_digest": "a" * 64,
        "source_chunk_ordinal": 4,
    }
    item = {
        "content": "new memory",
        "domain": ["工程"],
        "tags": [],
        "importance": 5,
        "valence": 0.5,
        "arousal": 0.3,
        "name": "new",
    }

    create_manager = _ImportBucketManager()
    create_engine = _import_engine(test_config, create_manager)
    assert not await create_engine._merge_or_create_item(
        item,
        x_provenance=provenance,
    )
    assert create_manager.creates[0]["x_provenance"] == provenance

    existing = {
        "id": "old",
        "content": "old memory",
        "score": 100,
        "metadata": {
            "tags": [],
            "domain": ["工程"],
            "importance": 5,
            "valence": 0.5,
            "arousal": 0.3,
        },
    }
    merge_manager = _ImportBucketManager(existing=[existing])
    merge_engine = _import_engine(test_config, merge_manager)
    assert await merge_engine._merge_or_create_item(
        item,
        x_provenance=provenance,
    )
    assert merge_manager.creates == []
    assert merge_manager.updates
    _, update_kwargs = merge_manager.updates[0]
    assert not (set(update_kwargs) & {
        "x_provenance",
        "x_schema_version",
        "source_kind",
        "source_digest",
        "source_chunk_ordinal",
        "source_buckets",
        "episode_buckets",
        "span_start",
        "span_end",
    })
