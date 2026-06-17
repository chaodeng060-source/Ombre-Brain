import pytest


SECRET_TEXT = (
    "api_key=sk-leaksecret123456789 然后她说好想我\n"
    "postgresql://u:p@10.0.0.9:5432/ombre 她亲了我一下\n"
    "Authorization: Bearer abc123XYZ._-token 但这里是她的真实情绪"
)


def _assert_secret_redacted(text: str):
    assert "sk-leaksecret123456789" not in text
    assert "10.0.0.9" not in text
    assert "abc123XYZ._-token" not in text


def _assert_emotion_kept(text: str):
    assert "好想我" in text
    assert "她亲了我一下" in text
    assert "真实情绪" in text


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeChatResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeChatCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeChatResponse(self.content)


class FakeChatClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeChatCompletions(content)


class FakeEmbeddingData:
    embedding = [0.1, 0.2, 0.3]


class FakeEmbeddingResponse:
    data = [FakeEmbeddingData()]


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeEmbeddingResponse()


class FakeEmbeddingClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


def _last_user_message(client: FakeChatClient) -> str:
    messages = client.chat.completions.calls[-1]["messages"]
    return [m["content"] for m in messages if m["role"] == "user"][-1]


def _dehydrator(test_config, response: str):
    from dehydrator import Dehydrator

    dh = Dehydrator(test_config)
    dh.api_available = True
    dh.model = "fake-model"
    dh.client = FakeChatClient(response)
    return dh


@pytest.mark.asyncio
async def test_analyze_redacts_external_payload_keeps_emotion(test_config):
    dh = _dehydrator(
        test_config,
        '{"domain":["feel"],"valence":0.7,"arousal":0.4,'
        '"tags":["真实情绪"],"suggested_name":"测试"}',
    )

    await dh.analyze(SECRET_TEXT)

    payload = _last_user_message(dh.client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)


@pytest.mark.asyncio
async def test_digest_redacts_external_payload_keeps_emotion(test_config):
    dh = _dehydrator(
        test_config,
        '{"entries":[{"name":"亲密记忆","content":"她亲了我一下",'
        '"domain":["feel"],"valence":0.8,"arousal":0.4,'
        '"tags":["真实情绪"],"importance":5}]}',
    )

    await dh.digest(SECRET_TEXT)

    payload = _last_user_message(dh.client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)


@pytest.mark.asyncio
async def test_infer_relations_redacts_new_content_and_candidates_keeps_ids(test_config):
    dh = _dehydrator(test_config, "[]")
    candidates = [
        {
            "id": "bucket-1",
            "name": f"候选 {SECRET_TEXT}",
            "summary": f"摘要 {SECRET_TEXT}",
        }
    ]

    await dh.infer_relations(SECRET_TEXT, candidates)

    payload = _last_user_message(dh.client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)
    assert "bucket-1" in payload


@pytest.mark.asyncio
async def test_embedding_engine_redacts_external_input_keeps_emotion(test_config):
    from embedding_engine import EmbeddingEngine

    eng = EmbeddingEngine(test_config)
    eng.enabled = True
    eng.client = FakeEmbeddingClient()

    await eng._generate_embedding(SECRET_TEXT)

    payload = eng.client.embeddings.calls[-1]["input"]
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)


@pytest.mark.asyncio
async def test_import_engine_redacts_chunk_before_llm(test_config):
    from import_memory import ImportEngine

    dh = _dehydrator(test_config, "[]")
    engine = ImportEngine(test_config, bucket_mgr=object(), dehydrator=dh)

    await engine._extract_memories(SECRET_TEXT)

    payload = _last_user_message(dh.client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)


@pytest.mark.asyncio
async def test_query_expand_redacts_llm_payload_keeps_original_first():
    from query_expand import expand_query

    client = FakeChatClient('["真实情绪","亲密记忆"]')

    out = await expand_query(SECRET_TEXT, client, "fake-model", config={"min_query_len": 1})

    assert out[0] == SECRET_TEXT
    payload = _last_user_message(client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)


@pytest.mark.asyncio
async def test_server_semantic_select_redacts_query_and_snippets(monkeypatch):
    import server

    client = FakeChatClient('{"keep":[0]}')
    fake_dehydrator = type("D", (), {"client": client, "model": "fake-model"})()
    monkeypatch.setattr(server, "dehydrator", fake_dehydrator)
    buckets = [{"id": "b1", "metadata": {"name": SECRET_TEXT}, "content": SECRET_TEXT}]

    await server._ds_semantic_select(SECRET_TEXT, buckets, keep=set(), max_results=5)

    payload = _last_user_message(client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)


@pytest.mark.asyncio
async def test_episode_extract_redacts_replay_before_llm():
    from datetime import datetime

    from episode_engine import EpisodeEngine

    class BucketMgr:
        def __init__(self):
            self.created = []

        async def create(self, **kwargs):
            self.created.append(kwargs)
            return "episode-1"

        async def update(self, *args, **kwargs):
            return None

    dh = type(
        "D",
        (),
        {
            "api_available": True,
            "model": "fake-model",
            "client": FakeChatClient('{"name":"情节","summary":"她亲了我一下"}'),
        },
    )()
    engine = EpisodeEngine({"narrative": {}}, BucketMgr(), embedding_engine=None, dehydrator=dh)
    cluster = [
        {
            "id": "b1",
            "_dt": datetime(2026, 6, 17),
            "content": SECRET_TEXT,
            "metadata": {"name": SECRET_TEXT, "created": "2026-06-17", "importance": 5},
        }
    ]

    await engine.extract_episode(cluster)

    payload = _last_user_message(dh.client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)


@pytest.mark.asyncio
async def test_saga_route_and_create_redact_payloads(test_config):
    from saga_engine import SagaEngine

    class BucketMgr:
        async def create(self, **kwargs):
            return "saga-1"

    route_dh = type(
        "D",
        (),
        {"model": "fake-model", "client": FakeChatClient("NEW")},
    )()
    engine = SagaEngine({"narrative": {}}, BucketMgr(), dehydrator=route_dh)
    episode = {"id": "ep1", "metadata": {"name": SECRET_TEXT}, "content": SECRET_TEXT}
    sagas = [{"id": "saga-raw-id", "metadata": {"name": SECRET_TEXT}, "content": SECRET_TEXT}]

    await engine._route_episode(episode, sagas)
    payload = _last_user_message(route_dh.client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)
    assert "saga-raw-id" in payload

    create_dh = type(
        "D",
        (),
        {
            "model": "fake-model",
            "client": FakeChatClient('{"title":"真实情绪","description":"她亲了我一下"}'),
        },
    )()
    engine = SagaEngine({"narrative": {}}, BucketMgr(), dehydrator=create_dh)

    await engine._create_saga(episode)
    payload = _last_user_message(create_dh.client)
    _assert_secret_redacted(payload)
    _assert_emotion_kept(payload)
