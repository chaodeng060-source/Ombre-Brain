from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


HOOK_PATH = (
    Path(__file__).resolve().parent.parent / ".claude" / "hooks" / "lmc5_hook.py"
)
PROJECT_ROOT = HOOK_PATH.parent.parent.parent
LAUNCHER_PATH = HOOK_PATH.parent / "run_lmc5_hook.mjs"
SPEC = importlib.util.spec_from_file_location("ombre_lmc5_hook", HOOK_PATH)
assert SPEC is not None and SPEC.loader is not None
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)


def test_load_transcript_preserves_exact_jsonl_record_text(tmp_path):
    transcript = tmp_path / "session.jsonl"
    first = '  {"uuid":"u-1","message":{"content":"朝灯"}}  '
    second = '{"type":"assistant","message":{"id":"m-2","content":"ok"}}'
    transcript.write_bytes(
        first.encode("utf-8") + b"\r\n\r\n" + second.encode("utf-8") + b"\n"
    )

    events = hook.load_transcript_events(transcript, "session-1")

    assert events == [
        {"source_event_id": "uuid:u-1", "payload": first},
        {"source_event_id": "message.id:m-2", "payload": second},
    ]


def test_duplicate_explicit_event_id_falls_back_to_exact_line_identity(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        '{"uuid":"same","n":1}\n{"uuid":"same","n":2}\n',
        encoding="utf-8",
    )

    first = hook.load_transcript_events(transcript, "session-1")
    second = hook.load_transcript_events(transcript, "session-1")

    assert first == second
    assert first[0]["source_event_id"] == "uuid:same"
    assert first[1]["source_event_id"].startswith("line-sha256:")
    assert first[0]["source_event_id"] != first[1]["source_event_id"]


@pytest.mark.parametrize(
    "contents, expected_code",
    [
        (b'{"ok":true}\\nnot-json\\n', "transcript_invalid_jsonl"),
        (b'["not-an-object"]\n', "transcript_event_not_object"),
        (b"\xff\n", "transcript_invalid_jsonl"),
    ],
)
def test_load_transcript_rejects_invalid_complete_batch(
    tmp_path, contents, expected_code
):
    transcript = tmp_path / "session.jsonl"
    transcript.write_bytes(contents)

    with pytest.raises(hook.HookInputError, match=expected_code):
        hook.load_transcript_events(transcript, "session-1")


def test_load_transcript_rejects_count_overflow_without_prefix_ack(
    tmp_path, monkeypatch
):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    monkeypatch.setattr(hook, "MAX_TRANSCRIPT_EVENTS", 1)

    with pytest.raises(hook.HookInputError, match="transcript_too_many_events"):
        hook.load_transcript_events(transcript, "session-1")


def test_session_end_posts_exact_batch_and_requires_explicit_ack(tmp_path):
    transcript = tmp_path / "session.jsonl"
    line = '{"uuid":"event-1","content":"private body"}'
    transcript.write_text(line + "\n", encoding="utf-8")
    seen = {}

    def post_json(path, payload):
        seen["path"] = path
        seen["payload"] = payload
        return {
            "ok": True,
            "session_id": "session-1",
            "acknowledged": 1,
            "inserted": 1,
        }

    hook.run_session_end(
        {"session_id": "session-1", "transcript_path": str(transcript)},
        post_json=post_json,
        outbox_dir=tmp_path / "outbox",
    )

    assert seen == {
        "path": "/lmc5/raw-events",
        "payload": {
            "schema_version": 1,
            "session_id": "session-1",
            "events": [
                {
                    "source_event_id": "uuid:event-1",
                    "payload": line,
                }
            ],
        },
    }


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"ok": True, "session_id": "other", "acknowledged": 1, "inserted": 1},
        {
            "ok": True,
            "session_id": "session-1",
            "acknowledged": 0,
            "inserted": 0,
        },
        {
            "ok": True,
            "session_id": "session-1",
            "acknowledged": 1,
            "inserted": 2,
        },
        {
            "ok": True,
            "session_id": "session-1",
            "acknowledged": True,
            "inserted": 1,
        },
    ],
)
def test_session_end_rejects_false_or_mismatched_ack(tmp_path, response):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text('{"id":"1"}\n', encoding="utf-8")

    with pytest.raises(hook.HookTransportError, match="raw_ack_invalid"):
        hook.run_session_end(
            {"sessionId": "session-1", "sessionLog": str(transcript)},
            post_json=lambda _path, _payload: response,
            outbox_dir=tmp_path / "outbox",
        )
    assert len(list((tmp_path / "outbox").glob("batch-*.json"))) == 1


def test_session_end_retry_accepts_zero_inserted_with_full_ack(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text('{"id":"1"}\n', encoding="utf-8")

    hook.run_session_end(
        {"session_id": "session-1", "transcript_path": str(transcript)},
        post_json=lambda _path, _payload: {
            "ok": True,
            "session_id": "session-1",
            "acknowledged": 1,
            "inserted": 0,
        },
        outbox_dir=tmp_path / "outbox",
    )
    assert list((tmp_path / "outbox").glob("batch-*.json")) == []


def test_session_end_spools_before_network_and_retries_durably(tmp_path):
    transcript = tmp_path / "session.jsonl"
    secret = "PRIVATE_TRANSCRIPT_BODY"
    transcript.write_text(
        json.dumps({"uuid": "event-1", "content": secret}) + "\n",
        encoding="utf-8",
    )
    outbox = tmp_path / "outbox"

    def unavailable(_path, _payload):
        pending = list(outbox.glob("batch-*.json"))
        assert len(pending) == 1
        raise hook.HookTransportError("hook_request_failed")

    with pytest.raises(hook.HookTransportError, match="hook_request_failed"):
        hook.run_session_end(
            {"session_id": "session-1", "transcript_path": str(transcript)},
            post_json=unavailable,
            outbox_dir=outbox,
        )

    pending = list(outbox.glob("batch-*.json"))
    assert len(pending) == 1
    if hook.os.name != "nt":
        assert pending[0].stat().st_mode & 0o777 == 0o600
        assert outbox.stat().st_mode & 0o777 == 0o700

    seen = []

    def available(path, payload):
        seen.append((path, payload["session_id"], len(payload["events"])))
        return {
            "ok": True,
            "session_id": "session-1",
            "acknowledged": 1,
            "inserted": 1,
        }

    assert hook.flush_outbox(post_json=available, outbox_dir=outbox) == 1
    assert seen == [("/lmc5/raw-events", "session-1", 1)]
    assert list(outbox.glob("batch-*.json")) == []


def test_corrupt_outbox_fails_closed_and_is_preserved(tmp_path):
    outbox = tmp_path / "outbox"
    outbox.mkdir(mode=0o700)
    path = outbox / f"batch-{'0' * 64}.json"
    path.write_text("{broken", encoding="utf-8")
    called = False

    def post_json(_path, _payload):
        nonlocal called
        called = True
        raise AssertionError("corrupt private spool must never be sent")

    with pytest.raises(hook.HookInputError, match="outbox_payload_corrupt"):
        hook.flush_outbox(post_json=post_json, outbox_dir=outbox)

    assert called is False
    assert path.exists()


def test_user_prompt_submit_posts_nontrivial_prompt_and_prints_context():
    seen = {}
    output = io.StringIO()

    def post_json(path, payload):
        seen["path"] = path
        seen["payload"] = payload
        return {"ok": True, "context": "[memory]\nremember this"}

    hook.run_user_prompt_submit(
        {"sessionId": "session-1", "prompt": "记得上次部署问题吗？"},
        post_json=post_json,
        output=output,
    )

    assert seen == {
        "path": "/lmc5/recall-hook",
        "payload": {
            "schema_version": 1,
            "prompt": "记得上次部署问题吗？",
            "session_id": "session-1",
        },
    }
    assert output.getvalue() == "[memory]\nremember this"


@pytest.mark.parametrize("prompt", ["", "嗯", "好", "继续", "OK", "？"])
def test_user_prompt_submit_skips_trivial_messages(prompt):
    called = False

    def post_json(_path, _payload):
        nonlocal called
        called = True
        raise AssertionError("trivial prompt must not reach recall")

    hook.run_user_prompt_submit({"prompt": prompt}, post_json=post_json)

    assert called is False


@pytest.mark.parametrize(
    "prompt",
    [
        "抱抱我",
        "今天好开心呀",
        "亲亲宝宝",
        "我正在吃饭",
    ],
)
def test_user_prompt_submit_skips_non_memory_companion_chatter(prompt):
    called = False

    def post_json(_path, _payload):
        nonlocal called
        called = True
        raise AssertionError("ordinary companion chat must not force recall")

    hook.run_user_prompt_submit({"prompt": prompt}, post_json=post_json)

    assert called is False


@pytest.mark.parametrize(
    "prompt",
    [
        "你记得我们上次搬家吗",
        "Ollama 当时跑在哪台机器",
        "我们 2026-07-24 做了什么",
        "那个仓库后来怎么样了？",
        "哪天",
        "在哪",
        "谁？",
    ],
)
def test_user_prompt_submit_recalls_on_memory_time_entity_or_question(prompt):
    called = []

    def post_json(path, payload):
        called.append((path, payload["prompt"]))
        return {"ok": True, "context": ""}

    hook.run_user_prompt_submit({"prompt": prompt}, post_json=post_json)

    assert called == [("/lmc5/recall-hook", prompt)]


def test_user_prompt_submit_does_not_stringify_whole_hook_event():
    called = False

    def post_json(_path, _payload):
        nonlocal called
        called = True
        raise AssertionError("event metadata must not become a recall query")

    hook.run_user_prompt_submit(
        {"cwd": "/private/path", "hook_token": "do-not-send"},
        post_json=post_json,
    )

    assert called is False


@pytest.mark.parametrize(
    "command",
    ["retry-outbox", "session-end", "user-prompt-submit"],
)
def test_main_honors_global_hook_skip(monkeypatch, command):
    monkeypatch.setenv("OMBRE_HOOK_SKIP", "1")
    monkeypatch.setattr(
        hook,
        "_read_hook_event",
        lambda _stream: (_ for _ in ()).throw(
            AssertionError("disabled hook must not read private input")
        ),
    )

    assert hook.main([command]) == 0


def test_main_session_end_fails_closed_without_leaking_body(
    tmp_path, monkeypatch, capsys
):
    transcript = tmp_path / "session.jsonl"
    secret = "VERY_PRIVATE_TRANSCRIPT_BODY"
    transcript.write_text(json.dumps({"content": secret}) + "\nBROKEN\n")
    event = {
        "session_id": "session-1",
        "transcript_path": str(transcript),
    }
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(event)))

    code = hook.main(["session-end"])
    captured = capsys.readouterr()

    assert code != 0
    assert captured.out == ""
    assert "transcript_invalid_jsonl" in captured.err
    assert secret not in captured.err
    assert str(transcript) not in captured.err


def test_main_user_prompt_submit_fails_open_without_leaking_prompt(
    monkeypatch, capsys
):
    secret = "VERY_PRIVATE_PROMPT"
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        io.StringIO(json.dumps({"prompt": secret})),
    )
    monkeypatch.setattr(
        hook,
        "_post_json",
        lambda _path, _payload: (_ for _ in ()).throw(
            hook.HookTransportError("hook_request_failed")
        ),
    )

    code = hook.main(["user-prompt-submit"])
    captured = capsys.readouterr()

    assert code == 0
    assert captured.out == ""
    assert "hook_request_failed" in captured.err
    assert secret not in captured.err


def test_post_json_adds_token_header_without_logging_it(monkeypatch):
    token = "token-that-must-not-be-printed"
    monkeypatch.setenv("OMBRE_HOOK_TOKEN", token)
    seen = {}

    class Response:
        status = 200

        def getcode(self):
            return self.status

        def read(self, _limit):
            return b'{"ok":true,"context":""}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def urlopen(request, timeout):
        seen["token"] = request.get_header("X-ombre-hook-token")
        seen["timeout"] = timeout
        return Response()

    response = hook._post_json(
        "/lmc5/recall-hook",
        {"schema_version": 1, "prompt": "remember"},
        urlopen=urlopen,
    )

    assert response == {"ok": True, "context": ""}
    assert seen["token"] == token
    assert seen["timeout"] == hook.DEFAULT_TIMEOUT_SECONDS


def test_post_json_rejects_non_loopback_plain_http(monkeypatch):
    monkeypatch.setenv("OMBRE_HOOK_TOKEN", "secret")
    monkeypatch.setenv("OMBRE_HOOK_URL", "http://memory.example.test")

    with pytest.raises(hook.HookInputError, match="hook_plain_http_forbidden"):
        hook._post_json(
            "/lmc5/recall-hook",
            {"schema_version": 1, "prompt": "remember"},
            urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("unsafe HTTP request must not be attempted")
            ),
        )


def test_redirect_handler_refuses_forwarding_private_headers():
    handler = hook._RejectRedirects()

    with pytest.raises(hook.HookTransportError, match="hook_redirect_refused"):
        handler.redirect_request(
            object(),
            None,
            302,
            "Found",
            {},
            "https://other.example.test/",
        )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is unavailable")
@pytest.mark.parametrize(
    "command",
    ["retry-outbox", "session-breath", "user-prompt-submit"],
)
def test_node_launcher_executes_real_hook_command(command):
    environment = dict(os.environ)
    environment["OMBRE_HOOK_SKIP"] = "1"
    result = subprocess.run(
        ["node", str(LAUNCHER_PATH), command],
        input=b"{}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""


def test_claude_settings_use_cross_platform_exec_form():
    settings = json.loads(
        (PROJECT_ROOT / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    handlers = [
        handler
        for groups in settings["hooks"].values()
        for group in groups
        for handler in group["hooks"]
    ]

    assert handlers
    assert all(handler.get("shell") is None for handler in handlers)
    assert all(handler["command"] == "node" for handler in handlers)
    assert all(
        handler["args"][0]
        == "${CLAUDE_PROJECT_DIR}/.claude/hooks/run_lmc5_hook.mjs"
        for handler in handlers
    )
