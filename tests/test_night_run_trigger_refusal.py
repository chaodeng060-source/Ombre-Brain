from __future__ import annotations

import pytest

import night_run_trigger
from night_run_trigger import NightTriggerHTTPError


@pytest.mark.parametrize("code", ["run.busy", "run.raced"])
def test_trigger_records_refusal_code_from_error_body(monkeypatch, code: str) -> None:
    class Response:
        status = 503

        @staticmethod
        def getheader(_name: str, default: str = "") -> str:
            return "application/json"

        @staticmethod
        def read(_limit: int) -> bytes:
            return ('{"ok":false,"code":"' + code + '"}').encode()

    class Connection:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def request(*_args, **_kwargs) -> None:
            return None

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setenv("OMBRE_API_TOKEN", "test-token")
    monkeypatch.setattr(night_run_trigger, "HTTPConnection", Connection)

    with pytest.raises(NightTriggerHTTPError) as raised:
        night_run_trigger.trigger()

    assert raised.value.status == 503
    assert raised.value.code == code


@pytest.mark.parametrize("code", ["run.busy", "run.raced"])
def test_overlapping_night_run_is_not_reported_as_cron_failure(
    monkeypatch, capsys, code: str
) -> None:
    monkeypatch.setattr(
        night_run_trigger,
        "trigger",
        lambda: (_ for _ in ()).throw(NightTriggerHTTPError(503, code)),
    )

    assert night_run_trigger.main() == 0
    assert f"already in flight ({code})" in capsys.readouterr().err


def test_other_http_refusal_remains_a_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        night_run_trigger,
        "trigger",
        lambda: (_ for _ in ()).throw(NightTriggerHTTPError(503, "night.unavailable")),
    )

    assert night_run_trigger.main() == 1
    assert "HTTP 503 code=night.unavailable" in capsys.readouterr().err
