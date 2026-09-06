import asyncio
import inspect

import pytest

import server


def _candidate(
    bucket_id: str,
    *,
    literal: float = 0.0,
    vector: float = 0.0,
    evidence=None,
    anchor: float = 0.3,
    weak_entity: bool = False,
) -> dict:
    row = {
        "id": bucket_id,
        "content": f"body:{bucket_id}",
        "metadata": {"name": bucket_id},
        "_literal_relevance_score": literal,
        "_original_vector_relevance_score": vector,
        "_anchor_adapted_relevance_score": anchor,
    }
    if evidence is not None:
        row["_entity_recall_evidence"] = evidence
    if weak_entity:
        row["weak_entity"] = True
    return row


def _evidence(df, corpus_bucket_count) -> list[dict]:
    return [
        {
            "term": "rare-name",
            "df": df,
            "corpus_bucket_count": corpus_bucket_count,
        }
    ]


def _build_tickets(candidates, **kwargs):
    builder = getattr(server, "_build_weak_entity_tickets", None)
    if builder is None:
        pytest.fail(
            "missing server._build_weak_entity_tickets(entity_candidates, *, "
            "literal_candidate_floor, excluded_ids=None, max_tickets=2)"
        )
    return builder(candidates, literal_candidate_floor=40.0, **kwargs)


def _require_ds_ticket_parameter() -> None:
    parameters = inspect.signature(server._ds_filter_candidates).parameters
    if "weak_entity_tickets" not in parameters:
        pytest.fail(
            "server._ds_filter_candidates needs keyword-only "
            "weak_entity_tickets: list[dict] | None = None"
        )


def _enable_ds(monkeypatch, *, fallback: bool = False) -> None:
    monkeypatch.setenv("OMBRE_DS_FILTER_ENABLED", "1")
    monkeypatch.setenv("OMBRE_DS_FILTER_MODES", "search")
    monkeypatch.setenv(
        "OMBRE_DS_FAILURE_FALLBACK_ENABLED",
        "1" if fallback else "0",
    )
    monkeypatch.setenv("OMBRE_DS_FAILURE_ANCHOR_FLOOR", "0.45")


@pytest.mark.parametrize(
    ("literal", "vector", "evidence", "eligible"),
    [
        # Accepted B delta, including the frozen 6cc5995aea84 score shape.
        (16.8675, 0.0, _evidence(1, 10), True),
        (39.9999, 0.0, _evidence(199, 1000), True),
        # Either score crossing its strict boundary rejects the ticket.
        (40.0, 0.0, _evidence(1, 10), False),
        (0.0, 0.000001, _evidence(1, 10), False),
        # DF is strict: exactly 0.2 is common, while any valid rare evidence wins.
        (0.0, 0.0, _evidence(20, 100), False),
        (
            0.0,
            0.0,
            [
                {"term": "common", "df": 20, "corpus_bucket_count": 100},
                {"term": "rare", "df": 1, "corpus_bucket_count": 100},
            ],
            True,
        ),
        # Missing or malformed evidence must fail closed.
        (0.0, 0.0, None, False),
        (0.0, 0.0, _evidence("1", 10), False),
        (0.0, 0.0, _evidence(1, 0), False),
        (0.0, 0.0, _evidence(-1, 10), False),
        (0.0, 0.0, _evidence(11, 10), False),
    ],
)
def test_build_weak_entity_tickets_enforces_score_and_df_boundaries(
    literal,
    vector,
    evidence,
    eligible,
):
    candidate = _candidate(
        "entity-only",
        literal=literal,
        vector=vector,
        evidence=evidence,
    )

    tickets = _build_tickets([candidate])

    assert bool(tickets) is eligible
    if eligible:
        assert [row["id"] for row in tickets] == ["entity-only"]
        assert tickets[0]["weak_entity"] is True


def test_build_weak_entity_tickets_caps_two_excludes_normal_ids_and_copies():
    candidates = [
        _candidate(
            bucket_id,
            literal=10.0,
            evidence=_evidence(1, 100),
        )
        for bucket_id in ("already-normal", "ticket-1", "ticket-2", "ticket-3")
    ]
    for row in candidates:
        row["entity_match"] = True

    tickets = _build_tickets(
        candidates,
        excluded_ids={"already-normal"},
        max_tickets=2,
    )

    assert [row["id"] for row in tickets] == ["ticket-1", "ticket-2"]
    assert all(row["weak_entity"] is True for row in tickets)
    assert all("entity_match" not in row for row in tickets)
    assert all(ticket is not source for ticket, source in zip(tickets, candidates[1:]))
    assert all("weak_entity" not in row for row in candidates)
    assert all(row["entity_match"] is True for row in candidates)


@pytest.mark.parametrize(
    ("bucket_id", "evidence"),
    [
        ("013da98a75e5", _evidence(20, 100)),
        ("019af40158f7", _evidence(21, 100)),
    ],
)
def test_recorded_high_frequency_entity_noise_never_gets_a_ticket(
    bucket_id,
    evidence,
):
    candidate = _candidate(
        bucket_id,
        literal=0.0,
        vector=0.0,
        evidence=evidence,
    )

    assert _build_tickets([candidate]) == []


@pytest.mark.asyncio
async def test_model_kept_ticket_fills_vacancy_without_displacement(
    monkeypatch,
):
    _require_ds_ticket_parameter()
    _enable_ds(monkeypatch)
    normal = [_candidate(f"normal-{index}") for index in range(1, 4)]
    tickets = [
        _candidate(f"ticket-{index}", weak_entity=True)
        for index in range(1, 3)
    ]
    observed = {}

    async def select(_query, rows, keep, max_results):
        observed["ids_in"] = [row["id"] for row in rows]
        observed["forced"] = set(keep)
        explicitly_kept = {"normal-1", "normal-3", "ticket-1", "ticket-2"}
        return [row for row in rows if row["id"] in explicitly_kept][
            :max_results
        ]

    monkeypatch.setattr(server, "_ds_semantic_select", select)

    selected = await server._ds_filter_candidates(
        "rare-name",
        normal,
        mode="search",
        max_results=3,
        force_keep_ids={"normal-3", "ticket-2"},
        allow_empty=True,
        weak_entity_tickets=tickets,
    )

    assert observed["ids_in"] == [
        "normal-1",
        "normal-2",
        "normal-3",
        "ticket-1",
        "ticket-2",
    ]
    assert observed["forced"] == {"normal-3"}
    assert [row["id"] for row in selected] == [
        "normal-1",
        "normal-3",
        "ticket-1",
    ]
    assert "displaced_id" not in selected[-1]
    assert len(selected) == 3


@pytest.mark.asyncio
async def test_model_kept_ticket_displaces_lowest_anchor_normal_when_pool_full(
    monkeypatch,
):
    _require_ds_ticket_parameter()
    _enable_ds(monkeypatch)
    normal = [
        _candidate("normal-1", anchor=0.91),
        _candidate("normal-2", anchor=0.12),
        _candidate("normal-3", anchor=0.64),
        _candidate("normal-4", anchor=0.20),
    ]
    tickets = [
        _candidate("ticket-1", weak_entity=True),
        _candidate("ticket-2", weak_entity=True),
    ]

    async def keep_everything(_query, rows, _keep, _max_results):
        return list(rows)

    monkeypatch.setattr(server, "_ds_semantic_select", keep_everything)

    selected = await server._ds_filter_candidates(
        "rare-name",
        normal,
        mode="search",
        max_results=4,
        allow_empty=True,
        weak_entity_tickets=tickets,
    )

    assert [row["id"] for row in selected] == [
        "normal-1",
        "normal-3",
        "ticket-1",
        "ticket-2",
    ]
    assert [row["displaced_id"] for row in selected[-2:]] == [
        "normal-2",
        "normal-4",
    ]
    assert len(selected) == 4


@pytest.mark.asyncio
async def test_full_pool_displacement_preserves_forced_and_unscored_normals(
    monkeypatch,
):
    _require_ds_ticket_parameter()
    _enable_ds(monkeypatch)
    unscored = {
        "id": "normal-unscored",
        "content": "body:normal-unscored",
        "metadata": {"name": "normal-unscored"},
    }
    normal = [
        _candidate("normal-forced", anchor=0.01),
        unscored,
        _candidate("normal-tie-earlier", anchor=0.20),
        _candidate("normal-tie-later", anchor=0.20),
    ]
    ticket = _candidate("ticket", weak_entity=True)

    async def keep_everything(_query, rows, _keep, _max_results):
        return list(rows)

    monkeypatch.setattr(server, "_ds_semantic_select", keep_everything)

    selected = await server._ds_filter_candidates(
        "rare-name",
        normal,
        mode="search",
        max_results=4,
        force_keep_ids={"normal-forced"},
        allow_empty=True,
        weak_entity_tickets=[ticket],
    )

    assert [row["id"] for row in selected] == [
        "normal-forced",
        "normal-unscored",
        "normal-tie-earlier",
        "ticket",
    ]
    assert selected[-1]["displaced_id"] == "normal-tie-later"
    assert "displaced_id" not in ticket
    assert "_literal_collision_guard" not in unscored


@pytest.mark.asyncio
async def test_ticket_rejected_by_model_is_not_restored_by_nonempty_fallback(
    monkeypatch,
):
    _require_ds_ticket_parameter()
    _enable_ds(monkeypatch)
    normal = [_candidate("normal-1"), _candidate("normal-2")]
    ticket = _candidate("ticket", weak_entity=True)

    async def reject_everything(*_args, **_kwargs):
        return []

    monkeypatch.setattr(server, "_ds_semantic_select", reject_everything)

    selected = await server._ds_filter_candidates(
        "rare-name",
        normal,
        mode="search",
        max_results=2,
        allow_empty=False,
        weak_entity_tickets=[ticket],
    )

    assert selected == normal
    assert all(not row.get("weak_entity") for row in selected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        server.DSFilterInvalidPayloadError("no_complete_json"),
        asyncio.TimeoutError(),
        RuntimeError("provider failed"),
    ],
    ids=["invalid", "timeout", "error"],
)
async def test_ds_failures_drop_tickets_before_conservative_fallback(
    monkeypatch,
    failure,
):
    _require_ds_ticket_parameter()
    _enable_ds(monkeypatch, fallback=True)
    normal = [
        _candidate("normal-top", anchor=0.3),
        _candidate("normal-tail", anchor=0.2),
    ]
    ticket = _candidate("ticket", anchor=0.99, weak_entity=True)
    fallback_calls = []
    original_fallback = server._ds_conservative_failure_candidates

    async def fail(*_args, **_kwargs):
        raise failure

    def capture_fallback(rows, **kwargs):
        fallback_calls.append(
            {
                "ids": [row["id"] for row in rows],
                "forced": set(kwargs["force_keep_ids"]),
            }
        )
        return original_fallback(rows, **kwargs)

    monkeypatch.setattr(server, "_ds_semantic_select", fail)
    monkeypatch.setattr(
        server,
        "_ds_conservative_failure_candidates",
        capture_fallback,
    )

    selected = await server._ds_filter_candidates(
        "rare-name",
        normal,
        mode="search",
        max_results=2,
        force_keep_ids={"ticket"},
        allow_empty=True,
        weak_entity_tickets=[ticket],
    )

    assert [row["id"] for row in selected] == ["normal-top"]
    assert all(not row.get("weak_entity") for row in selected)
    assert fallback_calls == [
        {"ids": ["normal-top", "normal-tail"], "forced": set()}
    ]


@pytest.mark.asyncio
async def test_outer_cancellation_prepares_fallback_from_normals_only(monkeypatch):
    _require_ds_ticket_parameter()
    _enable_ds(monkeypatch, fallback=True)
    normal = [_candidate("normal-1"), _candidate("normal-2")]
    ticket = _candidate("ticket", anchor=0.99, weak_entity=True)
    entered = asyncio.Event()
    fallback_calls = []
    original_fallback = server._ds_conservative_failure_candidates

    async def hang(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    def capture_fallback(rows, **kwargs):
        fallback_calls.append(
            {
                "ids": [row["id"] for row in rows],
                "forced": set(kwargs["force_keep_ids"]),
            }
        )
        return original_fallback(rows, **kwargs)

    monkeypatch.setattr(server, "_ds_semantic_select", hang)
    monkeypatch.setattr(
        server,
        "_ds_conservative_failure_candidates",
        capture_fallback,
    )

    task = asyncio.create_task(
        server._ds_filter_candidates(
            "rare-name",
            normal,
            mode="search",
            max_results=2,
            force_keep_ids={"ticket"},
            allow_empty=True,
            weak_entity_tickets=[ticket],
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fallback_calls
    assert all(
        call == {"ids": ["normal-1", "normal-2"], "forced": set()}
        for call in fallback_calls
    )


@pytest.mark.asyncio
async def test_ds_disabled_drops_ticket_without_calling_model(monkeypatch):
    _require_ds_ticket_parameter()
    monkeypatch.delenv("OMBRE_DS_FILTER_ENABLED", raising=False)
    normal = [_candidate("normal-1"), _candidate("normal-2")]
    ticket = _candidate("ticket", weak_entity=True)

    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("disabled DS must not run the model")

    monkeypatch.setattr(server, "_ds_semantic_select", must_not_run)

    selected = await server._ds_filter_candidates(
        "rare-name",
        normal,
        mode="search",
        max_results=2,
        weak_entity_tickets=[ticket],
    )

    assert selected == normal


@pytest.mark.asyncio
async def test_ticket_off_path_is_exact_legacy_ds_behavior(monkeypatch):
    _enable_ds(monkeypatch)
    normal = [_candidate("normal-1"), _candidate("normal-2")]
    observed = []

    async def keep_second(_query, rows, _keep, _max_results):
        observed.extend(row["id"] for row in rows)
        return rows[1:2]

    monkeypatch.setattr(server, "_ds_semantic_select", keep_second)

    selected = await server._ds_filter_candidates(
        "ordinary query",
        normal,
        mode="search",
        max_results=2,
        allow_empty=True,
    )

    assert observed == ["normal-1", "normal-2"]
    assert [row["id"] for row in selected] == ["normal-2"]
