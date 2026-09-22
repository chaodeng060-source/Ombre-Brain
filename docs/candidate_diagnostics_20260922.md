# Candidate-only diagnostic seam (2026-09-22)

Local implementation only. This is an observation entry point, not a recall
quality fix or evidence that the recent-memory incident is resolved.

## Source and authority

Implements the task's 2026-09-22 08:27 B decision: one existing, sanitized and
truncated query embedding is permitted; query expansion, gate, dehydration,
seen/body/activity writes, shadows and request-started background work are not.
No NAS deployment, restart, reindex, bucket edits or real-provider trial was
performed. The next production operation belongs to the task creator in the
approved attended window, not to this local implementation.

Base `5ba981747dd4c0863eb9c0898020e69c36e07768`; the isolated baseline commit
`bda623f` aligns only `bucket_manager.py` with the already-installed NAS version.
It is NOT part of a new production rollout. Verified pre-change SHA-256:

| Source | Production baseline SHA-256 |
| --- | --- |
| server.py | 185c9ba326ff26ad197fd27407c28f0a1abf3410afa7b8493e4f83e386a37193 |
| embedding_engine.py | c8fc782dceaac675f211c665ed8beb52fd11d81cdac45efa5cb90af7efc742a7 |
| bucket_manager.py | 93f7d9426c26ff88c2630b40f103acfccf145f84063c221e4f6209d2ecb12c2c |

Feature review/diff must start at `bda623f`. Do not replay the baseline graft
over a newer production bucket manager. Deployment is separately authorized.

## Protocol

`POST /api/breath-candidates`, under the existing `/api/*` bearer middleware.
Independent `OMBRE_CANDIDATES_DIAGNOSTICS_ENABLED=1` enables this route; default
OFF returns 404 without entering breath. This flag never changes an ordinary
`breath`, MCP `breath`, or `/api/breath` request.

Required: `query` (nonblank text) and `target_ids` (1-64 bucket IDs). Optional
parameters retain breath semantics: `policy`, `world`, `domain`, `since`,
`until`, `session_id`, `max_results`, `relation_depth`, `valence`, `arousal`,
`self_valence`, `self_arousal`. Empty-query surfacing and `domain=feel` are not
candidate-search requests and return 400. Bodies are bounded at 64 KiB.
No query examples, real IDs or memory bodies are published in this document.

The route sets request-local context and calls **the real breath function**.
Observations are taken inside its existing keyword/vector/curated/entity/RRF/
authority/retention/escort/dedup/seen/anchor/worklog/state-link/topic stages.
Return occurs before pre-gate partial assembly, DS, dehydration and delivery.
Relation expansion normally happens after DS and is explicitly
`not_executed_post_gate`; it is not fabricated as a pre-gate channel.

Response fields contain only IDs, enum names, numbers and booleans:

- `stages`: target membership, actual 1-based stage rank, score and pool count.
- `target_drop_observations`: observed present-to-absent transitions per flow.
  A later escort can restore a target; a recorded earlier drop is not a claim
  that the target is absent from the final candidate list.
- `target_rejections`: explicit non-keyword authority, link, scoring reasons.
- `bm25`: resident index membership, positive-score corpus rank and normalized
  score, mode/dirty/readiness/status. BM25 rank is NOT blended keyword rank.
- `vector`: backend, actual ANN top-k rank, selected exact cosine score, and
  membership. Outside ANN top-k has null rank, not an invented global rank.
  SQLite can report an actual full-scan rank; malformed stored rows remain
  present but unscorable. Failed selected probes mean unknown, not absent.
- `candidates` and `state_links`: pre-gate IDs/scores, never text or titles.
- `differences`, `candidate_set_may_differ`, `completed` and `partial`: explicit
  observational limits. A failed shadow lexical observation is marked partial;
  primary keyword/vector failure returns 503, never successful absence.

## Read-only and comparability limits

The one embedding is the existing redaction-then-2000-character truncation
path. A request-local SDK copy has `max_retries=0`; the shared client and its
circuit-breaker state are unchanged. PG diagnostic connections enforce
`default_transaction_read_only=on`; SQLite fallback uses `mode=ro` and never
initializes/writes the DB or persistent caches. Setup loops, snapshot refresh,
BM25/hint refresh, PG shadow and fusion shadow are not scheduled by this request.
The existing resident snapshot must already be warm; a cold one returns 503.

This is deliberately NOT claimed to be byte-equivalent to a normal request:
query expansion and live chord are omitted, snapshots/hints are not refreshed,
and SQLite fallback reads the store instead of refreshing the live cache.
These differences are always reported. Ordinary requests with no diagnostic
context retain their previous behavior, including refresh and shadow hooks.
Application-level seen/access/body/sidecar writes are excluded. Filesystem
mount atime behavior and independently running maintenance jobs are not changed.

## Offline verification and delivery boundary

The primary executor integrated the bounded worker's vector changes, fixed
SDK retry and ordinary helper-signature issues, and ran the checks independently.
New tests use mechanical IDs/numeric vectors and recording/MockTransport clients;
they are not invented semantic acceptance cases and never call a real provider.

New seam tests: 30 pass, including full breath with the real vector core, SDK
transport retry suppression, no setup/gate/assembly/write hooks, dirty/cold
snapshot handling, file-open write guards, SQLite bytes/mtime unchanged, API
authentication, and ordinary flag-OFF/ON candidate equivalence.

Expanded regression: 158 passed, 11 skipped, 1 inherited failure. The failure
is `tests/test_entity_recall.py::test_entity_only_channel_reuses_all_authority_filters`:
its old DS mock rejects the existing `recall_policy` argument. The exact same
failure was independently reproduced on clean baseline `bda623f`; neither the
test nor production behavior was changed to hide it. The 11 skipped checks are
environment-conditional and are not counted as passing.
Final rerun after integration hardening, excluding that entire pre-existing
entity test file: 150 passed, 11 skipped. The 8 other entity tests passed in the
expanded run above; this is not presented as a fully green unfiltered suite.

Commands use the existing isolated test interpreter, with provider keys unset:

```text
env -u OMBRE_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY \
  /opt/claude-twin/data/ombre-vps-runtime/venv/bin/python -m pytest -q --tb=short \
  tests/test_candidate_vector_diagnostics.py tests/test_breath_candidate_diagnostics.py \
  tests/test_pg_recall_fastpath.py tests/test_keyword_live_prune_20260918.py \
  tests/test_breath_dehydration_read_only.py tests/test_entity_recall.py \
  tests/test_recall_authority.py tests/test_state_aware_recall.py \
  tests/test_recall_perf_caches.py tests/test_mcp_auth.py \
  tests/test_mood_congruent_recall.py tests/test_rg_literal_recall.py tests/test_timeline_breath.py
```

Next: creator reviews this default-OFF seam and owns attended deployment. Only
then run the single authorized original query with the three private targets,
and close the missing per-channel evidence in the original incident report.
Do not run a 35-query quality experiment or call the original incident fixed
merely because this diagnostic implementation passes local tests.
