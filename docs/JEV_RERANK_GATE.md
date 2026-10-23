# Native JEV at the recall relevance gate

JEV replaces the primary small-model relevance pass when explicitly enabled.
It scores existing candidates; it does not generate summaries, write facts,
embed memories, expand retrieval or edit the X/Y/Z/E/M data. The existing gate
switch and mode selection still decide whether any paid relevance pass runs.
Disabled deployments retain the previous provider path.

## Protocol and selection

- Native TypeSafe `POST /v1/systemone`, `questions.rel.type=score`, four relevance
  levels, numeric score from 0 through 3. The adapter maps the score to 0–100.
- `jev-latest` is an alias, not a version assertion. The real resolved model is
  recorded in aggregate diagnostics. A response such as `jev-1.14.0` is valid.
- A candidate contains its human title and **complete redacted body**; raw
  session references, file paths, bucket IDs and other metadata are omitted.
  Retrieved memory bodies remain the source, not a newly generated summary.
- On current NAS builds the existing `_ds_score_mode_select` owns the fixed
  threshold, key bonuses and topic deduplication. Domain/world/author/Z filters
  remain in the surrounding recall pipeline. No second primary DS/Gemini call
  runs after a successful native JEV pass.
- Older Git checkouts without that score-mode selector use a compatibility
  selection: stable descending relevance, threshold 70, preserve forced IDs.
  This path is not a replacement for the newer NAS selector.
- One invalid or failed score invalidates the batch. Cancellation drains
  outstanding tasks. The existing outer gate retains its timeout, secondary
  provider and fail-closed policy. The adapter never automatically resends a
  paid HTTP request. Failure is not treated as an irrelevant-memory verdict.

## Configuration

Use the disabled example in `config.example.yaml`. An enabled section is:

```yaml
jev_reranker:
  enabled: true
  model: jev-latest
  endpoint: https://api.typesafe.ai/v1/systemone
  api_key_file: /data/.secrets/jev-rerank.env
  concurrency: 4
  cache_ttl: 600
  cache_max: 512
```

The credential is read from `OMBRE_JEV_API_KEY`, then `JEV_API_KEY`, then the
specified private env file. The file must have no group/other access; its key
line is `JEV_API_KEY=...`. Keep it outside Git. Never paste it into config
examples, task cards or logs. `OMBRE_JEV_RERANK_ENABLED=0` overrides the YAML
switch and restores the previous primary provider after the normal reload.

The cache stores scores plus aggregate usage under digests covering provider,
model, credential identity, question, full body and prompt version. A change
after character 200 also invalidates it. TTL and entry count bound resident
memory; the cache is not another session archive. Repeated identical scoring
uses the cache and incurs no new API call. This is not an upstream cache-hit
claim or a promised account-wide cache percentage.

`/health.reranker` reports enabled/provider/revision and last-batch counts,
model, requests, cache hits and token usage. It contains no query/body/key.

## Applying to an already customized NAS module

Do not replace live `server.py` with an older Git copy. This repository's NAS
runtime has additional fixes. First snapshot **host code, container code and
the bind-mounted config separately**, then generate an incremental copy:

```bash
python tools/patch_jev_gate.py PATH_TO_SAVED_LIVE_SERVER > NEW_SERVER.py
```

The patch adds one import, the JEV provider branch in `_ds_semantic_select`,
and one health field. It preserves all other function ASTs. Reapplying it is
idempotent; an incompatible signature/partial patch is an explicit error.
Install `jev_memory_gate.py` and `jev_reranker.py` alongside the resulting
module, configure the private key path and reload the memory service normally.
The host and container copies may have different prior contents; patch each
from its own snapshot. Verify the running health revision and a real recall,
not only a source hash. For rollback restore each original entry point and
config from its matching snapshot, then reload; new inert helper files do not
need to be deleted to stop JEV.

## Verification and source attribution

Offline tests use synthetic memories and a mock HTTP transport. They cover
alias resolution, malformed scores, complete batches, cancellation, cache
isolation, concurrency, no retries, route bypass, full-body redaction and
compatibility with the existing NAS selector. Run with the project's normal
experimental environment flags cleared.

This integration follows the native request and score contract in
[LMC-5 AML JEV adapter at d02563a](https://github.com/wuxuyun0606-collab/lmc-5/blob/d02563ad7885c027778429287a29d35e3fc2d6e7/extras/aml/jev/rerank_jev.py)
and [TypeSafe System One API](https://docs.typesafe.ai/api/system-one).
The bounded async client, this repository's gate integration and compatibility
patch are local implementation decisions. The competition's financial ledger,
600-second evaluation budget and exam bridge are not production requirements
for this existing recall service. The author's reported Luna/JEV/Clef noise
percentages are not measurements of this installation; synthetic acceptance
does not establish those percentages either.

## 2026-10-23 deployment result: inactive on NAS

Implementation commit `fe931af` passed 173 selected tests on both the Git
checkout and a copy of the actual NAS runtime. The latter needed its vendor
package/static packaging fixtures and one test-double `recall_policy` keyword
to match the newer running API; assertions were unchanged. These are the same
tests on two baselines, not 346 distinct tests.

The direct credential works on the VPS: one synthetic score returned HTTP 200,
resolved `jev-1.14.0`, 164 ms, 342 input/17 output tokens. NAS activation was
then tested through the real authenticated recall hook. It returned HTTP 200
but the JEV last-batch receipt stayed empty: the legacy fallback had answered.
Server logs and a bounded direct probe identified HTTP 451, explicitly stating
TypeSafe is not available in that region. This was not counted as successful
JEV acceptance.

The latest saved OpenCode credential is valid: both its Go and Zen model
catalogs return 200. The older `.env`/Zen-profile keys returned 401. Official
OpenCode supports the same protocol at the current
`https://opencode.ai/inference/systemone/v1/systemone` endpoint, with
`jev-1.13` / `jev-1.13-free`; see
[OpenCode Console models](https://opencode.ai/v2/docs/console/models/#jev).
At acceptance the paid model returned 402 (insufficient account funds); the
free model returned 451 (endpoint unavailable) at both the current endpoint
and the documented legacy Zen endpoint. Changing to a normal chat-completions
request would not solve this protocol/service issue.

Therefore the NAS original host entry point, container entry point and YAML
configuration were restored from their separate exact snapshots. The installed
new helper modules remain inert; no active gate change remains. The runtime
feature is **blocked on a usable service route**, not marked delivered. No
account top-up, region work-around, repeated paid retries or new proxy was
performed. Future activation requires one successful NAS synthetic score,
then a real recall receipt showing JEV requests and a second identical query
showing cache hits. A healthy HTTP recall alone is insufficient evidence.
