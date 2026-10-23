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
