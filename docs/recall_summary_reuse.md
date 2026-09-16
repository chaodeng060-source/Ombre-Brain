# Exact-body recall summary reuse

`server._dehydrate_for_recall` uses `Dehydrator.dehydrate_recall_with_source`.
It shares raw derived summaries between buckets; each caller formats its own
metadata afterwards. Neither this path nor its background producer writes
Markdown, activity, embeddings, factual metadata or the legacy summary database.
The existing bounded LRU and `.recall_cache/recall_dehydration_cache.db` are reused.
The existing private-path checks, permissions and nonblocking failure behavior
remain in force. No new database, daemon or dependency is added.

The strict namespace binds exact raw-body SHA256, full redacted-input SHA256,
model, configured/client endpoint, prompt, max tokens, temperature, thinking
mode and redaction/output contract versions. The serialized request and client
are captured before scheduling; the same key labels request, memory and disk.
Change the output/redaction version when their behavior changes.

Unversioned bucket summaries and old cache namespaces are not imported into the
strict path: their original output parameters cannot be proved. Existing legacy
callers retain their behavior. Existing rows and bucket frontmatter are not
deleted or migrated. This introduces one cold computation per encountered
strict key; there is no eager model prewarming.

Same-key concurrent misses share one producer within one Dehydrator/event loop.
One cancelled waiter cannot cancel other waiters. If all waiters disappear the
producer is cancelled unless async fallback explicitly owns its completion.
Failures, cancellations and empty/short model outputs are never cached.
This is not a cross-process provider lock; separate service processes may each
compute the same cold key before either persists it.

The existing 300-character async fallback, work-hour deferral and two-background-
provider limit are retained. Cache hits can still be served during work hours.
`OMBRE_RECALL_DEHYDRATE_ASYNC=0` retains synchronous behavior. The old
`recall_frontmatter_cache_enabled=false` setting still disables async fallback
for compatibility, but this helper no longer reads or writes frontmatter.
The separate provider-free E-chord delivery renderer is unchanged by this patch;
DS selection, ranking, relations and source write paths are not changed.

## Measurement

Logs use `recall_summary_cache at=<UTC epoch> event=... key=<hash> saved_calls=N`.
They contain no body, summary, prompt, metadata or credentials.

- `request`: a long-content strict summary request (short passthroughs excluded).
- `memory_hit`, `persistent_hit`, `coalesced_hit`: successful summary reuse.
- `provider_start`, `computed`, `failure`, `cancelled`: producer outcomes.
- `passthrough_async`, `passthrough_deferred`: fallback, never counted as savings.

Each successful reuse avoids one logical client call except a work-hour deferred
async request, which would not schedule a call anyway. This is a reuse counter,
not a causal savings estimate against the previous release (which had caches),
HTTP retry counts, tokens or billing. Hit rate includes successful coalescing;
its denominator is logged long-content requests. Window-crossing requests may
start or finish outside the selected interval; inspect coverage before acceptance.

```sh
python tools/recall_summary_window.py /operator/all-rotated-logs.txt \
  --start 2026-01-01T06:30:00+08:00 --end 2026-01-02T06:30:00+08:00
```

Use the actual deployment interval, not the example dates. Preserve all rotated
logs and verify process lifetime/log continuity. `window_elapsed` is only a time
check, not proof of complete logs or production acceptance. No events means
unknown hit rate, not zero production cost.

## Read-only pending census

```sh
python tools/pending_body_duplicates.py --ledger /operator/pipeline.sqlite3 --records
python tools/pending_body_duplicates.py --snapshot /operator/private-census.json
```

The SQLite connection is `mode=ro` plus `query_only`, in one read transaction.
Pending means no terminal `zero_candidates`/`candidates_persisted` outcome.
The census verifies raw payload digests and groups the exact `text` field, not
the unique event/chunk envelope. It distinguishes chunk, distinct-event and
nonblank-event denominators. Repeated members and copies beyond one per group
are separate figures. Hash-only records enable independent recomputation;
retain them privately, never publish actual vault statistics or identifiers in
this repository. Identical text is not proof of identical event identity and
is not authorization to skip/delete proposer work.

## Safe integration and acceptance

This patch's parent predates a separately deployed DS fallback. Do not deploy
its whole-tree archive over live code. A controlled integration must first
preserve that live change and the independently reviewed ingress repair, apply
this narrow patch, test the combined tree and regenerate reconciliation.

Use the existing atomic deployer and reviewed ingress deployment/rollback
wrappers in that integrated tree. Capture a full source/container rollback
anchor before switching. Avoid 01:00–06:30 Asia/Shanghai; do not restart active
chat responses. Roll back code/container together; do not restore, delete or
rewrite the memory volume. The strict sidecar rows are disposable, namespaced
and ignored by old code, so rollback needs no data migration.

After authorized deployment, verify loaded source/process and natural recall,
then collect a complete 24-hour log window. Report hits, eligible requests,
logical invocations/avoided calls, failures/deferred requests and coverage.
Local fake-provider tests do not establish runtime loading, production savings
or the 24-hour result.
