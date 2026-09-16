# Event-bound hold replay

`POST /api/hold` accepts an optional `idempotency_key` (1–256 ASCII letters,
digits, `_ . : -`, starting with a letter or digit). Invalid explicit keys
return 400. Old unkeyed requests retain their existing response and merge rules.

The write identity is SHA256 of the versioned JSON tuple:

`["ombre.hold/v1", event_key, effective_world, feel, pinned, body_sha256]`.

HTTP content is trimmed as before; SHA256 is of that exact UTF-8 body. The first
24 hex digits of the identity are the bucket ID. The full identity and body
SHA256 are written into the first Markdown frontmatter, atomically with its body.
The existing per-bucket process/file lock protects check-and-create. A replay
re-reads the file, checks both hashes and actual body equality, logs
`hold replay skipped`, and returns the same ID without updating the bucket.
A conflicting existing file returns 409, never overwrites it.

Successful keyed responses add `bucket_id` and `replayed`. They attest only to
the body, not completion of embedding, entity edges or the separate provenance
receipt protocol. No source receipt is fabricated. Replay before completed
optional post-processing does not resume those operations in this change.

Same text in another event/world/storage mode, or changed text in the same
event, remains a separate write. Keyed writes bypass heuristic merging. This
is deliberate: world+text alone cannot distinguish two real occurrences of the
same words. Callers must provide a stable, namespaced occurrence identity.
The Twin imprint adapter uses its durable source event ID; legacy candidates
use occurrence timestamp, session and original conversation fields. Cached
pre-fix requests receive that key at send time, without another model judgment.

There is no backfill, scan-to-delete, migration or mutation of existing buckets.
Old unmarked buckets are not assumed to be the same event: a still-pending old
request may create one new canonical bucket on its first keyed retry, then no
more. Image-bearing keyed MCP hold is not supported. Unkeyed paths, including
manual hold/grow, are not globally deduplicated by body.

## Deployment and rollback

Independent write-loss review must precede production changes. Deploy Ombre
before loading the Twin writer change, and never restart an active Twin reply.
Prepare the source archive from the reviewed commit with `git archive`; verify
the production env/container anchor and the live/source reconciliation manifest.
The current manifest includes the narrow changed module hashes; unrelated live
drift must still pass the existing deployer's own checks. No secrets belong in
an archive or a public diff.

```sh
bash scripts/deploy_hold_idempotency.sh --env-file /operator/production.env \
  --archive /operator/reviewed-source.tar --archive-sha256 ARCHIVE_SHA256 \
  --commit FULL_REVIEWED_COMMIT --preflight
# After approval and preflight, the same command without --preflight deploys.
bash scripts/rollback_hold_idempotency.sh --env-file /operator/production.env
```

The wrapper refuses deployment from 01:00 through 06:29 Asia/Shanghai, then
uses the existing atomic deployer (backup, preserved old container/source,
failure recovery and health checks). The rollback helper reads that deployer's
receipt, verifies the exact current and preserved containers, and restores the
old source/container under the production lock. It retains the reverted source
and container; it never restores or deletes the data volume. Rollback removes
the new replay protection, so coordinate the Twin writer rollback too. Emergency
rollback is not time-gated. A health response is not behavioral acceptance.

## 24-hour comparison

At actual deployment record a timezone-aware cutover. Verify the historical
writer's timezone; do not guess it from the browser timezone. Before deployment
capture a baseline and preserve it privately, then rerun after 24 hours:

```sh
python tools/hold_duplicate_window.py --data-dir /data \
  --cutover ACTUAL_ISO_TIMESTAMP_WITH_OFFSET --naive-timezone UTC --hours 24
```

Run where the vault is read-only accessible and `python-frontmatter` is installed.
The helper starts no application, model or SQLite writer. It emits aggregate
JSON only; reports time/read coverage, created-date fallback and incomplete
observation windows. Counts distinguish within-window duplicate copies from
new bodies equal to earlier buckets. Grouping is exact body+declared world+type,
not semantic event identity. Historical unmarked/missing-world buckets remain
unknown; the report is not permission to skip or delete them. Preserve the
pre-cutover result because later legitimate edits can change a retrospective
baseline. Collect natural `hold replay skipped` log counts as supporting evidence.

Local tests do not establish production loading or the 24-hour outcome.
