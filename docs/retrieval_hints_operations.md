# Retrieval hints: pilot deployment and rollback

This is an operator handoff, not evidence of deployment or recall quality.
The implementation adds derived hints; it never migrates bucket bodies or
vectors. No scheduler is installed by the CLI. Keep the four switches OFF until
the deployment owner has connected the real worker and approved the pilot.

## Changes and runtime boundaries

- `retrieval_hints.py`: bounded body-only generation and source-bound v2 fields.
- `retrieval_hints_storage.py`: SQLite jobs/immutable versions/publication pointer
  plus a separate PostgreSQL side table. Only a dual-verified publication is read.
- `retrieval_tokenizer.py`, `bm25_index.py`, `bucket_manager.py`: an isolated
  dictionary and hint candidates; generated hints never become manual keep keys.
- `lmc5_proposer.py`, `night_run_coordinator.py`, `night_run_runtime.py`: reuse the
  existing proposal response, stage hints after a successful original write.
  There is no second model call for those fields. Publishing still requires PG.
- `server.py`: reject source-confirmed quoted candidates at the existing main
  and borrowed-candidate boundaries. Unknown does not mean quoted or certified self.
- `tools/retrieval_hints_{prepare,batch,export,compare,rollback}.py`: pilot tools.

The code baseline is `4d55e84`. The live server has independent changes. Apply
only the reviewed task diff to current production files; never overwrite the
whole server or use this worktree as an entire deployment image.

Default switches:

```text
OMBRE_RETRIEVAL_HINTS_ENABLED=0
OMBRE_RETRIEVAL_HINTS_BACKFILL_ENABLED=0
OMBRE_RETRIEVAL_ATTRIBUTION_ENABLED=0
OMBRE_PRIVATE_RECALL_DICT_ENABLED=0
```

Use existing `OMBRE_DS_FILTER_FALLBACK_*` and `OMBRE_PG_RECALL_DSN` bindings on the
deployment host. Do not copy credentials from a different host or put them on a
command line. The private dictionary and source audits are not repository assets.
Runtime dependencies include `psycopg`; verify them in the actual worker image.

## Before the first production pilot

The deployment owner must supply these environmental facts:

1. The real active bucket root and the frozen approved source list (at most 200).
2. The existing shared production lock, exposed to the worker namespace with
   the same device/inode identity. A new container-local file is not that lock.
3. A real read-only busy probe returning exactly JSON with a boolean `busy`.
   Missing, malformed or failing probes stop the run; do not substitute a script
   that always reports idle. The actual probe has not been delivered here.
4. Existing maintenance-barrier compatibility, PG connectivity and permissions
   for the new side table; confirm no competing writer during receipt capture.
5. The deployment and controlled runtime-loading procedure. This document does
   not authorize a restart or treat environment variables in a shell as proof
   that a running process has loaded the new code.

Stop/disable this worker and its readers before the first before-image. Keep the
operator's private evidence directory separate from bucket bodies and original
conversation mirrors. In the examples below, angle-bracket values are explicit
operator-supplied paths or observed lock identities, not literal shell syntax.

```text
python tools/retrieval_hints_rollback.py capture
  --buckets-dir <BUCKET_ROOT> --approved-sources <APPROVED_JSON>
  --host-lock <EXISTING_SHARED_LOCK> --lock-device <DEVICE> --lock-inode <INODE>
  --output <PRIVATE_BEFORE_JSON>
```

Capture is read-only for both databases, including when the side tables do not
yet exist. It saves exact scoped versions, publication pointers, PG readiness
and dictionary bytes. Receipts are exclusively created with mode 0600; they may
contain private derived text and must never be committed. Acquiring operator
leases may create the sidecar worker lock/maintenance metadata, not hint rows.

Install the approved private dictionary only after this before-image. Keep its
private source map alongside the pilot evidence. Verify every approved token
against the loaded isolated tokenizer; do not load it into global jieba state.

## One bounded night run

Only `[01:00,06:30)` Asia/Shanghai permits generation and publication. A pass is
limited to 200 source IDs and 30 minutes, single concurrency, 15 seconds per
generation, no provider retries. Three consecutive generation or publication
failures stop the batch durably. Existing uncertain attempts are not regenerated.
New writes are queued for the night, not promised to have immediate hints.

After the deployment owner enables the approved switches in the worker:

```text
python tools/retrieval_hints_batch.py
  --buckets-dir <BUCKET_ROOT> --approved-sources <APPROVED_JSON>
  --host-lock <EXISTING_SHARED_LOCK> --lock-device <DEVICE> --lock-inode <INODE>
  --source-contexts <PRIVATE_SOURCE_AUDIT_JSON> --limit 200
  --busy-probe <REAL_READ_ONLY_PROBE> <PROBE_ARGUMENTS>
```

`--busy-probe` must be last. Source audit text is used for local attribution
validation only; backfill generation still sends only the bucket body. The
existing nightly proposer keeps its already-authorized source envelope.

Do not call `pass_finished` acceptance. Report generated, published, deferred,
source-changed, uncertain and failed counts, including all 200 original rows.
Missing usage is unknown, not zero. Provider cost and complete breath replay cost
are separate. Capture an after-image with the same IDs and command as above,
after stopping the worker/readers and before another writer changes this scope.

## Export and comparison

`retrieval_hints_export.py` writes a versioned manifest to a separate private
directory. It labels evidence as bucket-body evidence, not original chat lines.
It does not connect a Twin consumer or edit the original conversation mirror.

`retrieval_hints_compare.py` compares OFF / dictionary / hints / both on the full
frozen corpus with zero provider calls. It reports BM25 candidates only, not
vector/DS/final breath or injected context. Final OFF/both logical breath replay
and human relevance labels remain separate required deliverables; the full
230-call replay runner is not supplied by the candidate comparison tool.

Idempotency is currently bucket + source version + prompt, not event identity
across different buckets. Source-confirmed event grouping needs a separately
settled manifest/reader contract; do not silently skip buckets or call all
near-similar memories duplicates. Original-event backlog completeness also
requires the source-event extraction ledger, not just existing-bucket hints.

## Scoped rollback and interruption recovery

First stop the worker and disable all four switches in the actual readers;
verify their runtime state. Preserve before/after receipts and usage records.
Use the same approved scope and real shared locks. Preview before applying:

```text
python tools/retrieval_hints_rollback.py restore
  --buckets-dir <BUCKET_ROOT>
  --host-lock <EXISTING_SHARED_LOCK> --lock-device <DEVICE> --lock-inode <INODE>
  --before <PRIVATE_BEFORE_JSON> --after <PRIVATE_AFTER_JSON>
```

Only add `--apply` for the explicitly authorized rollback. The tool compares all
scoped components before its first mutation; a later publication or dictionary
update is a conflict, not permission to overwrite it. Restoration keeps old
immutable versions, marks new SQLite versions withdrawn and new PG versions not
ready, and restores the previous publication pointers and dictionary. If the
dictionary did not exist before, the new file is removed; its bytes remain
recoverable in the after receipt. Jobs, attempts, usage and batch receipts remain
untouched so a rollback cannot cause paid regeneration.

PG, SQLite and dictionary changes are not one distributed transaction. Readers
must stay OFF during rollback. If interrupted, rerun the identical before/after
pair: already-restored components are accepted, conflicting new changes are not.
Export a fresh manifest after restoration before reconnecting any consumer; the
tool does not rewrite a separate consumer's `CURRENT.json` behind its back.
Rebuild/reload the affected in-process index through the deployment owner's
normal procedure before enabling readers. No source bucket, vector, raw event,
original conversation or other bucket's sidecar is deleted or replaced.
