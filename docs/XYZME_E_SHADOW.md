# XYZME E axis: E0 shadow collection

E0 is an observation pipeline, not a recall feature. Its automatic source
universe is the union of:

- validated, already-redacted X-axis candidate payloads from the LMC-5 ledger;
- Markdown memories recursively read from exactly the five authoritative
  curated roots: `permanent`, `dynamic`, `archive`, `feel`, and `涩涩`.

Both source kinds pass the official type, keyword, and emotional-link gate.
The curated reader is strictly read-only: it rejects unsafe ancestors,
symlinks, hardlinks, malformed frontmatter, and duplicate IDs. Timestamps are
timezone-aware by default. An audited legacy corpus may explicitly set the
strict boolean `e_axis_shadow.legacy_naive_timestamps_utc: true`; only complete
quoted `YYYY-MM-DDTHH:MM:SS[.ffffff]` strings or YAML-decoded naive full-second
`datetime` values are then interpreted as UTC. YAML dates, quoted date-only
values, missing-seconds values, and non-boolean configuration still fail
closed. The selected policy is recorded in every coverage report as
`curated_timestamp_policy=aware_only|legacy_naive_utc`. The reader never
creates, touches, or rewrites a memory file, never reads raw events, and never
traverses outside those five roots. Both curated Markdown and
the LMC-5 SQLite source are opened through held directory descriptors with
`O_NOFOLLOW|O_NOATIME`; unsupported filesystems or permissions fail closed
instead of falling back to metadata-touching reads. The SQLite main file is
copied into a private in-memory database, so reading never creates source
`-wal`/`-shm`; a non-empty WAL or rollback journal is an explicit failure.

It writes three private sidecars under <buckets_dir>/.axis/:

- e-shadow.jsonl: immutable success or failure rows;
- e-shadow-attempts.jsonl: one outcome per run and source, with source IDs
  hashed and no memory text or provider output;
- e-shadow-coverage.jsonl: aggregate coverage, failure, distribution, enum,
  and distinct-natural-day reports for the formal automatic cohort. Reports
  break down source kind, memory type, trigger reason, skip reason, and failure
  code, including retryable, terminal, and unresolved failures. Coverage
  includes scored, terminal, unresolved, score-rate, and resolved-rate fields;
  a zero denominator is explicit and the CLI exits non-zero with
  `source.no_eligible`.

Every score row carries provider, model, scorer, rubric, source digest, E run
ID, trigger reason, and timestamp. Valence is [-1, 1]; arousal, tension, and
confidence are [0, 1]; response tendency and growth delta are strict enums.
Before curated title/content leaves Ombre for the configured scorer, it passes
the existing secret/credential redactor; emotional wording is preserved.
Timeout, empty output, incomplete output, malformed JSON, schema errors, range
errors, and low confidence are separate failures. Retryable failures remain in
the next run's backlog. A terminal failure remains non-green on later runs even
when no new provider call is made; it cannot disappear behind a zero-new-error
exit code.

`provider_name` is required and must name the real provider; there is no
generic `openai-compatible` default. The scorer identity includes a digest of
provider endpoint, model, rubric, prompt contract, token/content limits,
confidence threshold, temperature, and timeout. Changing any of those starts a
new calibration cohort instead of silently mixing distributions.

The producer reads no raw events and no memory directory outside the five
curated roots. It never updates candidate status, memory body/frontmatter,
facts, relations, decay, RRF, or any recall score. Every annotation permanently
sets shadow_only=true and affects_ranking=false.

The authoritative score ledger is contract_version=2. It intentionally fails
closed on v1 rows; an existing v1 ledger must be archived or migrated in a
reviewed maintenance window before this producer starts. Deployment must first
inspect the exact production ledger state; it must not assume that the ledger
is empty.

The authenticated manual `/api/e-axis/shadow` route uses the same v2 contract.
Its request must contain exactly `bucket_id`, `source_digest`, `provider`,
`scorer`, `model`, `rubric_version`, `run_id`, `trigger_reason`, and `score`.
The server, not the client, stamps every such row as
`source_kind=manual_bucket`. The server re-computes the named bucket's digest
and records a terminal `source_digest.mismatch` row instead of accepting a
stale score. Manual rows are completely excluded from formal automatic cohort
membership, distinct-natural-day counts, coverage and score distributions,
and all promotion evidence; they can never seed or accelerate E1.

The E job has a separate cron entry from the conservative Stage-1 night run.
An E failure can alert independently but cannot roll back or re-label a
successful X/M night run.

Deployment order is mandatory:

1. inspect the exact production deployment directory, active image/container,
   E ledgers and sidecars, and host cron in read-only mode;
2. obtain review approval for the exact diff and exact deployment targets;
3. create exact, checksummed backups of every file, image/rollback anchor, cron
   entry, and existing ledger that will be changed;
4. only then may an authorized deployment write NAS files, rebuild or switch
   the image, install `cron/run-e-axis-shadow.sh`, and verify live shadow-only
   health and rollback.

The E script uses a separate non-blocking lock, returns exit 75 with
`run.busy` on overlap, and does not invoke the core X/M night runner. Current
delivery state (2026-07-31): this work is not deployed; this task has not
changed NAS files, the production image/container, or host cron.

## E1 promotion gate

E0 reports always set promotion_eligible=false. At least 30 distinct Shanghai
natural days within one formal automatic provider/model/scorer/rubric cohort
are necessary but not sufficient; manual API rows never count. E1 also
requires adequate coverage and sample size, stable numeric/enum distributions,
resolved failure debt, provider calibration, real-query validation, and
explicit human approval. There is no automatic promotion switch. Until then,
the E ledger must not be read by recall, emotion resonance, perception, or
ranking code.
