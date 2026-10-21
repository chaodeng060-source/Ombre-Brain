# Memory source fidelity

Night proposals use complete raw messages and exact parent/root IDs, preserving speaker and date metadata. They never feed a byte-truncated JSON prefix to the model. Parent IDs resolve only when unambiguous in the same conversation.

The proposer checks unsupported animal classes, colors, numbers, dates and explicit named entities. One existing contract-repair attempt remains; unresolved or unsupported proposals do not become facts. This lexical check is deliberately limited and does not prove universal semantic entailment.

New buckets carry raw SQLite row pointers. Legacy receipt options remain unchanged, so pointer enrichment does not break idempotency. Raw conversations stay outside bucket bodies and remain append-only.

Summary generation uses the same bounded source checks. Detected drift falls back to a literal source excerpt without an extra model call. Production has additional summary-cache paths: the included live patch preserves those paths and the existing speaker/date, omission and event-time fixes.

`tools/repair_memory_fidelity.py` is dry-run by default. An explicit owner-authorized plan must bind an exact bucket ID and previous body hash to independently source-reviewed body, summary and source IDs. It takes one SQLite/code backup per maintenance run and one original file per bucket, retains original events, updates vectors, clears stale summaries and rebuilds current cache entries. Primary-authored E content stays immutable.

`tools/backfill_memory_source_refs.py` resolves legacy source pointers without changing body prose, source events or existing provenance. Missing or ambiguous originals remain reported; it never mints a reference from a summary.

Validation: 313 related regression tests; 30 focused source/summary/maintenance tests; actual NAS source reconstruction returned the complete source and its exact parent, rejected the unsupported cat claim, and accepted the supported border-collie claim. The source patch was loaded by the NAS service on 2026-10-21; health remained OK and the canonical inventory stayed at 22,058 records. Historical correction progress and private evidence are intentionally kept outside Git.
