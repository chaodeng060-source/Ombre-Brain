# Score-gate reason receipt — local implementation, not deployed

Task: `task_wr_f9ad5e2cbf63_01`. Baseline: `5bbff3ef367042af9eafc8ec3c041a23674d6097`.
Its server.py SHA256 matches the read-only NAS source checked on 2026-09-20:
`779c9286c0f36c5c8da6105901e9a992848396cbab7829ccfeebf90348d2096a`.
No unaccepted focus experiments or dirty IDF changes are included.

- `OMBRE_DS_GATE_REASONS_ENABLED` defaults ON; `0` restores the old score/cache/response shape.
- The existing model reason is retained by candidate index, including parallel batches and cache hits.
  Cache keys, prompts, request count, thresholds, bonus rules and selected results are unchanged.
- Old two-element cache entries remain hits. Missing reasons are explicitly `absent_legacy_cache`,
  `absent_in_response` or `unparsable`; never reconstructed from a score or requested with another call.
- Only score-threshold rejections are recorded. Capacity/topic deduplication is not misreported as a
  model rejection. Legacy non-score modes do not invent reasons they never requested.
- HTTP `/api/breath` adds a separate optional `gate_rejections` receipt, not part of the content-free
  `timing` object or operational logs. ContextVar capture isolates concurrent requests and is reset.
  Nothing is written to NAS buckets; only the matching Twin consumer persists the bounded receipt.
- Schema: `{schema_version:1, items:[{id,score,final,reason,reason_status,reason_truncated}], omitted}`.
  Up to 64 rows, 240 Unicode characters per reason, 64 characters per id, and 32768 UTF-8 bytes
  including JSON escaping. Truncation/omission is visible; a truncated prefix is not a complete quote.

Verification uses synthetic responses through the actual parser, cache, selection and HTTP bridge:
25 reason tests plus 58 adjacent cache/noise/timing checks pass. No live provider calls were made.
The optional `TWIN_GATE_REASON_SOURCE_ROOT` test connects the actual provider HTTP result to the
Twin consumer and actual trace writer, asserting original reasons and 0600 file permissions.
This proves local wiring, not production loading or a measured reduction in false rejections.

Review correction (rollback: `68049c8bf247237efca5b96e2b82502f4a8928f7`):
the reviewer's two unchanged index-guard tests first reproduced `IndexError`, then passed.
Missing scores use the selector's existing zero fallback and an `unparsable` reason; missing
reason slots are also `unparsable`. Batch combination pads/trims reasons within each batch,
so one malformed vector cannot shift another candidate's reason. The capture call is isolated:
an unexpected diagnostic exception preserves selection and logs only its type, not private text.
Eight vector-length combinations compare selection to the original source; a fault-injection
case checks the exception boundary, and two batch cases check reason alignment. These are
local reproducible defects, not evidence that production previously encountered them.

Delivery requires both local repository patches and the assigned Claude review. No push, deployment,
restart or NAS modification is authorized by this implementation receipt.
