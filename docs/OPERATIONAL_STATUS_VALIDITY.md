# Operational status validity

This layer handles only mutable engineering status facts: deployment,
completion, and progress. It does not change Ombre's Markdown layout, migrate
the vault, or model every kind of fact.

## Upstream basis

The implementation follows Graphiti `401c59a` rather than a TTL rule:

- [`valid_at`, `invalid_at`, and `expired_at`](https://github.com/getzep/graphiti/blob/401c59a65bdeb22a44136901ff30231e6998a7fe/graphiti_core/edges.py#L263-L282)
  distinguish event validity from the time invalidation was recorded.
- [duplicate/contradiction classification](https://github.com/getzep/graphiti/blob/401c59a65bdeb22a44136901ff30231e6998a7fe/graphiti_core/prompts/dedupe_edges.py#L53-L97)
  separates an updated status from an unrelated event.
- [temporal conflict resolution](https://github.com/getzep/graphiti/blob/401c59a65bdeb22a44136901ff30231e6998a7fe/graphiti_core/utils/maintenance/edge_operations.py#L538-L573)
  closes the older interval; a backfilled older event cannot replace a newer
  current fact.

Graphiti retains both edges. Ombre likewise retains both Markdown buckets and
stores only lifecycle markers in
`.validity/operational_status.sqlite3`.

## Runtime behavior

- `grow` keeps its existing top-five recall and `new / merge / supersede`
  arbitration. A status `supersede` creates a new bucket instead of replacing
  the old text, then atomically marks old=`historical`, new=`current`.
- Current status questions such as `上线了吗`, `做完了吗`, and `进度怎么样`
  suppress explicitly historical buckets.
- A matching legacy status bucket with no audited marker remains visible but
  is labeled `[validity:unknown] [authority:not_current_status]`. It cannot be
  presented as confirmed current evidence.
- Historical questions retain historical buckets and label their validity
  interval.
- Any sidecar read/write failure is fail-open: Markdown remains intact and an
  unmarked result is not promoted to current.

`status_validity.enabled: false` restores the old read/write behavior. An old
binary also ignores the additive sidecar, so rollback never requires rewriting
memory files.
