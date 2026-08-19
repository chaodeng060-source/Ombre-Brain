# LMC-5 X timeline in Ombre

Ombre stores the upstream X field literally as `metadata.thread`. Values are
mutable narrative labels; an empty value normalizes to `other`. `other` is the
incubator, not a real line.

## Assignment boundary

A production bucket may leave `other` only when an explicit review manifest
maps its bucket ID to a narrative label. A typed `in_thread` component may
propagate one already reviewed label, but it cannot invent a label. Components
with no named anchor or more than one distinct named anchor fail closed.

The following are candidate evidence only and never write `thread`:

- proposer `thread_hint`;
- `source_session` or `source_event_ids`;
- episode membership;
- `kin` edges or free-text relation notes;
- one intense conversation or one isolated event.

Do not generate `event:<hash>`, `relation:<hash>`, or `episode:<id>` lines. A
new formal line should first meet the upstream review threshold: at least eight
memories, a span of at least fourteen days, and at least two recall hits. The
threshold makes a review candidate; it does not authorize an automatic write.

## Backfill and rollback

Use one reviewed manifest:

```json
{
  "schema": "ombre.timeline-review/v1",
  "reviewer": "claude",
  "reviewed_at": "2026-08-19T00:00:00+08:00",
  "assignments": [
    {"bucket_id": "example-id", "thread": "基础设施演进"}
  ]
}
```

Dry-run and apply use the same deterministic plan. Apply requires a snapshot
and holds the vault maintenance barrier from snapshot through the final
compare-and-set write:

```bash
python timeline_sweep_cli.py --dry-run \
  --reviewed-manifest reviewed.json --output timeline-dry-run.json

python timeline_sweep_cli.py --apply \
  --reviewed-manifest reviewed.json \
  --snapshot-root /path/outside/vault \
  --snapshot-id timeline-before-YYYYMMDD-HHMMSS \
  --output timeline-apply.json
```

Existing named buckets are preserved; this sweep does not silently rename
them. Missing legacy fields are written as `other`. The report fields mean:

- `assigned_count`: buckets promoted from missing/`other` to a named line;
- `new_line_count`: distinct reviewed labels not present before the sweep;
- `orphan_count`: eligible buckets still in `other`;
- `updated_count`: physical bucket writes, including missing to `other`.

Rollback restores the snapshot recorded in the apply report. Never copy an
older vault over an active writer without first entering the same maintenance
window.

## Night run and recall

Nightly `timeline_sweep` runs after the snapshot and this night's dispatch,
then before report-only M processing. Proposer hints remain candidates, so the
night step can backfill `other` and report counts without auto-naming a line.

`/api/breath` reserves one existing `max_results` slot only after proving that
a retained primary result has an eligible same-thread neighbor. X fills that
slot before Y relation expansion. Z, Y, E, and random side channels all remain
inside the same total result budget. A query whose displayed primary results
have no named neighbor keeps the pre-X result ordering and budget.
