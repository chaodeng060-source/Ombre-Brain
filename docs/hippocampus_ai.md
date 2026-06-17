# Hippocampus Implementation Notes For AI

This repository is a local adaptation of `https://github.com/P0luz/Ombre-Brain.git`. Keep the upstream mental model: the durable source of truth is a directory of Markdown memory buckets, and the server exposes MCP/HTTP tools around those files.

## Core Contract

Each memory is one Markdown file:

```markdown
---
id: 78c658e2b6e9
name: Example
tags: [tag]
domain: [工程]
valence: 0.5
arousal: 0.3
importance: 5
type: dynamic
created: 2026-06-10T00:00:00
last_active: 2026-06-10T00:00:00
activation_count: 1
pinned: true
resolved: false
digested: false
world: ""
chord_tag: ""
relations:
  - type: explains
    target: abc123
    note: optional
---
Human-readable memory body.
```

Frontmatter is the API contract. Body is the original readable content. Do not move important state into prose if a structured field already exists.

## Storage Layout

`bucket_manager.py` maps bucket type to folders under `config["buckets_dir"]`:

- `permanent/`: long-lived identity, rules, pinned buckets, and permanent records.
- `dynamic/`: ordinary memories that can decay, resolve, merge, or archive.
- `archive/`: old low-score memories moved by decay.
- `feel/`: first-person emotional sediment written by `hold(feel=True)`.
- `embeddings.db`: SQLite vector cache used by `embedding_engine.py`.
- `.session_surface/`: per-session seen IDs for de-duplicating surfaced material.

Files remain Obsidian-compatible. Human editing is allowed, but keep YAML valid.

## Main Modules

- `server.py`: MCP server and HTTP bridge. Registers `breath`, `hold`, `grow`, `trace`, `inspect`, `pulse`, `dream`, `briefing`, `backfill_relations`, world switching, and dashboard APIs.
- `bucket_manager.py`: bucket CRUD, safe path handling, Markdown frontmatter load/save, merge movement, fuzzy/topic search, relation edge storage.
- `dehydrator.py`: LLM analysis, tagging, digesting, briefing compression, relation inference.
- `decay_engine.py`: score calculation and archival using importance, activation count, time, arousal, resolved/digested state.
- `embedding_engine.py`: OpenAI-compatible embedding client, multi-chunk embeddings, SQLite storage, cosine search.
- `utils.py`: config loading, ID generation, world matching, relation type constants, protected domain guard.

## Write Path

Use `hold()` for one memory:

1. Resolve `world`: explicit argument first, otherwise global `current_world`.
2. Optional image is uploaded to R2 and prepended as Markdown image syntax.
3. If `feel=True`, create a `type=feel` bucket and optionally mark `source_bucket` as `digested`.
4. Otherwise call `dehydrator.analyze()` to infer `domain`, `tags`, `valence`, `arousal`, and suggested name.
5. If `pinned=True`, create directly as `permanent`, lock `importance=10`, skip merge.
6. Otherwise `_merge_or_create()` decides whether to append to an existing compatible bucket or create a new one.
7. New buckets may receive auto-inferred relation edges via `_auto_infer_edges()`.
8. Generate/update embeddings best-effort; embedding failure must not block memory writes.

Use `grow()` for diary-like bulk input. It digests a long text into multiple candidate memories, then sends each through the same merge/create path. Very short inputs bypass digest to save LLM calls.

Use `trace()` for edits. It can change metadata, content, `resolved`, `pinned`, `protected`, `digested`, `world`, `chord_tag`, and relation edges. `delete=True` removes a bucket unless protected.

## Read Path

Use `inspect(bucket_id)` when the exact bucket must be read verbatim. This bypasses summary/search and returns full content.

Use `breath()` for recall:

- Empty query means automatic surfacing.
- Non-empty query runs keyword/fuzzy scoring and vector recall, then fuses results with RRF where applicable.
- Filters include `domain`, `world`, `valence`, `arousal`, `since`, `until`, and relation expansion.
- Pinned/protected buckets are allowed to surface as core principles; resolved buckets are still searchable but rank lower.

Use `briefing()` at window start:

1. Load all active buckets.
2. Filter by domain and current world.
3. Always include pinned/protected core buckets.
4. Add top unresolved dynamic buckets by decay score.
5. Add recent-window material gated by `created`, not just `last_active`, so maintenance touches do not make old events look recent.
6. Add top `feel` buckets as emotional sediment.
7. Pull protected-domain buckets into a verbatim block before LLM compression.
8. Compress remaining material with `dehydrator.briefing()`.
9. Append `=== 锚索引 ===` with `bucket_id` labels so the AI can call `inspect()` instead of guessing.

The current local version also supports `format=json`, where `tier==0` buckets are separated into raw `slots[]` and the rest remains compressed.

## Scoring And Forgetting

`decay_engine.calculate_score()` returns:

- `999.0` for `pinned`, `protected`, and `type=permanent`.
- `50.0` for `type=feel`.
- Otherwise a weighted decay score based on `importance`, `activation_count`, days since `last_active`, and arousal.

`resolved=True` does not delete a memory. It applies a strong ranking penalty and lets the bucket settle unless explicitly queried. `digested=True` lowers it further after a feel bucket has absorbed it.

Protected resolve domains are defined in `utils.PROTECTED_RESOLVE_DOMAINS`. Buckets in persistent relationship/commitment/self-reflection domains must not be marked resolved by automation.

## Relations

Relations live in frontmatter as outgoing edges:

- `causes`
- `contributes`
- `improves`
- `explains`
- `updates`
- `kin`

Use `trace(add_relation="type:target_id:note")` or the HTTP bucket update relation bridge. Edges are deduplicated by `type + target`.

## Local Extensions Over Upstream

This local system keeps the upstream Markdown/MCP architecture but adds:

- NAS deployment and health-check workflow.
- HTTP dashboard bridges for buckets, search, hold, briefing, config, import, network graph.
- Briefing anchor index for reliable `inspect(bucket_id)` follow-up.
- Protected-domain verbatim briefing block for relationship red lines.
- `world` isolation so RP worlds do not leak into daily memory.
- `feel` buckets and body-state sidecar.
- Multi-vector chunk embeddings to avoid long-bucket dilution.
- Relation backfill and explicit relation graph.
- `chord_tag` emotional color index for cross-window tone tracking.

## Operating Rules For Future AI

Do not treat briefing as ground truth when exact wording matters; call `inspect(bucket_id)`.

Do not mark relationship, commitment, family, self-reflection, or feel buckets as resolved unless the user explicitly asks and the guard allows it.

Do not blindly commit generated benchmark/log notes with production code. Keep draft eval notes separate until the real endpoint exists.

When editing bucket files manually, preserve valid YAML frontmatter and keep `id` stable. The filename may move; the `id` is the durable key.
