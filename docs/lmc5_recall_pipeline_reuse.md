# LMC-5 recall pipeline reuse map

Upstream source is pinned to `wuxuyun0606-collab/lmc-5` commit
`53a4aaa944cdc64a1a56eaf62aee0a67d59a46f1`, file
`extras/pgvector_backend/recall_pipeline.py`. The vendored copy must remain
byte-identical to that file; the upstream MIT license is included beside it.

## Direct reuse

- `RecallHit`: the upstream channel-hit contract.
- `RecallPipeline._apply_score_fusion`: the upstream RRF implementation.
- `RecallPipeline._merge_dedup`: additive cross-channel evidence fusion.

These functions replace only Ombre's keyword/vector/entity RRF merge. The
existing retrieval results, weights, rank order and final safety gates remain
the inputs and outputs of the same stage.

## Thin adapter

- Ombre bucket ids are strings, while upstream `RecallHit.source_id` and the
  dedup key require integers. The adapter assigns deterministic call-local
  integer ids and maps the fused results back to the original bucket ids.
- Ombre retrieval is asynchronous. All searches finish before the synchronous
  upstream fusion code is called.
- Repeated ids inside one channel are removed before adaptation, matching the
  existing Ombre RRF contract and preventing one channel from voting twice.

## Intentionally not transplanted

- `RecallPipeline.recall`: its storage adapters are synchronous and target the
  reference PostgreSQL/pgvector layout. Calling it around Ombre would duplicate
  live retrieval and bypass Ombre's authority, Z/Y/E and provenance gates.
- Reference vector, FTS, raw-event and archive adapters: Ombre already has live
  file/SQLite/embedding implementations with production data contracts.
- Query expansion, graph expansion, emotion resonance, rerank, content/session
  dedup and injection formatting: these are already deployed in Ombre and are
  frozen by the task. They are not rewritten or replaced here.

## Acceptance boundary

The adapter must be exactly equivalent to the previous `rrf_fuse_channels`
result for empty, two-channel, entity-channel, duplicate-id and tied-score
inputs. Deployment remains a separate protected cutover after commit review.
