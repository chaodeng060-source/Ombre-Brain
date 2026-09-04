# Desire chord × E recall: shadow proposal and opt-in live delivery

This feature is deliberately not a new recall channel. Twin attaches one
`live_chord.v1` object to the existing `/api/breath` request. Ombre validates
that object against the same request's turn and agent, freezes a pre-E cohort,
then waits until all existing E, state, session, anchor and DS filters have
produced the retained candidate pool and computes a private arm-C proposal. The `/api/breath` response carries
that text-free proposal back to Twin. By default the served list remains arm B.
When both Twin's explicit `e_chord_live_enabled=true` request bit and Ombre's
`OMBRE_E_CHORD_BYPASS=1` gate are open, Ombre may additionally return one
source-bound rendered bypass block in `e_chord_bypass_delivery.v1`. Natural
`raw` is never mutated, so Twin can discard a malformed delivery and use the
exact old result.

## Ordering boundary

The receipt distinguishes an honest baseline from an unscorable placeholder:

- A: relevance plus the non-E tie score, frozen before existing E ordering and
  before every later first-wins or subtractive gate;
- B: the current E-aware order that production already uses;
- C: B plus the chord proposal. It remains shadow-only in receipt v1-v3;
  validated live bypass delivery is recorded separately by final selection v4.

`pre_e_cohort_ids` records that original text-free cohort, while
`post_e_cohort_ids` freezes the order immediately after existing-E ranking and
before every downstream gate. A receipt may say
`a_cohort_status=pure_semantic` only when final B exactly matches that post-E
order, all three cohorts have the same IDs, and the existing DS path reports
`disabled` or `deterministic_noop`. A real
model-derived DS decision, a DS fallback, an unobserved decision source, or any
membership or order change from cap/dedupe/session/anchor/fact gates makes the whole turn
`unscorable_*`. In that case `arms.a` is canonically set to B, never presented
as a post-hoc "pure semantic" ablation. The existing DS prompt, candidate cap,
served B result and number of provider calls are unchanged; the shadow adds no
second decision call and never widens the existing prompt.

C can swap only adjacent candidates. Both candidates must already be
primary-authored E records by the chord's reviewed author identity, share an explicit
`event_id`, `episode_id`, or exact `e_source_bucket_id`, be non-factual, and fall within
`near_tie_epsilon` under B. They must also share the relevance-band ID frozen
before any subtractive gate; bands are never reconstructed from a filtered
subset. Each candidate must pass the recorded existing-E admissibility guard:
an authenticated E annotation is present, resonance is at least the configured
floor (never below 0.55), and non-neutral input/experience polarities do not
oppose each other. The promoted candidate's resonance may be at most 0.03 below
the demoted candidate's. The guard stores only booleans, integer milli-scores
and `-1/0/1` polarities; the validator recomputes its enum. A matching response tendency contributes a
non-negative affinity. C may make at most one adjacent swap per event lock and
no candidate may move more than one position. It cannot add a candidate, revive
a filtered result, cross an event lock, move a fact, or turn zero candidates
into a non-zero result.

Ombre's retained top-k is not treated as the visible prompt. Twin still applies
its existing scene/domain/noise filters, local reranker, dedupe and prompt
cutoff. Final C is derived only from the IDs actually committed as final B:
Twin applies the validated receipt pairs only when every pair survives and
remains adjacent in demoted/promoted order. If any pair is missing, separated,
or unsafe, the entire final C arm falls back to B with no applied swaps.
Cutoff-excluded IDs can never enter C. Consequently final B and C have exactly
the same length and candidate set, empty B implies empty C, unrelated IDs never
move, and facts never move. Twin records the applied pairs with the exact fields
`promoted_id`, `demoted_id`, `from_index`, `to_index`, and
`event_lock_digest`; indices are positions in final B immediately before that
swap and must satisfy `from_index=to_index+1`. It verifies that B exactly matches
the IDs actually committed to the prompt, and attaches one text-free
`e_chord_final_selection.v1` object to the existing `/api/recall-receipt`
request. Empty final injection is recorded too. An ID outside the frozen pool
is retained as an explicit diagnostic and makes the case unscorable; it is
never silently discarded. No extra external/provider API is introduced.
`final_input_cohort_ids` freezes the primary B order after Twin's local filters
but before its final reranker. Only an exact ordered match with the Ombre pool
may use `final_input_cohort_status=pure_same_cohort`. Any membership or order
drift is `unscorable_final_cohort_drift`, forces final A=B, and is excluded
from quality scoring while its full latency remains in P95.
Non-empty recall reuses the existing background `/api/recall-receipt` call;
an empty final selection uses that same internal endpoint so the zero can be
audited even though there are no activation IDs.

`e_chord_shadow.enabled` defaults to false and only `mode: shadow` activates
receipt capture. Turning it off restores the exact old path without rewriting
any bucket or E annotation.

## Live bypass delivery

The shadow receipt remains text-free and continues to declare
`shadow_only=true` / `affects_ranking=false`: it is proposal evidence, not the
served payload. The sibling delivery has an independent schema and binds
`projection_digest`, `source_turn_digest`, `attempt_index`, candidate ID and
`e_source_bucket_id` to that validated v3 proposal. At most the first declared
adjacent bypass winner is delivered, preventing multiple independent swaps
from becoming a global leap.

Before building the block, Ombre reruns the established world, time, session,
Z/fact, current-annotation, semantic-resonance and Russell-resonance gates.
Rendering uses a valid frontmatter dehydration cache when present; otherwise it
uses the established deterministic 300-character passthrough shape. It never
starts a summarizer/provider, query, embedding, write or background backfill.
Twin validates the one-ID block, then sends it through its existing scene,
telemetry, association, mood, domain, dedupe, rerank and prompt cutoff. Only a
block that survives into the actual model prompt is recorded as
`e_chord_final_selection.v4` with `delivered_bypass_ids`, `served_arm=c` and
`live_applied=true`. Twin independently replays the same local rerank after
removing the delivered block to bind the natural OFF arm. If a full top-k
causes replacement, v4 names the exact `displaced_natural_ids`; therefore
`C-B` is exactly the one delivery and `B-C` is exactly zero or one declared
natural displacement.

Either live gate closed means no delivery field at all. Any proposal, delivery,
filter, rerank or receipt error leaves natural `raw` available and does not
block the assistant response.

## Privacy and call budget

The live payload contains at most two closed-vocabulary facets. Claude's
transport identity remains `claude`, while its reviewed E-author identity is
the production value `哥哥`; the receiver validates this mapping so another
agent's E cannot cross the boundary. It never
contains the query, thought text, action hints, reasons, raw session ID, or
memory content. Twin reduces the raw session to a domain-separated SHA-256
digest. This digest is a deterministic request-scope equality token, not keyed
authentication or a secrecy boundary: Ombre recomputes it from the actual
request `session_id` and rejects a mismatch, while the raw session itself is
not copied into the projection. A separate SHA-256 over agent, conversation
and the already-persisted message ID binds the projection to the exact natural
source turn without exporting that ID or its text. Ombre binds the projection
to the request-local turn and agent. Window retries reuse the same immutable
projection and carry monotonic attempt indices. The final-selection receipt
names the one attempt Twin actually served; missing, duplicate or later
attempts fail closed.
Evaluation sums every attempted Ombre proposal plus Twin's final local
simulation when computing added latency, including turns whose quality arms
are excluded as unscorable.

The private Ombre attempt ledger lives at
`<buckets_dir>/.axis/e-chord-shadow.jsonl`; Twin's authoritative post-filter
selection lives separately at
`<buckets_dir>/.axis/e-chord-final-selection.jsonl`. Both use directory mode
0700 and file/lock mode 0600. Identical final-selection retries are idempotent;
conflicting rows for one projection are rejected. Attempt-ledger fsync runs in
an observed background task, not on the recall response path; worker-side file
locking is blocking so concurrent turns serialize instead of dropping samples.
The ledgers store only candidate IDs, frozen relevance band IDs, A/B/C ordering,
cohort/DS decision status, applied swap metadata, existing-E guard integers,
opaque source-turn, projection and event-lock digests, bounded diagnostics and
measured request-path shadow overhead. They store no
query, memory body, drive key, motivation, session scope or turn ID. The code
does not call retrieval, query expansion, embedding, a model, or any other
external endpoint; `external_api_delta` is fixed and validated as zero.

## Offline evaluation

Human gold is a separate JSONL with no raw phrase. Each row must declare
`source_kind=natural_conversation`, `agent_id`, the receipt's unique opaque
`source_turn_digest`, a query-only `natural_turn_digest`, and
`annotation_method=human`. Its `source_evidence` binds those digests to an exact
user-message JSONL path, physical line, line SHA-256 and query SHA-256 under a
read-only evidence root. The verifier recomputes the source-turn digest from
that line's `convId` and `id`, so a real but unrelated message cannot be paired
with a favorable receipt. It is joined to the selected attempt and final
selection by `projection_digest` and must completely partition the frozen pool into
`acceptable_ids` and `noise_ids`; non-zero cases also name `expected_ids`,
while zero controls set `expected_zero=true`.

Run:

```bash
python3 tools/eval_e_chord_shadow.py \
  --receipts /private/path/e-chord-shadow.jsonl \
  --selections /private/path/e-chord-final-selection.jsonl \
  --gold /private/path/e-chord-human-gold.jsonl \
  --evidence-root /private/read-only/evidence \
  --min-cases 20 \
  --p95-budget-ms 5
```

The evaluator mechanically recomputes candidate-pool equality and requires the
receipt, final selection and gold digest sets to match exactly. It rejects
partial or selective gold, ambiguous retries, pool drift, mutated source lines,
non-finite values, duplicate case/turn/digest keys and any non-zero
external-call delta. A final selection must follow its selected attempt in
time, stay within the declared visible cutoff, reconstruct C exactly from final
B plus receipt-bound safe swaps, and preserve the final B candidate set.
Factual, cross-event, cross-author, non-admissible or fabricated applied swaps
are rejected. Unscorable A cohorts are excluded turn-by-turn and reported by
reason count. Metrics
come only from Twin's final A/B/C selections, not
Ombre's pre-Twin top-k. It reports precision, noise, completeness, gold-zero
recall (`correct_zero_rate`) and predicted-zero precision, plus nearest-rank P95
of every attempt and final-selection overhead.

A mechanically good cohort would need at least 20 fully labelled natural turns,
C precision and both zero metrics no worse than B, strictly better completeness,
no higher noise, no hard violation, no outside-pool injection, zero
external-call delta and P95 within budget. The v1-v3 shadow contract
deliberately requires final B and C to have the same set, so set-based
precision, noise and completeness are identical and the strict
completeness-improvement gate is unreachable. The current shadow evaluator
therefore remains `inconclusive` and does not score v4 membership delivery.
Live acceptance instead requires paired OFF/ON production queries,
receipt-backed reversal cases and the separate P95 gate.
All statuses return non-zero, so automation cannot confuse an unscored cohort
with live approval.
`annotated_by` is auditable attribution, not a cryptographic human signature;
the required named Claude/human final review remains an independent release
approval and cannot be replaced by the evaluator itself.

## First baseline and rollback

The checked-in `evals/e_chord_shadow/baseline_20260829.json` is intentionally
`inconclusive`, not a synthetic score. The existing 21 human-gold cases have no
expected-ID overlap with the 51 authored E buckets (哥哥 49, claude 1, xiaojuan
1; only the 49 哥哥 rows match Claude's current author boundary), and historical
traces do not preserve a frozen pre-E arm or chord proposal. The feature must
therefore stay disabled until a future natural cohort reaches the gate.

There is a second explicit evidence boundary: the current 51 primary-authored
E buckets contain zero `event_id`/`episode_id` locks. The current `experience`
writer can preserve an exact source-memory link as `e_source_bucket_id`, but the
observed source links are singletons, so no two candidates share any accepted
lock. Therefore the safe C arm is still a no-op on today's asset set. The three
lock namespaces are kept distinct: a matching raw value under different keys
does not match. This implementation does not weaken the lock to thread, wording
or emotion merely to manufacture movement. At least two audited E writes must
share the same explicit source/event link before a meaningful cohort can move.

Rollback is immediate from either side: set `TWIN_E_CHORD_RECALL_ENABLED=0` or
`OMBRE_E_CHORD_BYPASS=0`; disabling `e_chord_shadow.enabled` also removes all
proposal work. Existing E ordering, factual
recall, embeddings, buckets and sidecar annotations require no migration or
cleanup; the append-only shadow ledger can remain ignored for audit.
