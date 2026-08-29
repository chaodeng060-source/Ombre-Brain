# Desire chord × E recall: zero-call shadow contract

This feature is deliberately not a new recall channel. Twin attaches one
`live_chord.v1` object to the existing `/api/breath` request. Ombre validates
that object against the same request's turn and agent, waits until all semantic,
state, session, anchor and DS filters have produced the retained candidate pool,
and then computes a private arm-C proposal. The `/api/breath` response carries
that text-free proposal back to Twin, but the served list remains arm B.

## Ordering boundary

The three arms use the exact same retained candidate objects:

- A: existing relevance and non-E tie score, recomputed over B's frozen pool;
- B: the current E-aware order that production already uses;
- C: B plus the chord proposal, recorded but never returned.

C can swap only adjacent candidates. Both candidates must already be
primary-authored E records by the chord's reviewed author identity, share an explicit
`event_id` or `episode_id`, be non-factual, and fall within
`near_tie_epsilon` under B. They must also share the relevance-band ID frozen
before any subtractive DS/session gate; bands are never reconstructed from a
filtered subset. A matching response tendency contributes a
non-negative affinity. C may make at most one adjacent swap per event lock and
no candidate may move more than one position. It cannot add a candidate, revive
a filtered result, cross an event lock, move a fact, or turn zero candidates
into a non-zero result.

Ombre's retained top-k is not treated as the visible prompt. Twin still applies
its existing scene/domain/noise filters and local reranker. It replays the same
post-filter blocks through the frozen A/B/C orderings, verifies that B exactly
matches the IDs actually committed to the prompt, and attaches one text-free
`e_chord_final_selection.v1` object to the existing `/api/recall-receipt`
request. Empty final injection is recorded too. An ID outside the frozen pool
is retained as an explicit diagnostic and makes the case unscorable; it is
never silently discarded. No extra external/provider API is introduced.
Non-empty recall reuses the existing background `/api/recall-receipt` call;
an empty final selection uses that same internal endpoint so the zero can be
audited even though there are no activation IDs.

`e_chord_shadow.enabled` defaults to false and only `mode: shadow` activates
receipt capture. Turning it off restores the exact old path without rewriting
any bucket or E annotation.

## Privacy and call budget

The live payload contains at most two closed-vocabulary facets. Claude's
transport identity remains `claude`, while its reviewed E-author identity is
the production value `哥哥`; the receiver validates this mapping so another
agent's E cannot cross the boundary. It never
contains the query, thought text, action hints, reasons, raw session ID, or
memory content. The raw session is reduced in Twin with a process-private keyed
digest. A separate SHA-256 over agent, conversation and the already-persisted
message ID binds the projection to the exact natural source turn without
exporting that ID or its text. Ombre binds the projection to the request-local
turn and agent. Window retries reuse the same immutable projection and carry
monotonic attempt indices. The final-selection receipt names the one attempt
Twin actually served; missing, duplicate or later attempts fail closed.
Evaluation sums every attempted Ombre proposal plus Twin's final local
simulation when computing added latency.

The private Ombre attempt ledger lives at
`<buckets_dir>/.axis/e-chord-shadow.jsonl`; Twin's authoritative post-filter
selection lives separately at
`<buckets_dir>/.axis/e-chord-final-selection.jsonl`. Both use directory mode
0700 and file/lock mode 0600. Identical final-selection retries are idempotent;
conflicting rows for one projection are rejected. Attempt-ledger fsync runs in
an observed background task, not on the recall response path; worker-side file
locking is blocking so concurrent turns serialize instead of dropping samples.
The ledgers store only candidate IDs, frozen relevance band IDs, A/B/C ordering,
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
time, stay within the declared visible cutoff, and identical shadow-arm inputs
must produce identical Twin outputs; otherwise the batch is rejected. Metrics
come only from Twin's final A/B/C selections, not
Ombre's pre-Twin top-k. It reports precision, noise, completeness, gold-zero
recall (`correct_zero_rate`) and predicted-zero precision, plus nearest-rank P95
of every attempt and final-selection overhead.

A mechanically good cohort needs at least 20 fully labelled natural turns, C
precision and both zero metrics no worse than B, strictly better completeness,
no higher noise, no hard violation, no outside-pool injection, zero
external-call delta and P95 within budget. Even then the tool returns
`candidate_for_named_review`, never `eligible_for_live`; exit code 3 requires a
named independent reviewer. `failed` and `inconclusive` also return non-zero, so
automation cannot confuse a candidate or unscored cohort with live approval.
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
E buckets contain zero `event_id`/`episode_id` locks, and the current
`experience` writer does not create either field. Therefore the safe C arm is a
no-op on today's asset set. This implementation does not weaken the lock to
thread, wording or emotion merely to manufacture movement. An audited upstream
event-link write must exist before a meaningful 20-turn cohort can be scored.

Rollback is one config change: keep or set `e_chord_shadow.enabled: false`.
Twin may also stop attaching the optional fields. Existing E ordering, factual
recall, embeddings, buckets and sidecar annotations require no migration or
cleanup; the append-only shadow ledger can remain ignored for audit.
