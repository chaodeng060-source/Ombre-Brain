# Y-axis relation slot reservation

## Why this exists

The production Y relation walk already had typed edges, intent depth, world/Z
eligibility and strength gates.  It still almost never emitted evidence because
primary, X and Z selection consumed `max_results` before Y calculated its cap.
On the frozen 30 miss + 10 zero-control production batch:

- `relation_depth >= 1`: 40/40;
- token gate closed: 0/40;
- Y call gate open: 22/40;
- eligible relation misses blocked by `relation_neighbor_cap == 0`: 6 cases;
- after excluding the primary tail that would fund the slot, 5 stable cases
  still had a real propagating `explains` neighbor.

The exact per-query diagnosis is in
`evals/relation_readout_baseline_20260902.json`.  Exact queries, memory IDs and
bodies stay in the mode-0600 private receipt outside Git.

## Behavior

Set `OMBRE_RELATION_SLOT_RESERVATION=1` to enable the rollout.  The selector:

1. runs only after the existing candidate qualification and intent policy;
2. checks the retained primary seeds with the existing typed Y graph walk;
3. reserves one existing injection slot only when that local probe returns a
   real eligible neighbor;
4. excludes every original primary ID, so the dropped primary cannot simply be
   relabelled as relationship evidence;
5. restores the withheld primary in its original section if the neighbor later
   fails deduplication, token fitting or dehydration.

The probe reads the already-loaded bucket snapshot.  It does not call an LLM,
embedding endpoint or any provider, and it does not write memory or relation
data.  Storage authority, E semantics and the Z gate are unchanged.

## Rollback

Unset the variable or set it to `0`, `false`, `no` or `off`, then load that
environment in the service process.  The reservation branch is skipped; the
returned bucket IDs, order and text follow the prior selector.  Contract tests
cover unset versus explicit-off equality and failed-neighbor restoration.

## Frozen replay evidence

Run `tools/eval_relation_slot_reservation.py` against the private baseline
ledger and the current production bucket mirror.  The committed content-free
receipt is `evals/relation_slot_reservation_20260902.json`.

Current clean-room replay result:

- relation evidence: 1 baseline, 6 projected after reservation;
- five distinct concrete `explains` edges with both endpoint bodies present in
  the private receipt;
- zero-control selection changes: 0/10;
- return-count changes: 0/40;
- hit@1, hit@k, MRR, completeness and explicit-noise metrics unchanged;
- added external LLM/API/embedding/provider calls: 0;
- local preflight P95: 650.896 ms; projected batch P95 regression: 0%.

This receipt authorizes independent code review only.  It is not proof that a
container was built, deployed or loaded.  After GLM reviews the frozen commit,
Claude must run the same 40 real queries against production with the variable
off and on, verify byte equality for off, confirm P95 regression stays within
20%, and record the loaded image and rollback evidence.
