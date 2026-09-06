# Entity scoring exclusion: implementation and acceptance status

Task: `task_rrf-9-6-17-45`.

The implementation is committed as `02f1a91bc8b6fa9de21625a92bd4310b1ac6c20b`.
The rollback anchor before editing was `d29183cdab57f090b5cec1e6c890a240bb6c29bd`.
The switch remains off by default. No deployment or service reload was performed.

## Runtime contract

`OMBRE_ENTITY_SCORE_GUARD_ENABLED=1` excludes validated entity links from the
entity RRF channel, entity-only state seeds, and the entity Anchor scoring arm.
The same exclusion applies to fusion shadow and the Anchor probe. Entity alias
resolution and linked-bucket retrieval remain unchanged.

The weak-ticket issuer, ticket DS input, empty-slot filling, replacement, ticket
rendering and ticket-only DF helpers have been removed. A bucket independently
retrieved by keyword or vector retains the normal eligibility and scoring path.

With the switch off, runtime behavior is the `3e16a3f` baseline. Relative to that
commit, `server.py` adds only the switch helper and scoring-exclusion conditions;
`bm25_index.py` and `bucket_manager.py` are byte-identical to the baseline.
DS prompt, model, timeout, result budget, score thresholds and normal ranking
were not changed.

## Verification

- Three integration contracts passed: no ON ticket/weak DS input, no ON entity
  RRF/Anchor scoring, and exact OFF output/call shape.
- The full affected test selection passed: 179 tests covering entity recall,
  DS fallback, recall authority, lexical collision/Anchor limits, timeline,
  state-aware recall, read-only dehydration, E-axis and sensory integration.
- Independent code review found no P0/P1/P2 issue; `git diff --check` passed.
- A full-repository run stalled at the pre-existing
  `test_world.py::test_search_world_filter`; the same 30-second stall was
  reproduced on clean `d29183c`. This is not a full-repository green receipt.

## Real batch evidence

The accepted runner uses the same effective-query-first selection as the prior
four-pair runner, with the original captured query as fallback. All 22 inputs
come from the pinned real ledger; no query was invented. Derived caches and
receipts are isolated from a private copy of the local corpus.

| Batch | Completed | DS semantic attempts | DS outcomes | Behavior checks |
|---|---:|---:|---|---|
| Four OFF/ON pairs | 4/4 | 8/8 | 3 ok, 5 timeout/fallback | ON has two RRF channels, no ticket marker, all four required focus buckets retained |
| Full ON batch | 22/22 | 22/22 | 2 ok, 1 invalid/fallback, 19 timeout/fallback | No ticket marker or weak DS input; 12 changed rows, 26 additions, 16 removals relative to frozen OFF |

Both completed phases left the private corpus and source data fingerprints
unchanged. Two earlier incomplete attempts are preserved separately: the first
used the wrong query-field selection and consumed zero DS attempts; the second
consumed two DS attempts before stopping on provider timeouts. Total DS semantic
attempts across all attempts and both phases were 32, including the final ON
batch's exact 22. Counts refer to the semantic-select entry point, not HTTP
transport retries.

## Acceptance limitation

The behavior checks above are not a quality pass. Review found at least one
clearly relevant bucket missing from the current ON output relative to frozen
OFF, so the required zero-relevant-drop result has not been demonstrated.

The identified row resolves no entities. Its old OFF and old ON results were
identical, and the missing bucket remains in the current keyword channel but
disappears before the DS input. The observation therefore cannot be attributed
to this entity switch or to the current DS timeout. Live provider responses and
unpaired historical baselines must not be conflated with switch causality.

Relevance labels are agent assessments of the actual bucket content, not human
gold. Some corpus records were created after the historical query; the replay
does not establish what was available at the original event time.

The task remains blocked at quality acceptance. Further action requires review
of the recorded loss and provider degradation within an explicitly agreed
verification scope; the switch must not be enabled based on these results.
