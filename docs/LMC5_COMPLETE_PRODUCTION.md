# LMC-5 complete production contract

Date: 2026-08-04

This document replaces the old "code green equals done" wording for LMC-5.
Completion means the three live loops below are wired, deployed, and verified
with production evidence. A task card, a local test run, or an E0 shadow
ledger by itself is not completion.

## Loops

Write loop:

- X binds raw events to time, world, domain, provenance, and bounded candidate
  creation.
- Y writes only approved safe relation classes into graph recall. Audit or
  contradiction relations do not become default expansion evidence.
- Z keeps fact lifecycle authoritative: current, review, and superseded facts
  are separated before ranking or E shaping.

Night loop:

- X/M are triggered by the nightly cron with durable per-run ledgers.
- M is report-only unless an explicit apply mode is configured and approved.
- E0 runs as a separate shadow collection job over the formal curated cohort
  and candidate cohort.

Recall loop:

- Main factual recall remains relevance-first and Z-authoritative.
- E1 live projection may consume only current, digest-matching E0
  `curated_memory` success rows from approved rubric versions.
- E can only break ties inside existing relevance bands, add a bounded labelled
  "supporting experience" side channel for explicit emotional queries, and add
  a response-posture block.
- E never rewrites facts, changes Z status, creates buckets, updates relations,
  changes decay, or turns manual/candidate E0 rows into authority.

## E1 Safety Contract

Active E is reversible at config level:

```yaml
e_axis_recall:
  enabled: true
  mode: active
  activation_id: "lmc5-e1-20260804-v1"
  allowed_rubric_versions: ["lmc5-experience-20260731-v1"]
  min_confidence: 0.5
  tie_break_weight: 0.2
  side_channel_limit: 1
  side_channel_scan_limit: 128
  side_channel_min_resonance: 0.55
```

Setting `enabled: false` restores the legacy recall path without modifying any
bucket, fact, relation, or E0 ledger.

## Acceptance

Before calling LMC-5 complete, collect all of the following:

- one targeted LMC-5/E test run covering active E new behavior and old E0
  neutrality;
- one full regression run;
- isolated commit pushed to `chaodeng060-source/Ombre-Brain` main;
- protected NAS deployment with cold backup, SQLite `quick_check`, rollback
  hook, container health, and cron preservation;
- real Chinese production recall with E off/on evidence showing an emotional
  ranking, side-channel, or response-posture difference while factual authority
  still comes from the main recall and Z axis.
