# NAS 8001 production contract

This document defines the durable source, job, backup, and rebuild contract for
the Ombre instance used by Twin.  It is intentionally fail-closed: a green Git
test run is not permission to replace a live container whose code differs from
the tracked source.

## Audited state on 2026-08-31

- Twin and the configured MCP tunnel use NAS `127.0.0.1:8001`, container
  `ombre-vps-mirror`.
- `memory.zhaodeng.xyz` still points at the separate 8000 container
  `ombre-brain`.  Changing that public route is outside this deployment step.
- 8001 persists data at `/vol1/ombre-migrate/data`, snapshots at
  `/vol1/ombre-migrate/snapshots`, and configuration at
  `/vol1/ombre-migrate/code/config.yaml`.  Python source under `/app` is not a
  bind mount.
- The running 8001 image is
  `sha256:2629696526f287a6c2268addb8e101c2ebbbfa3a2de2636ce2761befc9b45255`.
  It is untagged and has writable-layer changes in `server.py`,
  `bucket_manager.py`, `bm25_index.py`, and `memory_signal.py`.
- The tag `ombre-vps-mirror:latest` points at a different, older image.  The
  legacy `nas_mirror_rewire.sh` removes the live container and uses that tag;
  running it would discard the writable-layer fixes.
- The 04:30 LMC-5 job and 05:30 E-axis job currently default to the 8000
  container.  The 04:00 backup only archives `/vol1/ombre-data` (also 8000).
  The 8001 data directory is not covered by that backup.

Git commit `e17e52a3d0b6ea893c3bdcbf0452e344cb8675a8` permanently implements
the bucket-ID receipt for short and long grow requests, for both create and
merge.  It is an ancestor of `66891d994594607a87e623c59b6bb841f139ad95`.
That proves the source behavior, not that either running container was built
from it.

## Authority after cutover

The intended single authority is:

1. GitHub `main`, pinned to an exact commit, is the code authority.
2. `/vol1/ombre-migrate/code` is an extracted copy of that commit and contains
   `.deployed_commit` plus `deployment-manifest.json`.
3. `ombre-vps-mirror` is built from that exact directory, carries the OCI
   revision label, publishes only `127.0.0.1:8001`, and mounts
   `/vol1/ombre-migrate/data` at `/data`.
4. LMC-5, E-axis shadow, Twin/MCP writes, and the 8001 backup all use the same
   production environment file.
5. The public `memory.zhaodeng.xyz` upstream is switched to 8001 and the 8000
   `ombre-brain` container is stopped.  A running 8000 container is not treated
   as read-only: the application has no enforced read-only mode, so stopping it
   is the single-writer boundary.
6. `/vol1/ombre-migrate/deployment-anchor.env` records the container and image
   IDs produced by the last successful cutover.  Deploy and backup scripts load
   it after the base environment, so a rebuild cannot leave cron pinned to the
   retired container.
7. Deploy and backup share `/vol1/ombre-migrate/ombre-production.lock`.
   A deploy holds it for the whole preflight/build/backup/cutover transaction;
   cron backup or a second deploy therefore exits before touching Docker or
   source paths.

Copy `deploy/nas-production.env.example` outside the repository, fill the
secret-free path and image values, and keep authentication secrets only in the
existing protected `.env` file.

The scheduler must expose a system-cron drop-in directory that it actually
loads.  Add `OMBRE_JOB_CRON_FILE` (for example
`/etc/cron.d/ombre-vps-mirror`) and `OMBRE_JOB_CRON_USER=zhaodeng` to the
protected production environment.  The target directory must already exist,
must not be a symlink, and the installer account must be allowed to replace one
file in it.  The 2026-08-31 read-only audit confirmed this NAS runs
`/usr/sbin/cron -f`, loads `/etc/cron.d`, and keeps that directory root-owned;
therefore run the installer as root only in the authorized cutover window while
the drop-in continues to execute its three jobs as `zhaodeng`.  If this NAS cron implementation has no supported drop-in directory
or the operator is not authorized to write it, job installation stops here;
the script does not fall back to replacing the user's shared crontab.
Also set
`OMBRE_PUBLIC_HEALTH_URL=https://memory.zhaodeng.xyz/health`; installation
requires that endpoint to stay healthy after `ombre-brain` is stopped.

## Rebuild gate

Do not build or stop 8001 until a full top-level `/app/*.py` manifest has been
compared with the incoming Git archive and every difference has a reviewed
record.  This includes code already baked into the old image, not only paths
reported by `docker diff`.  The 2026-08-31 inventory includes at least:

- BM25 indexing and stop-token handling;
- max-win scoring and literal key forcing;
- Z current/historical weighting;
- vector escort, raw-head, and gate-empty behavior;
- `memory_signal.py`;
- grow bucket-ID receipts;
- runtime dependencies including `rank-bm25` and `psycopg`.

This list is not a substitute for that complete file manifest.  Any unexplained
live/source addition, removal, or hash difference blocks deployment.
`deploy/nas-live-reconciliation.tsv`
records the audited live hash and the reviewed Git-source hash separately, so
new Git changes do not have to equal the older live file byte-for-byte.  Its
initial `UNRECONCILED` entries deliberately keep the gate closed.
`scripts/deploy_nas_atomic.sh` checks those records before build or stop.  It
never uses `docker commit` or an unpinned `latest` tag.  The Docker publish
workflow also skips image publication while any `UNRECONCILED` source remains;
passing the contract tests alone cannot overwrite `latest` with a knowingly
undeployable image.

## Authorized cutover procedure

The following procedure changes NAS state and must only run in an approved
window.  The scripts are tracked here so the procedure is reviewable; this
document does not claim it has been executed.

1. Export the live `/app` manifest and reconcile every difference into a clean
   Git commit.  Run the full adjacent recall suite and the grow receipt suite.
2. Create a commit-exact `git archive`; record its SHA-256 and copy it to NAS.
   The deploy script verifies both the SHA-256 and Git's embedded archive
   commit ID against `--commit`.
3. Run the deploy script in preflight mode.  Confirm the current container ID,
   image ID, data bind, source manifest, commit, and health URL all match the
   production environment.
4. Run `scripts/backup_nas_data.sh`; retain its code/data archives and SQLite
   `quick_check` output.
5. Run `scripts/deploy_nas_atomic.sh`.  It stages the archive, builds a pinned
   image, stops only 8001, and renames the original container instead of
   deleting it.  The old container therefore retains every writable-layer
   change and is renamed back and restarted on failure.  On success it remains
   stopped under the reported rollback name until production acceptance is
   complete.  The script switches source atomically, starts the new 8001,
   checks health, writes the new deployment anchor, and proves the backup
   `--check` accepts that anchor.
6. Switch the configured public upstream for `memory.zhaodeng.xyz` from 8000 to
   `127.0.0.1:8001`, reload that proxy through its normal validated procedure,
   and confirm a cache-bypassed public health request reaches the new 8001
   deployment.  Then run `docker stop ombre-brain`; do not merely rename it or
   describe it as read-only.  `docker inspect -f '{{.State.Running}}'
   ombre-brain` must report `false`.  The public health request must succeed
   while 8000 remains stopped.  If either check fails, restart 8000 before
   reverting the public route so rollback does not create an outage.
7. As root, run `scripts/install_nas_jobs.sh`.  It first reruns the anchored backup
   `--check`, scans the shared user crontab read-only for old wrappers and
   direct `docker exec ... night_run_trigger.py`, `patrol_night.py`, or
   `e_axis_night.py` writers, then atomically replaces only the configured
   system-cron drop-in.  It never rewrites the shared user crontab.
8. Verify the scheduler loaded that exact drop-in, 8000 remains stopped, no
   8000 data timestamp advances, and the 8001 backup can be extracted and
   passes SQLite `quick_check`.

## Production acceptance still required

After rebuild, execute real, uniquely named writes against each authorized
endpoint and retain the exact command and response:

- short create returns the original `action -> name | domain` prefix plus ID;
- short merge returns `📎` plus the existing target ID;
- long create returns `📝` plus ID;
- long merge returns `📎` plus the existing target ID.

GET every returned bucket and verify one authorized experience anchor resolves
to the same `e_source_bucket_id`.  Record the loaded commit, image digest,
container ID, health result, and rollback path.  Until those writes and a real
rebuild are performed, the honest status is “source and deployment contract
prepared; production persistence not yet accepted.”

Only after this acceptance may the operator remove the stopped rollback
container and the retained previous source directory; neither cleanup is part
of the automated cutover.

The VPS disk copy is a separate divergent runtime: as of this audit it has the
long-path suffix only and no running local Ombre process.  It must be reconciled
by targeted hunks and tested separately; never replace either side's whole
`server.py` with the other side's file.
