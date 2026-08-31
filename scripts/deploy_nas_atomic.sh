#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "usage: $0 --env-file PATH --archive PATH --archive-sha256 HEX --commit HEX [--preflight]" >&2
  exit 2
}

ENV_FILE=""
ARCHIVE=""
ARCHIVE_SHA256=""
COMMIT=""
PREFLIGHT_ONLY=0
while (($#)); do
  case "$1" in
    --env-file) [[ $# -ge 2 ]] || usage; ENV_FILE="$2"; shift 2 ;;
    --archive) [[ $# -ge 2 ]] || usage; ARCHIVE="$2"; shift 2 ;;
    --archive-sha256) [[ $# -ge 2 ]] || usage; ARCHIVE_SHA256="$2"; shift 2 ;;
    --commit) [[ $# -ge 2 ]] || usage; COMMIT="$2"; shift 2 ;;
    --preflight) PREFLIGHT_ONLY=1; shift ;;
    *) usage ;;
  esac
done

[[ -r "$ENV_FILE" && -r "$ARCHIVE" ]] || usage
[[ "$ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || usage

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

# The base environment carries the audited first-cutover anchors.  After a
# successful cutover the deployer atomically writes the current container and
# image IDs to a separate, secret-free anchor file.  Loading it last keeps
# later backups and deployments tied to the running instance instead of the
# pre-cutover IDs.
if [[ -n "${OMBRE_DEPLOYMENT_ANCHOR_FILE:-}" && -e "$OMBRE_DEPLOYMENT_ANCHOR_FILE" ]]; then
  [[ -r "$OMBRE_DEPLOYMENT_ANCHOR_FILE" && ! -L "$OMBRE_DEPLOYMENT_ANCHOR_FILE" ]] || {
    echo "deployment anchor file is not a readable regular path" >&2
    exit 2
  }
  set -a
  # shellcheck disable=SC1090
  . "$OMBRE_DEPLOYMENT_ANCHOR_FILE"
  set +a
fi

: "${OMBRE_CONTAINER_NAME:?missing OMBRE_CONTAINER_NAME}"
: "${OMBRE_SERVICE:?missing OMBRE_SERVICE}"
: "${OMBRE_ACTIVE_DIR:?missing OMBRE_ACTIVE_DIR}"
: "${OMBRE_DATA_DIR:?missing OMBRE_DATA_DIR}"
: "${OMBRE_SNAPSHOT_DIR:?missing OMBRE_SNAPSHOT_DIR}"
: "${OMBRE_CONFIG_FILE:?missing OMBRE_CONFIG_FILE}"
: "${OMBRE_SECRET_ENV_FILE:?missing OMBRE_SECRET_ENV_FILE}"
: "${OMBRE_HEALTH_URL:?missing OMBRE_HEALTH_URL}"
: "${OMBRE_IMAGE_REPOSITORY:?missing OMBRE_IMAGE_REPOSITORY}"
: "${OMBRE_DEPLOYMENT_ANCHOR_FILE:?missing OMBRE_DEPLOYMENT_ANCHOR_FILE}"
: "${OMBRE_MUTATION_LOCK_FILE:?missing OMBRE_MUTATION_LOCK_FILE}"
: "${OMBRE_EXPECTED_CONTAINER_ID:?missing OMBRE_EXPECTED_CONTAINER_ID}"
: "${OMBRE_EXPECTED_IMAGE_ID:?missing OMBRE_EXPECTED_IMAGE_ID}"

[[ "$OMBRE_CONTAINER_NAME" == "ombre-vps-mirror" ]] || {
  echo "refusing non-production container: $OMBRE_CONTAINER_NAME" >&2
  exit 2
}
[[ "${OMBRE_HOST_PORT:-}" == "8001" ]] || {
  echo "refusing non-production host port: ${OMBRE_HOST_PORT:-unset}" >&2
  exit 2
}
[[ "$OMBRE_HEALTH_URL" == "http://127.0.0.1:8001/health" ]] || {
  echo "refusing non-production health URL: $OMBRE_HEALTH_URL" >&2
  exit 2
}
if [[ -n "${OMBRE_CONTRACT_TEST_ROOT:-}" ]]; then
  test_root="$(realpath "$OMBRE_CONTRACT_TEST_ROOT")"
  [[ "$test_root" == /tmp/* ]]
  [[ "$(realpath "$OMBRE_ACTIVE_DIR")" == "$test_root/active" ]]
  [[ "$(realpath "$OMBRE_DATA_DIR")" == "$test_root/data" ]]
  [[ "$(realpath "$OMBRE_SNAPSHOT_DIR")" == "$test_root/snapshots" ]]
  [[ "$(realpath "$(dirname "$OMBRE_DEPLOYMENT_ANCHOR_FILE")")" == "$test_root" ]]
  [[ "$(basename "$OMBRE_DEPLOYMENT_ANCHOR_FILE")" == "deployment-anchor.env" ]]
  [[ "$(realpath "$(dirname "$OMBRE_MUTATION_LOCK_FILE")")" == "$test_root" ]]
  [[ "$(basename "$OMBRE_MUTATION_LOCK_FILE")" == "mutation.lock" ]]
else
  [[ "$(realpath "$OMBRE_ACTIVE_DIR")" == "/vol1/ombre-migrate/code" ]]
  [[ "$(realpath "$OMBRE_DATA_DIR")" == "/vol1/ombre-migrate/data" ]]
  [[ "$(realpath "$OMBRE_SNAPSHOT_DIR")" == "/vol1/ombre-migrate/snapshots" ]]
  [[ "$(realpath "$OMBRE_CONFIG_FILE")" == "/vol1/ombre-migrate/code/config.yaml" ]]
  [[ "$(realpath "$(dirname "$OMBRE_DEPLOYMENT_ANCHOR_FILE")")" == "/vol1/ombre-migrate" ]]
  [[ "$(basename "$OMBRE_DEPLOYMENT_ANCHOR_FILE")" == "deployment-anchor.env" ]]
  [[ "$OMBRE_MUTATION_LOCK_FILE" == "/vol1/ombre-migrate/ombre-production.lock" ]]
fi
[[ ! -L "$OMBRE_DEPLOYMENT_ANCHOR_FILE" ]]

DOCKER_BIN="${DOCKER_BIN:-docker}"
CURL_BIN="${CURL_BIN:-curl}"
TAR_BIN="${TAR_BIN:-tar}"
SHA256_BIN="${SHA256_BIN:-sha256sum}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GIT_BIN="${GIT_BIN:-git}"
FLOCK_BIN="${FLOCK_BIN:-flock}"
BACKUP_SCRIPT="$(cd "$(dirname "$0")" && pwd)/backup_nas_data.sh"

exec 8>"$OMBRE_MUTATION_LOCK_FILE"
"$FLOCK_BIN" -n 8 || {
  echo "production mutation lock is held" >&2
  exit 1
}
export OMBRE_MUTATION_LOCK_HELD_FD=8

actual_archive_sha="$($SHA256_BIN "$ARCHIVE" | awk '{print $1}')"
[[ "$actual_archive_sha" == "$ARCHIVE_SHA256" ]] || {
  echo "archive checksum mismatch" >&2
  exit 1
}
archive_commit="$($GIT_BIN get-tar-commit-id <"$ARCHIVE")"
[[ "$archive_commit" == "$COMMIT" ]] || {
  echo "archive commit mismatch: archive=$archive_commit declared=$COMMIT" >&2
  exit 1
}

active_parent="$(dirname "$OMBRE_ACTIVE_DIR")"
stage="$active_parent/stage-$COMMIT"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
compose_project="ombre-vps-mirror-${COMMIT:0:12}-$(date -u +%Y%m%d%H%M%S)"
previous="$active_parent/previous-$stamp"
failed="$active_parent/failed-$stamp-$COMMIT"
[[ ! -e "$stage" && ! -e "$previous" && ! -e "$failed" ]] || {
  echo "stage/rollback path already exists" >&2
  exit 1
}

umask 077
mkdir -m 700 "$stage"
stage_exists=1
cleanup_stage() {
  local rc=$?
  trap - EXIT
  if ((stage_exists)) && [[ -d "$stage" ]]; then
    find "$stage" -depth -delete
  fi
  exit "$rc"
}
trap cleanup_stage EXIT

"$TAR_BIN" -C "$stage" -xf "$ARCHIVE"
[[ -f "$stage/server.py" && -f "$stage/Dockerfile" && -f "$stage/docker-compose.yml" ]] || {
  echo "archive is not an Ombre source tree" >&2
  exit 1
}
cp -p "$OMBRE_CONFIG_FILE" "$stage/config.yaml"
cp -p "$OMBRE_SECRET_ENV_FILE" "$stage/.env"
printf '%s\n' "$COMMIT" >"$stage/.deployed_commit"

# The current live container is the migration baseline.  Every direct Python
# writable-layer change must have an explicit, tracked live-SHA -> source-SHA
# reconciliation record.  Incoming source is allowed to contain newer reviewed
# changes, so equality between the whole live and source files is not required.
container_id="$($DOCKER_BIN inspect --format '{{.Id}}' "$OMBRE_CONTAINER_NAME")"
old_image="$($DOCKER_BIN inspect --format '{{.Image}}' "$OMBRE_CONTAINER_NAME")"
[[ "$container_id" == "$OMBRE_EXPECTED_CONTAINER_ID" ]] || {
  echo "container anchor changed: $container_id" >&2
  exit 1
}
[[ "$old_image" == "$OMBRE_EXPECTED_IMAGE_ID" ]] || {
  echo "image anchor changed: $old_image" >&2
  exit 1
}

mount_source() {
  "$DOCKER_BIN" inspect --format "{{range .Mounts}}{{if eq .Destination \"$1\"}}{{.Source}}{{end}}{{end}}" "$OMBRE_CONTAINER_NAME"
}
mounted_data="$(mount_source /data)"
mounted_snapshots="$(mount_source /snapshots)"
mounted_config="$(mount_source /app/config.yaml)"
[[ "$(realpath "$mounted_data")" == "$(realpath "$OMBRE_DATA_DIR")" ]] || {
  echo "container /data does not match the production manifest" >&2
  exit 1
}
[[ "$(realpath "$mounted_snapshots")" == "$(realpath "$OMBRE_SNAPSHOT_DIR")" ]] || {
  echo "container /snapshots does not match the production manifest" >&2
  exit 1
}
[[ "$(realpath "$mounted_config")" == "$(realpath "$OMBRE_CONFIG_FILE")" ]] || {
  echo "container config does not match the production manifest" >&2
  exit 1
}
mounted_buckets="$($DOCKER_BIN inspect --format '{{range .Mounts}}{{if eq .Destination "/app/buckets"}}{{.Name}}{{end}}{{end}}' "$OMBRE_CONTAINER_NAME")"
[[ "$mounted_buckets" == "${OMBRE_BUCKETS_VOLUME:?missing OMBRE_BUCKETS_VOLUME}" ]] || {
  echo "container buckets volume does not match the production manifest" >&2
  exit 1
}
mapfile -t networks < <(
  $DOCKER_BIN inspect --format '{{range $name, $network := .NetworkSettings.Networks}}{{$name}}{{println}}{{end}}' "$OMBRE_CONTAINER_NAME" |
    awk 'NF'
)
[[ "${#networks[@]}" == 1 && "${networks[0]}" == "${OMBRE_NETWORK_MODE:?missing OMBRE_NETWORK_MODE}" ]] || {
  echo "container network does not match the production manifest" >&2
  exit 1
}
port_binding="$($DOCKER_BIN port "$OMBRE_CONTAINER_NAME" 8000/tcp)"
[[ "$port_binding" == "127.0.0.1:8001" ]] || {
  echo "container port does not match 127.0.0.1:8001: $port_binding" >&2
  exit 1
}

diff_file="$stage/.live-container.diff"
"$DOCKER_BIN" diff "$OMBRE_CONTAINER_NAME" >"$diff_file"
if awk '
  $2 !~ /^\/app(\/|$)/ { next }
  $2 == "/app" { next }
  $1 !~ /^[ACD]$/ { next }
  $2 ~ /^\/app\/(__pycache__|buckets)(\/|$)/ { next }
  $2 ~ /\/__pycache__(\/|$)/ { next }
  $2 ~ /\.pyc$/ { next }
  $2 ~ /^\/app\/[^/]+\.(bak-|disabled-)/ { next }
  $2 == "/app/vendor" || $2 == "/app/vendor/lmc5_pgvector" { next }
  $2 ~ /^\/app\/[^/]+\.py$/ { next }
  { print; bad=1 }
  END { exit bad }
' "$diff_file" >"$stage/.unsupported-live-drift"; then
  :
else
  echo "unsupported live /app drift remains:" >&2
  cat "$stage/.unsupported-live-drift" >&2
  exit 1
fi

reconciliation="$stage/deploy/nas-live-reconciliation.tsv"
[[ -f "$reconciliation" ]] || {
  echo "missing tracked live reconciliation manifest" >&2
  exit 1
}
if awk -F '\t' '
  NF && $1 !~ /^#/ && $3 == "UNRECONCILED" { unresolved=1 }
  END { exit unresolved ? 0 : 1 }
' "$reconciliation"; then
  echo "unresolved reconciliation sentinel remains" >&2
  exit 1
fi
live_python_manifest="$stage/.live-python.tsv"
source_python_manifest="$stage/.source-python.tsv"
"$DOCKER_BIN" exec "$OMBRE_CONTAINER_NAME" sh -c \
  'for path in /app/*.py; do [ -f "$path" ] || continue; sha256sum "$path"; done' |
  awk '{path=$2; sub("^/app/", "", path); print path "\t" $1}' |
  sort -u >"$live_python_manifest"
find "$stage" -maxdepth 1 -type f -name '*.py' -exec "$SHA256_BIN" {} + |
  awk -v root="$stage/" '{path=$2; sub("^" root, "", path); print path "\t" $1}' |
  sort -u >"$source_python_manifest"

mapfile -t python_paths < <(
  awk -F '\t' '{print $1}' "$live_python_manifest" "$source_python_manifest" | sort -u
)
reconciled_count=0
for relative in "${python_paths[@]}"; do
  [[ "$relative" =~ ^[A-Za-z0-9_.-]+\.py$ ]] || {
    echo "unsafe top-level Python path: $relative" >&2
    exit 1
  }
  live_sha="$(awk -F '\t' -v path="$relative" '$1 == path {print $2}' "$live_python_manifest")"
  source_sha="$(awk -F '\t' -v path="$relative" '$1 == path {print $2}' "$source_python_manifest")"
  live_sha="${live_sha:-ABSENT}"
  source_sha="${source_sha:-ABSENT}"
  [[ "$live_sha" != "$source_sha" ]] || continue

  mapfile -t records < <(awk -F '\t' -v path="$relative" '$1 == path {print $2 " " $3}' "$reconciliation")
  [[ "${#records[@]}" == 1 ]] || {
    echo "missing or duplicate reconciliation record: $relative" >&2
    exit 1
  }
  read -r recorded_live recorded_source <<<"${records[0]}"
  [[ ( "$recorded_live" == "ABSENT" || "$recorded_live" =~ ^[0-9a-f]{64}$ ) &&
     ( "$recorded_source" == "ABSENT" || "$recorded_source" =~ ^[0-9a-f]{64}$ ) ]] || {
    echo "unreconciled live Python drift: $relative" >&2
    exit 1
  }
  [[ "$live_sha" == "$recorded_live" ]] || {
    echo "live reconciliation anchor changed: $relative" >&2
    exit 1
  }
  [[ "$source_sha" == "$recorded_source" ]] || {
    echo "source reconciliation anchor changed: $relative" >&2
    exit 1
  }
  reconciled_count=$((reconciled_count + 1))
done

grep -Eiq '^rank-bm25([<=> ]|$)' "$stage/requirements.txt" || {
  echo "rank-bm25 runtime dependency is not tracked" >&2
  exit 1
}
grep -Eiq '^psycopg([<=>\[]|$)' "$stage/requirements.txt" || {
  echo "psycopg runtime dependency is not tracked" >&2
  exit 1
}

[[ "$(grep -Fc '_bid_suffix' "$stage/server.py")" -ge 2 ]] || {
  echo "grow bucket-ID receipt anchors are missing" >&2
  exit 1
}

rm -f -- "$diff_file" "$stage/.unsupported-live-drift" \
  "$live_python_manifest" "$source_python_manifest"

if ((PREFLIGHT_ONLY)); then
  printf 'PREFLIGHT_OK commit=%s container=%s image=%s changed_py=%s\n' \
    "$COMMIT" "$container_id" "$old_image" "$reconciled_count"
  exit 0
fi

export OMBRE_DEPLOY_COMMIT="$COMMIT"
export OMBRE_IMAGE_TAG="$COMMIT"
export OMBRE_CONFIG_FILE="$stage/config.yaml"

"$DOCKER_BIN" compose --project-directory "$stage" -p "$compose_project" \
  -f "$stage/docker-compose.yml" config >/dev/null
"$DOCKER_BIN" compose --project-directory "$stage" -p "$compose_project" \
  -f "$stage/docker-compose.yml" build "$OMBRE_SERVICE"
new_image="$($DOCKER_BIN image inspect "$OMBRE_IMAGE_REPOSITORY:$COMMIT" --format '{{.Id}}')"
new_revision="$($DOCKER_BIN image inspect "$new_image" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
[[ "$new_revision" == "$COMMIT" ]] || {
  echo "built image revision label mismatch: $new_revision" >&2
  exit 1
}

"$BACKUP_SCRIPT" --env-file "$ENV_FILE"

rollback_container="${OMBRE_CONTAINER_NAME}-rollback-$stamp"
if "$DOCKER_BIN" container inspect "$rollback_container" >/dev/null 2>&1; then
  echo "rollback container name already exists: $rollback_container" >&2
  exit 1
fi
anchor_had_previous=0
anchor_backup="$active_parent/.deployment-anchor-before-$stamp"
anchor_tmp="${OMBRE_DEPLOYMENT_ANCHOR_FILE}.tmp-$stamp"
if [[ -e "$OMBRE_DEPLOYMENT_ANCHOR_FILE" ]]; then
  cp -p "$OMBRE_DEPLOYMENT_ANCHOR_FILE" "$anchor_backup"
  anchor_had_previous=1
fi

wait_health() {
  local i
  for i in $(seq 1 30); do
    if "$CURL_BIN" -fsS --max-time 3 "$OMBRE_HEALTH_URL" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

rollback() {
  local rc=$?
  local rollback_id="" main_id=""
  trap - EXIT
  trap '' INT TERM HUP
  # Recovery is best-effort across independent surfaces.  A failed source or
  # anchor mv must never prevent the preserved container from being renamed
  # back and started.
  set +e
  echo "deployment failed; restoring the preserved 8001 container and source" >&2
  rollback_id="$($DOCKER_BIN inspect --format '{{.Id}}' "$rollback_container" 2>/dev/null || true)"
  main_id="$($DOCKER_BIN inspect --format '{{.Id}}' "$OMBRE_CONTAINER_NAME" 2>/dev/null || true)"

  # Filesystem existence is the recovery journal.  This remains correct even
  # when a signal arrives after mv(2) succeeds but before the next shell line.
  if [[ -d "$previous" ]]; then
    if [[ -d "$OMBRE_ACTIVE_DIR" ]]; then
      mv "$OMBRE_ACTIVE_DIR" "$failed" || true
    fi
    mv "$previous" "$OMBRE_ACTIVE_DIR"
  fi

  if ((anchor_had_previous)) && [[ -f "$anchor_backup" ]]; then
    mv "$anchor_backup" "$OMBRE_DEPLOYMENT_ANCHOR_FILE"
  else
    rm -f -- "$OMBRE_DEPLOYMENT_ANCHOR_FILE"
  fi
  rm -f -- "$anchor_tmp"

  # Container IDs, not post-command flags, decide recovery.  Therefore stop or
  # rename can be interrupted at any instruction boundary without losing the
  # original writable layer.
  if [[ "$rollback_id" == "$container_id" ]]; then
    if [[ -n "$main_id" && "$main_id" != "$container_id" ]]; then
      "$DOCKER_BIN" rm -f "$OMBRE_CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    "$DOCKER_BIN" rename "$rollback_container" "$OMBRE_CONTAINER_NAME" || true
    "$DOCKER_BIN" start "$OMBRE_CONTAINER_NAME" >/dev/null || true
    wait_health || true
  elif [[ "$main_id" == "$container_id" ]]; then
    "$DOCKER_BIN" start "$OMBRE_CONTAINER_NAME" >/dev/null || true
    wait_health || true
  else
    echo "original rollback container is missing: $container_id" >&2
  fi
  if [[ -d "$stage" ]]; then
    find "$stage" -depth -delete
  fi
  exit "$rc"
}
trap rollback EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

"$DOCKER_BIN" stop "$OMBRE_CONTAINER_NAME" >/dev/null
"$DOCKER_BIN" rename "$OMBRE_CONTAINER_NAME" "$rollback_container"
mv "$OMBRE_ACTIVE_DIR" "$previous"
mv "$stage" "$OMBRE_ACTIVE_DIR"
stage_exists=0

export OMBRE_CONFIG_FILE="$OMBRE_ACTIVE_DIR/config.yaml"
"$DOCKER_BIN" compose --project-directory "$OMBRE_ACTIVE_DIR" -p "$compose_project" \
  -f "$OMBRE_ACTIVE_DIR/docker-compose.yml" up -d --no-build "$OMBRE_SERVICE"
wait_health

running_image="$($DOCKER_BIN inspect --format '{{.Image}}' "$OMBRE_CONTAINER_NAME")"
running_container="$($DOCKER_BIN inspect --format '{{.Id}}' "$OMBRE_CONTAINER_NAME")"
[[ "$running_image" == "$new_image" ]]
[[ "$running_container" != "$container_id" ]]

"$PYTHON_BIN" - "$OMBRE_ACTIVE_DIR/deployment-manifest.json" <<PY
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = {
    "commit": "$COMMIT",
    "archive_sha256": "$ARCHIVE_SHA256",
    "container_id": "$running_container",
    "image_id": "$running_image",
    "previous_image_id": "$old_image",
    "rollback_container": "$rollback_container",
    "rollback_container_id": "$container_id",
    "compose_project": "$compose_project",
    "data_dir": "$OMBRE_DATA_DIR",
    "host_port": 8001,
}
path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
PY
printf '%s\n' "$running_image" >"$OMBRE_ACTIVE_DIR/.deployed_image"

umask 077
{
  printf 'OMBRE_EXPECTED_CONTAINER_ID=%s\n' "$running_container"
  printf 'OMBRE_EXPECTED_IMAGE_ID=%s\n' "$running_image"
} >"$anchor_tmp"
chmod 600 "$anchor_tmp"
mv "$anchor_tmp" "$OMBRE_DEPLOYMENT_ANCHOR_FILE"

# Prove that the same anchor file consumed by the nightly backup resolves to
# the newly running 8001 instance before the deployment can be reported green.
"$BACKUP_SCRIPT" --env-file "$ENV_FILE" --check

if ((anchor_had_previous)); then
  rm -f -- "$anchor_backup"
fi

trap - EXIT INT TERM HUP
printf 'DEPLOY_OK commit=%s image=%s container=%s previous=%s rollback_container=%s anchor=%s\n' \
  "$COMMIT" "$running_image" "$running_container" "$previous" \
  "$rollback_container" "$OMBRE_DEPLOYMENT_ANCHOR_FILE"
