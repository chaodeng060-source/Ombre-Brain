#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "usage: $0 --env-file PATH [--check]" >&2
  exit 2
}

ENV_FILE=""
CHECK_ONLY=0
while (($#)); do
  case "$1" in
    --env-file) [[ $# -ge 2 ]] || usage; ENV_FILE="$2"; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    *) usage ;;
  esac
done
[[ -n "$ENV_FILE" && -r "$ENV_FILE" ]] || usage

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

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
: "${OMBRE_ACTIVE_DIR:?missing OMBRE_ACTIVE_DIR}"
: "${OMBRE_DATA_DIR:?missing OMBRE_DATA_DIR}"
: "${OMBRE_BACKUP_ROOT:?missing OMBRE_BACKUP_ROOT}"
: "${OMBRE_HEALTH_URL:?missing OMBRE_HEALTH_URL}"
: "${OMBRE_DEPLOYMENT_ANCHOR_FILE:?missing OMBRE_DEPLOYMENT_ANCHOR_FILE}"
: "${OMBRE_MUTATION_LOCK_FILE:?missing OMBRE_MUTATION_LOCK_FILE}"

[[ "$OMBRE_CONTAINER_NAME" == "ombre-vps-mirror" ]] || {
  echo "refusing non-production container: $OMBRE_CONTAINER_NAME" >&2
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
  [[ "$(realpath "$OMBRE_BACKUP_ROOT")" == "$test_root/backups" ]]
  [[ "$(realpath "$(dirname "$OMBRE_DEPLOYMENT_ANCHOR_FILE")")" == "$test_root" ]]
  [[ "$(basename "$OMBRE_DEPLOYMENT_ANCHOR_FILE")" == "deployment-anchor.env" ]]
  [[ "$(realpath "$(dirname "$OMBRE_MUTATION_LOCK_FILE")")" == "$test_root" ]]
  [[ "$(basename "$OMBRE_MUTATION_LOCK_FILE")" == "mutation.lock" ]]
else
  [[ "$(realpath "$OMBRE_ACTIVE_DIR")" == "/vol1/ombre-migrate/code" ]]
  [[ "$(realpath "$OMBRE_DATA_DIR")" == "/vol1/ombre-migrate/data" ]]
  [[ "$(realpath "$OMBRE_BACKUP_ROOT")" == "/home/zhaodeng/ombre-backups/ombre-vps-mirror" ]]
  [[ "$(realpath "$(dirname "$OMBRE_DEPLOYMENT_ANCHOR_FILE")")" == "/vol1/ombre-migrate" ]]
  [[ "$(basename "$OMBRE_DEPLOYMENT_ANCHOR_FILE")" == "deployment-anchor.env" ]]
  [[ "$OMBRE_MUTATION_LOCK_FILE" == "/vol1/ombre-migrate/ombre-production.lock" ]]
fi
[[ ! -L "$OMBRE_DEPLOYMENT_ANCHOR_FILE" ]]

DOCKER_BIN="${DOCKER_BIN:-docker}"
CURL_BIN="${CURL_BIN:-curl}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TAR_BIN="${TAR_BIN:-tar}"
REALPATH_BIN="${REALPATH_BIN:-realpath}"
FLOCK_BIN="${FLOCK_BIN:-flock}"
KEEP="${OMBRE_BACKUP_KEEP:-7}"
HEALTH_WAIT_SECONDS="${OMBRE_HEALTH_WAIT_SECONDS:-180}"

if [[ -n "${OMBRE_MUTATION_LOCK_HELD_FD:-}" ]]; then
  [[ "$OMBRE_MUTATION_LOCK_HELD_FD" =~ ^[0-9]+$ ]]
  inherited_target="$(realpath "/proc/$$/fd/$OMBRE_MUTATION_LOCK_HELD_FD")"
  [[ "$inherited_target" == "$(realpath "$OMBRE_MUTATION_LOCK_FILE")" ]] || {
    echo "inherited production mutation lock does not match" >&2
    exit 1
  }
  "$FLOCK_BIN" -n "$OMBRE_MUTATION_LOCK_HELD_FD" || {
    echo "inherited production mutation lock is not held" >&2
    exit 1
  }
else
  exec 8>"$OMBRE_MUTATION_LOCK_FILE"
  "$FLOCK_BIN" -n 8 || {
    echo "production mutation lock is held" >&2
    exit 1
  }
fi

[[ "$KEEP" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid OMBRE_BACKUP_KEEP: $KEEP" >&2
  exit 2
}
[[ "$HEALTH_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] &&
  ((HEALTH_WAIT_SECONDS >= 30 && HEALTH_WAIT_SECONDS <= 600)) || {
  echo "invalid OMBRE_HEALTH_WAIT_SECONDS: $HEALTH_WAIT_SECONDS" >&2
  exit 2
}

declared_data="$($REALPATH_BIN "$OMBRE_DATA_DIR")"
declared_code="$($REALPATH_BIN "$OMBRE_ACTIVE_DIR")"
mounted_data="$($DOCKER_BIN inspect --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Source}}{{end}}{{end}}' "$OMBRE_CONTAINER_NAME")"
mounted_data="$($REALPATH_BIN "$mounted_data")"
[[ "$mounted_data" == "$declared_data" ]] || {
  echo "container /data mismatch: declared=$declared_data mounted=$mounted_data" >&2
  exit 1
}

container_id="$($DOCKER_BIN inspect --format '{{.Id}}' "$OMBRE_CONTAINER_NAME")"
image_id="$($DOCKER_BIN inspect --format '{{.Image}}' "$OMBRE_CONTAINER_NAME")"
if [[ -n "${OMBRE_EXPECTED_CONTAINER_ID:-}" ]]; then
  [[ "$container_id" == "$OMBRE_EXPECTED_CONTAINER_ID" ]] || {
    echo "container anchor changed: $container_id" >&2
    exit 1
  }
fi
if [[ -n "${OMBRE_EXPECTED_IMAGE_ID:-}" ]]; then
  [[ "$image_id" == "$OMBRE_EXPECTED_IMAGE_ID" ]] || {
    echo "image anchor changed: $image_id" >&2
    exit 1
  }
fi

[[ -d "$declared_data" && -d "$declared_code" ]] || {
  echo "declared code/data directory is missing" >&2
  exit 1
}

if ((CHECK_ONLY)); then
  printf 'CHECK_OK container=%s image=%s data=%s code=%s\n' \
    "$container_id" "$image_id" "$declared_data" "$declared_code"
  exit 0
fi

umask 077
mkdir -p "$OMBRE_BACKUP_ROOT"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_work="$OMBRE_BACKUP_ROOT/.incomplete-$stamp"
backup_dir="$OMBRE_BACKUP_ROOT/backup-$stamp"
restore_check="$OMBRE_BACKUP_ROOT/.restore-check-$stamp"
stopped=0

wait_health() {
  local deadline=$((SECONDS + HEALTH_WAIT_SECONDS))
  while ((SECONDS < deadline)); do
    if "$CURL_BIN" -fsS --max-time 3 "$OMBRE_HEALTH_URL" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

cleanup() {
  local rc=$?
  trap - EXIT
  trap '' INT TERM HUP
  if ((stopped)); then
    "$DOCKER_BIN" start "$OMBRE_CONTAINER_NAME" >/dev/null || rc=1
    wait_health || rc=1
  fi
  if [[ -d "$restore_check" ]]; then
    find "$restore_check" -depth -delete
  fi
  if [[ -d "$backup_work" ]]; then
    mv "$backup_work" "$OMBRE_BACKUP_ROOT/failed-$stamp"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

stopped=1
"$DOCKER_BIN" stop "$OMBRE_CONTAINER_NAME" >/dev/null
mkdir -m 700 "$backup_work" "$restore_check"

quick_check='import pathlib,sqlite3,sys; root=pathlib.Path(sys.argv[1]); dbs=sorted({*root.rglob("*.db"),*root.rglob("*.sqlite"),*root.rglob("*.sqlite3")}); assert dbs,"no sqlite files"; results=[]; [(lambda c,p:(results.append((str(p),c.execute("PRAGMA quick_check").fetchone()[0])),c.close()))(sqlite3.connect(f"file:{p}?mode=ro",uri=True),p) for p in dbs]; bad=[r for r in results if r[1] != "ok"]; assert not bad,bad; print(f"quick_check ok: {len(dbs)} sqlite files")'
"$PYTHON_BIN" -c "$quick_check" "$declared_data"

"$TAR_BIN" -C "$declared_code" -cf "$backup_work/code.tar" .
"$TAR_BIN" -C "$declared_data" -cf "$backup_work/data.tar" .
"$TAR_BIN" -C "$restore_check" -xf "$backup_work/data.tar"
"$PYTHON_BIN" -c "$quick_check" "$restore_check"

(cd "$backup_work" && sha256sum code.tar data.tar >SHA256SUMS)
printf '%s\n' "$container_id" > "$backup_work/container-id"
printf '%s\n' "$image_id" > "$backup_work/image-id"

"$DOCKER_BIN" start "$OMBRE_CONTAINER_NAME" >/dev/null
stopped=0
wait_health
mv "$backup_work" "$backup_dir"

mapfile -t backups < <(
  find "$OMBRE_BACKUP_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -name 'backup-[0-9]*T[0-9]*Z' -print | sort -r
)
if ((${#backups[@]} > KEEP)); then
  for old in "${backups[@]:KEEP}"; do
    [[ "$old" == "$OMBRE_BACKUP_ROOT"/backup-* ]] || {
      echo "unsafe retention target: $old" >&2
      exit 1
    }
    find "$old" -depth -delete
  done
fi

printf 'BACKUP_OK path=%s container=%s image=%s\n' \
  "$backup_dir" "$container_id" "$image_id"
