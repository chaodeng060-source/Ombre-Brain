#!/usr/bin/env bash
set -Eeuo pipefail

COMPOSE_DIR="${OMBRE_COMPOSE_DIR:-/vol1/ombre-deploy}"
COMPOSE_FILE="${OMBRE_COMPOSE_FILE:-${COMPOSE_DIR}/docker-compose.yml}"
DATA_DIR="${OMBRE_DATA_DIR:-/vol1/ombre-data}"
DATA_TARGET="${OMBRE_DATA_TARGET:-/data}"
EXPECTED_VOLUME_ROOT="${OMBRE_EXPECTED_VOLUME_ROOT:-/vol1}"
EXPECTED_VOLUME_UUID="${OMBRE_EXPECTED_VOLUME_UUID:-7f87385b-17c7-41d6-86d9-4b315d5f3b29}"
EXPECTED_DOCKER_ROOT="${OMBRE_EXPECTED_DOCKER_ROOT:-/vol1/docker}"
SERVICE="${OMBRE_SERVICE:-ombre-brain}"
MCP_URL="${OMBRE_MCP_URL:-http://127.0.0.1:8000/mcp}"
DOCKER_WAIT_SECONDS="${OMBRE_DOCKER_WAIT_SECONDS:-300}"
MCP_WAIT_SECONDS="${OMBRE_MCP_WAIT_SECONDS:-90}"
POLL_SECONDS="${OMBRE_POLL_SECONDS:-5}"
LOCK_FILE="${OMBRE_LOCK_FILE:-${XDG_RUNTIME_DIR:-/tmp}/ombre-brain-recovery-${UID}.lock}"
CRON_BEGIN="# BEGIN ombre-brain self-recovery"
CRON_END="# END ombre-brain self-recovery"
CRON_SAFE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
MCP_PAYLOAD='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"ombre-nas-recovery","version":"1.0"}}}'
TEMP_FILES=()

cleanup() {
  if ((${#TEMP_FILES[@]})); then
    rm -f -- "${TEMP_FILES[@]}"
  fi
}
trap cleanup EXIT

info() {
  printf '[ombre-recovery] %s\n' "$*"
}

event() {
  info "$*"
  if command -v logger >/dev/null 2>&1; then
    logger -t ombre-brain-recovery -- "$*" || true
  fi
}

die() {
  event "ERROR: $*"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

canonical_path() {
  realpath "$1" 2>/dev/null || die "cannot resolve path: $1"
}

install_cron() {
  require_command awk
  require_command bash
  require_command crontab
  require_command flock
  require_command grep
  require_command mktemp
  require_command realpath

  exec 8>"${LOCK_FILE}.cron"
  flock -n 8 || die "another cron installation is already running"

  local script_path bash_path current cron_error filtered updated
  script_path="$(canonical_path "$0")"
  bash_path="$(canonical_path "$(command -v bash)")"
  [[ "${script_path}" =~ ^/[A-Za-z0-9._/-]+$ ]] ||
    die "unsafe script path for cron: ${script_path}"
  [[ "${bash_path}" =~ ^/[A-Za-z0-9._/-]+$ ]] ||
    die "unsafe bash path for cron: ${bash_path}"

  current="$(mktemp)"
  cron_error="$(mktemp)"
  filtered="$(mktemp)"
  updated="$(mktemp)"
  TEMP_FILES+=("${current}" "${cron_error}" "${filtered}" "${updated}")

  if ! crontab -l >"${current}" 2>"${cron_error}"; then
    if [[ ! -s "${current}" ]] && grep -qi 'no crontab for' "${cron_error}"; then
      : >"${current}"
    else
      die "cannot safely read current crontab"
    fi
  fi

  if ! awk -v begin="${CRON_BEGIN}" -v end="${CRON_END}" '
    $0 == begin {
      if (inside || seen_begin) bad = 1
      inside = 1
      seen_begin = 1
      next
    }
    $0 == end {
      if (!inside || seen_end) bad = 1
      inside = 0
      seen_end = 1
      next
    }
    END {
      if (inside || seen_begin != seen_end) bad = 1
      exit bad
    }
  ' "${current}"; then
    die "existing recovery cron markers are malformed; refusing overwrite"
  fi

  awk -v begin="${CRON_BEGIN}" -v end="${CRON_END}" '
    $0 == begin { skip = 1; next }
    $0 == end { skip = 0; next }
    !skip { print }
  ' "${current}" >"${filtered}"

  {
    cat "${filtered}"
    printf '%s\n' "${CRON_BEGIN}"
    printf 'PATH=%s\n' "${CRON_SAFE_PATH}"
    printf '@reboot sleep 90 && %s %s --recover >/dev/null 2>&1\n' \
      "${bash_path}" "${script_path}"
    printf '*/5 * * * * %s %s --recover >/dev/null 2>&1\n' \
      "${bash_path}" "${script_path}"
    printf '%s\n' "${CRON_END}"
  } >"${updated}"

  crontab "${updated}"
  local marker_count
  marker_count="$(crontab -l | grep -Fxc "${CRON_BEGIN}" || true)"
  [[ "${marker_count}" == "1" ]] || die "cron installation verification failed"
  event "installed idempotent reboot and five-minute recovery checks"
}

wait_for_docker() {
  local deadline=$((SECONDS + DOCKER_WAIT_SECONDS))
  until docker info >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      die "Docker did not become ready within ${DOCKER_WAIT_SECONDS}s"
    fi
    sleep "${POLL_SECONDS}"
  done
}

probe_mcp() {
  local body http_code
  body="$(mktemp)"
  TEMP_FILES+=("${body}")
  if ! http_code="$(
    curl -sS \
      --connect-timeout 3 \
      --max-time 15 \
      -o "${body}" \
      -w '%{http_code}' \
      -X POST "${MCP_URL}" \
      -H 'Content-Type: application/json' \
      -H 'Accept: application/json, text/event-stream' \
      --data-binary "${MCP_PAYLOAD}" 2>/dev/null
  )"; then
    rm -f "${body}"
    return 1
  fi

  if [[ "${http_code}" == "200" ]] && grep -q '"serverInfo"' "${body}"; then
    rm -f "${body}"
    return 0
  fi

  rm -f "${body}"
  return 1
}

wait_for_mcp() {
  local deadline=$((SECONDS + MCP_WAIT_SECONDS))
  until probe_mcp; do
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep "${POLL_SECONDS}"
  done
  return 0
}

guard_storage() {
  [[ -d "${COMPOSE_DIR}" ]] || die "compose directory missing: ${COMPOSE_DIR}"
  [[ -r "${COMPOSE_FILE}" ]] || die "compose file missing or unreadable: ${COMPOSE_FILE}"
  [[ -d "${DATA_DIR}" ]] || die "data directory missing: ${DATA_DIR}"

  local expected_volume compose_mount data_mount compose_uuid data_uuid
  expected_volume="$(canonical_path "${EXPECTED_VOLUME_ROOT}")"
  compose_mount="$(findmnt -T "${COMPOSE_DIR}" -n -o TARGET 2>/dev/null || true)"
  data_mount="$(findmnt -T "${DATA_DIR}" -n -o TARGET 2>/dev/null || true)"
  compose_uuid="$(findmnt -T "${COMPOSE_DIR}" -n -o UUID 2>/dev/null || true)"
  data_uuid="$(findmnt -T "${DATA_DIR}" -n -o UUID 2>/dev/null || true)"
  [[ -n "${compose_mount}" ]] || die "compose directory is not on a mounted filesystem"
  [[ -n "${data_mount}" ]] || die "data directory is not on a mounted filesystem"
  [[ -n "${compose_uuid}" ]] || die "compose filesystem UUID is unavailable"
  [[ -n "${data_uuid}" ]] || die "data filesystem UUID is unavailable"
  [[ "$(canonical_path "${compose_mount}")" == "${expected_volume}" ]] ||
    die "compose directory is not mounted from ${expected_volume}"
  [[ "$(canonical_path "${data_mount}")" == "${expected_volume}" ]] ||
    die "data directory is not mounted from ${expected_volume}"
  [[ "${compose_uuid}" == "${EXPECTED_VOLUME_UUID}" ]] ||
    die "compose filesystem UUID mismatch"
  [[ "${data_uuid}" == "${EXPECTED_VOLUME_UUID}" ]] ||
    die "data filesystem UUID mismatch"

  if [[ "${OMBRE_ALLOW_EMPTY_DATA:-0}" != "1" ]]; then
    if ! find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      die "data directory is empty; refusing automatic recovery"
    fi
  fi
}

guard_docker_root() {
  local actual_root expected_root
  actual_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" ||
    die "cannot read DockerRootDir"
  expected_root="$(canonical_path "${EXPECTED_DOCKER_ROOT}")"
  [[ "$(canonical_path "${actual_root}")" == "${expected_root}" ]] ||
    die "DockerRootDir mismatch: expected ${expected_root}, got ${actual_root}"
}

compose() {
  docker compose \
    --project-directory "${COMPOSE_DIR}" \
    -f "${COMPOSE_FILE}" \
    "$@"
}

validate_compose_data_mount() {
  local compose_json expected_source
  compose_json="$(mktemp)"
  TEMP_FILES+=("${compose_json}")
  expected_source="$(canonical_path "${DATA_DIR}")"
  compose config --format json >"${compose_json}" ||
    die "cannot render docker compose configuration"

  python3 - "${compose_json}" "${SERVICE}" "${expected_source}" "${DATA_TARGET}" <<'PY'
import json
import os
import sys

config_path, service_name, expected_source, expected_target = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)

try:
    volumes = config["services"][service_name].get("volumes", [])
except KeyError:
    raise SystemExit(f"compose service not found: {service_name}")

for volume in volumes:
    if isinstance(volume, str):
        parts = volume.split(":")
        if len(parts) < 2:
            continue
        source, target = parts[0], parts[1]
        volume_type = "bind"
    else:
        source = volume.get("source")
        target = volume.get("target")
        volume_type = volume.get("type")
    if (
        volume_type == "bind"
        and source
        and os.path.realpath(source) == expected_source
        and target == expected_target
    ):
        raise SystemExit(0)

raise SystemExit(
    f"compose service {service_name} does not bind "
    f"{expected_source} to {expected_target}"
)
PY
}

validate_container_data_mount() {
  local container_id="$1" mounts_json expected_source
  mounts_json="$(mktemp)"
  TEMP_FILES+=("${mounts_json}")
  expected_source="$(canonical_path "${DATA_DIR}")"
  docker inspect --format '{{json .Mounts}}' "${container_id}" >"${mounts_json}" ||
    die "cannot inspect container mounts for ${container_id}"

  python3 - "${mounts_json}" "${expected_source}" "${DATA_TARGET}" <<'PY'
import json
import os
import sys

mounts_path, expected_source, expected_target = sys.argv[1:]
with open(mounts_path, encoding="utf-8") as handle:
    mounts = json.load(handle)

for mount in mounts:
    if (
        mount.get("Type") == "bind"
        and os.path.realpath(mount.get("Source", "")) == expected_source
        and mount.get("Destination") == expected_target
        and mount.get("RW") is True
    ):
        raise SystemExit(0)

raise SystemExit(
    f"container does not bind {expected_source} read-write to {expected_target}"
)
PY
}

recover_service() {
  require_command curl
  require_command docker
  require_command find
  require_command findmnt
  require_command flock
  require_command grep
  require_command mktemp
  require_command python3
  require_command realpath

  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    info "another recovery check is already running"
    return 0
  fi

  guard_storage
  wait_for_docker
  guard_docker_root

  compose config --quiet ||
    die "docker compose configuration is invalid"
  validate_compose_data_mount ||
    die "docker compose data mount validation failed"

  local container_id running
  container_id="$(compose ps -q --all "${SERVICE}")" ||
    die "cannot inspect compose service ${SERVICE}"

  if [[ -n "${container_id}" ]]; then
    validate_container_data_mount "${container_id}" ||
      die "container data mount validation failed"
    running="$(docker inspect --format '{{.State.Running}}' "${container_id}" 2>/dev/null)" ||
      die "cannot inspect container ${container_id}"
    if [[ "${running}" == "true" ]]; then
      if [[ "${1}" == "--check" ]]; then
        if wait_for_mcp; then
          info "service passed MCP initialize; no action needed"
          return 0
        fi
        die "container is running but MCP stayed unhealthy for ${MCP_WAIT_SECONDS}s; refusing restart loop"
      fi
      if probe_mcp; then
        info "service passed MCP initialize; no action needed"
        return 0
      fi
      die "container is running but MCP is unhealthy; refusing restart loop"
    fi

    if [[ "${1}" == "--check" ]]; then
      die "container exists but is stopped"
    fi
    event "container is stopped; starting existing compose service"
    compose up -d "${SERVICE}" ||
      die "failed to start existing compose service"
  else
    if [[ "${1}" == "--check" ]]; then
      die "container is missing"
    fi
    event "container is missing; rebuilding from the verified compose project"
    compose up -d --build "${SERVICE}" ||
      die "failed to rebuild missing compose service"
  fi

  container_id="$(compose ps -q --all "${SERVICE}")" ||
    die "cannot inspect recovered compose service ${SERVICE}"
  [[ -n "${container_id}" ]] || die "compose recovery did not create ${SERVICE}"
  validate_container_data_mount "${container_id}" ||
    die "recovered container data mount validation failed"
  wait_for_mcp ||
    die "MCP endpoint did not become ready within ${MCP_WAIT_SECONDS}s"
  event "service recovery completed and MCP initialize succeeded"
}

mode="${1:---recover}"
case "${mode}" in
  --recover|--check)
    recover_service "${mode}"
    ;;
  --install-cron)
    install_cron
    ;;
  *)
    printf 'Usage: %s [--recover|--check|--install-cron]\n' "$0" >&2
    exit 2
    ;;
esac
