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
[[ -f "$ENV_FILE" && ! -L "$ENV_FILE" ]] || {
  echo "job environment must be a regular non-symlink file" >&2
  exit 2
}

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

: "${OMBRE_CONTAINER_NAME:?missing OMBRE_CONTAINER_NAME}"
: "${OMBRE_JOB_ENV_FILE:?missing OMBRE_JOB_ENV_FILE}"
: "${OMBRE_JOB_REPO_DIR:?missing OMBRE_JOB_REPO_DIR}"
[[ "$OMBRE_CONTAINER_NAME" == "ombre-vps-mirror" ]] || {
  echo "refusing non-production container: $OMBRE_CONTAINER_NAME" >&2
  exit 2
}

for value in "$OMBRE_JOB_ENV_FILE" "$OMBRE_JOB_REPO_DIR"; do
  [[ "$value" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "unsafe cron path: $value" >&2
    exit 2
  }
done

[[ -f "$OMBRE_JOB_ENV_FILE" && -r "$OMBRE_JOB_ENV_FILE" && ! -L "$OMBRE_JOB_ENV_FILE" ]] || {
  echo "configured job environment must be a readable regular non-symlink file" >&2
  exit 2
}
env_file_real="$(realpath "$ENV_FILE")"
job_env_real="$(realpath "$OMBRE_JOB_ENV_FILE")"
[[ "$env_file_real" == "$job_env_real" ]] || {
  echo "configured job environment is not the validated --env-file" >&2
  exit 2
}

CRONTAB_BIN="${CRONTAB_BIN:-crontab}"
repo="$OMBRE_JOB_REPO_DIR"
env_path="$env_file_real"
backup_script="$repo/scripts/backup_nas_data.sh"
[[ -f "$backup_script" ]] || {
  echo "tracked 8001 backup script is missing: $backup_script" >&2
  exit 1
}

# Do not install a nightly job until its exact environment and post-deploy
# container/image anchors already pass the same check cron will run.
/usr/bin/bash "$backup_script" --env-file "$ENV_FILE" --check

BEGIN="# BEGIN ombre-vps-mirror production jobs"
END="# END ombre-vps-mirror production jobs"
tmp_root="${TMPDIR:-/tmp}"
current="$(mktemp "$tmp_root/ombre-cron-current.XXXXXX")"
cron_error="$(mktemp "$tmp_root/ombre-cron-error.XXXXXX")"
filtered="$(mktemp "$tmp_root/ombre-cron-filtered.XXXXXX")"
updated="$(mktemp "$tmp_root/ombre-cron-updated.XXXXXX")"
fresh="$(mktemp "$tmp_root/ombre-cron-fresh.XXXXXX")"
drop_tmp=""
trap 'rm -f -- "$current" "$cron_error" "$filtered" "$updated" "$fresh" ${drop_tmp:+"$drop_tmp"}' EXIT

# Replacing a user's whole crontab can never provide compare-and-swap: another
# editor may commit after the final `crontab -l` and before `crontab FILE`.
# Production therefore owns one scheduler drop-in and updates only that file by
# same-directory rename.  The old CRONTAB_BIN mutation path remains reachable
# solely by the bounded contract-test fixture used by the adjacent legacy tests.
cron_file="${OMBRE_JOB_CRON_FILE:-}"
cron_user="${OMBRE_JOB_CRON_USER:-}"
legacy_test_mode=0
if [[ -z "$cron_file" ]]; then
  if [[ -n "${OMBRE_CONTRACT_TEST_ROOT:-}" && -n "${CRON_STATE:-}" && "$CRONTAB_BIN" == /* ]]; then
    test_root_real="$(realpath "$OMBRE_CONTRACT_TEST_ROOT")"
    crontab_bin_real="$(realpath "$CRONTAB_BIN")"
    cron_state_real="$(realpath "$CRON_STATE")"
    if [[ "$crontab_bin_real" == "$test_root_real/"* && "$cron_state_real" == "$test_root_real/"* ]]; then
      legacy_test_mode=1
    fi
  fi
  if (( ! legacy_test_mode )); then
    echo "production job install requires an independent cron drop-in (OMBRE_JOB_CRON_FILE)" >&2
    exit 1
  fi
else
  [[ "$cron_file" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "unsafe cron drop-in path: $cron_file" >&2
    exit 2
  }
  [[ "$cron_file" == */cron.d/* ]] || {
    echo "cron job path is not an independent cron.d drop-in: $cron_file" >&2
    exit 2
  }
  [[ -n "$cron_user" && "$cron_user" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]] || {
    echo "missing or unsafe OMBRE_JOB_CRON_USER" >&2
    exit 2
  }
  cron_dir="$(dirname "$cron_file")"
  cron_base="$(basename "$cron_file")"
  [[ "$cron_base" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "unsafe cron drop-in name: $cron_base" >&2
    exit 2
  }
  [[ -d "$cron_dir" && ! -L "$cron_dir" ]] || {
    echo "cron drop-in directory must already exist and not be a symlink: $cron_dir" >&2
    exit 1
  }
  if [[ -e "$cron_file" || -L "$cron_file" ]]; then
    [[ -f "$cron_file" && ! -L "$cron_file" ]] || {
      echo "cron drop-in must be a regular non-symlink file: $cron_file" >&2
      exit 1
    }
  fi
fi

if ! "$CRONTAB_BIN" -l >"$current" 2>"$cron_error"; then
  if [[ ! -s "$current" ]] && grep -qi 'no crontab for' "$cron_error"; then
    : >"$current"
  else
    echo "cannot safely read current crontab" >&2
    exit 1
  fi
fi

awk -v begin="$BEGIN" -v end="$END" '
  $0 == begin { if (inside || seen) bad=1; inside=1; seen=1; next }
  $0 == end { if (!inside) bad=1; inside=0; next }
  END { if (inside) bad=1; exit bad }
' "$current" || {
  echo "malformed managed cron markers" >&2
  exit 1
}

if ((legacy_test_mode)); then
  awk -v begin="$BEGIN" -v end="$END" '
    $0 == begin { inside=1; next }
    $0 == end { inside=0; next }
    !inside { print }
  ' "$current" >"$filtered"
else
  # A managed block created by the retired whole-crontab installer is also an
  # old writer during drop-in migration, so scan the complete user crontab.
  cp "$current" "$filtered"
fi

if grep -Eq '^[[:space:]]*[^#].*(ombre_daily_backup\.sh|backup_nas_data\.sh|run-lmc5-night\.sh|run-e-axis-shadow\.sh|night_run_trigger\.py|patrol_night\.py|e_axis_night\.py)' "$filtered"; then
  echo "unmanaged Ombre writer/backup cron remains; remove it explicitly first" >&2
  exit 1
fi

if (( ! legacy_test_mode )); then
  DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
  CURL_BIN="${CURL_BIN:-/usr/bin/curl}"
  public_health_url="${OMBRE_PUBLIC_HEALTH_URL:-}"
  [[ "$public_health_url" == "https://memory.zhaodeng.xyz/health" ]] || {
    echo "OMBRE_PUBLIC_HEALTH_URL must identify the audited public health endpoint" >&2
    exit 1
  }
  if ! legacy_running="$("$DOCKER_BIN" inspect --format '{{.State.Running}}' ombre-brain 2>/dev/null)"; then
    echo "cannot verify that the legacy writer container is stopped" >&2
    exit 1
  fi
  [[ "$legacy_running" == "false" ]] || {
    echo "legacy writer container is still running: ombre-brain" >&2
    exit 1
  }
  if ! "$CURL_BIN" --fail --silent --show-error --max-time 10 \
      --header 'Cache-Control: no-cache' --header 'Pragma: no-cache' \
      "$public_health_url" >/dev/null; then
    echo "public health failed while the legacy writer was stopped" >&2
    exit 1
  fi
fi

if ((legacy_test_mode)); then
  {
    cat "$filtered"
    printf '%s\n' "$BEGIN"
    printf '0 4 * * * /usr/bin/bash %s/scripts/backup_nas_data.sh --env-file %s\n' "$repo" "$env_path"
    printf '30 4 * * * /usr/bin/env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/bash -c '\''set -a; . %s; set +a; exec %s/cron/run-lmc5-night.sh'\''\n' "$env_path" "$repo"
    printf '30 5 * * * /usr/bin/env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/bash -c '\''set -a; . %s; set +a; exec %s/cron/run-e-axis-shadow.sh'\''\n' "$env_path" "$repo"
    printf '%s\n' "$END"
  } >"$updated"
else
  {
    printf '%s\n' '# Managed by Ombre scripts/install_nas_jobs.sh; do not mix unrelated jobs here.'
    printf '%s\n' 'SHELL=/bin/sh'
    printf '%s\n' 'PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
    printf '0 4 * * * %s /usr/bin/bash %s/scripts/backup_nas_data.sh --env-file %s\n' "$cron_user" "$repo" "$env_path"
    printf '30 4 * * * %s /usr/bin/env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/bash -c '\''set -a; . %s; set +a; exec %s/cron/run-lmc5-night.sh'\''\n' "$cron_user" "$env_path" "$repo"
    printf '30 5 * * * %s /usr/bin/env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/bash -c '\''set -a; . %s; set +a; exec %s/cron/run-e-axis-shadow.sh'\''\n' "$cron_user" "$env_path" "$repo"
  } >"$updated"
fi

if ((CHECK_ONLY)); then
  cat "$updated"
  exit 0
fi

if (( ! legacy_test_mode )); then
  drop_tmp="$(mktemp "$cron_dir/.${cron_base}.tmp.XXXXXX")"
  cp "$updated" "$drop_tmp"
  chmod 0644 "$drop_tmp"
  mv -f "$drop_tmp" "$cron_file"
  drop_tmp=""
  if ! cmp -s "$updated" "$cron_file"; then
    echo "cron drop-in failed post-install verification" >&2
    exit 1
  fi
  echo "JOBS_OK container=$OMBRE_CONTAINER_NAME env=$env_path cron_file=$cron_file"
  exit 0
fi

# Compatibility path for the bounded contract-test fixture.  Production never
# reaches this whole-crontab mutation because it has no real compare-and-swap.
: >"$cron_error"
if ! "$CRONTAB_BIN" -l >"$fresh" 2>"$cron_error"; then
  if [[ ! -s "$fresh" ]] && grep -qi 'no crontab for' "$cron_error"; then
    : >"$fresh"
  else
    echo "cannot safely re-read current crontab" >&2
    exit 1
  fi
fi
cmp -s "$current" "$fresh" || {
  echo "crontab changed concurrently; refusing to overwrite it" >&2
  exit 1
}

"$CRONTAB_BIN" "$updated"
if ! installed="$($CRONTAB_BIN -l)" ||
   [[ "$(grep -Fxc "$BEGIN" <<<"$installed")" != 1 ]] ||
   [[ "$(grep -Fxc "$END" <<<"$installed")" != 1 ]] ||
   [[ "$(grep -Fc -- '--env-file' <<<"$installed")" != 1 ]] ||
   [[ "$(grep -Fc 'run-lmc5-night.sh' <<<"$installed")" != 1 ]] ||
   [[ "$(grep -Fc 'run-e-axis-shadow.sh' <<<"$installed")" != 1 ]]; then
  if "$CRONTAB_BIN" "$current" >/dev/null 2>&1; then
    echo "installed crontab failed verification; original restored" >&2
  else
    echo "installed crontab failed verification; original restoration also failed" >&2
  fi
  exit 1
fi
echo "JOBS_OK container=$OMBRE_CONTAINER_NAME env=$env_path"
