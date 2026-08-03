#!/bin/sh
set -eu

DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
CONTAINER="${OMBRE_CONTAINER_NAME:-ombre-brain}"
LOCK_FILE="${OMBRE_LMC5_LOCK_FILE:-/tmp/ombre-lmc5-night.lock}"
PATROL_STATE_DIR="${OMBRE_PATROL_STATE_DIR:-/data/.lmc5/patrol}"

exec 9>"$LOCK_FILE"
if ! /usr/bin/flock -n 9; then
    exit 0
fi

if "$DOCKER_BIN" exec "$CONTAINER" \
    python /app/night_run_trigger.py; then
    :
else
    night_status=$?
    exit "$night_status"
fi

exec "$DOCKER_BIN" exec "$CONTAINER" \
    python /app/patrol_night.py \
    --config /app/config.yaml \
    --state-dir "$PATROL_STATE_DIR"
