#!/bin/sh
set -eu

DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
: "${OMBRE_CONTAINER_NAME:?set OMBRE_CONTAINER_NAME to the intended Ombre writer}"
CONTAINER="$OMBRE_CONTAINER_NAME"
if [ "$CONTAINER" != "ombre-vps-mirror" ]; then
    echo "refusing non-production container: $CONTAINER" >&2
    exit 2
fi
LOCK_FILE="${OMBRE_LMC5_LOCK_FILE:-/tmp/ombre-lmc5-night.lock}"
PATROL_STATE_DIR="${OMBRE_PATROL_STATE_DIR:-/data/.lmc5/patrol}"

exec 9>"$LOCK_FILE"
if ! /usr/bin/flock -n 9; then
    exit 0
fi

night_status=0
patrol_status=0

if "$DOCKER_BIN" exec "$CONTAINER" \
    python /app/night_run_trigger.py; then
    :
else
    night_status=$?
fi

if "$DOCKER_BIN" exec "$CONTAINER" \
    python /app/patrol_night.py \
    --config /app/config.yaml \
    --state-dir "$PATROL_STATE_DIR"; then
    :
else
    patrol_status=$?
fi

if [ "$night_status" -ne 0 ]; then
    exit "$night_status"
fi
exit "$patrol_status"
