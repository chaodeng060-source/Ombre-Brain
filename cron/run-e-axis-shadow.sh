#!/bin/sh
set -eu

DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
: "${OMBRE_CONTAINER_NAME:?set OMBRE_CONTAINER_NAME to the intended Ombre writer}"
CONTAINER="$OMBRE_CONTAINER_NAME"
if [ "$CONTAINER" != "ombre-vps-mirror" ]; then
    echo "refusing non-production container: $CONTAINER" >&2
    exit 2
fi

exec "$DOCKER_BIN" exec "$CONTAINER" \
    python /app/e_axis_night.py
