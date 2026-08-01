#!/bin/sh
set -eu

DOCKER_BIN="${DOCKER_BIN:-/usr/bin/docker}"
CONTAINER="${OMBRE_CONTAINER_NAME:-ombre-brain}"

exec "$DOCKER_BIN" exec "$CONTAINER" \
    python /app/e_axis_night.py
