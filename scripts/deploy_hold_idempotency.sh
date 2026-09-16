#!/usr/bin/env bash
# Run only after this task's review and production authorization.
set -Eeuo pipefail
minute=$(TZ=Asia/Shanghai date +%H%M)
minute=$((10#$minute))
if ((minute >= 100 && minute < 630)); then
  echo 'refusing deployment during 01:00-06:30 Asia/Shanghai night run' >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
exec bash "$script_dir/deploy_nas_atomic.sh" "$@"
