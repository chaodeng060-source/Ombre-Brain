#!/usr/bin/env bash
# Restore the exact container/source preserved by deploy_nas_atomic.sh.
# Never restore the data volume: memories written since deployment survive.
set -Eeuo pipefail
[[ $# == 2 && $1 == --env-file && -r $2 ]] || {
  echo "usage: $0 --env-file PATH" >&2; exit 2;
}
set -a
. "$2"
set +a
: "${OMBRE_ACTIVE_DIR:?} ${OMBRE_CONTAINER_NAME:?} ${OMBRE_MUTATION_LOCK_FILE:?}"
: "${OMBRE_DEPLOYMENT_ANCHOR_FILE:?} ${OMBRE_HEALTH_URL:?}"
[[ $OMBRE_ACTIVE_DIR == /vol1/ombre-migrate/code && $OMBRE_CONTAINER_NAME == ombre-vps-mirror ]]
[[ $OMBRE_MUTATION_LOCK_FILE == /vol1/ombre-migrate/ombre-production.lock ]]
[[ $OMBRE_DEPLOYMENT_ANCHOR_FILE == /vol1/ombre-migrate/deployment-anchor.env ]]
[[ $OMBRE_HEALTH_URL == http://127.0.0.1:8001/health ]]
[[ -d $OMBRE_ACTIVE_DIR && ! -L $OMBRE_ACTIVE_DIR ]]
exec 8>"$OMBRE_MUTATION_LOCK_FILE"
flock -n 8 || { echo 'production mutation lock is held' >&2; exit 1; }
mapfile -t receipt < <(python3 - "$OMBRE_ACTIVE_DIR/deployment-manifest.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for key in ('container_id','rollback_container','rollback_container_id','previous_image_id','previous_source_dir'):
    value = d[key]
    assert isinstance(value,str) and value and '\n' not in value
    print(value)
PY
)
[[ ${#receipt[@]} == 5 ]]
current_id=${receipt[0]}; old_name=${receipt[1]}; old_id=${receipt[2]}
old_image=${receipt[3]}; previous=${receipt[4]}
[[ $old_name =~ ^ombre-vps-mirror-rollback-[0-9]{8}T[0-9]{6}Z$ ]]
[[ $previous =~ ^/vol1/ombre-migrate/previous-[0-9]{8}T[0-9]{6}Z$ ]]
[[ -d $previous && ! -L $previous && -f $previous/server.py ]]
[[ $(docker inspect --format '{{.Id}}' "$OMBRE_CONTAINER_NAME") == "$current_id" ]]
[[ $(docker inspect --format '{{.Id}}' "$old_name") == "$old_id" ]]
[[ $(docker inspect --format '{{.Image}}' "$old_name") == "$old_image" ]]
stamp=$(date -u +%Y%m%dT%H%M%SZ)
saved_name="ombre-vps-mirror-reverted-$stamp"
saved_source="/vol1/ombre-migrate/reverted-$stamp"
anchor_before="/vol1/ombre-migrate/anchor-before-rollback-$stamp"
[[ ! -e $saved_source && ! -e $anchor_before ]]
if docker inspect "$saved_name" >/dev/null 2>&1; then exit 1; fi
umask 077
cp -p "$OMBRE_DEPLOYMENT_ANCHOR_FILE" "$anchor_before"
wait_health() {
  local until=$((SECONDS+180))
  while ((SECONDS < until)); do
    if curl -fsS --max-time 3 "$OMBRE_HEALTH_URL" >/dev/null; then return 0; fi
    sleep 1
  done
  return 1
}
recover_forward() {
  local rc=$? active_id
  trap - EXIT INT TERM HUP
  set +e
  active_id=$(docker inspect --format '{{.Id}}' "$OMBRE_CONTAINER_NAME" 2>/dev/null)
  if [[ $active_id == "$old_id" ]]; then
    docker stop "$OMBRE_CONTAINER_NAME" >/dev/null
    docker rename "$OMBRE_CONTAINER_NAME" "$old_name"
  fi
  if [[ -d $saved_source ]]; then
    if [[ -d $OMBRE_ACTIVE_DIR ]]; then mv "$OMBRE_ACTIVE_DIR" "$previous"; fi
    mv "$saved_source" "$OMBRE_ACTIVE_DIR"
  fi
  if [[ $(docker inspect --format '{{.Id}}' "$saved_name" 2>/dev/null) == "$current_id" ]]; then
    docker rename "$saved_name" "$OMBRE_CONTAINER_NAME"
  fi
  cp -p "$anchor_before" "$OMBRE_DEPLOYMENT_ANCHOR_FILE"
  docker start "$OMBRE_CONTAINER_NAME" >/dev/null
  wait_health
  echo 'rollback failed; attempted restoration of the pre-rollback service' >&2
  exit "$rc"
}
trap recover_forward EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
docker stop "$OMBRE_CONTAINER_NAME" >/dev/null
docker rename "$OMBRE_CONTAINER_NAME" "$saved_name"
mv "$OMBRE_ACTIVE_DIR" "$saved_source"
mv "$previous" "$OMBRE_ACTIVE_DIR"
docker rename "$old_name" "$OMBRE_CONTAINER_NAME"
docker start "$OMBRE_CONTAINER_NAME" >/dev/null
wait_health
[[ $(docker inspect --format '{{.Id}}' "$OMBRE_CONTAINER_NAME") == "$old_id" ]]
anchor_next="${OMBRE_DEPLOYMENT_ANCHOR_FILE}.rollback-$stamp"
printf 'OMBRE_EXPECTED_CONTAINER_ID=%s\nOMBRE_EXPECTED_IMAGE_ID=%s\n' "$old_id" "$old_image" >"$anchor_next"
chmod 600 "$anchor_next"
mv "$anchor_next" "$OMBRE_DEPLOYMENT_ANCHOR_FILE"
trap - EXIT INT TERM HUP
printf 'ROLLBACK_OK container=%s retained_container=%s retained_source=%s data_untouched=true\n' "$old_id" "$saved_name" "$saved_source"
