#!/usr/bin/env bash
# XF-C7 tier 2: download the 40 fixed-seed AV2 Sensor val logs (av2_val_draw_seed20261101.txt) without the stereo
# cameras (not used). Public bucket, no account; CC BY-NC-SA 4.0. Resumable (aws s3 sync skips files present with the
# same size); each log retried up to 5 times. Saves per-log ETag listings for verification.
# Usage (Git Bash, from D:/paper):
#   set -a; . ./.env.xfc7.local; set +a
#   bash newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/tier2_2026-11-01/download_av2_val_draw.sh
set -uo pipefail
: "${AV2_SENSOR_ROOT:?set AV2_SENSOR_ROOT (see .env.xfc7.local)}"
AWS="${AWS:-aws}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="s3://argoverse/datasets/av2/sensor/val"
MAN="$AV2_SENSOR_ROOT/_manifest/tier2_val_draw"
mkdir -p "$MAN"
CFG="$MAN/aws_config"
printf '[default]\ns3 =\n  max_concurrent_requests = 32\n' > "$CFG"
export AWS_CONFIG_FILE="$CFG"
n=0
while read -r line; do
  id="$(printf '%s' "$line" | tr -cd '0-9a-f-')"   # keep only id characters (the draw file has CRLF endings)
  [ -z "$id" ] && continue
  n=$((n + 1))
  echo "$(date -u +%FT%TZ) [$n/40] listing $id"
  "$AWS" s3api list-objects-v2 --no-sign-request --bucket argoverse --prefix "datasets/av2/sensor/val/$id/" \
    --query 'Contents[].[Key,Size,ETag]' --output text > "$MAN/etags_val_$id.tsv" || echo "listing failed: $id"
  ok=0
  for attempt in 1 2 3 4 5; do
    if "$AWS" s3 sync --no-sign-request --only-show-errors --exclude "sensors/cameras/stereo_*" \
        "$SRC/$id/" "$AV2_SENSOR_ROOT/val/$id/"; then ok=1; break; fi
    echo "$(date -u +%FT%TZ) retry $id (attempt $attempt failed)"
  done
  [ "$ok" = 1 ] || { echo "$(date -u +%FT%TZ) FAILED: $id"; exit 1; }
done < "$HERE/av2_val_draw_seed20261101.txt"
echo "$(date -u +%FT%TZ) done: $n logs"
