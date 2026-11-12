#!/usr/bin/env bash
# XF-C7 secondary S3: the 10 Argoverse 2 Sensor logs used by P01 (Justo Miro et al., WACV 2026).
# Log ids from the README of github.com/alexandre-justo-miro/annotation-correction-3D-boxes (read 2026-10-18).
# Public AWS bucket, no account; licence CC BY-NC-SA 4.0. Listed size 2026-10-18: 12,103,584,153 bytes, 30,359 objects.
# Layout <root>/{train,val}/<log_id>/ matches P01's loader (its --base-directory flag points at the parent).
# Resumable: aws s3 sync skips files already present with the same size. Each log is retried up to 5 times
# (a connection reset stopped the first run on 2026-10-18). The logs are ~3,000 small files each; to speed up
# transfers, point AWS_CONFIG_FILE at a config with a higher s3 max_concurrent_requests for the run.
# Usage (Git Bash, from the workspace root):
#   set -a; . ./.env.xfc7.local; set +a
#   bash newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/pilot/download_av2.sh
set -euo pipefail
: "${AV2_SENSOR_ROOT:?set AV2_SENSOR_ROOT (see .env.xfc7.local)}"

SRC="s3://argoverse/datasets/av2/sensor"
LOGS="train/3ca11a5e-50b2-3cc3-af7a-ce7ab02b9954
train/58d01358-5927-36fa-9e11-d18d1dc1f4f0
train/6b14d7c0-20f9-390b-af38-507a5de5998c
train/81d2b40a-c579-3e9c-b520-bee26cda947d
train/c556f8e0-a001-3586-b2cf-d3256685c39f
train/c780d53a-2d37-3cd8-9e89-530966aef53e
train/f648b945-6c70-3105-bd23-9502894e37d4
train/fd5c6932-2ee2-3cfb-9bdc-0b30bfb33a91
val/9a448a80-0e9a-3bf0-90f3-21750dfef55a
val/d5d6f11c-3026-3e0e-9d67-c111233e22de"

MAN="$AV2_SENSOR_ROOT/_manifest"
mkdir -p "$MAN"
for p in $LOGS; do
  tag="${p//\//_}"
  echo "$(date -u +%FT%TZ) listing $p"
  aws s3 ls --no-sign-request --recursive "$SRC/$p/" > "$MAN/listing_${tag}.txt"
  for attempt in 1 2 3 4 5; do
    echo "$(date -u +%FT%TZ) syncing $p (attempt $attempt)"
    if aws s3 sync --no-sign-request --only-show-errors "$SRC/$p/" "$AV2_SENSOR_ROOT/$p/"; then
      break
    fi
    if [ "$attempt" = 5 ]; then
      echo "$(date -u +%FT%TZ) FAILED: $p after 5 attempts"
      exit 1
    fi
  done
done
echo "$(date -u +%FT%TZ) done: $(echo "$LOGS" | wc -l) logs"
