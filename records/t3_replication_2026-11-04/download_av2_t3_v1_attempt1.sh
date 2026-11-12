#!/usr/bin/env bash
# XF-C7 T3 (AV2 replication): download the fresh sample fixed in av2_t3_sample.txt (108 val + 300 train logs) to
# $AV2_T3_ROOT, without the stereo cameras (not used). Public bucket, no account; CC BY-NC-SA 4.0. Resumable (aws s3
# sync skips files already present with the same size); each log is retried up to 5 times. Saves a per-log listing of
# keys, sizes and ETags for verification (verify_av2_t3.py).
# Usage (Git Bash):  AV2_T3_ROOT=E:/datasets/argoverse2_t3 bash download_av2_t3.sh
set -uo pipefail
: "${AV2_T3_ROOT:?set AV2_T3_ROOT, e.g. E:/datasets/argoverse2_t3}"
AWS="${AWS:-aws}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MAN="$AV2_T3_ROOT/_manifest/t3"
mkdir -p "$MAN"
CFG="$MAN/aws_config"
printf '[default]\ns3 =\n  max_concurrent_requests = 32\n' > "$CFG"
export AWS_CONFIG_FILE="$CFG"
total=$(grep -c . "$HERE/av2_t3_sample.txt")
n=0
while read -r line; do
  entry="$(printf '%s' "$line" | tr -cd 'a-z0-9/-')"
  [ -z "$entry" ] && continue
  split="${entry%%/*}"; id="${entry#*/}"
  case "$split" in val|train) ;; *) echo "bad split in: $line"; exit 1 ;; esac
  n=$((n + 1))
  echo "$(date -u +%FT%TZ) [$n/$total] $split/$id"
  if [ ! -s "$MAN/etags_${split}_$id.tsv" ]; then
    "$AWS" s3api list-objects-v2 --no-sign-request --bucket argoverse --prefix "datasets/av2/sensor/$split/$id/" \
      --query 'Contents[].[Key,Size,ETag]' --output text > "$MAN/etags_${split}_$id.tsv" || echo "listing failed: $split/$id"
  fi
  ok=0
  for attempt in 1 2 3 4 5; do
    if "$AWS" s3 sync --no-sign-request --only-show-errors --exclude "sensors/cameras/stereo_*" \
        "s3://argoverse/datasets/av2/sensor/$split/$id/" "$AV2_T3_ROOT/$split/$id/"; then ok=1; break; fi
    echo "$(date -u +%FT%TZ) retry $split/$id (attempt $attempt failed)"
  done
  [ "$ok" = 1 ] || { echo "$(date -u +%FT%TZ) FAILED: $split/$id"; exit 1; }
done < "$HERE/av2_t3_sample.txt"
echo "$(date -u +%FT%TZ) done: $n logs"
