#!/bin/bash
#SBATCH -J xfc7-dl
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 3-00:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3: download AV2 Sensor logs listed as '<split>/<id>' to <root>/<split>/<id>, without the stereo cameras.
# Public bucket, no account; CC BY-NC-SA 4.0. Resumable; each log retried up to 5 times; per-log key/size/ETag listing.
# Usage: sbatch download.sbatch <list.txt> <root>
set -uo pipefail
X=/mnt/data/vietnq/xfc7
LIST="$1"; ROOT="$2"
. "$X/env/bin/activate"
MAN="$ROOT/_manifest"; mkdir -p "$MAN"
export AWS_CONFIG_FILE="$MAN/aws_config"
printf '[default]\ns3 =\n  max_concurrent_requests = 16\n' > "$AWS_CONFIG_FILE"
total=$(grep -c . "$LIST"); n=0
hostname; date -u
while read -r entry; do
  [ -z "$entry" ] && continue
  split="${entry%%/*}"; id="${entry#*/}"; n=$((n + 1))
  echo "$(date -u +%FT%TZ) [$n/$total] $split/$id"
  [ -s "$MAN/etags_${split}_$id.tsv" ] || aws s3api list-objects-v2 --no-sign-request --bucket argoverse \
      --prefix "datasets/av2/sensor/$split/$id/" --query 'Contents[].[Key,Size,ETag]' --output text > "$MAN/etags_${split}_$id.tsv"
  ok=0
  for attempt in 1 2 3 4 5; do
    if aws s3 sync --no-sign-request --only-show-errors --exclude "sensors/cameras/stereo_*" \
        "s3://argoverse/datasets/av2/sensor/$split/$id/" "$ROOT/$split/$id/"; then ok=1; break; fi
    echo "$(date -u +%FT%TZ) retry $split/$id (attempt $attempt failed)"
  done
  [ "$ok" = 1 ] || { echo "$(date -u +%FT%TZ) FAILED: $split/$id"; exit 1; }
done < "$LIST"
echo "$(date -u +%FT%TZ) done: $n logs"
