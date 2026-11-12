#!/usr/bin/env bash
# XF-C7 T3, step S2 (raw detector pass, images only), run on each log as soon as its download has finished.
# Instrument unchanged from T2: careport.xfc7_t2_detect (sha256 55f489b4...), batch 4, --per-process 3.
# A log counts as complete once download_av2_t3.sh has started the next log, or has printed "done:".
# The sample file's order is the download order. Stops after the download ends and every complete log has a raw table.
# Usage (Git Bash, from D:/paper):  bash <this script>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${AV2_T3_ROOT:-E:/datasets/argoverse2_t3}"
OUT="$ROOT/derived/t3_det/raw"
DLLOG="$HERE/download_av2_t3_2026-11-04.log"
SAMPLE="$HERE/av2_t3_sample.txt"
export PYTHONPATH=src
STOP="$ROOT/_manifest/t3/STOP"
while true; do
  [ -e "$STOP" ] && { echo "$(date -u +%FT%TZ) STOPPED by watchdog"; exit 3; }
  started=$(grep -cE '^[0-9TZ:-]+ \[[0-9]+/[0-9]+\] ' "$DLLOG")
  finished=0
  if grep -q ' done: ' "$DLLOG"; then complete=$started; finished=1
  else complete=$((started - 1)); grep -q 'FAILED' "$DLLOG" && finished=1; fi
  if [ "$complete" -gt 0 ]; then
    mapfile -t logs < <(head -n "$complete" "$SAMPLE" | tr -d '\r' | sed "s#^#$ROOT/#")
    python -X utf8 -m careport.xfc7_t2_detect "$OUT" 4 --per-process 3 "${logs[@]}"
  fi
  if [ "$finished" = 1 ]; then
    echo "$(date -u +%FT%TZ) download ended; detector covered $complete logs"
    break
  fi
  sleep 300
done
