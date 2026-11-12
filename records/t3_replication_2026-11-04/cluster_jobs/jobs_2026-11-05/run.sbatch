#!/bin/bash
#SBATCH -J xfc7-t3run
#SBATCH -p main
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 3-00:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3: the single post-freeze run (T3 protocol section 15). Submit ONLY after the user's freeze word, with the
# frozen protocol and the config uploaded. Their SHA-256 are given on the command line and checked by the runner,
# which also refuses an existing output. CPU only; S-d re-runs the static step five times (about 60 GB at its peak).
# Usage: sbatch run.sbatch <config.json> <config_sha256> <protocol.md> <protocol_sha256> <out.json>
set -euo pipefail
X=/mnt/data/vietnq/xfc7
[ $# -eq 5 ] || { echo "usage: run.sbatch <config> <config_sha256> <protocol> <protocol_sha256> <out.json>"; exit 64; }
. "$X/env311/bin/activate"
export PYTHONPATH="$X/code311/src" AV2_SENSOR_ROOT="$X"
hostname; date -u
cd "$X/code311"
sha256sum -c --quiet BUNDLE311_SHA256SUMS || { echo BUNDLE_MISMATCH; exit 3; }
echo BUNDLE_OK
FULL="$X/jobs/logs/tests311_${SLURM_JOB_ID}.txt"
if ! python -m unittest discover -s tests > "$FULL" 2>&1; then
  echo "unit tests failed; no run"; grep -E "^(ERROR|FAIL):|Ran [0-9]+ tests|^FAILED" "$FULL"; exit 4
fi
grep -E "Ran [0-9]+ tests|^OK" "$FULL"
python -X utf8 scripts/run_measured.py careport.xfc7_t3_run "$1" "$2" "$3" "$4" "$5"
sha256sum "$5"
date -u; echo T3RUN_DONE
