#!/bin/bash
#SBATCH -J xfc7-comb
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3 (SAMPLE_PIN addendum 7): the frozen label-side combine (careport.xfc7_t2_av2_labels combine) over the per-log
# tables in <label_dir>, writing <label_dir>_label_side.parquet and <summary.json>; then S0 (frozen s0_av2_timing.py)
# on the same per-log tables. Label side only: no image, no detector box.
# Usage: sbatch combine.sbatch <label_dir> <summary.json> <s0.json>
set -euo pipefail
X=/mnt/data/vietnq/xfc7
LDIR="$1"; SUMM="$2"; S0="$3"
. "$X/env311/bin/activate"
export PYTHONPATH="$X/code311/src" AV2_SENSOR_ROOT="$X"
hostname; date -u
cd "$X/code311"
python -X utf8 scripts/run_measured.py careport.xfc7_t2_av2_labels combine "$LDIR" "$SUMM"
python -X utf8 scripts/s0_av2_timing.py "$LDIR" "$S0"
sha256sum "${LDIR%/}_label_side.parquet" "$SUMM" "$S0"
date -u; echo COMBINE_DONE
