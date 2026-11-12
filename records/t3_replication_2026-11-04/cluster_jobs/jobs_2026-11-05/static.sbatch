#!/bin/bash
#SBATCH -J xfc7-stat
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3 (SAMPLE_PIN addendum 7): the frozen static step S1/S3/S4 (careport.xfc7_t2_static, rule far40) on
# <label_side.parquet> with the raw tables in <raw_dir> at <score> (class-agnostic NMS 0.7), then the weights-only
# design table (t3_design_table.py, keys and weights only). Parked vehicles, weights and label-side predictions only:
# no moving theta-hat, residual or displacement.
# Usage: sbatch static.sbatch <label_side.parquet> <raw_dir> <score> <summary.json> <out_dir> <design.json>
set -euo pipefail
X=/mnt/data/vietnq/xfc7
LAB="$1"; RAW="$2"; SC="$3"; SUMM="$4"; OUT="$5"; DES="$6"
. "$X/env311/bin/activate"
export PYTHONPATH="$X/code311/src" AV2_SENSOR_ROOT="$X"
hostname; date -u
mkdir -p "$(dirname "$SUMM")" "$(dirname "$DES")"
cd "$X/code311"
python -X utf8 scripts/run_measured.py careport.xfc7_t2_static "$LAB" "$RAW" "$SC" 0.7 "$SUMM" "$OUT" av2 far40 > "${SUMM%.json}_stdout.txt"
python -X utf8 scripts/t3_design_table.py "$SUMM" "$OUT/moving_analysis_set_t2.parquet" "$DES"
sha256sum "$SUMM" "$DES" "$OUT"/*.parquet
date -u; echo STATIC_DONE
