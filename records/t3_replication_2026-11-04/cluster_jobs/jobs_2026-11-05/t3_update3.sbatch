#!/bin/bash
#SBATCH -J xfc7-upd3
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3 (SAMPLE_PIN addendum 9): check the updated code311 bundle (xfc7_t3_run.py and t3_design_table.py: the
# margin-flag-excluded run's gate uses that run's own predicted theta-bar), run the unit tests and the design-table
# script's tests, then write the all-H100 T2 50-log design tables again under new names (_v2). Weights and label-side
# predictions only: no moving theta-hat.
set -uo pipefail
X=/mnt/data/vietnq/xfc7
. "$X/env311/bin/activate"
hostname; date -u
cd "$X/code311" || exit 2
sha256sum -c --quiet BUNDLE311_SHA256SUMS && echo BUNDLE_OK || { echo BUNDLE_MISMATCH; exit 3; }
FULL="$X/jobs/logs/tests311_${SLURM_JOB_ID}.txt"
PYTHONPATH=src python -m unittest discover -s tests > "$FULL" 2>&1; rc=$?
echo "unittest exit=$rc"; grep -E "^(ERROR|FAIL):|Ran [0-9]+ tests|^OK|^FAILED" "$FULL"
DT="$X/jobs/logs/tests_design_${SLURM_JOB_ID}.txt"
(cd scripts && PYTHONPATH=../src python -m unittest test_t3_design_table > "$DT" 2>&1); rc2=$?
echo "design-table tests exit=$rc2"; grep -E "^(ERROR|FAIL):|Ran [0-9]+ tests|^OK|^FAILED" "$DT"
{ [ $rc -eq 0 ] && [ $rc2 -eq 0 ]; } || exit 4
export PYTHONPATH="$X/code311/src" AV2_SENSOR_ROOT="$X"
for s in 050 030; do
  out="$X/derived/t2h100_static/design_s${s}_v2.json"
  [ -e "$out" ] && { echo "$out exists; stop"; exit 5; }
  python -X utf8 scripts/t3_design_table.py "$X/derived/t2h100_static/S3_summary_s$s.json" \
    "$X/derived/t2h100_static/s$s/moving_analysis_set_t2.parquet" "$out" || exit 6
done
sha256sum "$X"/derived/t2h100_static/design_s0*_v2.json
date -u; echo UPD3_DONE
