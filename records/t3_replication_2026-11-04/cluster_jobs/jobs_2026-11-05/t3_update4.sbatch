#!/bin/bash
#SBATCH -J xfc7-upd4
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3 (SAMPLE_PIN addendum 10): check the code311 bundle after update 4 (the runner with the audit's fixes, its
# tests, build_t3_run_config.py) and run the unit tests and the design-table script's tests. No data are read.
set -uo pipefail
X=/mnt/data/vietnq/xfc7
. "$X/env311/bin/activate"
hostname; date -u
cd "$X/code311" || exit 2
sha256sum -c --quiet BUNDLE311_SHA256SUMS || { echo BUNDLE_MISMATCH; exit 3; }
echo BUNDLE_OK
FULL="$X/jobs/logs/tests311_${SLURM_JOB_ID}.txt"
PYTHONPATH=src python -m unittest discover -s tests > "$FULL" 2>&1; rc=$?
echo "unittest exit=$rc"; grep -E "^(ERROR|FAIL):|Ran [0-9]+ tests|^OK|^FAILED" "$FULL"
DT="$X/jobs/logs/tests_design_${SLURM_JOB_ID}.txt"
(cd scripts && PYTHONPATH=../src python -m unittest test_t3_design_table > "$DT" 2>&1); rc2=$?
echo "design-table tests exit=$rc2"; grep -E "^(ERROR|FAIL):|Ran [0-9]+ tests|^OK|^FAILED" "$DT"
date -u; echo UPD4_DONE
{ [ $rc -eq 0 ] && [ $rc2 -eq 0 ]; } || exit 4
