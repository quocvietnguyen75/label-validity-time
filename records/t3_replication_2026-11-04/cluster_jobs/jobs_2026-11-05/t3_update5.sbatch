#!/bin/bash
#SBATCH -J xfc7-upd5
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3 (SAMPLE_PIN addendum 12): check the code311 bundle after update 5 (build_t3_run_config.py with blocks named
# for the analysed counts, three more provenance pins and no overwrite; its new tests), run the unit tests, the
# design-table tests and the builder's tests, then build the run config once. The builder reads summaries, hashes and
# log ids only; no moving set is opened.
set -uo pipefail
X=/mnt/data/vietnq/xfc7
. "$X/env311/bin/activate"
export AV2_SENSOR_ROOT="$X"
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
BT="$X/jobs/logs/tests_builder_${SLURM_JOB_ID}.txt"
(cd scripts && python -m unittest test_build_t3_run_config > "$BT" 2>&1); rc3=$?
echo "builder tests exit=$rc3"; grep -E "^(ERROR|FAIL):|Ran [0-9]+ tests|^OK|^FAILED" "$BT"
{ [ $rc -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ]; } || { echo "tests failed; no config"; exit 4; }
CFG="$X/config/xfc7_t3_v1_0.json"
[ -e "$CFG" ] && { echo "config exists; stop"; exit 5; }
mkdir -p "$X/config"
python -X utf8 scripts/build_t3_run_config.py "$X/lists/t3_408.txt" PILOT_PROTOCOL_XF-C7_T3_v1_0.md "$CFG" || exit 6
sha256sum "$CFG"
date -u; echo UPD5_DONE
