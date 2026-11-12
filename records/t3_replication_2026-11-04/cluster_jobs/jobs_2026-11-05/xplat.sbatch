#!/bin/bash
#SBATCH -J xfc7-xplat
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3, SAMPLE_PIN addendum 7, the cross-platform check: (1) the frozen static step on the cluster with T2's PC
# inputs against T2's frozen PC outputs (xplat_check.py); (2) xfc7_t3_run.analysis on T2's frozen tables against T2's
# verdicts and R4's intervals (t3_repro_check.py). Parked vehicles and known T2 outcomes only.
set -uo pipefail
X=/mnt/data/vietnq/xfc7
. "$X/env311/bin/activate"
export PYTHONPATH="$X/code311/src"
F="$X/data/t2_pc_frozen"; OUT="$X/derived/xplat_t2"
mkdir -p "$OUT"
hostname; date -u
(cd "$F" && sha256sum -c --quiet SHA256SUMS) || { echo "T2 frozen inputs: hash check failed; stop"; exit 1; }
echo "T2 frozen inputs: 60 files verified"
cd "$X/code311"
python -X utf8 scripts/run_measured.py careport.xfc7_t2_static "$F/t2_labels_all50_label_side.parquet" "$F/t2_det_raw" 0.5 0.7 \
    "$OUT/S3_summary_xplat_s050.json" "$OUT/static_s050" av2 far40 > "$OUT/static_stdout.txt" 2> "$OUT/static_stderr.txt"
echo "static exit=$?"; grep MEASURED "$OUT/static_stderr.txt"
python -X utf8 scripts/xplat_check.py "$F/S3_summary_all50_s050.json" "$OUT/S3_summary_xplat_s050.json" "$F/all50_s050" \
    "$OUT/static_s050" "$OUT/xplat_check_1.json" > /dev/null
echo "check1 exit=$?"
python -X utf8 scripts/t3_repro_check.py "$F/S3_summary_all50_s050.json" "$F/T2_RESULTS_2026-11-01.json" "$F/r4_results.json" \
    "$F/all50_s050/moving_analysis_set_t2.parquet" "$OUT/xplat_check_2.json" > /dev/null
echo "check2 exit=$?"
python -c "import json;a=json.load(open('$OUT/xplat_check_1.json'));b=json.load(open('$OUT/xplat_check_2.json'));print('check1 pass',a['pass'],a['summary']['failures'],{t:v['pass_'] for t,v in a['tables'].items()});print('check2 pass',b['pass'],b['max_abs_diff_from_r4_ms'])"
date -u; echo XPLAT_DONE
