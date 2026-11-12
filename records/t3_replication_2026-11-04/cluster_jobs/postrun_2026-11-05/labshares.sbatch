#!/bin/bash
#SBATCH -J xfc7-labshares
#SBATCH -p main
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3 (SAMPLE_PIN addendum 15): seam-rule and fallback shares of the 407-log and pooled 457-log label sides,
# by T2's definition (T3 protocol section 16, T2 section 5). Label-side columns only; written after the single run.
set -uo pipefail
X=/mnt/data/vietnq/xfc7
. "$X/env311/bin/activate"
hostname; date -u
cd "$X/postrun" || exit 2
sha256sum -c --quiet POSTRUN_SHA256SUMS || { echo POSTRUN_MISMATCH; exit 3; }
echo POSTRUN_OK
OUT="$X/derived/postrun"
mkdir -p "$OUT"
python -X utf8 t3_label_shares.py "$X/derived/t3_labels_label_side.parquet" \
  5336fa031a14942adaa82e7d4667c724b64a64cf697505807e7de189ca8c0f83 "$OUT/t3_label_shares_407.json" || exit 4
python -X utf8 t3_label_shares.py "$X/derived/pooled457_labels_label_side.parquet" \
  e8ee3e2f0f1603526796db1555ff7773c7f90b8df53b68671f83eb22e58e3d01 "$OUT/pooled457_label_shares.json" || exit 5
sha256sum "$OUT/t3_label_shares_407.json" "$OUT/pooled457_label_shares.json"
date -u; echo LABSHARES_DONE
