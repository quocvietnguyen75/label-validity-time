#!/bin/bash
#SBATCH -J xfc7-annchk
#SBATCH -p main
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 0-02:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3: label-side data check over the 408 logs. Annotation timestamps against the LiDAR sweeps on disk and in
# the saved bucket listing (jobs/t3_ann_sweep_check.py). Reads no image, no detection, no LiDAR point, no outcome.
# Usage: sbatch annchk.sbatch <list.txt> <root> <out.json>
set -euo pipefail
X=/mnt/data/vietnq/xfc7
LIST="$1"; ROOT="$2"; OUT="$3"
. "$X/env/bin/activate"
hostname; date -u
sha256sum "$X/jobs/t3_ann_sweep_check.py"
python -I -X utf8 "$X/jobs/t3_ann_sweep_check.py" "$ROOT" "$LIST" "$OUT"
sha256sum "$OUT"
date -u; echo ANNCHK_DONE
