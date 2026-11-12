#!/bin/bash
#SBATCH -J xfc7-lab
#SBATCH -p main
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 2-00:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3: the frozen AV2 label side (careport.xfc7_t2_av2_labels run) over the logs in <list.txt> under <root>,
# writing per-log tables to <out_dir>. Reads annotations, poses, calibration and LiDAR; never an image.
# Usage: sbatch labels.sbatch <list.txt> <root> <out_dir>
set -euo pipefail
X=/mnt/data/vietnq/xfc7
LIST="$1"; ROOT="$2"; OUT="$3"
. "$X/env/bin/activate"
export PYTHONPATH="$X/code/src"
hostname; date -u
mapfile -t logs < <(sed "s#^#$ROOT/#" "$LIST")
mkdir -p "$OUT"
cd "$X/code"
python -X utf8 -m careport.xfc7_t2_av2_labels run "$OUT" "${logs[@]}"
date -u; echo LAB_DONE
