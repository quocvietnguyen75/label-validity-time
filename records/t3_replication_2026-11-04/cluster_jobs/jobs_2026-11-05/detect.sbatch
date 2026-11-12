#!/bin/bash
#SBATCH -J xfc7-det
#SBATCH -p main
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 2-00:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3, step S2: the frozen raw detector pass (careport.xfc7_t2_detect, batch 4, --per-process 3, as T2) over the
# logs in <list.txt> under <root>, writing raw tables to <out_dir>. Reads images only.
# Usage: sbatch detect.sbatch <list.txt> <root> <out_dir>
set -euo pipefail
X=/mnt/data/vietnq/xfc7
LIST="$1"; ROOT="$2"; OUT="$3"
. "$X/env/bin/activate"
export TORCH_HOME="$X/torch_home" PYTHONPATH="$X/code/src"
hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; date -u
mapfile -t logs < <(sed "s#^#$ROOT/#" "$LIST")
cd "$X/code"
python -X utf8 -m careport.xfc7_t2_detect "$OUT" 4 --per-process 3 "${logs[@]}"
date -u; echo DET_DONE
