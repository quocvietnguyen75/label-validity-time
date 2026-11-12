#!/bin/bash
#SBATCH -J xfc7-env
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3: build the cluster environment, check S3 access from a compute node, run the bundled unit tests.
# numpy, pandas and pyarrow pinned to the T2 environment (FREEZE_RECORD_XF-C7_T2_v1_0.md); torch and torchvision are
# the cluster's CUDA build (the parity check in SAMPLE_PIN addendum 4 tests the difference).
set -euo pipefail
X=/mnt/data/vietnq/xfc7
hostname; date -u
python3 -m venv "$X/env"
. "$X/env/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy==2.2.6 pandas==2.3.3 pyarrow==21.0.0 awscli
python -m pip install --quiet "torch==2.11.*" "torchvision==0.26.*" --index-url https://download.pytorch.org/whl/cu128 \
  || python -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip freeze > "$X/env_freeze.txt"
python -c "import sys, numpy, pandas, pyarrow, torch, torchvision; print(sys.version.split()[0], 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'pyarrow', pyarrow.__version__, 'torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda build', torch.version.cuda)"
curl -sS -o /dev/null -w "s3 from compute node: http %{http_code}\n" "https://argoverse.s3.amazonaws.com/?list-type=2&prefix=datasets/av2/sensor/val/&max-keys=1"
aws --version
cd "$X/code" && sha256sum -c ../BUNDLE_SHA256SUMS
cd "$X/code" && PYTHONPATH=src python -m unittest discover -s tests -v 2>&1 | tail -25
date -u; echo ENV_DONE
