#!/bin/bash
#SBATCH -J xfc7-env311
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3 (SAMPLE_PIN addendum 7): CPython 3.11.9 from python-build-standalone release 20240814, verified against
# that release's SHA256SUMS, and venv env311 with numpy 2.2.6, pandas 2.3.3, pyarrow 21.0.0 and scipy 1.16.3 (the
# PC's versions). Then the code311 bundle hash check and its unit tests.
set -euo pipefail
X=/mnt/data/vietnq/xfc7
REL=20240814
PY="cpython-3.11.9+$REL-x86_64-unknown-linux-gnu-install_only.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/$REL"
hostname; date -u
mkdir -p "$X/py311_dl"; cd "$X/py311_dl"
curl -sSfL --retry 5 -o SHA256SUMS "$URL/SHA256SUMS"
curl -sSfL --retry 5 -o "$PY" "$URL/${PY//+/%2B}"
grep -F "  $PY" SHA256SUMS > want.txt
cat want.txt
sha256sum -c want.txt
rm -rf "$X/py311" "$X/env311"
mkdir -p "$X/py311"; tar -xzf "$PY" -C "$X/py311"
"$X/py311/python/bin/python3" -m venv "$X/env311"
. "$X/env311/bin/activate"
python -m pip install --no-input --quiet --upgrade pip
python -m pip install --no-input --quiet numpy==2.2.6 pandas==2.3.3 pyarrow==21.0.0 scipy==1.16.3
python -c "import sys,numpy,pandas,pyarrow,scipy;print(sys.version);print('numpy',numpy.__version__,'pandas',pandas.__version__,'pyarrow',pyarrow.__version__,'scipy',scipy.__version__)"
python -m pip freeze
cd "$X/code311"
sha256sum -c BUNDLE311_SHA256SUMS
PYTHONPATH=src python -m unittest discover -s tests 2>&1 | tail -6
date -u; echo ENV311_DONE
