#!/bin/bash
#SBATCH -J xfc7-t311
#SBATCH -p main
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3 (SAMPLE_PIN addendum 7): pin pandas' helper packages in env311 to the PC's versions, check the code311
# bundle, and run its unit tests with the full output kept in jobs/logs/tests311_<jobid>.txt.
set -uo pipefail
X=/mnt/data/vietnq/xfc7
. "$X/env311/bin/activate"
hostname; date -u
python -m pip install --no-input --quiet python-dateutil==2.9.0.post0 pytz==2025.2 tzdata==2025.2 six==1.17.0
python -m pip freeze
cd "$X/code311" && sha256sum -c --quiet BUNDLE311_SHA256SUMS && echo BUNDLE_OK
FULL="$X/jobs/logs/tests311_${SLURM_JOB_ID}.txt"
PYTHONPATH=src python -m unittest discover -s tests > "$FULL" 2>&1
rc=$?
echo "unittest exit=$rc"
grep -E "^(ERROR|FAIL):|Ran [0-9]+ tests|^OK|^FAILED" "$FULL"
date -u; echo T311_DONE
exit $rc
