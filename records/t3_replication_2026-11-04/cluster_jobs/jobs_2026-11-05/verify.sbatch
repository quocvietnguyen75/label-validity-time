#!/bin/bash
#SBATCH -J xfc7-ver
#SBATCH -p main
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 1-00:00:00
#SBATCH -o /mnt/data/vietnq/xfc7/jobs/logs/%x-%j.out
# XF-C7 T3, G1 analogue: verify the downloaded logs of <list.txt> under <root> against the bucket listings that
# download.sbatch saved (t3_follow.py verify: every non-stereo object present at its listed size, and MD5 equal to
# every single-part ETag). One marker per log in <marker_dir>, totals in <marker_dir>/_DONE.json.
# Exits 0 only if every log verifies, so jobs submitted with afterok on this one never see an unverified log.
# Usage: sbatch --dependency=afterany:<download job ids> verify.sbatch <list.txt> <root> <marker_dir> <download_job_ids>
set -uo pipefail
X=/mnt/data/vietnq/xfc7
LIST="$1"; ROOT="$2"; MARK="$3"; IDS="$4"
. "$X/env/bin/activate"
hostname; date -u
python -X utf8 "$X/jobs/t3_follow.py" verify "$ROOT" "$LIST" "$MARK" "$IDS" 1
rc=$?
date -u; echo "VERIFY_DONE rc=$rc"
exit $rc
