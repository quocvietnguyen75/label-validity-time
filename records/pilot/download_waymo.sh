#!/usr/bin/env bash
# XF-C7 primary data: Waymo Open Dataset Perception v2.0.1, validation split (202 segments).
# Layout in the bucket and on disk: validation/<component>/<segment_context_name>.parquet
# (component tags from the v2 tutorial and GitHub issue #688; bucket existence confirmed 2026-10-18).
# Access needs a Google account registered at waymo.com/open with the licence accepted, then `gcloud auth login`.
# Licence: non-commercial; copies only to registered users; never commit these files.
#
# Usage (Git Bash, from the workspace root):
#   set -a; . ./.env.xfc7.local; set +a
#   bash .../pilot/download_waymo.sh manifest   # 1. listings with sizes and hashes; stops on any billing request
#   bash .../pilot/download_waymo.sh tables     # 2. small tables, about 1 GB
#   bash .../pilot/download_waymo.sh images     # 3. camera_image, about 68-78 GB
#   bash .../pilot/download_waymo.sh verify     # check every table and image file against the listing (size + MD5)
#   bash .../pilot/download_waymo.sh subset     # 4. after the protocol freeze only: lidar + lidar_pose, 40 segments
# Every step is resumable; rerunning skips files already present.
set -euo pipefail
: "${WAYMO_V2_ROOT:?set WAYMO_V2_ROOT (see .env.xfc7.local)}"
: "${GCLOUD:?set GCLOUD (see .env.xfc7.local)}"

BUCKET="gs://waymo_open_dataset_v_2_0_1/validation"
TABLES="camera_box lidar_box lidar_camera_synced_box projected_lidar_box camera_to_lidar_box_association vehicle_pose camera_calibration lidar_calibration stats camera_segmentation"
IMAGES="camera_image"
SUBSET_TAGS="lidar lidar_pose"
SUBSET_SEED=20261018
SUBSET_N=40
MAN="$WAYMO_V2_ROOT/_manifest"
stamp() { date -u +%FT%TZ; }

# Copies only the data files and _metadata. The bucket also holds a folder-marker object per component,
# which rsync tried to write over the local folder on Windows ("Access is denied", 2026-10-20).
# -n skips files already present, so a rerun resumes; `verify` then checks size and MD5 against the listing.
fetch_tags() {
  for t in "$@"; do
    echo "$(stamp) copy $t"
    mkdir -p "$WAYMO_V2_ROOT/validation/$t"
    "$GCLOUD" storage cp -n "$BUCKET/$t/*.parquet" "$BUCKET/$t/_metadata" "$WAYMO_V2_ROOT/validation/$t/"
  done
}
HERE="$(cd "$(dirname "$0")" && pwd)"

case "${1:-}" in
  manifest)
    mkdir -p "$MAN"
    for t in $TABLES $IMAGES $SUBSET_TAGS; do
      echo "$(stamp) listing $t"
      if ! "$GCLOUD" storage objects list "$BUCKET/$t/*" --format=json > "$MAN/listing_$t.json" 2> "$MAN/listing_$t.err"; then
        if grep -qiE "billing|requester|user.?project" "$MAN/listing_$t.err"; then
          echo "STOP: the bucket asks for a billing project. Nothing was downloaded. Report this before going on."
          exit 2
        fi
        cat "$MAN/listing_$t.err"
        exit 1
      fi
    done
    python -I - "$MAN" "$TABLES" "$IMAGES" "$SUBSET_TAGS" "$SUBSET_N" <<'PY'
import json, os, sys
man, tables, images, subset, n_sub = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4].split(), int(sys.argv[5])
def size(o):
    v = o.get("size")
    if v is None:
        v = (o.get("metadata") or {}).get("size", 0)
    return int(v)
rows, plan, later = [], 0, 0
for t in tables + images + subset:
    objs = json.load(open(os.path.join(man, f"listing_{t}.json"), encoding="utf-8"))
    n, b = len(objs), sum(size(o) for o in objs)
    if t in subset:
        b_sub = b * n_sub / n if n else 0
        later += b_sub
        rows.append(f"{t}\t{n}\t{b}\tlater: about {b_sub/1e9:.2f} GB for {n_sub} segments")
    else:
        plan += b
        rows.append(f"{t}\t{n}\t{b}")
rows.append(f"PLANNED NOW (tables + images)\t\t{plan}\t{plan/1e9:.2f} GB")
rows.append(f"LATER (subset estimate)\t\t{int(later)}\t{later/1e9:.2f} GB")
out = os.path.join(man, "size_summary.tsv")
open(out, "w", encoding="utf-8").write("component\tfiles\tbytes\tnote\n" + "\n".join(rows) + "\n")
print(open(out, encoding="utf-8").read())
PY
    ;;
  tables) fetch_tags $TABLES ;;
  images) fetch_tags $IMAGES ;;
  verify) python -I "$HERE/verify_listing.py" "$WAYMO_V2_ROOT" $TABLES $IMAGES ;;
  subset)
    [ "${XFC7_PROTOCOL_FROZEN:-0}" = 1 ] || { echo "The lidar subset waits for the protocol freeze (set XFC7_PROTOCOL_FROZEN=1 after freezing)."; exit 3; }
    python -I - "$MAN/listing_camera_image.json" "$MAN/subset_${SUBSET_N}.txt" "$SUBSET_SEED" "$SUBSET_N" <<'PY'
import json, os, random, sys
objs = json.load(open(sys.argv[1], encoding="utf-8"))
def name(o):
    u = o.get("name") or o.get("url") or o.get("storage_url") or ""
    b = os.path.basename(u)
    return b[:-8] if b.endswith(".parquet") else b
names = sorted(name(o) for o in objs if str(o.get("name", "")).endswith(".parquet"))
assert len(names) == 202, f"expected 202 validation segments, found {len(names)}"
pick = sorted(random.Random(int(sys.argv[3])).sample(names, int(sys.argv[4])))
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write("\n".join(pick) + "\n")  # LF only (the 2026-10-20 list was written with CRLF)
print(f"{len(pick)} of {len(names)} segments written to {sys.argv[2]} (seed {sys.argv[3]}, Python {sys.version.split()[0]})")
PY
    for t in $SUBSET_TAGS; do
      mkdir -p "$WAYMO_V2_ROOT/validation/$t"
      while read -r s; do
        [ -n "$s" ] && "$GCLOUD" storage cp -n "$BUCKET/$t/$s.parquet" "$WAYMO_V2_ROOT/validation/$t/"
      done < "$MAN/subset_${SUBSET_N}.txt"
    done
    ;;
  *) echo "usage: $0 manifest|tables|images|verify|subset"; exit 64 ;;
esac
echo "$(stamp) done: ${1}"
