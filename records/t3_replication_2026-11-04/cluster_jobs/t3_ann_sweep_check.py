"""XF-C7 T3: label-side data check. For each log, annotation timestamps against the LiDAR sweeps.

Reads only annotations.feather (timestamp_ns, category), city_SE3_egovehicle.feather (timestamp_ns),
the bucket listing that download.sbatch saved (_manifest/etags_<split>_<id>.tsv) and the on-disk file names.
Reads no image, no detection, no LiDAR point and no outcome.

Usage: python -I t3_ann_sweep_check.py <root> <list.txt> <out.json>
"""
import glob
import json
import os
import sys

import pyarrow.feather as feather

# Copied verbatim from careport.xfc7_t2_av2_labels (aeb058ab...), not imported: that module pulls in the detector module.
VEHICLE = ["REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "SCHOOL_BUS", "ARTICULATED_BUS", "BOX_TRUCK", "TRUCK",
           "TRUCK_CAB", "MOTORCYCLE"]
RING = ("ring_front_center", "ring_front_left", "ring_front_right", "ring_rear_left", "ring_rear_right",
        "ring_side_left", "ring_side_right")


def stamps(paths, ext):
    out = []
    for p in paths:
        b = os.path.basename(p)
        if b.endswith(ext):
            out.append(int(b[: -len(ext)]))
    return sorted(out)


def check_log(root, entry):
    split, lid = entry.split("/", 1)
    L = os.path.join(root, split, lid)
    ann = feather.read_table(os.path.join(L, "annotations.feather"), columns=["timestamp_ns", "category"]).to_pandas()
    pose = feather.read_table(os.path.join(L, "city_SE3_egovehicle.feather"), columns=["timestamp_ns"]).to_pandas()
    disk_sweeps = stamps(glob.glob(os.path.join(L, "sensors", "lidar", "*.feather")), ".feather")
    tsv = os.path.join(root, "_manifest", f"etags_{split}_{lid}.tsv")
    bucket_sweeps, bucket_last_img = [], {}
    with open(tsv, encoding="utf-8") as f:
        for line in f:
            key = line.split("\t", 1)[0]
            if "/sensors/lidar/" in key and key.endswith(".feather"):
                bucket_sweeps.append(int(os.path.basename(key)[: -len(".feather")]))
            for c in RING:
                if f"/sensors/cameras/{c}/" in key and key.endswith(".jpg"):
                    t = int(os.path.basename(key)[: -len(".jpg")])
                    bucket_last_img[c] = max(bucket_last_img.get(c, t), t)
    bucket_sweeps.sort()
    sw = set(disk_sweeps)
    ats = sorted(set(int(t) for t in ann["timestamp_ns"].to_numpy()))
    missing = [t for t in ats if t not in sw]
    veh_rows = {}
    v = ann[ann["category"].isin(VEHICLE)]
    for t in missing:
        veh_rows[str(t)] = int((v["timestamp_ns"] == t).sum())
    all_rows = {str(t): int((ann["timestamp_ns"] == t).sum()) for t in missing}
    first, last = (disk_sweeps[0], disk_sweeps[-1]) if disk_sweeps else (None, None)
    where = {}
    for t in missing:
        if first is None:
            where[str(t)] = "no sweeps"
        elif t < first:
            where[str(t)] = "before first sweep"
        elif t > last:
            where[str(t)] = "after last sweep"
        else:
            where[str(t)] = "inside sweep range"
    pts = pose["timestamp_ns"].to_numpy()
    return {
        "log": entry,
        "n_annotation_stamps": len(ats),
        "n_sweeps_disk": len(disk_sweeps),
        "n_sweeps_bucket": len(bucket_sweeps),
        "disk_equals_bucket": disk_sweeps == bucket_sweeps,
        "first_sweep": first,
        "last_sweep": last,
        "annotation_stamps_without_sweep": missing,
        "where": where,
        "rows_all_categories": all_rows,
        "rows_vehicle": veh_rows,
        "pose_min": int(pts.min()),
        "pose_max": int(pts.max()),
        "last_ring_image": bucket_last_img,
        "sweeps_without_annotation": len([t for t in disk_sweeps if t not in set(ats)]),
    }


def main(argv):
    root, lst, out = argv[1], argv[2], argv[3]
    entries = [x.strip() for x in open(lst, encoding="utf-8") if x.strip()]
    res = []
    for i, e in enumerate(entries, 1):
        r = check_log(root, e)
        res.append(r)
        if r["annotation_stamps_without_sweep"] or not r["disk_equals_bucket"]:
            print(i, e, "missing:", r["annotation_stamps_without_sweep"], "where:", r["where"],
                  "veh rows:", r["rows_vehicle"], "disk==bucket:", r["disk_equals_bucket"], flush=True)
    bad = [r["log"] for r in res if r["annotation_stamps_without_sweep"]]
    summ = {"n_logs": len(res), "logs_with_annotation_stamps_without_sweep": bad,
            "n_bad": len(bad), "all_disk_equals_bucket": all(r["disk_equals_bucket"] for r in res),
            "vehicle_categories": VEHICLE, "per_log": res}
    with open(out, "x", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    print("n_logs", len(res), "n_bad", len(bad), "all_disk_equals_bucket", summ["all_disk_equals_bucket"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
