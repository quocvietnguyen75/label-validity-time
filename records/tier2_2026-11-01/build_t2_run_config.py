"""Build the XF-C7 T2 run config (input of careport.xfc7_t2_run) with every input pinned by SHA-256.

Reads hashes and metadata only: the S3 summaries, the moving analysis sets, the TruckScenes descriptive tables and
the label-side tables. Cross-checks each summary's recorded hashes against the files. Paths are written relative to
$AV2_SENSOR_ROOT / $TRUCKSCENES_DERIVED (data) or to the repository root (summaries), never as machine paths.
Usage (repository root, env loaded): PYTHONPATH=src python <this> <out_config.json>"""
import hashlib
import json
import os
import sys

import pandas as pd

from careport.xfc7_t2_ts_labels import TS_CAMERAS, TS_LIDARS

T2 = "newpaper/autonomous_driving_crossmodal_fit_selection_2026-10-13/tier2_2026-11-01"
P01_TRAIN = ["3ca11a5e-50b2-3cc3-af7a-ce7ab02b9954", "58d01358-5927-36fa-9e11-d18d1dc1f4f0",
             "6b14d7c0-20f9-390b-af38-507a5de5998c", "81d2b40a-c579-3e9c-b520-bee26cda947d",
             "c556f8e0-a001-3586-b2cf-d3256685c39f", "c780d53a-2d37-3cd8-9e89-530966aef53e",
             "f648b945-6c70-3105-bd23-9502894e37d4", "fd5c6932-2ee2-3cfb-9bdc-0b30bfb33a91"]
P01_VAL = ["9a448a80-0e9a-3bf0-90f3-21750dfef55a", "d5d6f11c-3026-3e0e-9d67-c111233e22de"]
ROOTS = {"$AV2_SENSOR_ROOT": "AV2_SENSOR_ROOT", "$TRUCKSCENES_DERIVED": "TRUCKSCENES_DERIVED"}
# Protocol constants (T2 sections 8, 9, 11), asserted again by the runner against the pinned summaries.
EXPECTED_S1_RULE = "far40"
EXPECTED_DECISION = ["ring_front_left", "ring_rear_left", "ring_rear_right", "ring_side_left"]
TS_LABELS = "$TRUCKSCENES_DERIVED/t2_labels_rowthird_label_side.parquet"
TS_STATIC = "$TRUCKSCENES_DERIVED/t2_static/s050_rowthird"
TS_MIN_TRACKS = 30                     # section 11: a TruckScenes camera is reported with >= 30 assigned moving tracks


def real(p):
    for k, env in ROOTS.items():
        if p.startswith(k):
            return os.environ[env] + p[len(k):]
    return p


def sha(p):
    return hashlib.sha256(open(real(p), "rb").read()).hexdigest()


def main(argv):
    out_path = argv[1]
    inputs, cfg = [], {}

    def pin(what, p):
        h = sha(p)
        inputs.append({"what": what, "path": p, "sha256": h})
        return h

    lab = "$AV2_SENSOR_ROOT/derived/t2_labels_all50_label_side.parquet"
    lab_h = pin("AV2 label-side table, 50 logs", lab)
    for tag, s in (("primary", "050"), ("sensitivity_score_0_3", "030")):
        summ = f"{T2}/S3_summary_all50_s{s}.json"
        sdir = f"$AV2_SENSOR_ROOT/derived/t2_static/all50_s{s}"
        mset, zl = f"{sdir}/moving_analysis_set_t2.parquet", f"{sdir}/static_zero_lined.parquet"
        pin(f"AV2 S3 summary, score >= 0.{s[1]}", summ)
        mh = pin(f"AV2 moving analysis set, score >= 0.{s[1]}", mset)
        zh = pin(f"AV2 zero-lined parked table, score >= 0.{s[1]}", zl)
        with open(summ, encoding="utf-8") as f:
            js = json.load(f)
        if (js["moving_analysis_set_sha256"] != mh or js["descriptive_tables_sha256"]["static_zero_lined.parquet"] != zh
                or js["inputs"]["label_side_sha256"] != lab_h or js["inputs"]["logs"] != 50):
            raise SystemExit(f"{summ}: recorded hashes or log count disagree with the files; stop")
        if js["S1_rule"] != EXPECTED_S1_RULE or sorted(js["decision_cameras"]) != EXPECTED_DECISION:
            raise SystemExit(f"{summ}: S1 rule or decision cameras differ from the protocol; stop")
        cfg[tag] = {"s3_summary": summ, "moving_set": mset, "static_zero_lined": zl}
    ts_summ = f"{T2}/S3_summary_ts_s050.json"
    pin("TruckScenes S3 summary, score >= 0.5", ts_summ)
    with open(ts_summ, encoding="utf-8") as f:
        ts_js = json.load(f)
    ts_lab_h = pin("TruckScenes label-side table (centre row third)", TS_LABELS)
    if ts_js["inputs"]["label_side_sha256"] != ts_lab_h:
        raise SystemExit("TruckScenes label-side hash disagrees with its summary; stop")
    ts = {}
    for key, name in (("static_all", "static_all.parquet"), ("moving_assigned", "moving_assigned.parquet")):
        p = f"{TS_STATIC}/{name}"
        if pin(f"TruckScenes {key}", p) != ts_js["descriptive_tables_sha256"][name]:
            raise SystemExit(f"{name}: hash disagrees with the TruckScenes summary; stop")
        ts[key] = p
    ma = pd.read_parquet(real(ts["moving_assigned"]), columns=["segment", "object_id", "camera"])
    tracks = ma.drop_duplicates().groupby("camera").size()
    report = [TS_CAMERAS[int(c) - 1] for c, n in tracks.items() if n >= TS_MIN_TRACKS]
    ts.update({"hypotheses": ["sample"] + TS_LIDARS, "camera_names": {str(i): n for i, n in enumerate(TS_CAMERAS, 1)},
               "report_cameras": report, "moving_tracks_by_camera": {TS_CAMERAS[int(c) - 1]: int(n) for c, n in tracks.items()}})
    exp = {"S1_rule": EXPECTED_S1_RULE, "decision_cameras": {t: EXPECTED_DECISION for t in ("primary", "sensitivity_score_0_3")}}
    cfg = {"protocol": "PILOT_PROTOCOL_XF-C7_T2_v1_0.md", "inputs": inputs, "expected": exp, **cfg,
           "p01_logs": sorted(P01_TRAIN + P01_VAL), "truckscenes": ts}
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(out_path, hashlib.sha256(open(out_path, "rb").read()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
