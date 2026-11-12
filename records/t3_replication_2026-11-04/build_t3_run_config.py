"""XF-C7 T3: build config/preanalysis/xfc7_t3_v1_0.json for xfc7_t3_run (T3 protocol section 15), on the cluster.

Pins every file the run reads by SHA-256 (paths under $AV2_SENSOR_ROOT = /mnt/data/vietnq/xfc7), fills each analysis
block's `expected` decision cameras from its S3 summary, and the S-g subgroups from the sample file. Reads summaries,
hashes and log ids only. Usage (env311 active):
  python build_t3_run_config.py <av2_t3_sample.txt> <protocol file name> <out_config.json>
5 Nov (SAMPLE_PIN addendum 12): the blocks are named for the analysed counts (407 logs; pooled 457), the exclusion
check and the pooled label summaries are pinned as provenance, and an existing config is never overwritten."""
import hashlib
import json
import os
import re
import sys

ROOT = os.environ.get("AV2_SENSOR_ROOT", "/mnt/data/vietnq/xfc7")
D = "$AV2_SENSOR_ROOT/derived"


def real(p):
    return p.replace("$AV2_SENSOR_ROOT", ROOT)


def sha(p):
    h = hashlib.sha256()
    with open(real(p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def static_reads(tag_dir, score_tag, label_side=None):
    """The static step's outputs for one run: summary <tag_dir>/S3_summary_<score_tag>.json, tables in <tag_dir>/<score_tag>/."""
    o = f"{D}/{tag_dir}/{score_tag}"
    r = {"s3_summary": f"{D}/{tag_dir}/S3_summary_{score_tag}.json", "moving_set": f"{o}/moving_analysis_set_t2.parquet",
         "static_zero_lined": f"{o}/static_zero_lined.parquet"}
    if label_side:
        r["label_side"] = label_side
        r["reference_table"] = f"{o}/reference_table.parquet"
    return r


def main(argv):
    sample, proto, out = argv[1:4]
    ids = re.findall(r"(val|train)/([0-9a-f-]{36})", open(sample, encoding="utf-8").read())
    comparison = {"t2_s3_summary": "$AV2_SENSOR_ROOT/data/t2_pc_frozen/S3_summary_all50_s050.json",
                  "t2_results": "$AV2_SENSOR_ROOT/data/t2_pc_frozen/T2_RESULTS_2026-11-01.json",
                  "r4_results": "$AV2_SENSOR_ROOT/data/t2_pc_frozen/r4_results.json",
                  "t2_moving_set": "$AV2_SENSOR_ROOT/data/t2_pc_frozen/all50_s050/moving_analysis_set_t2.parquet"}
    analyses = {
        "primary": {"role": "T3 primary: 407 analysed logs (408 drawn, one excluded; SAMPLE_PIN addendum 11), "
                            "score >= 0.5 (sections 9-10)",
                    "reads": static_reads("t3_static", "s050", f"{D}/t3_labels_label_side.parquet"),
                    "compositions": True, "e4": True,
                    "subgroups": {"val": [i for s, i in ids if s == "val"], "train": [i for s, i in ids if s == "train"]}},
        "sensitivity_score_0_3": {"role": "S-e: score >= 0.3, own budgets and decision cameras",
                                  "reads": static_reads("t3_static", "s030")},
        "t2_h100_with_t2_frozen_budgets": {"role": "S-b (i): T2's 50 logs on H100 tables with T2's frozen S3 summary",
                                           "reads": {"s3_summary": comparison["t2_s3_summary"],
                                                     "moving_set": f"{D}/t2h100_static/s050/moving_analysis_set_t2.parquet",
                                                     "static_zero_lined": f"{D}/t2h100_static/s050/static_zero_lined.parquet"}},
        "t2_h100_recomputed": {"role": "S-b (ii): T2's 50 logs on H100 tables with the H100 budgets and cameras",
                               "reads": static_reads("t2h100_static", "s050"), "compositions": True},
        "t2_h100_recomputed_score_0_3": {"role": "S-b (ii) at score >= 0.3", "reads": static_reads("t2h100_static", "s030")},
        "pooled_457": {"role": "S-c: pooled 457 logs (T2's 50 on H100 plus T3's 407 analysed), score >= 0.5",
                       "reads": static_reads("pooled_static", "s050"), "compositions": True},
    }
    for blk in analyses.values():
        s3 = json.load(open(real(blk["reads"]["s3_summary"]), encoding="utf-8"))
        blk["expected"] = {"S1_rule": s3["S1_rule"], "decision_cameras": sorted(s3["decision_cameras"])}
    extra = {"t3_label_summary": f"{D}/t3_labels_summary.json", "t3_s0": f"{D}/s0_t3.json",
             "t3_det_manifest_a": f"{D}/t3_det_a/raw/run_manifest.json", "t3_det_manifest_b": f"{D}/t3_det_b/raw/run_manifest.json",
             "t3_verify": f"{D}/t3_verify/_DONE.json",
             "t3_annotation_check": f"{D}/t3_ann_check/ann_sweep_check_408.json",
             "pooled_label_summary": f"{D}/pooled457_labels_summary.json", "pooled_s0": f"{D}/s0_pooled457.json"}
    paths = sorted({p for b in analyses.values() for p in b["reads"].values()} | set(comparison.values()) | set(extra.values()))
    cfg = {"protocol": proto, "built_by": "t3_replication_2026-11-04/build_t3_run_config.py",
           "inputs": [{"what": os.path.basename(p), "path": p, "sha256": sha(p)} for p in paths],
           "bootstrap": {"draws": 99999, "seed": 20261103, "weights": "webb6"},
           "comparison": comparison, "analyses": analyses, "provenance_only": extra}
    with open(out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, indent=1)
    print(out, len(cfg["inputs"]), "inputs pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
