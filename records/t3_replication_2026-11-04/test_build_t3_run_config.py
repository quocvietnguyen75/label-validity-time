"""Tests for build_t3_run_config.py (T3 section 15), written before its 5 Nov change (SAMPLE_PIN addendum 12):
blocks named for the analysed counts (407, pooled 457), the exclusion evidence and the pooled label summaries pinned,
and no overwrite of an existing config. Run from this folder: python -m unittest test_build_t3_run_config -v"""
import hashlib
import json
import os
import tempfile
import unittest

import build_t3_run_config as b

R = "$AV2_SENSOR_ROOT"
CAMS = ["ring_side_left", "ring_front_left"]
IDS = [("val", "00000000-0000-0000-0000-000000000001"), ("train", "00000000-0000-0000-0000-000000000002"),
       ("train", "00000000-0000-0000-0000-000000000003")]
PROTO = "PILOT_PROTOCOL_XF-C7_T3_v1_0.md"
NEW_PROVENANCE = (f"{R}/derived/t3_ann_check/ann_sweep_check_408.json", f"{R}/derived/pooled457_labels_summary.json",
                  f"{R}/derived/s0_pooled457.json")
OLD_PROVENANCE = (f"{R}/derived/t3_labels_summary.json", f"{R}/derived/s0_t3.json",
                  f"{R}/derived/t3_det_a/raw/run_manifest.json", f"{R}/derived/t3_det_b/raw/run_manifest.json",
                  f"{R}/derived/t3_verify/_DONE.json")


def write(root, rel, data):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as fh:
        fh.write(data)
    return p


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


class BuildConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.old_root = b.ROOT
        b.ROOT = self.root
        self.bytes = {}
        cams = {"t3_static": CAMS + ["ring_rear_left"], "t2h100_static": CAMS, "pooled_static": CAMS + ["ring_side_right"]}
        for d, tags in (("t3_static", ("s050", "s030")), ("t2h100_static", ("s050", "s030")), ("pooled_static", ("s050",))):
            for t in tags:
                self.put(f"derived/{d}/S3_summary_{t}.json",
                         json.dumps({"S1_rule": "far40", "decision_cameras": cams[d], "tag": f"{d}/{t}"}).encode())
                for f in ("moving_analysis_set_t2.parquet", "static_zero_lined.parquet", "reference_table.parquet"):
                    self.put(f"derived/{d}/{t}/{f}", f"{d}/{t}/{f}".encode())
        self.put("derived/t3_labels_label_side.parquet", b"label side")
        self.put("data/t2_pc_frozen/S3_summary_all50_s050.json",
                 json.dumps({"S1_rule": "far40", "decision_cameras": CAMS}).encode())
        for rel in ("data/t2_pc_frozen/T2_RESULTS_2026-11-01.json", "data/t2_pc_frozen/r4_results.json",
                    "data/t2_pc_frozen/all50_s050/moving_analysis_set_t2.parquet"):
            self.put(rel, rel.encode())
        for p in OLD_PROVENANCE + NEW_PROVENANCE:
            self.put(p.replace(R + "/", ""), p.encode())
        self.sample = write(self.root, "sample.txt", "".join(f"{s}/{i}\n" for s, i in IDS).encode())
        self.out = os.path.join(self.root, "cfg.json")
        b.main(["build", self.sample, PROTO, self.out])
        with open(self.out, encoding="utf-8") as fh:
            self.cfg = json.load(fh)

    def tearDown(self):
        b.ROOT = self.old_root
        self.tmp.cleanup()

    def put(self, rel, data):
        write(self.root, rel, data)
        self.bytes[f"{R}/{rel}"] = data

    def test_blocks_are_named_for_the_analysed_counts(self):
        a = self.cfg["analyses"]
        self.assertEqual(set(a), {"primary", "sensitivity_score_0_3", "t2_h100_with_t2_frozen_budgets",
                                  "t2_h100_recomputed", "t2_h100_recomputed_score_0_3", "pooled_457"})
        self.assertIn("407", a["primary"]["role"])
        self.assertIn("457", a["pooled_457"]["role"])
        for blk in a.values():
            self.assertNotIn("408 fresh", blk["role"])
            self.assertNotIn("458", blk["role"])

    def test_exclusion_evidence_and_pooled_summaries_are_pinned(self):
        prov = set(self.cfg["provenance_only"].values())
        pinned = {i["path"]: i["sha256"] for i in self.cfg["inputs"]}
        for p in NEW_PROVENANCE + OLD_PROVENANCE:
            self.assertIn(p, prov)
            self.assertEqual(pinned[p], sha_bytes(self.bytes[p]))

    def test_every_read_is_pinned_with_its_hash(self):
        pinned = {i["path"]: i["sha256"] for i in self.cfg["inputs"]}
        reads = [p for blk in self.cfg["analyses"].values() for p in blk["reads"].values()]
        reads += list(self.cfg["comparison"].values())
        for p in reads:
            self.assertEqual(pinned[p], sha_bytes(self.bytes[p]), p)
        self.assertEqual(self.cfg["analyses"]["primary"]["reads"]["label_side"], f"{R}/derived/t3_labels_label_side.parquet")
        self.assertEqual(self.cfg["analyses"]["pooled_457"]["reads"]["s3_summary"],
                         f"{R}/derived/pooled_static/S3_summary_s050.json")

    def test_expected_cameras_come_from_each_summary(self):
        a = self.cfg["analyses"]
        self.assertEqual(a["primary"]["expected"], {"S1_rule": "far40", "decision_cameras": sorted(CAMS + ["ring_rear_left"])})
        self.assertEqual(a["pooled_457"]["expected"]["decision_cameras"], sorted(CAMS + ["ring_side_right"]))
        self.assertEqual(a["t2_h100_with_t2_frozen_budgets"]["expected"]["decision_cameras"], sorted(CAMS))

    def test_protocol_name_bootstrap_and_subgroups(self):
        self.assertEqual(self.cfg["protocol"], PROTO)
        self.assertEqual(self.cfg["bootstrap"], {"draws": 99999, "seed": 20261103, "weights": "webb6"})
        self.assertEqual(self.cfg["analyses"]["primary"]["subgroups"],
                         {"val": [IDS[0][1]], "train": [IDS[1][1], IDS[2][1]]})
        self.assertTrue(self.cfg["analyses"]["primary"]["e4"])
        self.assertTrue(self.cfg["analyses"]["pooled_457"]["compositions"])

    def test_refuses_to_overwrite_a_config(self):
        with self.assertRaises(FileExistsError):
            b.main(["build", self.sample, PROTO, self.out])


if __name__ == "__main__":
    unittest.main()
