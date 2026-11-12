"""Tests for t3_label_shares.py (written before it). Run from this folder: python -m unittest test_t3_label_shares -v

The tool must give T2's seam-rule and fallback shares by T2's own definition (tier2_2026-11-01/diagnostics/
t2_label_and_s1_tables.py, label_shares), read label-side columns only, check its input's SHA-256 and never
overwrite its output.
"""
import importlib.util
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import t3_label_shares as t

HERE = os.path.dirname(os.path.abspath(__file__))
T2_DIAG = os.path.join(HERE, "..", "tier2_2026-11-01", "diagnostics", "t2_label_and_s1_tables.py")


def t2_module():
    spec = importlib.util.spec_from_file_location("t2_label_and_s1_tables", T2_DIAG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0, 20, n)
    speed[rng.random(n) < 0.05] = np.nan
    return pd.DataFrame({
        "camera": rng.integers(1, 8, n).astype("int8"),
        "most_visible": rng.random(n) < 0.9, "num_pts": rng.integers(0, 200, n), "inside_all": rng.random(n) < 0.9,
        "overlap": rng.random(n) < 0.1, "static_proj_ok": rng.random(n) < 0.95, "speed": speed,
        "sdot": rng.uniform(0, 3, n), "along_axis_deg": rng.uniform(0, 90, n), "seam": rng.random(n) < 0.15,
        "theta_fallback_A": rng.random(n) < 0.02, "theta_fallback_B": rng.random(n) < 0.05})


class SameAsT2(unittest.TestCase):
    def test_identical_to_t2_definition(self):
        lab = synthetic()
        self.assertEqual(t.label_shares(lab), t2_module().label_shares(lab))

    def test_reads_t2_columns_only(self):
        self.assertEqual(t.COLS, ["camera", "most_visible", "num_pts", "inside_all", "overlap", "static_proj_ok",
                                  "speed", "sdot", "along_axis_deg", "seam", "theta_fallback_A", "theta_fallback_B"])


class Refusals(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.lab = os.path.join(self.d.name, "lab.parquet")
        synthetic(300).assign(extra=1.0).to_parquet(self.lab)
        self.out = os.path.join(self.d.name, "out.json")

    def tearDown(self):
        self.d.cleanup()

    def test_wrong_hash_stops_before_writing(self):
        with self.assertRaises(SystemExit):
            t.main(["t3_label_shares.py", self.lab, "0" * 64, self.out])
        self.assertFalse(os.path.exists(self.out))

    def test_right_hash_writes_once(self):
        h = t.sha(self.lab)
        t.main(["t3_label_shares.py", self.lab, h, self.out])
        with open(self.out, encoding="utf-8") as fh:
            got = json.load(fh)
        self.assertEqual(got["input_sha256"], h)
        self.assertIn("_all", got["label_side_shares"])
        with self.assertRaises(FileExistsError):
            t.main(["t3_label_shares.py", self.lab, h, self.out])


if __name__ == "__main__":
    unittest.main()
