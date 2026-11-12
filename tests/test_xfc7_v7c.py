"""Synthetic tests for V7C (plan V7C_PLAN_2026-11-10.md): how far the AV2 camera means move boxes, against AV2's
detection thresholds. No dataset is read."""
import hashlib
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from careport import xfc7_v7c as v


def cell(est, los, his):
    """A stored three-method cell: the same est, the given lo and hi per method (cr1, cr2_bm, bootstrap)."""
    return {m: {"est": est, "lo": lo, "hi": hi} for m, lo, hi in zip(("cr1", "cr2_bm", "bootstrap"), los, his)}


class DisplacementTests(unittest.TestCase):
    def test_metres_from_speed_and_ms(self):
        self.assertEqual(v.displacement(10.0, 50.0), 0.5)
        np.testing.assert_allclose(v.displacement(np.array([4.0, 20.0]), 25.0), [0.1, 0.5])

    def test_shares_are_strict(self):
        d = np.array([0.5, 0.51, 1.0, 1.5, 2.0, 4.0, 4.01, 0.0])
        s = v.shares(d)
        self.assertAlmostEqual(s[0.5], 6 / 8)   # 0.5 itself does not count
        self.assertAlmostEqual(s[1.0], 4 / 8)   # 1.5, 2.0, 4.0, 4.01 (1.0 itself does not count)
        self.assertAlmostEqual(s[2.0], 2 / 8)   # 4.0, 4.01
        self.assertAlmostEqual(s[4.0], 1 / 8)   # 4.01

    def test_summary_has_shares_median_p90(self):
        s = v.summarize(np.array([0.1, 0.2, 0.6, 1.2]))
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["share_gt"]["0.5"], 0.5)
        self.assertAlmostEqual(s["median_m"], 0.4)
        self.assertAlmostEqual(s["p90_m"], float(np.percentile([0.1, 0.2, 0.6, 1.2], 90)))


class OffsetTests(unittest.TestCase):
    def test_main_lower_upper(self):
        o = v.offsets(cell(52.0, (47.6, 47.5, 47.4), (56.5, 56.6, 56.7)), b_c=15.0)
        self.assertEqual(o["M"], 52.0)
        self.assertAlmostEqual(o["L"], 47.4 - 15.0)
        self.assertEqual(o["U"], 56.7)

    def test_lower_floor_at_zero(self):
        o = v.offsets(cell(13.0, (7.6, -31.6, 8.1), (19.0, 58.2, 18.6)), b_c=68.6)
        self.assertEqual(o["L"], 0.0)
        self.assertEqual(v.displacement(10.0, o["L"]), 0.0)

    def test_methods_must_share_est(self):
        bad = cell(52.0, (47.6, 47.5, 47.4), (56.5, 56.6, 56.7))
        bad["bootstrap"]["est"] = 51.0
        with self.assertRaises(ValueError):
            v.offsets(bad, b_c=15.0)


class UnitTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "segment": ["a", "a", "a", "a", "b"],
            "camera": [2, 2, 2, 4, 2],
            "object_id": ["o1", "o1", "o1", "o1", "o9"],
            "speed": [3.0, 10.0, 20.0, 8.0, 5.0],
        })

    def test_track_median_speed(self):
        t = v.tracks(self.df)
        self.assertEqual(len(t), 3)  # (a, 2, o1), (a, 4, o1), (b, 2, o9)
        row = t[(t.segment == "a") & (t.camera == 2) & (t.object_id == "o1")].iloc[0]
        self.assertEqual(row.speed, 10.0)

    def test_population_counts(self):
        offs = {2: {"M": 60.0, "L": 30.0, "U": 90.0}, 4: {"M": 50.0, "L": 0.0, "U": 70.0}}
        r = v.population(self.df, offs, cameras=(2, 4))
        self.assertEqual(r["counts"], {"observations": 5, "tracks": 3, "logs": 2})
        # observation unit, offset M: d = speed * 60 ms (camera 2) or * 50 ms (camera 4)
        d = np.array([0.18, 0.6, 1.2, 0.4, 0.3])
        self.assertAlmostEqual(r["observation"]["M"]["share_gt"]["0.5"], float(np.mean(d > 0.5)))
        # track unit, offset M: medians 10 (cam 2) -> 0.6 m, 8 (cam 4) -> 0.4 m, 5 (cam 2) -> 0.3 m
        self.assertAlmostEqual(r["track"]["M"]["share_gt"]["0.5"], 1 / 3)
        self.assertEqual(r["observation"]["L"]["share_gt"]["2.0"], 0.0)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for log, spd, n in (("slow", 4.0, 30), ("mid", 10.0, 30), ("fast", 20.0, 30)):
            rows += [{"segment": log, "camera": 2, "object_id": f"{log}{i}", "speed": spd} for i in range(n)]
        self.df = pd.DataFrame(rows)

    def test_deterministic_and_brackets(self):
        offs = {2: {"M": 60.0, "L": 0.0, "U": 60.0}}
        a = v.bootstrap_shares(self.df, offs, seed=7, n_boot=500)
        b = v.bootstrap_shares(self.df, offs, seed=7, n_boot=500)
        self.assertEqual(a, b)
        point = 2 / 3  # mid (0.6 m) and fast (1.2 m) exceed 0.5 m
        lo, hi = a["0.5"]
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)
        self.assertGreater(hi - lo, 0.0)  # logs differ, so resampling logs moves the share

    def test_resamples_logs_not_rows(self):
        # one log only: every resample is that log, so the interval collapses to the point
        offs = {2: {"M": 60.0, "L": 0.0, "U": 60.0}}
        one = self.df[self.df.segment == "fast"]
        lo, hi = v.bootstrap_shares(one, offs, seed=7, n_boot=200)["0.5"]
        self.assertEqual((lo, hi), (1.0, 1.0))


class InputTests(unittest.TestCase):
    def test_hash_check(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.bin")
            with open(p, "wb") as fh:
                fh.write(b"abc")
            good = hashlib.sha256(b"abc").hexdigest()
            v.check_sha256(p, good)
            with self.assertRaises(ValueError):
                v.check_sha256(p, "0" * 64)

    def test_split_map(self):
        lines = ["val/a", "train/b", "train/c"]
        self.assertEqual(v.split_map(lines), {"a": "val", "b": "train", "c": "train"})
        with self.assertRaises(ValueError):
            v.split_map(["val/a", "train/a"])
        with self.assertRaises(ValueError):
            v.split_map(["test/a"])


if __name__ == "__main__":
    unittest.main()
