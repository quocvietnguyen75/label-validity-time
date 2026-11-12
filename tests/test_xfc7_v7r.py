"""Synthetic tests for the V7R post hoc checks (V7R plan of 8 Nov 2026 with amendment A1). No dataset is read."""
import math
import unittest

import numpy as np
import pandas as pd

from careport import xfc7_v7r as v


class DeviationTests(unittest.TestCase):
    def setUp(self):
        rows = []
        # segment s1: worst |res| 12 ms; s2: worst 6 ms; s3: worst 2 ms
        for seg, cams in (("s1", [("FRONT", "L2R", 12.0), ("SIDE_LEFT", "L2R", -3.0)]),
                          ("s2", [("FRONT", "R2L", -6.0)]),
                          ("s3", [("FRONT_LEFT", "R2L", 2.0), ("SIDE_RIGHT", "R2L", 1.0)])):
            for cam, ro, r in cams:
                rows.append({"segment": seg, "camera": cam, "readout": ro, "res_ms": r})
        self.df = pd.DataFrame(rows)

    def test_worst_camera_counts(self):
        self.assertEqual(v.worst_camera_counts(self.df), (3, 2, 1))

    def test_summary_shares_and_quantiles(self):
        s = v.deviation_summary(self.df)
        self.assertEqual(s["cells"], 5)
        self.assertAlmostEqual(s["overall"]["share_gt5"], 2 / 5)
        self.assertAlmostEqual(s["overall"]["median_abs"], 3.0)
        self.assertAlmostEqual(s["worst"]["median_abs"], 6.0)
        self.assertEqual(s["by_cell"]["FRONT|L2R"]["n"], 1)

    def test_comparator_share(self):
        comp = {("FRONT", "L2R"): 10.0, ("SIDE_LEFT", "L2R"): 30.0, ("FRONT", "R2L"): 50.0,
                ("FRONT_LEFT", "R2L"): 40.0, ("SIDE_RIGHT", "R2L"): 70.0}
        # only s1 FRONT (12 >= 10) is not below its comparator
        self.assertAlmostEqual(v.share_below_comparator(self.df, comp), 4 / 5)


class TwoFrameTests(unittest.TestCase):
    def test_bound(self):
        r = v.two_frame_bound(lo=-5.0, hi=2.0, theta_bar=52.0, theta_a=28.0, theta_b=78.0)
        self.assertAlmostEqual(r["gap_ms"], 50.0)
        self.assertAlmostEqual(r["kappa"], 0.1)
        self.assertAlmostEqual(r["shift_ms"], 0.1 * 26.0)


class BootstrapTests(unittest.TestCase):
    def test_cluster_bootstrap_median_is_deterministic_and_brackets(self):
        vals = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.05, 0.95])
        cl = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
        e1 = v.cluster_bootstrap(vals, cl, np.median, n=500, seed=1)
        e2 = v.cluster_bootstrap(vals, cl, np.median, n=500, seed=1)
        self.assertEqual(e1, e2)
        self.assertLessEqual(e1[1], e1[0])
        self.assertGreaterEqual(e1[2], e1[0])


class LengthTests(unittest.TestCase):
    def test_track_ranges_and_slope(self):
        rows = []
        for i, (L, spd) in enumerate([(4.5, 5.0), (4.6, 7.0), (4.7, 9.0), (4.8, 11.0)]):
            for k in range(6):
                rows.append({"log": f"g{i % 2}", "track_uuid": f"t{i}", "category": "REGULAR_VEHICLE",
                             "length_m": L, "speed": spd})
        rows.append({"log": "g0", "track_uuid": "tv", "category": "REGULAR_VEHICLE", "length_m": 4.0, "speed": 5.0})
        rows.append({"log": "g0", "track_uuid": "tv", "category": "REGULAR_VEHICLE", "length_m": 4.3, "speed": 5.0})
        ann = pd.DataFrame(rows)
        rng = v.track_length_ranges(ann)
        self.assertEqual(int((rng["length_range"] > 0.01).sum()), 1)
        tr = v.moving_tracks(ann, min_frames=5)
        self.assertEqual(len(tr), 4)
        self.assertAlmostEqual(v.ols_slope(tr["speed"].to_numpy(), tr["length_m"].to_numpy()), 0.05)


class FaceTests(unittest.TestCase):
    """A box of length 4 m at the origin, heading +x; the vehicle moves along +x at 10 m/s.
    LiDAR A passes at 40 ms and B at 90 ms, so D = 10 * 0.05 = 0.5 m and B is the later copy."""

    L, W, H = 4.0, 2.0, 1.5

    def _points(self, face_a, face_b, z=0.2):
        n = 30
        pa = np.column_stack([np.full(n, face_a), np.linspace(-0.8, 0.8, n), np.full(n, z)])
        pb = np.column_stack([np.full(n, face_b), np.linspace(-0.8, 0.8, n), np.full(n, z)])
        xyz = np.vstack([pa, pb])
        laser = np.r_[np.full(n, 3), np.full(n, 40)]
        off = np.r_[np.full(n, 40.0), np.full(n, 90.0)]
        return xyz, laser, off

    def _box(self, cx):
        return {"cx": cx, "cy": 0.0, "cz": 0.0, "l": self.L, "w": self.W, "h": self.H, "yaw": 0.0}

    def test_approaching_fitted_to_earlier(self):
        # ego ahead of the box (box centre at x = -20 relative to ego? use ego at origin, box at x = -20, moving +x)
        box = self._box(-20.0)
        xyz, laser, off = self._points(-20.0 + 2.0, -20.0 + 2.5)   # A face at the box front, B 0.5 m beyond
        r = v.copy_face_gaps(xyz, laser, off, box, v_xy=(10.0, 0.0))
        self.assertEqual(r["cls"], "approaching")
        self.assertEqual(r["earlier"], "A")
        self.assertAlmostEqual(r["D"], 0.5)
        self.assertAlmostEqual(r["g_earlier"] / r["D"], 0.0, places=6)
        self.assertAlmostEqual(r["g_later"] / r["D"], 1.0, places=6)
        self.assertAlmostEqual(r["separation"], 1.0, places=6)

    def test_approaching_fitted_to_later(self):
        box = self._box(-20.0)
        xyz, laser, off = self._points(-20.0 + 1.5, -20.0 + 2.0)
        r = v.copy_face_gaps(xyz, laser, off, box, v_xy=(10.0, 0.0))
        self.assertAlmostEqual(r["g_earlier"] / r["D"], -1.0, places=6)
        self.assertAlmostEqual(r["g_later"] / r["D"], 0.0, places=6)

    def test_receding_fitted_to_earlier(self):
        box = self._box(20.0)                                          # box ahead of ego, moving away (+x)
        xyz, laser, off = self._points(20.0 - 2.0, 20.0 - 1.5)         # rear faces: A at the box rear, B 0.5 m ahead
        r = v.copy_face_gaps(xyz, laser, off, box, v_xy=(10.0, 0.0))
        self.assertEqual(r["cls"], "receding")
        self.assertAlmostEqual(r["g_earlier"] / r["D"], 0.0, places=6)
        self.assertAlmostEqual(r["g_later"] / r["D"], -1.0, places=6)
        self.assertAlmostEqual(r["separation"], 1.0, places=6)

    def test_ground_points_are_excluded(self):
        box = self._box(-20.0)
        xyz, laser, off = self._points(-20.0 + 2.0, -20.0 + 2.5)
        ground = np.column_stack([np.linspace(-17.5, -17.0, 20), np.zeros(20), np.full(20, -0.75)])  # road beyond the front
        xyz2 = np.vstack([xyz, ground])
        laser2 = np.r_[laser, np.full(20, 3)]
        off2 = np.r_[off, np.full(20, 40.0)]
        r = v.copy_face_gaps(xyz2, laser2, off2, box, v_xy=(10.0, 0.0))
        self.assertAlmostEqual(r["g_earlier"] / r["D"], 0.0, places=6)

    def test_too_few_points_returns_none(self):
        box = self._box(-20.0)
        xyz, laser, off = self._points(-20.0 + 2.0, -20.0 + 2.5)
        keep = np.r_[np.arange(5), np.arange(30, 60)]
        self.assertIsNone(v.copy_face_gaps(xyz[keep], laser[keep], off[keep], box, v_xy=(10.0, 0.0)))

    def test_parked_near_face_gap(self):
        box = self._box(-20.0)                                         # ego at origin is ahead of the box (+x)
        xyz = np.column_stack([np.full(30, -20.0 + 1.9), np.linspace(-0.8, 0.8, 30), np.full(30, 0.2)])
        g = v.parked_near_face_gap(xyz, box)
        self.assertAlmostEqual(g, -0.1, places=6)

    def test_classify(self):
        sig = v.SIGNATURES
        med = {"approaching": ((-0.05, 0.1), (0.9, 1.1)), "receding": ((-0.1, 0.1), (-1.1, -0.9))}
        self.assertEqual(v.consistent_routes(med, sig), ["earlier copy"])


if __name__ == "__main__":
    unittest.main()
