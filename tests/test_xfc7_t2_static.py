import unittest

import numpy as np
import pandas as pd


class TCritTest(unittest.TestCase):
    def test_student_t_with_g_minus_1_df(self):
        from careport.xfc7_t2_static import t_crit
        self.assertAlmostEqual(t_crit(10), 2.2622, places=3)
        self.assertAlmostEqual(t_crit(50), 2.0096, places=3)
        self.assertTrue(np.isnan(t_crit(1)))


class CapacityTest(unittest.TestCase):
    """T2 protocol section 9: per camera and contrast, decidable if |gap| > threshold; a contrast is decidable for
    the dataset if >= 2 cameras clear it."""

    def test_contrasts_and_dataset_rule(self):
        from careport.xfc7_t2_static import capacity
        theta = {"FL": {"start": 0.0, "A": 27.0, "B": 77.0, "mid": 50.0},
                 "SL": {"start": 0.0, "A": 14.0, "B": 62.0, "mid": 50.0},
                 "RL": {"start": 0.0, "A": 92.0, "B": 48.0, "mid": 50.0}}
        thr = {"FL": 30.0, "SL": 30.0, "RL": 30.0}
        cap = capacity(theta, thr)
        self.assertFalse(cap["cameras"]["FL"]["start-A"]["decidable"])      # 27 < 30
        self.assertTrue(cap["cameras"]["FL"]["start-B"]["decidable"])
        self.assertTrue(cap["cameras"]["RL"]["start-A"]["decidable"])
        self.assertTrue(cap["dataset"]["start-B"])                          # FL, SL, RL all clear 30
        self.assertFalse(cap["dataset"]["start-A"])                         # only RL
        self.assertTrue(cap["dataset"]["start-mid"])
        self.assertAlmostEqual(cap["cameras"]["SL"]["A-B"]["gap"], 48.0)


class ProjectionCheckTest(unittest.TestCase):
    def test_median_centre_offset_per_camera_against_15px(self):
        from careport.xfc7_t2_static import projection_check
        st = pd.DataFrame({"camera": [2, 2, 2, 4, 4, 4], "p0_u": [100.0, 200, 300, 100, 100, 100],
                           "p0_v": [50.0] * 6, "u_ref": [103.0, 196, 300, 140, 130, 120], "v_ref": [54.0, 50, 50, 50, 50, 50]})
        out = projection_check(st, {2: "ring_front_left", 4: "ring_side_left"})
        self.assertAlmostEqual(out["ring_front_left"]["median_px"], 4.0)
        self.assertTrue(out["ring_front_left"]["pass"])
        self.assertFalse(out["ring_side_left"]["pass"])                     # median 30 px


class HalfWidthTest(unittest.TestCase):
    def test_predicted_halfwidth_uses_t_with_g_minus_1(self):
        from careport.xfc7_t2_static import predicted_halfwidth
        hw = predicted_halfwidth(de=4.0, v=1.0, g=10)
        self.assertAlmostEqual(hw, 2.2622 * 2.0, places=3)


class CustomContrastTest(unittest.TestCase):
    def test_capacity_with_dataset_specific_contrasts(self):
        from careport.xfc7_t2_static import capacity
        theta = {"LB": {"sample": 0.0, "LIDAR_LEFT": -32.0, "LIDAR_RIGHT": 40.0},
                 "RF": {"sample": 0.0, "LIDAR_LEFT": -5.0, "LIDAR_RIGHT": 3.0}}
        cap = capacity(theta, {"LB": 20.0, "RF": 20.0}, contrasts=[("sample", "LIDAR_LEFT"), ("sample", "LIDAR_RIGHT")])
        self.assertTrue(cap["cameras"]["LB"]["sample-LIDAR_LEFT"]["decidable"])
        self.assertFalse(cap["dataset"]["sample-LIDAR_LEFT"])               # one camera only
        self.assertEqual(sorted(cap["dataset"]), ["sample-LIDAR_LEFT", "sample-LIDAR_RIGHT"])


class DecisionCameraTest(unittest.TestCase):
    def test_cameras_need_30_tracks_and_an_s1_pass(self):
        from careport.xfc7_t2_static import decision_cameras
        cams = {"a": {"tracks": 40}, "b": {"tracks": 40}, "c": {"tracks": 29}, "d": {"tracks": 35}}
        s1 = {"a": {"pass": True}, "b": {"pass": False}, "c": {"pass": True}}       # d has no parked rows
        self.assertEqual(decision_cameras(cams, s1), ["a"])

    def test_far_rule_uses_parked_rows_beyond_40_m_only(self):
        from careport.xfc7_t2_static import s1_table
        st = pd.DataFrame({"camera": [1] * 6, "range_m": [5, 5, 5, 50, 50, 50],
                           "p0_u": [100.0, 100, 100, 100, 100, 100], "u_ref": [140.0, 140, 140, 110, 110, 110],
                           "p0_v": [0.0] * 6, "v_ref": [0.0] * 6})
        self.assertFalse(s1_table(st, {1: "c"}, "written")["c"]["pass"])          # median 40 px
        far = s1_table(st, {1: "c"}, "far40")["c"]
        self.assertTrue(far["pass"])
        self.assertEqual(far["n"], 3)
        with self.assertRaises(ValueError):
            s1_table(st, {1: "c"}, "other")


def _tiny_lab():
    """Camera 1: 40 parked rows in one bin plus 6 moving rows in that bin (one unassigned); camera 2: 5 moving rows
    and no parked rows, so no zero line."""
    rows = []
    rng = np.random.default_rng(1)
    for i in range(40):
        rows.append(dict(segment=f"s{i % 4}", frame=i, camera=1, object_id=f"p{i}", incl_static=True, incl_moving=False,
                         ego_speed=float(i % 5)))
    for i in range(6):
        rows.append(dict(segment=f"s{i % 4}", frame=100 + i, camera=1, object_id=f"m{i}", incl_static=False,
                         incl_moving=True, ego_speed=5.0 + i))
    for i in range(5):
        rows.append(dict(segment=f"s{i % 4}", frame=200 + i, camera=2, object_id=f"n{i}", incl_static=False,
                         incl_moving=True, ego_speed=5.0 + i))
    lab = pd.DataFrame(rows)
    n = len(lab)
    lab = lab.assign(p0_u=rng.uniform(100, 900, n), p0_v=rng.uniform(100, 500, n), e_u=1.0, e_v=0.0, viewpoint_bin=0,
                     range_bin=1, column_third=1, ego_class=0, frame_type="A", sdot=1.0, theta_start=0.0)
    assoc = lab[["segment", "frame", "camera", "object_id"]].copy()
    assoc["camera_object_id"] = [f"h{i}" for i in range(n)]
    assoc.loc[(lab["object_id"] == "m5").to_numpy(), "camera_object_id"] = np.nan
    assoc = assoc.assign(u_ref=lab["p0_u"] + rng.normal(0, 3, n), v_ref=lab["p0_v"], w_ref=50.0, cost=0.1, margin=np.inf)
    assoc["camera_object_id"] = assoc["camera_object_id"].astype(object)
    return lab, assoc


class DescriptiveTablesTest(unittest.TestCase):
    def test_assigned_moving_rows_kept_before_the_zero_line_filter_and_saved(self):
        import os
        import tempfile
        from unittest import mock
        from careport import xfc7_t2_static as m
        lab, assoc = _tiny_lab()
        with mock.patch.object(m, "associate_images", return_value=assoc):
            res = m.run_reference_t2(lab, None, {"start": "theta_start"}, {1: "c1", 2: "c2"})
        self.assertEqual(sorted(res["mo"]["camera"].unique()), [1])                     # camera 2 has no zero line
        ma = res["mo_assigned"]
        self.assertEqual(int((ma["camera"] == 2).sum()), 5)
        self.assertEqual(int((ma["camera"] == 1).sum()), 5)                            # m5 unassigned
        self.assertNotIn("theta_hat", ma.columns)
        with tempfile.TemporaryDirectory() as d:
            h = m.save_descriptive_tables(res, d)
            self.assertEqual(sorted(h), ["moving_assigned.parquet", "static_all.parquet", "static_zero_lined.parquet"])
            zl = pd.read_parquet(os.path.join(d, "static_zero_lined.parquet"))
            self.assertEqual(len(zl), 40)                                              # all 40 parked rows share a bin
            self.assertIn("r", zl.columns)
            st = pd.read_parquet(os.path.join(d, "static_all.parquet"))
            self.assertEqual(len(st), 40)
            self.assertIn("e", st.columns)
            self.assertEqual(len(pd.read_parquet(os.path.join(d, "moving_assigned.parquet"))), 10)
            self.assertEqual(len(h["static_all.parquet"]), 64)


if __name__ == "__main__":
    unittest.main()
