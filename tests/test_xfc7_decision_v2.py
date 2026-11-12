import unittest

from careport.xfc7_decision_v2 import combine_runs, decide_rejection

CAMS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT"]
B = {"FRONT": 16.3, "FRONT_LEFT": 18.8, "FRONT_RIGHT": 36.9, "SIDE_LEFT": 12.6}


def _d(est_by_cam, hw=3.0):
    return {c: (e, e - hw, e + hw) for c, e in est_by_cam.items()}


class AdditiveRuleTest(unittest.TestCase):
    def test_frame_ruled_out_pose_rejected_once_lidar_and_cam_kept(self):
        delta = {
            "frame": _d({"FRONT": 49.0, "FRONT_LEFT": 39.0, "FRONT_RIGHT": 60.0, "SIDE_LEFT": 24.0}),
            "lidar": _d({"FRONT": 0.5, "FRONT_LEFT": -1.0, "FRONT_RIGHT": 1.0, "SIDE_LEFT": 0.2}),
            "pose": _d({"FRONT": -0.9, "FRONT_LEFT": -11.0, "FRONT_RIGHT": 10.0, "SIDE_LEFT": -25.8}),
            "cam": _d({"FRONT": -3.0, "FRONT_LEFT": 2.0, "FRONT_RIGHT": -2.5, "SIDE_LEFT": -2.4}),
        }
        r = decide_rejection(delta, B, CAMS)
        # additive thresholds 5 + B: 21.3, 23.8, 41.9, 17.6; CI lower bounds 46, 36, 57, 21 -> all rejected
        self.assertEqual(r["frame"]["rejected_in"], CAMS)
        self.assertTrue(r["frame"]["ruled_out"])
        self.assertEqual(r["pose"]["rejected_in"], ["SIDE_LEFT"])       # CI upper -22.8 < -17.6
        self.assertFalse(r["pose"]["ruled_out"])
        self.assertFalse(r["lidar"]["ruled_out"])
        self.assertFalse(r["cam"]["ruled_out"])

    def test_budget_widens_the_margin(self):
        # FRONT: CI (17, 23) lies outside +-5 and the estimate exceeds B = 16.3, but the CI is not outside +-21.3
        delta = {"frame": {"FRONT": (20.0, 17.0, 23.0), "SIDE_LEFT": (20.0, 18.0, 22.0)}}
        add = decide_rejection(delta, B, ["FRONT", "SIDE_LEFT"])
        self.assertEqual(add["frame"]["rejected_in"], ["SIDE_LEFT"])     # 18 > 17.6
        pair = decide_rejection(delta, B, ["FRONT", "SIDE_LEFT"], rule="pair")
        self.assertEqual(pair["frame"]["rejected_in"], ["FRONT", "SIDE_LEFT"])

    def test_negative_side(self):
        delta = {"pose": {"FRONT_LEFT": (-30.0, -34.0, -26.0), "SIDE_LEFT": (-26.0, -30.0, -22.0)}}
        self.assertTrue(decide_rejection(delta, B, ["FRONT_LEFT", "SIDE_LEFT"])["pose"]["ruled_out"])


class CombineRunsTest(unittest.TestCase):
    def test_ruled_out_requires_both_runs(self):
        p = {"frame": {"rejected_in": ["FRONT", "SIDE_LEFT"], "ruled_out": True},
             "pose": {"rejected_in": ["SIDE_LEFT", "FRONT_LEFT"], "ruled_out": True},
             "lidar": {"rejected_in": [], "ruled_out": False}}
        m = {"frame": {"rejected_in": ["FRONT", "SIDE_LEFT"], "ruled_out": True},
             "pose": {"rejected_in": ["SIDE_LEFT"], "ruled_out": False},
             "lidar": {"rejected_in": [], "ruled_out": False}}
        v = combine_runs(p, m)
        self.assertEqual(v["frame"], "ruled out")
        self.assertEqual(v["pose"], "association-sensitive, not ruled out")
        self.assertEqual(v["lidar"], "not ruled out")


if __name__ == "__main__":
    unittest.main()
