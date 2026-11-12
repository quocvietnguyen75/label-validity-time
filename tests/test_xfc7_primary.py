import unittest

import numpy as np
import pandas as pd

from careport.xfc7_primary import analyse

R, S = 44.1, 13.8
BASE = {2: 37.5, 4: 25.0, 1: 50.0}          # FRONT_LEFT, SIDE_LEFT, FRONT image-centre times (ms)


def _synthetic(truth, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for seg in range(90):
        ro = 2 if seg % 2 == 0 else 4
        for trk in range(6):
            cam = [1, 2, 4][(seg + trk) % 3]
            for fr in range(5):
                xi = rng.uniform(-0.45, 0.45)
                t_lidar = BASE[cam] + S * xi
                t_cam = BASE[cam] + 1.0 + (R * xi if ro == 2 else -R * xi)
                t_pose = 49.8
                tau = {"frame": 0.0, "lidar": t_lidar, "pose": t_pose, "cam": t_cam}[truth]
                sdot = rng.uniform(0.4, 2.0)
                s = 1 if rng.random() < 0.9 else -1
                ang = rng.uniform(0, 2 * np.pi)
                e = np.array([np.cos(ang), np.sin(ang)])
                d = s * e
                z_b = rng.normal(2.0, 1.0)
                p0 = rng.uniform(200, 1700, 2)
                noise = rng.normal(0, 6.0)
                u_ref = p0 - (sdot * tau + s * z_b + noise) * d
                rows.append({"segment": f"s{seg}", "object_id": f"o{trk}", "camera": cam, "frame": fr, "readout": ro,
                             "p0_u": p0[0], "p0_v": p0[1], "u_ref": u_ref[0], "v_ref": u_ref[1], "d_u": d[0], "d_v": d[1],
                             "s": s, "z_b": z_b, "sdot": sdot, "sigma_b": 6.0, "xi": xi,
                             "tau_frame": 0.0, "tau_lidar": t_lidar, "tau_pose": t_pose, "tau_cap": t_cam})
    df = pd.DataFrame(rows)
    df["n_track"] = df.groupby(["segment", "object_id", "camera"])["frame"].transform("size")
    df["w"] = df["sdot"] ** 2 / (df["sigma_b"] ** 2 * df["n_track"])
    return df


class PrimaryTest(unittest.TestCase):
    def _run(self, truth):
        return analyse(_synthetic(truth), ["FRONT", "FRONT_LEFT", "SIDE_LEFT"], ["L2R", "R2L"])

    def test_recovers_lidar(self):
        r = self._run("lidar")
        self.assertEqual(r["decision"]["supported"], ["lidar"])
        self.assertAlmostEqual(r["cameras"]["SIDE_LEFT"]["delta"]["lidar"]["est"], 0.0, delta=1.5)

    def test_recovers_frame(self):
        self.assertEqual(self._run("frame")["decision"]["supported"], ["frame"])

    def test_v2_rejection_glue_on_lidar_truth(self):
        from careport.xfc7_primary_v2_run import verdicts
        res = analyse(_synthetic("lidar"), [], [])
        v = verdicts(res, ["FRONT", "FRONT_LEFT", "SIDE_LEFT"], {"FRONT": 10.0, "FRONT_LEFT": 10.0, "SIDE_LEFT": 10.0})
        self.assertTrue(v["frame"]["ruled_out"])
        self.assertFalse(v["lidar"]["ruled_out"])
        self.assertFalse(v["cam"]["ruled_out"])
        # additive rule, B = 10: thresholds 15 ms; pose is 12.3 ms (FRONT_LEFT) and 24.8 ms (SIDE_LEFT) from LiDAR time
        self.assertEqual(v["pose"]["rejected_in"], ["SIDE_LEFT"])

    def test_recovers_pose(self):
        self.assertEqual(self._run("pose")["decision"]["supported"], ["pose"])

    def test_recovers_cam_by_the_readout_sign_flip(self):
        r = self._run("cam")
        self.assertEqual(r["decision"]["supported"], ["cam"])
        b = r["slopes"]
        self.assertGreater(b["L2R"]["lidar"]["est"], 20.0)     # tau-hat - t_lidar rises with xi under left-to-right
        self.assertLess(b["R2L"]["lidar"]["est"], -40.0)       # and falls under right-to-left


if __name__ == "__main__":
    unittest.main()
