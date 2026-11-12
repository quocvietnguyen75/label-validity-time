import unittest

import numpy as np
import pandas as pd


def _quat_z(yaw):
    return np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)


class GeometryTest(unittest.TestCase):
    def test_quaternion_yaw_round_trip(self):
        from careport.xfc7_t2_av2_labels import quat_to_R, yaw_from_quat
        for yaw in (-2.5, -0.3, 0.0, 1.2, 3.0):
            self.assertAlmostEqual(yaw_from_quat(*_quat_z(yaw)), yaw, places=9)
            R = quat_to_R(*_quat_z(yaw))
            np.testing.assert_allclose(R @ [1, 0, 0], [np.cos(yaw), np.sin(yaw), 0], atol=1e-12)

    def test_pose_interpolation_translation_and_rotation(self):
        from careport.xfc7_t2_av2_labels import PoseTrack, yaw_from_quat
        ts = np.array([0, 100_000_000], dtype=np.int64)
        q = np.array([_quat_z(0.0), _quat_z(0.2)])
        t = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        P = PoseTrack(ts, q, t)
        T = P.at(50_000_000)
        np.testing.assert_allclose(T[:3, 3], [0.5, 0, 0], atol=1e-12)
        self.assertAlmostEqual(np.arctan2(T[1, 0], T[0, 0]), 0.1, places=9)

    def test_point_on_optical_axis_projects_to_principal_point(self):
        from careport.xfc7_t2_av2_labels import project_points
        K = np.array([[1000.0, 0, 960], [0, 1000.0, 600], [0, 0, 1]])
        uv, ok = project_points(K, np.eye(4), np.array([[0.0, 0.0, 20.0], [0.0, 0.0, -5.0]]))
        np.testing.assert_allclose(uv[0], [960, 600])
        self.assertTrue(ok[0])
        self.assertFalse(ok[1])                                    # behind the camera


class InteriorTest(unittest.TestCase):
    def test_median_offsets_per_lidar_inside_the_box_only(self):
        from careport.xfc7_t2_av2_labels import interior_offsets
        pts = pd.DataFrame({"x": [10.0, 10.5, 9.8, 10.1, 30.0], "y": [0.0, 0.2, -0.3, 0.1, 0.0], "z": [0.5] * 5,
                            "laser_number": [3, 5, 40, 41, 3], "offset_ns": [20e6, 22e6, 70e6, 74e6, 1e6]})
        box = {"cx": 10.0, "cy": 0.0, "cz": 0.5, "l": 4.0, "w": 2.0, "h": 1.5, "yaw": 0.0}
        (ta, na), (tb, nb) = interior_offsets(pts, [box])[0]
        self.assertEqual((na, nb), (2, 2))
        self.assertAlmostEqual(ta, 21.0)
        self.assertAlmostEqual(tb, 72.0)


class FramesTest(unittest.TestCase):
    def test_frames_within_each_sweep(self):
        from careport.xfc7_t2_av2_labels import frames_in_sweeps
        sw = np.array([0, 100, 200], dtype=np.int64) * 1_000_000
        cam = np.array([40, 90, 140, 190, 240], dtype=np.int64) * 1_000_000
        pairs = frames_in_sweeps(sw, cam)
        self.assertEqual([(s // 1_000_000, f // 1_000_000) for s, f in pairs], [(0, 40), (0, 90), (100, 140), (100, 190), (200, 240)])

    def test_frame_type_from_crossing_offsets(self):
        from careport.xfc7_t2_av2_labels import frame_type
        self.assertEqual(frame_type(39.9, cross_a=40.0, cross_b=90.0), "A")
        self.assertEqual(frame_type(2.5, cross_a=54.0, cross_b=3.5), "B")


class EstimatorRecoveryTest(unittest.TestCase):
    """End to end on synthetic geometry: a reference drawn at the true validity time theta is recovered by
    theta-hat = (p_start - u_ref) . d_hat / sdot (no zero-line offset)."""

    def test_theta_hat_recovers_truth(self):
        from careport.xfc7_t2_av2_labels import box_image_terms
        K = np.array([[1700.0, 0, 1024], [0, 1700.0, 775], [0, 0, 1]])
        # camera looks along ego +x: camera z = ego x, camera x = -ego y, camera y = -ego z
        R_ce = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
        T_cam_from_ego = np.eye(4)
        T_cam_from_ego[:3, :3] = R_ce
        T_ego_from_city = np.eye(4)
        centre = np.array([25.0, 4.0, 0.8])
        size = np.array([4.5, 1.9, 1.6])
        v = np.array([0.0, -12.0, 0.0])                            # crossing right to left in the image
        dt_fs_ms = 40.0
        terms = box_image_terms(K, T_cam_from_ego, T_ego_from_city, centre, size, -np.pi / 2, v, dt_fs_ms, [0.0, 30.0])
        for theta_true in (0.0, 25.0, 60.0):
            u_ref = box_image_terms(K, T_cam_from_ego, T_ego_from_city, centre, size, -np.pi / 2, v, dt_fs_ms, [theta_true])["p"][0]
            theta_hat = float((terms["p"][0] - u_ref) @ terms["d_hat"] / terms["sdot"])
            self.assertAlmostEqual(theta_hat, theta_true, delta=0.02 * max(theta_true, 1.0) + 0.05)


class SeamTest(unittest.TestCase):
    def test_spread_flags_a_box_straddling_the_sweep_seam(self):
        from careport.xfc7_t2_av2_labels import interior_spreads
        pts = pd.DataFrame({"x": [10.0, 10.2, 9.9, 10.1], "y": [0.0, 0.1, -0.1, 0.0], "z": [0.5] * 4,
                            "laser_number": [3, 4, 40, 41], "offset_ns": [1e6, 99e6, 60e6, 62e6]})
        box = {"cx": 10.0, "cy": 0.0, "cz": 0.5, "l": 4.0, "w": 2.0, "h": 1.5, "yaw": 0.0}
        sa, sb = interior_spreads(pts, [box])[0]
        self.assertAlmostEqual(sa, 98.0)                            # LiDAR A straddles its seam
        self.assertAlmostEqual(sb, 2.0)


if __name__ == "__main__":
    unittest.main()
