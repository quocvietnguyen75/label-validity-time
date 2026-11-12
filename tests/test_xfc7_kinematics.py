import unittest

import numpy as np

from careport.xfc7_kinematics import (
    implied_capture_offset,
    pose_time_offset,
    rotate_to_world,
    speed_agreement,
    track_velocity_world,
)


def _rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


class PoseTimeTest(unittest.TestCase):
    def test_recovers_time_on_linear_trajectory(self):
        t = np.array([0.026, 0.038, 0.051, 0.064, 0.076])
        xyz = np.array([100.0, 50.0, 0.0]) + np.outer(t, [10.0, 2.0, 0.0])
        frame_xyz = np.array([100.0, 50.0, 0.0]) + 0.0473 * np.array([10.0, 2.0, 0.0])
        self.assertAlmostEqual(pose_time_offset(frame_xyz, t, xyz), 0.0473, places=9)

    def test_static_ego_returns_nan(self):
        t = np.array([0.026, 0.038, 0.051, 0.064, 0.076])
        xyz = np.tile([1.0, 2.0, 3.0], (5, 1))
        self.assertTrue(np.isnan(pose_time_offset(np.array([1.0, 2.0, 3.0]), t, xyz)))


class TrackVelocityTest(unittest.TestCase):
    def test_central_differences_with_gap_rule(self):
        t = np.array([0.0, 0.1, 0.2, 0.3, 0.7])
        c = np.outer(t, [4.0, 3.0, 0.0])
        v = track_velocity_world(t, c, max_gap=0.15)
        self.assertTrue(np.isnan(v[0]).all())        # no left neighbour
        np.testing.assert_allclose(v[1], [4.0, 3.0, 0.0])
        np.testing.assert_allclose(v[2], [4.0, 3.0, 0.0])
        self.assertTrue(np.isnan(v[3]).all())        # right neighbour 0.4 s away
        self.assertTrue(np.isnan(v[4]).all())

    def test_vehicle_frame_speed_matches_after_rotation(self):
        v_world = np.array([5.0, 5.0, 0.0])
        rots = np.stack([_rot_z(30.0), _rot_z(40.0)])
        v_vehicle = np.einsum("nji,j->ni", rots, v_world)   # R^T v
        back = rotate_to_world(rots, v_vehicle)
        ratio, angle = speed_agreement(np.tile(v_world, (2, 1)), back)
        np.testing.assert_allclose(ratio, 1.0)
        np.testing.assert_allclose(angle, 0.0, atol=1e-9)
        ratio_w, angle_w = speed_agreement(np.tile(v_world, (2, 1)), v_vehicle)  # wrong convention
        np.testing.assert_allclose(ratio_w, 1.0)
        np.testing.assert_allclose(angle_w, [30.0, 40.0])


class ImpliedCaptureTest(unittest.TestCase):
    def test_capture_offset_and_perpendicular_residual(self):
        c = np.array([[10.0, 0.0, 1.0], [5.0, 5.0, 0.0]])
        v = np.array([[5.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        cs = c + v * np.array([[0.04], [0.06]]) + np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        t, perp = implied_capture_offset(c, cs, v)
        np.testing.assert_allclose(t, [0.04, 0.06])
        np.testing.assert_allclose(perp, [0.0, 0.01], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
