import unittest

import numpy as np

from careport.xfc7_observations import (
    along_axis_angle_deg,
    anchor_and_gate,
    column_third,
    ego_speed_class,
    hull_iou,
    image_speed,
    motion_sign,
    overlap_flags,
    range_bin,
    translate_hull,
    viewpoint_bin,
)


class MotionTest(unittest.TestCase):
    def test_motion_sign_forward_and_reverse(self):
        h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        v = np.array([[5.0, 0.5, 0.0], [0.2, -4.0, 0.0]])
        np.testing.assert_array_equal(motion_sign(v, h), [1, -1])

    def test_image_speed_along_direction(self):
        p0 = np.array([[110.0, 50.0]])
        p10 = np.array([[100.0, 50.0]])
        d = np.array([[1.0, 0.0]])
        np.testing.assert_allclose(image_speed(p0, p10, d), [1.0])   # 10 px in 10 ms
        np.testing.assert_allclose(image_speed(p0, p10, -d), [-1.0])


class AnchorTest(unittest.TestCase):
    def test_anchor_midpoint_and_gate(self):
        p0 = np.array([[100.0, 200.0]])
        pm = np.array([[140.0, 230.0]])          # 50 px apart
        a, g = anchor_and_gate(p0, pm, np.array([80.0]), np.array([100.0]))
        np.testing.assert_allclose(a, [[120.0, 215.0]])
        np.testing.assert_allclose(g, [0.5 * 50.0 + 40.0 + 0.25 * 90.0])

    def test_translate_hull_keeps_size(self):
        h = np.array([[10.0, 20.0, 30.0, 60.0]])
        np.testing.assert_allclose(translate_hull(h, np.array([[100.0, 100.0]])), [[90.0, 80.0, 110.0, 120.0]])


class OverlapTest(unittest.TestCase):
    def test_iou_and_overlap_flags_within_image(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [100.0, 100.0, 110.0, 110.0]])
        iou = hull_iou(a, a)
        self.assertAlmostEqual(iou[0, 1], 50.0 / 150.0)
        self.assertAlmostEqual(iou[0, 2], 0.0)
        flags = overlap_flags(a, np.array([7, 7, 7]), threshold=0.1)
        np.testing.assert_array_equal(flags, [True, True, False])
        flags2 = overlap_flags(a, np.array([7, 8, 7]), threshold=0.1)   # different images do not overlap
        np.testing.assert_array_equal(flags2, [False, False, False])


class GeometryBinsTest(unittest.TestCase):
    def test_along_axis_angle_is_acute(self):
        cam = np.array([[0.0, 0.0, 0.0]] * 3)
        c = np.array([[10.0, 0.0, 0.0]] * 3)
        v = np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        np.testing.assert_allclose(along_axis_angle_deg(v, cam, c), [0.0, 0.0, 90.0], atol=1e-9)

    def test_viewpoint_bin_signed_12_bins(self):
        heading = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        ray = np.array([[1.0, 0.0]] * 4)
        np.testing.assert_array_equal(viewpoint_bin(heading, ray), [6, 9, 0, 3])

    def test_range_column_and_ego_classes(self):
        np.testing.assert_array_equal(range_bin(np.array([5.0, 15.0, 29.9, 45.0, 79.0, 120.0])), [0, 1, 1, 2, 3, 3])
        np.testing.assert_array_equal(column_third(np.array([0.0, 700.0, 1300.0, 1919.0]), 1920), [0, 1, 2, 2])
        np.testing.assert_array_equal(ego_speed_class(np.array([0.0, 1.99, 2.0, 7.9, 8.0, 30.0])), [0, 0, 1, 1, 2, 2])


class FrustumTest(unittest.TestCase):
    def test_margin_to_nearest_edge(self):
        from careport.xfc7_observations import frustum_margin_deg
        f, W, H = 1000.0, 2000.0, 1000.0            # half-FOV: 45 deg horizontally, 26.57 deg vertically
        m = frustum_margin_deg(np.array([1000.0, 1900.0, 1000.0]), np.array([500.0, 500.0, 990.0]), f, f, 1000.0, 500.0, W, H)
        np.testing.assert_allclose(m[0], np.degrees(np.arctan(0.5)), atol=1e-9)          # centre: vertical edge nearest
        np.testing.assert_allclose(m[1], 45.0 - np.degrees(np.arctan(0.9)), atol=1e-9)   # near the right edge
        np.testing.assert_allclose(m[2], np.degrees(np.arctan(0.5)) - np.degrees(np.arctan(0.49)), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
