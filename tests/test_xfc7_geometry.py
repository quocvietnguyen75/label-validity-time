import unittest

import numpy as np

from careport.xfc7_geometry import box_corners, hull_2d, hull_inside_image, transform_points


class BoxCornersTest(unittest.TestCase):
    def test_axis_aligned_box(self):
        c = box_corners(np.array([[10.0, 0.0, 1.0]]), np.array([[4.0, 2.0, 1.5]]), np.array([0.0]))
        self.assertEqual(c.shape, (1, 8, 3))
        np.testing.assert_allclose(c[0].min(axis=0), [8.0, -1.0, 0.25])
        np.testing.assert_allclose(c[0].max(axis=0), [12.0, 1.0, 1.75])

    def test_heading_rotates_length_axis(self):
        c = box_corners(np.zeros((1, 3)), np.array([[4.0, 2.0, 2.0]]), np.array([np.pi / 2]))
        np.testing.assert_allclose(c[0].min(axis=0), [-1.0, -2.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(c[0].max(axis=0), [1.0, 2.0, 1.0], atol=1e-12)


class TransformTest(unittest.TestCase):
    def test_single_transform_translation_and_rotation(self):
        t = np.eye(4)
        t[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # 90 degrees about z
        t[:3, 3] = [5.0, 0.0, 0.0]
        p = transform_points(t, np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 3.0]]))
        np.testing.assert_allclose(p, [[5.0, 1.0, 0.0], [3.0, 0.0, 3.0]], atol=1e-12)

    def test_batched_transforms(self):
        t = np.stack([np.eye(4), np.eye(4)])
        t[1, :3, 3] = [0.0, 0.0, 1.0]
        p = transform_points(t, np.zeros((2, 8, 3)))
        self.assertEqual(p.shape, (2, 8, 3))
        np.testing.assert_allclose(p[1, :, 2], 1.0)
        np.testing.assert_allclose(p[0], 0.0)


class HullTest(unittest.TestCase):
    def test_hull_and_validity(self):
        uv = np.array([[[10, 20], [30, 5], [15, 40], [12, 22], [11, 21], [13, 23], [14, 24], [16, 25]]], dtype=float)
        valid = np.ones((1, 8), dtype=bool)
        h, ok = hull_2d(uv, valid)
        np.testing.assert_allclose(h[0], [10, 5, 30, 40])
        self.assertTrue(ok[0])
        valid[0, 3] = False
        h2, ok2 = hull_2d(uv, valid)
        self.assertFalse(ok2[0])

    def test_inside_image_with_margin(self):
        h = np.array([[10.0, 10.0, 100.0, 50.0], [5.0, 10.0, 100.0, 50.0], [10.0, 10.0, 1915.0, 50.0]])
        np.testing.assert_array_equal(hull_inside_image(h, 1920, 1280, margin=10.0), [True, False, False])


if __name__ == "__main__":
    unittest.main()
