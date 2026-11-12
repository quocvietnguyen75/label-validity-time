import unittest

import numpy as np

from careport.xfc7_posthoc import (cell_mean_ci, clip_boxes, closest_row, compose_budget, crop_window, n_in_gate,
                                   paired_iou, ruled_out_up_to, shift_along, tipping_factor)


class TippingTest(unittest.TestCase):
    def test_factor_for_interval_above_and_below(self):
        self.assertAlmostEqual(tipping_factor(25.0, 30.0, 10.0), 2.0)      # (25 - 5) / 10
        self.assertAlmostEqual(tipping_factor(-40.0, -15.0, 5.0), 2.0)     # (15 - 5) / 5
        self.assertEqual(tipping_factor(-3.0, 4.0, 10.0), 0.0)             # never rejected

    def test_ruled_out_threshold_is_second_largest(self):
        self.assertAlmostEqual(ruled_out_up_to({"A": 2.6, "B": 1.5, "C": 1.4, "D": 1.0}), 1.5)
        self.assertEqual(ruled_out_up_to({"A": 3.0, "B": 0.0}), 0.0)


class GateCountTest(unittest.TestCase):
    def test_boxes_inside_each_gate(self):
        anchors = np.array([[0.0, 0.0], [100.0, 0.0]])
        gates = np.array([10.0, 50.0])
        boxes = np.array([[3.0, 4.0], [20.0, 0.0], [130.0, 40.0], [100.0, 49.0]])
        np.testing.assert_array_equal(n_in_gate(anchors, gates, boxes), [1, 2])   # 5 <= 10; 50 <= 50 and 49 <= 50
        np.testing.assert_array_equal(n_in_gate(anchors, gates, np.zeros((0, 2))), [0, 0])


class BudgetCompositionTest(unittest.TestCase):
    def test_max_quadrature_and_sum(self):
        c = (3.0, 4.0, 0.0)
        self.assertAlmostEqual(compose_budget(c, "max"), 4.0)
        self.assertAlmostEqual(compose_budget(c, "quadrature"), 5.0)
        self.assertAlmostEqual(compose_budget(c, "sum"), 7.0)
        with self.assertRaises(ValueError):
            compose_budget(c, "mean")


class IoUTest(unittest.TestCase):
    def test_clip_then_paired_iou(self):
        a = np.array([[-10.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
        b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], [0.0, 0.0, 1.0, 1.0]])
        ca = clip_boxes(a, 100.0, 100.0)
        np.testing.assert_allclose(ca[0], [0.0, 0.0, 10.0, 10.0])
        np.testing.assert_allclose(paired_iou(ca, b), [1.0, 50.0 / 150.0, 0.0])


class ShiftTest(unittest.TestCase):
    def test_shift_along_direction_moves_tau_hat_by_delta(self):
        # tau_hat = ((p0 - u) . d - s z) / sdot ; shifting u by -delta * sdot * d raises tau_hat by delta
        u = np.array([[100.0, 50.0]])
        d = np.array([[0.6, 0.8]])
        sdot = np.array([2.0])
        u2 = shift_along(u, d, sdot, 10.0)
        p0 = np.array([[120.0, 70.0]])
        t1 = ((p0 - u) * d).sum(axis=1) / sdot
        t2 = ((p0 - u2) * d).sum(axis=1) / sdot
        np.testing.assert_allclose(t2 - t1, [10.0])


class DisplayTest(unittest.TestCase):
    def test_cell_mean_interval_only_with_enough_clusters(self):
        y = np.array([1.0, 3.0, 2.0, 6.0, 4.0, 8.0])
        w = np.ones(6)
        seg = np.array(["a", "a", "b", "b", "c", "c"])
        full = cell_mean_ci(y, w, seg, min_clusters=3)
        self.assertAlmostEqual(full["est"], 4.0)
        self.assertTrue(full["lo"] < 4.0 < full["hi"])
        self.assertEqual((full["obs"], full["clusters"]), (6, 3))
        few = cell_mean_ci(y, w, seg, min_clusters=5)
        self.assertAlmostEqual(few["est"], 4.0)
        self.assertTrue(np.isnan(few["lo"]) and np.isnan(few["hi"]))

    def test_closest_row_breaks_ties_by_keys(self):
        v = np.array([47.0, 53.0, 50.0, 49.0, 51.0])
        seg = np.array(["s2", "s1", "s9", "s3", "s1"])
        frame = np.array([5, 9, 1, 2, 3])
        self.assertEqual(closest_row(v, 50.2, seg, frame), 2)            # |50.0 - 50.2| is smallest
        self.assertEqual(closest_row(v, 50.0 + 1.0, seg, frame), 4)      # 51.0 exact
        # all three tie at |diff| = 1: the smaller segment wins, then the smaller frame
        self.assertEqual(closest_row(np.array([49.0, 51.0, 51.0]), 50.0, np.array(["s3", "s1", "s1"]), np.array([1, 7, 4])), 2)

    def test_crop_window_pads_widens_and_clips(self):
        boxes = np.array([[100.0, 100.0, 140.0, 120.0], [120.0, 105.0, 150.0, 125.0]])
        u0, v0, u1, v1 = crop_window(boxes, pad=0.25, min_width=160, width=1920, height=1280)
        self.assertEqual(u1 - u0, 160)                                    # union 50 px wide, padded to 75, widened to 160
        self.assertTrue(u0 <= 100 and u1 >= 150 and v0 <= 100 and v1 >= 125)
        self.assertEqual((v0, v1), (94, 131))                             # union 100-125 (25 px), pad 6.25 each side
        e0, f0, e1, f1 = crop_window(np.array([[5.0, 5.0, 25.0, 20.0]]), pad=0.25, min_width=160, width=1920, height=1280)
        self.assertEqual((e0, f0), (0, 1))                                # shifted inside the image edge, width kept
        self.assertEqual(e1 - e0, 160)


if __name__ == "__main__":
    unittest.main()
