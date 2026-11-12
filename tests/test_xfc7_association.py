import unittest

import numpy as np

from careport.xfc7_association import assign_image, cost_matrix


class CostMatrixTest(unittest.TestCase):
    def test_distance_over_gate_plus_width_term_and_gating(self):
        anchors = np.array([[100.0, 100.0], [300.0, 100.0]])
        gates = np.array([50.0, 50.0])
        hull_w = np.array([40.0, 40.0])
        refs = np.array([[110.0, 100.0], [400.0, 100.0]])
        ref_w = np.array([80.0, 40.0])
        c = cost_matrix(anchors, gates, hull_w, refs, ref_w)
        self.assertAlmostEqual(c[0, 0], 10.0 / 50.0 + abs(np.log(2.0)))
        self.assertTrue(np.isinf(c[0, 1]))          # 300 px away, outside the gate
        self.assertTrue(np.isinf(c[1, 1]))          # 100 px away, outside a 50 px gate
        self.assertTrue(np.isinf(c[1, 0]))


class AssignTest(unittest.TestCase):
    def test_hungarian_prefers_global_optimum_and_reports_margins(self):
        # Row 0 is closest to ref 0 but row 1 can only use ref 0 cheaply; global optimum swaps.
        cost = np.array([[0.1, 0.3],
                         [0.2, np.inf]])
        col, margin = assign_image(cost)
        np.testing.assert_array_equal(col, [1, 0])
        self.assertAlmostEqual(margin[0], 0.1 - 0.3)       # second-best (0.1) minus assigned (0.3): negative
        self.assertTrue(np.isinf(margin[1]))               # only one finite candidate

    def test_unassignable_rows_get_minus_one(self):
        cost = np.array([[np.inf, np.inf], [0.5, np.inf], [0.6, 0.7]])
        col, margin = assign_image(cost)
        self.assertEqual(col[0], -1)
        self.assertEqual(sorted(c for c in col if c >= 0), [0, 1])
        self.assertTrue(np.isnan(margin[0]))

    def test_empty_inputs(self):
        col, margin = assign_image(np.zeros((3, 0)))
        np.testing.assert_array_equal(col, [-1, -1, -1])
        col, margin = assign_image(np.zeros((0, 2)))
        self.assertEqual(len(col), 0)


class AssociateImagesTest(unittest.TestCase):
    def test_per_image_matching_and_gate_scale(self):
        import pandas as pd
        from careport.xfc7_association import associate_images
        roster = pd.DataFrame({
            "segment": ["s"] * 3, "frame": [1, 1, 2], "camera": [1, 1, 1], "object_id": ["a", "b", "a"],
            "anchor_u": [100.0, 400.0, 100.0], "anchor_v": [100.0, 100.0, 100.0], "gate": [50.0, 50.0, 50.0],
            "hull_w0": [40.0, 60.0, 40.0], "hull_wmax": [40.0, 60.0, 40.0]})
        boxes = pd.DataFrame({
            "segment": ["s", "s", "s"], "frame": [1, 1, 2], "camera": [1, 1, 1], "camera_object_id": ["h1", "h2", "h3"],
            "u": [105.0, 470.0, 160.0], "v": [100.0, 100.0, 100.0], "w": [40.0, 60.0, 40.0]})
        a = associate_images(roster, boxes, gate_scale=1.0).set_index(["frame", "object_id"])
        self.assertEqual(a.loc[(1, "a"), "camera_object_id"], "h1")
        self.assertTrue(pd.isna(a.loc[(1, "b"), "camera_object_id"]))   # 70 px > 50 px gate
        self.assertTrue(pd.isna(a.loc[(2, "a"), "camera_object_id"]))   # 60 px > 50 px gate
        w = associate_images(roster, boxes, gate_scale=1.5).set_index(["frame", "object_id"])
        self.assertEqual(w.loc[(1, "b"), "camera_object_id"], "h2")     # 70 < 75
        self.assertEqual(w.loc[(2, "a"), "camera_object_id"], "h3")     # 60 < 75


if __name__ == "__main__":
    unittest.main()
