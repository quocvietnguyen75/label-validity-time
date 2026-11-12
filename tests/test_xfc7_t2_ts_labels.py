import unittest

import numpy as np


class PointsToGlobalTest(unittest.TestCase):
    def test_each_point_uses_the_ego_pose_at_its_own_time(self):
        from careport.xfc7_t2_av2_labels import PoseTrack
        from careport.xfc7_t2_ts_labels import points_to_global
        # ego moves +x at 20 m/s; identity rotation; sensor at the ego origin
        ts = np.array([0, 100_000], dtype=np.int64)                     # microseconds
        P = PoseTrack(ts, np.array([[1.0, 0, 0, 0]] * 2), np.array([[0.0, 0, 0], [2.0, 0, 0]]))
        xyz = np.array([[10.0, 0, 0], [10.0, 0, 0]])
        g = points_to_global(xyz, np.array([0, 50_000], dtype=np.int64), np.eye(4), P)
        np.testing.assert_allclose(g[:, 0], [10.0, 11.0])


class HitTimeTest(unittest.TestCase):
    def test_median_point_time_inside_the_box_relative_to_the_sample(self):
        from careport.xfc7_t2_ts_labels import box_hit_time_ms
        g = np.array([[20.0, 0.0, 1.0], [20.5, 0.3, 1.0], [19.6, -0.2, 1.2], [40.0, 0.0, 1.0]])
        t_us = np.array([1_000_000 - 30_000, 1_000_000 - 34_000, 1_000_000 - 32_000, 1_000_000])
        box = {"cx": 20.0, "cy": 0.0, "cz": 1.0, "l": 4.5, "w": 1.9, "h": 1.6, "yaw": 0.0}
        th, n = box_hit_time_ms(g, t_us, box, sample_us=1_000_000)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(th, -32.0)


class VehicleTest(unittest.TestCase):
    def test_vehicle_categories(self):
        from careport.xfc7_t2_ts_labels import TS_VEHICLE
        self.assertIn("vehicle.car", TS_VEHICLE)
        self.assertNotIn("vehicle.trailer", TS_VEHICLE)
        self.assertNotIn("vehicle.ego_trailer", TS_VEHICLE)


class RowThirdTest(unittest.TestCase):
    def test_inclusion_needs_the_centre_row_third_not_the_centre_column_third(self):
        import pandas as pd
        from careport.xfc7_t2_ts_labels import inclusion
        base = dict(most_visible=True, num_pts=50, inside_all=True, overlap=False, static_proj_ok=True, speed=10.0,
                    sdot=1.0, along_axis_deg=60.0, width=1980.0, height=943.0)
        d = pd.DataFrame([dict(base, p0_u=100.0, p0_v=470.0),          # left column, centre row: in
                          dict(base, p0_u=990.0, p0_v=100.0),          # centre column, top row: out
                          dict(base, p0_u=990.0, p0_v=900.0),          # centre column, bottom row: out
                          dict(base, p0_u=1900.0, p0_v=600.0, speed=0.2)])   # parked, centre row: static
        r = inclusion(d)
        self.assertEqual(r["row_third"].tolist(), [1, 0, 2, 1])
        self.assertEqual(r["incl_moving"].tolist(), [True, False, False, False])
        self.assertEqual(r["incl_static"].tolist(), [False, False, False, True])


if __name__ == "__main__":
    unittest.main()
