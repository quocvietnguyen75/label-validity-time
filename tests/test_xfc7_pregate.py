import unittest

import numpy as np
import pandas as pd

from careport.xfc7_pregate import (
    CAMERA_NAMES,
    EXPECTED_SWEEP_ORDER,
    camera_timing_table,
    expected_order_holds,
    planar_speed,
    population_table,
)


def _timing_frame():
    rows = []
    frame_us = 1_600_000_000_000_000
    delays_ms = {4: 25.0, 2: 37.5, 1: 50.0, 3: 62.5, 5: 75.0}
    for k in range(3):
        f = frame_us + k * 100_000
        for cam, d in delays_ms.items():
            trig = f / 1e6 + d / 1000.0 - 0.010
            rows.append({
                "camera_name": cam,
                "frame_timestamp_micros": f,
                "camera_trigger_time": trig,
                "camera_readout_done_time": trig + 0.020,
                "pose_timestamp": f / 1e6 + d / 1000.0,
                "shutter": 0.004,
            })
    return pd.DataFrame(rows)


class CameraTimingTest(unittest.TestCase):
    def test_table_reports_per_camera_delays_in_ms(self):
        t = camera_timing_table(_timing_frame()).set_index("camera")
        self.assertEqual(t.loc["FRONT", "n_images"], 3)
        self.assertAlmostEqual(t.loc["FRONT", "pose_delay_ms_median"], 50.0, places=3)
        self.assertAlmostEqual(t.loc["SIDE_RIGHT", "trigger_delay_ms_median"], 65.0, places=3)
        self.assertAlmostEqual(t.loc["SIDE_LEFT", "readout_ms_median"], 20.0, places=3)
        self.assertAlmostEqual(t.loc["FRONT_LEFT", "shutter_ms_median"], 4.0, places=3)

    def test_expected_sweep_order(self):
        self.assertEqual(EXPECTED_SWEEP_ORDER, ("SIDE_LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "SIDE_RIGHT"))
        good = {"SIDE_LEFT": 25.0, "FRONT_LEFT": 37.5, "FRONT": 50.0, "FRONT_RIGHT": 62.5, "SIDE_RIGHT": 75.0}
        self.assertTrue(expected_order_holds(good))
        swapped = dict(good, FRONT=70.0)
        self.assertFalse(expected_order_holds(swapped))
        self.assertEqual(CAMERA_NAMES[1], "FRONT")
        self.assertEqual(CAMERA_NAMES[5], "SIDE_RIGHT")


class PopulationTest(unittest.TestCase):
    def setUp(self):
        key = ["seg", 1]
        self.boxes = pd.DataFrame([
            # id, type, points, vx, vy
            {"segment": "s", "frame": 1, "object_id": "moving", "type": 1, "points": 25, "vx": 6.0, "vy": 8.0},
            {"segment": "s", "frame": 1, "object_id": "static", "type": 1, "points": 40, "vx": 0.1, "vy": 0.2},
            {"segment": "s", "frame": 1, "object_id": "ped", "type": 2, "points": 50, "vx": 5.0, "vy": 0.0},
            {"segment": "s", "frame": 1, "object_id": "sparse", "type": 1, "points": 10, "vx": 9.0, "vy": 0.0},
            {"segment": "s", "frame": 1, "object_id": "nosync", "type": 1, "points": 30, "vx": 9.0, "vy": 0.0},
            {"segment": "s", "frame": 1, "object_id": "slow", "type": 1, "points": 30, "vx": 1.0, "vy": 0.0},
        ])
        self.synced = pd.DataFrame([
            {"segment": "s", "frame": 1, "object_id": "moving", "most_visible_camera_name": 1},
            {"segment": "s", "frame": 1, "object_id": "static", "most_visible_camera_name": 1},
            {"segment": "s", "frame": 1, "object_id": "ped", "most_visible_camera_name": 1},
            {"segment": "s", "frame": 1, "object_id": "sparse", "most_visible_camera_name": 1},
            {"segment": "s", "frame": 1, "object_id": "slow", "most_visible_camera_name": 5},
        ])
        del key

    def test_planar_speed(self):
        np.testing.assert_allclose(planar_speed(np.array([3.0, 0.0]), np.array([4.0, 2.0])), [5.0, 2.0])

    def test_population_counts_and_predicted_shift(self):
        t = population_table(self.boxes, self.synced, {"FRONT": 50.0, "SIDE_RIGHT": 75.0}).set_index("camera")
        front = t.loc["FRONT"]
        self.assertEqual(front["eligible_vehicles"], 2)      # moving + static; ped, sparse excluded; nosync has no camera
        self.assertEqual(front["moving_ge_3"], 1)
        self.assertEqual(front["moving_ge_8"], 1)
        self.assertEqual(front["static_le_0_5"], 1)
        self.assertAlmostEqual(front["moving_shift_m_median"], 10.0 * 0.050, places=6)
        side_right = t.loc["SIDE_RIGHT"]
        self.assertEqual(side_right["eligible_vehicles"], 1)  # slow vehicle: neither moving nor static
        self.assertEqual(side_right["moving_ge_3"], 0)
        self.assertEqual(side_right["static_le_0_5"], 0)
        self.assertTrue(np.isnan(side_right["moving_shift_m_median"]))


if __name__ == "__main__":
    unittest.main()
