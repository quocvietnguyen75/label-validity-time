import unittest

import numpy as np

from careport.xfc7_timing_model import (
    LEFT_TO_RIGHT,
    RIGHT_TO_LEFT,
    azimuth_about,
    column_time,
    lidar_sweep_time_ms,
)


class AzimuthTest(unittest.TestCase):
    def test_vehicle_axes_about_sensor_position(self):
        sensor = np.array([1.43, 0.0])
        pts = np.array([[11.43, 0.0, 1.0], [1.43, 5.0, 0.0], [1.43, -5.0, 0.0], [-8.57, 0.0, 0.0]])
        np.testing.assert_allclose(np.degrees(azimuth_about(pts, sensor)), [0.0, 90.0, -90.0, 180.0], atol=1e-9)


class LidarSweepTest(unittest.TestCase):
    def test_sweep_phase_follows_sun_et_al_convention(self):
        # Frame start at the backward direction; SIDE_LEFT (+90 deg) at 25 ms, FRONT (0) at 50 ms, SIDE_RIGHT (-90) at 75 ms.
        th = np.radians([90.0, 45.0, 0.0, -45.0, -90.0, 179.999])
        np.testing.assert_allclose(lidar_sweep_time_ms(th), [25.0, 37.5, 50.0, 62.5, 75.0, 0.0], atol=1e-3)

    def test_period_scales(self):
        self.assertAlmostEqual(float(lidar_sweep_time_ms(np.array([0.0]), period_ms=99.0)[0]), 49.5)

    def test_amendment_a1_phase_and_period(self):
        from careport.xfc7_timing_model import A1_PERIOD_MS, A1_PHASE_MS
        self.assertEqual((A1_PHASE_MS, A1_PERIOD_MS), (0.50, 98.96))
        t = lidar_sweep_time_ms(np.radians([0.0, -90.0]), period_ms=A1_PERIOD_MS, phase_ms=A1_PHASE_MS)
        np.testing.assert_allclose(t, [0.50 + 49.48, 0.50 + 74.22])


class ColumnTimeTest(unittest.TestCase):
    def test_left_to_right_and_right_to_left(self):
        trig, shutter, done, w = 10.0, 1.0, 56.0, 1920
        r = done - trig - shutter  # 45
        x = np.array([0.0, 960.0, 1920.0])
        np.testing.assert_allclose(column_time(x, w, trig, shutter, done, LEFT_TO_RIGHT), [10.5, 10.5 + r / 2, 10.5 + r])
        np.testing.assert_allclose(column_time(x, w, trig, shutter, done, RIGHT_TO_LEFT), [10.5 + r, 10.5 + r / 2, 10.5])

    def test_other_directions_rejected(self):
        with self.assertRaises(ValueError):
            column_time(np.array([0.0]), 1920, 0.0, 1.0, 46.0, 1)


if __name__ == "__main__":
    unittest.main()
