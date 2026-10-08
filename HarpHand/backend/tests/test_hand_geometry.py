import unittest

from core.hand_geometry import adaptive_touch_distance_px, normalized_touch_confidence


class HandGeometryTests(unittest.TestCase):
    def test_touch_distance_scales_with_resolution_and_is_bounded(self):
        self.assertAlmostEqual(adaptive_touch_distance_px(1920, 1080), 19.98)
        self.assertEqual(adaptive_touch_distance_px(160, 120), 6.0)
        self.assertEqual(adaptive_touch_distance_px(7680, 4320), 32.0)

    def test_touch_confidence_uses_the_runtime_threshold(self):
        self.assertEqual(normalized_touch_confidence(0, 20), 1.0)
        self.assertEqual(normalized_touch_confidence(10, 20), 0.5)
        self.assertEqual(normalized_touch_confidence(25, 20), 0.0)


if __name__ == "__main__":
    unittest.main()
