from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from holistic_camera import classify_gaze_direction  # noqa: E402


class HolisticCameraGazeTests(unittest.TestCase):
    def test_small_upward_vertical_bias_remains_center(self) -> None:
        self.assertEqual(classify_gaze_direction(horizontal=0.50, vertical=-0.06), "center")

    def test_clear_upward_iris_offset_is_up(self) -> None:
        self.assertEqual(classify_gaze_direction(horizontal=0.50, vertical=-0.09), "up")

    def test_clear_downward_iris_offset_is_down(self) -> None:
        self.assertEqual(classify_gaze_direction(horizontal=0.50, vertical=0.09), "down")


if __name__ == "__main__":
    unittest.main()
