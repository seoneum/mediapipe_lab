from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_facial_movement import analyze_facial_movements  # noqa: E402
from ondamm_sensing_cli import build_preview_lines  # noqa: E402


class OnDammSensingCliTests(unittest.TestCase):
    def test_build_preview_lines_includes_expression_when_present(self) -> None:
        lines = build_preview_lines(
            face_present=True,
            pose_present=True,
            gaze_zone="left",
            posture_proxy="right_shifted",
            expression_label="smile",
            blendshape_pairs=[("mouthSmileLeft", 0.92), ("mouthSmileRight", 0.90), ("jawOpen", 0.11)],
        )
        self.assertEqual(lines[0], "face=True")
        self.assertIn("gaze=left", lines)
        self.assertIn("posture=right_shifted", lines)
        self.assertIn("expression=smile", lines[-1])
        self.assertIn("mouthSmileLeft:0.92", lines[-1])

    def test_build_preview_lines_omits_expression_line_when_absent(self) -> None:
        lines = build_preview_lines(
            face_present=False,
            pose_present=False,
            gaze_zone="unknown",
            posture_proxy="unavailable",
            expression_label=None,
            blendshape_pairs=None,
        )
        self.assertEqual(lines, [
            "face=False",
            "pose=False",
            "gaze=unknown",
            "posture=unavailable",
        ])

    def test_preview_prioritizes_eye_closure_and_lists_multiple_movement_hints(self) -> None:
        class Category:
            def __init__(self, name, score):
                self.category_name = name
                self.score = score

        analysis = analyze_facial_movements([
            Category("eyeBlinkLeft", 0.5),
            Category("eyeBlinkRight", 0.46),
            Category("mouthSmileLeft", 0.8),
            Category("mouthSmileRight", 0.82),
        ])
        lines = build_preview_lines(
            face_present=True,
            pose_present=True,
            gaze_zone="center",
            posture_proxy="centered",
            expression_label=None,
            blendshape_pairs=None,
            facial_movement_analysis=analysis,
        )

        self.assertIn("eyes=both_closed", lines)
        self.assertTrue(any("movements=eyes_closed,mouth_smile" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
