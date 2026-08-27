from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_facial_movement import (  # noqa: E402
    MovementRule,
    analyze_facial_movements,
    rules_from_approved_profiles,
)


def blendshapes(**scores: float) -> list[SimpleNamespace]:
    return [SimpleNamespace(category_name=name, score=score) for name, score in scores.items()]


class OnDammFacialMovementTests(unittest.TestCase):
    def test_bilateral_eye_closure_is_reported_independently_from_primary_mouth_movement(self) -> None:
        result = analyze_facial_movements(
            blendshapes(
                eyeBlinkLeft=0.48,
                eyeBlinkRight=0.44,
                mouthSmileLeft=0.91,
                mouthSmileRight=0.89,
            )
        )

        self.assertEqual(result.eye_closure_state, "both_closed")
        self.assertEqual(result.primary_label, "eyes_closed")
        self.assertIn("mouth_smile", result.active_labels)
        self.assertAlmostEqual(result.eye_blink_left, 0.48)
        self.assertAlmostEqual(result.eye_blink_right, 0.44)

    def test_multiple_observable_movement_types_can_be_active_in_one_frame(self) -> None:
        result = analyze_facial_movements(
            blendshapes(
                mouthSmileLeft=0.8,
                mouthSmileRight=0.78,
                jawOpen=0.72,
                browInnerUp=0.65,
                browOuterUpLeft=0.6,
                browOuterUpRight=0.62,
                mouthPucker=0.7,
            )
        )

        self.assertTrue({"mouth_smile", "jaw_open", "brow_raise", "lip_pucker"}.issubset(result.active_labels))
        self.assertEqual(result.eye_closure_state, "open_or_uncertain")
        self.assertNotIn("emotion", result.to_dict())

    def test_approved_dossier_profile_can_add_or_override_a_movement_rule(self) -> None:
        profiles = [
            {
                "label": "lip_corner_pull",
                "display_name": "입꼬리 당김 움직임",
                "blendshape_names": ["mouthDimpleLeft", "mouthDimpleRight"],
                "aggregation": "mean",
                "activation_threshold": 0.35,
                "approved_by": "teacher-a",
                "source_session_ids": ["session-a"],
                "status": "approved",
            }
        ]

        rules = rules_from_approved_profiles(profiles)
        result = analyze_facial_movements(
            blendshapes(mouthDimpleLeft=0.6, mouthDimpleRight=0.5),
            rules=rules,
        )

        self.assertIn("lip_corner_pull", result.active_labels)
        self.assertEqual(result.rule_display_names["lip_corner_pull"], "입꼬리 당김 움직임")

    def test_unapproved_profile_is_not_accepted_as_runtime_rule(self) -> None:
        with self.assertRaisesRegex(ValueError, "approved"):
            rules_from_approved_profiles(
                [
                    {
                        "label": "custom",
                        "display_name": "사용자 정의 움직임",
                        "blendshape_names": ["mouthDimpleLeft"],
                        "aggregation": "max",
                        "activation_threshold": 0.4,
                        "approved_by": "teacher-a",
                        "source_session_ids": [],
                        "status": "draft",
                    }
                ]
            )

    def test_movement_rule_rejects_diagnostic_or_emotion_labels(self) -> None:
        for label in ("emotion_happy", "attention_low", "asd_state", "preference_like"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                MovementRule(label, label, ("mouthSmileLeft",), "max", 0.4)


if __name__ == "__main__":
    unittest.main()
