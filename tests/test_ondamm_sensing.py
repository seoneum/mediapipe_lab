from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_sensing import ObservationTally, build_sensing_draft  # noqa: E402


class OnDammSensingTests(unittest.TestCase):
    def test_build_sensing_draft_marks_non_authoritative(self) -> None:
        tally = ObservationTally()
        for _ in range(6):
            tally.add_frame(face_present=True, pose_present=True, gaze_zone="center", posture_proxy="centered")
        for _ in range(2):
            tally.add_frame(face_present=True, pose_present=False, gaze_zone="left", posture_proxy="unavailable")

        draft = build_sensing_draft(
            child_id="child-a",
            local_session_id="sensing-1234",
            duration_seconds=8.0,
            tally=tally,
            optional_audio_presence_note="짧은 발성이 들렸음",
        )

        self.assertEqual(draft.child_id, "child-a")
        self.assertEqual(draft.face_present_ratio, 1.0)
        self.assertEqual(draft.pose_present_ratio, 0.75)
        self.assertEqual(draft.gaze_zone_counts["center"], 6)
        self.assertIn("보조 메모 초안", draft.reviewed_note_draft[0])
        self.assertIn("짧은 발성이 들렸음", " ".join(draft.reviewed_note_draft))
        self.assertFalse(draft.storage_policy["raw_media_saved"])
        self.assertFalse(draft.storage_policy["auto_writeback_to_dossier"])

    def test_reviewed_note_draft_avoids_scoring_language(self) -> None:
        tally = ObservationTally()
        for _ in range(5):
            tally.add_frame(face_present=True, pose_present=True, gaze_zone="center", posture_proxy="centered")

        draft = build_sensing_draft(
            child_id="child-b",
            local_session_id="sensing-5678",
            duration_seconds=5.0,
            tally=tally,
        )
        note_text = " ".join(draft.reviewed_note_draft).lower()
        self.assertNotIn("compliance", note_text)
        self.assertNotIn("ranking", note_text)
        self.assertIn("해석하지 말고", note_text)
        self.assertIn("보조 메모 초안", note_text)

    def test_expression_labels_are_counted_as_movement_hints_not_emotions(self) -> None:
        tally = ObservationTally()
        for label in ["smile", "smile", "neutral", None]:
            tally.add_frame(
                face_present=True,
                pose_present=True,
                gaze_zone="center",
                posture_proxy="centered",
                expression_label=label,
            )

        draft = build_sensing_draft(
            child_id="child-expression",
            local_session_id="sensing-expression",
            duration_seconds=4.0,
            tally=tally,
        )

        self.assertEqual(draft.expression_label_counts, {"neutral": 1, "smile": 2})
        note_text = " ".join(draft.reviewed_note_draft)
        self.assertIn("표정 움직임 힌트", note_text)
        self.assertIn("감정 상태로 확정하지", note_text)
        self.assertNotIn("행복", note_text)


if __name__ == "__main__":
    unittest.main()
