from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ondamm_paths  # noqa: E402
import ondamm_store  # noqa: E402
from ondamm_models import Dossier, FacialMovementProfile, SessionSummary  # noqa: E402
from ondamm_recommendations import build_baseline_recommendation  # noqa: E402
from ondamm_cli import render_handoff_markdown  # noqa: E402


class OnDammMvpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ondamm-test-"))
        self.dossiers_dir = self.temp_dir / "dossiers"
        self.exports_dir = self.temp_dir / "exports"

        ondamm_paths.ONDAMM_DATA = self.temp_dir
        ondamm_paths.ONDAMM_DOSSIERS = self.dossiers_dir
        ondamm_paths.ONDAMM_EXPORTS = self.exports_dir
        ondamm_store.ONDAMM_DOSSIERS = self.dossiers_dir
        ondamm_store.ONDAMM_EXPORTS = self.exports_dir

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_dossier_round_trip_preserves_core_fields(self) -> None:
        dossier = Dossier.create(
            child_id="child-a",
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서 + 짧은 구두 지시",
            confirmed_preferences=["동물 카드"],
            effective_strategies=["짧은 단계 제시"],
        )
        ondamm_store.create_dossier(dossier)

        loaded = ondamm_store.load_dossier("child-a")
        self.assertEqual(loaded.child_id, "child-a")
        self.assertEqual(loaded.display_name, "Demo Child")
        self.assertEqual(loaded.confirmed_preferences, ["동물 카드"])
        self.assertEqual(loaded.effective_strategies, ["짧은 단계 제시"])
        self.assertEqual(loaded.canonical_status, "active")

    def test_approved_recommendation_is_persistable(self) -> None:
        dossier = Dossier.create(
            child_id="child-b",
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서",
            confirmed_preferences=["동물 카드"],
            confirmed_avoidances=["갑작스러운 큰 소리"],
            effective_strategies=["짧은 단계 제시"],
            triggers_and_calming_supports=["전환 전에 예고하기"],
        )
        recommendation = build_baseline_recommendation(
            dossier,
            goal="분류 활동 5분 유지",
            caregiver_input="짧은 피드백이 있을 때 더 잘 참여함",
            drafted_by="teacher-a",
            approved_by="teacher-a",
        )
        dossier.add_recommendation(recommendation)
        ondamm_store.save_dossier(dossier)

        loaded = ondamm_store.load_dossier("child-b")
        self.assertEqual(len(loaded.approved_plan_history), 1)
        saved = loaded.approved_plan_history[0]
        self.assertEqual(saved.status, "approved")
        self.assertIn("분류 활동 5분 유지", saved.goal)
        self.assertTrue(any("선호 자극" in item for item in saved.suggested_activities))

    def test_handoff_markdown_contains_manual_reestablishment_notice(self) -> None:
        dossier = Dossier.create(
            child_id="child-c",
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서",
            handoff_notes=["전환 전에 예고하면 안정적임"],
        )
        dossier.add_session_summary(
            SessionSummary.create(
                title="분류 활동 1회차",
                activity_name="동물 분류",
                observed_response="선호 카드 제시 후 참여함",
                educator_interpretation="짧은 단계 제시가 유효함",
                approved_by="teacher-a",
                tags=["전환성공"],
            )
        )

        markdown = render_handoff_markdown(dossier)
        self.assertIn("human-readable handoff artifact", markdown)
        self.assertIn("수동으로 continuity dossier를 다시 작성", markdown)
        self.assertIn("전환 전에 예고하면 안정적임", markdown)
        self.assertIn("분류 활동 1회차", markdown)

    def test_approved_facial_movement_profile_round_trip_and_audit(self) -> None:
        dossier = Dossier.create(
            child_id="child-facial-profile",
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서",
        )
        profile = FacialMovementProfile.create(
            label="lip_corner_pull",
            display_name="입꼬리 당김 움직임",
            blendshape_names=["mouthDimpleLeft", "mouthDimpleRight"],
            aggregation="mean",
            activation_threshold=0.35,
            approved_by="teacher-a",
            source_session_ids=["session-observation-a"],
        )

        dossier.add_facial_movement_profile(profile)
        ondamm_store.create_dossier(dossier)

        loaded = ondamm_store.load_dossier("child-facial-profile")
        self.assertEqual(len(loaded.approved_facial_movement_profiles), 1)
        saved = loaded.approved_facial_movement_profiles[0]
        self.assertEqual(saved.label, "lip_corner_pull")
        self.assertEqual(saved.status, "approved")
        self.assertEqual(saved.source_session_ids, ["session-observation-a"])
        self.assertEqual(loaded.access_audit_records[-1]["event_type"], "facial_movement_profile_approved")

    def test_facial_profile_requires_approved_source_and_never_auto_updates(self) -> None:
        with self.assertRaises(ValueError):
            FacialMovementProfile.create(
                label="lip_corner_pull",
                display_name="입꼬리 당김 움직임",
                blendshape_names=["mouthDimpleLeft"],
                aggregation="max",
                activation_threshold=0.35,
                approved_by="",
                source_session_ids=[],
            )


if __name__ == "__main__":
    unittest.main()
