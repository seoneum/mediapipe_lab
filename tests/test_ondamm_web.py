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
import ondamm_security  # noqa: E402
import ondamm_store  # noqa: E402
from ondamm_web import OndammWebService, ValidationError  # noqa: E402


class OndammWebServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ondamm-web-test-"))
        self.dossiers_dir = self.temp_dir / "dossiers"
        self.exports_dir = self.temp_dir / "exports"
        self.secrets_dir = self.temp_dir / "secrets"

        ondamm_paths.ONDAMM_DATA = self.temp_dir
        ondamm_paths.ONDAMM_DOSSIERS = self.dossiers_dir
        ondamm_paths.ONDAMM_EXPORTS = self.exports_dir
        ondamm_paths.ONDAMM_SECRETS = self.secrets_dir
        ondamm_store.ONDAMM_DOSSIERS = self.dossiers_dir
        ondamm_store.ONDAMM_EXPORTS = self.exports_dir
        ondamm_security.ONDAMM_SECRETS = self.secrets_dir

        self.service = OndammWebService()
        self.service.create_dossier(
            {
                "child_id": "child-web",
                "display_name": "하늘",
                "age_band": "초등 저학년",
                "communication_modality": "시각 단서 + 짧은 문장",
                "confirmed_preferences": ["동물 카드"],
                "effective_strategies": ["한 번에 한 단계"],
                "triggers_and_calming_supports": ["전환 10초 전 예고"],
            }
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_list_dossiers_returns_ui_summary(self) -> None:
        items = self.service.list_dossiers()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["child_id"], "child-web")
        self.assertEqual(items[0]["display_name"], "하늘")
        self.assertEqual(items[0]["session_count"], 0)
        self.assertEqual(items[0]["plan_count"], 0)

    def test_recommendation_preview_does_not_modify_dossier(self) -> None:
        preview = self.service.preview_recommendation(
            "child-web",
            {
                "goal": "분류 활동 5분 참여",
                "caregiver_input": "짧은 피드백에 반응함",
                "drafted_by": "teacher-a",
            },
        )

        self.assertEqual(preview["status"], "draft")
        self.assertIsNone(preview["approved_by"])
        dossier = self.service.get_dossier("child-web")
        self.assertEqual(dossier["approved_plan_history"], [])

    def test_approved_recommendation_is_persisted(self) -> None:
        saved = self.service.approve_recommendation(
            "child-web",
            {
                "goal": "분류 활동 5분 참여",
                "caregiver_input": "짧은 피드백에 반응함",
                "drafted_by": "teacher-a",
                "approved_by": "teacher-a",
            },
        )

        self.assertEqual(saved["status"], "approved")
        dossier = self.service.get_dossier("child-web")
        self.assertEqual(len(dossier["approved_plan_history"]), 1)

    def test_withdrawn_dossier_blocks_session_write(self) -> None:
        self.service.change_status(
            "child-web",
            {
                "status": "withdrawn_locked",
                "actor_id": "guardian-a",
                "reason_code": "consent_withdrawn",
                "reason": "보호자 요청",
            },
        )

        with self.assertRaisesRegex(RuntimeError, "withdrawn_locked"):
            self.service.add_session(
                "child-web",
                {
                    "title": "분류 활동",
                    "activity_name": "동물 분류",
                    "observed_response": "카드를 선택함",
                    "educator_interpretation": "시각 단서가 도움이 됨",
                    "approved_by": "teacher-a",
                },
            )

    def test_export_handoff_creates_signed_readable_artifacts(self) -> None:
        result = self.service.export_handoff("child-web", {"actor_id": "teacher-a"})

        export_path = Path(result["export_path"])
        manifest_path = Path(result["manifest_path"])
        self.assertTrue(export_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertIn("수동으로 continuity dossier를 다시 작성", result["markdown"])
        self.assertIn("artifact-", result["manifest"]["artifact_id"])

    def test_child_id_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValidationError):
            self.service.create_dossier(
                {
                    "child_id": "../escape",
                    "display_name": "테스트",
                    "age_band": "초등",
                    "communication_modality": "시각 단서",
                }
            )


if __name__ == "__main__":
    unittest.main()
