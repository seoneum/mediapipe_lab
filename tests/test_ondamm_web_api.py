from __future__ import annotations

import json
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
from ondamm_web import ApiRouter, OndammWebService, ValidationError  # noqa: E402


class OndammWebApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ondamm-web-api-test-"))
        dossiers_dir = self.temp_dir / "dossiers"
        exports_dir = self.temp_dir / "exports"
        ondamm_paths.ONDAMM_DOSSIERS = dossiers_dir
        ondamm_paths.ONDAMM_EXPORTS = exports_dir
        ondamm_store.ONDAMM_DOSSIERS = dossiers_dir
        ondamm_store.ONDAMM_EXPORTS = exports_dir
        clip_dir = exports_dir / "artifacts" / "run-ui" / "event-clips"
        clip_dir.mkdir(parents=True)
        self.clip_path = clip_dir / "event-ui.mp4"
        self.clip_path.write_bytes(b"test-video")
        (clip_dir.parent / "event_recording.json").write_text(
            json.dumps(
                {
                    "child_id": "child-api",
                    "mode": "camera",
                    "events": [
                        {
                            "event_id": "event-ui",
                            "event_type": "gaze_diverted",
                            "start_timestamp": 1.0,
                            "end_timestamp": 3.0,
                            "trigger_values": {"gaze_zone": "left"},
                            "clip_path": str(self.clip_path),
                            "created_at": "2026-07-24T10:00:00+00:00",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.gpt_review_calls = []

        class FakeReviewer:
            model = "fake-gpt"

            def review(inner_self, *, frame_data_urls, event_metadata):
                self.gpt_review_calls.append((frame_data_urls, event_metadata))
                return {
                    "response_id": "resp-ui",
                    "model": inner_self.model,
                    "review_text": "관찰 가능한 움직임만 검토했습니다.",
                    "remote_frame_count": len(frame_data_urls),
                    "whole_video_uploaded": False,
                    "dossier_auto_updated": False,
                }

        self.service = OndammWebService(
            clip_analyzer=lambda path: {
                "analysis_engine": "fake_mediapipe",
                "expression_label_counts": {"smile": 2},
                "dominant_expression_hint": "smile",
                "dossier_auto_updated": False,
            },
            gpt_reviewer=FakeReviewer(),
            frame_extractor=lambda path: ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"],
        )
        self.service.create_dossier(
            {
                "child_id": "child-api",
                "display_name": "다온",
                "age_band": "초등 저학년",
                "communication_modality": "시각 단서",
                "confirmed_preferences": ["퍼즐"],
            }
        )
        self.router = ApiRouter(self.service)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_learning_plan_preview_is_five_step_non_authoritative_plan(self) -> None:
        result = self.service.preview_learning_plan(
            "child-api",
            {"goal": "퍼즐 활동 5분 참여", "caregiver_input": "완료 카드에 반응함"},
        )

        self.assertEqual(len(result["steps"]), 5)
        self.assertEqual(result["goal"], "퍼즐 활동 5분 참여")
        self.assertIn("진단", result["support_boundary_notice"])
        self.assertEqual(self.service.get_dossier("child-api")["approved_plan_history"], [])

    def test_router_health_and_dossier_routes(self) -> None:
        health_status, health = self.router.dispatch("GET", "/api/health", None)
        item_status, item = self.router.dispatch("GET", "/api/dossiers/child-api", None)

        self.assertEqual(health_status, 200)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(item_status, 200)
        self.assertEqual(item["display_name"], "다온")

    def test_router_returns_404_for_unknown_route(self) -> None:
        status, payload = self.router.dispatch("GET", "/api/unknown", None)

        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "not_found")

    def test_sensing_demo_returns_unstored_review_draft(self) -> None:
        result = self.service.preview_sensing_demo(
            "child-api",
            {"duration_seconds": 8, "audio_presence_note": "짧은 발성이 들렸음"},
        )

        self.assertGreater(result["frame_count"], 0)
        self.assertFalse(result["storage_policy"]["raw_media_saved"])
        self.assertFalse(result["storage_policy"]["auto_writeback_to_dossier"])
        self.assertEqual(result["expression_label_counts"], {"mouth_smile": 6, "neutral": 24})
        self.assertEqual(result["facial_movement_counts"], {"mouth_smile": 6})
        self.assertEqual(result["eye_closure_state_counts"], {"open_or_uncertain": 30})
        self.assertIn("교사 검토용", " ".join(result["reviewed_note_draft"]))
        self.assertEqual(self.service.get_dossier("child-api")["approved_session_summaries"], [])

    def test_clip_routes_list_local_video_and_run_mediapipe_analysis(self) -> None:
        list_status, clips = self.router.dispatch("GET", "/api/dossiers/child-api/clips", None)
        clip_id = clips[0]["clip_id"]
        analysis_status, result = self.router.dispatch(
            "POST",
            f"/api/dossiers/child-api/clips/{clip_id}/mediapipe",
            {},
        )

        self.assertEqual(list_status, 200)
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["event_type"], "gaze_diverted")
        self.assertEqual(analysis_status, 200)
        self.assertEqual(result["analysis_engine"], "fake_mediapipe")
        self.assertEqual(result["dominant_expression_hint"], "smile")
        self.assertFalse(result["dossier_auto_updated"])

    def test_gpt_clip_review_requires_explicit_remote_frame_consent(self) -> None:
        clip_id = self.service.list_local_clips("child-api")[0]["clip_id"]

        with self.assertRaises(ValidationError):
            self.service.review_local_clip_with_gpt(
                "child-api",
                clip_id,
                {"confirm_remote_frame_upload": False},
            )

        self.assertEqual(self.gpt_review_calls, [])

    def test_gpt_clip_review_sends_only_bounded_frames_and_never_updates_dossier(self) -> None:
        _, clips = self.router.dispatch("GET", "/api/dossiers/child-api/clips", None)
        clip_id = clips[0]["clip_id"]
        status, result = self.router.dispatch(
            "POST",
            f"/api/dossiers/child-api/clips/{clip_id}/gpt-review",
            {"confirm_remote_frame_upload": True},
        )

        self.assertEqual(status, 200)
        self.assertEqual(result["review_text"], "관찰 가능한 움직임만 검토했습니다.")
        self.assertEqual(result["remote_frame_count"], 2)
        self.assertFalse(result["whole_video_uploaded"])
        self.assertFalse(result["dossier_auto_updated"])
        self.assertEqual(len(self.gpt_review_calls), 1)
        self.assertEqual(self.service.get_dossier("child-api")["approved_session_summaries"], [])

    def test_local_ollama_clip_review_requires_no_remote_upload_consent(self) -> None:
        calls = []

        class LocalClient:
            embedding_model = "embeddinggemma"

            @staticmethod
            def ping():
                return {"version": "test-local"}

        class LocalReviewer:
            provider = "ollama"
            local_only = True
            requires_remote_frame_consent = False
            model = "qwen3-vl:2b-instruct"
            client = LocalClient()

            @staticmethod
            def review(*, frame_data_urls, event_metadata):
                calls.append((frame_data_urls, event_metadata))
                return {
                    "provider": "ollama",
                    "model": "qwen3-vl:2b-instruct",
                    "review_text": "로컬에서만 검토했습니다.",
                    "local_frame_count": len(frame_data_urls),
                    "remote_frame_count": 0,
                    "whole_video_uploaded": False,
                    "local_only": True,
                    "dossier_auto_updated": False,
                }

        self.service.frame_reviewer = LocalReviewer()
        self.service.gpt_reviewer = self.service.frame_reviewer
        self.service.llm_provider = "ollama"
        clip_id = self.service.list_local_clips("child-api")[0]["clip_id"]

        status, result = self.router.dispatch(
            "POST",
            f"/api/dossiers/child-api/clips/{clip_id}/llm-review",
            {},
        )

        self.assertEqual(status, 200)
        self.assertEqual(result["local_frame_count"], 2)
        self.assertEqual(result["remote_frame_count"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.service.get_dossier("child-api")["approved_session_summaries"], [])

    def test_local_rag_route_uses_child_dossier_and_never_writes_back(self) -> None:
        calls = []

        class FakeRag:
            @staticmethod
            def answer(*, dossier, question, top_k, clips, scope, history):
                calls.append((dossier.child_id, question, top_k, clips, scope, history))
                return {
                    "provider": "ollama",
                    "answer": "승인된 로컬 근거만 사용했습니다.",
                    "sources": [{"source_id": "dossier:confirmed_preferences:1"}],
                    "video_results": [],
                    "local_only": True,
                    "vectors_persisted": False,
                    "dossier_auto_updated": False,
                }

        self.service.rag_assistant = FakeRag()
        before = self.service.get_dossier("child-api")
        status, result = self.router.dispatch(
            "POST",
            "/api/dossiers/child-api/assistant/query",
            {
                "question": "어떤 활동부터 시작할까요?",
                "top_k": 3,
                "scope": "all",
                "history": [{"role": "user", "content": "이전 질문"}],
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(calls[0][0:3], ("child-api", "어떤 활동부터 시작할까요?", 3))
        self.assertEqual(calls[0][3][0]["child_id"], "child-api")
        self.assertEqual(calls[0][4], "all")
        self.assertEqual(calls[0][5], [{"role": "user", "content": "이전 질문"}])
        self.assertTrue(result["local_only"])
        self.assertFalse(result["dossier_auto_updated"])
        self.assertEqual(self.service.get_dossier("child-api"), before)

    def test_ollama_integration_status_reports_local_provider(self) -> None:
        class LocalClient:
            embedding_model = "embeddinggemma"

            @staticmethod
            def ping():
                return {"version": "test-local"}

        class LocalReviewer:
            provider = "ollama"
            local_only = True
            requires_remote_frame_consent = False
            model = "qwen3-vl:2b-instruct"
            client = LocalClient()

        self.service.frame_reviewer = LocalReviewer()
        self.service.gpt_reviewer = self.service.frame_reviewer
        self.service.llm_provider = "ollama"
        self.service.rag_assistant = type("Rag", (), {"client": LocalClient()})()

        status, integrations = self.router.dispatch("GET", "/api/integrations", None)

        self.assertEqual(status, 200)
        self.assertTrue(integrations["ollama"]["configured"])
        self.assertTrue(integrations["ollama"]["available"])
        self.assertTrue(integrations["llm"]["local_only"])
        self.assertFalse(integrations["llm"]["requires_explicit_frame_consent"])

    def test_event_clip_cross_review_route_keeps_roles_separate_from_dossier(self) -> None:
        clip_id = self.service.list_local_clips("child-api")[0]["clip_id"]

        for role, decision in [
            ("guardian", "accepted"),
            ("teacher", "accepted"),
            ("institutional_social_worker", "uncertain"),
        ]:
            status, result = self.router.dispatch(
                "POST",
                f"/api/dossiers/child-api/clips/{clip_id}/reviews",
                {
                    "reviewer_role": role,
                    "reviewer_name": f"{role}-a",
                    "decision": decision,
                    "observed_facts": "시선 방향이 짧게 왼쪽으로 이동함",
                    "context_comment": "활동 전환 직후",
                },
            )
            self.assertEqual(status, 201)

        read_status, bundle = self.router.dispatch(
            "GET",
            f"/api/dossiers/child-api/clips/{clip_id}/reviews",
            None,
        )
        self.assertEqual(read_status, 200)
        self.assertEqual(bundle["summary"]["status"], "needs_context")
        self.assertEqual(bundle["summary"]["pending_roles"], [])
        self.assertFalse(bundle["dossier_auto_updated"])
        self.assertEqual(self.service.get_dossier("child-api")["approved_session_summaries"], [])

    def test_event_review_rejects_unknown_role(self) -> None:
        clip_id = self.service.list_local_clips("child-api")[0]["clip_id"]
        with self.assertRaises(ValidationError):
            self.service.add_event_review(
                "child-api",
                clip_id,
                {
                    "reviewer_role": "admin",
                    "reviewer_name": "admin-a",
                    "decision": "accepted",
                    "observed_facts": "움직임 확인",
                },
            )

    def test_integrations_route_reports_configured_model(self) -> None:
        status, integrations = self.router.dispatch("GET", "/api/integrations", None)

        self.assertEqual(status, 200)
        self.assertTrue(integrations["openai"]["configured"])
        self.assertEqual(integrations["openai"]["model"], "fake-gpt")
        self.assertTrue(integrations["mediapipe"]["configured"])

    def test_approved_session_backed_facial_profile_updates_dossier_and_runtime_contract(self) -> None:
        session = self.service.add_session(
            "child-api",
            {
                "title": "입꼬리 움직임 검토",
                "activity_name": "표정 따라 하기",
                "observed_response": "mouthDimple 계열 점수가 반복 관찰됨",
                "educator_interpretation": "사용자별 움직임 규칙 후보로 검토함",
                "approved_by": "teacher-a",
                "tags": ["facial-movement-calibration"],
            },
        )

        status, profile = self.router.dispatch(
            "POST",
            "/api/dossiers/child-api/facial-movement-profiles/approve",
            {
                "label": "lip_corner_pull",
                "display_name": "입꼬리 당김 움직임",
                "blendshape_names": ["mouthDimpleLeft", "mouthDimpleRight"],
                "aggregation": "mean",
                "activation_threshold": 0.35,
                "approved_by": "teacher-a",
                "source_session_ids": [session["session_id"]],
            },
        )

        self.assertEqual(status, 201)
        self.assertEqual(profile["status"], "approved")
        saved = self.service.get_dossier("child-api")
        self.assertEqual(saved["approved_facial_movement_profiles"][0]["label"], "lip_corner_pull")
        self.assertEqual(saved["access_audit_records"][-1]["event_type"], "facial_movement_profile_approved")

    def test_facial_profile_rejects_unknown_dossier_session(self) -> None:
        with self.assertRaises(ValidationError):
            self.service.approve_facial_movement_profile(
                "child-api",
                {
                    "label": "lip_corner_pull",
                    "display_name": "입꼬리 당김 움직임",
                    "blendshape_names": ["mouthDimpleLeft"],
                    "aggregation": "max",
                    "activation_threshold": 0.35,
                    "approved_by": "teacher-a",
                    "source_session_ids": ["missing-session"],
                },
            )


if __name__ == "__main__":
    unittest.main()
