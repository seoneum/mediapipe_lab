from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_models import Dossier, RecommendationEntry, SessionSummary  # noqa: E402
from ondamm_ollama import (  # noqa: E402
    OllamaClient,
    OllamaDossierRag,
    OllamaFrameReviewer,
    approved_dossier_chunks,
    local_clip_chunks,
)


class FakeOllamaTransport:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, method, url, payload, timeout):
        self.calls.append((method, url, payload, timeout))
        if url.endswith("/api/version"):
            return {"version": "test"}
        if url.endswith("/api/tags"):
            return {
                "models": [
                    {"name": "qwen3-vl:2b-instruct", "digest": "chat-digest", "size": 123},
                    {"name": "embeddinggemma:latest", "digest": "embed-digest", "size": 45},
                ]
            }
        if url.endswith("/api/show"):
            if payload["model"] == "embeddinggemma":
                return {"capabilities": ["embedding"], "model_info": {"general.architecture": "gemma3"}}
            return {
                "capabilities": ["completion", "vision", "thinking"],
                "model_info": {
                    "general.architecture": "qwen3_5",
                    "general.parameter_count": 2_000_000_000,
                    "qwen3_5.context_length": 262144,
                },
            }
        if url.endswith("/api/embed"):
            vectors = []
            for text in payload["input"]:
                if "질문" in text or "한 번에 한 단계" in text:
                    vectors.append([1.0, 0.0, 0.0])
                else:
                    vectors.append([0.0, 1.0, 0.0])
            return {"model": payload["model"], "embeddings": vectors}
        if url.endswith("/api/chat"):
            return {"message": {"role": "assistant", "content": "승인 근거를 바탕으로 한 로컬 초안입니다."}}
        raise AssertionError(url)


class OndammOllamaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = FakeOllamaTransport()
        self.client = OllamaClient(
            chat_model="qwen3-vl:2b-instruct",
            embedding_model="embeddinggemma",
            transport=self.transport,
        )

    def test_client_is_fail_closed_to_loopback_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "localhost"):
            OllamaClient(base_url="http://192.168.0.20:11434")
        self.assertEqual(self.client.ping()["version"], "test")

    def test_local_frame_review_uses_native_chat_images_without_remote_upload(self) -> None:
        reviewer = OllamaFrameReviewer(self.client)
        result = reviewer.review(
            frame_data_urls=["data:image/jpeg;base64,/9j/2Q=="],
            event_metadata={"event_type": "repeating_micro_motion"},
        )

        _, url, payload, _ = self.transport.calls[-1]
        self.assertTrue(url.endswith("/api/chat"))
        self.assertEqual(payload["messages"][1]["images"], ["/9j/2Q=="])
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_ctx"], 16384)
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["model"], "qwen3-vl:2b-instruct")
        self.assertEqual(result["local_frame_count"], 1)
        self.assertEqual(result["remote_frame_count"], 0)
        self.assertTrue(result["local_only"])
        self.assertFalse(result["dossier_auto_updated"])

    def test_model_provenance_and_capabilities_are_verified(self) -> None:
        status = self.client.verify_models(require_vision=True)

        self.assertEqual(status["chat_digest"], "chat-digest")
        self.assertEqual(status["chat_architecture"], "qwen3_5")
        self.assertEqual(status["native_context_length"], 262144)
        self.assertEqual(status["configured_context_length"], 16384)
        self.assertIn("vision", status["chat_capabilities"])

    def test_model_digest_mismatch_fails_closed(self) -> None:
        client = OllamaClient(
            chat_model="qwen3-vl:2b-instruct",
            embedding_model="embeddinggemma",
            expected_chat_digest="different-digest",
            transport=self.transport,
        )
        with self.assertRaisesRegex(RuntimeError, "digest"):
            client.verify_models()

    def test_rag_indexes_only_confirmed_and_approved_dossier_text(self) -> None:
        dossier = Dossier.create(
            child_id="child-local-rag",
            display_name="다온",
            age_band="초등 저학년",
            communication_modality="시각 단서",
            confirmed_preferences=["동물 카드"],
            effective_strategies=["한 번에 한 단계"],
        )
        dossier.add_session_summary(
            SessionSummary.create(
                title="승인된 수업",
                activity_name="분류",
                observed_response="카드를 한 장 선택함",
                educator_interpretation="짧은 단계가 도움이 됨",
                approved_by="teacher-a",
            )
        )
        dossier.approved_plan_history.extend(
            [
                RecommendationEntry.create(
                    goal="미승인 목표",
                    summary="이 초안은 검색되면 안 됨",
                    suggested_activities=["미승인 활동"],
                    rationale_lines=["미승인 근거"],
                    drafted_by="teacher-a",
                ),
                RecommendationEntry.create(
                    goal="승인 목표",
                    summary="승인된 계획",
                    suggested_activities=["승인 활동"],
                    rationale_lines=["승인 근거"],
                    drafted_by="teacher-a",
                    approved_by="teacher-a",
                ),
            ]
        )

        chunks = approved_dossier_chunks(dossier)
        all_text = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("승인된 수업", all_text)
        self.assertIn("승인된 계획", all_text)
        self.assertNotIn("검색되면 안 됨", all_text)

    def test_rag_retrieves_child_scoped_evidence_without_persisting_vectors(self) -> None:
        dossier = Dossier.create(
            child_id="child-local-rag",
            display_name="다온",
            age_band="초등 저학년",
            communication_modality="시각 단서",
            confirmed_preferences=["동물 카드"],
            effective_strategies=["한 번에 한 단계"],
        )
        result = OllamaDossierRag(self.client, default_top_k=1).answer(
            dossier=dossier,
            question="질문: 활동을 어떻게 나누면 되나요?",
        )

        self.assertEqual(result["sources"][0]["section"], "effective_strategies")
        self.assertIn("로컬 초안", result["answer"])
        self.assertTrue(result["local_only"])
        self.assertFalse(result["vectors_persisted"])
        self.assertFalse(result["dossier_auto_updated"])

    def test_rag_without_approved_evidence_does_not_call_models(self) -> None:
        dossier = Dossier.create(
            child_id="empty-child",
            display_name="빈 기록",
            age_band="미입력",
            communication_modality="미입력",
        )
        result = OllamaDossierRag(self.client).answer(dossier=dossier, question="무엇을 하나요?")

        self.assertEqual(result["sources"], [])
        self.assertIn("근거를 찾지 못했습니다", result["answer"])
        self.assertEqual(self.transport.calls, [])

    def test_video_search_uses_safe_metadata_and_returns_player_locator(self) -> None:
        dossier = Dossier.create(
            child_id="child-local-rag",
            display_name="다온",
            age_band="초등 저학년",
            communication_modality="시각 단서",
        )
        clips = [
            {
                "clip_id": "clip-safe",
                "child_id": dossier.child_id,
                "event_id": "event-safe",
                "event_type": "repeating_micro_motion",
                "created_at": "2026-08-30T08:00:00+00:00",
                "duration_seconds": 2.7,
                "mode": "camera",
                "media_url": "/media/clips/clip-safe",
                "trigger_values": {
                    "candidate_id": "C001",
                    "occurrence_count": 3,
                    "embedding": [0.1, 0.2],
                    "secret": "not-indexed",
                },
            }
        ]

        chunk_text = local_clip_chunks(clips)[0].text
        self.assertIn("C001", chunk_text)
        self.assertNotIn("embedding", chunk_text)
        self.assertNotIn("not-indexed", chunk_text)

        result = OllamaDossierRag(self.client).answer(
            dossier=dossier,
            question="질문: 세 번째 반복 영상을 찾아줘",
            clips=clips,
            scope="videos",
            history=[{"role": "user", "content": "앞서 반복 후보를 이야기했어"}],
        )

        self.assertEqual(result["scope"], "videos")
        self.assertEqual(result["history_used"], 1)
        self.assertFalse(result["history_persisted"])
        self.assertEqual(result["video_results"][0]["clip_id"], "clip-safe")
        self.assertNotIn("embedding", result["video_results"][0]["trigger_values"])


if __name__ == "__main__":
    unittest.main()
