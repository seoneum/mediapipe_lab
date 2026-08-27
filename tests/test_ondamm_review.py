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

from ondamm_review import (  # noqa: E402
    EventReviewStore,
    LocalClipCatalog,
    analyze_clip_with_mediapipe,
    analyze_video_frames,
    ensure_browser_compatible_mp4,
)


class OndammLocalClipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ondamm-review-test-"))
        self.outputs = self.temp_dir / "outputs" / "ondamm"
        self.run_dir = self.outputs / "artifacts" / "run-a"
        self.clip_dir = self.run_dir / "event-clips"
        self.clip_dir.mkdir(parents=True)
        self.clip_path = self.clip_dir / "event-smile.mp4"
        self.clip_path.write_bytes(b"fake-mp4")
        outside = self.temp_dir / "outside.mp4"
        outside.write_bytes(b"outside")
        (self.run_dir / "event_recording.json").write_text(
            json.dumps(
                {
                    "child_id": "child-a",
                    "mode": "camera",
                    "events": [
                        {
                            "event_id": "event-smile",
                            "event_type": "expression_shifted",
                            "start_timestamp": 1.0,
                            "end_timestamp": 3.5,
                            "trigger_values": {"expression_hint": "smile"},
                            "clip_path": str(self.clip_path),
                            "created_at": "2026-07-24T10:00:00+00:00",
                        },
                        {
                            "event_id": "event-outside",
                            "event_type": "face_missing",
                            "start_timestamp": 4.0,
                            "end_timestamp": 6.0,
                            "trigger_values": {},
                            "clip_path": str(outside),
                            "created_at": "2026-07-24T10:01:00+00:00",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.catalog = LocalClipCatalog(self.outputs)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_catalog_lists_only_existing_clips_inside_outputs_root(self) -> None:
        clips = self.catalog.list_clips("child-a")

        self.assertEqual(len(clips), 1)
        clip = clips[0]
        self.assertEqual(clip["event_id"], "event-smile")
        self.assertEqual(clip["event_type"], "expression_shifted")
        self.assertEqual(clip["duration_seconds"], 2.5)
        self.assertNotIn(str(self.temp_dir), clip["relative_path"])
        self.assertTrue(clip["media_url"].startswith("/media/clips/"))

    def test_catalog_resolves_stable_clip_id_to_safe_path(self) -> None:
        clip = self.catalog.list_clips("child-a")[0]

        resolved = self.catalog.resolve_clip(clip["clip_id"], child_id="child-a")

        self.assertEqual(resolved.path, self.clip_path.resolve())
        with self.assertRaises(FileNotFoundError):
            self.catalog.resolve_clip(clip["clip_id"], child_id="child-b")

    def test_cross_review_store_tracks_latest_role_views_and_consensus(self) -> None:
        clip = self.catalog.resolve_clip(self.catalog.list_clips("child-a")[0]["clip_id"], child_id="child-a")
        store = EventReviewStore(self.outputs / "event-reviews")

        for role, name in [
            ("guardian", "guardian-a"),
            ("teacher", "teacher-a"),
            ("institutional_social_worker", "worker-a"),
        ]:
            bundle = store.add_review(
                child_id="child-a",
                clip=clip,
                reviewer_role=role,
                reviewer_name=name,
                decision="accepted",
                observed_facts="고개가 왼쪽으로 이동한 구간을 확인함",
                context_comment="환경 변화 직후였음",
            )

        self.assertEqual(bundle["summary"]["status"], "consensus_accepted")
        self.assertTrue(bundle["summary"]["ready_for_human_promotion"])
        self.assertEqual(bundle["summary"]["pending_roles"], [])
        self.assertFalse(bundle["dossier_auto_updated"])
        self.assertTrue((self.outputs / "event-reviews" / "child-a" / f"{clip.clip_id}.json").is_file())

    def test_cross_review_store_preserves_revision_and_surfaces_disagreement(self) -> None:
        clip = self.catalog.resolve_clip(self.catalog.list_clips("child-a")[0]["clip_id"], child_id="child-a")
        store = EventReviewStore(self.outputs / "event-reviews")
        for role, decision in [
            ("guardian", "accepted"),
            ("teacher", "rejected"),
            ("institutional_social_worker", "accepted"),
        ]:
            bundle = store.add_review(
                child_id="child-a",
                clip=clip,
                reviewer_role=role,
                reviewer_name=f"{role}-a",
                decision=decision,
                observed_facts="동일한 짧은 움직임 구간을 확인함",
            )

        self.assertEqual(bundle["revision"], 3)
        self.assertEqual(bundle["summary"]["status"], "disagreement")
        self.assertFalse(bundle["summary"]["ready_for_human_promotion"])

        revised = store.add_review(
            child_id="child-a",
            clip=clip,
            reviewer_role="teacher",
            reviewer_name="teacher-a",
            decision="uncertain",
            observed_facts="조명이 바뀌어 추가 맥락이 필요함",
        )
        latest = revised["summary"]["latest_by_role"]["teacher"]
        self.assertIsNotNone(latest["supersedes_review_id"])
        self.assertEqual(revised["summary"]["status"], "needs_context")

    def test_analyze_video_frames_samples_video_with_injected_analyzer(self) -> None:
        import cv2
        import numpy as np

        video = self.clip_dir / "sample.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
        for index in range(10):
            writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
        writer.release()

        result = analyze_video_frames(
            video,
            analyze_frame=lambda frame: "smile" if float(frame.mean()) > 40 else "neutral",
            max_samples=5,
        )

        self.assertEqual(result["sampled_frame_count"], 5)
        self.assertEqual(sum(result["expression_label_counts"].values()), 5)
        self.assertIn(result["dominant_expression_hint"], {"neutral", "smile"})
        self.assertIn("감정 상태", result["non_diagnostic_notice"])

    def test_mediapipe_clip_analysis_uses_injected_face_analyzer(self) -> None:
        import cv2
        import numpy as np

        video = self.clip_dir / "mediapipe-sample.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
        for index in range(4):
            writer.write(np.full((48, 64, 3), 60 + index, dtype=np.uint8))
        writer.release()

        class FakeAnalyzer:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def analyze(self, frame):
                return "smile"

        result = analyze_clip_with_mediapipe(
            video,
            max_samples=3,
            analyzer_factory=FakeAnalyzer,
        )

        self.assertEqual(result["analysis_engine"], "google_mediapipe_holistic_blendshapes")
        self.assertEqual(result["expression_label_counts"], {"smile": 3})
        self.assertFalse(result["dossier_auto_updated"])

    def test_browser_compatible_cache_transcodes_mpeg4_once(self) -> None:
        source = self.clip_dir / "legacy-mpeg4.mp4"
        source.write_bytes(b"legacy-mpeg4")
        cache_dir = self.outputs / ".web-cache"
        calls = []

        def fake_transcoder(input_path: Path, output_path: Path) -> None:
            calls.append((input_path, output_path))
            output_path.write_bytes(b"browser-h264")

        first = ensure_browser_compatible_mp4(
            source,
            cache_dir=cache_dir,
            probe_codec=lambda path: "mpeg4",
            transcoder=fake_transcoder,
        )
        second = ensure_browser_compatible_mp4(
            source,
            cache_dir=cache_dir,
            probe_codec=lambda path: "mpeg4",
            transcoder=fake_transcoder,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.read_bytes(), b"browser-h264")
        self.assertEqual(len(calls), 1)
        self.assertEqual(first.parent, cache_dir.resolve())


if __name__ == "__main__":
    unittest.main()
