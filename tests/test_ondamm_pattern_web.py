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
from ondamm_pattern_memory import PatternMemoryPolicy, PatternMemoryStore  # noqa: E402
from ondamm_web import ApiRouter, OndammWebService  # noqa: E402


class TemporalPatternWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ondamm-pattern-web-"))
        dossiers = self.temp_dir / "dossiers"
        exports = self.temp_dir / "exports"
        ondamm_paths.ONDAMM_DOSSIERS = dossiers
        ondamm_paths.ONDAMM_EXPORTS = exports
        ondamm_store.ONDAMM_DOSSIERS = dossiers
        ondamm_store.ONDAMM_EXPORTS = exports
        self.service = OndammWebService(pattern_memory_root=exports / "pattern-memory")
        self.service.create_dossier(
            {
                "child_id": "child-pattern",
                "display_name": "가온",
                "age_band": "초등",
                "communication_modality": "시각 단서",
            }
        )
        self.router = ApiRouter(self.service)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_explicit_promotion_creates_known_pattern(self) -> None:
        store = PatternMemoryStore(
            self.service.pattern_memory_root,
            child_id="child-pattern",
            encoder_digest="b" * 64,
            embedding_dimension=4,
            policy=PatternMemoryPolicy(min_occurrences_for_clip=3, strong_candidate_occurrences=5),
        )
        decision = None
        for index in range(3):
            decision = store.observe_episode(
                episode_id=f"episode-{index}",
                embedding=[1.0, 0.0, 0.0, 0.0],
                start_timestamp=float(index),
                end_timestamp=float(index) + 0.4,
                quality_score=0.9,
            )
        candidate_id = decision.candidate_id
        store.attach_source_event(candidate_id=candidate_id, event_id="event-pattern")

        run_dir = Path(ondamm_paths.ONDAMM_EXPORTS) / "learning" / "run-pattern"
        clips_dir = run_dir / "event-clips"
        clips_dir.mkdir(parents=True)
        clip_path = clips_dir / "event-pattern.mp4"
        clip_path.write_bytes(b"local-video")
        (run_dir / "event_recording.json").write_text(
            json.dumps(
                {
                    "child_id": "child-pattern",
                    "mode": "camera-temporal-pattern",
                    "events": [
                        {
                            "event_id": "event-pattern",
                            "event_type": "repeating_micro_motion",
                            "start_timestamp": 2.0,
                            "end_timestamp": 2.4,
                            "trigger_values": {"candidate_id": candidate_id, "occurrence_count": 3},
                            "clip_path": str(clip_path),
                            "created_at": "2026-08-30T10:00:00+00:00",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        clip = self.service.list_local_clips("child-pattern")[0]
        for role in ("guardian", "teacher", "institutional_social_worker"):
            self.service.add_event_review(
                "child-pattern",
                clip["clip_id"],
                {
                    "reviewer_role": role,
                    "reviewer_name": role,
                    "decision": "accepted",
                    "observed_facts": "독립된 짧은 입 주변 반복 움직임이 보임",
                },
            )

        status, pattern = self.router.dispatch(
            "POST",
            f"/api/dossiers/child-pattern/patterns/candidates/{candidate_id}/promote",
            {
                "clip_id": clip["clip_id"],
                "display_name": "입 주변 짧은 반복 움직임",
                "approved_by": "review-board-a",
            },
        )

        self.assertEqual(status, 201)
        self.assertEqual(pattern["support_count"], 3)
        self.assertEqual(pattern["encoder_digest"], "b" * 64)
        list_status, state = self.router.dispatch("GET", "/api/dossiers/child-pattern/patterns", None)
        self.assertEqual(list_status, 200)
        self.assertEqual(len(state["known_patterns"]), 1)
        self.assertEqual(state["candidates"], [])
        dossier = self.service.get_dossier("child-pattern")
        audit = dossier["access_audit_records"][-1]
        self.assertEqual(audit["event_type"], "temporal_movement_pattern_approved")
        self.assertEqual(audit["details"]["prototype_digest"], pattern["prototype_digest"])
        self.assertEqual(audit["details"]["display_name"], pattern["display_name"])
        self.assertEqual(audit["details"]["approved_by"], pattern["approved_by"])

    def test_review_does_not_auto_promote_pattern(self) -> None:
        store = PatternMemoryStore(
            self.service.pattern_memory_root,
            child_id="child-pattern",
            encoder_digest="b" * 64,
            embedding_dimension=4,
            policy=PatternMemoryPolicy(min_occurrences_for_clip=3, strong_candidate_occurrences=5),
        )
        decision = None
        for index in range(3):
            decision = store.observe_episode(
                episode_id=f"episode-review-only-{index}",
                embedding=[1.0, 0.0, 0.0, 0.0],
                start_timestamp=float(index),
                end_timestamp=float(index) + 0.4,
                quality_score=0.9,
            )
        candidate_id = decision.candidate_id
        store.attach_source_event(candidate_id=candidate_id, event_id="event-review-only")

        run_dir = Path(ondamm_paths.ONDAMM_EXPORTS) / "learning" / "run-review-only"
        clips_dir = run_dir / "event-clips"
        clips_dir.mkdir(parents=True)
        clip_path = clips_dir / "event-review-only.mp4"
        clip_path.write_bytes(b"local-video")
        (run_dir / "event_recording.json").write_text(
            json.dumps(
                {
                    "child_id": "child-pattern",
                    "mode": "camera-temporal-pattern",
                    "events": [
                        {
                            "event_id": "event-review-only",
                            "event_type": "repeating_micro_motion",
                            "start_timestamp": 2.0,
                            "end_timestamp": 2.4,
                            "trigger_values": {
                                "candidate_id": candidate_id,
                                "occurrence_count": 3,
                                "occurrence_threshold": 3,
                            },
                            "clip_path": str(clip_path),
                            "created_at": "2026-08-30T10:00:00+00:00",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        clip = self.service.list_local_clips("child-pattern")[0]
        for role in ("guardian", "teacher", "institutional_social_worker"):
            bundle = self.service.add_event_review(
                "child-pattern",
                clip["clip_id"],
                {
                    "reviewer_role": role,
                    "reviewer_name": role,
                    "decision": "accepted",
                    "observed_facts": "독립된 짧은 입 주변 반복 움직임이 보임",
                    "context_comment": "review only; no promotion action",
                },
            )

        self.assertEqual(bundle["summary"]["status"], "consensus_accepted")
        self.assertTrue(bundle["summary"]["ready_for_human_promotion"])
        state = store.public_state()
        self.assertEqual(state["known_patterns"], [])
        self.assertEqual(state["candidates"][0]["candidate_id"], candidate_id)
        dossier = self.service.get_dossier("child-pattern")
        self.assertNotIn(
            "temporal_movement_pattern_approved",
            [entry["event_type"] for entry in dossier["access_audit_records"]],
        )


if __name__ == "__main__":
    unittest.main()
