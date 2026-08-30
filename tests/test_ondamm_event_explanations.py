from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_event_recording import EventMetadata, EventRecordingPolicy, LocalEventClipRecorder  # noqa: E402
from ondamm_movement_explanation import (  # noqa: E402
    build_selection_explanation,
    compare_region_profiles,
    summarize_temporal_features,
)
from ondamm_pattern_memory import PatternMemoryPolicy, PatternMemoryStore  # noqa: E402


class OndammEventExplanationTests(unittest.TestCase):
    def test_temporal_summary_names_dominant_region_and_visible_change(self) -> None:
        summary = summarize_temporal_features(
            [
                {
                    "motion_mouth": 0.8,
                    "motion_left_eye": 0.1,
                    "motion_right_eye": 0.1,
                    "motion_left_brow": 0.05,
                    "motion_right_brow": 0.05,
                    "bs_mouthPucker": 0.1,
                },
                {
                    "motion_mouth": 1.0,
                    "motion_left_eye": 0.1,
                    "motion_right_eye": 0.1,
                    "motion_left_brow": 0.05,
                    "motion_right_brow": 0.05,
                    "bs_mouthPucker": 0.5,
                },
            ]
        )

        self.assertEqual(summary["dominant_region"], "mouth")
        self.assertEqual(summary["top_changes"][0]["label"], "입술 오므리기")
        self.assertEqual(summary["top_changes"][0]["change_points"], 40.0)
        self.assertIn("입·턱 주변", summary["plain_summary"])

    def test_pattern_memory_compares_current_regions_with_previous_occurrences(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-explanation-memory-") as temp_dir:
            store = PatternMemoryStore(
                Path(temp_dir),
                child_id="child-a",
                encoder_digest="a" * 64,
                embedding_dimension=4,
                policy=PatternMemoryPolicy(min_occurrences_for_clip=3),
            )
            summary = {
                "dominant_region_label": "입·턱 주변",
                "region_distribution": {
                    "mouth": 0.7,
                    "left_eye": 0.1,
                    "right_eye": 0.1,
                    "left_brow": 0.05,
                    "right_brow": 0.05,
                },
            }
            decision = None
            for index in range(3):
                decision = store.observe_episode(
                    episode_id=f"episode-{index}",
                    embedding=[1.0, 0.0, 0.0, 0.0],
                    start_timestamp=float(index),
                    end_timestamp=float(index) + 0.4,
                    quality_score=1.0,
                    movement_summary=summary,
                )

            self.assertIsNotNone(decision)
            self.assertEqual(decision.occurrence_count, 3)
            self.assertEqual(decision.regional_comparison["similarity_percent"], 100.0)
            explanation, facts = build_selection_explanation(
                occurrence_count=decision.occurrence_count,
                occurrence_threshold=3,
                embedding_distance=decision.distance,
                movement_summary=decision.movement_summary,
                regional_comparison=decision.regional_comparison,
            )
            self.assertIn("검토 기준 3회", explanation)
            self.assertEqual(facts["embedding_similarity_percent"], 100.0)
            candidate = store.public_state()["candidates"][0]
            self.assertEqual(candidate["movement_profile_count"], 3)

    def test_recorded_clip_metadata_separates_video_context_from_event_duration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-explanation-clip-") as temp_dir:
            recorder = LocalEventClipRecorder(
                policy=EventRecordingPolicy(pre_event_buffer_seconds=1.0, clip_tail_seconds=1.0),
                output_dir=Path(temp_dir),
                buffer_enabled=True,
                persist_enabled=True,
                output_format="npz",
            )
            for index in range(41):
                timestamp = index / 10
                recorder.add_frame(
                    frame=np.zeros((2, 2, 3), dtype=np.uint8),
                    timestamp=timestamp,
                )
            event = EventMetadata.create(
                event_type="repeating_micro_motion",
                start_timestamp=2.0,
                end_timestamp=2.5,
                trigger_values={"duration_seconds": 0.5},
            )
            recorder.record_event(event)
            [recorded] = recorder.finalize_ready(current_timestamp=4.0)

            self.assertEqual(recorded.trigger_values["clip_duration_seconds"], 2.5)
            self.assertEqual(recorded.trigger_values["clip_pre_context_seconds"], 1.0)
            self.assertEqual(recorded.trigger_values["clip_post_context_seconds"], 1.0)

    def test_region_similarity_is_not_presented_as_probability(self) -> None:
        current = {"region_distribution": {"mouth": 0.8, "left_eye": 0.2}}
        comparison = compare_region_profiles(current, {"mouth": 0.7, "left_eye": 0.3})

        self.assertEqual(comparison["similarity_percent"], 90.0)
        explanation, _ = build_selection_explanation(
            occurrence_count=3,
            occurrence_threshold=3,
            embedding_distance=0.02,
            movement_summary={"dominant_region_label": "입·턱 주변"},
            regional_comparison=comparison,
        )
        self.assertIn("확정하는 값은 아닙니다", explanation)


if __name__ == "__main__":
    unittest.main()
