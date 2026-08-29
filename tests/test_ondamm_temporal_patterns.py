from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_event_recording import EventRecordingPolicy, LocalEventClipRecorder  # noqa: E402
from ondamm_micro_motion import EpisodePolicy, MicroMotionEpisodeDetector  # noqa: E402
from ondamm_micro_motion_runtime import MicroMotionRuntime  # noqa: E402
from ondamm_pattern_memory import PatternMemoryPolicy, PatternMemoryStore  # noqa: E402
from ondamm_temporal_encoder import (  # noqa: E402
    TemporalEncoder,
    TemporalEncoderSpec,
    build_torch_encoder,
    export_temporal_encoder_checkpoint,
)


ENCODER_DIGEST = "a" * 64


def fake_encoder(*, sequence_length: int = 3, stride_frames: int = 1) -> TemporalEncoder:
    spec = TemporalEncoderSpec(
        feature_names=("bs_test", "geom_abs_test", "motion_mean"),
        sequence_length=sequence_length,
        stride_frames=stride_frames,
        channels=(4,),
        embedding_dim=4,
    )

    def encode_batch(batch: np.ndarray) -> np.ndarray:
        result = np.zeros((batch.shape[0], 4), dtype=np.float32)
        result[:, 0] = 1.0
        return result

    return TemporalEncoder(spec=spec, encode_batch=encode_batch, encoder_digest=ENCODER_DIGEST)


class TemporalPatternTests(unittest.TestCase):
    def test_temporal_encoder_rejects_qc_and_nuisance_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            TemporalEncoderSpec(feature_names=("bs_test", "yaw_deg"))
        with self.assertRaisesRegex(ValueError, "forbidden"):
            TemporalEncoderSpec(feature_names=("bs_test", "dino_pca_available"))

    def test_temporal_encoder_checkpoint_round_trip(self) -> None:
        spec = TemporalEncoderSpec(
            feature_names=("bs_a", "geom_abs_b", "motion_c"),
            sequence_length=4,
            stride_frames=1,
            channels=(4,),
            embedding_dim=4,
        )
        model = build_torch_encoder(spec)
        for value in model.state_dict().values():
            value.fill_(0.1)
        with tempfile.TemporaryDirectory(prefix="ondamm-encoder-") as temp_dir:
            path = Path(temp_dir) / "encoder.pt"
            digest = export_temporal_encoder_checkpoint(
                path,
                spec=spec,
                model_state_dict=model.state_dict(),
                normalization_mean=[0.0, 0.0, 0.0],
                normalization_std=[1.0, 1.0, 1.0],
                metadata={"training_split": "session-held-out"},
            )
            encoder = TemporalEncoder.from_checkpoint(path)
            embedding = encoder.encode(np.ones((4, 3), dtype=np.float32))

        self.assertEqual(encoder.encoder_digest, digest)
        self.assertEqual(embedding.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)

    def test_overlapping_endpoints_form_one_episode_and_refractory_separates_next(self) -> None:
        detector = MicroMotionEpisodeDetector(
            EpisodePolicy(onset_threshold=0.2, offset_threshold=0.1, min_duration_seconds=0.2, refractory_seconds=0.5)
        )
        embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.assertIsNone(detector.add_endpoint(timestamp=1.0, motion_score=0.3, embedding=embedding))
        self.assertIsNone(detector.add_endpoint(timestamp=1.2, motion_score=0.3, embedding=embedding))
        episode = detector.add_endpoint(timestamp=1.4, motion_score=0.0, embedding=embedding)

        self.assertIsNotNone(episode)
        self.assertEqual(episode.endpoint_count, 2)
        self.assertEqual(episode.duration_seconds, 0.2)
        self.assertIsNone(detector.add_endpoint(timestamp=1.6, motion_score=0.5, embedding=embedding))
        self.assertIsNone(detector.add_endpoint(timestamp=2.0, motion_score=0.3, embedding=embedding))
        self.assertIsNone(detector.add_endpoint(timestamp=2.2, motion_score=0.3, embedding=embedding))
        second = detector.add_endpoint(timestamp=2.4, motion_score=0.0, embedding=embedding)
        self.assertIsNotNone(second)
        self.assertNotEqual(episode.episode_id, second.episode_id)

    def test_pattern_memory_counts_episodes_then_promotes_frozen_encoder_prototype(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-pattern-memory-") as temp_dir:
            store = PatternMemoryStore(
                Path(temp_dir),
                child_id="child-a",
                encoder_digest=ENCODER_DIGEST,
                embedding_dimension=4,
                policy=PatternMemoryPolicy(min_occurrences_for_clip=3, strong_candidate_occurrences=5),
            )
            embedding = [1.0, 0.0, 0.0, 0.0]
            decisions = [
                store.observe_episode(
                    episode_id=f"episode-{index}",
                    embedding=embedding,
                    start_timestamp=float(index),
                    end_timestamp=float(index) + 0.4,
                    quality_score=0.9,
                )
                for index in range(1, 4)
            ]

            self.assertEqual([item.occurrence_count for item in decisions], [1, 2, 3])
            self.assertEqual([item.clip_required for item in decisions], [False, False, True])
            candidate_id = decisions[-1].candidate_id
            self.assertIsNotNone(candidate_id)
            store.attach_source_event(candidate_id=candidate_id, event_id="event-source")
            pattern = store.promote_candidate(
                candidate_id=candidate_id,
                display_name="입꼬리 짧은 반복 움직임",
                approved_by="teacher-a",
                source_event_ids=["event-source"],
            )
            self.assertEqual(pattern["support_count"], 3)
            self.assertEqual(pattern["encoder_digest"], ENCODER_DIGEST)
            self.assertEqual(pattern["distance_threshold"], 0.05)

            known = store.observe_episode(
                episode_id="episode-known",
                embedding=embedding,
                start_timestamp=10.0,
                end_timestamp=10.4,
                quality_score=0.95,
            )
            self.assertEqual(known.lifecycle, "KNOWN_OCCURRENCE")
            self.assertEqual(known.pattern_id, pattern["pattern_id"])
            self.assertFalse(known.clip_required)

    def test_runtime_persists_only_third_episode_after_post_tail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-temporal-runtime-") as temp_dir:
            root = Path(temp_dir)
            encoder = fake_encoder()
            memory = PatternMemoryStore(
                root / "pattern-memory",
                child_id="child-a",
                encoder_digest=encoder.encoder_digest,
                embedding_dimension=encoder.spec.embedding_dim,
                policy=PatternMemoryPolicy(min_occurrences_for_clip=3, strong_candidate_occurrences=5),
            )
            recorder = LocalEventClipRecorder(
                policy=EventRecordingPolicy(pre_event_buffer_seconds=1.5, clip_tail_seconds=1.0),
                output_dir=root / "run" / "event-clips",
                buffer_enabled=True,
                persist_enabled=True,
                output_format="mp4",
                fps=5.0,
            )
            runtime = MicroMotionRuntime(
                child_id="child-a",
                encoder=encoder,
                episode_detector=MicroMotionEpisodeDetector(
                    EpisodePolicy(onset_threshold=0.2, offset_threshold=0.1, min_duration_seconds=0.2, refractory_seconds=0.5)
                ),
                pattern_memory=memory,
                clip_recorder=recorder,
                event_metadata_path=root / "run" / "event_recording.json",
            )
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            features = [0.1, 0.1, 0.1]
            for timestamp in (0.0, 0.1, 0.2):
                runtime.add_observation(timestamp=timestamp, features=features, frame=frame, motion_score=0.0)

            outcomes = []
            for start in (1.0, 2.0, 3.0):
                runtime.add_observation(timestamp=start, features=features, frame=frame, motion_score=0.3)
                runtime.add_observation(timestamp=start + 0.2, features=features, frame=frame, motion_score=0.3)
                outcomes.append(
                    runtime.add_observation(timestamp=start + 0.4, features=features, frame=frame, motion_score=0.0)
                )

            self.assertEqual([item.decision["occurrence_count"] for item in outcomes], [1, 2, 3])
            self.assertIsNone(outcomes[0].requested_event)
            self.assertIsNone(outcomes[1].requested_event)
            self.assertIsNotNone(outcomes[2].requested_event)
            self.assertIsNone(outcomes[2].requested_event["clip_path"])
            self.assertEqual(recorder.pending_event_count, 1)
            self.assertFalse((root / "run" / "event_recording.json").exists())

            for timestamp in (3.6, 3.8, 4.2):
                final = runtime.add_observation(
                    timestamp=timestamp,
                    features=features,
                    frame=frame,
                    motion_score=0.0,
                )
            self.assertEqual(len(final.finalized_events), 1)
            clip_path = Path(final.finalized_events[0]["clip_path"])
            self.assertTrue(clip_path.is_file())
            metadata = json.loads((root / "run" / "event_recording.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["recorded_event_count"], 1)
            self.assertEqual(metadata["events"][0]["trigger_values"]["occurrence_count"], 3)
            self.assertEqual(len(list((root / "run" / "event-clips").glob("*.mp4"))), 1)


if __name__ == "__main__":
    unittest.main()
