from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_event_recording import EventRecordingPolicy, LocalEventClipRecorder  # noqa: E402
from ondamm_micro_motion import EpisodePolicy, MicroMotionEpisodeDetector  # noqa: E402
from ondamm_micro_motion_runtime import MicroMotionRuntime  # noqa: E402
from ondamm_pattern_memory import PatternMemoryPolicy, PatternMemoryStore  # noqa: E402
from ondamm_review import LocalClipCatalog  # noqa: E402
from ondamm_temporal_encoder import (  # noqa: E402
    TemporalEncoder,
    TemporalEncoderSpec,
    build_torch_encoder,
    export_temporal_encoder_checkpoint,
)


ENCODER_DIGEST = "a" * 64


def product_feature_names() -> tuple[str, ...]:
    return tuple(
        [f"bs_fixture_{index:02d}" for index in range(52)]
        + [f"geom_abs_fixture_{index:02d}" for index in range(18)]
        + [f"motion_fixture_{index:02d}" for index in range(9)]
    )


def export_product_fixture(root: Path) -> Path:
    spec = TemporalEncoderSpec(
        feature_names=product_feature_names(),
        sequence_length=60,
        stride_frames=5,
        channels=(64, 64, 64),
        kernel_size=3,
        dropout=0.2,
        embedding_dim=64,
    )
    model = build_torch_encoder(spec)
    checkpoint = root / "encoder_fixture.pt"
    digest = export_temporal_encoder_checkpoint(
        checkpoint,
        spec=spec,
        model_state_dict=model.state_dict(),
        normalization_mean=np.zeros(79, dtype=np.float32),
        normalization_std=np.ones(79, dtype=np.float32),
        metadata={
            "held_out_participant": "fixture-held-out",
            "train_participants": ["fixture-train-a", "fixture-train-b"],
            "best_epoch": 1,
            "normalization": "fixture robust center/scale",
        },
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "features": list(spec.feature_names),
                "feature_counts": {"blendshape": 52, "geometry": 18, "motion": 9, "total": 79},
                "sequence": {"causal": True, "seq_len_frames": 60, "stride_frames": 5},
                "model": {"channels": "64,64,64", "kernel_size": 3},
                "encoder_checkpoints": {
                    "fixture": {"path": str(checkpoint), "sha256": digest}
                },
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


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


def build_runtime_fixture(
    root: Path,
    *,
    output_format: str = "mp4",
) -> tuple[MicroMotionRuntime, LocalEventClipRecorder, PatternMemoryStore, np.ndarray, list[float]]:
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
        output_format=output_format,
        fps=5.0,
    )
    runtime = MicroMotionRuntime(
        child_id="child-a",
        encoder=encoder,
        episode_detector=MicroMotionEpisodeDetector(
            EpisodePolicy(
                onset_threshold=0.2,
                offset_threshold=0.1,
                min_duration_seconds=0.2,
                refractory_seconds=0.5,
            )
        ),
        pattern_memory=memory,
        clip_recorder=recorder,
        event_metadata_path=root / "run" / "event_recording.json",
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    features = [0.1, 0.1, 0.1]
    for timestamp in (0.0, 0.1, 0.2):
        runtime.add_observation(timestamp=timestamp, features=features, frame=frame, motion_score=0.0)
    return runtime, recorder, memory, frame, features


def emit_episode(
    runtime: MicroMotionRuntime,
    *,
    start: float,
    frame: np.ndarray,
    features: list[float],
):
    runtime.add_observation(timestamp=start, features=features, frame=frame, motion_score=0.3)
    runtime.add_observation(timestamp=start + 0.2, features=features, frame=frame, motion_score=0.3)
    return runtime.add_observation(
        timestamp=start + 0.4,
        features=features,
        frame=frame,
        motion_score=0.0,
    )


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
            encoder = TemporalEncoder.from_checkpoint(path, product_contract=False)
            embedding = encoder.encode(np.ones((4, 3), dtype=np.float32))

        self.assertEqual(encoder.encoder_digest, digest)
        self.assertEqual(embedding.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=6)

    def test_overlapping_windows_are_one_episode(self) -> None:
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

    def test_separated_motion_is_multiple_occurrences(self) -> None:
        detector = MicroMotionEpisodeDetector(
            EpisodePolicy(
                onset_threshold=0.2,
                offset_threshold=0.1,
                min_duration_seconds=0.2,
                refractory_seconds=0.5,
            )
        )
        embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        episodes = []
        for start in (1.0, 2.0, 3.0):
            detector.add_endpoint(timestamp=start, motion_score=0.3, embedding=embedding)
            detector.add_endpoint(timestamp=start + 0.2, motion_score=0.3, embedding=embedding)
            episodes.append(
                detector.add_endpoint(timestamp=start + 0.4, motion_score=0.0, embedding=embedding)
            )
        self.assertEqual(len([episode for episode in episodes if episode is not None]), 3)
        self.assertEqual(len({episode.episode_id for episode in episodes}), 3)

    def test_face_loss_resets_temporal_history(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-face-loss-") as temp_dir:
            runtime, _, _, frame, features = build_runtime_fixture(Path(temp_dir))
            self.assertEqual(runtime.temporal_history_count, 3)
            runtime.add_observation(
                timestamp=0.3, features=features, frame=frame, motion_score=0.3
            )
            runtime.reset_temporal_history()
            self.assertEqual(runtime.temporal_history_count, 0)
            self.assertIsNone(runtime.episode_detector.flush(timestamp=0.4))

    def test_reacquisition_requires_warmup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-face-reacquire-") as temp_dir:
            runtime, _, _, frame, features = build_runtime_fixture(Path(temp_dir))
            runtime.reset_temporal_history()
            first = runtime.add_observation(
                timestamp=1.0, features=features, frame=frame, motion_score=0.5
            )
            second = runtime.add_observation(
                timestamp=1.1, features=features, frame=frame, motion_score=0.5
            )
            self.assertIsNone(first.episode)
            self.assertIsNone(second.episode)
            self.assertEqual(runtime.temporal_history_count, 2)

    def test_headless_runtime_still_operates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-headless-runtime-") as temp_dir:
            runtime, _, _, frame, features = build_runtime_fixture(Path(temp_dir))
            outcome = emit_episode(runtime, start=1.0, frame=frame, features=features)
            self.assertEqual(outcome.decision["lifecycle"], "UNKNOWN_OCCURRENCE")
            self.assertEqual(outcome.decision["occurrence_count"], 1)

    def test_missing_checkpoint_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-missing-encoder-") as temp_dir:
            with self.assertRaises(FileNotFoundError):
                TemporalEncoder.from_checkpoint(Path(temp_dir) / "missing.pt")

    def test_feature_order_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-feature-order-") as temp_dir:
            root = Path(temp_dir)
            checkpoint = export_product_fixture(root)
            manifest_path = root / "config.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["features"][0], manifest["features"][1] = manifest["features"][1], manifest["features"][0]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "feature order"):
                TemporalEncoder.from_checkpoint(checkpoint)

    def test_encoder_digest_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-encoder-digest-") as temp_dir:
            root = Path(temp_dir)
            checkpoint = export_product_fixture(root)
            manifest_path = root / "config.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["encoder_checkpoints"]["fixture"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "digest"):
                TemporalEncoder.from_checkpoint(checkpoint)

    def test_promoted_pattern_is_detected_as_known(self) -> None:
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

    def test_rejected_pattern_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-pattern-suppression-") as temp_dir:
            store = PatternMemoryStore(
                Path(temp_dir),
                child_id="child-a",
                encoder_digest=ENCODER_DIGEST,
                embedding_dimension=4,
            )
            first = store.observe_episode(
                episode_id="episode-first",
                embedding=[1.0, 0.0, 0.0, 0.0],
                start_timestamp=1.0,
                end_timestamp=1.4,
                quality_score=0.9,
            )
            store.suppress_candidate(
                candidate_id=first.candidate_id,
                approved_by="review-board-a",
                reason="three-role consensus rejected",
            )

            suppressed = store.observe_episode(
                episode_id="episode-suppressed",
                embedding=[1.0, 0.0, 0.0, 0.0],
                start_timestamp=2.0,
                end_timestamp=2.4,
                quality_score=0.9,
            )
            self.assertEqual(suppressed.lifecycle, "SUPPRESSED")
            self.assertFalse(suppressed.clip_required)
            self.assertEqual(store.public_state()["candidates"], [])

    def test_known_match_precedes_suppression(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-known-suppression-") as temp_dir:
            store = PatternMemoryStore(
                Path(temp_dir),
                child_id="child-a",
                encoder_digest=ENCODER_DIGEST,
                embedding_dimension=4,
            )
            known_vector = [1.0, 0.0, 0.0, 0.0]
            source = [
                store.observe_episode(
                    episode_id=f"episode-known-source-{index}",
                    embedding=known_vector,
                    start_timestamp=float(index),
                    end_timestamp=float(index) + 0.4,
                    quality_score=0.9,
                )
                for index in range(3)
            ]
            store.attach_source_event(candidate_id=source[-1].candidate_id, event_id="event-known")
            pattern = store.promote_candidate(
                candidate_id=source[-1].candidate_id,
                display_name="known",
                approved_by="reviewer",
                source_event_ids=["event-known"],
                distance_threshold=0.05,
            )
            nearby = [0.9, np.sqrt(0.19), 0.0, 0.0]
            rejected = store.observe_episode(
                episode_id="episode-rejected",
                embedding=nearby,
                start_timestamp=5.0,
                end_timestamp=5.4,
                quality_score=0.9,
            )
            store.suppress_candidate(
                candidate_id=rejected.candidate_id,
                approved_by="reviewer",
                reason="rejected nearby pattern",
            )
            decision = store.observe_episode(
                episode_id="episode-known-query",
                embedding=known_vector,
                start_timestamp=6.0,
                end_timestamp=6.4,
                quality_score=0.9,
            )
            self.assertEqual(decision.lifecycle, "KNOWN_OCCURRENCE")
            self.assertEqual(decision.pattern_id, pattern["pattern_id"])

    def test_known_pattern_does_not_create_unknown_cluster(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-known-pattern-") as temp_dir:
            store = PatternMemoryStore(
                Path(temp_dir),
                child_id="child-a",
                encoder_digest=ENCODER_DIGEST,
                embedding_dimension=4,
            )
            decisions = [
                store.observe_episode(
                    episode_id=f"episode-source-{index}",
                    embedding=[1.0, 0.0, 0.0, 0.0],
                    start_timestamp=float(index),
                    end_timestamp=float(index) + 0.4,
                    quality_score=0.9,
                )
                for index in range(3)
            ]
            candidate_id = decisions[-1].candidate_id
            store.attach_source_event(candidate_id=candidate_id, event_id="event-source")
            pattern = store.promote_candidate(
                candidate_id=candidate_id,
                display_name="승인 패턴",
                approved_by="review-board-a",
                source_event_ids=["event-source"],
            )

            known = store.observe_episode(
                episode_id="episode-known-only",
                embedding=[1.0, 0.0, 0.0, 0.0],
                start_timestamp=5.0,
                end_timestamp=5.4,
                quality_score=0.9,
            )
            self.assertEqual(known.lifecycle, "KNOWN_OCCURRENCE")
            self.assertEqual(known.pattern_id, pattern["pattern_id"])
            self.assertEqual(store.public_state()["candidates"], [])

    def test_unknown_first_occurrence_does_not_write_video(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-first-occurrence-") as temp_dir:
            runtime, _, _, frame, features = build_runtime_fixture(Path(temp_dir))
            outcome = emit_episode(runtime, start=1.0, frame=frame, features=features)
            self.assertEqual(outcome.decision["occurrence_count"], 1)
            self.assertIsNone(outcome.requested_event)
            self.assertEqual(list(Path(temp_dir).rglob("*.mp4")), [])

    def test_unknown_second_occurrence_does_not_write_video(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-second-occurrence-") as temp_dir:
            runtime, _, _, frame, features = build_runtime_fixture(Path(temp_dir))
            emit_episode(runtime, start=1.0, frame=frame, features=features)
            outcome = emit_episode(runtime, start=2.0, frame=frame, features=features)
            self.assertEqual(outcome.decision["occurrence_count"], 2)
            self.assertIsNone(outcome.requested_event)
            self.assertEqual(list(Path(temp_dir).rglob("*.mp4")), [])

    def test_unknown_third_occurrence_requests_clip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-third-occurrence-") as temp_dir:
            runtime, recorder, memory, frame, features = build_runtime_fixture(Path(temp_dir))
            emit_episode(runtime, start=1.0, frame=frame, features=features)
            emit_episode(runtime, start=2.0, frame=frame, features=features)
            outcome = emit_episode(runtime, start=3.0, frame=frame, features=features)
            self.assertEqual(outcome.decision["occurrence_count"], 3)
            self.assertEqual(outcome.requested_event["event_type"], "repeating_micro_motion")
            self.assertIsNone(outcome.requested_event["clip_path"])
            self.assertEqual(recorder.pending_event_count, 1)
            self.assertEqual(list(Path(temp_dir).rglob("*.mp4")), [])
            candidate = memory.public_state()["candidates"][0]
            self.assertEqual(candidate["source_event_ids"], [])

    def test_third_clip_failure_retries_on_fourth_occurrence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-clip-retry-") as temp_dir:
            runtime, recorder, _, frame, features = build_runtime_fixture(Path(temp_dir))
            for start in (1.0, 2.0, 3.0):
                emit_episode(runtime, start=start, frame=frame, features=features)
            with patch.object(recorder, "_persist_event", side_effect=RuntimeError("codec failed")):
                with self.assertRaisesRegex(RuntimeError, "codec failed"):
                    runtime.add_observation(
                        timestamp=4.2,
                        features=features,
                        frame=frame,
                        motion_score=0.0,
                    )
            self.assertEqual(recorder.pending_event_count, 0)
            fourth = emit_episode(runtime, start=5.0, frame=frame, features=features)
            self.assertEqual(fourth.decision["occurrence_count"], 4)
            self.assertIsNotNone(fourth.requested_event)
            self.assertEqual(recorder.pending_event_count, 1)

    def test_duplicate_clip_is_not_created(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-no-duplicate-clip-") as temp_dir:
            runtime, _, memory, frame, features = build_runtime_fixture(
                Path(temp_dir), output_format="npz"
            )
            for start in (1.0, 2.0, 3.0):
                emit_episode(runtime, start=start, frame=frame, features=features)
            final = runtime.add_observation(
                timestamp=4.2, features=features, frame=frame, motion_score=0.0
            )
            self.assertEqual(len(final.finalized_events), 1)
            fourth = emit_episode(runtime, start=5.0, frame=frame, features=features)
            self.assertIsNone(fourth.requested_event)
            self.assertEqual(len(list(Path(temp_dir).rglob("*.npz"))), 2)  # vectors.npz + one clip
            self.assertEqual(len(memory.public_state()["candidates"][0]["source_event_ids"]), 1)

    def test_clip_waits_for_post_tail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-post-tail-") as temp_dir:
            runtime, recorder, _, frame, features = build_runtime_fixture(Path(temp_dir))
            for start in (1.0, 2.0, 3.0):
                outcome = emit_episode(runtime, start=start, frame=frame, features=features)
            self.assertIsNotNone(outcome.requested_event)
            before_tail = runtime.add_observation(
                timestamp=4.1,
                features=features,
                frame=frame,
                motion_score=0.0,
            )
            self.assertEqual(before_tail.finalized_events, ())
            self.assertEqual(recorder.pending_event_count, 1)
            at_tail = runtime.add_observation(
                timestamp=4.2,
                features=features,
                frame=frame,
                motion_score=0.0,
            )
            self.assertEqual(len(at_tail.finalized_events), 1)
            self.assertEqual(recorder.pending_event_count, 0)

    def test_finalized_clip_has_pre_and_post_context(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ondamm-clip-context-") as temp_dir:
            runtime, _, memory, frame, features = build_runtime_fixture(
                Path(temp_dir),
                output_format="npz",
            )
            emit_episode(runtime, start=1.0, frame=frame, features=features)
            runtime.add_observation(timestamp=1.5, features=features, frame=frame, motion_score=0.0)
            emit_episode(runtime, start=2.0, frame=frame, features=features)
            emit_episode(runtime, start=3.0, frame=frame, features=features)
            for timestamp in (3.6, 3.8, 4.0, 4.2):
                final = runtime.add_observation(
                    timestamp=timestamp,
                    features=features,
                    frame=frame,
                    motion_score=0.0,
                )
            clip_path = Path(final.finalized_events[0]["clip_path"])
            with np.load(clip_path, allow_pickle=False) as archive:
                timestamps = archive["timestamps"].tolist()
            self.assertEqual(timestamps[0], 1.5)
            self.assertEqual(timestamps[-1], 4.2)
            candidate = memory.public_state()["candidates"][0]
            self.assertEqual(candidate["source_event_ids"], [final.finalized_events[0]["event_id"]])

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
            trigger = metadata["events"][0]["trigger_values"]
            self.assertEqual(trigger["occurrence_count"], 3)
            self.assertEqual(trigger["occurrence_threshold"], 3)
            self.assertIn("motion_score", trigger)
            self.assertIn("nearest_known_pattern", trigger)
            self.assertIn("nearest_known_distance", trigger)
            self.assertEqual(metadata["events"][0]["event_type"], "repeating_micro_motion")
            self.assertEqual(len(list((root / "run" / "event-clips").glob("*.mp4"))), 1)
            catalog_items = LocalClipCatalog(root).list_clips("child-a")
            self.assertEqual(len(catalog_items), 1)
            self.assertEqual(catalog_items[0]["event_type"], "repeating_micro_motion")
            self.assertEqual(catalog_items[0]["trigger_values"]["occurrence_threshold"], 3)


if __name__ == "__main__":
    unittest.main()
