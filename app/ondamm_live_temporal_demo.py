"""Live demo adapter joining MediaPipe signals to the temporal product runtime."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ondamm_event_recording import EventMetadata, EventRecordingPolicy, LocalEventClipRecorder
from ondamm_micro_motion import EpisodePolicy, MicroMotionEpisodeDetector
from ondamm_micro_motion_runtime import MicroMotionRuntime, RuntimeOutcome
from ondamm_pattern_memory import PatternMemoryPolicy, PatternMemoryStore
from ondamm_temporal_encoder import TemporalEncoder
from ondamm_temporal_features import PersonalMotionCalibrator, TemporalFeatureAdapter, raw_motion_magnitude


def _event_from_dict(payload: Mapping[str, Any]) -> EventMetadata:
    return EventMetadata(
        event_id=str(payload["event_id"]),
        event_type=str(payload["event_type"]),
        start_timestamp=float(payload["start_timestamp"]),
        end_timestamp=float(payload["end_timestamp"]),
        trigger_values=dict(payload.get("trigger_values", {})),
        clip_path=str(payload["clip_path"]) if payload.get("clip_path") else None,
        created_at=str(payload["created_at"]),
    )


@dataclass(frozen=True)
class LiveTemporalResult:
    requested_events: tuple[EventMetadata, ...]
    finalized_events: tuple[EventMetadata, ...]
    outcome: RuntimeOutcome


class LiveTemporalDemo:
    """Own temporal inference, recurrence state, and demo overlay status."""

    def __init__(
        self,
        *,
        child_id: str,
        checkpoint_path: Path,
        pattern_memory_root: Path,
        clips_dir: Path,
        event_metadata_path: Path,
        record_events: bool,
        clip_fps: float,
        calibration_seconds: float = 3.0,
        onset_z: float = 4.0,
        offset_z: float = 2.0,
        min_episode_seconds: float = 0.2,
        refractory_seconds: float = 0.5,
        min_occurrences_for_clip: int = 3,
        strong_candidate_occurrences: int = 5,
        pre_seconds: float = 1.5,
        post_seconds: float = 1.0,
    ) -> None:
        self.encoder = TemporalEncoder.from_checkpoint(checkpoint_path)
        self.feature_adapter = TemporalFeatureAdapter(self.encoder.spec.feature_names)
        self.motion_calibrator = PersonalMotionCalibrator(calibration_seconds=calibration_seconds)
        self.onset_z = float(onset_z)
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        policy = PatternMemoryPolicy(
            min_occurrences_for_clip=min_occurrences_for_clip,
            strong_candidate_occurrences=max(strong_candidate_occurrences, min_occurrences_for_clip),
        )
        memory_file = pattern_memory_root.expanduser().resolve() / child_id / "memory.json"
        if memory_file.is_file():
            memory = PatternMemoryStore.open_existing(pattern_memory_root, child_id=child_id)
            if memory.encoder_digest != self.encoder.encoder_digest:
                raise RuntimeError(
                    "existing child pattern memory was created by a different temporal encoder; "
                    "use the matching checkpoint or a separate demo child"
                )
        else:
            memory = PatternMemoryStore(
                pattern_memory_root,
                child_id=child_id,
                encoder_digest=self.encoder.encoder_digest,
                embedding_dimension=self.encoder.spec.embedding_dim,
                policy=policy,
            )
        self.occurrence_threshold = memory.policy.min_occurrences_for_clip
        recorder = LocalEventClipRecorder(
            policy=EventRecordingPolicy(
                pre_event_buffer_seconds=pre_seconds,
                clip_tail_seconds=post_seconds,
            ),
            output_dir=clips_dir,
            buffer_enabled=True,
            persist_enabled=record_events,
            output_format="mp4",
            fps=clip_fps,
        )
        detector = MicroMotionEpisodeDetector(
            EpisodePolicy(
                onset_threshold=onset_z,
                offset_threshold=offset_z,
                min_duration_seconds=min_episode_seconds,
                refractory_seconds=refractory_seconds,
            )
        )
        self.runtime = MicroMotionRuntime(
            child_id=child_id,
            encoder=self.encoder,
            episode_detector=detector,
            pattern_memory=memory,
            clip_recorder=recorder,
            event_metadata_path=event_metadata_path,
        )
        self._latest_decision: dict[str, Any] | None = None
        self._motion_score = 0.0
        self._event_saved_until = float("-inf")

    def process(
        self,
        *,
        timestamp: float,
        signal: Mapping[str, Any],
        frame_for_record: np.ndarray,
    ) -> LiveTemporalResult:
        face_detected = bool(signal.get("face_detected"))
        raw_motion = raw_motion_magnitude(signal) if face_detected else 0.0
        self._motion_score = self.motion_calibrator.add(
            timestamp=timestamp,
            raw_motion=raw_motion,
            face_detected=face_detected,
        )
        features = self.feature_adapter.from_signal(signal)
        outcome = self.runtime.add_observation(
            timestamp=timestamp,
            features=features,
            frame=frame_for_record,
            motion_score=self._motion_score if face_detected else 0.0,
            quality_score=1.0 if face_detected else 0.0,
        )
        if outcome.decision:
            self._latest_decision = dict(outcome.decision)
        requested = (
            (_event_from_dict(outcome.requested_event),)
            if outcome.requested_event is not None
            else ()
        )
        finalized = tuple(_event_from_dict(payload) for payload in outcome.finalized_events)
        if finalized:
            self._event_saved_until = float(timestamp) + 3.5
        return LiveTemporalResult(requested, finalized, outcome)

    def overlay_status(self, *, timestamp: float) -> dict[str, Any]:
        decision = self._latest_decision or {}
        return {
            "temporal_enabled": True,
            "checkpoint": self.checkpoint_path.name,
            "calibration_remaining": self.motion_calibrator.remaining_at(timestamp),
            "motion_score": self._motion_score,
            "motion_active": self.motion_calibrator.ready and self._motion_score >= self.onset_z,
            "lifecycle": decision.get("lifecycle"),
            "candidate_id": decision.get("candidate_id"),
            "pattern_id": decision.get("pattern_id"),
            "occurrence_count": decision.get("occurrence_count", 0),
            "occurrence_threshold": self.occurrence_threshold,
            "event_saved": float(timestamp) <= self._event_saved_until,
        }

    def close(self, *, timestamp: float) -> LiveTemporalResult:
        outcome = self.runtime.close(timestamp=timestamp, allow_incomplete_tail=True)
        if outcome.decision:
            self._latest_decision = dict(outcome.decision)
        requested = (
            (_event_from_dict(outcome.requested_event),)
            if outcome.requested_event is not None
            else ()
        )
        finalized = tuple(_event_from_dict(payload) for payload in outcome.finalized_events)
        return LiveTemporalResult(requested, finalized, outcome)
