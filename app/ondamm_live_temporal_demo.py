"""Live demo adapter joining MediaPipe signals to the temporal product runtime."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ondamm_event_recording import EventMetadata, EventRecordingPolicy, LocalEventClipRecorder
from ondamm_micro_motion import EpisodePolicy, MicroMotionEpisodeDetector
from ondamm_micro_motion_runtime import MicroMotionRuntime, RuntimeOutcome
from ondamm_pattern_memory import PatternMemoryPolicy, PatternMemoryStore
from ondamm_temporal_encoder import TemporalEncoder
from ondamm_child_metric_encoder import load_child_metric_encoder
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
        metric_checkpoint_path: Path | None = None,
        pattern_memory_root: Path,
        clips_dir: Path,
        event_metadata_path: Path,
        record_events: bool,
        clip_fps: float,
        session_id: str = "runtime-session",
        calibration_seconds: float = 3.0,
        calibration_min_valid_samples: int = 60,
        calibration_min_face_coverage: float = 0.80,
        calibration_min_effective_seconds: float = 2.5,
        face_loss_reset_seconds: float = 0.5,
        onset_z: float = 4.0,
        offset_z: float = 2.0,
        min_episode_seconds: float = 0.2,
        refractory_seconds: float = 0.5,
        min_occurrences_for_clip: int = 3,
        strong_candidate_occurrences: int = 5,
        candidate_distance_threshold: float | None = None,
        pre_seconds: float = 3.0,
        post_seconds: float = 3.0,
        review_frame_size: tuple[int, int] = (960, 540),
        review_buffer_fps: float = 12.0,
    ) -> None:
        self.child_id = str(child_id).strip()
        self.session_id = str(session_id).strip()
        if not self.child_id or not self.session_id:
            raise ValueError("child_id and session_id are required")
        if metric_checkpoint_path is not None:
            self.encoder = load_child_metric_encoder(
                base_checkpoint_path=checkpoint_path,
                metric_checkpoint_path=metric_checkpoint_path,
                child_id=self.child_id,
            )
        else:
            self.encoder = TemporalEncoder.from_checkpoint(
                checkpoint_path
            )

        self.feature_adapter = TemporalFeatureAdapter(
            self.encoder.spec.feature_names
        )
        self.motion_calibrator = PersonalMotionCalibrator(
            calibration_seconds=calibration_seconds,
            minimum_valid_samples=calibration_min_valid_samples,
            minimum_face_coverage=calibration_min_face_coverage,
            minimum_effective_duration=calibration_min_effective_seconds,
        )
        if face_loss_reset_seconds < 0:
            raise ValueError("face_loss_reset_seconds must be non-negative")
        self.face_loss_reset_seconds = float(face_loss_reset_seconds)
        self.onset_z = float(onset_z)
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.metric_checkpoint_path = (
            metric_checkpoint_path.expanduser().resolve()
            if metric_checkpoint_path is not None
            else None
        )
        self.detection_log_path = (
            event_metadata_path.expanduser().resolve().parent
            / "temporal_detections.csv"
        )
        self._logged_episode_ids: set[str] = set()
        if candidate_distance_threshold is None:
            candidate_distance_threshold = float(
                getattr(
                    self.encoder,
                    "candidate_distance_threshold",
                    0.05,
                )
            )

        policy = PatternMemoryPolicy(
            candidate_distance_threshold=float(
                candidate_distance_threshold
            ),
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
            buffer_frame_size=review_frame_size,
            buffer_fps=review_buffer_fps,
        )
        detector = MicroMotionEpisodeDetector(
            EpisodePolicy(
                onset_threshold=onset_z,
                offset_threshold=offset_z,
                min_duration_seconds=min_episode_seconds,
                refractory_seconds=refractory_seconds,
                offset_hold_seconds=0.8,
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
        self._latest_episode: dict[str, Any] | None = None
        self._motion_score = 0.0
        self._event_saved_until = float("-inf")
        self._face_missing_since: float | None = None
        self._face_loss_reset = False

    def process(
        self,
        *,
        timestamp: float,
        signal: Mapping[str, Any],
        frame_for_record: np.ndarray,
    ) -> LiveTemporalResult:
        face_detected = bool(signal.get("face_detected"))

        raw_motion = (
            raw_motion_magnitude(signal)
            if face_detected
            else 0.0
        )

        # calibration 이전 상태를 기억한다.
        calibration_was_ready = self.motion_calibrator.ready

        self._motion_score = self.motion_calibrator.add(
            timestamp=timestamp,
            raw_motion=raw_motion,
            face_detected=face_detected,
        )

        calibration_ready = self.motion_calibrator.ready

        # ---------------------------------------------------------
        # Calibration이 끝나기 전에는 TCN feature history를
        # 절대 쌓지 않는다.
        #
        # frame은 review ring/post-tail 용도로만 전달한다.
        # ---------------------------------------------------------
        if not calibration_ready:
            if not face_detected:
                if self._face_missing_since is None:
                    self._face_missing_since = float(timestamp)
            else:
                self._face_missing_since = None

            outcome = self.runtime.add_frame_only(
                timestamp=timestamp,
                frame=frame_for_record,
            )

        else:
            # -----------------------------------------------------
            # 방금 calibration이 완료된 순간부터 fresh TCN
            # history를 시작한다.
            # -----------------------------------------------------
            if not calibration_was_ready:
                self.runtime.reset_temporal_history()
                self._face_loss_reset = False
                self._face_missing_since = None

            if not face_detected:
                if self._face_missing_since is None:
                    self._face_missing_since = float(timestamp)

                if (
                    not self._face_loss_reset
                    and float(timestamp) - self._face_missing_since + 1e-9
                    >= self.face_loss_reset_seconds
                ):
                    self.runtime.reset_temporal_history()
                    self._face_loss_reset = True

                outcome = self.runtime.add_frame_only(
                    timestamp=timestamp,
                    frame=frame_for_record,
                )

            else:
                if self._face_missing_since is not None:
                    self._face_missing_since = None

                features = self.feature_adapter.from_signal(signal)

                outcome = self.runtime.add_observation(
                    timestamp=timestamp,
                    features=features,
                    frame=frame_for_record,
                    motion_score=self._motion_score,
                    quality_score=1.0,
                )

                # face loss 뒤에는 full causal sequence가 다시 채워져야
                # 정상 temporal detection으로 돌아간다.
                if (
                    self.runtime.temporal_history_count
                    >= self.encoder.spec.sequence_length
                ):
                    self._face_loss_reset = False

        if outcome.decision:
            self._latest_decision = dict(outcome.decision)

        if outcome.episode:
            self._latest_episode = dict(outcome.episode)

        self._append_detection(
            outcome,
            eventized_timestamp=timestamp,
        )

        requested = (
            (_event_from_dict(outcome.requested_event),)
            if outcome.requested_event is not None
            else ()
        )

        finalized = tuple(
            _event_from_dict(payload)
            for payload in outcome.finalized_events
        )

        if finalized:
            self._event_saved_until = float(timestamp) + 3.5

        return LiveTemporalResult(
            requested,
            finalized,
            outcome,
        )

    def overlay_status(self, *, timestamp: float) -> dict[str, Any]:
        decision = self._latest_decision or {}
        episode = self._latest_episode or {}
        tail_ready = self.runtime.clip_recorder.has_ready_pending(current_timestamp=timestamp)
        calibration = self.motion_calibrator.status(timestamp)
        face_lost = self._face_missing_since is not None
        warming_up = (
            self.motion_calibrator.ready
            and not face_lost
            and self.runtime.temporal_history_count < self.encoder.spec.sequence_length
        )
        return {
            "temporal_enabled": True,
            "checkpoint": self.checkpoint_path.name,
            "metric_checkpoint": (
                self.metric_checkpoint_path.name
                if self.metric_checkpoint_path
                else None
            ),
            "embedding_dimension": self.encoder.spec.embedding_dim,
            "calibration_remaining": self.motion_calibrator.remaining_at(timestamp),
            "calibration_ready": self.motion_calibrator.ready,
            "calibration_status": calibration,
            "face_lost": face_lost,
            "warming_up": warming_up,
            "warmup_frames": self.runtime.temporal_history_count,
            "warmup_required_frames": self.encoder.spec.sequence_length,
            "motion_score": self._motion_score,
            "motion_active": self.motion_calibrator.ready and self._motion_score >= self.onset_z,
            "lifecycle": decision.get("lifecycle"),
            "candidate_id": decision.get("candidate_id"),
            "pattern_id": decision.get("pattern_id"),
            "occurrence_count": decision.get("occurrence_count", 0),
            "occurrence_threshold": self.occurrence_threshold,
            "nearest_known_pattern": decision.get("nearest_known_pattern"),
            "nearest_known_distance": decision.get("nearest_known_distance"),
            "quality_score": episode.get("quality_score", 1.0),
            # The tail-ready frame is included in the atomic persistence call
            # immediately after this status is rendered.
            "event_saved": tail_ready or float(timestamp) <= self._event_saved_until,
        }

    def close(self, *, timestamp: float) -> LiveTemporalResult:
        outcome = self.runtime.close(timestamp=timestamp, allow_incomplete_tail=False)
        if outcome.decision:
            self._latest_decision = dict(outcome.decision)
        self._append_detection(outcome, eventized_timestamp=timestamp)
        requested = (
            (_event_from_dict(outcome.requested_event),)
            if outcome.requested_event is not None
            else ()
        )
        finalized = tuple(_event_from_dict(payload) for payload in outcome.finalized_events)
        return LiveTemporalResult(requested, finalized, outcome)

    def set_event_recording(
        self,
        enabled: bool,
    ) -> None:
        """Toggle event clip persistence while live analysis continues."""
        self.runtime.clip_recorder.set_persist_enabled(
            enabled
        )

    @property
    def event_recording_enabled(self) -> bool:
        return bool(
            self.runtime.clip_recorder.persist_enabled
        )

    def abort_without_saving(self) -> None:
        self.runtime.abort_without_saving()

    def _append_detection(
        self,
        outcome: RuntimeOutcome,
        *,
        eventized_timestamp: float,
    ) -> None:
        """Persist derived episode decisions needed for future-session metrics."""
        if outcome.episode is None or outcome.decision is None:
            return
        episode_id = str(outcome.episode["episode_id"])
        if episode_id in self._logged_episode_ids:
            return
        self.detection_log_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "child_id",
            "session_id",
            "detection_id",
            "start_timestamp",
            "end_timestamp",
            "lifecycle",
            "pattern_id",
            "candidate_id",
            "occurrence_count",
            "eventized_timestamp",
        ]
        write_header = not self.detection_log_path.exists()
        with self.detection_log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "child_id": self.child_id,
                    "session_id": self.session_id,
                    "detection_id": episode_id,
                    "start_timestamp": outcome.episode["start_timestamp"],
                    "end_timestamp": outcome.episode["end_timestamp"],
                    "lifecycle": outcome.decision["lifecycle"],
                    "pattern_id": outcome.decision.get("pattern_id") or "",
                    "candidate_id": outcome.decision.get("candidate_id") or "",
                    "occurrence_count": outcome.decision.get("occurrence_count", 0),
                    "eventized_timestamp": float(eventized_timestamp),
                }
            )
        self._logged_episode_ids.add(episode_id)
