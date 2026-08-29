"""Camera/source-agnostic orchestration for temporal movement discovery.

Frame extraction stays in the existing MediaPipe stack.  This runtime receives
the agreed label-free feature vector and optional frame, then connects:

    frozen TCN -> episode segmentation -> personal pattern memory
    -> count-triggered delayed clip persistence -> existing clip catalog/UI
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ondamm_event_recording import EventMetadata, LocalEventClipRecorder
from ondamm_micro_motion import MicroMotionEpisodeDetector, MotionEpisode, generic_motion_score
from ondamm_pattern_memory import PatternDecision, PatternMemoryStore
from ondamm_temporal_encoder import TemporalEncoder


@dataclass(frozen=True)
class RuntimeOutcome:
    episode: dict[str, Any] | None
    decision: dict[str, Any] | None
    requested_event: dict[str, Any] | None
    finalized_events: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "decision": self.decision,
            "requested_event": self.requested_event,
            "finalized_events": list(self.finalized_events),
        }


class MicroMotionRuntime:
    def __init__(
        self,
        *,
        child_id: str,
        encoder: TemporalEncoder,
        episode_detector: MicroMotionEpisodeDetector,
        pattern_memory: PatternMemoryStore,
        clip_recorder: LocalEventClipRecorder,
        event_metadata_path: Path,
        mode: str = "camera-temporal-pattern",
    ) -> None:
        if pattern_memory.child_id != child_id:
            raise ValueError("pattern memory child_id does not match runtime")
        if pattern_memory.encoder_digest != encoder.encoder_digest:
            raise ValueError("pattern memory encoder digest does not match runtime encoder")
        if pattern_memory.embedding_dimension != encoder.spec.embedding_dim:
            raise ValueError("pattern memory embedding dimension does not match encoder")
        if not clip_recorder.buffer_enabled:
            raise ValueError("temporal runtime requires an ephemeral RAM frame buffer")
        self.child_id = child_id
        self.encoder = encoder
        self.episode_detector = episode_detector
        self.pattern_memory = pattern_memory
        self.clip_recorder = clip_recorder
        self.event_metadata_path = event_metadata_path.expanduser().resolve()
        self.mode = mode
        self._features: deque[np.ndarray] = deque(maxlen=encoder.spec.sequence_length)
        self._frame_index = 0

    def add_observation(
        self,
        *,
        timestamp: float,
        features: Mapping[str, float] | Sequence[float],
        frame: np.ndarray | None = None,
        motion_score: float | None = None,
        quality_score: float = 1.0,
    ) -> RuntimeOutcome:
        timestamp = float(timestamp)
        vector = self._feature_vector(features)
        if frame is not None:
            self.clip_recorder.add_frame(frame=frame, timestamp=timestamp)
        finalized = self._finalize_ready(timestamp)
        self._features.append(vector)
        self._frame_index += 1
        if len(self._features) < self.encoder.spec.sequence_length:
            return RuntimeOutcome(None, None, None, tuple(event.to_dict() for event in finalized))
        if (self._frame_index - self.encoder.spec.sequence_length) % self.encoder.spec.stride_frames != 0:
            return RuntimeOutcome(None, None, None, tuple(event.to_dict() for event in finalized))

        sequence = np.stack(self._features)
        embedding = self.encoder.encode(sequence)
        score = generic_motion_score(self.encoder.spec.feature_names, vector) if motion_score is None else float(motion_score)
        episode = self.episode_detector.add_endpoint(
            timestamp=timestamp,
            motion_score=score,
            embedding=embedding,
            quality_score=quality_score,
        )
        if episode is None:
            return RuntimeOutcome(None, None, None, tuple(event.to_dict() for event in finalized))
        decision, requested = self._handle_episode(episode)
        if requested and requested.clip_path:
            self._append_recorded_event(requested)
            finalized.append(requested)
        return RuntimeOutcome(
            episode.to_summary(),
            decision.to_dict(),
            requested.to_dict() if requested else None,
            tuple(event.to_dict() for event in finalized),
        )

    def close(self, *, timestamp: float, allow_incomplete_tail: bool = False) -> RuntimeOutcome:
        episode = self.episode_detector.flush(timestamp=timestamp)
        decision = None
        requested = None
        if episode is not None:
            decision, requested = self._handle_episode(episode)
            if requested and requested.clip_path:
                self._append_recorded_event(requested)
        finalized = self.clip_recorder.flush_pending(
            current_timestamp=timestamp,
            allow_incomplete_tail=allow_incomplete_tail,
        )
        for event in finalized:
            if event.clip_path:
                self._append_recorded_event(event)
        return RuntimeOutcome(
            episode.to_summary() if episode else None,
            decision.to_dict() if decision else None,
            requested.to_dict() if requested else None,
            tuple(event.to_dict() for event in finalized),
        )

    def _handle_episode(self, episode: MotionEpisode) -> tuple[PatternDecision, EventMetadata | None]:
        decision = self.pattern_memory.observe_episode(
            episode_id=episode.episode_id,
            embedding=episode.embedding,
            start_timestamp=episode.start_timestamp,
            end_timestamp=episode.end_timestamp,
            quality_score=episode.quality_score,
        )
        if (
            not decision.clip_required
            or not decision.candidate_id
            or not self.clip_recorder.persist_enabled
        ):
            return decision, None
        event = EventMetadata.create(
            event_type="temporal_movement_candidate",
            start_timestamp=episode.start_timestamp,
            end_timestamp=episode.end_timestamp,
            trigger_values={
                "lifecycle": decision.lifecycle,
                "candidate_id": decision.candidate_id,
                "occurrence_count": decision.occurrence_count,
                "novelty_score": decision.novelty_score,
                "nearest_distance": decision.distance,
                "duration_seconds": episode.duration_seconds,
                "quality_score": episode.quality_score,
                "encoder_digest": self.encoder.encoder_digest,
                "raw_media_policy": "current threshold-crossing episode only",
            },
        )
        self.pattern_memory.attach_source_event(candidate_id=decision.candidate_id, event_id=event.event_id)
        return decision, self.clip_recorder.record_event(event)

    def _finalize_ready(self, timestamp: float) -> list[EventMetadata]:
        finalized = self.clip_recorder.finalize_ready(current_timestamp=timestamp)
        for event in finalized:
            if event.clip_path:
                self._append_recorded_event(event)
        return finalized

    def _feature_vector(self, features: Mapping[str, float] | Sequence[float]) -> np.ndarray:
        if isinstance(features, Mapping):
            missing = [name for name in self.encoder.spec.feature_names if name not in features]
            extras = sorted(set(features) - set(self.encoder.spec.feature_names))
            if missing:
                raise ValueError(f"missing temporal features: {', '.join(missing[:5])}")
            if extras:
                raise ValueError(f"unexpected temporal features: {', '.join(extras[:5])}")
            vector = np.asarray([features[name] for name in self.encoder.spec.feature_names], dtype=np.float32)
        else:
            vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (len(self.encoder.spec.feature_names),) or not np.isfinite(vector).all():
            raise ValueError("temporal feature vector has invalid shape or values")
        return vector

    def _append_recorded_event(self, event: EventMetadata) -> None:
        if not event.clip_path:
            return
        path = self.event_metadata_path
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("could not read temporal event metadata") from exc
        else:
            payload = {
                "schema_version": 1,
                "child_id": self.child_id,
                "mode": self.mode,
                "recording_enabled": True,
                "buffer_policy": "ephemeral RAM; no full-session recording",
                "persistence_policy": "only the threshold-crossing recurrence is persisted",
                "events": [],
            }
        events = payload.setdefault("events", [])
        if any(item.get("event_id") == event.event_id for item in events if isinstance(item, dict)):
            return
        events.append(event.to_dict())
        payload["recorded_event_count"] = len(events)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
