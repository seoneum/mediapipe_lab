from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


DEFAULT_GAZE_DIVERTED_ZONES = ("left", "right", "up", "down")
DEFAULT_POSTURE_SHIFTED_VALUES = ("left_shifted", "right_shifted")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EventRecordingPolicy:
    face_missing_min_seconds: float = 2.0
    gaze_diverted_min_seconds: float = 2.0
    posture_shifted_min_seconds: float = 3.0
    facial_movement_min_seconds: float = 0.4
    pre_event_buffer_seconds: float = 1.5
    clip_tail_seconds: float = 0.0
    gaze_diverted_zones: tuple[str, ...] = DEFAULT_GAZE_DIVERTED_ZONES
    posture_shifted_values: tuple[str, ...] = DEFAULT_POSTURE_SHIFTED_VALUES
    target_facial_movement_labels: tuple[str, ...] = ()

    def threshold_for(self, event_type: str) -> float:
        if event_type == "face_missing":
            return self.face_missing_min_seconds
        if event_type == "gaze_diverted":
            return self.gaze_diverted_min_seconds
        if event_type == "posture_shifted":
            return self.posture_shifted_min_seconds
        if event_type == "facial_movement_detected":
            return self.facial_movement_min_seconds
        raise ValueError(f"Unsupported event type: {event_type}")

    def history_window_seconds(self) -> float:
        return max(
            self.face_missing_min_seconds,
            self.gaze_diverted_min_seconds,
            self.posture_shifted_min_seconds,
            self.facial_movement_min_seconds,
        ) + self.pre_event_buffer_seconds + self.clip_tail_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventObservation:
    timestamp: float
    face_present: bool
    gaze_zone: str
    posture_proxy: str
    facial_movement_labels: tuple[str, ...] = ()


@dataclass
class EventMetadata:
    event_id: str
    event_type: str
    start_timestamp: float
    end_timestamp: float
    trigger_values: dict[str, Any]
    clip_path: str | None = None
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        start_timestamp: float,
        end_timestamp: float,
        trigger_values: dict[str, Any],
        clip_path: str | None = None,
    ) -> "EventMetadata":
        return cls(
            event_id=f"event-{uuid4().hex[:8]}",
            event_type=event_type,
            start_timestamp=round(float(start_timestamp), 3),
            end_timestamp=round(float(end_timestamp), 3),
            trigger_values=trigger_values,
            clip_path=clip_path,
        )

    def with_clip_path(self, clip_path: str | None) -> "EventMetadata":
        payload = self.to_dict()
        payload["clip_path"] = clip_path
        return EventMetadata(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PendingEventState:
    start_timestamp: float | None = None
    emitted: bool = False
    trigger_values: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.start_timestamp = None
        self.emitted = False
        self.trigger_values = {}


class SustainedEventDetector:
    def __init__(self, policy: EventRecordingPolicy | None = None) -> None:
        self.policy = policy or EventRecordingPolicy()
        self._states = {
            "face_missing": _PendingEventState(),
            "gaze_diverted": _PendingEventState(),
            "posture_shifted": _PendingEventState(),
            "facial_movement_detected": _PendingEventState(),
        }

    def add_observation(self, observation: EventObservation) -> list[EventMetadata]:
        events: list[EventMetadata] = []
        for event_type, abnormal, trigger_values in self._iter_checks(observation):
            state = self._states[event_type]
            if abnormal:
                if state.start_timestamp is None:
                    state.start_timestamp = observation.timestamp
                    state.trigger_values = dict(trigger_values)
                else:
                    state.trigger_values.update(trigger_values)
                if not state.emitted:
                    duration = observation.timestamp - state.start_timestamp
                    if duration + 1e-9 >= self.policy.threshold_for(event_type):
                        emitted_trigger_values = dict(state.trigger_values)
                        emitted_trigger_values["duration_seconds"] = round(duration, 3)
                        events.append(
                            EventMetadata.create(
                                event_type=event_type,
                                start_timestamp=state.start_timestamp,
                                end_timestamp=observation.timestamp,
                                trigger_values=emitted_trigger_values,
                            )
                        )
                        state.emitted = True
                continue
            state.reset()
        return events

    def reset(self) -> None:
        for state in self._states.values():
            state.reset()

    def _iter_checks(self, observation: EventObservation) -> list[tuple[str, bool, dict[str, Any]]]:
        gaze_zone = observation.gaze_zone.strip().lower()
        posture_proxy = observation.posture_proxy.strip().lower()
        active_movements = tuple(
            sorted({label.strip() for label in observation.facial_movement_labels if label.strip()})
        )
        target_movements = set(self.policy.target_facial_movement_labels)
        matched_movements = tuple(label for label in active_movements if label in target_movements)
        return [
            (
                "face_missing",
                not observation.face_present,
                {
                    "face_present": observation.face_present,
                    "gaze_zone": gaze_zone,
                    "posture_proxy": posture_proxy,
                },
            ),
            (
                "gaze_diverted",
                observation.face_present and gaze_zone in self.policy.gaze_diverted_zones,
                {
                    "face_present": observation.face_present,
                    "gaze_zone": gaze_zone,
                },
            ),
            (
                "posture_shifted",
                posture_proxy in self.policy.posture_shifted_values,
                {
                    "face_present": observation.face_present,
                    "posture_proxy": posture_proxy,
                },
            ),
            (
                "facial_movement_detected",
                observation.face_present and bool(matched_movements),
                {
                    "face_present": observation.face_present,
                    "facial_movement_labels": list(matched_movements),
                },
            ),
        ]


@dataclass
class _BufferedFrame:
    timestamp: float
    frame: np.ndarray


class LocalEventClipRecorder:
    def __init__(
        self,
        *,
        policy: EventRecordingPolicy | None = None,
        output_dir: Path | None = None,
        recording_enabled: bool = False,
        buffer_enabled: bool | None = None,
        persist_enabled: bool | None = None,
        output_format: str = "npz",
        fps: float | None = None,
        buffer_frame_size: tuple[int, int] | None = None,
        buffer_fps: float | None = None,
    ) -> None:
        self.policy = policy or EventRecordingPolicy()
        self.output_dir = output_dir
        self.buffer_enabled = recording_enabled if buffer_enabled is None else bool(buffer_enabled)
        self.persist_enabled = recording_enabled if persist_enabled is None else bool(persist_enabled)
        # Backward-compatible public attribute used by ondamm_learning_cli.
        self.recording_enabled = self.persist_enabled
        if output_format not in {"npz", "mp4"}:
            raise ValueError("output_format must be 'npz' or 'mp4'")
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive")
        if buffer_frame_size is not None and (
            len(buffer_frame_size) != 2 or any(int(value) <= 0 for value in buffer_frame_size)
        ):
            raise ValueError("buffer_frame_size must contain positive width and height")
        if buffer_fps is not None and buffer_fps <= 0:
            raise ValueError("buffer_fps must be positive")
        self.output_format = output_format
        self.fps = float(fps) if fps is not None else None
        self.buffer_frame_size = (
            (int(buffer_frame_size[0]), int(buffer_frame_size[1]))
            if buffer_frame_size is not None
            else None
        )
        self.buffer_fps = float(buffer_fps) if buffer_fps is not None else None
        self._last_buffered_timestamp = float("-inf")
        self._frames: deque[_BufferedFrame] = deque()
        self._pending_events: dict[str, EventMetadata] = {}

    def add_frame(self, *, frame: np.ndarray, timestamp: float) -> None:
        if not self.buffer_enabled:
            return
        timestamp = float(timestamp)
        if (
            self.buffer_fps is not None
            and timestamp - self._last_buffered_timestamp + 1e-9 < 1.0 / self.buffer_fps
        ):
            return
        values = np.asarray(frame)
        if self.buffer_frame_size is not None and values.shape[1::-1] != self.buffer_frame_size:
            import cv2

            values = cv2.resize(values, self.buffer_frame_size, interpolation=cv2.INTER_AREA)
        self._frames.append(_BufferedFrame(timestamp=timestamp, frame=np.array(values, copy=True)))
        self._last_buffered_timestamp = timestamp
        self._prune_frames(current_timestamp=timestamp)

    def record_event(self, event: EventMetadata) -> EventMetadata:
        """Persist now when no tail is requested, otherwise queue delayed finalization.

        Buffering and disk persistence are intentionally separate.  A runtime may
        keep a short, ephemeral RAM buffer while candidate occurrences remain
        metadata-only, then call this method only after the recurrence policy has
        crossed its clip threshold.
        """
        if not self.persist_enabled or self.output_dir is None:
            return event
        if self.policy.clip_tail_seconds > 0:
            self._pending_events.setdefault(event.event_id, event)
            return event
        return self._persist_event(event)

    def finalize_ready(self, *, current_timestamp: float) -> list[EventMetadata]:
        """Write pending clips only after their post-event tail exists in RAM."""
        ready: list[EventMetadata] = []
        for event_id, event in list(self._pending_events.items()):
            required_until = event.end_timestamp + self.policy.clip_tail_seconds
            if float(current_timestamp) + 1e-9 < required_until:
                continue
            try:
                ready.append(self._persist_event(event))
            except Exception:
                # A failed codec/write attempt must not permanently occupy the
                # candidate's one pending slot. Pattern memory has no source
                # event yet, so a later independent occurrence can retry.
                del self._pending_events[event_id]
                raise
            del self._pending_events[event_id]
        return ready

    def has_ready_pending(self, *, current_timestamp: float) -> bool:
        """Return whether the current frame completes a requested post tail."""
        return any(
            float(current_timestamp) + 1e-9
            >= event.end_timestamp + self.policy.clip_tail_seconds
            for event in self._pending_events.values()
        )

    def flush_pending(self, *, current_timestamp: float, allow_incomplete_tail: bool = False) -> list[EventMetadata]:
        """Finalize ready clips at shutdown; incomplete tails stay pending by default."""
        if not allow_incomplete_tail:
            return self.finalize_ready(current_timestamp=current_timestamp)
        completed: list[EventMetadata] = []
        for event_id, event in list(self._pending_events.items()):
            completed.append(self._persist_event(event, clip_end_override=float(current_timestamp)))
            del self._pending_events[event_id]
        return completed

    @property
    def buffered_frame_count(self) -> int:
        return len(self._frames)

    @property
    def pending_event_count(self) -> int:
        return len(self._pending_events)

    def has_pending_candidate(self, candidate_id: str) -> bool:
        return any(
            event.trigger_values.get("candidate_id") == candidate_id
            for event in self._pending_events.values()
        )

    def discard_buffered(self) -> None:
        """아동의 중단 요청처럼 저장이 허용되지 않는 종료에서 RAM 자료를 버린다."""
        self._pending_events.clear()
        self._frames.clear()

    def _persist_event(
        self,
        event: EventMetadata,
        *,
        clip_end_override: float | None = None,
    ) -> EventMetadata:
        if not self.persist_enabled or self.output_dir is None:
            return event

        self.output_dir.mkdir(parents=True, exist_ok=True)
        clip_frames = self._select_frames(event, clip_end_override=clip_end_override)
        if not clip_frames:
            raise RuntimeError(f"No buffered frames are available for event clip: {event.event_id}")

        if self.output_format == "mp4":
            clip_path = self._write_mp4(event, clip_frames)
            return event.with_clip_path(str(clip_path))

        clip_path = self.output_dir / f"{event.event_id}.npz"
        frames = np.stack([entry.frame for entry in clip_frames])
        timestamps = np.array([entry.timestamp for entry in clip_frames], dtype=float)
        metadata = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "start_timestamp": event.start_timestamp,
            "end_timestamp": event.end_timestamp,
            "trigger_values": event.trigger_values,
            "clip_path": str(clip_path),
        }
        np.savez_compressed(
            clip_path,
            frames=frames,
            timestamps=timestamps,
            event_metadata=json.dumps(metadata, ensure_ascii=False),
        )
        return event.with_clip_path(str(clip_path))

    def _write_mp4(self, event: EventMetadata, clip_frames: list[_BufferedFrame]) -> Path:
        import cv2

        first = clip_frames[0].frame
        if first.ndim == 2:
            first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
        elif first.ndim == 3 and first.shape[2] == 4:
            first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)
        if first.ndim != 3 or first.shape[2] != 3:
            raise ValueError("event frames must be grayscale, BGR, or BGRA images")
        height, width = first.shape[:2]
        fps = self.fps or self._estimate_fps(clip_frames)
        clip_path = self.output_dir / f"{event.event_id}.mp4"
        temporary = self.output_dir / f".{event.event_id}.tmp.mp4"
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {clip_path}")
        try:
            try:
                for entry in clip_frames:
                    frame = entry.frame
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif frame.ndim == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
            finally:
                writer.release()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        if not temporary.is_file() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Event clip writer produced no data: {clip_path}")
        temporary.replace(clip_path)
        return clip_path

    def _select_frames(
        self,
        event: EventMetadata,
        *,
        clip_end_override: float | None = None,
    ) -> list[_BufferedFrame]:
        clip_start = max(0.0, event.start_timestamp - self.policy.pre_event_buffer_seconds)
        clip_end = (
            float(clip_end_override)
            if clip_end_override is not None
            else event.end_timestamp + self.policy.clip_tail_seconds
        )
        return [entry for entry in self._frames if clip_start <= entry.timestamp <= clip_end]

    def _estimate_fps(self, clip_frames: list[_BufferedFrame]) -> float:
        if len(clip_frames) < 2:
            return 30.0
        deltas = np.diff(np.array([entry.timestamp for entry in clip_frames], dtype=float))
        positive = deltas[deltas > 1e-9]
        if not len(positive):
            return 30.0
        return float(np.clip(1.0 / np.median(positive), 1.0, 240.0))

    def _prune_frames(self, *, current_timestamp: float) -> None:
        threshold = current_timestamp - self.policy.history_window_seconds()
        while self._frames and self._frames[0].timestamp < threshold:
            self._frames.popleft()
