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
    pre_event_buffer_seconds: float = 1.5
    clip_tail_seconds: float = 0.0
    gaze_diverted_zones: tuple[str, ...] = DEFAULT_GAZE_DIVERTED_ZONES
    posture_shifted_values: tuple[str, ...] = DEFAULT_POSTURE_SHIFTED_VALUES

    def threshold_for(self, event_type: str) -> float:
        if event_type == "face_missing":
            return self.face_missing_min_seconds
        if event_type == "gaze_diverted":
            return self.gaze_diverted_min_seconds
        if event_type == "posture_shifted":
            return self.posture_shifted_min_seconds
        raise ValueError(f"Unsupported event type: {event_type}")

    def history_window_seconds(self) -> float:
        return max(
            self.face_missing_min_seconds,
            self.gaze_diverted_min_seconds,
            self.posture_shifted_min_seconds,
        ) + self.pre_event_buffer_seconds + self.clip_tail_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventObservation:
    timestamp: float
    face_present: bool
    gaze_zone: str
    posture_proxy: str


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
                    if duration >= self.policy.threshold_for(event_type):
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
    ) -> None:
        self.policy = policy or EventRecordingPolicy()
        self.output_dir = output_dir
        self.recording_enabled = recording_enabled
        self._frames: deque[_BufferedFrame] = deque()

    def add_frame(self, *, frame: np.ndarray, timestamp: float) -> None:
        if not self.recording_enabled:
            return
        self._frames.append(_BufferedFrame(timestamp=float(timestamp), frame=np.array(frame, copy=True)))
        self._prune_frames(current_timestamp=float(timestamp))

    def record_event(self, event: EventMetadata) -> EventMetadata:
        if not self.recording_enabled or self.output_dir is None:
            return event

        self.output_dir.mkdir(parents=True, exist_ok=True)
        clip_frames = self._select_frames(event)
        if not clip_frames:
            return event

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

    def _select_frames(self, event: EventMetadata) -> list[_BufferedFrame]:
        clip_start = max(0.0, event.start_timestamp - self.policy.pre_event_buffer_seconds)
        clip_end = event.end_timestamp + self.policy.clip_tail_seconds
        return [entry for entry in self._frames if clip_start <= entry.timestamp <= clip_end]

    def _prune_frames(self, *, current_timestamp: float) -> None:
        threshold = current_timestamp - self.policy.history_window_seconds()
        while self._frames and self._frames[0].timestamp < threshold:
            self._frames.popleft()
