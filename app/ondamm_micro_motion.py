"""Causal micro-motion episode segmentation.

TCN windows overlap heavily.  This module converts those window endpoints into
independent episodes so recurrence counts never equal the number of windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from ondamm_movement_explanation import summarize_temporal_features


@dataclass(frozen=True)
class EpisodePolicy:
    onset_threshold: float = 0.15
    offset_threshold: float = 0.08
    min_duration_seconds: float = 0.2
    refractory_seconds: float = 0.5

    # Motion이 잠깐 offset 아래로 떨어졌다고 즉시 episode를 닫지 않는다.
    # 이 시간 동안 계속 조용해야 하나의 physical movement가 끝난 것으로 본다.
    offset_hold_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.onset_threshold <= 0:
            raise ValueError("onset_threshold must be positive")
        if not 0 <= self.offset_threshold < self.onset_threshold:
            raise ValueError("offset_threshold must be non-negative and below onset_threshold")
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive")
        if self.refractory_seconds < 0:
            raise ValueError("refractory_seconds must be non-negative")
        if self.offset_hold_seconds < 0:
            raise ValueError("offset_hold_seconds must be non-negative")


@dataclass(frozen=True)
class MotionEpisode:
    episode_id: str
    start_timestamp: float
    end_timestamp: float
    embedding: np.ndarray
    peak_motion_score: float
    mean_motion_score: float
    quality_score: float
    endpoint_count: int
    movement_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return round(max(0.0, self.end_timestamp - self.start_timestamp), 3)

    def to_summary(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
            "peak_motion_score": self.peak_motion_score,
            "mean_motion_score": self.mean_motion_score,
            "quality_score": self.quality_score,
            "endpoint_count": self.endpoint_count,
            "movement_summary": self.movement_summary,
        }


@dataclass
class _ActiveEpisode:
    start_timestamp: float
    last_active_timestamp: float
    embeddings: list[np.ndarray] = field(default_factory=list)
    motion_scores: list[float] = field(default_factory=list)
    quality_scores: list[float] = field(default_factory=list)
    movement_feature_samples: list[dict[str, float]] = field(default_factory=list)
    quiet_since: float | None = None


class MicroMotionEpisodeDetector:
    def __init__(self, policy: EpisodePolicy | None = None) -> None:
        self.policy = policy or EpisodePolicy()
        self._active: _ActiveEpisode | None = None
        self._refractory_until = float("-inf")
        self._last_timestamp = float("-inf")

    def add_endpoint(
        self,
        *,
        timestamp: float,
        motion_score: float,
        embedding: Sequence[float],
        quality_score: float = 1.0,
        movement_features: Mapping[str, float] | None = None,
    ) -> MotionEpisode | None:
        timestamp = float(timestamp)
        motion_score = float(motion_score)
        quality_score = float(quality_score)
        vector = np.asarray(embedding, dtype=np.float32)

        if timestamp + 1e-9 < self._last_timestamp:
            raise ValueError("episode timestamps must be monotonic")
        self._last_timestamp = timestamp

        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("embedding must be a finite 1D vector")

        if not np.isfinite([motion_score, quality_score]).all():
            raise ValueError("motion_score and quality_score must be finite")

        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        # ---------------------------------------------------------
        # 1. 아직 episode가 없다면 onset threshold를 넘어야 시작.
        # ---------------------------------------------------------
        if self._active is None:
            if timestamp < self._refractory_until:
                return None

            if motion_score < self.policy.onset_threshold:
                return None

            self._active = _ActiveEpisode(
                start_timestamp=timestamp,
                last_active_timestamp=timestamp,
            )

        active = self._active

        # ---------------------------------------------------------
        # 2. offset 아래로 내려갔다고 즉시 종료하지 않는다.
        #
        # 예:
        # neutral -> lip purse -> brief stop -> return neutral
        #
        # brief stop을 episode 경계로 잘못 인식하지 않기 위해
        # offset_hold_seconds 동안 계속 조용해야 종료한다.
        # ---------------------------------------------------------
        if motion_score <= self.policy.offset_threshold:
            if active.quiet_since is None:
                active.quiet_since = timestamp

            quiet_duration = timestamp - active.quiet_since

            if quiet_duration + 1e-9 < self.policy.offset_hold_seconds:
                return None

            episode = self._finish_active()

            self._refractory_until = (
                timestamp + self.policy.refractory_seconds
            )

            return episode

        # ---------------------------------------------------------
        # 3. quiet hold가 끝나기 전에 움직임이 다시 올라오면
        # 같은 physical episode로 이어 붙인다.
        # ---------------------------------------------------------
        active.quiet_since = None
        active.last_active_timestamp = timestamp

        active.embeddings.append(vector.copy())
        active.motion_scores.append(motion_score)
        active.quality_scores.append(quality_score)
        if movement_features:
            active.movement_feature_samples.append(
                {str(name): float(value) for name, value in movement_features.items()}
            )

        return None

    def flush(self, *, timestamp: float | None = None) -> MotionEpisode | None:
        if self._active is None:
            return None
        if timestamp is not None and timestamp + 1e-9 < self._active.last_active_timestamp:
            raise ValueError("flush timestamp cannot precede the active episode")
        episode = self._finish_active()
        if timestamp is not None:
            self._refractory_until = float(timestamp) + self.policy.refractory_seconds
        return episode

    def reset(self) -> None:
        self._active = None
        self._refractory_until = float("-inf")
        self._last_timestamp = float("-inf")

    def _finish_active(self) -> MotionEpisode | None:
        active = self._active
        self._active = None
        if active is None or not active.embeddings:
            return None
        duration = active.last_active_timestamp - active.start_timestamp
        if duration + 1e-9 < self.policy.min_duration_seconds:
            return None
        centroid = np.mean(np.stack(active.embeddings), axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 1e-8:
            return None
        return MotionEpisode(
            episode_id=f"episode-{uuid4().hex[:12]}",
            start_timestamp=round(active.start_timestamp, 3),
            end_timestamp=round(active.last_active_timestamp, 3),
            embedding=(centroid / norm).astype(np.float32),
            peak_motion_score=round(max(active.motion_scores), 6),
            mean_motion_score=round(float(np.mean(active.motion_scores)), 6),
            quality_score=round(float(np.mean(active.quality_scores)), 6),
            endpoint_count=len(active.embeddings),
            movement_summary=summarize_temporal_features(active.movement_feature_samples),
        )


def generic_motion_score(feature_names: Sequence[str], values: Sequence[float]) -> float:
    """Robust score from the generic motion subset; nuisance columns are absent."""
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(feature_names),) or not np.isfinite(vector).all():
        raise ValueError("feature vector does not match feature_names")
    indices = [index for index, name in enumerate(feature_names) if name.startswith("motion_")]
    if not indices:
        raise ValueError("feature_names contain no motion_* values")
    selected = np.abs(vector[indices])
    # Median limits the influence of a single noisy ROI while the upper quartile
    # preserves a localized movement episode.
    return float(0.5 * np.median(selected) + 0.5 * np.quantile(selected, 0.75))
