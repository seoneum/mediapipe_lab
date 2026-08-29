"""Live MediaPipe signal adapter for the strict ON DAMM TCN feature contract."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from micro_expression_signals import LEFT_BROW, LEFT_EYE, MOUTH, RIGHT_BROW, RIGHT_EYE
from ondamm_temporal_encoder import validate_feature_names


NOSE = (1, 2, 4, 5, 6, 19, 94, 97, 98, 129, 168, 195, 197, 326, 327, 358)
REGIONS = {
    "mouth": tuple(sorted(set(MOUTH))),
    "eyes": tuple(sorted(set(LEFT_EYE + RIGHT_EYE))),
    "brow": tuple(sorted(set(LEFT_BROW + RIGHT_BROW))),
    "nose": NOSE,
}
LIP_APERTURE_PAIRS = ((13, 14), (82, 87), (312, 317))
EYE_APERTURE_PAIRS = ((159, 145), (158, 153), (386, 374), (387, 380))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _point_distance(points: np.ndarray, left: int, right: int) -> float:
    if left >= len(points) or right >= len(points):
        return 0.0
    return _finite(np.linalg.norm(points[left, :2] - points[right, :2]))


def _mean_pair_distance(points: np.ndarray, pairs: Sequence[tuple[int, int]]) -> float:
    values = [_point_distance(points, left, right) for left, right in pairs]
    return float(np.mean(values)) if values else 0.0


def absolute_geometry(canonical_landmarks: np.ndarray) -> dict[str, float]:
    points = np.asarray(canonical_landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError("canonical_landmarks must be a finite Nx2/Nx3 array")
    output: dict[str, float] = {}
    for name, raw_indices in REGIONS.items():
        indices = [index for index in raw_indices if index < len(points)]
        if not indices:
            center = np.zeros(2, dtype=np.float32)
            spread = 0.0
        else:
            region = points[indices, :2]
            center = region.mean(axis=0)
            spread = _finite(np.linalg.norm(region - center, axis=1).mean())
        output[f"geom_abs_{name}_cx"] = _finite(center[0])
        output[f"geom_abs_{name}_cy"] = _finite(center[1])
        output[f"geom_abs_{name}_spread"] = spread
    output.update(
        {
            "geom_abs_mouth_width": _point_distance(points, 61, 291),
            "geom_abs_lip_aperture": _mean_pair_distance(points, LIP_APERTURE_PAIRS),
            "geom_abs_eye_aperture": _mean_pair_distance(points, EYE_APERTURE_PAIRS),
            "geom_abs_inner_brow_distance": _point_distance(points, 107, 336),
            "geom_abs_mouth_corner_y": _finite(np.mean(points[[61, 291], 1])) if len(points) > 291 else 0.0,
            "geom_abs_brow_y": _finite(np.mean(points[list(REGIONS["brow"]), 1])) if len(points) > 336 else 0.0,
        }
    )
    return output


def motion_features(signal: Mapping[str, Any]) -> dict[str, float]:
    output = {
        name: _finite(signal.get(name))
        for name in (
            "motion_mean",
            "motion_max",
            "motion_mouth",
            "motion_left_eye",
            "motion_right_eye",
            "motion_left_brow",
            "motion_right_brow",
        )
    }
    output["motion_eyes"] = float(
        np.mean([output["motion_left_eye"], output["motion_right_eye"]])
    )
    output["motion_brow"] = float(
        np.mean([output["motion_left_brow"], output["motion_right_brow"]])
    )
    return output


def raw_motion_magnitude(signal: Mapping[str, Any]) -> float:
    motion = motion_features(signal)
    localized = np.asarray(
        [motion["motion_mouth"], motion["motion_eyes"], motion["motion_brow"]],
        dtype=np.float32,
    )
    return float(0.5 * np.median(localized) + 0.5 * np.max(localized))


class TemporalFeatureAdapter:
    """Create an exact checkpoint-ordered feature mapping from one live signal."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names = validate_feature_names(feature_names)
        self._last_geometry: dict[str, float] = {}

    def from_signal(self, signal: Mapping[str, Any]) -> dict[str, float]:
        face_detected = bool(signal.get("face_detected"))
        blendshapes = signal.get("blendshapes") if face_detected else {}
        if not isinstance(blendshapes, Mapping):
            blendshapes = {}
        motion = motion_features(signal) if face_detected else {}
        geometry: dict[str, float] = {}
        canonical = signal.get("canonical_landmarks")
        if face_detected and canonical is not None:
            geometry = absolute_geometry(np.asarray(canonical))
            self._last_geometry = dict(geometry)
        elif self._last_geometry:
            # A short missed frame should not manufacture a giant geometry jump.
            geometry = dict(self._last_geometry)

        output: dict[str, float] = {}
        for name in self.feature_names:
            if name.startswith("bs_"):
                output[name] = _finite(blendshapes.get(name.removeprefix("bs_")))
            elif name.startswith("geom_abs_"):
                output[name] = _finite(geometry.get(name))
            else:
                output[name] = _finite(motion.get(name))
        return output


@dataclass
class PersonalMotionCalibrator:
    """Short neutral calibration that converts raw motion into a personal z score."""

    calibration_seconds: float = 3.0
    scale_floor: float = 1e-5
    max_score: float = 20.0
    _started_at: float | None = None
    _samples: list[float] = field(default_factory=list)
    center: float = 0.0
    scale: float = 1.0
    ready: bool = False

    def add(self, *, timestamp: float, raw_motion: float, face_detected: bool) -> float:
        timestamp = float(timestamp)
        raw_motion = max(0.0, _finite(raw_motion))
        if self._started_at is None:
            if not face_detected:
                return 0.0
            self._started_at = timestamp
        if not self.ready:
            if face_detected:
                self._samples.append(raw_motion)
            if timestamp - self._started_at + 1e-9 >= self.calibration_seconds:
                self._finish()
            return 0.0
        return float(np.clip((raw_motion - self.center) / self.scale, 0.0, self.max_score))

    def remaining_at(self, timestamp: float) -> float:
        if self.ready:
            return 0.0
        if self._started_at is None:
            return self.calibration_seconds
        return max(0.0, self.calibration_seconds - (float(timestamp) - self._started_at))

    def _finish(self) -> None:
        values = np.asarray(self._samples, dtype=np.float32)
        if not len(values):
            self.center = 0.0
            self.scale = self.scale_floor
        else:
            self.center = float(np.median(values))
            mad = float(np.median(np.abs(values - self.center)) * 1.4826)
            q25, q75 = np.quantile(values, [0.25, 0.75]) if len(values) > 1 else (self.center, self.center)
            self.scale = max(mad, float((q75 - q25) / 1.349), self.scale_floor)
        self.ready = True
