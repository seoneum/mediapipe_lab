"""Future-session metrics for one strongly personalized child model.

The product objective is deliberately within-child.  Training/personalization
sessions are historical sessions from one child, and every metric in this
module is computed on a later session from that same child.  No cross-person
aggregation or unseen-person score is defined here.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ACTIVE_LIFECYCLES = {
    "KNOWN_OCCURRENCE",
    "UNKNOWN_OCCURRENCE",
    "REPEATING_CANDIDATE",
}


def _required_text(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} is required")
    return result


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class GroundTruthEvent:
    child_id: str
    session_id: str
    event_id: str
    pattern_id: str
    start_timestamp: float
    end_timestamp: float
    known_at_session_start: bool

    def __post_init__(self) -> None:
        for name in ("child_id", "session_id", "event_id", "pattern_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "start_timestamp",
            _finite_float(self.start_timestamp, "start_timestamp"),
        )
        object.__setattr__(
            self,
            "end_timestamp",
            _finite_float(self.end_timestamp, "end_timestamp"),
        )
        if self.end_timestamp < self.start_timestamp:
            raise ValueError("ground-truth event end must not precede start")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GroundTruthEvent":
        return cls(
            child_id=value.get("child_id", ""),
            session_id=value.get("session_id", ""),
            event_id=value.get("event_id", ""),
            pattern_id=value.get("pattern_id", ""),
            start_timestamp=value.get("start_timestamp"),
            end_timestamp=value.get("end_timestamp"),
            known_at_session_start=_boolean(
                value.get("known_at_session_start"),
                "known_at_session_start",
            ),
        )


@dataclass(frozen=True)
class TemporalDetection:
    child_id: str
    session_id: str
    detection_id: str
    start_timestamp: float
    end_timestamp: float
    lifecycle: str
    pattern_id: str | None = None
    candidate_id: str | None = None
    occurrence_count: int = 0
    eventized_timestamp: float | None = None

    def __post_init__(self) -> None:
        for name in ("child_id", "session_id", "detection_id", "lifecycle"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        start = _finite_float(self.start_timestamp, "start_timestamp")
        end = _finite_float(self.end_timestamp, "end_timestamp")
        object.__setattr__(self, "start_timestamp", start)
        object.__setattr__(self, "end_timestamp", end)
        if end < start:
            raise ValueError("detection end must not precede start")
        object.__setattr__(self, "pattern_id", _optional_text(self.pattern_id))
        object.__setattr__(self, "candidate_id", _optional_text(self.candidate_id))
        count = int(self.occurrence_count)
        if count < 0:
            raise ValueError("occurrence_count must be non-negative")
        object.__setattr__(self, "occurrence_count", count)
        if self.eventized_timestamp is not None:
            object.__setattr__(
                self,
                "eventized_timestamp",
                _finite_float(self.eventized_timestamp, "eventized_timestamp"),
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TemporalDetection":
        raw_eventized = value.get("eventized_timestamp")
        if raw_eventized is not None and str(raw_eventized).strip() == "":
            raw_eventized = None
        raw_count = value.get("occurrence_count", 0)
        if raw_count is None or str(raw_count).strip() == "":
            raw_count = 0
        return cls(
            child_id=value.get("child_id", ""),
            session_id=value.get("session_id", ""),
            detection_id=value.get("detection_id", ""),
            start_timestamp=value.get("start_timestamp"),
            end_timestamp=value.get("end_timestamp"),
            lifecycle=value.get("lifecycle", ""),
            pattern_id=value.get("pattern_id"),
            candidate_id=value.get("candidate_id"),
            occurrence_count=raw_count,
            eventized_timestamp=raw_eventized,
        )


def temporal_iou(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    if union <= 0:
        return 1.0 if left_start == right_start and left_end == right_end else 0.0
    return intersection / union


def match_events(
    truth: Sequence[GroundTruthEvent],
    detections: Sequence[TemporalDetection],
    *,
    iou_threshold: float,
) -> tuple[dict[int, int], dict[int, int]]:
    """Greedy, deterministic one-to-one temporal matching within a session."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    candidates: list[tuple[float, int, int]] = []
    for truth_index, target in enumerate(truth):
        for detection_index, detection in enumerate(detections):
            if target.session_id != detection.session_id:
                continue
            score = temporal_iou(
                target.start_timestamp,
                target.end_timestamp,
                detection.start_timestamp,
                detection.end_timestamp,
            )
            if score >= iou_threshold:
                candidates.append((score, truth_index, detection_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    truth_to_detection: dict[int, int] = {}
    detection_to_truth: dict[int, int] = {}
    for _, truth_index, detection_index in candidates:
        if truth_index in truth_to_detection or detection_index in detection_to_truth:
            continue
        truth_to_detection[truth_index] = detection_index
        detection_to_truth[detection_index] = truth_index
    return truth_to_detection, detection_to_truth


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return float(sum(materialized) / len(materialized)) if materialized else None


def _median(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def evaluate_future_session(
    truth: Sequence[GroundTruthEvent],
    detections: Sequence[TemporalDetection],
    *,
    child_id: str,
    future_session: str,
    session_duration_seconds: float,
    iou_threshold: float = 0.2,
    min_unknown_repetitions: int = 3,
) -> dict[str, Any]:
    """Compute the product metrics on one held-out future session.

    ``known_at_session_start`` is frozen before the future session begins.
    Unknown discovery metrics therefore cannot relabel a pattern as known after
    seeing the test session.  Candidate identity is evaluated only against
    human-annotated temporal ground truth.
    """
    child_id = _required_text(child_id, "child_id")
    future_session = _required_text(future_session, "future_session")
    duration = _finite_float(session_duration_seconds, "session_duration_seconds")
    if duration <= 0:
        raise ValueError("session_duration_seconds must be positive")
    if min_unknown_repetitions < 2:
        raise ValueError("min_unknown_repetitions must be at least two")

    all_rows: list[GroundTruthEvent | TemporalDetection] = [*truth, *detections]
    for row in all_rows:
        if row.child_id != child_id:
            raise ValueError("evaluation rows must belong to exactly the target child")
        if row.session_id != future_session:
            raise ValueError("evaluation rows must belong to the held-out future session")

    active_detections = [item for item in detections if item.lifecycle in ACTIVE_LIFECYCLES]
    truth_to_detection, detection_to_truth = match_events(
        truth,
        active_detections,
        iou_threshold=iou_threshold,
    )

    known_truth_indices = [index for index, item in enumerate(truth) if item.known_at_session_start]
    known_detection_indices = [
        index
        for index, item in enumerate(active_detections)
        if item.lifecycle == "KNOWN_OCCURRENCE"
    ]
    correct_known_truth: set[int] = set()
    correct_known_detections: set[int] = set()
    for truth_index in known_truth_indices:
        detection_index = truth_to_detection.get(truth_index)
        if detection_index is None:
            continue
        target = truth[truth_index]
        detection = active_detections[detection_index]
        if (
            detection.lifecycle == "KNOWN_OCCURRENCE"
            and detection.pattern_id == target.pattern_id
        ):
            correct_known_truth.add(truth_index)
            correct_known_detections.add(detection_index)

    unmatched_activations = sum(
        index not in detection_to_truth
        for index in range(len(active_detections))
    )
    duration_minutes = duration / 60.0

    unknown_truth_indices = [
        index for index, item in enumerate(truth) if not item.known_at_session_start
    ]
    unknown_counts = Counter(truth[index].pattern_id for index in unknown_truth_indices)
    eligible_unknown_patterns = {
        pattern_id
        for pattern_id, count in unknown_counts.items()
        if count >= min_unknown_repetitions
    }

    candidate_detection_indices: dict[str, list[int]] = defaultdict(list)
    discovered_candidates: set[str] = set()
    for detection_index, detection in enumerate(active_detections):
        if detection.candidate_id:
            candidate_detection_indices[detection.candidate_id].append(detection_index)
            if detection.lifecycle == "REPEATING_CANDIDATE":
                discovered_candidates.add(detection.candidate_id)

    candidate_truth_patterns: dict[str, set[str]] = {}
    candidate_has_known_truth: dict[str, bool] = {}
    for candidate_id in discovered_candidates:
        patterns: set[str] = set()
        has_known = False
        for detection_index in candidate_detection_indices[candidate_id]:
            truth_index = detection_to_truth.get(detection_index)
            if truth_index is None:
                continue
            patterns.add(truth[truth_index].pattern_id)
            has_known = has_known or truth[truth_index].known_at_session_start
        candidate_truth_patterns[candidate_id] = patterns
        candidate_has_known_truth[candidate_id] = has_known

    true_discovery_candidates = {
        candidate_id
        for candidate_id, patterns in candidate_truth_patterns.items()
        if (
            len(patterns) == 1
            and not candidate_has_known_truth[candidate_id]
            and next(iter(patterns)) in eligible_unknown_patterns
        )
    }
    candidates_by_pattern: dict[str, list[str]] = defaultdict(list)
    for candidate_id in true_discovery_candidates:
        pattern_id = next(iter(candidate_truth_patterns[candidate_id]))
        candidates_by_pattern[pattern_id].append(candidate_id)

    duplicate_count = sum(max(0, len(values) - 1) for values in candidates_by_pattern.values())
    false_merge_count = sum(
        len(candidate_truth_patterns[candidate_id]) > 1
        for candidate_id in discovered_candidates
        if candidate_truth_patterns[candidate_id]
    )
    mapped_discovery_count = sum(
        bool(candidate_truth_patterns[candidate_id])
        for candidate_id in discovered_candidates
    )

    occurrences_until_discovery: list[float] = []
    discovery_latencies: list[float] = []
    for pattern_id, candidate_ids in candidates_by_pattern.items():
        repeating = [
            active_detections[index]
            for candidate_id in candidate_ids
            for index in candidate_detection_indices[candidate_id]
            if active_detections[index].lifecycle == "REPEATING_CANDIDATE"
        ]
        if not repeating:
            continue
        first_discovery = min(
            repeating,
            key=lambda item: (
                item.eventized_timestamp
                if item.eventized_timestamp is not None
                else item.end_timestamp,
                item.detection_id,
            ),
        )
        occurrences_until_discovery.append(float(first_discovery.occurrence_count))
        first_observation = min(
            item.start_timestamp
            for item in truth
            if not item.known_at_session_start and item.pattern_id == pattern_id
        )
        eventized = (
            first_discovery.eventized_timestamp
            if first_discovery.eventized_timestamp is not None
            else first_discovery.end_timestamp
        )
        discovery_latencies.append(max(0.0, float(eventized - first_observation)))

    per_known_pattern_recall: list[float] = []
    for pattern_id in sorted({truth[index].pattern_id for index in known_truth_indices}):
        indices = [
            index
            for index in known_truth_indices
            if truth[index].pattern_id == pattern_id
        ]
        per_known_pattern_recall.append(
            sum(index in correct_known_truth for index in indices) / len(indices)
        )

    return {
        "objective": "within-child-future-session",
        "child_id": child_id,
        "future_session": future_session,
        "iou_threshold": float(iou_threshold),
        "session_duration_seconds": duration,
        "known_pattern_event_recall": _safe_ratio(
            len(correct_known_truth),
            len(known_truth_indices),
        ),
        "known_pattern_precision": _safe_ratio(
            len(correct_known_detections),
            len(known_detection_indices),
        ),
        "false_activations_per_min": float(unmatched_activations / duration_minutes),
        "unknown_repeated_pattern_discovery_precision": _safe_ratio(
            len(true_discovery_candidates),
            len(discovered_candidates),
        ),
        "duplicate_cluster_rate": _safe_ratio(
            duplicate_count,
            len(true_discovery_candidates),
        ),
        "false_merge_rate": _safe_ratio(false_merge_count, mapped_discovery_count),
        "occurrences_required_until_discovery_mean": _mean(occurrences_until_discovery),
        "occurrences_required_until_discovery_median": _median(occurrences_until_discovery),
        "first_observation_to_eventization_latency_seconds_mean": _mean(discovery_latencies),
        "first_observation_to_eventization_latency_seconds_median": _median(discovery_latencies),
        "future_session_stability": _mean(per_known_pattern_recall),
        "counts": {
            "ground_truth_events": len(truth),
            "active_detections": len(active_detections),
            "known_ground_truth_events": len(known_truth_indices),
            "known_detections": len(known_detection_indices),
            "unmatched_activations": unmatched_activations,
            "eligible_unknown_patterns": len(eligible_unknown_patterns),
            "discovered_candidate_clusters": len(discovered_candidates),
            "true_discovery_candidate_clusters": len(true_discovery_candidates),
            "duplicate_candidate_clusters": duplicate_count,
            "false_merge_candidate_clusters": false_merge_count,
        },
    }


__all__ = [
    "ACTIVE_LIFECYCLES",
    "GroundTruthEvent",
    "TemporalDetection",
    "evaluate_future_session",
    "match_events",
    "temporal_iou",
]
