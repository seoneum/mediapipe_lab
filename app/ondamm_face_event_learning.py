"""Bounded, offline event learning from facial movement proxies.

This module deliberately eventizes *observable proxy changes*, not internal states.
A sample contains small numeric movement summaries, camera-relative zones, and
quality metadata; no image, video, emotion, concentration, preference, diagnosis,
or compliance field is accepted.  Windows are short and thresholds explicit so an
event is a reproducible review cue rather than a claim about a person.

The learner is a per-person prototype model.  It consumes only independently
approved, positive teacher/reviewer observations and emits a possible support
strategy match with abstention.  Every emitted object remains a human-review
candidate and cannot mutate a learning plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

# Bounded thresholds: equality at the boundary is intentionally safe/passive.
FACIAL_WINDOW = 3
FACIAL_CHANGE_THRESHOLD = 0.20
GAZE_DWELL_THRESHOLD = 2.0
HEAD_TRANSITION_THRESHOLD = 2
QUALITY_THRESHOLD = 0.60
MATCH_CONFIDENCE_THRESHOLD = 0.70
NOTICE = "Observational candidate only; human review required; non-diagnostic."
# Maximum encoded input accepted by the CLI before JSON parsing; bounded reads
# keep stdin and local-file handling from materializing untrusted input.
MAX_INPUT_BYTES = 1_048_576
MAX_EVENT_SPAN_SECONDS = 30.0
MAX_SAFE_COLLECTION = 4096
MAX_ZONE_COUNT = FACIAL_WINDOW
MAX_QUALITY_FLAGS = 32
MAX_PROVENANCE = FACIAL_WINDOW
MAX_LABEL_PROVENANCE = MAX_SAFE_COLLECTION
MAX_TRAINING_MANIFEST = MAX_SAFE_COLLECTION
MAX_STRATEGY_PROTOTYPES = 64
MAX_SOURCE_SAMPLES = MAX_SAFE_COLLECTION
MAX_GRAPH_NODES = 131072
MAX_CONTAINER_BREADTH = MAX_SAFE_COLLECTION
_FORBIDDEN = {
    "emotion", "emotions", "concentration", "attention", "preference", "preferences",
    "asd", "autism", "diagnosis", "diagnostic", "compliance", "raw_image", "raw_video",
    "image", "video", "media", "frame", "eye_contact", "eye-contact", "emotion_label",
}
_ALLOWED_SAMPLE = {
    "timestamp", "person_id", "session_id", "context_id", "goal_id",
    "facial_movement_proxy_values", "facial_movement", "gaze_zone", "gaze_dwell_seconds",
    "head_orientation_zone", "head_transition_count", "quality_score", "quality_flags",
}
_ALLOWED_STRATEGIES = {
    "pause", "break", "visual_pause", "visual_break", "movement_break",
    "quiet_space", "first_then", "choice", "prompt", "wait", "modeling",
}
_ALLOWED_FEATURE_SUMMARY = {
    "reason", "sample_count", "feature_vector", "gaze_zones", "head_zones",
    "support_strategy_candidate", "abstained",
}
_CANDIDATE_FIELDS = {
    "event_type", "candidate_id", "person_id", "session_id", "context_id",
    "start_timestamp", "end_timestamp", "feature_summary", "provenance",
    "quality_score", "quality_flags", "requires_human_review", "notice",
    "source_model_digest", "confidence", "evidence_ids",
    "source_sample_digest",
}
_MODEL_FIELDS = {
    "person_id", "strategy_prototypes", "training_candidate_ids",
    "training_fingerprints", "label_provenance", "model_digest",
    "min_quality", "abstention_threshold",
}
_HEX_ID = re.compile(r"^[0-9a-f]+$")
def _forbidden_text(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError(f"{name} must be a non-empty short string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if any(token in normalized for token in _FORBIDDEN):
        raise ValueError(f"forbidden diagnostic/raw value: {name}")
    return value.strip()


def _safe_context(value: Any, name: str) -> str:
    return _forbidden_text(_text(value, name), name)


def _strategy(value: Any, name: str) -> str:
    text = _forbidden_text(_text(value, name), name).lower()
    if text not in _ALLOWED_STRATEGIES:
        raise ValueError(f"unsupported {name}")
    return text


def _hex(value: Any, name: str, lengths: set[int]) -> str:
    if not isinstance(value, str) or len(value) not in lengths or not _HEX_ID.fullmatch(value):
        raise ValueError(f"{name} must be a canonical hexadecimal identifier")
    return value


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canon(value).encode("utf-8")).hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"{name} must be a non-empty short string")
    return value.strip()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric")
    try:
        converted = float(value)
        finite = math.isfinite(converted)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{name} must be finite numeric") from None
    if not finite:
        raise ValueError(f"{name} must be finite numeric")
    return converted


def _check_forbidden(value: Any, path: str = "", *, max_depth: int = 32) -> None:
    """Reject forbidden fields and hostile container graphs without recursive descent."""
    # This walk is deliberately iterative: malformed JSON-like objects can be
    # deeply nested or cyclic, so recursive traversal would turn hostile input
    # into a RecursionError before the caller can return the fixed safe error.
    # `active` tracks only the current path, which detects cycles while still
    # permitting repeated immutable values. Depth, breadth, and node budgets
    # keep validation bounded before feature logic runs.
    active: set[int] = set()
    stack: list[tuple[Any, str, int, bool]] = [(value, path, 0, False)]
    nodes = 1
    while stack:
        item, item_path, depth, exiting = stack.pop()
        if depth > max_depth:
            raise ValueError("nested value exceeds safety depth")
        if isinstance(item, str):
            if item != NOTICE:
                _forbidden_text(item, item_path)
            continue
        if not isinstance(item, (Mapping, list, tuple)):
            continue
        marker = id(item)
        if exiting:
            active.remove(marker)
            continue
        if marker in active:
            raise ValueError("cyclic nested value")
        active.add(marker)
        stack.append((item, item_path, depth, True))
        if isinstance(item, Mapping):
            try:
                breadth = len(item)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("container breadth cannot be determined") from None
            if breadth > MAX_CONTAINER_BREADTH:
                raise ValueError("container breadth exceeds safety bound")
            for key, child in item.items():
                key_text = str(key).lower().replace("-", "_").replace(" ", "_")
                if key_text in _FORBIDDEN or any(token in key_text for token in _FORBIDDEN):
                    raise ValueError(f"forbidden diagnostic/raw field: {item_path}{key}")
                nodes += 1
                if nodes > MAX_GRAPH_NODES:
                    raise ValueError("nested value graph exceeds safety node budget")
                stack.append((child, f"{item_path}{key}.", depth + 1, False))
        else:
            if len(item) > MAX_CONTAINER_BREADTH:
                raise ValueError("container breadth exceeds safety bound")
            for index, child in enumerate(item):
                nodes += 1
                if nodes > MAX_GRAPH_NODES:
                    raise ValueError("nested value graph exceeds safety node budget")
                stack.append((child, f"{item_path}{index}.", depth + 1, False))
def _check_json_numbers(value: Any, path: str = "", *, max_depth: int = 32) -> None:
    """Validate generic JSON numbers without recursive descent or non-finite output."""
    active: set[int] = set()
    stack: list[tuple[Any, str, int, bool]] = [(value, path, 0, False)]
    nodes = 1
    while stack:
        item, item_path, depth, exiting = stack.pop()
        if depth > max_depth:
            raise ValueError("nested value exceeds safety depth")
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            try:
                numeric = float(item)
            except (OverflowError, TypeError, ValueError):
                raise ValueError(f"{item_path} must be finite numeric") from None
            if not math.isfinite(numeric):
                raise ValueError(f"{item_path} must be finite numeric")
            continue
        if not isinstance(item, (Mapping, list, tuple)):
            continue
        marker = id(item)
        if exiting:
            active.remove(marker)
            continue
        if marker in active:
            raise ValueError("cyclic nested value")
        active.add(marker)
        stack.append((item, item_path, depth, True))
        if isinstance(item, Mapping):
            try:
                breadth = len(item)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("container breadth cannot be determined") from None
            if breadth > MAX_CONTAINER_BREADTH:
                raise ValueError("container breadth exceeds safety bound")
            for key, child in item.items():
                nodes += 1
                if nodes > MAX_GRAPH_NODES:
                    raise ValueError("nested value graph exceeds safety node budget")
                stack.append((child, f"{item_path}{key}.", depth + 1, False))
        else:
            if len(item) > MAX_CONTAINER_BREADTH:
                raise ValueError("container breadth exceeds safety bound")
            for index, child in enumerate(item):
                nodes += 1
                if nodes > MAX_GRAPH_NODES:
                    raise ValueError("nested value graph exceeds safety node budget")
                stack.append((child, f"{item_path}{index}.", depth + 1, False))

def _freeze_map(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ObservationSample:
    timestamp: float
    person_id: str
    session_id: str
    context_id: str
    facial_movement_proxy_values: tuple[float, ...]
    gaze_zone: str
    gaze_dwell_seconds: float
    head_orientation_zone: str
    head_transition_count: int
    quality_score: float
    quality_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        timestamp = _finite(self.timestamp, "timestamp")
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        object.__setattr__(self, "timestamp", timestamp)
        for name in ("person_id", "session_id", "gaze_zone", "head_orientation_zone", "context_id"):
            object.__setattr__(self, name, _safe_context(getattr(self, name), name))
        if isinstance(self.facial_movement_proxy_values, (str, bytes)) or not isinstance(self.facial_movement_proxy_values, (list, tuple)):
            raise ValueError("facial movement proxies must be a list or tuple")
        if len(self.facial_movement_proxy_values) > 32:
            raise ValueError("facial movement proxies exceed safe collection bound")
        values = tuple(_finite(v, "facial_movement_proxy_values") for v in self.facial_movement_proxy_values)
        if not values or len(values) > 32 or any(v < 0 or v > 1 for v in values):
            raise ValueError("facial movement proxies must be 1-32 values in [0,1]")
        object.__setattr__(self, "facial_movement_proxy_values", values)
        dwell = _finite(self.gaze_dwell_seconds, "gaze_dwell_seconds")
        if dwell < 0 or dwell > 3600:
            raise ValueError("gaze dwell must be in [0,3600]")
        object.__setattr__(self, "gaze_dwell_seconds", dwell)
        if isinstance(self.head_transition_count, bool) or not isinstance(self.head_transition_count, int) or not 0 <= self.head_transition_count <= 10000:
            raise ValueError("head_transition_count must be an integer in [0,10000]")
        score = _finite(self.quality_score, "quality_score")
        if not 0 <= score <= 1:
            raise ValueError("quality_score must be in [0,1]")
        object.__setattr__(self, "quality_score", score)
        if isinstance(self.quality_flags, (str, bytes)) or not isinstance(self.quality_flags, (list, tuple)):
            raise ValueError("quality_flags must be a list or tuple")
        if len(self.quality_flags) > MAX_QUALITY_FLAGS:
            raise ValueError("quality_flags exceed safe collection bound")
        flags = tuple(_forbidden_text(_text(f, "quality_flag"), "quality_flag").lower() for f in self.quality_flags)
        if len(set(flags)) != len(flags):
            raise ValueError("quality_flags must be unique")
        object.__setattr__(self, "quality_flags", flags)

    @property
    def facial_movement(self) -> tuple[float, ...]:
        return self.facial_movement_proxy_values

    @property
    def goal_id(self) -> str:
        return self.context_id

    @property
    def sample_id(self) -> str:
        return _digest(self.to_dict())[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp, "person_id": self.person_id, "session_id": self.session_id,
            "context_id": self.context_id, "facial_movement_proxy_values": list(self.facial_movement_proxy_values),
            "gaze_zone": self.gaze_zone, "gaze_dwell_seconds": self.gaze_dwell_seconds,
            "head_orientation_zone": self.head_orientation_zone, "head_transition_count": self.head_transition_count,
            "quality_score": self.quality_score, "quality_flags": list(self.quality_flags),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationSample":
        if not isinstance(value, Mapping):
            raise ValueError("sample must be an object")
        movement_value = value.get("facial_movement_proxy_values", value.get("facial_movement", ()))
        flags_value = value.get("quality_flags", ())
        if isinstance(movement_value, (list, tuple)) and len(movement_value) > 32:
            raise ValueError("facial movement proxies exceed safe collection bound")
        if isinstance(flags_value, (list, tuple)) and len(flags_value) > MAX_QUALITY_FLAGS:
            raise ValueError("quality_flags exceed safe collection bound")
        _check_forbidden(value)
        unknown = set(value) - _ALLOWED_SAMPLE
        if unknown:
            raise ValueError(f"unknown sample fields: {sorted(unknown)}")
        if "facial_movement_proxy_values" in value and "facial_movement" in value:
            raise ValueError("provide one facial movement proxy field")
        if "context_id" in value and "goal_id" in value:
            raise ValueError("provide one context/goal identifier")
        movement = value.get("facial_movement_proxy_values", value.get("facial_movement"))
        context = value.get("context_id", value.get("goal_id"))
        required = {"timestamp", "person_id", "session_id", "gaze_zone", "gaze_dwell_seconds", "head_orientation_zone", "head_transition_count", "quality_score"}
        if not required.issubset(value) or movement is None or context is None:
            raise ValueError("sample missing required proxy fields")
        flags_value = value.get("quality_flags", ())
        if isinstance(flags_value, (str, bytes)) or not isinstance(flags_value, (list, tuple)):
            raise ValueError("quality_flags must be a list")
        if not isinstance(movement, (list, tuple)):
            raise ValueError("facial movement proxies must be a list")
        return cls(value["timestamp"], value["person_id"], value["session_id"], context, movement, value["gaze_zone"], value["gaze_dwell_seconds"], value["head_orientation_zone"], value["head_transition_count"], value["quality_score"], flags_value)


def _summary_strings(value: Any, name: str, depth: int = 0, active: set[int] | None = None) -> None:
    if depth > 32:
        raise ValueError(f"{name} exceeds safety depth")
    if isinstance(value, str):
        _forbidden_text(value, name)
        return
    if not isinstance(value, (Mapping, list, tuple)):
        return
    if active is None:
        active = set()
    marker = id(value)
    if marker in active:
        raise ValueError(f"{name} contains a cycle")
    active.add(marker)
    try:
        if isinstance(value, Mapping):
            for key, child in value.items():
                _summary_strings(child, f"{name}.{key}", depth + 1, active)
        else:
            for index, child in enumerate(value):
                _summary_strings(child, f"{name}.{index}", depth + 1, active)
    finally:
        active.remove(marker)


def _feature_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("feature_summary must be an object")
    for key, limit in (("feature_vector", 64), ("gaze_zones", MAX_ZONE_COUNT), ("head_zones", MAX_ZONE_COUNT)):
        nested = value.get(key)
        if isinstance(nested, (list, tuple)) and len(nested) > limit:
            raise ValueError(f"feature_summary.{key} exceeds safe collection bound")
    _check_forbidden(value)
    _summary_strings(value, "feature_summary")
    unknown = set(value) - _ALLOWED_FEATURE_SUMMARY
    if unknown:
        raise ValueError(f"unknown feature summary fields: {sorted(unknown)}")
    if not {"reason", "sample_count", "feature_vector", "gaze_zones", "head_zones"}.issubset(value):
        raise ValueError("feature_summary missing required fields")
    _forbidden_text(_text(value["reason"], "feature_summary.reason"), "feature_summary.reason")
    count = value["sample_count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= FACIAL_WINDOW:
        raise ValueError("feature_summary.sample_count must be 1-3")
    vector = value["feature_vector"]
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("feature_summary.feature_vector must be non-empty")
    if len(vector) > 64 or any(_finite(item, "feature_summary.feature_vector") < 0 or _finite(item, "feature_summary.feature_vector") > 1000 for item in vector):
        raise ValueError("feature_summary.feature_vector values are out of bounds")
    for name in ("gaze_zones", "head_zones"):
        zones = value[name]
        if not isinstance(zones, (list, tuple)) or not zones:
            raise ValueError(f"feature_summary.{name} must be non-empty")
        if len(zones) > count or len(zones) > MAX_ZONE_COUNT:
            raise ValueError(f"feature_summary.{name} exceeds sample-count bound")
        for zone in zones:
            _forbidden_text(_text(zone, f"feature_summary.{name}"), f"feature_summary.{name}")
    if "support_strategy_candidate" in value:
        strategy = _strategy(value["support_strategy_candidate"], "support_strategy_candidate")
    else:
        strategy = None
    if "abstained" in value and not isinstance(value["abstained"], bool):
        raise ValueError("feature_summary.abstained must be boolean")
    result = dict(value)
    if strategy is not None:
        result["support_strategy_candidate"] = strategy
    return result
@dataclass(frozen=True, slots=True)
class EventCandidate:
    event_type: str
    candidate_id: str
    person_id: str
    session_id: str
    context_id: str
    start_timestamp: float
    end_timestamp: float
    feature_summary: Mapping[str, Any]
    provenance: tuple[str, ...]
    quality_score: float
    quality_flags: tuple[str, ...]
    requires_human_review: bool = True
    notice: str = NOTICE
    source_model_digest: str | None = None
    confidence: float | None = None
    evidence_ids: tuple[str, ...] = ()
    source_sample_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or self.event_type not in {"facial_movement", "gaze", "head_orientation", "quality"}:
            raise ValueError("invalid event_type")
        object.__setattr__(self, "candidate_id", _hex(self.candidate_id, "candidate_id", {24}))
        for name in ("person_id", "session_id"):
            object.__setattr__(self, name, _safe_context(getattr(self, name), name))
        object.__setattr__(self, "context_id", _safe_context(self.context_id, "context_id"))
        start, end = _finite(self.start_timestamp, "start_timestamp"), _finite(self.end_timestamp, "end_timestamp")
        if start < 0 or end < 0:
            raise ValueError("candidate timestamps must be non-negative")
        if end < start:
            raise ValueError("candidate end precedes start")
        if end - start > MAX_EVENT_SPAN_SECONDS:
            raise ValueError("candidate span exceeds bounded short window")
        object.__setattr__(self, "start_timestamp", start)
        object.__setattr__(self, "end_timestamp", end)
        object.__setattr__(self, "feature_summary", _freeze_map(_feature_summary(self.feature_summary)))
        if isinstance(self.provenance, (str, bytes)) or not isinstance(self.provenance, (list, tuple)):
            raise ValueError("provenance must be a list or tuple")
        if len(self.provenance) > MAX_PROVENANCE:
            raise ValueError("provenance exceeds safe collection bound")
        if isinstance(self.evidence_ids, (str, bytes)) or not isinstance(self.evidence_ids, (list, tuple)):
            raise ValueError("evidence_ids must be a list or tuple")
        if len(self.evidence_ids) > MAX_PROVENANCE:
            raise ValueError("evidence_ids exceeds safe collection bound")
        provenance = tuple(_hex(v, "provenance", {20, 24, 40, 64}) for v in self.provenance)
        evidence = tuple(_hex(v, "evidence_id", {20, 24, 40, 64}) for v in self.evidence_ids)
        if not provenance or not evidence or len(set(provenance)) != len(provenance) or evidence != provenance:
            raise ValueError("evidence_ids must exactly match non-empty provenance")
        if len(provenance) != self.feature_summary["sample_count"]:
            raise ValueError("feature_summary.sample_count must match provenance length")
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "evidence_ids", evidence)
        score = _finite(self.quality_score, "quality_score")
        if not 0 <= score <= 1:
            raise ValueError("candidate quality must be in [0,1]")
        object.__setattr__(self, "quality_score", score)
        if isinstance(self.quality_flags, (str, bytes)) or not isinstance(self.quality_flags, (list, tuple)):
            raise ValueError("quality_flags must be a list or tuple")
        if len(self.quality_flags) > MAX_QUALITY_FLAGS:
            raise ValueError("quality_flags exceed safe collection bound")
        flags = tuple(_forbidden_text(_text(v, "quality_flag"), "quality_flag") for v in self.quality_flags)
        if len(set(flags)) != len(flags):
            raise ValueError("quality_flags must be unique")
        object.__setattr__(self, "quality_flags", flags)
        if not isinstance(self.requires_human_review, bool) or self.requires_human_review is not True:
            raise ValueError("candidates must remain human-review, non-diagnostic observations")
        if self.notice != NOTICE:
            raise ValueError("candidate notice must be the fixed NOTICE constant")
        if self.source_model_digest is not None:
            object.__setattr__(self, "source_model_digest", _hex(self.source_model_digest, "source_model_digest", {64}))
        if self.confidence is not None:
            confidence = _finite(self.confidence, "confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be in [0,1]")
            object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_sample_digest", _hex(self.source_sample_digest, "source_sample_digest", {64}))
        if self.candidate_id != _digest(self.to_dict(include_id=False))[:24]:
            raise ValueError("candidate_id does not match candidate contents")

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict(include_id=False))
    @property
    def predicted_support_strategy(self) -> str | None:
        value = self.feature_summary.get("support_strategy_candidate")
        return value if isinstance(value, str) else None

    def to_dict(self, include_id: bool = True) -> dict[str, Any]:
        out = {"event_type": self.event_type, "person_id": self.person_id, "session_id": self.session_id, "context_id": self.context_id, "start_timestamp": self.start_timestamp, "end_timestamp": self.end_timestamp, "feature_summary": _json_value(self.feature_summary), "provenance": list(self.provenance), "quality_score": self.quality_score, "quality_flags": list(self.quality_flags), "requires_human_review": True, "notice": self.notice, "source_model_digest": self.source_model_digest, "confidence": self.confidence, "evidence_ids": list(self.evidence_ids), "source_sample_digest": self.source_sample_digest}
        identity = _digest(out)[:24]
        if self.candidate_id != identity:
            raise ValueError("candidate_id does not match candidate contents")
        if include_id:
            out["candidate_id"] = self.candidate_id
        return out


@dataclass(frozen=True, slots=True)
class LabelRecord:
    reviewer_role: str
    reviewer_id: str
    candidate_id: str
    observed_support_strategy: str
    observed_context: str
    outcome: str
    approved: bool

    def __post_init__(self) -> None:
        if not isinstance(self.reviewer_role, str) or self.reviewer_role not in {"expert", "teacher", "parent"}:
            raise ValueError("reviewer_role must be expert, teacher, or parent")
        object.__setattr__(self, "reviewer_id", _safe_context(self.reviewer_id, "reviewer_id"))
        object.__setattr__(self, "candidate_id", _hex(self.candidate_id, "candidate_id", {24}))
        object.__setattr__(self, "observed_support_strategy", _strategy(self.observed_support_strategy, "observed_support_strategy"))
        object.__setattr__(self, "observed_context", _safe_context(self.observed_context, "observed_context"))
        if not isinstance(self.outcome, str) or self.outcome not in {"helpful", "not_helpful", "uncertain", "not_observed"}:
            raise ValueError("invalid outcome")
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be explicit boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"reviewer_role": self.reviewer_role, "reviewer_id": self.reviewer_id, "candidate_id": self.candidate_id, "observed_support_strategy": self.observed_support_strategy, "observed_context": self.observed_context, "outcome": self.outcome, "approved": self.approved}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LabelRecord":
        if not isinstance(value, Mapping):
            raise ValueError("label must be an object")
        _check_forbidden(value)
        needed = {"reviewer_role", "reviewer_id", "candidate_id", "observed_support_strategy", "observed_context", "outcome", "approved"}
        try:
            fields = set(value)
        except (TypeError, ValueError):
            raise ValueError("label fields must be explicit and exact") from None
        if fields != needed:
            raise ValueError("label fields must be explicit and exact")
        return cls(**dict(value))


def _model_payload(model: "PrototypeModel") -> dict[str, Any]:
    return {
        "person_id": model.person_id,
        "strategy_prototypes": {k: list(v) for k, v in sorted(model.strategy_prototypes.items())},
        "training_candidate_ids": sorted(model.training_candidate_ids),
        "training_fingerprints": dict(sorted(model.training_fingerprints.items())),
        "label_provenance": sorted(model.label_provenance),
        "min_quality": model.min_quality,
        "abstention_threshold": model.abstention_threshold,
    }


def _validate_model_structure(model: "PrototypeModel") -> None:
    if not isinstance(model.strategy_prototypes, Mapping) or not model.strategy_prototypes:
        raise ValueError("model needs positive prototypes")
    if len(model.strategy_prototypes) > MAX_STRATEGY_PROTOTYPES:
        raise ValueError("strategy prototypes exceed safe collection bound")
    dimensions: set[int] = set()
    for strategy, vector in model.strategy_prototypes.items():
        if not isinstance(strategy, str):
            raise ValueError("strategy prototype keys must be strings")
        _strategy(strategy, "strategy prototype")
        if not isinstance(vector, (list, tuple)) or isinstance(vector, (str, bytes)) or not vector or len(vector) > 64:
            raise ValueError("strategy prototypes must have bounded non-empty vectors")
        values = tuple(_finite(item, "strategy prototype") for item in vector)
        if any(item < 0 or item > 1000 for item in values):
            raise ValueError("strategy prototype values are out of bounds")
        dimensions.add(len(values))
    if len(dimensions) != 1:
        raise ValueError("strategy prototypes must share one vector dimension")
    if isinstance(model.training_candidate_ids, (str, bytes)) or not isinstance(model.training_candidate_ids, (list, tuple)):
        raise ValueError("training candidate IDs must be a list or tuple")
    if len(model.training_candidate_ids) > MAX_TRAINING_MANIFEST:
        raise ValueError("training candidate manifest exceeds safe collection bound")
    ids = tuple(model.training_candidate_ids)
    if not ids or any(_hex(item, "training candidate ID", {24}) != item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("training candidate IDs must be canonical and unique")
    if not isinstance(model.training_fingerprints, Mapping):
        raise ValueError("training fingerprints must be an object")
    if len(model.training_fingerprints) > MAX_TRAINING_MANIFEST:
        raise ValueError("training fingerprint manifest exceeds safe collection bound")
    fingerprints = dict(model.training_fingerprints)
    if set(ids) != set(fingerprints):
        raise ValueError("training candidate IDs must exactly equal fingerprint keys")
    for candidate_id, fingerprint in fingerprints.items():
        _hex(candidate_id, "training fingerprint key", {24})
        _hex(fingerprint, "training fingerprint", {64})
    if isinstance(model.label_provenance, (str, bytes)) or not isinstance(model.label_provenance, (list, tuple)):
        raise ValueError("label provenance must be a list or tuple")
    if len(model.label_provenance) > MAX_LABEL_PROVENANCE:
        raise ValueError("label provenance exceeds safe collection bound")
    provenance = tuple(model.label_provenance)
    if not provenance or any(_hex(item, "label provenance", {64}) != item for item in provenance):
        raise ValueError("label provenance must contain valid digests")
    if len(set(provenance)) != len(provenance):
        raise ValueError("label provenance digests must be unique")

@dataclass(frozen=True, slots=True)
class PrototypeModel:
    person_id: str
    strategy_prototypes: Mapping[str, tuple[float, ...]]
    training_candidate_ids: tuple[str, ...]
    training_fingerprints: Mapping[str, str]
    label_provenance: tuple[str, ...]
    model_digest: str
    min_quality: float = QUALITY_THRESHOLD
    abstention_threshold: float = MATCH_CONFIDENCE_THRESHOLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "person_id", _safe_context(self.person_id, "person_id"))
        if not isinstance(self.strategy_prototypes, Mapping):
            raise ValueError("strategy_prototypes must be an object")
        if len(self.strategy_prototypes) > MAX_STRATEGY_PROTOTYPES:
            raise ValueError("strategy prototypes exceed safe collection bound")
        prototypes: dict[str, tuple[float, ...]] = {}
        for key, vector in self.strategy_prototypes.items():
            if not isinstance(key, str):
                raise ValueError("strategy prototype keys must be strings")
            strategy = _strategy(key, "strategy prototype")
            if strategy in prototypes:
                raise ValueError("duplicate strategy prototype keys")
            if isinstance(vector, (str, bytes)) or not isinstance(vector, (list, tuple)):
                raise ValueError("strategy prototype vectors must be lists or tuples")
            prototypes[strategy] = tuple(vector)
        object.__setattr__(self, "strategy_prototypes", _freeze_map(prototypes))
        if isinstance(self.training_candidate_ids, (str, bytes)) or not isinstance(self.training_candidate_ids, (list, tuple)):
            raise ValueError("training candidate IDs must be a list or tuple")
        if len(self.training_candidate_ids) > MAX_TRAINING_MANIFEST:
            raise ValueError("training candidate manifest exceeds safe collection bound")
        object.__setattr__(self, "training_candidate_ids", tuple(sorted(self.training_candidate_ids, key=lambda item: str(item))))
        if not isinstance(self.training_fingerprints, Mapping):
            raise ValueError("training fingerprints must be an object")
        if len(self.training_fingerprints) > MAX_TRAINING_MANIFEST:
            raise ValueError("training fingerprint manifest exceeds safe collection bound")
        object.__setattr__(
            self,
            "training_fingerprints",
            _freeze_map(dict(sorted(self.training_fingerprints.items(), key=lambda item: str(item[0])))),
        )
        if isinstance(self.label_provenance, (str, bytes)) or not isinstance(self.label_provenance, (list, tuple)):
            raise ValueError("label provenance must be a list or tuple")
        if len(self.label_provenance) > MAX_LABEL_PROVENANCE:
            raise ValueError("label provenance exceeds safe collection bound")
        object.__setattr__(self, "label_provenance", tuple(sorted(self.label_provenance, key=lambda item: str(item))))
        for name in ("min_quality", "abstention_threshold"):
            value = _finite(getattr(self, name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
            object.__setattr__(self, name, value)
        _hex(self.model_digest, "model_digest", {64})
        _validate_model_structure(self)
        if self.model_digest != _digest(_model_payload(self)):
            raise ValueError("model_digest does not match normalized model payload")

    def to_dict(self) -> dict[str, Any]:
        _verify_model(self)
        return {"person_id": self.person_id, "strategy_prototypes": {k: list(v) for k, v in sorted(self.strategy_prototypes.items())}, "training_candidate_ids": list(self.training_candidate_ids), "training_fingerprints": dict(sorted(self.training_fingerprints.items())), "label_provenance": list(self.label_provenance), "model_digest": self.model_digest, "min_quality": self.min_quality, "abstention_threshold": self.abstention_threshold}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrototypeModel":
        if not isinstance(value, Mapping):
            raise ValueError("model must be an object")
        for key, limit in (
            ("strategy_prototypes", MAX_STRATEGY_PROTOTYPES),
            ("training_candidate_ids", MAX_TRAINING_MANIFEST),
            ("training_fingerprints", MAX_TRAINING_MANIFEST),
            ("label_provenance", MAX_LABEL_PROVENANCE),
        ):
            nested = value.get(key)
            if isinstance(nested, (Mapping, list, tuple)) and len(nested) > limit:
                raise ValueError(f"{key} exceeds safe collection bound")
        _check_forbidden(value)
        try:
            fields = set(value)
        except (TypeError, ValueError):
            raise ValueError("model fields must be explicit and exact") from None
        if fields != _MODEL_FIELDS:
            raise ValueError("model fields must be explicit and exact")
        try:
            model = cls(
                value["person_id"],
                value["strategy_prototypes"],
                value["training_candidate_ids"],
                value["training_fingerprints"],
                value["label_provenance"],
                value["model_digest"],
                value["min_quality"],
                value["abstention_threshold"],
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("malformed model manifest") from exc
        _verify_model(model)
        return model


def _samples(samples: Iterable[ObservationSample | Mapping[str, Any]]) -> list[ObservationSample]:
    out: list[ObservationSample] = []
    for sample in samples:
        if len(out) >= MAX_SOURCE_SAMPLES:
            raise ValueError("samples exceed safe collection bound")
        out.append(sample if isinstance(sample, ObservationSample) else ObservationSample.from_dict(sample))
    if any(a.timestamp > b.timestamp for a, b in zip(out, out[1:])): raise ValueError("samples must be ordered by timestamp")
    ids = [s.sample_id for s in out]
    if len(ids) != len(set(ids)): raise ValueError("duplicate sample IDs")
    return out


def _vector(window: Sequence[ObservationSample]) -> tuple[float, ...]:
    width = max(len(s.facial_movement_proxy_values) for s in window)
    movement = [sum((s.facial_movement_proxy_values[i] if i < len(s.facial_movement_proxy_values) else 0) for s in window) / len(window) for i in range(width)]
    dwell = sum(s.gaze_dwell_seconds for s in window) / len(window)
    gaze_transitions = sum(a.gaze_zone != b.gaze_zone for a, b in zip(window, window[1:]))
    head = sum(s.head_transition_count for s in window) / len(window)
    quality = sum(s.quality_score for s in window) / len(window)
    return tuple(round(v, 6) for v in (*movement, dwell / 10, gaze_transitions / max(1, len(window)-1), head / 10, quality))
def _source_sample_digest(window: Sequence[ObservationSample]) -> str:
    return _digest([sample.to_dict() for sample in window])


def _make_candidate(kind: str, window: Sequence[ObservationSample], reason: str) -> EventCandidate:
    quality = round(sum(s.quality_score for s in window) / len(window), 6)
    flags = tuple(sorted({f for s in window for f in s.quality_flags}))
    features = {"reason": reason, "sample_count": len(window), "feature_vector": list(_vector(window)), "gaze_zones": sorted({s.gaze_zone for s in window}), "head_zones": sorted({s.head_orientation_zone for s in window})}
    provenance = tuple(s.sample_id for s in window)
    source_sample_digest = _source_sample_digest(window)
    base = {"event_type": kind, "person_id": window[0].person_id, "session_id": window[0].session_id, "context_id": window[0].context_id, "start_timestamp": window[0].timestamp, "end_timestamp": window[-1].timestamp, "feature_summary": features, "provenance": list(provenance), "quality_score": quality, "quality_flags": list(flags), "requires_human_review": True, "notice": NOTICE, "source_model_digest": None, "confidence": None, "evidence_ids": list(provenance), "source_sample_digest": source_sample_digest}
    return EventCandidate(candidate_id=_digest(base)[:24], **base)


def extract_event_candidates(samples: Iterable[ObservationSample | Mapping[str, Any]]) -> list[EventCandidate]:
    """Extract bounded observational windows; never infer a mental state.

    A three-sample window keeps eventization local.  Facial movement uses mean
    absolute proxy change (>= .20), gaze uses dwell (>= 2 seconds) or a zone
    transition, head uses >= 2 transitions, and quality uses score < .60 or a
    quality flag.  Thresholds are deliberately conservative and documented.
    """
    ordered = _samples(samples)
    groups: dict[tuple[str, str, str], list[ObservationSample]] = {}
    for sample in ordered: groups.setdefault((sample.person_id, sample.session_id, sample.context_id), []).append(sample)
    result: list[EventCandidate] = []
    def add_candidate(kind: str, window: Sequence[ObservationSample], reason: str) -> None:
        if len(result) >= MAX_SAFE_COLLECTION:
            raise ValueError("extracted candidates exceed safe collection bound")
        result.append(_make_candidate(kind, window, reason))

    for group in groups.values():
        for index, sample in enumerate(group):
            window = group[max(0, index - FACIAL_WINDOW + 1): index + 1]
            while len(window) > 1 and window[-1].timestamp - window[0].timestamp > MAX_EVENT_SPAN_SECONDS:
                window = window[1:]
            previous = window[-2] if len(window) > 1 else None
            if previous is not None:
                width = max(len(previous.facial_movement_proxy_values), len(sample.facial_movement_proxy_values))
                delta = sum(abs((sample.facial_movement_proxy_values[i] if i < len(sample.facial_movement_proxy_values) else 0) - (previous.facial_movement_proxy_values[i] if i < len(previous.facial_movement_proxy_values) else 0)) for i in range(width)) / width
                if delta >= FACIAL_CHANGE_THRESHOLD: add_candidate("facial_movement", window, "proxy_change_at_or_above_0.20")
            if sample.gaze_dwell_seconds >= GAZE_DWELL_THRESHOLD or (previous is not None and sample.gaze_zone != previous.gaze_zone):
                add_candidate("gaze", window, "dwell_at_or_above_2s_or_zone_transition")
            if sum(s.head_transition_count for s in window) >= HEAD_TRANSITION_THRESHOLD: add_candidate("head_orientation", window, "head_transitions_at_or_above_2")
            if sample.quality_score < QUALITY_THRESHOLD or sample.quality_flags: add_candidate("quality", window, "sensor_unavailable_or_quality_below_0.60")
    # IDs may converge only for identical windows/reasons; preserve one deterministic event.
    unique = {c.candidate_id: c for c in result}
    return [unique[key] for key in sorted(unique, key=lambda k: (unique[k].start_timestamp, unique[k].event_type, k))]


def _candidate_map(candidates: Iterable[EventCandidate | Mapping[str, Any]]) -> dict[str, EventCandidate]:
    parsed: list[EventCandidate] = []
    for candidate in candidates:
        if len(parsed) >= MAX_SAFE_COLLECTION:
            raise ValueError("candidates exceed safe collection bound")
        parsed.append(candidate if isinstance(candidate, EventCandidate) else _candidate_from_dict(candidate))
    out: dict[str, EventCandidate] = {}
    for candidate in parsed:
        if candidate.candidate_id in out or candidate.candidate_id != _digest(candidate.to_dict(include_id=False))[:24]:
            raise ValueError("malformed, tampered, or duplicate candidate ID")
        out[candidate.candidate_id] = candidate
    return out


def _candidate_from_dict(value: Mapping[str, Any]) -> EventCandidate:
    if not isinstance(value, Mapping):
        raise ValueError("candidate must be an object")
    for key, limit in (("provenance", MAX_PROVENANCE), ("evidence_ids", MAX_PROVENANCE), ("quality_flags", MAX_QUALITY_FLAGS)):
        nested = value.get(key)
        if isinstance(nested, (list, tuple)) and len(nested) > limit:
            raise ValueError(f"{key} exceeds safe collection bound")
    summary = value.get("feature_summary")
    if isinstance(summary, Mapping):
        for key in ("gaze_zones", "head_zones"):
            zones = summary.get(key)
            if isinstance(zones, (list, tuple)) and len(zones) > MAX_ZONE_COUNT:
                raise ValueError(f"feature_summary.{key} exceeds safe collection bound")
    _check_forbidden(value)
    try:
        fields = set(value)
    except (TypeError, ValueError):
        raise ValueError("candidate fields must be explicit and exact") from None
    if fields != _CANDIDATE_FIELDS:
        raise ValueError("candidate fields must be explicit and exact")
    data = dict(value)
    candidate_id = data.pop("candidate_id")
    try:
        return EventCandidate(candidate_id=candidate_id, **data)
    except TypeError as exc:
        raise ValueError("malformed candidate") from exc


def train_per_person_model(candidates: Iterable[EventCandidate | Mapping[str, Any]], labels: Iterable[LabelRecord | Mapping[str, Any]], *, source_samples: Iterable[ObservationSample | Mapping[str, Any]], person_id: str | None = None) -> PrototypeModel:
    """Train prototypes only after independent human approval and no conflict."""
    # Re-extract candidates from the original structured samples and compare
    # fingerprints before fitting. A caller cannot make a modified candidate
    # trustworthy merely by recomputing its candidate_id or manifest digest.
    cmap = _candidate_map(candidates)
    if not cmap: raise ValueError("no candidates")
    source_candidates = _candidate_map(extract_event_candidates(source_samples))
    for candidate_id, candidate in cmap.items():
        source_candidate = source_candidates.get(candidate_id)
        if source_candidate is None or source_candidate.fingerprint != candidate.fingerprint:
            raise ValueError("candidate does not match supplied source samples")
    people = {c.person_id for c in cmap.values()}
    if len(people) != 1:
        raise ValueError("mixed people are rejected; train one person")
    only_person = next(iter(people))
    if person_id is None:
        person_id = only_person
    if person_id != only_person: raise ValueError("person has no candidates")
    parsed: list[LabelRecord] = []
    for label in labels:
        if len(parsed) >= MAX_SAFE_COLLECTION:
            raise ValueError("labels exceed safe collection bound")
        parsed.append(label if isinstance(label, LabelRecord) else LabelRecord.from_dict(label))
    seen_reviewer: set[tuple[str, str, str]] = set(); positives: list[tuple[EventCandidate, LabelRecord]] = []
    by_candidate: dict[str, list[LabelRecord]] = {}
    for label in parsed:
        if label.candidate_id not in cmap:
            raise ValueError("label references unknown candidate")
        candidate = cmap[label.candidate_id]
        if candidate.person_id != person_id:
            raise ValueError("mixed-person label leakage rejected")
        if label.observed_context != candidate.context_id:
            raise ValueError("label context does not match candidate context")
        key = (label.reviewer_role, label.reviewer_id, label.candidate_id)
        if key in seen_reviewer:
            raise ValueError("duplicate reviewer label")
        seen_reviewer.add(key)
        by_candidate.setdefault(label.candidate_id, []).append(label)
        if label.approved and label.outcome == "helpful":
    # Only explicitly approved, helpful outcomes become positive prototype
    # examples. Rejected, uncertain, or unobserved labels remain review history
    # and cannot pull a support strategy toward a prediction.
            positives.append((candidate, label))
    if not positives:
        raise ValueError("no approved positive labels")
    for cid, rows in by_candidate.items():
        approved_rows = [r for r in rows if r.approved]
        outcomes = {r.outcome for r in approved_rows}
        strategies = {r.observed_support_strategy for r in approved_rows if r.outcome == "helpful"}
        if len(outcomes) > 1 or len(strategies) > 1:
            raise ValueError("approved reviewer disagreement")
        if outcomes & {"uncertain", "not_observed"}:
            raise ValueError("approved abstention is rejected")
        if any(r.outcome == "helpful" for r in approved_rows):
            independent = {r.reviewer_id for r in approved_rows}
            if len(independent) < 2:
                raise ValueError("each positive candidate requires two independent approvals")
    if any(l.outcome in {"uncertain", "not_observed"} and l.approved for l in parsed):
        raise ValueError("approved abstention is rejected")
    if any(c.quality_score < QUALITY_THRESHOLD for c, _ in positives):
        raise ValueError("low-quality positive candidate rejected")
    prototypes: dict[str, list[tuple[float, ...]]] = {}
    training_ids: list[str] = []; fingerprints: dict[str, str] = {}; provenance: list[str] = []
    for candidate, label in sorted(positives, key=lambda pair: (pair[0].candidate_id, pair[1].reviewer_id)):
        prototypes.setdefault(label.observed_support_strategy, []).append(tuple(candidate.feature_summary["feature_vector"]))
        if candidate.candidate_id not in training_ids:
            training_ids.append(candidate.candidate_id)
        fingerprints[candidate.candidate_id] = candidate.fingerprint
        provenance.append(_digest(label.to_dict()))
    if any(len({len(row) for row in rows}) != 1 for rows in prototypes.values()):
        raise ValueError("positive feature vectors must share one dimension per strategy")
    averaged = {strategy: tuple(round(sum(row[i] for row in rows) / len(rows), 6) for i in range(len(rows[0]))) for strategy, rows in prototypes.items()}
    # A prototype is the arithmetic mean of approved feature vectors for one
    # strategy. This is intentionally explainable: distance can be inspected
    # by a reviewer and does not claim a mental state or diagnosis.
    payload = {"person_id": person_id, "strategy_prototypes": {k: list(v) for k, v in sorted(averaged.items())}, "training_candidate_ids": sorted(training_ids), "training_fingerprints": dict(sorted(fingerprints.items())), "label_provenance": sorted(provenance), "min_quality": QUALITY_THRESHOLD, "abstention_threshold": MATCH_CONFIDENCE_THRESHOLD}
    return PrototypeModel(model_digest=_digest(payload), **payload)


def train_model(candidates: Iterable[EventCandidate | Mapping[str, Any]], labels: Iterable[LabelRecord | Mapping[str, Any]], *, source_samples: Iterable[ObservationSample | Mapping[str, Any]], person_id: str | None = None) -> PrototypeModel:
    return train_per_person_model(candidates, labels, source_samples=source_samples, person_id=person_id)


def _verify_model(model: PrototypeModel) -> None:
    if not isinstance(model, PrototypeModel):
        raise ValueError("model must be a PrototypeModel")
    _validate_model_structure(model)
    if model.model_digest != _digest(_model_payload(model)):
        raise ValueError("tampered model digest")


def match_reviewable_candidates(model: PrototypeModel | Mapping[str, Any], samples: Iterable[ObservationSample | Mapping[str, Any]]) -> list[EventCandidate]:
    """Return only review candidates; low confidence abstains instead of deciding."""
    # Matching is nearest-prototype retrieval, not an automatic decision. The
    # model refuses another person's data, training windows, low-quality input,
    # and confidence at/below the abstention threshold; surviving matches are
    # still EventCandidate objects requiring human review.
    model = model if isinstance(model, PrototypeModel) else PrototypeModel.from_dict(model)
    _verify_model(model)
    candidates = extract_event_candidates(samples)
    out: list[EventCandidate] = []
    for candidate in candidates:
        if candidate.person_id != model.person_id or candidate.quality_score < model.min_quality or candidate.candidate_id in model.training_candidate_ids: continue
        vector = tuple(candidate.feature_summary["feature_vector"])
        scored = []
        for strategy, prototype in model.strategy_prototypes.items():
            if len(prototype) != len(vector): continue
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, prototype)))
            scored.append((distance, strategy))
        if not scored: continue
        distance, strategy = sorted(scored)[0]
        confidence = round(max(0.0, 1.0 - distance / 2.0), 6)
        if confidence <= model.abstention_threshold: continue
        summary = dict(candidate.feature_summary); summary["support_strategy_candidate"] = strategy; summary["abstained"] = False
        base = candidate.to_dict(include_id=False); base.update({"feature_summary": summary, "source_model_digest": model.model_digest, "confidence": confidence, "evidence_ids": list(candidate.provenance)})
        out.append(EventCandidate(candidate_id=_digest(base)[:24], **base))
    return sorted(out, key=lambda c: (c.start_timestamp, c.candidate_id))


def dumps(value: Any) -> str:
    if isinstance(value, ObservationSample): payload = value.to_dict()
    elif isinstance(value, EventCandidate): payload = value.to_dict()
    elif isinstance(value, LabelRecord): payload = value.to_dict()
    elif isinstance(value, PrototypeModel): payload = value.to_dict()
    else: payload = value
    _check_forbidden(payload)
    _check_json_numbers(payload, "payload.")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _bounded_json_loads(text: str | bytes) -> Any:
    """Parse bounded UTF-8 JSON text and normalize parser failures to ValueError."""
    if isinstance(text, str):
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("input is not valid UTF-8") from None
    elif isinstance(text, bytes):
        encoded = text
    else:
        raise ValueError("input must be UTF-8 text")
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds MAX_INPUT_BYTES")
    try:
        decoded = encoded.decode("utf-8")
        return json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(str(exc)) from None


def loads_sample(text: str | bytes) -> ObservationSample: return ObservationSample.from_dict(_bounded_json_loads(text))
def loads_candidate(text: str | bytes) -> EventCandidate: return _candidate_from_dict(_bounded_json_loads(text))
def loads_label(text: str | bytes) -> LabelRecord: return LabelRecord.from_dict(_bounded_json_loads(text))
def loads_model(text: str | bytes) -> PrototypeModel: return PrototypeModel.from_dict(_bounded_json_loads(text))
def serialize_sample(sample: ObservationSample) -> str:
    return dumps(sample)


def serialize_candidate(candidate: EventCandidate) -> str:
    return dumps(candidate)


def serialize_label(label: LabelRecord) -> str:
    return dumps(label)


def serialize_model(model: PrototypeModel) -> str:
    return dumps(model)
# Small public aliases keep the vocabulary ergonomic while retaining one canonical
# implementation and one serialization format.
StructuredObservationSample = ObservationSample
CandidateEvent = EventCandidate
ReviewLabel = LabelRecord
PerPersonPrototypeModel = PrototypeModel
extract_candidates = extract_event_candidates
sample_to_json = lambda sample: dumps(sample)
candidate_to_json = lambda candidate: dumps(candidate)
label_to_json = lambda label: dumps(label)
model_to_json = lambda model: dumps(model)
sample_from_json = loads_sample
candidate_from_json = loads_candidate
label_from_json = loads_label
model_from_json = loads_model


def _demo_samples() -> list[ObservationSample]:
    return [ObservationSample(float(i), "adult-demo", "session-demo", "goal-demo", (0.1 + (0.25 if i >= 2 else 0), 0.2), "center" if i < 2 else "left", 2.0 if i >= 2 else 0.5, "level", 1 if i >= 2 else 0, 0.95) for i in range(4)]


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _read_bounded(handle: Any) -> str:
    """Read at most MAX_INPUT_BYTES UTF-8 bytes from a text or binary stream."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = handle.read(min(8192, MAX_INPUT_BYTES - total + 1))
        if chunk in (b"", ""):
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        elif not isinstance(chunk, bytes):
            raise TypeError("input stream must return text or bytes")
        total += len(chunk)
        if total > MAX_INPUT_BYTES:
            raise ValueError("input exceeds MAX_INPUT_BYTES")
        chunks.append(chunk)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        raise UnicodeError("input is not valid UTF-8") from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = _JsonArgumentParser(description="offline proxy event candidates (non-diagnostic)")
    # The CLI accepts bounded JSON summaries only. It never opens a camera,
    # reads raw media, calls network/GPT services, or mutates a learning plan.
    # Parser, I/O, schema, and resource errors share one safe JSON boundary.
    parser.add_argument("--demo", action="store_true"); parser.add_argument("--train", action="store_true"); parser.add_argument("--match", action="store_true"); parser.add_argument("--input", type=str, help="JSON file, or stdin when omitted")
    try:
        args = parser.parse_args(argv)
        if sum((bool(args.demo), bool(args.train), bool(args.match))) > 1:
            raise ValueError("modes are mutually exclusive")
        if args.demo:
            samples = _demo_samples(); candidates = extract_event_candidates(samples)
            labels = [LabelRecord("expert", "expert-demo", candidates[0].candidate_id, "visual_pause", "goal-demo", "helpful", True), LabelRecord("teacher", "teacher-demo", candidates[0].candidate_id, "visual_pause", "goal-demo", "helpful", True)] if candidates else []
            model = train_model(candidates, labels, source_samples=samples) if candidates else None
            matches = match_reviewable_candidates(model, samples) if model else []
            payload = {"samples": [s.to_dict() for s in samples], "candidates": [c.to_dict() for c in candidates], "model": model.to_dict() if model else None, "matches": [c.to_dict() for c in matches], "notice": NOTICE}
        else:
            if args.input:
                with open(args.input, "rb") as handle:
                    source = _read_bounded(handle)
            else:
                source = _read_bounded(sys.stdin)
            data = json.loads(source)
            if not isinstance(data, Mapping):
                raise ValueError("input JSON root must be an object")
            if args.train:
                allowed_root = {"samples", "labels", "person_id"}
            elif args.match:
                allowed_root = {"samples", "model"}
            else:
                allowed_root = {"samples"}
            _check_forbidden(data, "root.")
            unknown_root = set(data) - allowed_root
            if unknown_root:
                raise ValueError(f"unknown root fields: {sorted(unknown_root)}")
            samples = _samples(data.get("samples", []))
            candidates = extract_event_candidates(samples)
            if args.train:
                model = train_model(candidates, data.get("labels", []), source_samples=samples, person_id=data.get("person_id")); payload = {"model": model.to_dict(), "candidates": [c.to_dict() for c in candidates], "notice": NOTICE}
            elif args.match:
                payload = {"matches": [c.to_dict() for c in match_reviewable_candidates(data["model"], samples)], "notice": NOTICE}
            else: payload = {"candidates": [c.to_dict() for c in candidates], "notice": NOTICE}
        print(json.dumps(payload, sort_keys=True, separators=(",", ":"))); return 0
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError, UnicodeError, RecursionError) as exc:
        print(json.dumps({"error": str(exc), "notice": NOTICE}, sort_keys=True, separators=(",", ":"))); return 2


if __name__ == "__main__": raise SystemExit(main())
