"""Offline, deterministic, teacher-approved personalization primitives.

This module deliberately consumes event *summaries* only.  It does not import a
camera, UI, network client, or machine-learning runtime.
"""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
MODEL_CONFIG_VERSION = "centroid-baseline-v1"
MAX_DURATION_SECONDS = 300.0
MIN_QUALITY_SCORE = 0.5

# These are proxy event names, not interpretations of a person's internal state.
def _strict_schema_version(value: Any, *, name: str) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ValueError(f"unsupported {name} schema version")

ALLOWED_EVENT_TYPES = (
    "face_missing",
    "gaze_diverted",
    "posture_shifted",
    "task_start",
    "task_complete",
    "transition",
    "support_used",
    "session_start",
    "session_end",
    "prompt",
)
ALLOWED_ZONE_TRANSITIONS = ("none", "left", "right", "up", "down")
ALLOWED_TEACHER_CONTEXTS = (
    "none",
    "visual_schedule",
    "short_prompt",
    "transition_preview",
    "brief_break",
    "reinforcement",
)
SUPPORT_LABELS = (
    "visual_schedule",
    "short_prompt",
    "transition_preview",
    "brief_break",
    "reinforcement",
)
OUTCOME_LABELS = (
    "helpful",
    "not_helpful",
    "neutral",
    "effective",
    "ineffective",
    "mixed",
    "not_observed",
)

# Keys that are never safe input, even when nested in a purported summary.
_FORBIDDEN_KEY_PARTS = (
    "landmark",
    "frame_score",
    "per_frame",
    "emotion",
    "attention",
    "preference",
    "diagnos",
    "compliance",
    "asd",
)
_FEATURE_NAMES = (
    *(f"event_type:{name}" for name in ALLOWED_EVENT_TYPES),
    "duration_ratio",
    "quality_ok",
    "quality_complete",
    "teacher_reviewed",
    *(f"zone_transition:{name}" for name in ALLOWED_ZONE_TRANSITIONS),
    *(f"teacher_context:{name}" for name in ALLOWED_TEACHER_CONTEXTS),
)
_ALLOWED_SUMMARY_KEYS = {
    "person_id",
    "event_id",
    "event_type",
    "duration_seconds",
    "quality_flags",
    "quality_score",
    "zone_transition",
    "teacher_context",
    "trigger_values",
}
_ALLOWED_TRIGGER_KEYS = {
    "duration_seconds",
    "quality_flags",
    "quality_score",
    "zone_transition",
    "teacher_context",
}
_ALLOWED_QUALITY_KEYS = {"quality_ok", "quality_complete", "teacher_reviewed"}


def _reject_forbidden(value: Any) -> None:
    """Reject forbidden nested keys without recursion or cycle failures."""
    active: set[int] = set()
    stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
    while stack:
        current, depth, exiting = stack.pop()
        if exiting:
            active.discard(id(current))
            continue
        if depth > 128:
            raise ValueError("nested event fields exceed permitted depth")
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in active:
                raise ValueError("cyclic nested event fields are not allowed")
            active.add(marker)
            stack.append((current, depth, True))
            for key, nested in current.items():
                lowered = str(key).lower()
                if any(part in lowered for part in _FORBIDDEN_KEY_PARTS):
                    raise ValueError(f"forbidden event field: {key}")
                stack.append((nested, depth + 1, False))
        elif isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in active:
                raise ValueError("cyclic nested event fields are not allowed")
            active.add(marker)
            stack.append((current, depth, True))
            stack.extend((nested, depth + 1, False) for nested in current)


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _strict_keys(mapping: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unknown {name} field(s): {sorted(map(str, unknown))}")


def _clean_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class EventFeatureRow:
    """Immutable bounded numeric representation of one safe event summary."""

    person_id: str
    event_id: str
    event_type: str
    features: tuple[float, ...]
    quality_ok: bool
    quality_score: float
    quality_complete: bool = False
    teacher_reviewed: bool = False

    def __post_init__(self) -> None:
        person_id = _clean_id(self.person_id, name="person_id")
        event_id = _clean_id(self.event_id, name="event_id")
        if not isinstance(self.event_type, str) or self.event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError("unknown event type")
        if not isinstance(self.features, (tuple, list)) or len(self.features) != len(_FEATURE_NAMES):
            raise ValueError("feature vector shape mismatch")
        features = tuple(_finite_number(value, name="feature") for value in self.features)
        if any(value < 0.0 or value > 1.0 for value in features):
            raise ValueError("feature vector value outside bounds")
        event_count = len(ALLOWED_EVENT_TYPES)
        event_features = features[:event_count]
        if any(value not in (0.0, 1.0) for value in event_features):
            raise ValueError("event feature encoding must be binary")
        if sum(event_features) != 1.0 or event_features[ALLOWED_EVENT_TYPES.index(self.event_type)] != 1.0:
            raise ValueError("event feature encoding mismatch")
        if not isinstance(self.quality_ok, bool):
            raise ValueError("quality_ok must be boolean")
        if not isinstance(self.quality_complete, bool):
            raise ValueError("quality_complete must be boolean")
        if not isinstance(self.teacher_reviewed, bool):
            raise ValueError("teacher_reviewed must be boolean")
        quality_score = _finite_number(self.quality_score, name="quality_score")
        if not 0.0 <= quality_score <= 1.0:
            raise ValueError("quality_score must be between 0 and 1")
        if features[event_count + 1] != float(self.quality_ok):
            raise ValueError("quality_ok feature encoding mismatch")
        if features[event_count + 2] != float(self.quality_complete):
            raise ValueError("quality_complete feature encoding mismatch")
        if features[event_count + 3] != float(self.teacher_reviewed):
            raise ValueError("teacher_reviewed feature encoding mismatch")
        zone_start = event_count + 4
        zone_features = features[zone_start : zone_start + len(ALLOWED_ZONE_TRANSITIONS)]
        if any(value not in (0.0, 1.0) for value in zone_features) or sum(zone_features) != 1.0:
            raise ValueError("zone feature encoding mismatch")
        context_start = zone_start + len(ALLOWED_ZONE_TRANSITIONS)
        context_features = features[context_start:]
        if any(value not in (0.0, 1.0) for value in context_features) or sum(context_features) != 1.0:
            raise ValueError("context feature encoding mismatch")
        object.__setattr__(self, "person_id", person_id)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "quality_score", quality_score)

    @classmethod
    def from_summary(
        cls,
        summary: Mapping[str, Any],
        *,
        person_id: str | None = None,
        event_id: str | None = None,
    ) -> "EventFeatureRow":
        if not isinstance(summary, Mapping):
            raise ValueError("event summary must be a mapping")
        _reject_forbidden(summary)
        _strict_keys(summary, _ALLOWED_SUMMARY_KEYS, name="event summary")
        trigger = summary.get("trigger_values", {})
        if trigger is None:
            trigger = {}
        if not isinstance(trigger, Mapping):
            raise ValueError("trigger_values must be a mapping")
        _strict_keys(trigger, _ALLOWED_TRIGGER_KEYS, name="trigger")

        def field(name: str, default: Any = None) -> Any:
            return summary[name] if name in summary else trigger.get(name, default)

        pid = _clean_id(person_id if person_id is not None else summary.get("person_id"), name="person_id")
        eid = _clean_id(event_id if event_id is not None else summary.get("event_id"), name="event_id")
        event_type = field("event_type")
        if not isinstance(event_type, str) or event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError("unknown event type")
        duration = _finite_number(field("duration_seconds", 0.0), name="duration_seconds")
        if duration < 0 or duration > MAX_DURATION_SECONDS:
            raise ValueError("duration_seconds outside permitted bounds")

        quality_flags = field("quality_flags", {})
        if quality_flags is None:
            quality_flags = {}
        if not isinstance(quality_flags, Mapping):
            raise ValueError("quality_flags must be a mapping")
        _strict_keys(quality_flags, _ALLOWED_QUALITY_KEYS, name="quality")
        flags: dict[str, bool] = {}
        for key in _ALLOWED_QUALITY_KEYS:
            value = quality_flags.get(key, key == "quality_ok")
            if not isinstance(value, bool):
                raise ValueError(f"quality flag {key} must be boolean")
            flags[key] = value
        quality_score = _finite_number(field("quality_score", 1.0), name="quality_score")
        if not 0.0 <= quality_score <= 1.0:
            raise ValueError("quality_score must be between 0 and 1")
        zone = field("zone_transition", "none")
        if zone is None:
            zone = "none"
        if not isinstance(zone, str) or zone not in ALLOWED_ZONE_TRANSITIONS:
            raise ValueError("unknown zone transition")
        context = field("teacher_context", "none")
        if context is None:
            context = "none"
        if not isinstance(context, str) or context not in ALLOWED_TEACHER_CONTEXTS:
            raise ValueError("unknown teacher context")

        features = tuple(
            [float(event_type == name) for name in ALLOWED_EVENT_TYPES]
            + [round(duration / MAX_DURATION_SECONDS, 8)]
            + [float(flags[name]) for name in ("quality_ok", "quality_complete", "teacher_reviewed")]
            + [float(zone == name) for name in ALLOWED_ZONE_TRANSITIONS]
            + [float(context == name) for name in ALLOWED_TEACHER_CONTEXTS]
        )
        return cls(
            pid,
            eid,
            event_type,
            features,
            flags["quality_ok"],
            quality_score,
            flags["quality_complete"],
            flags["teacher_reviewed"],
        )

    @property
    def feature_vector(self) -> tuple[float, ...]:
        return self.features

    from_event_summary = from_summary
    from_event = from_summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "person_id": self.person_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "features": list(self.features),
            "quality_ok": self.quality_ok,
            "quality_score": self.quality_score,
            "quality_complete": self.quality_complete,
            "teacher_reviewed": self.teacher_reviewed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EventFeatureRow":
        if not isinstance(payload, Mapping):
            raise ValueError("feature row must be a mapping")
        _strict_keys(
            payload,
            {
                "schema_version",
                "person_id",
                "event_id",
                "event_type",
                "features",
                "quality_ok",
                "quality_score",
                "quality_complete",
                "teacher_reviewed",
            },
            name="feature row",
        )
        _strict_schema_version(payload.get("schema_version"), name="feature row")
        person_id = _clean_id(payload.get("person_id"), name="person_id")
        event_id = _clean_id(payload.get("event_id"), name="event_id")
        event_type = payload.get("event_type")
        if event_type not in ALLOWED_EVENT_TYPES:
            raise ValueError("unknown event type")
        raw_features = payload.get("features")
        if not isinstance(raw_features, list) or len(raw_features) != len(_FEATURE_NAMES):
            raise ValueError("feature vector shape mismatch")
        features = tuple(_finite_number(value, name="feature") for value in raw_features)
        quality_ok = payload.get("quality_ok")
        quality_complete = payload.get("quality_complete")
        teacher_reviewed = payload.get("teacher_reviewed")
        return cls(
            person_id,
            event_id,
            event_type,
            features,
            quality_ok,
            payload.get("quality_score"),
            quality_complete,
            teacher_reviewed,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "EventFeatureRow":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid feature row JSON") from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class TeacherLabel:
    """A human-approved support/outcome annotation; never a generated diagnosis."""

    event_id: str
    support_label: str
    outcome_label: str
    teacher_approved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _clean_id(self.event_id, name="event_id"))
        if self.support_label not in SUPPORT_LABELS:
            raise ValueError("unknown support label")
        if self.outcome_label not in OUTCOME_LABELS:
            raise ValueError("unknown outcome label")
        if not isinstance(self.teacher_approved, bool):
            raise ValueError("teacher_approved must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "support_label": self.support_label,
            "outcome_label": self.outcome_label,
            "teacher_approved": self.teacher_approved,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeacherLabel":
        if not isinstance(payload, Mapping):
            raise ValueError("label must be a mapping")
        _strict_keys(payload, {"schema_version", "event_id", "support_label", "outcome_label", "teacher_approved"}, name="label")
        _strict_schema_version(payload.get("schema_version"), name="label")
        return cls(
            event_id=payload.get("event_id"),
            support_label=payload.get("support_label"),
            outcome_label=payload.get("outcome_label"),
            teacher_approved=payload.get("teacher_approved", False),
        )
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "TeacherLabel":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid label JSON") from exc
        return cls.from_dict(payload)

def _clean_evidence_ids(value: Any, *, name: str = "evidence_ids") -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence")
    raw = tuple(value)
    cleaned = tuple(sorted({_clean_id(item, name="evidence_id") for item in raw}))
    if not cleaned:
        raise ValueError(f"{name} must be non-empty")
    if len(cleaned) != len(raw):
        raise ValueError(f"{name} must not contain duplicates")
    return cleaned
def _row_fingerprint(row: "EventFeatureRow") -> str:
    return hashlib.sha256(row.to_json().encode("utf-8")).hexdigest()


def _manifest_entries_payload(
    entries: Sequence[tuple[str, str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "event_id": event_id,
            "support_label": support_label,
            "row_fingerprint": row_fingerprint,
        }
        for event_id, support_label, row_fingerprint in entries
    ]


def _manifest_digest(
    entries: Sequence[tuple[str, str, str]],
    *,
    model_binding: Mapping[str, Any],
) -> str:
    payload = {"entries": _manifest_entries_payload(entries), "model_binding": model_binding}
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalize_manifest(value: Any) -> tuple[tuple[str, str, str], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("approved evidence manifest must be a sequence")
    normalized: list[tuple[str, str, str]] = []
    for item in value:
        if isinstance(item, Mapping):
            _strict_keys(item, {"event_id", "support_label", "row_fingerprint"}, name="manifest entry")
            event_id = item.get("event_id")
            support_label = item.get("support_label")
            fingerprint = item.get("row_fingerprint")
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            event_id, support_label, fingerprint = item
        else:
            raise ValueError("approved evidence manifest entry must be a mapping")
        event_id = _clean_id(event_id, name="event_id")
        if support_label not in SUPPORT_LABELS:
            raise ValueError("unknown manifest support label")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise ValueError("invalid manifest row fingerprint")
        normalized.append((event_id, support_label, fingerprint))
    if not normalized:
        raise ValueError("approved evidence manifest must be non-empty")
    if len({entry[0] for entry in normalized}) != len(normalized):
        raise ValueError("approved evidence manifest contains duplicate event_id values")
    if tuple(sorted(normalized, key=lambda item: item[0])) != tuple(normalized):
        raise ValueError("approved evidence manifest must be event-id sorted")
    return tuple(normalized)
def _validate_centroid_values(raw_values: Sequence[Any]) -> tuple[float, ...]:
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence) or len(raw_values) != len(_FEATURE_NAMES):
        raise ValueError("invalid model label centroid")
    values = tuple(_finite_number(value, name="centroid") for value in raw_values)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("centroid value outside bounds")
    event_count = len(ALLOWED_EVENT_TYPES)
    categorical_groups = (
        values[:event_count],
        values[event_count + 4 : event_count + 4 + len(ALLOWED_ZONE_TRANSITIONS)],
        values[event_count + 4 + len(ALLOWED_ZONE_TRANSITIONS) :],
    )
    for group in categorical_groups:
        if any(value not in (0.0, 1.0) for value in group) or sum(group) != 1.0:
            raise ValueError("invalid categorical centroid encoding")
    if not 0.0 <= values[event_count] <= 1.0:
        raise ValueError("invalid centroid duration")
    for index in range(event_count + 1, event_count + 4):
        if values[index] not in (0.0, 1.0):
            raise ValueError("invalid centroid quality encoding")
    if values[event_count + 1] != 1.0 or values[event_count + 2] != 1.0 or values[event_count + 3] != 1.0:
        raise ValueError("fit centroids require complete teacher-reviewed quality")
    return values


def _categorical_mode(rows: Sequence[EventFeatureRow], start: int, size: int) -> tuple[float, ...]:
    counts = [sum(row.features[start + index] for row in rows) for index in range(size)]
    best = max(range(size), key=lambda index: (counts[index], -index))
    return tuple(float(index == best) for index in range(size))
_AUTHORIZATION_SENTINEL = object()
_RECOMMENDATION_ISSUANCE_TOKEN = object()


@dataclass(frozen=True)
class _RecommendationAuthorization:
    event_id: str
    event_fingerprint: str
    approved_event_ids: tuple[str, ...]
    support_label: str
    manifest_digest: str
    issued: object = None


@dataclass(frozen=True)
class PersonalizationPrediction:
    support_label: str
    confidence: float
    sample_count: int
    evidence_ids: tuple[str, ...]
    _recommendation_authorization: _RecommendationAuthorization | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.support_label not in SUPPORT_LABELS:
            raise ValueError("unknown support label")
        confidence = _finite_number(self.confidence, name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 1:
            raise ValueError("sample_count must be a positive integer")
        evidence_ids = _clean_evidence_ids(self.evidence_ids)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_ids", evidence_ids)


@dataclass(frozen=True)
class Recommendation:
    support_label: str
    hint: str
    teacher_approved: bool
    evidence_ids: tuple[str, ...]
    provenance: str = "teacher_approved_personalization"
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RECOMMENDATION_ISSUANCE_TOKEN:
            raise ValueError("recommendation issuance is internal-only")
        if self.support_label not in SUPPORT_LABELS:
            raise ValueError("unknown support label")
        if not isinstance(self.hint, str) or not self.hint.strip():
            raise ValueError("hint must be a non-empty string")
        if self.teacher_approved is not True:
            raise ValueError("recommendations require explicit teacher approval")
        evidence_ids = _clean_evidence_ids(self.evidence_ids)
        if self.provenance != "teacher_approved_personalization":
            raise ValueError("unsupported recommendation provenance")
        object.__setattr__(self, "hint", self.hint.strip())
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "provenance", self.provenance.strip())


class InsufficientEvidenceError(ValueError):
    """Raised when validated inputs do not provide enough fitting evidence."""


@dataclass(frozen=True)
class PersonalizationModel:
    """Deterministic nearest-centroid baseline for one target person."""

    target_person_id: str
    sample_count: int
    min_samples: int
    min_confidence: float
    centroids: tuple[tuple[str, tuple[float, ...]], ...]
    label_counts: tuple[tuple[str, int], ...]
    feature_names: tuple[str, ...] = _FEATURE_NAMES
    schema_version: int = SCHEMA_VERSION
    model_config_version: str = MODEL_CONFIG_VERSION

    approved_event_ids: tuple[str, ...] = ()
    approved_evidence_manifest: tuple[tuple[str, str, str], ...] = ()
    approved_manifest_digest: str = ""
    def __post_init__(self) -> None:
        target = _clean_id(self.target_person_id, name="target_person_id")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("sample_count must be a positive integer")
        if (
            isinstance(self.min_samples, bool)
            or not isinstance(self.min_samples, int)
            or self.min_samples < 1
        ):
            raise ValueError("min_samples must be a positive integer")
        if self.sample_count < self.min_samples:
            raise ValueError("sample count is below minimum")
        min_confidence = _finite_number(self.min_confidence, name="min_confidence")
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if (
            type(self.schema_version) is not int
            or self.schema_version != SCHEMA_VERSION
            or self.model_config_version != MODEL_CONFIG_VERSION
        ):
            raise ValueError("unsupported model schema/config version")
        if not isinstance(self.feature_names, (tuple, list)) or tuple(self.feature_names) != _FEATURE_NAMES:
            raise ValueError("feature schema mismatch")
        if not isinstance(self.centroids, (tuple, list)) or not self.centroids:
            raise ValueError("model centroids must be non-empty")
        if not isinstance(self.label_counts, (tuple, list)) or not self.label_counts:
            raise ValueError("model label counts must be non-empty")

        centroid_values: list[tuple[str, tuple[float, ...]]] = []
        for item in self.centroids:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("invalid model label centroid")
            label, raw_values = item
            if label not in SUPPORT_LABELS:
                raise ValueError("unknown model support label")
            if any(existing_label == label for existing_label, _ in centroid_values):
                raise ValueError("duplicate model support label")
            if not isinstance(raw_values, (tuple, list)) or len(raw_values) != len(_FEATURE_NAMES):
                raise ValueError("invalid model label centroid")
            values = _validate_centroid_values(raw_values)
            centroid_values.append((label, values))

        count_values: list[tuple[str, int]] = []
        for item in self.label_counts:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("invalid model label count")
            label, count = item
            if label not in SUPPORT_LABELS:
                raise ValueError("unknown model support label")
            if any(existing_label == label for existing_label, _ in count_values):
                raise ValueError("duplicate model support label")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("invalid model label count")
            count_values.append((label, count))

        if {label for label, _ in centroid_values} != {label for label, _ in count_values}:
            raise ValueError("model centroids and label counts mismatch")
        if sum(count for _, count in count_values) != self.sample_count:
            raise ValueError("model sample count mismatch")
        manifest = _normalize_manifest(self.approved_evidence_manifest)
        if len(manifest) != self.sample_count:
            raise ValueError("model approved evidence count mismatch")
        approved_event_ids = _clean_evidence_ids(
            tuple(entry[0] for entry in manifest), name="approved_event_ids"
        )
        supplied_ids = _clean_evidence_ids(
            self.approved_event_ids, name="approved_event_ids"
        )
        if supplied_ids != approved_event_ids:
            raise ValueError("model approved evidence IDs do not match manifest")
        model_binding = {
            "schema_version": SCHEMA_VERSION,
            "model_config_version": MODEL_CONFIG_VERSION,
            "target_person_id": target,
            "sample_count": self.sample_count,
            "min_samples": self.min_samples,
            "min_confidence": min_confidence,
            "feature_names": list(_FEATURE_NAMES),
            "centroids": {label: list(values) for label, values in centroid_values},
            "label_counts": {label: count for label, count in count_values},
        }
        expected_digest = _manifest_digest(manifest, model_binding=model_binding)
        if (
            not isinstance(self.approved_manifest_digest, str)
            or self.approved_manifest_digest != expected_digest
        ):
            raise ValueError("model evidence manifest integrity check failed")

        object.__setattr__(self, "target_person_id", target)
        object.__setattr__(self, "min_confidence", min_confidence)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(self, "centroids", tuple(centroid_values))
        object.__setattr__(self, "label_counts", tuple(count_values))
        object.__setattr__(self, "approved_event_ids", approved_event_ids)
        object.__setattr__(self, "approved_evidence_manifest", manifest)
        object.__setattr__(self, "approved_manifest_digest", expected_digest)

    @classmethod
    def fit(
        cls,
        rows: Sequence[EventFeatureRow],
        labels: Sequence[TeacherLabel] | Mapping[str, TeacherLabel],
        *,
        target_person_id: str,
        min_samples: int = 3,
        min_confidence: float = 0.55,
    ) -> "PersonalizationModel":
        target = _clean_id(target_person_id, name="target_person_id")
        if isinstance(labels, Mapping):
            by_event = dict(labels)
            for key, label in by_event.items():
                if not isinstance(key, str) or not isinstance(label, TeacherLabel):
                    raise ValueError("labels mapping must contain TeacherLabel values")
                if key != label.event_id:
                    raise ValueError("label mapping key must equal label.event_id")
        else:
            label_values = tuple(labels)
            if any(not isinstance(label, TeacherLabel) for label in label_values):
                raise ValueError("labels must contain TeacherLabel values")
            label_ids = [label.event_id for label in label_values]
            if len(set(label_ids)) != len(label_ids):
                raise ValueError("labels contain duplicate event_id values")
            by_event = {label.event_id: label for label in label_values}
        row_values = tuple(rows)
        if any(not isinstance(row, EventFeatureRow) for row in row_values):
            raise ValueError("rows must contain EventFeatureRow values")
        event_ids = [row.event_id for row in row_values]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("rows contain duplicate event_id values")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
            or not 0.0 < min_confidence <= 1.0
            or not math.isfinite(min_confidence)
        ):
            raise ValueError("min_confidence must be between 0 and 1")
        selected: list[tuple[EventFeatureRow, TeacherLabel]] = []
        for row in sorted(row_values, key=lambda item: (item.person_id, item.event_id)):
            if row.person_id != target:
                continue
            label = by_event.get(row.event_id)
            if (
                label is not None
                and label.outcome_label in {"helpful", "effective"}
                and label.teacher_approved is True
                and row.quality_ok
                and row.quality_complete
                and row.teacher_reviewed
                and row.quality_score >= MIN_QUALITY_SCORE
            ):
                selected.append((row, label))
        if len(selected) < min_samples:
            raise InsufficientEvidenceError(f"at least {min_samples} teacher-approved samples are required")
        grouped: dict[str, list[EventFeatureRow]] = {}
        for row, label in selected:
            grouped.setdefault(label.support_label, []).append(row)
        centroids = []
        counts = []
        for support_label in sorted(grouped):
            members = grouped[support_label]
            centroid_values = [
                round(sum(row.features[i] for row in members) / len(members), 8)
                for i in range(len(_FEATURE_NAMES))
            ]
            event_count = len(ALLOWED_EVENT_TYPES)
            zone_start = event_count + 4
            context_start = zone_start + len(ALLOWED_ZONE_TRANSITIONS)
            centroid_values[:event_count] = _categorical_mode(members, 0, event_count)
            centroid_values[zone_start:context_start] = _categorical_mode(
                members, zone_start, len(ALLOWED_ZONE_TRANSITIONS)
            )
            centroid_values[context_start:] = _categorical_mode(
                members, context_start, len(ALLOWED_TEACHER_CONTEXTS)
            )
            centroid = _validate_centroid_values(centroid_values)
            centroids.append((support_label, centroid))
            counts.append((support_label, len(members)))
        manifest = tuple(
            sorted(
                (
                    row.event_id,
                    label.support_label,
                    _row_fingerprint(row),
                )
                for row, label in selected
            )
        )
        model_binding = {
            "schema_version": SCHEMA_VERSION,
            "model_config_version": MODEL_CONFIG_VERSION,
            "target_person_id": target,
            "sample_count": len(selected),
            "min_samples": min_samples,
            "min_confidence": float(min_confidence),
            "feature_names": list(_FEATURE_NAMES),
            "centroids": {label: list(values) for label, values in centroids},
            "label_counts": {label: count for label, count in counts},
        }
        return cls(
            target,
            len(selected),
            min_samples,
            min_confidence,
            tuple(centroids),
            tuple(counts),
            approved_event_ids=tuple(entry[0] for entry in manifest),
            approved_evidence_manifest=manifest,
            approved_manifest_digest=_manifest_digest(
                manifest, model_binding=model_binding
            ),
        )

    def predict(self, row: EventFeatureRow | Mapping[str, Any]) -> PersonalizationPrediction | None:
        try:
            self.__post_init__()
        except (TypeError, ValueError, AttributeError, KeyError, IndexError, RecursionError):
            return None
        if isinstance(row, Mapping):
            try:
                row = EventFeatureRow.from_summary(row)
            except (TypeError, ValueError):
                return None
        if not isinstance(row, EventFeatureRow):
            return None
        try:
            if row.person_id != self.target_person_id:
                return None
            row.__post_init__()
        except (TypeError, ValueError, AttributeError, KeyError, IndexError, RecursionError):
            return None
        if row.event_type not in ALLOWED_EVENT_TYPES or len(row.features) != len(_FEATURE_NAMES):
            return None
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in row.features):
            return None
        if (
            not isinstance(row.quality_ok, bool)
            or not isinstance(row.quality_complete, bool)
            or not isinstance(row.teacher_reviewed, bool)
            or not math.isfinite(row.quality_score)
        ):
            return None
        if not 0.0 <= row.quality_score <= 1.0:
            return None
        if (
            not row.quality_ok
            or not row.quality_complete
            or not row.teacher_reviewed
            or row.quality_score < MIN_QUALITY_SCORE
            or not self.centroids
        ):
            return None
        if any(len(centroid) != len(row.features) for _, centroid in self.centroids):
            return None
        distances = sorted(
            ((sum((a - b) ** 2 for a, b in zip(row.features, centroid)), label) for label, centroid in self.centroids),
            key=lambda item: (item[0], item[1]),
        )
        best_distance, best_label = distances[0]
        second_distance = distances[1][0] if len(distances) > 1 else best_distance + 1.0
        base = 1.0 / (1.0 + math.sqrt(best_distance))
        margin = max(0.0, second_distance - best_distance)
        confidence = round(base * (0.5 + 0.5 * margin / (margin + 1.0)), 6)
        if confidence < self.min_confidence:
            return None
        counts = dict(self.label_counts)
        row_fingerprint = _row_fingerprint(row)
        bound_entry = next(
            (entry for entry in self.approved_evidence_manifest if entry[0] == row.event_id),
            None,
        )
        if bound_entry is not None and (
            bound_entry[1] != best_label or bound_entry[2] != row_fingerprint
        ):
            return None
        label_manifest = tuple(
            entry for entry in self.approved_evidence_manifest if entry[1] == best_label
        )
        approved_entry = next(
            (entry for entry in label_manifest if entry[0] == row.event_id),
            None,
        )
        authorization = None
        if (
            approved_entry is not None
            and approved_entry[2] == row_fingerprint
        ):
            authorization = _RecommendationAuthorization(
                row.event_id,
                approved_entry[2],
                tuple(entry[0] for entry in label_manifest),
                best_label,
                self.approved_manifest_digest,
                _AUTHORIZATION_SENTINEL,
            )
        return PersonalizationPrediction(
            best_label,
            confidence,
            counts[best_label],
            (row.event_id,),
            authorization,
        )

    def to_dict(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version,
            "model_config_version": self.model_config_version,
            "target_person_id": self.target_person_id,
            "sample_count": self.sample_count,
            "min_samples": self.min_samples,
            "min_confidence": self.min_confidence,
            "feature_names": list(self.feature_names),
            "centroids": {label: list(values) for label, values in self.centroids},
            "label_counts": {label: count for label, count in self.label_counts},
            "approved_event_ids": list(self.approved_event_ids),
            "approved_evidence_manifest": _manifest_entries_payload(
                self.approved_evidence_manifest
            ),
            "approved_manifest_digest": self.approved_manifest_digest,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PersonalizationModel":
        required = {
            "schema_version",
            "model_config_version",
            "target_person_id",
            "sample_count",
            "min_samples",
            "min_confidence",
            "feature_names",
            "centroids",
            "label_counts",
            "approved_event_ids",
            "approved_evidence_manifest",
            "approved_manifest_digest",
        }
        if not isinstance(payload, Mapping):
            raise ValueError("model must be a mapping")
        _strict_keys(payload, required, name="model")
        _strict_schema_version(payload.get("schema_version"), name="model")
        if payload.get("model_config_version") != MODEL_CONFIG_VERSION:
            raise ValueError("unsupported model schema/config version")
        raw_names = payload.get("feature_names")
        if isinstance(raw_names, (str, bytes)) or not isinstance(raw_names, Sequence):
            raise ValueError("feature schema mismatch")
        names = tuple(raw_names)
        if names != _FEATURE_NAMES:
            raise ValueError("feature schema mismatch")
        target = _clean_id(payload.get("target_person_id"), name="target_person_id")
        sample_count = payload.get("sample_count")
        min_samples = payload.get("min_samples")
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 1
            or isinstance(min_samples, bool)
            or not isinstance(min_samples, int)
            or min_samples < 1
        ):
            raise ValueError("invalid model sample counts")
        if sample_count < min_samples:
            raise ValueError("model sample count is below minimum")
        min_confidence = _finite_number(payload.get("min_confidence"), name="min_confidence")
        if not 0 < min_confidence <= 1:
            raise ValueError("invalid model confidence threshold")
        raw_centroids = payload.get("centroids")
        raw_counts = payload.get("label_counts")
        if (
            not isinstance(raw_centroids, Mapping)
            or not isinstance(raw_counts, Mapping)
            or any(not isinstance(label, str) for label in raw_centroids)
            or any(not isinstance(label, str) for label in raw_counts)
            or set(raw_centroids) != set(raw_counts)
        ):
            raise ValueError("invalid model centroids")
        centroids: list[tuple[str, tuple[float, ...]]] = []
        counts: list[tuple[str, int]] = []
        for label in sorted(raw_centroids):
            if label not in SUPPORT_LABELS:
                raise ValueError("unknown model support label")
            values = _validate_centroid_values(raw_centroids[label])
            count = raw_counts[label]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("invalid model label count")
            centroids.append((label, values))
            counts.append((label, count))
        if sum(count for _, count in counts) != sample_count:
            raise ValueError("model sample count mismatch")
        approved_event_ids = _clean_evidence_ids(
            payload.get("approved_event_ids"), name="approved_event_ids"
        )
        manifest = _normalize_manifest(payload.get("approved_evidence_manifest"))
        if len(manifest) != sample_count:
            raise ValueError("model approved evidence count mismatch")
        if tuple(entry[0] for entry in manifest) != approved_event_ids:
            raise ValueError("model approved evidence IDs do not match manifest")
        manifest_digest = payload.get("approved_manifest_digest")
        if not isinstance(manifest_digest, str):
            raise ValueError("invalid model evidence manifest digest")
        return cls(
            target,
            sample_count,
            min_samples,
            min_confidence,
            tuple(centroids),
            tuple(counts),
            approved_event_ids=approved_event_ids,
            approved_evidence_manifest=manifest,
            approved_manifest_digest=manifest_digest,
        )

    @classmethod
    def from_json(cls, text: str) -> "PersonalizationModel":
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid model JSON") from exc
        return cls.from_dict(payload)


def grouped_person_split(
    rows: Sequence[EventFeatureRow], *, test_fraction: float = 0.2
) -> tuple[tuple[EventFeatureRow, ...], tuple[EventFeatureRow, ...]]:
    """Split rows deterministically by person, never splitting one person's rows."""
    if isinstance(test_fraction, bool) or not isinstance(test_fraction, (int, float)) or not 0.0 <= test_fraction <= 1.0 or not math.isfinite(test_fraction):
        raise ValueError("test_fraction must be between 0 and 1")
    row_values = tuple(rows)
    if any(not isinstance(row, EventFeatureRow) for row in row_values):
        raise ValueError("rows must contain EventFeatureRow values")
    event_ids = [row.event_id for row in row_values]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("rows contain duplicate event_id values")
    row_values = tuple(sorted(row_values, key=lambda row: (row.person_id, row.event_id)))
    people = sorted({row.person_id for row in row_values})
    if test_fraction == 0.0 or not people:
        return row_values, ()
    if len(people) == 1:
        raise ValueError("positive test_fraction requires at least two people")
    test_count = min(len(people) - 1, max(1, math.ceil(len(people) * float(test_fraction))))
    test_people = set(people[-test_count:])
    train = tuple(row for row in row_values if row.person_id not in test_people)
    test = tuple(row for row in row_values if row.person_id in test_people)
    return train, test


def split_person_events(
    rows: Sequence[EventFeatureRow], *, validation_fraction: float = 0.2
) -> tuple[tuple[EventFeatureRow, ...], tuple[EventFeatureRow, ...]]:
    """Deterministically split one person's rows without event-ID leakage."""
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0.0 < validation_fraction < 1.0
        or not math.isfinite(validation_fraction)
    ):
        raise ValueError("validation_fraction must be between 0 and 1")
    row_values = tuple(rows)
    if any(not isinstance(row, EventFeatureRow) for row in row_values):
        raise ValueError("rows must contain EventFeatureRow values")
    if len(row_values) < 2:
        raise ValueError("at least two rows are required")
    people = {row.person_id for row in row_values}
    if len(people) != 1:
        raise ValueError("rows must belong to one person")
    event_ids = [row.event_id for row in row_values]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("rows contain duplicate event_id values")
    ordered = tuple(sorted(row_values, key=lambda row: row.event_id))
    validation_count = min(
        len(ordered) - 1,
        max(1, math.ceil(len(ordered) * float(validation_fraction))),
    )
    return ordered[:-validation_count], ordered[-validation_count:]


grouped_person_split_within_target = split_person_events

split_by_person = grouped_person_split
fit_personalization_model = PersonalizationModel.fit
extract_event_features = EventFeatureRow.from_summary

_SUPPORT_HINTS = {
    "visual_schedule": "시각 순서표와 first-then 단서를 먼저 제시합니다.",
    "short_prompt": "한 번에 한 문장의 짧은 prompt를 사용합니다.",
    "transition_preview": "다음 단계와 종료 조건을 짧게 미리 예고합니다.",
    "brief_break": "필요할 때 짧고 예측 가능한 회복 시간을 제공합니다.",
    "reinforcement": "성공 직후 구체적인 칭찬 또는 승인된 강화 단서를 제공합니다.",
}


def _issue_recommendation(
    support_label: str,
    hint: str,
    evidence_ids: Sequence[str],
) -> Recommendation:
    return Recommendation(
        support_label,
        hint,
        True,
        tuple(evidence_ids),
        _issuance_token=_RECOMMENDATION_ISSUANCE_TOKEN,
    )


def generate_recommendations(
    support_label: str | PersonalizationPrediction | None,
    *,
    teacher_approved: bool,
    evidence_ids: Sequence[str] = (),
) -> tuple[Recommendation, ...]:
    """Map an approved support label to a bounded learning hint."""
    prediction: PersonalizationPrediction | None = None
    if not isinstance(support_label, PersonalizationPrediction):
        return ()
    try:
        support_label.__post_init__()
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("invalid personalization prediction") from exc
    prediction = support_label
    label = prediction.support_label
    if teacher_approved is not True or label not in _SUPPORT_HINTS:
        return ()
    authorization = prediction._recommendation_authorization
    if (
        not isinstance(authorization, _RecommendationAuthorization)
        or authorization.issued is not _AUTHORIZATION_SENTINEL
    ):
        return ()
    if authorization.support_label != label:
        return ()
    if authorization.event_id not in prediction.evidence_ids:
        return ()
    if (
        not isinstance(authorization.event_id, str)
        or not isinstance(authorization.event_fingerprint, str)
        or len(authorization.event_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in authorization.event_fingerprint)
        or not isinstance(authorization.manifest_digest, str)
        or len(authorization.manifest_digest) != 64
        or any(char not in "0123456789abcdef" for char in authorization.manifest_digest)
        or not authorization.approved_event_ids
    ):
        return ()
    if not set(prediction.evidence_ids).issubset(authorization.approved_event_ids):
        return ()
    if not evidence_ids:
        return ()
    evidence = _clean_evidence_ids(evidence_ids)
    if not set(prediction.evidence_ids).issubset(evidence):
        return ()
    if not set(evidence).issubset(authorization.approved_event_ids):
        return ()
    return (_issue_recommendation(label, _SUPPORT_HINTS[label], evidence),)


def serialize_model(model: PersonalizationModel) -> str:
    return model.to_json()


def deserialize_model(text: str) -> PersonalizationModel:
    return PersonalizationModel.from_json(text)


__all__ = [
    "ALLOWED_EVENT_TYPES", "ALLOWED_ZONE_TRANSITIONS", "ALLOWED_TEACHER_CONTEXTS",
    "SUPPORT_LABELS", "OUTCOME_LABELS", "EventFeatureRow", "TeacherLabel",
    "PersonalizationPrediction", "Recommendation", "InsufficientEvidenceError", "PersonalizationModel",
    "grouped_person_split", "split_by_person", "split_person_events",
    "grouped_person_split_within_target", "fit_personalization_model",
    "extract_event_features",
    "generate_recommendations", "serialize_model", "deserialize_model",
]
