from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


_FORBIDDEN_LABEL_PARTS = (
    "emotion",
    "happy",
    "sad",
    "angry",
    "fear",
    "attention",
    "concentration",
    "preference",
    "diagnos",
    "autism",
    "asd",
    "compliance",
)
_ALLOWED_AGGREGATIONS = {"mean", "max", "min"}

# MediaPipe Face Blendshapes follows the ARKit-style category names. Keep this
# allowlist explicit so dossier-provided profiles cannot turn arbitrary fields
# into a learned runtime feature.
ALLOWED_BLENDSHAPE_NAMES = frozenset(
    {
        "_neutral",
        "browDownLeft",
        "browDownRight",
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
        "jawForward",
        "jawLeft",
        "jawOpen",
        "jawRight",
        "mouthClose",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthFunnel",
        "mouthLeft",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthPucker",
        "mouthRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "noseSneerLeft",
        "noseSneerRight",
        "tongueOut",
    }
)


@dataclass(frozen=True)
class MovementRule:
    label: str
    display_name: str
    blendshape_names: tuple[str, ...]
    aggregation: str
    activation_threshold: float
    priority: int = 100

    def __post_init__(self) -> None:
        label = self.label.strip() if isinstance(self.label, str) else ""
        display_name = self.display_name.strip() if isinstance(self.display_name, str) else ""
        if not label or not display_name:
            raise ValueError("movement rule label and display_name are required")
        lowered = f"{label} {display_name}".lower()
        if any(part in lowered for part in _FORBIDDEN_LABEL_PARTS):
            raise ValueError("movement rules must describe observable movement, not emotion, diagnosis, or preference")
        if self.aggregation not in _ALLOWED_AGGREGATIONS:
            raise ValueError("movement rule aggregation must be mean, max, or min")
        names = tuple(self.blendshape_names)
        if not names or len(names) > 8 or len(set(names)) != len(names):
            raise ValueError("movement rule requires 1 to 8 unique blendshape names")
        unknown = sorted(set(names) - ALLOWED_BLENDSHAPE_NAMES)
        if unknown:
            raise ValueError(f"unsupported blendshape names: {unknown}")
        if isinstance(self.activation_threshold, bool) or not isinstance(self.activation_threshold, (int, float)):
            raise ValueError("activation_threshold must be numeric")
        threshold = float(self.activation_threshold)
        if not 0.05 <= threshold <= 0.95:
            raise ValueError("activation_threshold must be between 0.05 and 0.95")
        if not isinstance(self.priority, int) or not 0 <= self.priority <= 1000:
            raise ValueError("priority must be an integer between 0 and 1000")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "blendshape_names", names)
        object.__setattr__(self, "activation_threshold", threshold)

    def score(self, scores: Mapping[str, float]) -> float:
        values = [float(scores.get(name, 0.0)) for name in self.blendshape_names]
        if self.aggregation == "max":
            return max(values)
        if self.aggregation == "min":
            return min(values)
        return sum(values) / len(values)


DEFAULT_MOVEMENT_RULES: tuple[MovementRule, ...] = (
    MovementRule("eyes_closed", "양눈 닫힘 움직임", ("eyeBlinkLeft", "eyeBlinkRight"), "min", 0.25, 0),
    MovementRule("left_eye_closed", "왼눈 닫힘 움직임", ("eyeBlinkLeft",), "max", 0.35, 5),
    MovementRule("right_eye_closed", "오른눈 닫힘 움직임", ("eyeBlinkRight",), "max", 0.35, 5),
    MovementRule("mouth_smile", "입꼬리 상승 움직임", ("mouthSmileLeft", "mouthSmileRight"), "mean", 0.35, 20),
    MovementRule("jaw_open", "턱 열림 움직임", ("jawOpen",), "max", 0.4, 20),
    MovementRule("brow_raise", "눈썹 올림 움직임", ("browInnerUp", "browOuterUpLeft", "browOuterUpRight"), "max", 0.4, 30),
    MovementRule("brow_lower", "눈썹 내림 움직임", ("browDownLeft", "browDownRight"), "mean", 0.35, 30),
    MovementRule("eye_squint", "눈 가늘게 뜸 움직임", ("eyeSquintLeft", "eyeSquintRight"), "mean", 0.4, 30),
    MovementRule("eyes_wide", "눈 크게 뜸 움직임", ("eyeWideLeft", "eyeWideRight"), "mean", 0.4, 30),
    MovementRule("mouth_frown", "입꼬리 하강 움직임", ("mouthFrownLeft", "mouthFrownRight"), "mean", 0.35, 40),
    MovementRule("lip_pucker", "입술 오므림 움직임", ("mouthPucker",), "max", 0.4, 40),
    MovementRule("lip_press", "입술 누름 움직임", ("mouthPressLeft", "mouthPressRight"), "mean", 0.4, 40),
    MovementRule("mouth_stretch", "입 늘림 움직임", ("mouthStretchLeft", "mouthStretchRight"), "mean", 0.4, 40),
    MovementRule("mouth_dimple", "입꼬리 당김 움직임", ("mouthDimpleLeft", "mouthDimpleRight"), "mean", 0.4, 40),
    MovementRule("cheek_puff", "볼 부풀림 움직임", ("cheekPuff",), "max", 0.4, 50),
    MovementRule("nose_sneer", "코 주변 올림 움직임", ("noseSneerLeft", "noseSneerRight"), "max", 0.4, 50),
    MovementRule("mouth_left", "입 왼쪽 이동", ("mouthLeft",), "max", 0.45, 60),
    MovementRule("mouth_right", "입 오른쪽 이동", ("mouthRight",), "max", 0.45, 60),
    MovementRule("tongue_out", "혀 내밈 움직임", ("tongueOut",), "max", 0.5, 60),
)


@dataclass(frozen=True)
class FacialMovementAnalysis:
    primary_label: str
    active_labels: tuple[str, ...]
    movement_scores: dict[str, float]
    rule_display_names: dict[str, str]
    eye_closure_state: str
    eye_blink_left: float
    eye_blink_right: float
    top_blendshapes: tuple[tuple[str, float], ...]
    notice: str = (
        "얼굴 blendshape 기반 관찰 가능한 움직임 proxy이며 감정, 집중도, 선호도, 진단으로 해석하지 않습니다."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_label": self.primary_label,
            "active_labels": list(self.active_labels),
            "movement_scores": dict(self.movement_scores),
            "rule_display_names": dict(self.rule_display_names),
            "eye_closure_state": self.eye_closure_state,
            "eye_blink_left": self.eye_blink_left,
            "eye_blink_right": self.eye_blink_right,
            "top_blendshapes": [[name, score] for name, score in self.top_blendshapes],
            "notice": self.notice,
        }


def _blendshape_scores(face_blendshapes: Iterable[Any] | Mapping[str, float] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(face_blendshapes, Mapping):
        for name, score in face_blendshapes.items():
            if name in ALLOWED_BLENDSHAPE_NAMES and isinstance(score, (int, float)) and not isinstance(score, bool):
                result[name] = max(0.0, min(1.0, float(score)))
        return result
    for category in face_blendshapes or ():
        name = getattr(category, "category_name", None)
        score = getattr(category, "score", None)
        if name in ALLOWED_BLENDSHAPE_NAMES and isinstance(score, (int, float)) and not isinstance(score, bool):
            result[name] = max(0.0, min(1.0, float(score)))
    return result


def merge_rules(
    defaults: Sequence[MovementRule] = DEFAULT_MOVEMENT_RULES,
    overrides: Sequence[MovementRule] = (),
) -> tuple[MovementRule, ...]:
    by_label = {rule.label: rule for rule in defaults}
    for rule in overrides:
        by_label[rule.label] = rule
    return tuple(sorted(by_label.values(), key=lambda rule: (rule.priority, rule.label)))


def rules_from_approved_profiles(profiles: Iterable[Mapping[str, Any] | Any]) -> tuple[MovementRule, ...]:
    rules: list[MovementRule] = []
    for profile in profiles:
        to_dict = getattr(profile, "to_dict", None)
        if callable(to_dict):
            profile = to_dict()
        if not isinstance(profile, Mapping):
            raise ValueError("facial movement profile must be a mapping")
        if profile.get("status") != "approved" or not str(profile.get("approved_by", "")).strip():
            raise ValueError("facial movement profile must be explicitly approved")
        source_ids = profile.get("source_session_ids", [])
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError("approved facial movement profile requires source session IDs")
        rules.append(
            MovementRule(
                label=str(profile.get("label", "")),
                display_name=str(profile.get("display_name", "")),
                blendshape_names=tuple(profile.get("blendshape_names", ())),
                aggregation=str(profile.get("aggregation", "")),
                activation_threshold=float(profile.get("activation_threshold", -1.0)),
                priority=int(profile.get("priority", 80)),
            )
        )
    return merge_rules(overrides=rules)


def analyze_facial_movements(
    face_blendshapes: Iterable[Any] | Mapping[str, float] | None,
    *,
    rules: Sequence[MovementRule] = DEFAULT_MOVEMENT_RULES,
    top_limit: int = 10,
) -> FacialMovementAnalysis:
    scores = _blendshape_scores(face_blendshapes)
    left = scores.get("eyeBlinkLeft", 0.0)
    right = scores.get("eyeBlinkRight", 0.0)
    rule_by_label = {rule.label: rule for rule in rules}
    both_threshold = rule_by_label.get("eyes_closed", DEFAULT_MOVEMENT_RULES[0]).activation_threshold
    left_threshold = rule_by_label.get("left_eye_closed", DEFAULT_MOVEMENT_RULES[1]).activation_threshold
    right_threshold = rule_by_label.get("right_eye_closed", DEFAULT_MOVEMENT_RULES[2]).activation_threshold
    if min(left, right) >= both_threshold:
        eye_state = "both_closed"
    elif left >= left_threshold and right < both_threshold:
        eye_state = "left_closed"
    elif right >= right_threshold and left < both_threshold:
        eye_state = "right_closed"
    else:
        eye_state = "open_or_uncertain"

    movement_scores = {rule.label: round(rule.score(scores), 6) for rule in rules}
    active_rules = [rule for rule in rules if movement_scores[rule.label] >= rule.activation_threshold]
    active_rules.sort(key=lambda rule: (rule.priority, -movement_scores[rule.label], rule.label))
    active_labels_list = [rule.label for rule in active_rules]

    if eye_state == "both_closed":
        primary = "eyes_closed"
    elif eye_state == "left_closed":
        primary = "left_eye_closed"
    elif eye_state == "right_closed":
        primary = "right_eye_closed"
    elif active_rules:
        primary = max(active_rules, key=lambda rule: (movement_scores[rule.label], -rule.priority)).label
    else:
        primary = "neutral"

    # Bilateral closure supersedes the two unilateral component labels in the
    # human-facing list while preserving their numeric scores for diagnostics.
    if eye_state == "both_closed":
        active_labels_list = [
            label for label in active_labels_list if label not in {"left_eye_closed", "right_eye_closed"}
        ]
        if "eyes_closed" not in active_labels_list:
            active_labels_list.insert(0, "eyes_closed")
    active_labels = tuple(active_labels_list)

    top = tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[: max(1, top_limit)])
    return FacialMovementAnalysis(
        primary_label=primary,
        active_labels=active_labels,
        movement_scores=movement_scores,
        rule_display_names={rule.label: rule.display_name for rule in rules},
        eye_closure_state=eye_state,
        eye_blink_left=round(left, 6),
        eye_blink_right=round(right, 6),
        top_blendshapes=top,
    )
