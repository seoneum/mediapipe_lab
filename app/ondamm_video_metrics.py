"""Rule-based engagement metric fusion over SignalSample sequences.

Pure standard-library functions only: no third-party imports, no file or
network access, deterministic float math. Inputs are duck-typed against the
frozen cross-module SignalSample schema (attribute objects or plain dicts).

Non-diagnostic framing: every metric here is a behavioral proxy estimated
from tracked face signals. It is NOT a medical, psychological, or
educational diagnosis of any person, and output labels must never be
presented as one.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

# --- attention proxy rules -------------------------------------------------
EYES_OPEN_BLINK_MAX = 0.5        # blink score strictly below this -> eyes open
GAZE_INWARD_DIFF_MAX = 0.35      # |eyeLookInwardL - eyeLookInwardR| strict bound
GAZE_VERTICAL_SUM_MAX = 0.5      # sum of eyeLookUp* + eyeLookDown* strict bound
CONE_YAW_MAX_DEG = 30.0          # inclusive half-width around reference yaw
CONE_PITCH_DOWN_DEG = 25.0       # inclusive pitch drop below reference
CONE_PITCH_UP_DEG = 15.0         # inclusive pitch rise above reference
ATTENTION_WINDOW_RATIO_MIN = 0.5  # window counts as focused only if ratio > this
REFERENCE_WARMUP_SAMPLES = 30    # per-ID reference pose uses the first N samples

# --- interest score --------------------------------------------------------
# Behavioral-proxy heuristic combining head-pose stability, non-negative
# valence, and motion energy. Thresholds are presentation bands, not clinical
# or diagnostic cut-offs.
INTEREST_WEIGHT_STABILITY = 0.4
INTEREST_WEIGHT_VALENCE = 0.35
INTEREST_WEIGHT_MOTION = 0.25
INTEREST_LOW_THRESHOLD = 0.33    # score < this -> "낮음" (exclusive)
INTEREST_MEDIUM_THRESHOLD = 0.66  # score < this -> "중간", otherwise "높음"
INTEREST_LOW_LABEL = "낮음"
INTEREST_MEDIUM_LABEL = "중간"
INTEREST_HIGH_LABEL = "높음"

EXPRESSION_MEDIAN_FILTER_K = 5   # default median-filter window for timelines
FOCUS_WINDOW_DEFAULT = 5
FOCUS_HOP_DEFAULT = 1

_VERTICAL_LOOK_KEYS = (
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeLookDownLeft", "eyeLookDownRight",
)


def _num(value: Any) -> float:
    """Coerce to float; missing/None/unconvertible values count as 0.0."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _safe(value: float) -> float:
    """Non-finite floats degrade to 0.0 instead of poisoning comparisons."""
    v = _num(value)
    return v if _finite(v) else 0.0


def _field(sample: Any, key: str, default: Any = None) -> Any:
    """Read a frozen-schema field from an object or a mapping."""
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


def eyes_open(blink: float) -> bool:
    b = _num(blink)
    return _finite(b) and b < EYES_OPEN_BLINK_MAX


def gaze_forward(blendshapes: Mapping[str, float] | None) -> bool:
    bs = blendshapes if blendshapes is not None else {}
    left = _num(bs.get("eyeLookInwardLeft"))
    right = _num(bs.get("eyeLookInwardRight"))
    vertical = sum(_num(bs.get(key)) for key in _VERTICAL_LOOK_KEYS)
    inward_diff = abs(left - right)
    if not (_finite(inward_diff) and _finite(vertical)):
        return False
    return inward_diff < GAZE_INWARD_DIFF_MAX and vertical < GAZE_VERTICAL_SUM_MAX


def head_in_cone(yaw_deg: float, pitch_deg: float,
                 ref: tuple[float, float]) -> bool:
    ref_yaw, ref_pitch = ref
    dyaw = _num(yaw_deg) - _num(ref_yaw)
    pitch = _num(pitch_deg)
    lo = _num(ref_pitch) - CONE_PITCH_DOWN_DEG
    hi = _num(ref_pitch) + CONE_PITCH_UP_DEG
    if not (_finite(dyaw) and _finite(pitch)):
        return False
    return abs(dyaw) <= CONE_YAW_MAX_DEG and lo <= pitch <= hi


def reference_pose(samples: Sequence[Any]) -> tuple[float, float]:
    """Per-ID reference head pose: median of the first N samples."""
    warmup = list(samples[:REFERENCE_WARMUP_SAMPLES])
    if not warmup:
        return (0.0, 0.0)
    yaws = [_safe(_field(s, "yaw_deg")) for s in warmup]
    pitches = [_safe(_field(s, "pitch_deg")) for s in warmup]
    return (statistics.median(yaws), statistics.median(pitches))


def is_attentive(sample: Any, ref: tuple[float, float]) -> bool:
    in_cone = head_in_cone(_field(sample, "yaw_deg"),
                           _field(sample, "pitch_deg"), ref)
    return (in_cone
            and eyes_open(_field(sample, "blink"))
            and gaze_forward(_field(sample, "blendshapes")))


def attention_ratio(window: Sequence[Any],
                    ref: tuple[float, float] | None = None) -> float:
    if not window:
        return 0.0
    r = reference_pose(window) if ref is None else ref
    hits = sum(1 for s in window if is_attentive(s, r))
    return hits / len(window)


def cumulative_focus_seconds(samples: Sequence[Any], fps: float,
                             win: int = FOCUS_WINDOW_DEFAULT,
                             hop: int = FOCUS_HOP_DEFAULT,
                             ref: tuple[float, float] | None = None) -> float:
    """Focus time as the frame-union of sliding windows with ratio > threshold.

    Each window [start, start+win) stepping by hop marks its frames when its
    attention ratio exceeds ATTENTION_WINDOW_RATIO_MIN; marked frames are
    counted once each at dt=1/fps, so overlapping windows never double-count
    and appending samples can never decrease the total. Sequences shorter
    than win fall back to one partial window over what exists.
    """
    n = len(samples)
    fps_safe = _safe(fps)
    if n == 0 or fps_safe <= 0.0:
        return 0.0
    r = reference_pose(samples) if ref is None else ref
    step = max(1, int(hop))
    size = max(1, int(win))
    focused = [False] * n
    if n < size:
        if attention_ratio(samples, r) > ATTENTION_WINDOW_RATIO_MIN:
            focused = [True] * n
    else:
        for start in range(0, n - size + 1, step):
            if attention_ratio(samples[start:start + size], r) > ATTENTION_WINDOW_RATIO_MIN:
                for j in range(start, start + size):
                    focused[j] = True
    return sum(1 for flag in focused if flag) / fps_safe


def interest_level(pose_jitter_norm: float, valence: float,
                   motion_energy_norm: float) -> str:
    jitter = min(1.0, max(0.0, _safe(pose_jitter_norm)))
    motion = min(1.0, max(0.0, _safe(motion_energy_norm)))
    positive_valence = max(_safe(valence), 0.0)
    score = (INTEREST_WEIGHT_STABILITY * (1.0 - jitter)
             + INTEREST_WEIGHT_VALENCE * positive_valence
             + INTEREST_WEIGHT_MOTION * motion)
    if score < INTEREST_LOW_THRESHOLD:
        return INTEREST_LOW_LABEL
    if score < INTEREST_MEDIUM_THRESHOLD:
        return INTEREST_MEDIUM_LABEL
    return INTEREST_HIGH_LABEL


def dominant_expression_timeline(
    probs_seq: Sequence[Sequence[float]],
    emotion_labels: Sequence[str],
    frame_indices: Sequence[int] | None = None,
    fps: float = 30.0,
    k: int = EXPRESSION_MEDIAN_FILTER_K,
) -> list[dict[str, Any]]:
    """Median-filter each emotion class over a k-window, then argmax per frame."""
    n = len(probs_seq)
    if n == 0:
        return []
    size = max(1, int(k))
    half = size // 2
    idxs = [int(i) for i in frame_indices] if frame_indices is not None else list(range(n))
    fps_safe = _safe(fps) or 30.0
    cleaned = [[_safe(p) for p in row] for row in probs_seq]
    timeline: list[dict[str, Any]] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        columns = list(zip(*cleaned[lo:hi]))
        medians = [statistics.median(col) for col in columns]
        best = max(range(len(medians)), key=medians.__getitem__)
        label = emotion_labels[best] if best < len(emotion_labels) else str(best)
        timeline.append({"t_sec": round(idxs[i] / fps_safe, 3), "label": label})
    return timeline


def _pose_jitter_norm(samples: Sequence[Any]) -> float:
    if len(samples) < 2:
        return 0.0
    deltas = []
    for prev, curr in zip(samples, samples[1:]):
        dyaw = _safe(_field(curr, "yaw_deg")) - _safe(_field(prev, "yaw_deg"))
        dpitch = _safe(_field(curr, "pitch_deg")) - _safe(_field(prev, "pitch_deg"))
        deltas.append(math.hypot(dyaw, dpitch))
    mean_delta = statistics.fmean(deltas)
    return min(1.0, max(0.0, mean_delta / 90.0))


def _motion_energy_norm(samples: Sequence[Any]) -> float:
    if len(samples) < 2:
        return 0.0
    diffs = []
    for prev, curr in zip(samples, samples[1:]):
        prev_bs = _field(prev, "blendshapes") or {}
        curr_bs = _field(curr, "blendshapes") or {}
        keys = set(prev_bs) | set(curr_bs)
        if keys:
            l1 = sum(abs(_num(curr_bs.get(key)) - _num(prev_bs.get(key)))
                     for key in keys)
            diffs.append(l1 / len(keys))
        else:
            diffs.append(0.0)
    mean_per_key = statistics.fmean(diffs)
    return min(1.0, mean_per_key * 10.0)


@dataclasses.dataclass(frozen=True)
class PersonMetrics:
    """Frozen per-person summary matching the cross-module schema contract."""

    global_id: str
    attention_pct: float
    focus_seconds: float
    interest: str
    expression_timeline: list[dict[str, Any]]
    frames_covered: int
    total_frames: int
    low_confidence: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "attention_pct": float(self.attention_pct),
            "focus_seconds": float(self.focus_seconds),
            "interest": self.interest,
            "expression_timeline": [dict(entry) for entry in self.expression_timeline],
            "frames_covered": int(self.frames_covered),
            "total_frames": int(self.total_frames),
            "low_confidence": bool(self.low_confidence),
        }

    def to_csv_row(self) -> dict[str, str]:
        row: dict[str, str] = {}
        for key, value in self.to_json().items():
            if isinstance(value, str):
                row[key] = value
            elif isinstance(value, list):
                row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                row[key] = str(value)
        return row


def summarize_person(global_id: str, samples: Iterable[Any], total_frames: int,
                     fps: float, low_confidence: bool = False) -> PersonMetrics:
    """Fuse one person's SignalSample sequence into frozen PersonMetrics."""
    collected = list(samples)
    covered = len(collected)
    if covered == 0:
        return PersonMetrics(
            global_id=global_id,
            attention_pct=0.0,
            focus_seconds=0.0,
            interest=INTEREST_LOW_LABEL,
            expression_timeline=[],
            frames_covered=0,
            total_frames=int(total_frames),
            low_confidence=bool(low_confidence),
        )
    ref = reference_pose(collected)
    attention_pct = attention_ratio(collected, ref) * 100.0
    focus = cumulative_focus_seconds(collected, fps, win=FOCUS_WINDOW_DEFAULT,
                                    hop=FOCUS_HOP_DEFAULT, ref=ref)
    labels = list(_field(collected[0], "emotion_labels") or [])
    probs_seq = [list(_field(s, "emotion_probs") or []) for s in collected]
    frame_indices = [int(_safe(_field(s, "frame_idx"))) for s in collected]
    timeline = dominant_expression_timeline(probs_seq, labels,
                                            frame_indices=frame_indices,
                                            fps=fps)
    valences = [_safe(_field(s, "valence")) for s in collected]
    mean_valence = statistics.fmean(valences)
    interest = interest_level(_pose_jitter_norm(collected), mean_valence,
                              _motion_energy_norm(collected))
    return PersonMetrics(
        global_id=global_id,
        attention_pct=attention_pct,
        focus_seconds=focus,
        interest=interest,
        expression_timeline=timeline,
        frames_covered=covered,
        total_frames=int(total_frames),
        low_confidence=bool(low_confidence),
    )
