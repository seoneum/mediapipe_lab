"""TDD boundary-table tests for the ON DAMM video metric fusion engine.

Written BEFORE app/ondamm_video_metrics.py exists (RED phase of todo 4 in
.omo/plans/ondamm-video-pipeline.md). All float comparisons go through
pytest.approx / math.isclose; inputs are deterministic, so no seeds needed.

The module under test must be PURE stdlib: no sklearn/torch/numpy imports,
no file/network access. SignalSample-shaped inputs are plain dicts here on
purpose -- that is also the manual-QA surface (see task-4 evidence snippet).
"""

from __future__ import annotations

import json
import math
import sys
import dataclasses
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_video_metrics import (  # noqa: E402
    ATTENTION_WINDOW_RATIO_MIN,
    CONE_PITCH_DOWN_DEG,
    CONE_PITCH_UP_DEG,
    CONE_YAW_MAX_DEG,
    EYES_OPEN_BLINK_MAX,
    EXPRESSION_MEDIAN_FILTER_K,
    GAZE_INWARD_DIFF_MAX,
    GAZE_VERTICAL_SUM_MAX,
    INTEREST_LOW_THRESHOLD,
    INTEREST_MEDIUM_THRESHOLD,
    PersonMetrics,
    attention_ratio,
    cumulative_focus_seconds,
    dominant_expression_timeline,
    eyes_open,
    gaze_forward,
    head_in_cone,
    interest_level,
    reference_pose,
    summarize_person,
)

EMOTION_LABELS = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]

FPS = 30.0


# --------------------------------------------------------------------------
# factories (plain dicts == frozen SignalSample schema)
# --------------------------------------------------------------------------

def make_sample(frame_idx=0, yaw=0.0, pitch=0.0, roll=0.0, blink=0.0,
                blendshapes=None, valence=0.0, arousal=0.0,
                emotion_probs=None, global_id="p1"):
    return {
        "frame_idx": frame_idx,
        "global_id": global_id,
        "blendshapes": dict(blendshapes or {}),
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "blink": blink,
        "emotion_labels": list(EMOTION_LABELS),
        "emotion_probs": list(emotion_probs or [0.125] * 8),
        "valence": valence,
        "arousal": arousal,
    }


def attentive(i, **kw):
    kw.setdefault("frame_idx", i)
    kw.setdefault("yaw", 0.0)
    kw.setdefault("pitch", 0.0)
    kw.setdefault("blink", 0.0)
    return make_sample(**kw)


def looking_away(i, yaw=60.0):
    """Fails ONLY the head-cone rule (eyes open, gaze forward)."""
    return make_sample(frame_idx=i, yaw=yaw)


def probs_for(dominant_index, dominant=0.9, rest=0.1 / 7):
    p = [rest] * 8
    p[dominant_index] = dominant
    return p


# --------------------------------------------------------------------------
# threshold constants named at module level (spec: exclusive boundaries)
# --------------------------------------------------------------------------

class TestSpecConstants:
    def test_interest_thresholds_match_spec(self):
        assert math.isclose(INTEREST_LOW_THRESHOLD, 0.33)
        assert math.isclose(INTEREST_MEDIUM_THRESHOLD, 0.66)

    def test_cone_and_gaze_constants_match_spec(self):
        assert math.isclose(CONE_YAW_MAX_DEG, 30.0)
        assert math.isclose(CONE_PITCH_DOWN_DEG, 25.0)
        assert math.isclose(CONE_PITCH_UP_DEG, 15.0)
        assert math.isclose(GAZE_INWARD_DIFF_MAX, 0.35)
        assert math.isclose(GAZE_VERTICAL_SUM_MAX, 0.5)
        assert math.isclose(EYES_OPEN_BLINK_MAX, 0.5)
        assert math.isclose(ATTENTION_WINDOW_RATIO_MIN, 0.5)
        assert EXPRESSION_MEDIAN_FILTER_K == 5


# --------------------------------------------------------------------------
# eyes_open: blink < 0.5 (strict)
# --------------------------------------------------------------------------

class TestEyesOpen:
    @pytest.mark.parametrize("blink,expected", [
        (0.0, True), (0.49, True), (0.4999999, True),
        (0.5, False), (0.51, False), (1.0, False),
    ])
    def test_strict_threshold(self, blink, expected):
        assert eyes_open(blink) is expected

    def test_nan_blink_is_not_open(self):
        # documented NaN guard: non-finite never counts as attentive
        assert eyes_open(float("nan")) is False


# --------------------------------------------------------------------------
# gaze_forward: |inwardL - inwardR| < 0.35 AND sum(eyeLookUp*) + sum(eyeLookDown*) < 0.5
# --------------------------------------------------------------------------

class TestGazeForward:
    def test_fully_forward_is_true(self):
        bs = {"eyeLookInwardLeft": 0.1, "eyeLookInwardRight": 0.1}
        assert gaze_forward(bs) is True

    @pytest.mark.parametrize("left,right,expected", [
        (0.3499999, 0.0, True),
        (0.35, 0.0, False),      # exact boundary excluded (strict <)
        (-0.35, 0.0, False),     # |diff| symmetric
        (0.3500001, 0.0, False),
    ])
    def test_inward_diff_boundary(self, left, right, expected):
        bs = {"eyeLookInwardLeft": left, "eyeLookInwardRight": right}
        assert gaze_forward(bs) is expected

    @pytest.mark.parametrize("up,down,expected", [
        (0.1249, 0.1249, True),   # four keys x 0.1249 -> sum 0.4996
        (0.125, 0.125, False),    # four keys x 0.125 -> exact 0.5 excluded
        (0.3, 0.3, False),
    ])
    def test_vertical_sum_boundary(self, up, down, expected):
        bs = {
            "eyeLookUpLeft": up, "eyeLookUpRight": up,
            "eyeLookDownLeft": down, "eyeLookDownRight": down,
        }
        assert gaze_forward(bs) is expected

    def test_missing_blendshape_keys_treated_as_zero(self):
        # KeyError is unacceptable: absent keys behave as 0.0
        assert gaze_forward({}) is True
        assert gaze_forward({"eyeLookInwardLeft": 0.9}) is False
        assert gaze_forward({"eyeLookUpLeft": 0.6}) is False

    def test_none_values_treated_as_zero(self):
        bs = {"eyeLookInwardLeft": None, "eyeLookInwardRight": None}
        assert gaze_forward(bs) is True

    def test_nan_value_is_not_forward(self):
        bs = {"eyeLookInwardLeft": float("nan"), "eyeLookInwardRight": 0.0}
        assert gaze_forward(bs) is False


# --------------------------------------------------------------------------
# head_in_cone: |dyaw| <= 30 AND pitch in [ref-25, ref+15]
# --------------------------------------------------------------------------

class TestHeadInCone:
    REF = (10.0, -5.0)

    @pytest.mark.parametrize("yaw,expected", [
        (10.0, True), (40.0, True), (-20.0, True),       # inclusive edges
        (40.0001, False), (-20.0001, False),
    ])
    def test_yaw_boundary_inclusive(self, yaw, expected):
        assert head_in_cone(yaw, -5.0, self.REF) is expected

    @pytest.mark.parametrize("pitch,expected", [
        (-30.0, True),   # ref - 25 (inclusive bottom edge)
        (+10.0, True),   # ref + 15 (inclusive top edge)
        (-30.0001, False),
        (+10.0001, False),
    ])
    def test_pitch_asymmetric_boundary(self, pitch, expected):
        assert head_in_cone(10.0, pitch, self.REF) is expected

    def test_yaw_inside_pitch_outside_rejected(self):
        assert head_in_cone(10.0, 45.0, self.REF) is False

    def test_pitch_inside_yaw_outside_rejected(self):
        assert head_in_cone(90.0, -5.0, self.REF) is False


# --------------------------------------------------------------------------
# reference_pose: per-ID median of FIRST 30 samples only
# --------------------------------------------------------------------------

class TestReferencePose:
    def test_median_of_first_30_ignores_later_drift(self):
        samples = [attentive(i, yaw=10.0) for i in range(30)]
        samples += [attentive(i, yaw=200.0) for i in range(30, 40)]
        ref_yaw, ref_pitch = reference_pose(samples)
        assert math.isclose(ref_yaw, 10.0)
        assert math.isclose(ref_pitch, 0.0)

    def test_exactly_31_samples_excludes_the_31st(self):
        samples = [attentive(i, yaw=0.0) for i in range(30)]
        samples.append(attentive(30, yaw=180.0))
        ref_yaw, _ = reference_pose(samples)
        assert math.isclose(ref_yaw, 0.0)

    def test_even_count_median_averages_middle_pair(self):
        samples = [attentive(i, yaw=(0.0 if i % 2 == 0 else 20.0)) for i in range(30)]
        ref_yaw, _ = reference_pose(samples)
        assert math.isclose(ref_yaw, 10.0)

    def test_empty_sequence_returns_zero_reference(self):
        assert reference_pose([]) == (0.0, 0.0)


# --------------------------------------------------------------------------
# attention_ratio(window): mean(head_in_cone AND eyes_open AND gaze_forward)
# --------------------------------------------------------------------------

class TestAttentionRatio:
    def test_fully_attentive_window_is_one(self):
        window = [attentive(i) for i in range(12)]
        assert attention_ratio(window, ref=(0.0, 0.0)) == pytest.approx(1.0)

    def test_half_looking_away_is_half(self):
        window = [attentive(i) for i in range(5)] + [looking_away(i) for i in range(5, 10)]
        assert attention_ratio(window, ref=(0.0, 0.0)) == pytest.approx(0.5)

    def test_blink_only_failure_counts_against_ratio(self):
        window = [attentive(i) for i in range(5)]
        window.append(attentive(5, blink=0.9))
        assert attention_ratio(window, ref=(0.0, 0.0)) == pytest.approx(5 / 6)

    def test_gaze_only_failure_counts_against_ratio(self):
        window = [attentive(i) for i in range(5)]
        window.append(attentive(5, blendshapes={"eyeLookUpLeft": 0.8}))
        assert attention_ratio(window, ref=(0.0, 0.0)) == pytest.approx(5 / 6)

    def test_empty_window_is_zero(self):
        assert attention_ratio([], ref=(0.0, 0.0)) == pytest.approx(0.0)

    def test_single_sample_window(self):
        assert attention_ratio([attentive(0)], ref=(0.0, 0.0)) == pytest.approx(1.0)
        assert attention_ratio([looking_away(0)], ref=(0.0, 0.0)) == pytest.approx(0.0)

    def test_ref_defaults_to_window_reference(self):
        window = [attentive(i, yaw=7.0, pitch=-2.0) for i in range(6)]
        assert attention_ratio(window) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# cumulative_focus_seconds: frame-union of windows with ratio > 0.5, dt = 1/fps
# --------------------------------------------------------------------------

class TestCumulativeFocusSeconds:
    def test_monotonic_non_decreasing_win5_hop1(self):
        stream = [attentive(i) for i in range(60)]
        values = [cumulative_focus_seconds(stream[:n], fps=FPS, win=5, hop=1)
                  for n in range(1, 61)]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
        assert values[-1] == pytest.approx(60 / FPS)

    def test_monotonic_on_mixed_stream_with_explicit_ref(self):
        stream = []
        for block in range(8):
            base = block * 5
            if block % 2 == 0:
                stream += [attentive(base + j) for j in range(5)]
            else:
                stream += [looking_away(base + j) for j in range(5)]
        values = [cumulative_focus_seconds(stream[:n], fps=FPS, win=5, hop=1, ref=(0.0, 0.0))
                  for n in range(1, len(stream) + 1)]
        assert all(b >= a - 1e-12 for a, b in zip(values, values[1:]))

    def test_all_distracted_is_zero(self):
        stream = [looking_away(i) for i in range(40)]
        assert cumulative_focus_seconds(stream, fps=FPS, win=5, hop=1,
                                        ref=(0.0, 0.0)) == pytest.approx(0.0)

    def test_empty_sequence_is_zero(self):
        assert cumulative_focus_seconds([], fps=FPS, win=5, hop=1) == pytest.approx(0.0)

    def test_single_sample_partial_window_counts_once(self):
        # n < win falls back to one partial window over what exists
        got = cumulative_focus_seconds([attentive(0)], fps=FPS, win=5, hop=1)
        assert got == pytest.approx(1 / FPS)

    def test_no_double_counting_across_overlapping_windows(self):
        # win=5 hop=1 overlapping windows over an attentive stream must not
        # inflate focus beyond wall-clock duration of the covered frames
        stream = [attentive(i) for i in range(30)]
        got = cumulative_focus_seconds(stream, fps=FPS, win=5, hop=1)
        assert got == pytest.approx(1.0)  # exactly 30 frames / 30 fps

    def test_windows_at_or_below_half_threshold_add_nothing(self):
        # every 5-sample slice of the periodic [A,A,D,D,D] pattern holds exactly
        # 2/5 attentive (< 0.5) -> no window qualifies -> zero focus seconds
        stream = []
        for i in range(15):
            stream.append(attentive(i) if i % 5 in (0, 1) else looking_away(i))
        got = cumulative_focus_seconds(stream, fps=FPS, win=5, hop=1, ref=(0.0, 0.0))
        assert got == pytest.approx(0.0)

    def test_window_ratio_exactly_half_is_excluded(self):
        # win=4 admits an exact 0.5 ratio; spec counts only ratio > 0.5
        stream = [attentive(0), attentive(1), looking_away(2), looking_away(3),
                  looking_away(4)]
        got = cumulative_focus_seconds(stream, fps=FPS, win=4, hop=1, ref=(0.0, 0.0))
        assert got == pytest.approx(0.0)


# --------------------------------------------------------------------------
# interest_level: score = 0.4*(1-jitter) + 0.35*max(valence,0) + 0.25*motion
#   score < 0.33 -> "낮음"; < 0.66 -> "중간"; else "높음" (exclusive bounds)
# --------------------------------------------------------------------------

class TestInterestLevel:
    def test_well_separated_bands(self):
        assert interest_level(0.0, 1.0, 1.0) == "높음"    # 0.75
        assert interest_level(0.5, 0.5, 0.5) == "중간"    # 0.2+0.175+0.125 = 0.5
        assert interest_level(1.0, -1.0, 0.0) == "낮음"   # 0.0

    def test_valence_clamped_at_zero(self):
        # clamped: 0.2 + 0 + 0.25 = 0.45 -> 중간; unclamped would be 0.10 -> 낮음
        assert interest_level(0.5, -1.0, 1.0) == "중간"

    def test_boundary_033_is_exclusive_low_side(self):
        # score(j) = 0.4*(1-j); d(score)/dj = -0.4 -> dj=2.5e-6 shifts 1e-6
        just_above = interest_level(0.175 - 2.5e-6, 0.0, 0.0)   # 0.33 + 1e-6
        just_below = interest_level(0.175 + 2.5e-6, 0.0, 0.0)   # 0.33 - 1e-6
        assert just_above == "중간"
        assert just_below == "낮음"

    def test_boundary_066_is_exclusive_with_high_inclusive(self):
        # base(j=0.4, v=0.6) = 0.24+0.21 = 0.45; m=0.84 adds 0.21 -> 0.66
        just_below = interest_level(0.4, 0.6, 0.84 - 4e-6)      # 0.66 - 1e-6
        just_above = interest_level(0.4, 0.6, 0.84 + 4e-6)      # 0.66 + 1e-6
        assert just_below == "중간"
        assert just_above == "높음"


# --------------------------------------------------------------------------
# dominant_expression_timeline: per-class median filter (k=5) then argmax
# --------------------------------------------------------------------------

class TestDominantExpressionTimeline:
    def test_median_filter_absorbs_single_frame_spike(self):
        probs = [probs_for(4) for _ in range(9)]        # happiness-dominant
        probs[4] = probs_for(7)                          # one surprise spike
        timeline = dominant_expression_timeline(probs, EMOTION_LABELS, k=5)
        assert len(timeline) == 9
        assert all(entry["label"] == "happiness" for entry in timeline)

    def test_k1_keeps_raw_argmax_spike(self):
        probs = [probs_for(4) for _ in range(9)]
        probs[4] = probs_for(7)
        timeline = dominant_expression_timeline(probs, EMOTION_LABELS, k=1)
        assert [e["label"] for e in timeline].count("surprise") == 1

    def test_sustained_switch_survives_smoothing(self):
        probs = [probs_for(4) for _ in range(10)] + [probs_for(6) for _ in range(5)]
        timeline = dominant_expression_timeline(probs, EMOTION_LABELS, k=5)
        labels = [e["label"] for e in timeline]
        assert labels[-1] == "sadness"
        assert labels.count("sadness") >= 3

    def test_t_sec_from_frame_indices_and_fps(self):
        probs = [probs_for(4) for _ in range(3)]
        timeline = dominant_expression_timeline(probs, EMOTION_LABELS,
                                                frame_indices=[0, 3, 6], fps=30.0)
        assert [e["t_sec"] for e in timeline] == [pytest.approx(0.0),
                                                  pytest.approx(0.1),
                                                  pytest.approx(0.2)]

    def test_default_k_is_five(self):
        probs = [probs_for(4) for _ in range(9)]
        probs[4] = probs_for(7)
        timeline = dominant_expression_timeline(probs, EMOTION_LABELS)
        assert all(entry["label"] == "happiness" for entry in timeline)

    def test_empty_probs_seq_returns_empty_timeline(self):
        assert dominant_expression_timeline([], EMOTION_LABELS) == []


# --------------------------------------------------------------------------
# summarize_person -> frozen PersonMetrics + to_json()/to_csv_row()
# --------------------------------------------------------------------------

SCHEMA_KEYS = {
    "global_id", "attention_pct", "focus_seconds", "interest",
    "expression_timeline", "frames_covered", "total_frames", "low_confidence",
}


def happy_stream(n=150, valence=0.8):
    return [make_sample(frame_idx=i, yaw=0.0, pitch=0.0, blink=0.0,
                        valence=valence, emotion_probs=probs_for(4))
            for i in range(n)]


class TestSummarizePerson:
    def test_returns_personmetrics_with_exact_schema_fields(self):
        pm = summarize_person("p1", happy_stream(20), total_frames=20, fps=FPS)
        assert isinstance(pm, PersonMetrics)
        assert {f.name for f in dataclasses.fields(PersonMetrics)} == SCHEMA_KEYS

    def test_personmetrics_is_frozen(self):
        pm = summarize_person("p1", happy_stream(20), total_frames=20, fps=FPS)
        with pytest.raises(dataclasses.FrozenInstanceError):
            pm.attention_pct = 42.0

    def test_attentive_stream_full_scores(self):
        pm = summarize_person("p1", happy_stream(150), total_frames=150, fps=FPS)
        assert pm.attention_pct == pytest.approx(100.0)
        assert pm.focus_seconds == pytest.approx(5.0)   # 150 frames / 30 fps
        assert pm.interest == "높음"                     # jitter 0, valence .8 -> 0.68
        assert pm.frames_covered == 150
        assert pm.total_frames == 150
        assert pm.low_confidence is False
        assert len(pm.expression_timeline) == 150
        assert all(e["label"] == "happiness" for e in pm.expression_timeline)

    def test_empty_sequence_zero_flags(self):
        pm = summarize_person("p1", [], total_frames=100, fps=FPS)
        assert pm.attention_pct == pytest.approx(0.0)
        assert pm.focus_seconds == pytest.approx(0.0)
        assert pm.frames_covered == 0
        assert pm.total_frames == 100
        assert pm.expression_timeline == []
        assert pm.interest == "낮음"

    def test_low_confidence_passthrough(self):
        pm = summarize_person("unknown_1", happy_stream(10), total_frames=10,
                              fps=FPS, low_confidence=True)
        assert pm.low_confidence is True
        assert pm.global_id == "unknown_1"

    def test_accepts_plain_dict_samples(self):
        # manual-QA surface parity: duck-typed SignalSample dicts
        pm = summarize_person("p1", happy_stream(6), total_frames=6, fps=FPS)
        assert pm.attention_pct == pytest.approx(100.0)

    def test_to_json_contains_all_schema_keys(self):
        pm = summarize_person("p1", happy_stream(12), total_frames=12, fps=FPS)
        payload = pm.to_json()
        assert set(payload.keys()) == SCHEMA_KEYS
        round_trip = json.loads(json.dumps(payload))
        assert round_trip == payload
        assert round_trip["interest"] == "높음"

    def test_to_csv_row_contains_all_schema_keys_as_strings(self):
        pm = summarize_person("p1", happy_stream(12), total_frames=12, fps=FPS)
        row = pm.to_csv_row()
        assert set(row.keys()) == SCHEMA_KEYS
        assert all(isinstance(v, str) for v in row.values())
        restored_timeline = json.loads(row["expression_timeline"])
        assert isinstance(restored_timeline, list)
        assert restored_timeline[0]["t_sec"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# purity guardrail: implementation stays stdlib-only, no I/O
# --------------------------------------------------------------------------

class TestPurityGuardrail:
    BANNED = ("sklearn", "torch", "numpy", "cv2", "requests",
              "urllib", "socket", "subprocess", "pathlib",
              "import os", "import io")

    def test_module_source_has_no_heavy_or_io_imports(self):
        source = (APP_DIR / "ondamm_video_metrics.py").read_text(encoding="utf-8")
        for token in self.BANNED:
            assert token not in source, f"banned token in pure module: {token}"
