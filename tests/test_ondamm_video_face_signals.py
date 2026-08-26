"""ON DAMM 얼굴 신호 추출(app/ondamm_video_face_signals.py) hermetic 테스트.

원칙:
- 기본 실행은 네트워크/실제 모델 로드 없이 래퍼 로직만 검증(monkeypatch fixture).
- 오일러 분해는 항등행렬/단축 회전/합성 90°급 행렬로 exact-value 검증.
- 실체 모델 경로는 smoke 마크 + ONDAMM_SMOKE_CLIP 게이트, 체크포인트 교차검증은
  캐시 부재 시 skip-with-reason(가짜 성공 금지).
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ondamm_video_face_signals as fs
from ondamm_video_face_signals import (
    BLENDSHAPE_NAMES,
    EMOTION_LABELS_8,
    CROP_SCALE,
    EmotionSignals,
    FaceSignalError,
    FaceSignalExtractor,
    MediaPipeSignals,
    elementary_rotation_matrix,
    ensure_strictly_monotonic,
    normalize_blendshapes,
    padded_crop,
    resolve_default_device,
    rotation_matrix_to_euler_degrees,
    signals_from_result,
    validate_rgb_frame,
)

TOL = 1e-6


# ------------------------------------------------------------------ fixtures


class StubRecognizer:
    """EmotiEffLibRecognizerTorch와 동일한 표면적을 가진 결정론적 스텁."""

    def __init__(self, seed: int = 7, wrong_slice: bool = False, with_bias: bool = True):
        rng = np.random.default_rng(seed)
        self.classifier_weights = rng.standard_normal((10, 1280)).astype(np.float32)
        self.classifier_bias = rng.standard_normal(10).astype(np.float32) if with_bias else None
        self.idx_to_emotion_class = {i: name for i, name in enumerate(EMOTION_LABELS_8)}
        self._wrong_slice = wrong_slice
        self._seed = seed

    def extract_features(self, crops):
        rng = np.random.default_rng(self._seed + len(crops))
        return rng.standard_normal((len(crops), 1280)).astype(np.float32)

    def predict_emotions(self, crops, logits: bool = True):
        features = self.extract_features(crops)
        scores = features @ self.classifier_weights.T
        if self.classifier_bias is not None:
            scores = scores + self.classifier_bias
        window = scores[:, -2:] if self._wrong_slice else scores[:, :-2]  # 라이브러리 MTL 관례
        preds = np.argmax(window, axis=1)
        return [self.idx_to_emotion_class[int(p)] for p in preds], scores


class FakeLandmarker:
    """MediaPipeSignals.detect 와 동일한 반환 계약의 스텁."""

    def __init__(self):
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def detect(self, frame, ts_ms_monotonic):
        self.calls.append((tuple(frame.shape), int(ts_ms_monotonic)))
        return {
            "blendshapes": normalize_blendshapes(
                [("eyeBlinkLeft", 0.9), ("eyeBlinkRight", 0.2), ("jawOpen", 0.1)]
            ),
            "yaw_deg": 4.2,
            "pitch_deg": -1.5,
            "roll_deg": 0.7,
            "blink": 0.9,
        }

    def close(self) -> None:
        pass


class FakeEmotions:
    def predict(self, crops):
        assert len(crops) == 1 and crops[0].ndim == 3
        probs = [1.0 / 8.0] * 8
        return {
            "labels": list(EMOTION_LABELS_8),
            "probs": probs,
            "valence": 0.25,
            "arousal": -0.5,
        }


def _bare_mediapipe_signals(categories, matrix):
    """__init__(실제 모델 로드) 없이 detect 래퍼 로직만 검증하는 최소 인스턴스."""
    obj = MediaPipeSignals.__new__(MediaPipeSignals)
    obj._last_ts_ms = None
    obj._infer = lambda frame, ts: _FakeResult(categories, matrix)
    return obj


class _FakeCategory:
    def __init__(self, category_name, score):
        self.category_name = category_name
        self.score = score


class _FakeResult:
    def __init__(self, categories, matrix):
        self.face_blendshapes = [categories] if categories is not None else None
        self.facial_transformation_matrixes = [matrix] if matrix is not None else None


@pytest.fixture()
def rgb_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)


# ------------------------------------------------------------------- euler


def test_euler_identity_matrix_is_zero():
    yaw, pitch, roll = rotation_matrix_to_euler_degrees(np.eye(4))
    assert abs(yaw) < TOL and abs(pitch) < TOL and abs(roll) < TOL


def test_euler_known_yaw_matrix():
    # Ry(30°): r20=-sin30 → yaw=atan2(sin30, cos30)=30°, pitch=roll=0
    mat = np.eye(4)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    mat[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    yaw, pitch, roll = rotation_matrix_to_euler_degrees(mat)
    assert math.isclose(yaw, 30.0, abs_tol=TOL)
    assert abs(pitch) < TOL and abs(roll) < TOL


def test_euler_composite_90ish_rotation_roundtrip():
    # Rz(-25°)·Ry(90°)·Rx(15°) 를 구성 규약대로 만들면 분해가 정확히 되돌려야 한다.
    expected = (90.0, 15.0, -25.0)
    mat = elementary_rotation_matrix(*expected)
    got = rotation_matrix_to_euler_degrees(mat)
    for g, e in zip(got, expected):
        assert math.isclose(g, e, abs_tol=1e-6)


def test_euler_rejects_bad_matrix_shape():
    with pytest.raises(FaceSignalError, match="shape"):
        rotation_matrix_to_euler_degrees(np.zeros(3))
    with pytest.raises(FaceSignalError, match="shape"):
        rotation_matrix_to_euler_degrees([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(FaceSignalError, match="non-finite"):
        rotation_matrix_to_euler_degrees(np.full((4, 4), np.nan))


# ------------------------------------------------------------- blendshapes


def test_blendshape_dict_has_52_keys_and_blink_max():
    raw = [
        _FakeCategory("eyeBlinkLeft", 0.9),
        _FakeCategory("eyeBlinkRight", 0.2),
        _FakeCategory("_neutral", 0.01),
        _FakeCategory("mouthSmileLeft", 0.55),
    ]
    signals = signals_from_result(raw, np.eye(4))
    blendshapes = signals["blendshapes"]
    assert len(blendshapes) == 52
    assert set(blendshapes) == set(BLENDSHAPE_NAMES)
    assert all(isinstance(v, float) for v in blendshapes.values())
    assert signals["blink"] == pytest.approx(max(0.9, 0.2))
    assert blendshapes["mouthSmileLeft"] == pytest.approx(0.55)
    assert blendshapes["jawOpen"] == 0.0  # 누락 범주는 0.0


def test_missing_blendshape_categories_tolerated_with_zero():
    signals = signals_from_result([("eyeBlinkRight", 0.7)], None)
    assert len(signals["blendshapes"]) == 52
    assert signals["blendshapes"]["eyeBlinkLeft"] == 0.0
    assert signals["blendshapes"]["eyeBlinkRight"] == pytest.approx(0.7)
    assert signals["blink"] == pytest.approx(0.7)  # max(0.0, 0.7)
    # 변환 행렬 부재는 머리포즈 0.0 강등(허용된 우아한 강등).
    assert (signals["yaw_deg"], signals["pitch_deg"], signals["roll_deg"]) == (0.0, 0.0, 0.0)


def test_unknown_extra_categories_dropped_and_no_face_returns_none():
    signals = signals_from_result([("notARealBlendshape", 0.9)], None)
    assert len(signals["blendshapes"]) == 52
    assert "notARealBlendshape" not in signals["blendshapes"]
    assert signals_from_result(None, None) is None


# ---------------------------------------------------------------- VA slice


def test_va_slice_shapes_and_ranges():
    emotions = EmotionSignals(device="cpu", recognizer=StubRecognizer(seed=11))
    crops = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(2)]
    out = emotions.predict(crops)
    assert len(out["labels"]) == 8
    assert out["labels"] == list(EMOTION_LABELS_8)
    assert len(out["probs"]) == 8
    assert all(0.0 <= p <= 1.0 for p in out["probs"])
    assert sum(out["probs"]) == pytest.approx(1.0, abs=1e-9)
    assert isinstance(out["valence"], float) and math.isfinite(out["valence"])
    assert isinstance(out["arousal"], float) and math.isfinite(out["arousal"])
    assert -1.0 <= out["valence"] <= 1.0
    assert -1.0 <= out["arousal"] <= 1.0


def test_va_ordering_cross_check_hermetic_match_and_mismatch_detection():
    ok_rec = StubRecognizer(seed=3)
    emotions_ok = EmotionSignals(device="cpu", recognizer=ok_rec)
    crop = np.zeros((24, 24, 3), dtype=np.uint8)
    verdict = emotions_ok.verify_va_column_order(crop)
    assert verdict["ok"] is True
    assert verdict["our_slice_label"] == verdict["library_label"]

    bad_rec = StubRecognizer(seed=3, wrong_slice=True)
    emotions_bad = EmotionSignals(device="cpu", recognizer=bad_rec)
    verdict_bad = emotions_bad.verify_va_column_order(crop)
    # 의도적으로 틀린 슬라이싱 비교 창은 정직하게 mismatch로 보고되어야 한다.
    assert verdict_bad["our_slice_label"] != verdict_bad["library_label"]
    assert "MISMATCH" in verdict_bad["detail"]


# ------------------------------------------------------- timestamp / input


def test_timestamp_monotonicity_violation_raises(rgb_frame):
    raw = [("eyeBlinkLeft", 0.5)]
    signals = _bare_mediapipe_signals(raw, np.eye(4))
    assert signals.detect(rgb_frame, 100) is not None
    with pytest.raises(FaceSignalError, match="strictly monotonic"):
        signals.detect(rgb_frame, 100)  # 동일 타임스탬프 금지
    assert signals.detect(rgb_frame, 200) is not None
    with pytest.raises(FaceSignalError, match="strictly monotonic"):
        signals.detect(rgb_frame, 150)  # 역행 금지
    with pytest.raises(FaceSignalError, match="strictly monotonic"):
        ensure_strictly_monotonic(500, 500)


def test_detect_rejects_non_rgb_input():
    with pytest.raises(FaceSignalError, match="HxWx3"):
        validate_rgb_frame(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(FaceSignalError, match="uint8"):
        validate_rgb_frame(np.zeros((10, 10, 3), dtype=np.float32))
    with pytest.raises(FaceSignalError, match="dtype"):
        padded_crop(np.zeros((10, 10, 3), dtype=np.float64), [0, 0, 5, 5])


# ------------------------------------------------------------- model path


def test_bad_model_path_raises_actionable_error():
    with pytest.raises(FaceSignalError) as excinfo:
        MediaPipeSignals(model_path="/bad/path.task")
    message = str(excinfo.value)
    assert "scripts/download_video_models.sh" in message
    assert "/bad/path.task" in message


# --------------------------------------------------------------- extractor


def test_extractor_happy_path_signal_sample_json_roundtrip(rgb_frame):
    extractor = FaceSignalExtractor(
        device="cpu", sample_every=3, landmarker=FakeLandmarker(), emotions=FakeEmotions()
    )
    sample = extractor.extract(3, "person_a", [20, 20, 40, 40], rgb_frame, ts_ms_monotonic=33)
    assert sample is not None
    assert set(sample.keys()) == {
        "frame_idx", "global_id", "blendshapes", "yaw_deg", "pitch_deg", "roll_deg",
        "blink", "emotion_labels", "emotion_probs", "valence", "arousal",
    }
    assert sample["frame_idx"] == 3 and sample["global_id"] == "person_a"
    assert len(sample["blendshapes"]) == 52
    assert len(sample["emotion_labels"]) == 8 and len(sample["emotion_probs"]) == 8
    assert -1.0 <= sample["valence"] <= 1.0 and -1.0 <= sample["arousal"] <= 1.0
    # JSON 직렬화 round-trip 동일성(frozen schema 직렬화 계약).
    restored = json.loads(json.dumps(sample))
    assert restored == sample


def test_extractor_sample_every_gate_and_no_face(rgb_frame):
    class NoFaceLandmarker(FakeLandmarker):
        def detect(self, frame, ts_ms_monotonic):
            return None

    extractor = FaceSignalExtractor(
        device="cpu", sample_every=3, landmarker=FakeLandmarker(), emotions=FakeEmotions()
    )
    assert extractor.extract(4, "p", [20, 20, 40, 40], rgb_frame, 10) is None  # 게이트 미통과
    assert extractor.extract(6, "p", [20, 20, 40, 40], rgb_frame, 20) is not None

    none_extractor = FaceSignalExtractor(
        device="cpu", sample_every=1, landmarker=NoFaceLandmarker(), emotions=FakeEmotions()
    )
    assert none_extractor.extract(1, "p", [20, 20, 40, 40], rgb_frame, 5) is None


def test_padded_crop_scales_and_clips(rgb_frame):
    crop = padded_crop(rgb_frame, [30, 20, 50, 40])  # 중심 (40,30), 1.5x → 25..55 / 15..45
    assert crop.shape == (30, 30, 3)
    edge = padded_crop(rgb_frame, [0, 0, 10, 10])  # 경계 클리핑
    assert edge.shape[0] >= 10 and edge.shape[1] >= 10
    assert edge.shape[0] <= rgb_frame.shape[0] and edge.shape[1] <= rgb_frame.shape[1]
    with pytest.raises(FaceSignalError, match="degenerate bbox"):
        padded_crop(rgb_frame, [10, 10, 10, 20])
    with pytest.raises(FaceSignalError, match="4 elements"):
        padded_crop(rgb_frame, [1, 2, 3])
    assert CROP_SCALE == 1.5  # frozen contract


# ------------------------------------------------------------ device resolve


def test_resolve_default_device_prefers_env_report(tmp_path, monkeypatch):
    report = tmp_path / "env_report.json"
    report.write_text(json.dumps({"default_device": "mps"}), encoding="utf-8")
    assert resolve_default_device(report) == "mps"


def test_resolve_default_device_falls_back_gracefully(tmp_path, monkeypatch):
    garbage = tmp_path / "env_report.json"
    garbage.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        fs.ondamm_video_env, "run_check", lambda report_path=None: ({"default_device": "cuda"}, 0)
    )
    assert resolve_default_device(garbage) == "cuda"

    monkeypatch.setattr(
        fs.ondamm_video_env,
        "run_check",
        lambda report_path=None: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    missing = tmp_path / "missing" / "env_report.json"
    assert resolve_default_device(missing) == "cpu"  # 안전값


# ---------------------------------------------------------------- fp32 rule


def test_module_never_calls_half_precision():
    source = Path(fs.__file__).read_text(encoding="utf-8")
    assert not re.search(r"\.half\s*\(", source), "반정밀(half) 호출 금지 위반"
    assert "float16" not in source, "float16 리터럴 금지 위반"
    assert "torch.half" not in source


# ----------------------------------------------------- real-model (gated)


@pytest.mark.smoke
def test_real_face_blendshapes_smoke():
    clip = __import__("os").environ.get("ONDAMM_SMOKE_CLIP")
    if not clip:
        pytest.skip("ONDAMM_SMOKE_CLIP 미설정 — 실제 영상 smoke은 opt-in")
    import cv2

    capture = cv2.VideoCapture(clip)
    assert capture.isOpened(), f"cannot open smoke clip: {clip}"
    ok, frame_bgr = capture.read()
    capture.release()
    assert ok, "smoke clip first frame read failed"
    frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])

    landmarker_path = ROOT / "models" / "face_landmarker.task"
    if not landmarker_path.is_file():
        pytest.skip(f"landmarker bundle missing — run bash scripts/download_video_models.sh ({landmarker_path})")

    signals = MediaPipeSignals(landmarker_path)
    try:
        first = signals.detect(frame_rgb, 1)
        second = signals.detect(frame_rgb, 2)
    finally:
        signals.close()
    observed = first or second  # 첫 프레임 워밍업 미검출 허용, 두 타임스탬프 시도
    assert observed is not None, "no face detected in ONDAMM_SMOKE_CLIP first frames"
    assert len(observed["blendshapes"]) == 52
    assert any(v > 0.0 for v in observed["blendshapes"].values()), "blendshapes empty (all zero)"


def test_va_cross_check_real_checkpoint():
    checkpoint = Path.home() / ".emotiefflib" / "enet_b0_8_va_mtl.pt"
    if not checkpoint.is_file():
        pytest.skip(
            f"EmotiEffLib checkpoint not cached at {checkpoint} — "
            "run scripts/download_video_models.sh first; skipping instead of fake-pass"
        )
    device = resolve_default_device()
    emotions = EmotionSignals(device=device)
    rng = np.random.default_rng(2026)
    crop = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    verdict = emotions.verify_va_column_order(crop)
    assert verdict["ok"] is True, (
        f"VA column ordering mismatch against real checkpoint: {verdict} — adapt slicing!"
    )
