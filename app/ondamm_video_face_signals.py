"""ON DAMM 영상 분석기 얼굴 신호 추출 (todo 3).

MediaPipe FaceLandmarker(52 blendshapes + 4x4 변환 행렬)와 EmotiEffLib
``enet_b0_8_va_mtl``(8-class 표정 + valence/arousal)를 감싸, 실행 전략의
frozen schema contract에 정확히 맞는 ``SignalSample`` 호환 dict를 만든다.

SignalSample dict 계약 (변경 금지 — todos 2~5가 함께 사용):
    frame_idx:int, global_id:str,
    blendshapes:dict[str,float]  # 정확히 52키 (누락 범주는 0.0으로 채움)
    yaw_deg:float, pitch_deg:float, roll_deg:float,
    blink:float,                 # max(eyeBlinkLeft, eyeBlinkRight)
    emotion_labels:list[str]     # 8개 (probs와 순서 병렬)
    emotion_probs:list[float]    # 8개, softmax 합 ≈ 1
    valence:float                # [-1,1]
    arousal:float                # [-1,1]

오일러 분해 (4x4 facial transformation matrix → yaw/pitch/roll):
회전 블록 R = Rz(roll) · Ry(yaw) · Rx(pitch) (Tait-Bryan ZYX 관례)로 두면
    r20 = -sin(yaw)
    r21 = cos(yaw)·sin(pitch),  r22 = cos(yaw)·cos(pitch)
    r10 = sin(roll)·cos(yaw),   r00 = cos(roll)·cos(yaw)
이므로 표준 분해식은:
    yaw   = atan2(-r20, sqrt(r00² + r10²))
    pitch = atan2(r21, r22)
    roll  = atan2(r10, r00)
yaw=±90°(짐벌 록)에서는 pitch/roll이 수치적으로 유일하지 않지만 위 식은
여전히 정의되며 연속적으로 동작한다(MediaPipe 출력에서 실질적으로 발생하지
않는 극단 자세까지 포함해 안전하다). 각도는 도(degree)로 반환한다.

VA 열 순서: EmotiEffLib 라이브러리 자체가 MTL 모델의 감정 점수를
``scores[:, :-2]``로 슬라이스하므로(logit 앞 8열 = 감정, 마지막 2열 =
valence, arousal) 우리 슬라이스 ``[:, :8] / [:, 8] / [:, 9]``과 일치한다.
추가로 :meth:`EmotionSignals.verify_va_column_order` 가 실제 체크포인트에서
``predict_emotions`` argmax와 교차검증한다(결과는 증거 파일에 기록).

fp32 강제: 어떤 텐서/배열에도 반정밀(half) 변환을 호출하지 않는다. MPS fp16 NaN
회귀를 막기 위한 프로젝트 규칙이다(소스 내 반정밀 호출 부재를 테스트가 검사한다).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "app"))

import ondamm_video_env  # noqa: E402  (repo convention: flat import via app/ on sys.path)

DEFAULT_LANDMARKER_PATH = "models/face_landmarker.task"
DOWNLOAD_SCRIPT_HINT = "scripts/download_video_models.sh"
EMOTIEFF_MODEL_NAME = "enet_b0_8_va_mtl"
CROP_SCALE = 1.5
NUM_BLENDSHAPES = 52
NUM_EMOTIONS = 8

# MediaPipe face_blendshapes_graph.cc의 표준 52개 범주 이름(순서 불문, 키 집합이 계약).
BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)

# enet_b0_8_va_mtl(idx_to_emotion_class)의 8-class 라벨 순서.
EMOTION_LABELS_8: tuple[str, ...] = (
    "Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise",
)

VALID_DEVICES = ("cpu", "mps", "cuda")
SIGNAL_SAMPLE_KEYS = (
    "frame_idx", "global_id", "blendshapes", "yaw_deg", "pitch_deg", "roll_deg",
    "blink", "emotion_labels", "emotion_probs", "valence", "arousal",
)


class FaceSignalError(RuntimeError):
    """얼굴 신호 추출 단계의 실패(모델 부재/잘못된 입력 등)를 알린다."""


# --------------------------------------------------------------------- euler


def rotation_matrix_to_euler_degrees(matrix: Any) -> tuple[float, float, float]:
    """4x4(또는 3x3) 변환 행렬을 (yaw_deg, pitch_deg, roll_deg)로 분해한다.

    R = Rz(roll)·Ry(yaw)·Rx(pitch) 관례의 표준 식(모듈 docstring 참고)을 그대로
    구현했다. 잘못된 모양/비유한 값은 명확한 FaceSignalError로 거절한다.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape not in ((4, 4), (3, 3)):
        raise FaceSignalError(
            f"facial transformation matrix must be 4x4 (or 3x3), got shape {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise FaceSignalError("facial transformation matrix contains non-finite values")
    r00, r01, _r02 = arr[0, 0], arr[0, 1], arr[0, 2]
    r10, _r11, _r12 = arr[1, 0], arr[1, 1], arr[1, 2]
    r20, r21, r22 = arr[2, 0], arr[2, 1], arr[2, 2]

    yaw = math.degrees(math.atan2(-r20, math.sqrt(r00 * r00 + r10 * r10)))
    pitch = math.degrees(math.atan2(r21, r22))
    roll = math.degrees(math.atan2(r10, r00))
    return float(yaw), float(pitch), float(roll)


def elementary_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """테스트/검증용: Rz(roll)·Ry(yaw)·Rx(pitch) 4x4 행렬을 생성한다."""
    y, p, r = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rot = np.array(
        [
            [cr * cy, cr * sy * sp - sr * cp, cr * sy * cp + sr * sp],
            [sr * cy, sr * sy * sp + cr * cp, sr * sy * cp - cr * sp],
            [-sy, cy * sp, cy * cp],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    return mat


# ---------------------------------------------------------------- blendshapes


def normalize_blendshapes(raw_categories: Iterable[Any]) -> dict[str, float]:
    """mediapipe Category 목록((name, score) 튜플도 허용)을 정확히 52키 dict로.

    계약상 누락 범주는 0.0으로 채우고(허용), 알 수 없는 여분 범주는 버린다.
    """
    raw: dict[str, float] = {}
    for cat in raw_categories:
        name, score = _category_name_score(cat)
        raw[name] = float(score)
    out = {name: 0.0 for name in BLENDSHAPE_NAMES}
    for name, score in raw.items():
        if name in out:
            out[name] = score
    return out


def _category_name_score(cat: Any) -> tuple[str, float]:
    if isinstance(cat, (tuple, list)) and len(cat) == 2:
        return str(cat[0]), float(cat[1])
    name = getattr(cat, "category_name", None)
    score = getattr(cat, "score", None)
    if name is None or score is None:
        raise FaceSignalError(
            f"blendshape category-like object expected (category_name/score or (name, score)), got {type(cat).__name__}"
        )
    return str(name), float(score)


def signals_from_result(blendshape_categories: Iterable[Any] | None, matrix: Any | None) -> dict:
    """랜더마커 1-face 결과를 공통 신호 dict로 변환한다(순수 함수 — hermetic 테스트 대상).

    blendshape가 None이면(얼굴 없음) None을 반환한다. 변환 행렬이 없으면
    머리포즈는 0.0으로 강등한다(허용된 우아한 강등).
    """
    if blendshape_categories is None:
        return None
    blendshapes = normalize_blendshapes(blendshape_categories)
    blink = max(blendshapes["eyeBlinkLeft"], blendshapes["eyeBlinkRight"])
    if matrix is not None:
        yaw, pitch, roll = rotation_matrix_to_euler_degrees(np.asarray(matrix))
    else:
        yaw = pitch = roll = 0.0
    return {
        "blendshapes": blendshapes,
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "blink": float(blink),
    }


def ensure_strictly_monotonic(last_ts_ms: int | None, ts_ms: int) -> int:
    """랜더마커 VIDEO 모드 계약: 타임스탬프는 인스턴스당 엄격 단조 증가."""
    ts = int(ts_ms)
    if last_ts_ms is not None and ts <= last_ts_ms:
        raise FaceSignalError(
            f"landmarker timestamps must be strictly monotonically increasing per instance: "
            f"got {ts} after {last_ts_ms}"
        )
    return ts


def validate_rgb_frame(frame: Any) -> np.ndarray:
    """uint8 HxWx3 RGB 프레임 검증. 잘못된 입력은 명확한 오류로 거절한다."""
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise FaceSignalError(f"expected HxWx3 RGB frame, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise FaceSignalError(
            f"expected uint8 RGB frame, got dtype {arr.dtype} (convert before calling detect)"
        )
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------- mediapipe


class MediaPipeSignals:
    """FaceLandmarker VIDEO 모드 래퍼: blendshapes 52 + yaw/pitch/roll + blink."""

    def __init__(self, model_path: str | Path = DEFAULT_LANDMARKER_PATH) -> None:
        path = Path(model_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file() or path.stat().st_size == 0:
            raise FaceSignalError(
                f"face landmarker model not found at '{path}' — run "
                f"`bash {DOWNLOAD_SCRIPT_HINT}` from the repo root first"
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - 환경 손상 시에만
            raise FaceSignalError(
                f"mediapipe tasks API unavailable ({type(exc).__name__}: {exc}); "
                f"see requirements.txt and {DOWNLOAD_SCRIPT_HINT}"
            ) from exc

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        try:
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise FaceSignalError(
                f"failed to create FaceLandmarker from '{path}' ({exc}); the bundle may be "
                f"corrupt — re-run `bash {DOWNLOAD_SCRIPT_HINT}`"
            ) from exc
        self._last_ts_ms: int | None = None

    def _infer(self, rgb_frame: np.ndarray, ts_ms: int):
        """실제 mediapipe 호출(테스트에서는 인스턴스 속성으로 대체된다)."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_image(mp_image, ts_ms)

    def detect(self, rgb_frame_or_crop: Any, ts_ms_monotonic: int) -> dict | None:
        """RGB 프레임/크롭 1장에서 얼굴 신호를 추출한다. 얼굴 없으면 None."""
        frame = validate_rgb_frame(rgb_frame_or_crop)
        ts = ensure_strictly_monotonic(self._last_ts_ms, ts_ms_monotonic)
        self._last_ts_ms = ts
        result = self._infer(frame, ts)
        categories = result.face_blendshapes[0] if result.face_blendshapes else None
        matrices = getattr(result, "facial_transformation_matrixes", None)
        matrix = matrices[0] if matrices else None
        return signals_from_result(categories, matrix)

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()


# ---------------------------------------------------------------- emotions


def resolve_default_device(report_path: str | Path | None = None) -> str:
    """todo 1의 env 모듈 경유로 기본 디바이스를 결정한다(import 상수 맹신 금지).

    1) outputs/ondamm/video/env_report.json 의 default_device (--check 산출물)
    2) 없으면 ondamm_video_env.run_check() 로 즉시 검사 후 판정
    3) 그마저 실패하면 안전값 "cpu"
    """
    path = Path(report_path) if report_path is not None else ondamm_video_env.REPORT_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            device = json.load(fh).get("default_device")
        if device in VALID_DEVICES:
            return str(device)
    except Exception:
        pass
    try:
        report, _code = ondamm_video_env.run_check()
        device = report.get("default_device")
        if device in VALID_DEVICES:
            return str(device)
    except Exception:
        pass
    return "cpu"


def softmax(logits: np.ndarray) -> np.ndarray:
    """행별 수치안정 softmax(fp64 내부 계산 — 반정밀 변환은 어디에도 없음)."""
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _create_recognizer(device: str):
    """EmotiEffLib 인식기 팩토리(테스트에서 monkeypatch 대상)."""
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    return EmotiEffLibRecognizer(engine="torch", model_name=EMOTIEFF_MODEL_NAME, device=device)


class EmotionSignals:
    """EmotiEffLib enet_b0_8_va_mtl 래퍼: 8-class probs + valence/arousal."""

    def __init__(self, device: str | None = None, recognizer: Any | None = None) -> None:
        self.device = device if device is not None else resolve_default_device()
        self._rec = recognizer if recognizer is not None else _create_recognizer(self.device)

    def predict(self, rgb_crops: Sequence[np.ndarray]) -> dict:
        """RGB 크롭 목록 → labels(8)/probs(8)/valence/arousal.

        logits = features @ classifier_weights.T (+ bias, 속성이 존재하면).
        열 순서는 [감정 8 | valence | arousal] (모듈 docstring의 검증 근거 참고).
        N>1 크롭이면 logit 평균 후 슬라이스한다. VA 선형 출력은 tanh로 [-1,1]에
        묶어 frozen schema 범위를 보장한다(AffectNet VA 레이블 범위와 동일).
        """
        crops = list(rgb_crops)
        if not crops:
            raise FaceSignalError("predict requires at least one RGB crop")
        features = np.asarray(self._rec.extract_features(crops))
        if features.ndim != 2:
            raise FaceSignalError(f"expected 2D feature matrix, got shape {features.shape}")
        weights = np.asarray(self._rec.classifier_weights)
        if weights.ndim != 2 or weights.shape[1] != features.shape[1]:
            raise FaceSignalError(
                f"classifier_weights shape {weights.shape} incompatible with features {features.shape}"
            )
        logits = features @ weights.T
        bias = getattr(self._rec, "classifier_bias", None)
        if bias is not None:
            logits = logits + np.asarray(bias)
        if logits.shape[1] != NUM_EMOTIONS + 2:
            raise FaceSignalError(
                f"expected {NUM_EMOTIONS + 2} logit columns (8 emotions + VA), got {logits.shape[1]}"
            )

        mean_logits = logits.reshape(-1, logits.shape[1]).mean(axis=0, keepdims=True)
        probs = softmax(mean_logits[:, :NUM_EMOTIONS])[0]
        valence = float(np.tanh(mean_logits[0, NUM_EMOTIONS]))
        arousal = float(np.tanh(mean_logits[0, NUM_EMOTIONS + 1]))

        idx_map = getattr(self._rec, "idx_to_emotion_class", None)
        if isinstance(idx_map, dict) and len(idx_map) == NUM_EMOTIONS:
            labels = [str(idx_map[i]) for i in range(NUM_EMOTIONS)]
        else:
            labels = list(EMOTION_LABELS_8)
        return {
            "labels": labels,
            "probs": [float(p) for p in probs],
            "valence": valence,
            "arousal": arousal,
        }

    def verify_va_column_order(self, rgb_crop: np.ndarray) -> dict:
        """실제 크롭 1장에서 VA 열 순서를 교차검증한다.

        우리 슬라이스 argmax(logits[:, :8]) 라벨이 라이브러리
        ``predict_emotions(crop)[0]`` 라벨과 같으면 ok. 어긋나면 슬라이싱을
        수정해야 하므로 결과를 있는 그대로 반환한다(위장 성공 금지).
        """
        crops = [np.asarray(rgb_crop)]
        features = np.asarray(self._rec.extract_features(crops))
        weights = np.asarray(self._rec.classifier_weights)
        logits = features @ weights.T
        bias = getattr(self._rec, "classifier_bias", None)
        if bias is not None:
            logits = logits + np.asarray(bias)
        idx_map = getattr(self._rec, "idx_to_emotion_class", None)
        labels = (
            [str(idx_map[i]) for i in range(NUM_EMOTIONS)]
            if isinstance(idx_map, dict) and len(idx_map) == NUM_EMOTIONS
            else list(EMOTION_LABELS_8)
        )
        ours = labels[int(np.argmax(logits[0, :NUM_EMOTIONS]))]
        lib_labels, _scores = self._rec.predict_emotions(crops, logits=True)
        library = str(lib_labels[0])
        return {
            "ok": ours == library,
            "our_slice_label": ours,
            "library_label": library,
            "detail": (
                "argmax(logits[:, :8]) matches rec.predict_emotions — slicing "
                "[emotions=:8, valence=8, arousal=9] confirmed"
                if ours == library
                else "ORDERING MISMATCH — adapt slicing before shipping"
            ),
        }


# ---------------------------------------------------------------- extractor


def padded_crop(rgb_frame: np.ndarray, bbox_xyxy: Sequence[float], scale: float = CROP_SCALE) -> np.ndarray:
    """bbox를 scale배(기본 1.5x)로 패딩해 이미지 경계로 클리핑한 RGB 크롭."""
    frame = validate_rgb_frame(rgb_frame)
    if len(bbox_xyxy) != 4:
        raise FaceSignalError(f"bbox_xyxy must have 4 elements [x1,y1,x2,y2], got {len(bbox_xyxy)}")
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    if not (x2 > x1 and y2 > y1):
        raise FaceSignalError(f"degenerate bbox_xyxy: {list(bbox_xyxy)} (need x2>x1 and y2>y1)")
    h, w = frame.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w, half_h = (x2 - x1) * scale / 2.0, (y2 - y1) * scale / 2.0
    ix1 = max(0, int(math.floor(cx - half_w)))
    iy1 = max(0, int(math.floor(cy - half_h)))
    ix2 = min(w, int(math.ceil(cx + half_w)))
    iy2 = min(h, int(math.ceil(cy + half_h)))
    ix2, iy2 = max(ix2, ix1 + 1), max(iy2, iy1 + 1)
    return frame[iy1:iy2, ix1:ix2]


class FaceSignalExtractor:
    """global-ID별 패딩 크롭 → frozen SignalSample 호환 dict 오케스트레이션."""

    def __init__(
        self,
        device: str | None = None,
        sample_every: int = 3,
        landmarker: Any | None = None,
        emotions: Any | None = None,
        model_path: str | Path = DEFAULT_LANDMARKER_PATH,
    ) -> None:
        self.sample_every = max(1, int(sample_every))
        self.landmarker = landmarker if landmarker is not None else MediaPipeSignals(model_path)
        self.emotions = emotions if emotions is not None else EmotionSignals(device=device)

    def extract(
        self,
        frame_idx: int,
        global_id: str,
        bbox_xyxy: Sequence[float],
        rgb_frame: np.ndarray,
        ts_ms_monotonic: int,
    ) -> dict | None:
        """sample_every 게이트를 통과하고 얼굴이 보일 때만 SignalSample dict 반환."""
        if int(frame_idx) % self.sample_every != 0:
            return None
        crop = padded_crop(rgb_frame, bbox_xyxy)
        signals = self.landmarker.detect(crop, ts_ms_monotonic)
        if signals is None:
            return None
        emo = self.emotions.predict([crop])
        sample = {
            "frame_idx": int(frame_idx),
            "global_id": str(global_id),
            "blendshapes": signals["blendshapes"],
            "yaw_deg": float(signals["yaw_deg"]),
            "pitch_deg": float(signals["pitch_deg"]),
            "roll_deg": float(signals["roll_deg"]),
            "blink": float(signals["blink"]),
            "emotion_labels": list(emo["labels"]),
            "emotion_probs": [float(p) for p in emo["probs"]],
            "valence": float(emo["valence"]),
            "arousal": float(emo["arousal"]),
        }
        assert set(sample.keys()) == set(SIGNAL_SAMPLE_KEYS)
        return sample

    def close(self) -> None:
        close = getattr(self.landmarker, "close", None)
        if callable(close):
            close()
