"""ON DAMM 캘리브레이션 데이터 준비 (todo 6, option B).

data/calib/{person}/{session}_{class}.mp4 관례로 녹화된 클립을 스캔해
frozen EmotiEffLib 특징(1280-d)을 캐시한다. 백본 학습/가중치 수정은 일절
하지 않는다(추론만).

파이프라인:
  1) data/calib/{person}/{session}_{class}.mp4 스캔 (파일명에서 session/class 파싱)
  2) target_fps(기본 5)로 프레임 추출
  3) 흐림 프레임 제거: cv2.Laplacian(gray, CV_64F) 분산 < 60
  4) 얼굴 없음 프레임 제거: MediaPipe FaceDetection(model_selection=0, 저렴한
     검출기 경로)을 다운스케일된 프로브 프레임에 실행. 기본 probe_every=1 이면
     모든 후보 프레임을 (다운스케일본으로) 검사한다. 전체 FaceLandmarker를
     프레임마다 돌리기엔 무거우므로 의도적으로 저렴한 detector를 쓴다. 더
     줄여야 하면 --probe-every N: N번째 후보마다 프로브를 실행하고, 각 후보는
     "가장 가까운 프로브"의 얼굴 여부를 상속받아 생존/폐기된다(아래
     nearest_probe_ok 참고 — 문서화된 대체 경로).
  5) 생존 프레임 → emotiefflib extract_features로 1280-d 임베딩(디바이스는
     todo 3과 동일하게 outputs/ondamm/video/env_report.json의 default_device,
     없으면 cpu). 사전학습 8-class probs도 함께 저장(베이스라인/Label Studio
     사전주석용).
  6) 출력(outputs 기본 outputs/ondamm/calib/{person}/):
        features_{person}.npy  (N,1280) float32
        labels.npy             (N,)    int   — class_names 인덱스
        sessions.npy           (N,)    str   — 파일명에서 파싱한 session id
        probs_{person}.npy     (N,8)   float64 — 사전학습 감정 확률
        frames_{person}.json   행별 프레임 메타(image 파일명 포함 — LS 매칭키)
        meta.json              클래스별 카운트/드랍 사유/임계값/제외·경고 플래그

클래스 규칙(실행 전략 문서의 레시피 수치):
    usable < 80   → HARD-EXCLUDED (features/labels에서 행 자체를 제외, exit 0)
    usable < 150  → WARNED (target ≥150)
생존 클래스가 2 미만이면 ValueError("need ≥2 classes ...") — 깨끗한 실패.

hermetic 테스트 계약: cv2 / mediapipe / emotiefflib는 함수 내부 지연 임포트.
테스트는 select_frame_indices / scan_frames / extract_features_for /
nearest_probe_ok / classify_counts 같은 순수 조각들을 numpy 합성 프레임으로
검증하고, prepare_person에는 가짜 probe/backend를 주입한다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "app"))

import ondamm_video_env  # noqa: E402  (repo convention: flat import via app/ on sys.path)
from ondamm_video_face_signals import EMOTION_LABELS_8, resolve_default_device, softmax  # noqa: E402

DEFAULT_FPS = 5
BLUR_LAPLACIAN_THRESHOLD = 60.0
MIN_CLASS_FRAMES = 80   # 미만 → HARD-EXCLUDED
WARN_CLASS_FRAMES = 150 # 미만 → WARNED (목표 ≥150)
FEATURE_DIM = 1280
PROBE_MAX_SIDE = 320    # 얼굴 프로브 다운스케일 최대 변(저렴한 경로 유지)
DEFAULT_BATCH_SIZE = 32
VIDEO_SUFFIX = ".mp4"
EMOTIEFF_MODEL_NAME = "enet_b0_8_va_mtl"

# {session}_{class}.mp4 — class에 underscore가 들어가지 않는 관례(마지막 '_'로 분리).
_CLIP_NAME_RE = re.compile(r"^.+_.+$")


class CalibPrepError(RuntimeError):
    """캘리브레이션 준비 단계의 명확한 실패."""


# ------------------------------------------------------------------ naming


def parse_clip_name(video_path: str | Path) -> tuple[str, str]:
    """{session}_{class}.mp4 → (session, class). 관례 위반은 CalibPrepError."""
    stem = Path(video_path).stem
    if not _CLIP_NAME_RE.match(stem) or "_" not in stem:
        raise CalibPrepError(
            f"clip filename '{Path(video_path).name}' violates the "
            f"'{{session}}_{{class}}{VIDEO_SUFFIX}' convention (e.g. s01_neutral.mp4)"
        )
    session, cls = stem.rsplit("_", 1)
    if not session or not cls:
        raise CalibPrepError(
            f"clip filename '{Path(video_path).name}' has an empty session or class part"
        )
    return session, cls


def discover_clips(data_root: str | Path) -> list[Path]:
    """data_root/{person}/*.mp4 나열. 하나도 없으면 기대 레이아웃을 알려주며 실패."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"calibration data root not found: '{root}' — expected layout "
            f"{root}/{{person}}/{{session}}_{{class}}{VIDEO_SUFFIX} "
            f"(e.g. {root}/minsu/s01_neutral.mp4)"
        )
    clips = sorted(p for p in root.glob(f"*/*{VIDEO_SUFFIX}") if p.is_file())
    if not clips:
        raise FileNotFoundError(
            f"no {VIDEO_SUFFIX} clips under '{root}' — expected layout "
            f"{root}/{{person}}/{{session}}_{{class}}{VIDEO_SUFFIX} "
            f"(e.g. {root}/minsu/s01_neutral.mp4)"
        )
    return clips


# ------------------------------------------------------------------ filtering


def laplacian_variance(bgr_frame: np.ndarray) -> float:
    """cv2.Laplacian(gray, CV_64F) 분산. 낮을수록 흐림(cv2는 함수 내 지연 임포트)."""
    import cv2

    frame = np.asarray(bgr_frame)
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise CalibPrepError(f"expected HxWx3 BGR (or HxW gray) frame, got shape {frame.shape}")
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_blurred(lap_var: float, threshold: float = BLUR_LAPLACIAN_THRESHOLD) -> bool:
    return float(lap_var) < float(threshold)


def select_frame_indices(src_fps: float, target_fps: float, n_frames: int) -> list[int]:
    """소스 fps→target_fps 등간격 샘플링 인덱스(결정적)."""
    if src_fps <= 0:
        src_fps = 30.0  # 메타데이터 누락 시 관례 폴백
    if target_fps <= 0:
        raise CalibPrepError(f"target fps must be > 0, got {target_fps}")
    step = src_fps / float(target_fps)
    idx: list[int] = []
    k = 0
    while True:
        i = int(round(k * step))
        if i >= n_frames:
            break
        idx.append(i)
        k += 1
    return idx


def nearest_probe_ok(n_candidates: int, probe_positions: Sequence[int], probe_flags: Sequence[bool]) -> list[bool]:
    """후보 i의 생존 여부 = 가장 가까운 프로브 위치의 얼굴 플래그(동률은 앞쪽).

    --probe-every N > 1 일 때의 문서화된 근사: 프로브 사이 후보들은 인접 프로브의
    판정을 상속한다. probe_positions는 오름차순이라고 가정한다.
    """
    if len(probe_positions) != len(probe_flags):
        raise CalibPrepError("probe_positions and probe_flags must have equal length")
    if n_candidates and not probe_positions:
        raise CalibPrepError("no probes scheduled for non-empty candidate set")
    out = [False] * int(n_candidates)
    for i in range(int(n_candidates)):
        best_j = 0
        best_dist = None
        for j, pos in enumerate(probe_positions):
            dist = abs(i - pos)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_j = j
        out[i] = bool(probe_flags[best_j])
    return out


def scan_frames(
    frames_bgr: Sequence[np.ndarray],
    *,
    blur_threshold: float = BLUR_LAPLACIAN_THRESHOLD,
    probe: Any | None = None,
    probe_every: int = 1,
) -> tuple[list[int], dict[str, int]]:
    """후보 프레임에서 흐림/얼굴없음 필터. → (생존 로컬 인덱스, 드랍 카운트).

    probe는 .detect(bgr)->bool 인터페이스(None이면 얼굴 검사 생략=모두 통과).
    """
    dropped = {"blurred": 0, "faceless": 0}
    blur_ok: list[int] = []
    for i, fr in enumerate(frames_bgr):
        if is_blurred(laplacian_variance(fr), blur_threshold):
            dropped["blurred"] += 1
        else:
            blur_ok.append(i)

    if probe is None:
        return blur_ok, dropped

    probe_every = max(1, int(probe_every))
    probe_positions = list(range(0, len(blur_ok), probe_every))
    probe_flags = [bool(probe.detect(frames_bgr[p])) for p in probe_positions]
    keep_flags = nearest_probe_ok(len(blur_ok), probe_positions, probe_flags)
    kept = [idx for idx, ok in zip(blur_ok, keep_flags) if ok]
    dropped["faceless"] = len(blur_ok) - len(kept)
    return kept, dropped


# ------------------------------------------------------------------ features


class FaceProbe:
    """저렴한 MediaPipe FaceDetection 기반 얼굴 존재 프로브(다운스케일 입력)."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_detection_confidence
        )

    def detect(self, bgr_frame: np.ndarray) -> bool:
        frame = np.asarray(bgr_frame)
        h, w = frame.shape[:2]
        scale = PROBE_MAX_SIDE / float(max(h, w))
        if scale < 1.0:
            frame = self._cv2.resize(
                frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=self._cv2.INTER_AREA
            )
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        result = self._fd.process(rgb)
        return bool(result.detections)

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            fd.close()


def make_emotion_backend(device: str | None = None) -> Any:
    """EmotiEffLib 백엔드 팩토리(테스트 monkeypatch 대상).

    반환 객체 계약: .embed(list_of_rgb)->(N,1280), .emotion_probs(features)->(N,8),
    .device 속성. emotiefflib/torch는 이 함수 안에서만 임포트된다(hermetic 보장).
    """
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    resolved = device if device in ("cpu", "mps", "cuda") else resolve_default_device()
    rec = EmotiEffLibRecognizer(engine="torch", model_name=EMOTIEFF_MODEL_NAME, device=resolved)

    class _Backend:
        device = resolved

        @staticmethod
        def embed(rgb_crops: Sequence[np.ndarray]) -> np.ndarray:
            feats = np.asarray(rec.extract_features(list(rgb_crops)), dtype=np.float32)
            if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
                raise CalibPrepError(
                    f"expected ({len(rgb_crops)},{FEATURE_DIM}) frozen features, got {feats.shape}"
                )
            return feats

        @staticmethod
        def emotion_probs(features: np.ndarray) -> np.ndarray:
            feats = np.asarray(features, dtype=np.float64)
            logits = feats @ np.asarray(rec.classifier_weights, dtype=np.float64).T
            bias = getattr(rec, "classifier_bias", None)
            if bias is not None:
                logits = logits + np.asarray(bias, dtype=np.float64)
            return softmax(logits[:, : len(EMOTION_LABELS_8)])

    return _Backend()


def extract_features_for(
    frames_bgr: Sequence[np.ndarray],
    kept_local_idx: Sequence[int],
    backend: Any,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """생존 프레임(BGR) → RGB 배치 변환 후 (features(K,1280), probs(K,8))."""
    batch_size = max(1, int(batch_size))
    feats_out: list[np.ndarray] = []
    for start in range(0, len(kept_local_idx), batch_size):
        chunk = [frames_bgr[i][:, :, ::-1] for i in kept_local_idx[start : start + batch_size]]  # BGR→RGB
        feats_out.append(backend.embed(chunk))
    features = (
        np.concatenate(feats_out, axis=0)
        if feats_out
        else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    )
    probs = backend.emotion_probs(features) if len(features) else np.zeros((0, len(EMOTION_LABELS_8)))
    return features, np.asarray(probs)


# ------------------------------------------------------------------ video io


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[int, float, np.ndarray]]:
    """(source_frame_idx, ts_sec, bgr) 순차 열람(cv2 지연 임포트)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise CalibPrepError(f"cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, idx / fps, frame
            idx += 1
    finally:
        cap.release()


def video_src_fps(video_path: str | Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise CalibPrepError(f"cannot open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 30.0
    finally:
        cap.release()


def scan_clip(
    video_path: str | Path,
    *,
    target_fps: float = DEFAULT_FPS,
    blur_threshold: float = BLUR_LAPLACIAN_THRESHOLD,
    probe: Any | None = None,
    probe_every: int = 1,
) -> tuple[list[int], dict[str, int], int]:
    """1차 패스: 프레임을 메모리에 쌓지 않고 생존 소스 인덱스만 산출.

    → (kept_source_indices, dropped_counts, n_extracted_candidates)
    """
    src_fps = video_src_fps(video_path)
    step = src_fps / float(target_fps)
    candidates: list[np.ndarray] = []
    source_of_local: list[int] = []
    next_k = 0
    next_wanted = 0
    for src_idx, _ts, frame in iter_video_frames(video_path):
        if src_idx >= next_wanted:
            candidates.append(frame)
            source_of_local.append(src_idx)
            next_k += 1
            next_wanted = int(round(next_k * step))
    kept_local, dropped = scan_frames(
        candidates, blur_threshold=blur_threshold, probe=probe, probe_every=probe_every
    )
    kept_source = [source_of_local[i] for i in kept_local]
    return kept_source, dropped, len(candidates)


def extract_clip_features(
    video_path: str | Path,
    kept_source_indices: Sequence[int],
    backend: Any,
    *,
    target_fps: float = DEFAULT_FPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """2차 패스: 생존 소스 인덱스만 다시 디코드해 특징 추출(메모리 절약)."""
    import cv2

    wanted = set(int(i) for i in kept_source_indices)
    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise CalibPrepError(f"cannot open video: {video_path}")
        idx = 0
        while wanted:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                frames[idx] = frame
                wanted.discard(idx)
            idx += 1
    finally:
        cap.release()
    missing = sorted(wanted)
    if missing:
        raise CalibPrepError(f"video '{video_path}' lost frames on second pass: {missing[:5]}")
    ordered = [frames[int(i)] for i in kept_source_indices]
    return extract_features_for(ordered, range(len(ordered)), backend, batch_size)


# ------------------------------------------------------------------ assembly


@dataclass
class ClipResult:
    video: Path
    session: str
    cls: str
    n_candidates: int
    dropped: dict[str, int] = field(default_factory=dict)
    features: np.ndarray | None = None
    probs: np.ndarray | None = None


def classify_counts(counts: dict[str, int]) -> dict[str, list[str]]:
    """usable 카운트 → excluded(<80) / warned(<150) 목록. 순수 함수(경계 테스트 대상)."""
    excluded = sorted(c for c, n in counts.items() if n < MIN_CLASS_FRAMES)
    warned = sorted(
        c for c, n in counts.items() if MIN_CLASS_FRAMES <= n < WARN_CLASS_FRAMES
    )
    return {"excluded": excluded, "warned": warned}


def prepare_person(
    person: str,
    clips: Sequence[Path],
    out_dir: str | Path,
    *,
    fps: float = DEFAULT_FPS,
    blur_threshold: float = BLUR_LAPLACIAN_THRESHOLD,
    probe_every: int = 1,
    device: str | None = None,
    probe_factory: Callable[[], Any] | None = None,
    backend_factory: Callable[[str | None], Any] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """개인 단위 파이프라인: 스캔→특징→클래스 규칙→npy/meta 기록. meta dict 반환."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = (probe_factory or FaceProbe)() if probe_every >= 1 else None
    backend = (backend_factory or make_emotion_backend)(device)

    results: list[ClipResult] = []
    for clip in clips:
        session, cls = parse_clip_name(clip)
        kept_source, dropped, n_cand = scan_clip(
            clip, target_fps=fps, blur_threshold=blur_threshold, probe=probe, probe_every=probe_every
        )
        res = ClipResult(video=clip, session=session, cls=cls, n_candidates=n_cand, dropped=dropped)
        if kept_source:
            res.features, res.probs = extract_clip_features(
                clip, kept_source, backend, target_fps=fps, batch_size=batch_size
            )
        results.append(res)

    per_class_kept: dict[str, int] = {}
    per_class_dropped: dict[str, dict[str, int]] = {}
    for res in results:
        per_class_dropped.setdefault(res.cls, {"blurred": 0, "faceless": 0})
        for reason, n in res.dropped.items():
            per_class_dropped[res.cls][reason] = per_class_dropped[res.cls].get(reason, 0) + n
        if res.features is not None:
            per_class_kept[res.cls] = per_class_kept.get(res.cls, 0) + int(res.features.shape[0])
        else:
            per_class_kept.setdefault(res.cls, 0)

    flags = classify_counts(per_class_kept)
    excluded_set = set(flags["excluded"])
    class_names = sorted(c for c in per_class_kept if c not in excluded_set)

    rows: list[tuple[np.ndarray, np.ndarray, int, str]] = []
    for res in results:
        if res.cls in excluded_set or res.features is None or res.probs is None:
            continue
        cls_idx = class_names.index(res.cls)
        for k in range(res.features.shape[0]):
            rows.append((res.features[k], res.probs[k], cls_idx, res.session))

    if len(class_names) < 2:
        raise ValueError(
            f"need ≥2 classes after exclusion for person '{person}' "
            f"(usable per class: {per_class_kept}, excluded: {flags['excluded']})"
        )

    features = np.stack([r[0] for r in rows]).astype(np.float32)
    probs = np.stack([r[1] for r in rows]).astype(np.float64)
    labels = np.array([r[2] for r in rows], dtype=np.int64)
    sessions = np.array([r[3] for r in rows])

    cursor_by_cls: dict[str, int] = {c: 0 for c in class_names}
    frame_records: list[dict] = []
    for i, (_f, _p, cls_idx, session) in enumerate(rows):
        cls = class_names[cls_idx]
        seq = cursor_by_cls[cls]
        cursor_by_cls[cls] = seq + 1
        name = f"{person}_{session}_{cls}_{seq:05d}.jpg"
        frame_records.append({"index": i, "image": name, "person": person, "session": session, "class": cls})

    np.save(out_dir / f"features_{person}.npy", features)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "sessions.npy", sessions)
    np.save(out_dir / f"probs_{person}.npy", probs)
    with open(out_dir / "frames_{p}.json".format(p=person), "w", encoding="utf-8") as fh:
        json.dump(frame_records, fh, ensure_ascii=False, indent=2)

    meta = {
        "person": person,
        "device": getattr(backend, "device", device or "cpu"),
        "thresholds": {
            "fps": float(fps),
            "blur_laplacian": float(blur_threshold),
            "min_class_frames": MIN_CLASS_FRAMES,
            "warn_class_frames": WARN_CLASS_FRAMES,
            "probe_every": int(probe_every),
        },
        "class_names": class_names,
        "classes": {
            c: {
                "usable": int(per_class_kept[c]),
                "dropped_blurred": int(per_class_dropped[c]["blurred"]),
                "dropped_faceless": int(per_class_dropped[c]["faceless"]),
                "status": (
                    "excluded" if c in excluded_set else ("warned" if c in flags["warned"] else "kept")
                ),
            }
            for c in sorted(per_class_kept)
        },
        "excluded_classes": flags["excluded"],
        "warned_classes": flags["warned"],
        "total_rows": int(features.shape[0]),
        "clips": [
            {
                "video": res.video.name,
                "session": res.session,
                "class": res.cls,
                "candidates": res.n_candidates,
                "dropped": res.dropped,
            }
            for res in results
        ],
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return meta


# ------------------------------------------------------------------ cli


def print_class_report(person: str, meta: dict) -> None:
    print(f"[{person}] usable frames per class (target ≥{WARN_CLASS_FRAMES}, hard floor {MIN_CLASS_FRAMES}):")
    for name, info in meta["classes"].items():
        mark = {"excluded": "EXCLUDED", "warned": "WARN", "kept": "ok"}[info["status"]]
        print(
            f"  {name:<16} usable={info['usable']:<5} blurred={info['dropped_blurred']:<4} "
            f"faceless={info['dropped_faceless']:<4} [{mark}]"
        )
    if meta["excluded_classes"]:
        print(f"  excluded (<{MIN_CLASS_FRAMES}): {', '.join(meta['excluded_classes'])}")
    if meta["warned_classes"]:
        print(f"  warned (<{WARN_CLASS_FRAMES}): {', '.join(meta['warned_classes'])}")
    print(f"  rows written: {meta['total_rows']} → {meta['device']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ondamm_calib_prep",
        description=(
            "ON DAMM calibration prep: scan data/calib/{person}/{session}_{class}.mp4, "
            "filter blurry/faceless frames, cache frozen EmotiEffLib features."
        ),
    )
    parser.add_argument("--data-root", default="data/calib")
    parser.add_argument("--out", default="outputs/ondamm/calib")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--blur-threshold", type=float, default=BLUR_LAPLACIAN_THRESHOLD)
    parser.add_argument("--probe-every", type=int, default=1,
                        help="run the cheap face probe every Nth surviving candidate (default 1)")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args(argv)

    device = None if args.device == "auto" else args.device
    try:
        clips = discover_clips(args.data_root)
    except (FileNotFoundError, CalibPrepError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    by_person: dict[str, list[Path]] = {}
    for clip in clips:
        by_person.setdefault(clip.parent.name, []).append(clip)

    failed = False
    for person in sorted(by_person):
        try:
            meta = prepare_person(
                person,
                by_person[person],
                Path(args.out) / person,
                fps=args.fps,
                blur_threshold=args.blur_threshold,
                probe_every=args.probe_every,
                device=device,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failed = True
            continue
        print_class_report(person, meta)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
