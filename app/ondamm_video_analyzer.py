"""ON DAMM 영상 분석기 오케스트레이터 (todo 5).

입력 MP4 1개를 받아 다음 파이프라인을 단일 패스 스트리밍으로 수행한다:

    FrameReader → PersonTracker(TrackObservation) → FaceSignalExtractor(SignalSample dict)
    → 누적 샘플 → ondamm_video_metrics.summarize_person(PersonMetrics)
    → draw_overlays로 주석 프레임 스트리밍 생성 → render.encode(mp4v temp → ffmpeg libx264)
    → metrics JSON/CSV export

라이브 vs 최종 라벨의 정직한 차이 (설계 문서화):
트래킹은 상태를 가지므로(persist=True) 두 번째 패스에서 재검출하면 관측값이 달라진다.
따라서 두 패스는 하지 않고, "단일 패스 + 결과 영상은 러닝(running) 추정치로 렌더,
메트릭 파일(JSON/CSV)은 루프 종료 후 전체 윈도우로 최종 계산" 전략을 쓴다.
즉 영상에 새겨진 집중%/흥미/표정 라벨은 그 시점까지 누적된 부분 윈도우 추정값이고,
metrics JSON/CSV의 값은 영상 전체를 본 최종값이다. 두 값이 프레임 단위로 일치한다는
보장은 의도적으로 하지 않으며, 이는 문서화된 동작이다(README의 지표 표는 JSON/CSV 기준).
렌더 자체는 결정적이다: 같은 입력+같은 스텁/모델이면 같은 출력 비트가 나온다.

타임스탬프 계약: MediaPipe VIDEO 모드는 인스턴스당 엄격 단조 증가 ms 타임스탬프를
요구하므로, extract 호출마다 ``frame_idx * 1000 + 프레임 내 슬롯`` 을 넘겨 한 프레임에
여러 사람이 있어도 항상 증가하도록 보장한다.

비진단 고지: 모든 지표는 행동 프록시 추정이며 의학적·교육적 진단이 아니다. 이 문구는
렌더러가 모든 프레임에 새긴다(burned-in).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "app"))

import ondamm_video_env  # noqa: E402  (repo convention: flat import via app/ on sys.path)
import ondamm_video_metrics as metrics_mod  # noqa: E402
import ondamm_video_render as render_mod  # noqa: E402
from ondamm_video_face_signals import (  # noqa: E402
    EmotionSignals,
    FaceSignalExtractor,
    MediaPipeSignals,
    resolve_default_device,
)
from ondamm_video_metrics import PersonMetrics, summarize_person  # noqa: E402
from ondamm_video_render import RenderError, argmax_expression, draw_overlays, encode  # noqa: E402
from ondamm_video_tracking import (  # noqa: E402
    FaceEmbedder,
    FrameReader,
    PersonTracker,
    VideoInputError,
)

DOWNLOAD_SCRIPT_HINT = "scripts/download_video_models.sh"
DEFAULT_WEIGHTS = "models/yolo26s.pt"
DEFAULT_LANDMARKER = "models/face_landmarker.task"
VALID_DEVICES = ("auto", "cpu", "mps", "cuda")


class ModelMissingError(RuntimeError):
    """필요한 로컬 모델 자산이 없을 때(exit 3로 매핑됨)."""


def missing_model_files() -> list[str]:
    """실행에 필요한 로컬 모델 자산 중 누락된 것들의 설명 목록(네트워크 접촉 없음)."""
    missing: list[str] = []
    yolo = REPO_ROOT / DEFAULT_WEIGHTS
    if not yolo.is_file() or yolo.stat().st_size == 0:
        missing.append(DEFAULT_WEIGHTS)
    landmarker = REPO_ROOT / DEFAULT_LANDMARKER
    if not landmarker.is_file() or landmarker.stat().st_size <= 0:
        missing.append(DEFAULT_LANDMARKER)
    buffalo = Path.home() / ".insightface" / "models" / "buffalo_l"
    if not buffalo.is_dir() or not any(buffalo.iterdir()):
        missing.append("~/.insightface/models/buffalo_l")
    return missing


def resolve_device(device: str | None) -> str:
    """``auto``면 env 리포트/즉시 검사로 판정하고, 명시값은 그대로 신뢰한다."""
    value = (device or "auto").strip().lower()
    if value == "auto":
        return resolve_default_device()
    if value not in VALID_DEVICES:
        raise ValueError(f"--device must be one of {VALID_DEVICES}, got '{device}'")
    return value


class _PassState:
    """단일 패스 동안의 누적 상태(샘플/신뢰도/프레임 수)."""

    def __init__(self) -> None:
        self.samples_by_gid: dict[str, list[dict]] = {}
        self.low_confidence_by_gid: dict[str, bool] = {}
        self.total_frames = 0


def _annotated_frames(
    reader: FrameReader,
    tracker: Any,
    extractor: Any,
    state: _PassState,
    font_path: str | None,
) -> Iterator[np.ndarray]:
    """한 프레임씩 처리해 주석 프레임을 yield하는 제너레이터(스트리밍, 무대량버퍼)."""
    running_metrics: dict[str, PersonMetrics] = {}
    fallback_expr: dict[str, str] = {}
    sample_every = max(1, int(getattr(extractor, "sample_every", 1)))

    for frame_idx, ts_sec, frame in reader:
        observations = tracker.process_frame(frame, frame_idx=frame_idx, ts_sec=ts_sec)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        new_samples: set[str] = set()
        for slot, obs in enumerate(observations):
            gid = obs.global_id
            state.low_confidence_by_gid[gid] = bool(obs.low_confidence)
            ts_ms = frame_idx * 1000 + slot  # 엄격 단조 보장(모듈 docstring 참고)
            sample = extractor.extract(
                frame_idx, gid, obs.bbox_xyxy, rgb, ts_ms_monotonic=ts_ms
            )
            if sample is not None:
                state.samples_by_gid.setdefault(gid, []).append(sample)
                new_samples.add(gid)
                probs = sample.get("emotion_probs") or []
                labels = sample.get("emotion_labels") or []
                if probs and labels:
                    fallback_expr[gid] = argmax_expression(labels, probs)

        # 라이브 라벨용 러닝 추정치: 새 샘플이 들어온 프레임에서만 재계산(O(n²) 회피).
        for gid in new_samples:
            running_metrics[gid] = summarize_person(
                gid,
                state.samples_by_gid[gid],
                total_frames=frame_idx + 1,  # 러닝 추정치(최종값은 루프 종료 후)
                fps=reader.fps,
                low_confidence=state.low_confidence_by_gid.get(gid, False),
            )

        annotated = draw_overlays(
            frame,
            observations,
            running_metrics,
            font_path,
            fallback_expr_by_gid=fallback_expr,
        )
        state.total_frames = frame_idx + 1
        yield annotated


def write_metrics_json(metrics: Iterable[PersonMetrics], path: str | Path | None) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [m.to_json() for m in metrics]
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp.replace(out)
    return out


def write_metrics_csv(metrics: Iterable[PersonMetrics], path: str | Path | None) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.to_csv_row() for m in metrics]
    fieldnames = list(PersonMetrics.__dataclass_fields__.keys())  # 헤더 = 스키마 필드명 고정
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(out)
    return out


def finalize_metrics(state: _PassState, fps: float) -> list[PersonMetrics]:
    """루프 종료 후 전체 윈도우로 최종 PersonMetrics를 계산한다(gid 사전순 결정적 정렬)."""
    final: list[PersonMetrics] = []
    for gid in sorted(state.samples_by_gid):
        final.append(
            summarize_person(
                gid,
                state.samples_by_gid[gid],
                total_frames=state.total_frames,
                fps=fps,
                low_confidence=state.low_confidence_by_gid.get(gid, False),
            )
        )
    return final


def build_real_pipeline(device: str) -> tuple[PersonTracker, FaceSignalExtractor]:
    """실모델 파이프라인 구성(YOLO26+BoT-SORT+ArcFace / MediaPipe+EmotiEffLib)."""
    tracker = PersonTracker(device=device, embedder=FaceEmbedder())
    extractor = FaceSignalExtractor(
        device=device,
        landmarker=MediaPipeSignals(DEFAULT_LANDMARKER),
        emotions=EmotionSignals(device=device),
    )
    return tracker, extractor


def run(
    args: Any,
    *,
    tracker: Any | None = None,
    extractor: Any | None = None,
    require_models: bool = True,
    font_path: str | None = None,
    render: Any | None = None,
) -> int:
    """CLI 계약 본체. 반환값 = 종료 코드(0 성공).

    ``args`` 는 ``input/output/device/sample_every/metrics_json/metrics_csv``
    속성을 가진 객체(argparse.Namespace 또는 동형). hermetic 테스트를 위해
    tracker/extractor 주입과 require_models=False 를 허용한다(프로덕션 경로는
    build_real_pipeline 사용). ``render`` 는 encode 함수 주입 지점(테스트용).
    """
    encode_fn = render.encode if render is not None else encode

    input_path = Path(getattr(args, "input"))
    output_path = Path(getattr(args, "output"))
    device_arg = getattr(args, "device", "auto")
    sample_every = int(getattr(args, "sample_every", 3) or 3)
    metrics_json = getattr(args, "metrics_json", None)
    metrics_csv = getattr(args, "metrics_csv", None)

    # 1) 입력 검증(가장 싼 실패부터): VideoInputError → exit 2 (CLI가 변환)
    reader = FrameReader(input_path)  # raises VideoInputError
    try:
        # 2) 모델 자산 선확인 → exit 3 (CLI가 변환)
        if require_models:
            missing = missing_model_files()
            if missing:
                raise ModelMissingError(
                    "missing model assets: " + ", ".join(missing)
                    + f" — run `bash {DOWNLOAD_SCRIPT_HINT}` from the repo root first"
                )
        device = resolve_device(device_arg)
        if tracker is None or extractor is None:
            real_tracker, real_extractor = build_real_pipeline(device)
            tracker = tracker if tracker is not None else real_tracker
            extractor = extractor if extractor is not None else real_extractor

        state = _PassState()
        frames = _annotated_frames(reader, tracker, extractor, state, font_path)
        try:
            encode_fn(frames, output_path, reader.fps)
        finally:
            close = getattr(extractor, "close", None)
            if callable(close):
                close()
    finally:
        reader.close()

    final_metrics = finalize_metrics(state, reader.fps)
    write_metrics_json(final_metrics, metrics_json)
    write_metrics_csv(final_metrics, metrics_csv)
    return 0
