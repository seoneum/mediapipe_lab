"""ON DAMM 영상 분석기 환경 검증 (todo 1).

`python -m app.ondamm_video_env --check` 으로 실행한다.

- 임포트 검증 (torch / ultralytics / insightface / emotiefflib / mediapipe / cv2 / sklearn)
- outputs/ondamm/video/env_report.json 작성:
    {"mps_available": bool, "emotiefflib_mps_ok": bool,
     "ffmpeg_found": bool, "models_present": bool}
- MPS 스파이크: 시드 고정 입력 한 개를 fp32로 CPU와 MPS에서 각각 순전파하여
  logit 최대절대차가 1e-3 이하면 emotiefflib_mps_ok=true.
- 스파이크 실패/MPS 부재/EmotiEffLib 부재 시에도 exit 0으로 보고서만 남긴다(우아한 강등).
  이때 모듈 상수 DEFAULT_DEVICE는 "cpu"를 유지한다(todo 3/5가 import해서 소비).
- fp16은 MPS에서 NaN을 만들 수 있으므로 금지다. 이 모듈은 어떤 텐서에도 반정밀(half) 변환을 호출하지 않는다.

기본 테스트(tests/test_ondamm_video_env.py)는 네트워크나 실제 모델 로드 없이
이 모듈의 헬퍼를 monkeypatch해 검증한다(hermetic).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "outputs" / "ondamm" / "video" / "env_report.json"

# EmotiEffLib 표정+VA 모델. todo 3/5도 동일 이름을 사용한다.
EMOTIEFF_MODEL_NAME = "enet_b0_8_va_mtl"
SPIKE_MAX_ABS_LOGIT_DIFF = 1e-3

# todo 3/5가 import하는 모듈 상수. 기본값은 항상 안전한 "cpu"이며,
# --check 실행에서 MPS 스파이크가 통과한 경우에만 "mps"로 갱신된다.
DEFAULT_DEVICE = "cpu"

# 임포트만 확인할 핵심 의존성. 여기엔 실제 모델 로드가 없다.
CORE_IMPORTS = (
    "torch",
    "ultralytics",
    "insightface",
    "emotiefflib",
    "mediapipe",
    "cv2",
    "sklearn",
)


def find_ffmpeg(which: Callable[[str], str | None] | None = None) -> bool:
    """PATH에서 ffmpeg 발견 여부. 테스트에서 which를 주입할 수 있다."""
    finder = which or shutil.which
    return finder("ffmpeg") is not None


def _asset_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def models_present(
    model_dir: Path | None = None,
    insightface_home: Path | None = None,
) -> bool:
    """사전 다운로드된 모델 자산 존재 여부(네트워크 없이 파일 크기만 확인).

    - models/face_landmarker.task > 1MB
    - models/yolo26s.pt 존재(비어있지 않음)
    - ~/.insightface/models/buffalo_l 디렉토리 비어있지 않음
    EmotiEffLib 체크포인트는 HuggingFace 존재 확인이 다운로드를 유발할 수 있어
    의도적으로 models_present 판정에서 제외한다(스파이크 결과가 별도 증거).
    """
    root = model_dir or (REPO_ROOT / "models")
    home = insightface_home or Path.home()
    landmarker = root / "face_landmarker.task"
    yolo = root / "yolo26s.pt"
    buffalo = home / ".insightface" / "models" / "buffalo_l"

    if _asset_bytes(landmarker) <= 1_048_576:
        return False
    if _asset_bytes(yolo) == 0:
        return False
    if not buffalo.is_dir() or not any(buffalo.iterdir()):
        return False
    return True


def torch_mps_available() -> bool:
    """torch MPS 사용 가능 여부. torch가 없으면 False."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def validate_imports() -> dict[str, bool]:
    """핵심 의존성 임포트 가능 여부(부작용 없는 임포트만)."""
    result: dict[str, bool] = {}
    for name in CORE_IMPORTS:
        try:
            __import__(name)
            result[name] = True
        except Exception:
            result[name] = False
    return result


def mps_spike(seed: int = 1234) -> tuple[bool, str]:
    """enet_b0_8_va_mtl을 CPU vs MPS로 fp32 순전파 1회씩 돌려 logit 차이를 검증.

    Returns:
        (emotiefflib_mps_ok, reason_or_detail)
    우아한 강등 계약: 어떤 실패에도 예외를 밖으로 던지지 않고 (False, 사유)를 반환.
    """
    try:
        import numpy as np
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    except Exception as exc:  # emotiefflib 미설치 등
        return False, f"emotiefflib unavailable: {type(exc).__name__}: {exc}"

    if not torch_mps_available():
        return False, "mps_unavailable"

    import torch

    # 시드 고정 + 결정적 입력(uint8 RGB 1장). fp32 전용 — 반정밀(half) 변환 호출 금지(MPS NaN 문제).
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)

    def logits_on(device: str):
        recognizer = EmotiEffLibRecognizer(
            engine="torch", model_name=EMOTIEFF_MODEL_NAME, device=device
        )
        features = recognizer.extract_features([image])
        # classifier: logits = features @ W.T + b (VA 포함 10열)
        return features @ recognizer.classifier_weights.T + recognizer.classifier_bias

    try:
        cpu_logits = logits_on("cpu")
        mps_logits = logits_on("mps")
    except Exception as exc:
        return False, f"spike forward failed: {type(exc).__name__}: {exc}"

    if cpu_logits.shape != mps_logits.shape:
        return False, f"shape mismatch: {cpu_logits.shape} vs {mps_logits.shape}"
    if not np.isfinite(mps_logits).all() or not np.isfinite(cpu_logits).all():
        # NaN/Inf는 즉시 실패 처리(fp16 NaN 회귀 감지용 방어선).
        return False, "non-finite logits detected"

    max_abs_diff = float(np.max(np.abs(cpu_logits - mps_logits)))
    ok = max_abs_diff <= SPIKE_MAX_ABS_LOGIT_DIFF
    detail = f"max_abs_logit_diff={max_abs_diff:.3e} tolerance={SPIKE_MAX_ABS_LOGIT_DIFF:g}"
    return ok, detail


def compute_default_device(mps_available: bool, emotiefflib_mps_ok: bool) -> str:
    """스파이크 결과에서 todo 3/5용 디바이스를 결정한다."""
    return "mps" if (mps_available and emotiefflib_mps_ok) else "cpu"


def build_report(
    ffmpeg_found: bool,
    models_present_flag: bool,
    spike_fn: Callable[[], tuple[bool, str]] | None = None,
    imports: dict[str, bool] | None = None,
) -> dict:
    """환경 보고서 dict를 만든다. 어떤 조합에서도 4개 필수 boolean 키를 보장한다."""
    mps_available = torch_mps_available()
    spike = spike_fn or mps_spike
    try:
        spike_ok, detail = spike()
    except Exception as exc:  # 스파이크 헬퍼 계약 위반 방어(우아한 강등).
        spike_ok, detail = False, f"spike crashed: {type(exc).__name__}: {exc}"
    report: dict = {
        "mps_available": bool(mps_available),
        "emotiefflib_mps_ok": bool(spike_ok),
        "ffmpeg_found": bool(ffmpeg_found),
        "models_present": bool(models_present_flag),
    }
    device = compute_default_device(mps_available, bool(spike_ok))
    report["default_device"] = device
    report["imports"] = imports if imports is not None else validate_imports()
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    if spike_ok:
        report["spike_detail"] = detail
    else:
        # 우아한 강등: 사유를 남기고 DEFAULT_DEVICE는 cpu 유지.
        report["reason"] = detail
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device
    return report


def write_report(report: dict, report_path: Path | None = None) -> Path:
    """보고서를 임시파일→원자적 치환으로 기록해 부분 JSON을 남기지 않는다."""
    path = report_path or REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp_path, path)
    return path


def run_check(report_path: Path | None = None) -> tuple[dict, int]:
    """--check 본체. (report, exit_code) 반환. 완료만 되면 항상 exit 0(강등 허용)."""
    report = build_report(
        ffmpeg_found=find_ffmpeg(),
        models_present_flag=models_present(),
    )
    path = write_report(report, report_path)
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ondamm_video_env",
        description="ON DAMM 영상 분석기 환경 검증 (--check).",
    )
    parser.add_argument("--check", action="store_true", help="환경 검증 실행 및 보고서 작성")
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return 2

    report, code = run_check()

    print("ON DAMM video analyzer environment check")
    print(f"  mps_available   : {report['mps_available']}")
    print(f"  emotiefflib_mps_ok: {report['emotiefflib_mps_ok']}")
    if not report["emotiefflib_mps_ok"]:
        print(f"    reason: {report.get('reason', 'unknown')}")
    print(f"  ffmpeg_found    : {report['ffmpeg_found']}")
    print(f"  models_present  : {report['models_present']}")
    print(f"  DEFAULT_DEVICE  : {DEFAULT_DEVICE}")
    if not report["ffmpeg_found"]:
        print("  ffmpeg not found — install it:")
        print("    macOS (Homebrew): brew install ffmpeg")
        print("    Ubuntu/Debian   : sudo apt install ffmpeg")
    print(f"  report written  : {REPORT_PATH}")
    return code


if __name__ == "__main__":
    sys.exit(main())
