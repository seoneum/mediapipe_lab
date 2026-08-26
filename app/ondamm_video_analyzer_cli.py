"""ON DAMM 영상 분석기 CLI (todo 5).

README "ON DAMM 영상 분석기" 섹션의 계약을 그대로 구현한다:

    .venv/bin/python -m app.ondamm_video_analyzer_cli \\
        --input input.mp4 --output outputs/ondamm/video/result.mp4 \\
        --device {auto,cpu,mps,cuda} --sample-every N \\
        --metrics-json PATH --metrics-csv PATH

종료 코드:
    0 성공 / 2 입력 영상 열기 실패 / 3 모델 파일 없음 / 4 렌더 실패.
그 외 예기치 않은 오류는 1(계약 외 내부 오류)로 마무리한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "app"))

from ondamm_video_analyzer import (  # noqa: E402
    DOWNLOAD_SCRIPT_HINT,
    ModelMissingError,
    run,
)
from ondamm_video_render import RenderError  # noqa: E402
from ondamm_video_tracking import VideoInputError  # noqa: E402

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_INPUT_FAIL = 2
EXIT_MODELS_MISSING = 3
EXIT_RENDER_FAIL = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ondamm_video_analyzer_cli",
        description=(
            "ON DAMM 영상 분석기: 사람별 고유 ID 유지 + 집중/흥미/표정 한글 자막 MP4와 "
            "개인별 지표 JSON/CSV를 만든다(오프라인, 비진단 행동 프록시)."
        ),
    )
    parser.add_argument("--input", required=True, help="입력 영상 파일. 열지 못하면 종료 코드 2")
    parser.add_argument("--output", required=True, help="자막이 새겨진 결과 MP4 저장 경로")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="연산 장치. auto는 환경에서 감지된 최적 장치를 고른다(기본값)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=3,
        help="N 프레임마다 얼굴 신호를 샘플링한다(기본값 3)",
    )
    parser.add_argument("--metrics-json", default=None, help="개인별 지표 JSON 출력 경로")
    parser.add_argument("--metrics-csv", default=None, help="개인별 지표 CSV 출력 경로")
    return parser


def main(argv: list[str] | None = None, *, run_fn=run) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(run_fn(args))
    except VideoInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT_FAIL
    except ModelMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"       run `bash {DOWNLOAD_SCRIPT_HINT}` first", file=sys.stderr)
        return EXIT_MODELS_MISSING
    except RenderError as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return EXIT_RENDER_FAIL
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
