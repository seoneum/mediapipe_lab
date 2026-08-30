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
import json
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
EXIT_RIGHTS_BLOCKED = 5


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
        "--child-id",
        required=True,
        help="서명된 연구 전용 분석 동의를 확인할 로컬 아동 ID",
    )
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


def main(argv: list[str] | None = None, *, run_fn=run, rights_check_fn=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if rights_check_fn is None:
            from ondamm_rights import require_research_metrics_consent

            rights_check_fn = require_research_metrics_consent
        rights_check_fn(args.child_id)
        result = int(run_fn(args))
        if result == EXIT_OK:
            artifacts = [args.output, args.metrics_json, args.metrics_csv]
            owner_path = Path(args.output).expanduser().resolve().with_suffix(".ondamm-owner.json")
            owner_path.write_text(
                json.dumps(
                    {"child_id": args.child_id, "purpose": "research_metrics", "artifacts": [str(Path(item).expanduser().resolve()) for item in artifacts if item]},
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
        return result
    except Exception as exc:
        from ondamm_rights import RightsBlockedError

        if isinstance(exc, RightsBlockedError):
            print(f"실행 차단: {exc}", file=sys.stderr)
            return EXIT_RIGHTS_BLOCKED
        if isinstance(exc, VideoInputError):
            print(f"입력 영상 오류: {exc}", file=sys.stderr)
            return EXIT_INPUT_FAIL
        if isinstance(exc, ModelMissingError):
            print(f"필요한 모델 파일이 없습니다: {exc}", file=sys.stderr)
            print(f"먼저 `bash {DOWNLOAD_SCRIPT_HINT}`를 실행해 주세요.", file=sys.stderr)
            return EXIT_MODELS_MISSING
        if isinstance(exc, RenderError):
            print(f"결과 영상 생성에 실패했습니다: {exc}", file=sys.stderr)
            return EXIT_RENDER_FAIL
        raise
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return EXIT_UNEXPECTED


if __name__ == "__main__":
    sys.exit(main())
