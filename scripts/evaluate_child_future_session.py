from __future__ import annotations

"""Evaluate one personalized child model on a later session of that child."""

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_child_temporal_evaluation import (  # noqa: E402
    GroundTruthEvent,
    TemporalDetection,
    evaluate_future_session,
)


def load_csv(path: Path) -> list[dict[str, str]]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing evaluation CSV: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Within-child future-session temporal pattern evaluation",
    )
    parser.add_argument("--child-id", required=True)
    parser.add_argument("--future-session", required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--session-duration-seconds", type=float, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--min-unknown-repetitions", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Default: outputs/micro_expression/children/<child-id>/"
            "future-session/<session>/metrics.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ground_truth = [GroundTruthEvent.from_mapping(row) for row in load_csv(args.ground_truth)]
    detections = [TemporalDetection.from_mapping(row) for row in load_csv(args.detections)]
    metrics = evaluate_future_session(
        ground_truth,
        detections,
        child_id=args.child_id,
        future_session=args.future_session,
        session_duration_seconds=args.session_duration_seconds,
        iou_threshold=args.iou_threshold,
        min_unknown_repetitions=args.min_unknown_repetitions,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else (
            ROOT
            / "outputs"
            / "micro_expression"
            / "children"
            / args.child_id
            / "future-session"
            / args.future_session
            / "metrics.json"
        ).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"objective: {metrics['objective']}")
    print(f"child_id: {metrics['child_id']}")
    print(f"future_session: {metrics['future_session']}")
    print(f"metrics: {output}")


if __name__ == "__main__":
    main()
