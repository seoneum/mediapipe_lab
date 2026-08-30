from __future__ import annotations

"""Audit whether the frozen TCN embedding separates instructed movement types.

One embedding is produced per independent action repeat by averaging the
L2-normalized causal-window embeddings whose endpoints are in an active phase.
Distances and retrieval are evaluated within participant; LOSO folds therefore
never compare vectors produced by different encoders.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_temporal_encoder import TemporalEncoder  # noqa: E402


ACTIVE_PHASES = {"onset", "hold", "release"}
PROTOCOLS = ("upper", "lower")


@dataclass(frozen=True)
class RepeatEmbedding:
    participant: str
    protocol: str
    action: str
    repeat_idx: int
    checkpoint: str
    endpoint_count: int
    face_coverage: float
    embedding: np.ndarray


def _normalized_centroid(vectors: list[np.ndarray]) -> np.ndarray:
    centroid = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 1e-8:
        raise RuntimeError("repeat embedding centroid is zero")
    return (centroid / norm).astype(np.float32)


def extract_repeat_embeddings(
    participant: str,
    encoder: TemporalEncoder,
    checkpoint: Path,
    *,
    session: str,
    min_face_coverage: float,
) -> list[RepeatEmbedding]:
    output: list[RepeatEmbedding] = []
    names = list(encoder.spec.feature_names)
    for protocol in PROTOCOLS:
        path = ROOT / "outputs" / "micro_expression" / participant / session / f"{protocol}_signals_v4.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing v4 signals: {path}")
        frame = pd.read_csv(path)
        missing = sorted(set(names) - set(frame.columns))
        if missing:
            raise RuntimeError(f"{path} is missing encoder feature: {missing[0]}")
        action_rows = frame[frame["action"].fillna("").astype(str).ne("neutral")].copy()
        for (action, repeat_idx), trial in action_rows.groupby(["action", "repeat_idx"], sort=True):
            trial = trial.sort_values("analysis_timestamp_ms").reset_index(drop=True)
            values = trial[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            face = pd.to_numeric(trial["face_detected"], errors="coerce").fillna(0).to_numpy(dtype=float)
            phases = trial["movement_phase"].fillna("").astype(str).to_numpy()
            endpoints: list[np.ndarray] = []
            for end_idx in range(encoder.spec.sequence_length - 1, len(trial), encoder.spec.stride_frames):
                if phases[end_idx] not in ACTIVE_PHASES:
                    continue
                start_idx = end_idx - encoder.spec.sequence_length + 1
                if float(np.mean(face[start_idx : end_idx + 1])) < min_face_coverage:
                    continue
                sequence = values[start_idx : end_idx + 1]
                if not np.isfinite(sequence).all():
                    continue
                endpoints.append(encoder.encode(sequence))
            if not endpoints:
                raise RuntimeError(f"no valid active causal windows for {participant}/{action}/R{repeat_idx}")
            output.append(
                RepeatEmbedding(
                    participant=participant,
                    protocol=protocol,
                    action=str(action),
                    repeat_idx=int(repeat_idx),
                    checkpoint=checkpoint.name,
                    endpoint_count=len(endpoints),
                    face_coverage=float(np.mean(face)),
                    embedding=_normalized_centroid(endpoints),
                )
            )
    return output


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(1.0 - float(np.dot(left, right)), 0.0, 2.0))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(repeats: list[RepeatEmbedding], *, threshold: float, output_dir: Path, mode: str) -> dict[str, object]:
    pair_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    actions = sorted({item.action for item in repeats})

    for participant in sorted({item.participant for item in repeats}):
        subset = [item for item in repeats if item.participant == participant]
        for left_index, left in enumerate(subset):
            for right in subset[left_index + 1 :]:
                distance = cosine_distance(left.embedding, right.embedding)
                same = left.action == right.action
                pair_rows.append(
                    {
                        "participant": participant,
                        "left_action": left.action,
                        "left_repeat": left.repeat_idx,
                        "right_action": right.action,
                        "right_repeat": right.repeat_idx,
                        "same_action": int(same),
                        "cosine_distance": round(distance, 8),
                        "within_candidate_threshold": int(distance <= threshold),
                    }
                )

        for query in subset:
            ranked = sorted(
                (
                    (cosine_distance(query.embedding, candidate.embedding), candidate)
                    for candidate in subset
                    if candidate is not query
                ),
                key=lambda pair: (pair[0], pair[1].action, pair[1].repeat_idx),
            )
            top_actions = [candidate.action for _, candidate in ranked[:3]]
            predicted = top_actions[0]
            confusion[(query.action, predicted)] += 1
            retrieval_rows.append(
                {
                    "participant": participant,
                    "query_action": query.action,
                    "query_repeat": query.repeat_idx,
                    "predicted_action": predicted,
                    "nearest_distance": round(ranked[0][0], 8),
                    "recall_at_1": int(predicted == query.action),
                    "recall_at_3": int(query.action in top_actions),
                    "top3_actions": "|".join(top_actions),
                }
            )

    same = np.asarray([row["cosine_distance"] for row in pair_rows if row["same_action"]], dtype=float)
    different = np.asarray([row["cosine_distance"] for row in pair_rows if not row["same_action"]], dtype=float)
    recall1 = float(np.mean([row["recall_at_1"] for row in retrieval_rows]))
    recall3 = float(np.mean([row["recall_at_3"] for row in retrieval_rows]))
    same_accept = float(np.mean(same <= threshold))
    different_reject = float(np.mean(different > threshold))
    threshold_rows: list[dict[str, object]] = []
    for candidate_threshold in np.linspace(0.0, 0.5, 501):
        candidate_same_accept = float(np.mean(same <= candidate_threshold))
        candidate_different_reject = float(np.mean(different > candidate_threshold))
        threshold_rows.append(
            {
                "threshold": round(float(candidate_threshold), 4),
                "same_action_accept_rate": candidate_same_accept,
                "different_action_reject_rate": candidate_different_reject,
                "balanced_accuracy": 0.5
                * (candidate_same_accept + candidate_different_reject),
            }
        )
    best_threshold = max(
        threshold_rows,
        key=lambda row: (row["balanced_accuracy"], row["different_action_reject_rate"]),
    )
    separation_probability = float(np.mean(same[:, None] < different[None, :]))
    relation_holds = bool(float(np.mean(same)) < float(np.mean(different)))
    summary: dict[str, object] = {
        "mode": mode,
        "participants": sorted({item.participant for item in repeats}),
        "action_count": len(actions),
        "repeat_embedding_count": len(repeats),
        "embedding_dimension": int(repeats[0].embedding.size),
        "same_action_pair_count": int(same.size),
        "different_action_pair_count": int(different.size),
        "same_action_mean_cosine_distance": float(np.mean(same)),
        "same_action_median_cosine_distance": float(np.median(same)),
        "different_action_mean_cosine_distance": float(np.mean(different)),
        "different_action_median_cosine_distance": float(np.median(different)),
        "pairwise_separation_probability": separation_probability,
        "same_action_is_closer_on_average": relation_holds,
        "recall_at_1": recall1,
        "recall_at_3": recall3,
        "candidate_distance_threshold": threshold,
        "same_action_accept_rate_at_threshold": same_accept,
        "different_action_reject_rate_at_threshold": different_reject,
        "threshold_balanced_accuracy": 0.5 * (same_accept + different_reject),
        "best_diagnostic_threshold": best_threshold["threshold"],
        "best_diagnostic_same_action_accept_rate": best_threshold["same_action_accept_rate"],
        "best_diagnostic_different_action_reject_rate": best_threshold["different_action_reject_rate"],
        "best_diagnostic_balanced_accuracy": best_threshold["balanced_accuracy"],
        "interpretation": (
            "same-action distance is lower on average, but retrieval and threshold error rates must determine fitness"
            if relation_holds
            else "same-action distance is not lower on average; the current embedding is not fit for action identity prototypes"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "pairwise_distances.csv", pair_rows, list(pair_rows[0]))
    _write_csv(output_dir / "retrieval.csv", retrieval_rows, list(retrieval_rows[0]))
    _write_csv(output_dir / "threshold_sweep.csv", threshold_rows, list(threshold_rows[0]))
    repeat_rows = [
        {
            "participant": item.participant,
            "protocol": item.protocol,
            "action": item.action,
            "repeat_idx": item.repeat_idx,
            "checkpoint": item.checkpoint,
            "endpoint_count": item.endpoint_count,
            "face_coverage": round(item.face_coverage, 8),
        }
        for item in repeats
    ]
    _write_csv(output_dir / "repeat_embeddings_index.csv", repeat_rows, list(repeat_rows[0]))
    confusion_rows = []
    for actual in actions:
        row: dict[str, object] = {"actual_action": actual}
        row.update({predicted: confusion[(actual, predicted)] for predicted in actions})
        confusion_rows.append(row)
    _write_csv(output_dir / "confusion_matrix.csv", confusion_rows, ["actual_action", *actions])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Temporal embedding audit",
        "",
        f"- mode: {mode}",
        f"- repeat embeddings: {len(repeats)} ({len(actions)} actions)",
        f"- same-action cosine distance: mean {np.mean(same):.4f}, median {np.median(same):.4f}",
        f"- different-action cosine distance: mean {np.mean(different):.4f}, median {np.median(different):.4f}",
        f"- Recall@1 / Recall@3: {recall1:.3f} / {recall3:.3f}",
        f"- threshold {threshold:.2f}: same accept {same_accept:.3f}, different reject {different_reject:.3f}, balanced {summary['threshold_balanced_accuracy']:.3f}",
        f"- best diagnostic threshold: {best_threshold['threshold']:.3f} (balanced {best_threshold['balanced_accuracy']:.3f})",
        f"- conclusion: {summary['interpretation']}",
        "",
        "This is a development-data engineering diagnostic, not clinical validation.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", nargs="+", default=["p1", "p2", "p3"])
    parser.add_argument("--session", default="s01")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--loso", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--min-face-coverage", type=float, default=0.80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "micro_expression" / "embedding_audit",
    )
    args = parser.parse_args()
    if args.loso == bool(args.checkpoint):
        raise ValueError("choose exactly one of --loso or --checkpoint")
    if not 0 < args.threshold <= 2:
        raise ValueError("--threshold must be in (0, 2]")

    repeats: list[RepeatEmbedding] = []
    for participant in args.participants:
        checkpoint = (
            ROOT / "outputs" / "micro_expression" / "v4_tcn" / f"encoder_held_out_{participant}.pt"
            if args.loso
            else args.checkpoint
        )
        assert checkpoint is not None
        encoder = TemporalEncoder.from_checkpoint(checkpoint)
        repeats.extend(
            extract_repeat_embeddings(
                participant,
                encoder,
                checkpoint,
                session=args.session,
                min_face_coverage=args.min_face_coverage,
            )
        )
    mode = "loso-held-out-per-participant" if args.loso else f"single-checkpoint:{args.checkpoint.name}"
    summary = evaluate(repeats, threshold=args.threshold, output_dir=args.output_dir, mode=mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
