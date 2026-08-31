from __future__ import annotations

"""
Personalized child metric head.

Pipeline
--------
79D facial temporal sequence
    -> frozen product TCN
    -> 64D L2-normalized temporal embedding
    -> trainable Linear(64 -> 32)
    -> L2 normalize
    -> personalized metric embedding

Split
-----
R1, R2, R4, R5 : train
R3             : validation / epoch + threshold selection
R6             : held-out historical test

s03 is NEVER loaded here.

Training objective
------------------
Supervised contrastive loss:
    same instructed action -> pull together
    different action       -> push apart

plus geometry preservation:
    preserve pairwise cosine structure of the pretrained embedding
    so the metric head does not destroy the space needed for later
    unknown-pattern discovery.
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
SCRIPTS_DIR = ROOT / "scripts"

for directory in (APP_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


from ondamm_temporal_encoder import TemporalEncoder  # noqa: E402
import train_v4_tcn as tcn  # noqa: E402
import run_v4_modality_ablation as ab  # noqa: E402


ACTIVE_PHASES = {
    "onset",
    "hold",
    "release",
}

PROTOCOLS = (
    "upper",
    "lower",
)

TRAIN_REPEATS = {
    1,
    2,
    4,
    5,
}

VAL_REPEATS = {
    3,
}

TEST_REPEATS = {
    6,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but not available"
            )
        return torch.device("mps")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available"
            )
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class MetricHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )

        nn.init.orthogonal_(
            self.projection.weight
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        z = self.projection(x)

        return F.normalize(
            z,
            p=2,
            dim=1,
            eps=1e-8,
        )


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("z must be 2D")

    if labels.ndim != 1:
        raise ValueError("labels must be 1D")

    n = z.shape[0]

    if n < 2:
        raise RuntimeError(
            "contrastive batch is too small"
        )

    logits = (
        z @ z.T
    ) / temperature

    logits = logits - logits.max(
        dim=1,
        keepdim=True,
    ).values.detach()

    eye = torch.eye(
        n,
        dtype=torch.bool,
        device=z.device,
    )

    same = (
        labels[:, None]
        == labels[None, :]
    )

    positive_mask = (
        same
        & ~eye
    )

    denominator_mask = ~eye

    exp_logits = (
        torch.exp(logits)
        * denominator_mask.float()
    )

    log_prob = (
        logits
        - torch.log(
            exp_logits.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-12
        )
    )

    positive_count = positive_mask.sum(
        dim=1
    )

    valid = positive_count > 0

    if not bool(valid.any()):
        raise RuntimeError(
            "batch contains no positive pairs"
        )

    mean_log_prob_positive = (
        (
            log_prob
            * positive_mask.float()
        ).sum(dim=1)
        / positive_count.clamp_min(1)
    )

    return -mean_log_prob_positive[
        valid
    ].mean()


def geometry_preservation_loss(
    original: torch.Tensor,
    projected: torch.Tensor,
) -> torch.Tensor:
    """
    Preserve pairwise cosine relationships from frozen 64D space.

    original embeddings are already L2-normalized.
    projected embeddings are also L2-normalized.
    """

    original = F.normalize(
        original,
        p=2,
        dim=1,
        eps=1e-8,
    )

    original_similarity = (
        original
        @ original.T
    ).detach()

    projected_similarity = (
        projected
        @ projected.T
    )

    n = original.shape[0]

    mask = ~torch.eye(
        n,
        dtype=torch.bool,
        device=original.device,
    )

    return F.mse_loss(
        projected_similarity[mask],
        original_similarity[mask],
    )


def build_child_normalization(
    child_id: str,
    sessions: list[str],
    feature_names: list[str],
    calibration_start: float,
    calibration_end: float,
):
    frames = []

    for session in sessions:
        frame = tcn.load_subject(
            child_id,
            session,
            calibration_start,
            calibration_end,
        )

        frame["session_id"] = session
        frames.append(frame)

    historical = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    calibration = ab.calibration_frame(
        historical
    )

    stats = ab.robust_stats(
        calibration,
        feature_names,
    )

    center = np.asarray(
        [
            float(stats.center[name])
            for name in feature_names
        ],
        dtype=np.float32,
    )

    scale = np.asarray(
        [
            float(stats.scale[name])
            for name in feature_names
        ],
        dtype=np.float32,
    )

    if (
        not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise RuntimeError(
            "invalid child normalization statistics"
        )

    return center, scale, len(calibration)


def extract_window_embeddings(
    child_id: str,
    sessions: list[str],
    encoder: TemporalEncoder,
    *,
    min_face_coverage: float,
):
    embeddings: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []

    features = list(
        encoder.spec.feature_names
    )

    sequence_length = (
        encoder.spec.sequence_length
    )

    stride = (
        encoder.spec.stride_frames
    )

    for session in sessions:
        for protocol in PROTOCOLS:
            path = (
                ROOT
                / "outputs"
                / "micro_expression"
                / child_id
                / session
                / f"{protocol}_signals_v4.csv"
            )

            if not path.is_file():
                raise FileNotFoundError(
                    f"missing V4 signals: {path}"
                )

            frame = pd.read_csv(path)

            missing = sorted(
                set(features)
                - set(frame.columns)
            )

            if missing:
                raise RuntimeError(
                    f"{path} missing feature: "
                    f"{missing[0]}"
                )

            action_rows = frame[
                frame["action"]
                .fillna("")
                .astype(str)
                .ne("neutral")
            ].copy()

            for (
                action,
                repeat_idx,
            ), trial in action_rows.groupby(
                [
                    "action",
                    "repeat_idx",
                ],
                sort=True,
            ):
                trial = (
                    trial
                    .sort_values(
                        "analysis_timestamp_ms"
                    )
                    .reset_index(drop=True)
                )

                values = (
                    trial[features]
                    .apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    .to_numpy(
                        dtype=np.float32
                    )
                )

                face = (
                    pd.to_numeric(
                        trial["face_detected"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .to_numpy(
                        dtype=np.float32
                    )
                )

                phases = (
                    trial["movement_phase"]
                    .fillna("")
                    .astype(str)
                    .to_numpy()
                )

                timestamps = (
                    pd.to_numeric(
                        trial[
                            "analysis_timestamp_ms"
                        ],
                        errors="coerce",
                    )
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                for end_idx in range(
                    sequence_length - 1,
                    len(trial),
                    stride,
                ):
                    if (
                        phases[end_idx]
                        not in ACTIVE_PHASES
                    ):
                        continue

                    start_idx = (
                        end_idx
                        - sequence_length
                        + 1
                    )

                    face_coverage = float(
                        np.mean(
                            face[
                                start_idx:
                                end_idx + 1
                            ]
                        )
                    )

                    if (
                        face_coverage
                        < min_face_coverage
                    ):
                        continue

                    sequence = values[
                        start_idx:
                        end_idx + 1
                    ]

                    if not np.isfinite(
                        sequence
                    ).all():
                        continue

                    embedding = encoder.encode(
                        sequence
                    ).astype(
                        np.float32
                    )

                    embeddings.append(
                        embedding
                    )

                    metadata.append({
                        "session": session,
                        "protocol": protocol,
                        "action": str(action),
                        "repeat_idx": int(
                            repeat_idx
                        ),
                        "endpoint_idx": int(
                            end_idx
                        ),
                        "analysis_timestamp_ms":
                            float(
                                timestamps[
                                    end_idx
                                ]
                            ),
                        "face_coverage":
                            face_coverage,
                    })

    if not embeddings:
        raise RuntimeError(
            "no temporal embeddings extracted"
        )

    matrix = np.stack(
        embeddings
    ).astype(
        np.float32
    )

    metadata_frame = pd.DataFrame(
        metadata
    )

    return matrix, metadata_frame


def encode_actions(
    metadata: pd.DataFrame,
):
    actions = sorted(
        metadata["action"]
        .astype(str)
        .unique()
        .tolist()
    )

    mapping = {
        action: index
        for index, action
        in enumerate(actions)
    }

    labels = np.asarray(
        [
            mapping[action]
            for action
            in metadata["action"].astype(str)
        ],
        dtype=np.int64,
    )

    return actions, mapping, labels


def split_mask(
    metadata: pd.DataFrame,
    repeats: set[int],
):
    repeat_idx = pd.to_numeric(
        metadata["repeat_idx"],
        errors="raise",
    ).astype(int)

    return repeat_idx.isin(
        repeats
    ).to_numpy()


def balanced_batch_indices(
    labels: np.ndarray,
    sessions: np.ndarray,
    *,
    samples_per_class_session: int,
    rng: np.random.Generator,
):
    selected: list[int] = []

    unique_labels = sorted(
        np.unique(labels).tolist()
    )

    unique_sessions = sorted(
        np.unique(sessions).tolist()
    )

    for label in unique_labels:
        for session in unique_sessions:
            pool = np.flatnonzero(
                (labels == label)
                & (sessions == session)
            )

            if len(pool) == 0:
                raise RuntimeError(
                    "missing train samples for "
                    f"class={label}, "
                    f"session={session}"
                )

            replace = (
                len(pool)
                < samples_per_class_session
            )

            chosen = rng.choice(
                pool,
                size=samples_per_class_session,
                replace=replace,
            )

            selected.extend(
                chosen.tolist()
            )

    selected = np.asarray(
        selected,
        dtype=np.int64,
    )

    rng.shuffle(selected)

    return selected


def project_numpy(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
):
    output: list[np.ndarray] = []

    model.eval()

    with torch.inference_mode():
        for start in range(
            0,
            len(x),
            batch_size,
        ):
            batch = torch.as_tensor(
                x[
                    start:
                    start + batch_size
                ],
                dtype=torch.float32,
                device=device,
            )

            z = model(
                batch
            )

            output.append(
                z.detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(
        output,
        axis=0,
    ).astype(
        np.float32
    )


def normalized_centroid(
    vectors: np.ndarray,
):
    centroid = np.mean(
        vectors,
        axis=0,
    )

    norm = float(
        np.linalg.norm(
            centroid
        )
    )

    if norm <= 1e-8:
        raise RuntimeError(
            "zero centroid"
        )

    return (
        centroid
        / norm
    ).astype(
        np.float32
    )


def cosine_distance(
    a: np.ndarray,
    b: np.ndarray,
):
    return float(
        np.clip(
            1.0
            - float(
                np.dot(
                    a,
                    b,
                )
            ),
            0.0,
            2.0,
        )
    )


def build_repeat_centroids(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
):
    items = []

    work = metadata.reset_index(
        drop=True
    )

    groups = work.groupby(
        [
            "session",
            "action",
            "repeat_idx",
        ],
        sort=True,
    ).groups

    for (
        session,
        action,
        repeat_idx,
    ), indices in groups.items():
        idx = np.asarray(
            list(indices),
            dtype=np.int64,
        )

        centroid = normalized_centroid(
            embeddings[idx]
        )

        items.append({
            "session": str(session),
            "action": str(action),
            "repeat_idx": int(
                repeat_idx
            ),
            "window_count": int(
                len(idx)
            ),
            "embedding": centroid,
        })

    return items


def cross_session_metrics(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
):
    repeats = build_repeat_centroids(
        embeddings,
        metadata,
    )

    session_names = sorted({
        item["session"]
        for item in repeats
    })

    if session_names != [
        "s01",
        "s02",
    ]:
        raise RuntimeError(
            "cross-session audit expects "
            "exactly s01 and s02"
        )

    left = [
        item
        for item in repeats
        if item["session"] == "s01"
    ]

    right = [
        item
        for item in repeats
        if item["session"] == "s02"
    ]

    pair_rows = []
    same = []
    different = []

    for a in left:
        for b in right:
            distance = cosine_distance(
                a["embedding"],
                b["embedding"],
            )

            same_action = (
                a["action"]
                == b["action"]
            )

            pair_rows.append({
                "left_session": "s01",
                "left_action": a["action"],
                "left_repeat":
                    a["repeat_idx"],
                "right_session": "s02",
                "right_action":
                    b["action"],
                "right_repeat":
                    b["repeat_idx"],
                "same_action":
                    int(same_action),
                "cosine_distance":
                    distance,
            })

            if same_action:
                same.append(
                    distance
                )
            else:
                different.append(
                    distance
                )

    same = np.asarray(
        same,
        dtype=float,
    )

    different = np.asarray(
        different,
        dtype=float,
    )

    if (
        len(same) == 0
        or len(different) == 0
    ):
        raise RuntimeError(
            "invalid pairwise evaluation"
        )

    retrieval_rows = []

    def retrieve(
        query_items,
        reference_items,
    ):
        for query in query_items:
            ranked = sorted(
                (
                    (
                        cosine_distance(
                            query["embedding"],
                            reference[
                                "embedding"
                            ],
                        ),
                        reference,
                    )
                    for reference
                    in reference_items
                ),
                key=lambda pair: (
                    pair[0],
                    pair[1]["action"],
                ),
            )

            top3 = ranked[:3]

            retrieval_rows.append({
                "query_session":
                    query["session"],
                "query_action":
                    query["action"],
                "query_repeat":
                    query["repeat_idx"],
                "reference_session":
                    top3[0][1]["session"],
                "nearest_action":
                    top3[0][1]["action"],
                "nearest_distance":
                    top3[0][0],
                "recall_at_1":
                    int(
                        top3[0][1]["action"]
                        == query["action"]
                    ),
                "recall_at_3":
                    int(
                        any(
                            candidate["action"]
                            == query["action"]
                            for _, candidate
                            in top3
                        )
                    ),
                "top3_actions":
                    "|".join(
                        candidate["action"]
                        for _, candidate
                        in top3
                    ),
            })

    retrieve(
        left,
        right,
    )

    retrieve(
        right,
        left,
    )

    recall1 = float(
        np.mean([
            row["recall_at_1"]
            for row
            in retrieval_rows
        ])
    )

    recall3 = float(
        np.mean([
            row["recall_at_3"]
            for row
            in retrieval_rows
        ])
    )

    separation_probability = float(
        np.mean(
            same[:, None]
            < different[None, :]
        )
    )

    action_rows = []

    actions = sorted({
        item["action"]
        for item in repeats
    })

    # Keep the same direction as the previous
    # s01 -> s02 centroid audit.
    for action in actions:
        left_match = [
            item
            for item in left
            if item["action"] == action
        ]

        right_match = [
            item
            for item in right
            if item["action"] == action
        ]

        if (
            len(left_match) != 1
            or len(right_match) != 1
        ):
            raise RuntimeError(
                "expected one held-out repeat "
                "per action/session"
            )

        own_distance = cosine_distance(
            left_match[0]["embedding"],
            right_match[0]["embedding"],
        )

        competitors = []

        for other in right:
            if (
                other["action"]
                == action
            ):
                continue

            competitors.append(
                (
                    cosine_distance(
                        left_match[0][
                            "embedding"
                        ],
                        other["embedding"],
                    ),
                    other["action"],
                )
            )

        competitors.sort(
            key=lambda item: item[0]
        )

        nearest_distance, nearest_action = (
            competitors[0]
        )

        action_rows.append({
            "action": action,
            "same_action_distance":
                own_distance,
            "nearest_different_action":
                nearest_action,
            "nearest_different_distance":
                nearest_distance,
            "margin":
                nearest_distance
                - own_distance,
            "correct_relation":
                int(
                    own_distance
                    < nearest_distance
                ),
        })

    relation_rate = float(
        np.mean([
            row[
                "correct_relation"
            ]
            for row in action_rows
        ])
    )

    metrics = {
        "repeat_centroids":
            int(len(repeats)),
        "same_action_pair_count":
            int(len(same)),
        "different_action_pair_count":
            int(len(different)),
        "same_action_mean_distance":
            float(
                np.mean(same)
            ),
        "same_action_median_distance":
            float(
                np.median(same)
            ),
        "different_action_mean_distance":
            float(
                np.mean(different)
            ),
        "different_action_median_distance":
            float(
                np.median(different)
            ),
        "same_action_closer_on_average":
            bool(
                np.mean(same)
                < np.mean(different)
            ),
        "separation_probability":
            separation_probability,
        "recall_at_1":
            recall1,
        "recall_at_3":
            recall3,
        "centroid_relation_rate":
            relation_rate,
    }

    return {
        "metrics": metrics,
        "same": same,
        "different": different,
        "pairs": pd.DataFrame(
            pair_rows
        ),
        "retrieval": pd.DataFrame(
            retrieval_rows
        ),
        "actions": pd.DataFrame(
            action_rows
        ),
    }


def choose_threshold(
    same: np.ndarray,
    different: np.ndarray,
):
    best = None

    rows = []

    for threshold in np.linspace(
        0.0,
        0.5,
        501,
    ):
        same_accept = float(
            np.mean(
                same <= threshold
            )
        )

        different_reject = float(
            np.mean(
                different > threshold
            )
        )

        balanced = (
            same_accept
            + different_reject
        ) / 2.0

        row = {
            "threshold":
                float(threshold),
            "same_accept":
                same_accept,
            "different_reject":
                different_reject,
            "balanced_accuracy":
                balanced,
        }

        rows.append(
            row
        )

        key = (
            balanced,
            different_reject,
            -threshold,
        )

        if (
            best is None
            or key > best[0]
        ):
            best = (
                key,
                row,
            )

    assert best is not None

    return (
        best[1],
        pd.DataFrame(rows),
    )


def threshold_metrics(
    same: np.ndarray,
    different: np.ndarray,
    threshold: float,
):
    same_accept = float(
        np.mean(
            same <= threshold
        )
    )

    different_reject = float(
        np.mean(
            different > threshold
        )
    )

    return {
        "threshold":
            float(threshold),
        "same_accept":
            same_accept,
        "different_reject":
            different_reject,
        "balanced_accuracy":
            (
                same_accept
                + different_reject
            )
            / 2.0,
    }


def evaluation_key(
    metrics: dict[str, object],
):
    """
    Epoch selection is validation-only.

    Priority:
      1. Recall@1
      2. centroid relation
      3. separation probability
      4. Recall@3
    """

    return (
        float(
            metrics["recall_at_1"]
        ),
        float(
            metrics[
                "centroid_relation_rate"
            ]
        ),
        float(
            metrics[
                "separation_probability"
            ]
        ),
        float(
            metrics["recall_at_3"]
        ),
    )


def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    metadata: pd.DataFrame,
    device: torch.device,
):
    projected = project_numpy(
        model,
        x,
        device,
    )

    return cross_session_metrics(
        projected,
        metadata,
    )


def evaluate_raw(
    x: np.ndarray,
    metadata: pd.DataFrame,
):
    normalized = (
        x
        / np.clip(
            np.linalg.norm(
                x,
                axis=1,
                keepdims=True,
            ),
            1e-8,
            None,
        )
    )

    return cross_session_metrics(
        normalized,
        metadata,
    )


def print_metrics(
    title: str,
    result,
):
    m = result["metrics"]

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    print(
        "same-action distance      :",
        f"mean={m['same_action_mean_distance']:.4f}",
        f"median={m['same_action_median_distance']:.4f}",
    )

    print(
        "different-action distance :",
        f"mean={m['different_action_mean_distance']:.4f}",
        f"median={m['different_action_median_distance']:.4f}",
    )

    print(
        "same closer on average    :",
        m[
            "same_action_closer_on_average"
        ],
    )

    print(
        "separation probability    :",
        f"{m['separation_probability']:.3f}",
    )

    print(
        "Recall@1 / Recall@3       :",
        f"{m['recall_at_1']:.3f}",
        "/",
        f"{m['recall_at_3']:.3f}",
    )

    print(
        "action centroid relation  :",
        f"{m['centroid_relation_rate']:.3f}",
    )

    print()
    print("Per-action:")

    for row in result[
        "actions"
    ].to_dict(
        orient="records"
    ):
        print(
            f"{row['action']:14s}",
            f"same={row['same_action_distance']:.4f}",
            f"nearest={row['nearest_different_action']:14s}",
            f"other={row['nearest_different_distance']:.4f}",
            f"margin={row['margin']:+.4f}",
            (
                "PASS"
                if row[
                    "correct_relation"
                ]
                else "FAIL"
            ),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train child-specific metric head "
            "on frozen temporal encoder"
        )
    )

    parser.add_argument(
        "--child-id",
        required=True,
    )

    parser.add_argument(
        "--sessions",
        nargs="+",
        default=[
            "s01",
            "s02",
        ],
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "micro_expression"
            / "v4_tcn"
            / "encoder_product.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--projection-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--samples-per-class-session",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--preserve-weight",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--min-face-coverage",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--calibration-start",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--calibration-end",
        type=float,
        default=7.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.sessions != [
        "s01",
        "s02",
    ]:
        raise ValueError(
            "this experiment is intentionally "
            "locked to s01 + s02"
        )

    seed_everything(
        args.seed
    )

    device = choose_device(
        args.device
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            ROOT
            / "outputs"
            / "micro_expression"
            / "children"
            / args.child_id
            / "metric_head"
        ).resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("device:", device)
    print(
        "loading frozen encoder:",
        args.checkpoint,
    )

    # CPU loading makes child normalization assignment
    # explicit. encode() itself remains frozen.
    encoder = TemporalEncoder.from_checkpoint(
        args.checkpoint,
        device="cpu",
    )

    feature_names = list(
        encoder.spec.feature_names
    )

    print(
        "TCN features:",
        len(feature_names),
    )
    print(
        "TCN sequence:",
        encoder.spec.sequence_length,
    )
    print(
        "TCN embedding:",
        encoder.spec.embedding_dim,
    )

    child_center, child_scale, calibration_rows = (
        build_child_normalization(
            args.child_id,
            args.sessions,
            feature_names,
            args.calibration_start,
            args.calibration_end,
        )
    )

    encoder.mean = child_center
    encoder.std = child_scale

    print(
        "child calibration rows:",
        calibration_rows,
    )

    print(
        "extracting frozen TCN windows..."
    )

    x, metadata = extract_window_embeddings(
        args.child_id,
        args.sessions,
        encoder,
        min_face_coverage=
            args.min_face_coverage,
    )

    actions, action_to_index, labels = (
        encode_actions(
            metadata
        )
    )

    print(
        "window embeddings:",
        x.shape,
    )
    print(
        "actions:",
        len(actions),
        actions,
    )

    train_mask = split_mask(
        metadata,
        TRAIN_REPEATS,
    )

    val_mask = split_mask(
        metadata,
        VAL_REPEATS,
    )

    test_mask = split_mask(
        metadata,
        TEST_REPEATS,
    )

    if np.any(
        train_mask
        & val_mask
    ):
        raise RuntimeError(
            "train/val leakage"
        )

    if np.any(
        train_mask
        & test_mask
    ):
        raise RuntimeError(
            "train/test leakage"
        )

    if np.any(
        val_mask
        & test_mask
    ):
        raise RuntimeError(
            "val/test leakage"
        )

    x_train = x[
        train_mask
    ]

    y_train = labels[
        train_mask
    ]

    meta_train = (
        metadata.loc[
            train_mask
        ]
        .reset_index(drop=True)
    )

    x_val = x[
        val_mask
    ]

    meta_val = (
        metadata.loc[
            val_mask
        ]
        .reset_index(drop=True)
    )

    x_test = x[
        test_mask
    ]

    meta_test = (
        metadata.loc[
            test_mask
        ]
        .reset_index(drop=True)
    )

    print()
    print(
        "TRAIN windows:",
        len(x_train),
        "R1,R2,R4,R5",
    )
    print(
        "VAL windows  :",
        len(x_val),
        "R3",
    )
    print(
        "TEST windows :",
        len(x_test),
        "R6",
    )

    # Baseline before the metric head.
    baseline_val = evaluate_raw(
        x_val,
        meta_val,
    )

    baseline_test = evaluate_raw(
        x_test,
        meta_test,
    )

    print_metrics(
        "BASELINE R3 / CHILD NORMALIZATION",
        baseline_val,
    )

    print_metrics(
        "BASELINE R6 / CHILD NORMALIZATION",
        baseline_test,
    )

    model = MetricHead(
        input_dim=x.shape[1],
        output_dim=args.projection_dim,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    sessions_train = (
        meta_train["session"]
        .astype(str)
        .to_numpy()
    )

    rng = np.random.default_rng(
        args.seed
    )

    best_state = None
    best_epoch = None
    best_key = None
    best_val_result = None

    patience_left = (
        args.patience
    )

    history_rows = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        epoch_losses = []
        epoch_supcon = []
        epoch_preserve = []

        for _ in range(
            args.steps_per_epoch
        ):
            batch_idx = (
                balanced_batch_indices(
                    y_train,
                    sessions_train,
                    samples_per_class_session=
                        args.samples_per_class_session,
                    rng=rng,
                )
            )

            xb = torch.as_tensor(
                x_train[
                    batch_idx
                ],
                dtype=torch.float32,
                device=device,
            )

            yb = torch.as_tensor(
                y_train[
                    batch_idx
                ],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            z = model(
                xb
            )

            supcon = (
                supervised_contrastive_loss(
                    z,
                    yb,
                    args.temperature,
                )
            )

            preserve = (
                geometry_preservation_loss(
                    xb,
                    z,
                )
            )

            loss = (
                supcon
                + args.preserve_weight
                * preserve
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

            epoch_supcon.append(
                float(
                    supcon.detach()
                    .cpu()
                )
            )

            epoch_preserve.append(
                float(
                    preserve.detach()
                    .cpu()
                )
            )

        val_result = evaluate_model(
            model,
            x_val,
            meta_val,
            device,
        )

        val_metrics = (
            val_result["metrics"]
        )

        key = evaluation_key(
            val_metrics
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss":
                float(
                    np.mean(
                        epoch_losses
                    )
                ),
            "supcon_loss":
                float(
                    np.mean(
                        epoch_supcon
                    )
                ),
            "preserve_loss":
                float(
                    np.mean(
                        epoch_preserve
                    )
                ),
            "val_recall_at_1":
                val_metrics[
                    "recall_at_1"
                ],
            "val_recall_at_3":
                val_metrics[
                    "recall_at_3"
                ],
            "val_centroid_relation":
                val_metrics[
                    "centroid_relation_rate"
                ],
            "val_separation_probability":
                val_metrics[
                    "separation_probability"
                ],
            "val_same_mean":
                val_metrics[
                    "same_action_mean_distance"
                ],
            "val_different_mean":
                val_metrics[
                    "different_action_mean_distance"
                ],
        })

        improved = (
            best_key is None
            or key > best_key
        )

        if improved:
            best_key = key
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_val_result = (
                val_result
            )

            patience_left = (
                args.patience
            )
        else:
            patience_left -= 1

        if (
            epoch == 1
            or epoch % 10 == 0
            or improved
        ):
            print(
                f"epoch={epoch:03d}",
                f"loss={np.mean(epoch_losses):.4f}",
                f"R1={val_metrics['recall_at_1']:.3f}",
                f"R3={val_metrics['recall_at_3']:.3f}",
                f"relation={val_metrics['centroid_relation_rate']:.3f}",
                f"sep={val_metrics['separation_probability']:.3f}",
                (
                    "*"
                    if improved
                    else ""
                ),
            )

        if patience_left <= 0:
            print(
                "early stopping"
            )
            break

    if (
        best_state is None
        or best_epoch is None
        or best_val_result is None
    ):
        raise RuntimeError(
            "failed to select metric head"
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    # Recompute validation using selected epoch.
    val_result = evaluate_model(
        model,
        x_val,
        meta_val,
        device,
    )

    # Threshold is selected ONLY from R3.
    best_threshold, threshold_sweep = (
        choose_threshold(
            val_result["same"],
            val_result["different"],
        )
    )

    frozen_threshold = float(
        best_threshold[
            "threshold"
        ]
    )

    val_threshold_metrics = (
        threshold_metrics(
            val_result["same"],
            val_result["different"],
            frozen_threshold,
        )
    )

    # R6 is touched only after model epoch and
    # threshold have been fixed.
    test_result = evaluate_model(
        model,
        x_test,
        meta_test,
        device,
    )

    test_threshold_metrics = (
        threshold_metrics(
            test_result["same"],
            test_result["different"],
            frozen_threshold,
        )
    )

    print_metrics(
        "METRIC HEAD R3 / VALIDATION",
        val_result,
    )

    print()
    print(
        "R3 selected threshold:",
        f"{frozen_threshold:.3f}",
    )
    print(
        "R3 threshold balanced:",
        f"{val_threshold_metrics['balanced_accuracy']:.3f}",
    )

    print_metrics(
        "METRIC HEAD R6 / HELD-OUT TEST",
        test_result,
    )

    print()
    print(
        "R6 frozen threshold:",
        f"{frozen_threshold:.3f}",
    )
    print(
        "R6 same accept:",
        f"{test_threshold_metrics['same_accept']:.3f}",
    )
    print(
        "R6 different reject:",
        f"{test_threshold_metrics['different_reject']:.3f}",
    )
    print(
        "R6 threshold balanced:",
        f"{test_threshold_metrics['balanced_accuracy']:.3f}",
    )

    test_m = test_result[
        "metrics"
    ]

    gate = {
        "recall_at_1": (
            float(
                test_m[
                    "recall_at_1"
                ]
            )
            >= 0.70
        ),
        "recall_at_3": (
            float(
                test_m[
                    "recall_at_3"
                ]
            )
            >= 0.85
        ),
        "centroid_relation": (
            float(
                test_m[
                    "centroid_relation_rate"
                ]
            )
            >= 0.80
        ),
        "separation_probability": (
            float(
                test_m[
                    "separation_probability"
                ]
            )
            >= 0.85
        ),
    }

    gate_pass = all(
        gate.values()
    )

    print()
    print("=" * 72)
    print("R6 ENGINEERING GATE")
    print("=" * 72)

    for name, passed in gate.items():
        print(
            f"{name:28s}",
            (
                "PASS"
                if passed
                else "FAIL"
            ),
        )

    print()
    print(
        "FINAL GATE:",
        (
            "PASS"
            if gate_pass
            else "FAIL"
        ),
    )

    history = pd.DataFrame(
        history_rows
    )

    history.to_csv(
        output_dir
        / "epoch_history.csv",
        index=False,
    )

    threshold_sweep.to_csv(
        output_dir
        / "validation_threshold_sweep.csv",
        index=False,
    )

    val_result["pairs"].to_csv(
        output_dir
        / "validation_r3_pairs.csv",
        index=False,
    )

    val_result[
        "retrieval"
    ].to_csv(
        output_dir
        / "validation_r3_retrieval.csv",
        index=False,
    )

    val_result[
        "actions"
    ].to_csv(
        output_dir
        / "validation_r3_actions.csv",
        index=False,
    )

    test_result["pairs"].to_csv(
        output_dir
        / "heldout_r6_pairs.csv",
        index=False,
    )

    test_result[
        "retrieval"
    ].to_csv(
        output_dir
        / "heldout_r6_retrieval.csv",
        index=False,
    )

    test_result[
        "actions"
    ].to_csv(
        output_dir
        / "heldout_r6_actions.csv",
        index=False,
    )

    metadata.to_csv(
        output_dir
        / "window_embedding_index.csv",
        index=False,
    )

    product_digest = hashlib.sha256(
        args.checkpoint
        .expanduser()
        .resolve()
        .read_bytes()
    ).hexdigest()

    checkpoint_path = (
        output_dir
        / "metric_head.pt"
    )

    payload = {
        "schema_version": 1,
        "checkpoint_role":
            "child-metric-head",
        "child_id":
            args.child_id,
        "training_sessions":
            args.sessions,
        "future_session":
            "s03",
        "future_session_used":
            False,
        "source_encoder_path":
            str(
                args.checkpoint
                .expanduser()
                .resolve()
            ),
        "source_encoder_sha256":
            product_digest,
        "source_embedding_dim":
            int(
                encoder.spec.embedding_dim
            ),
        "projection_dim":
            int(
                args.projection_dim
            ),
        "train_repeats":
            sorted(
                TRAIN_REPEATS
            ),
        "validation_repeats":
            sorted(
                VAL_REPEATS
            ),
        "heldout_test_repeats":
            sorted(
                TEST_REPEATS
            ),
        "best_epoch":
            int(
                best_epoch
            ),
        "validation_selected_threshold":
            frozen_threshold,
        "child_normalization_mean":
            child_center.tolist(),
        "child_normalization_std":
            child_scale.tolist(),
        "feature_names":
            feature_names,
        "metric_head_state_dict":
            {
                key:
                    value.detach()
                    .cpu()
                for key, value
                in model.state_dict().items()
            },
        "training_objective": {
            "metric":
                "supervised_contrastive",
            "temperature":
                args.temperature,
            "geometry_preservation_weight":
                args.preserve_weight,
            "cross_session_balanced_batches":
                True,
        },
    }

    torch.save(
        payload,
        checkpoint_path,
    )

    metric_digest = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    baseline_summary = {
        "validation_r3":
            baseline_val["metrics"],
        "heldout_r6":
            baseline_test["metrics"],
    }

    final_summary = {
        "child_id":
            args.child_id,
        "sessions":
            args.sessions,
        "s03_used":
            False,
        "best_epoch":
            int(best_epoch),
        "projection_dim":
            args.projection_dim,
        "metric_head_sha256":
            metric_digest,
        "validation_threshold":
            frozen_threshold,
        "baseline":
            baseline_summary,
        "metric_head": {
            "validation_r3":
                val_result["metrics"],
            "validation_threshold_metrics":
                val_threshold_metrics,
            "heldout_r6":
                test_result["metrics"],
            "heldout_r6_threshold_metrics":
                test_threshold_metrics,
        },
        "engineering_gate":
            gate,
        "engineering_gate_pass":
            gate_pass,
    }

    (
        output_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            final_summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    config = {
        "architecture": {
            "frozen_encoder_dim":
                encoder.spec.embedding_dim,
            "metric_projection_dim":
                args.projection_dim,
            "metric_head":
                "single bias-free linear layer + L2 normalization",
        },
        "data_split": {
            "train":
                sorted(
                    TRAIN_REPEATS
                ),
            "validation":
                sorted(
                    VAL_REPEATS
                ),
            "heldout_test":
                sorted(
                    TEST_REPEATS
                ),
            "future_s03_used":
                False,
        },
        "normalization": (
            "child s01+s02 historical "
            "calibration robust center/scale"
        ),
        "metric_threshold": {
            "value":
                frozen_threshold,
            "selected_from":
                "R3 only",
            "R6_used_for_selection":
                False,
        },
    }

    (
        output_dir
        / "config.json"
    ).write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "best epoch:",
        best_epoch,
    )

    print(
        "metric head:",
        checkpoint_path,
    )

    print(
        "sha256:",
        metric_digest,
    )

    print(
        "summary:",
        output_dir
        / "summary.json",
    )


if __name__ == "__main__":
    main()
