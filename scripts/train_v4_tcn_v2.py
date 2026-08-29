from __future__ import annotations

"""
V4 causal TCN baseline for person-held-out facial-action detection.

Core representation
-------------------
- MediaPipe blendshapes: bs_*
- label-free absolute canonical geometry: geom_abs_*
- generic facial motion: motion_*

Deliberately excluded from the first TCN:
- DINO
- yaw/pitch/roll
- gaze proxies
- blink
- face_ratio
- v4 session-relative geom_delta_*/geom_state_*/dino_change_*

Reason:
The nuisance audit showed that pitch, vertical gaze, blink, and face ratio can
co-vary with instructed actions. The first temporal baseline therefore learns
from facial representation only, while control recordings are used as hard
negative examples.

Evaluation
----------
Outer LOSO:
    train P2+P3 -> test P1
    train P1+P3 -> test P2
    train P1+P2 -> test P3

Within outer-train subjects:
- action trials are split by repeat at the TRIAL level, not by frames
- validation is used for early stopping and threshold selection
- held-out subject is untouched until final evaluation

Sequence
--------
Causal window ending at time t:
    X[t-L+1 : t] -> y[t]

Default:
    60 frames ~= 2 s at 30 FPS
    training/eval stride = 5 frames

The sequence may cross PRE -> ONSET -> HOLD -> RELEASE -> POST within the same
action trial, which is intentional. It never crosses action-repeat boundaries
or recording boundaries.

Outputs
-------
outputs/micro_expression/v4_tcn/
    fold_results.csv
    epoch_history.csv
    predictions.csv
    summary.csv
    config.json
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required."
    ) from exc

try:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required.\n"
        "Install with:\n"
        "  uv pip install --python .venv/bin/python scikit-learn"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPTS_DIR),
    )

import run_v4_modality_ablation as ab  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "micro_expression"
PROTOCOLS = [
    "upper",
    "lower",
    "control",
]

ACTIVE_PHASES = {
    "onset",
    "hold",
    "release",
}

NEUTRAL_PHASES = {
    "pre_neutral",
    "post_neutral",
}

MOTION_CANDIDATES = [
    "motion_mean",
    "motion_max",
    "motion_mouth",
    "motion_left_eye",
    "motion_right_eye",
    "motion_left_brow",
    "motion_right_brow",
    "motion_eyes",
    "motion_brow",
]


def seed_everything(
    seed: int,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(
            seed
        )


def choose_device(
    requested: str,
):
    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but not available."
            )

        return torch.device(
            "mps"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available."
            )

        return torch.device(
            "cuda"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if torch.backends.mps.is_available():
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def discover_core_features(
    frames,
):
    common = set(
        frames[0].columns
    )

    for frame in frames[1:]:
        common &= set(
            frame.columns
        )

    blendshape = sorted(
        c
        for c in common
        if c.startswith(
            "bs_"
        )
    )

    geometry = sorted(
        c
        for c in common
        if c.startswith(
            "geom_abs_"
        )
    )

    motion = [
        c
        for c in MOTION_CANDIDATES
        if c in common
    ]

    if not blendshape:
        raise RuntimeError(
            "No common bs_* columns."
        )

    if not geometry:
        raise RuntimeError(
            "No common geom_abs_* columns."
        )

    if not motion:
        raise RuntimeError(
            "No generic motion columns."
        )

    feature_cols = (
        blendshape
        + geometry
        + motion
    )

    forbidden_prefixes = [
        "geom_delta_",
        "geom_state_",
        "dino_",
    ]

    for feature in feature_cols:
        if any(
            feature.startswith(
                prefix
            )
            for prefix in forbidden_prefixes
        ):
            raise RuntimeError(
                f"Forbidden personal/DINO feature leaked: {feature}"
            )

    forbidden_exact = {
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "blink",
        "gaze_horizontal",
        "gaze_vertical",
        "face_ratio",
        "target",
        "is_calibration",
        "face_detected",
    }

    leaked = sorted(
        forbidden_exact
        & set(
            feature_cols
        )
    )

    if leaked:
        raise RuntimeError(
            f"Nuisance/QC features leaked into TCN core: {leaked}"
        )

    return (
        blendshape,
        geometry,
        motion,
        feature_cols,
    )


def load_subject(
    participant,
    session,
    calibration_start,
    calibration_end,
):
    pieces = []

    for protocol in PROTOCOLS:
        df = ab.load_csv(
            participant,
            session,
            protocol,
        )

        df = ab.label_targets(
            df,
            calibration_start,
            calibration_end,
        )

        pieces.append(
            df
        )

    return pd.concat(
        pieces,
        ignore_index=True,
        sort=False,
    )


def make_trial_key(
    frame: pd.DataFrame,
):
    frame = frame.copy()

    trial_key = pd.Series(
        "",
        index=frame.index,
        dtype=object,
    )

    action_mask = frame[
        "protocol"
    ].isin(
        [
            "upper",
            "lower",
        ]
    )

    action = (
        frame.get(
            "action",
            pd.Series(
                "",
                index=frame.index,
            ),
        )
        .fillna("")
        .astype(str)
    )

    repeat_idx = pd.to_numeric(
        frame.get(
            "repeat_idx",
            pd.Series(
                np.nan,
                index=frame.index,
            ),
        ),
        errors="coerce",
    )

    trial_key.loc[
        action_mask
    ] = (
        frame.loc[
            action_mask,
            "recording_key",
        ].astype(str)
        + "|"
        + action.loc[
            action_mask
        ]
        + "|R"
        + repeat_idx.loc[
            action_mask
        ]
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    control_mask = frame[
        "protocol"
    ].eq(
        "control"
    )

    trial_key.loc[
        control_mask
    ] = (
        frame.loc[
            control_mask,
            "recording_key",
        ].astype(str)
        + "|CONTROL"
    )

    frame[
        "trial_key"
    ] = trial_key

    return frame


def trial_level_split(
    train_df,
    validation_repeat,
):
    """
    Keep one repeat per action as validation.

    Default validation_repeat=3 means:
        R1/R2 -> train
        R3    -> validation

    Control is split in time at 70/30 within each train subject recording.
    """
    train_df = train_df.copy()

    action_mask = train_df[
        "protocol"
    ].isin(
        [
            "upper",
            "lower",
        ]
    )

    repeat_idx = pd.to_numeric(
        train_df.get(
            "repeat_idx",
            pd.Series(
                np.nan,
                index=train_df.index,
            ),
        ),
        errors="coerce",
    )

    action_val = (
        action_mask
        & repeat_idx.eq(
            validation_repeat
        )
    )

    action_train = (
        action_mask
        & ~repeat_idx.eq(
            validation_repeat
        )
    )

    control_mask = train_df[
        "protocol"
    ].eq(
        "control"
    )

    control_train = pd.Series(
        False,
        index=train_df.index,
    )

    control_val = pd.Series(
        False,
        index=train_df.index,
    )

    for recording_key, group in train_df[
        control_mask
    ].groupby(
        "recording_key",
        sort=False,
    ):
        ordered = group.sort_values(
            "analysis_timestamp_ms"
        )

        cut = int(
            round(
                len(ordered)
                * 0.70
            )
        )

        control_train.loc[
            ordered.index[:cut]
        ] = True

        control_val.loc[
            ordered.index[cut:]
        ] = True

    train_mask = (
        action_train
        | control_train
    )

    val_mask = (
        action_val
        | control_val
    )

    overlap = (
        train_mask
        & val_mask
    )

    if overlap.any():
        raise RuntimeError(
            "Train/validation frame overlap."
        )

    return (
        train_df[
            train_mask
        ].copy(),
        train_df[
            val_mask
        ].copy(),
    )


def normalize_frames(
    train_outer,
    train_split,
    val_split,
    test_df,
    feature_cols,
):
    """
    Global robust baseline from OUTER-TRAIN calibration frames only.

    The same transform is applied to train/val/test.
    """
    global_cal = ab.calibration_frame(
        train_outer
    )

    stats = ab.robust_stats(
        global_cal,
        feature_cols,
    )

    return (
        ab.z_transform(
            train_split,
            feature_cols,
            stats,
        ),
        ab.z_transform(
            val_split,
            feature_cols,
            stats,
        ),
        ab.z_transform(
            test_df,
            feature_cols,
            stats,
        ),
        stats,
    )


@dataclass
class SequencePack:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


def build_sequences(
    frame,
    normalized,
    feature_cols,
    *,
    seq_len,
    stride,
    min_face_coverage,
    include_control,
):
    work = frame.copy()

    z = normalized[
        feature_cols
    ].copy()

    z.columns = [
        "__z__"
        + c
        for c in feature_cols
    ]

    work = pd.concat(
        [
            work.reset_index(
                drop=True
            ),
            z.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    z_cols = [
        "__z__"
        + c
        for c in feature_cols
    ]

    usable = work[
        work[
            "target"
        ].notna()
        & ~work[
            "is_calibration"
        ].fillna(
            False
        )
    ].copy()

    if not include_control:
        usable = usable[
            usable[
                "protocol"
            ].isin(
                [
                    "upper",
                    "lower",
                ]
            )
        ].copy()

    X_rows = []
    y_rows = []
    metadata = []

    for trial_key, group in usable.groupby(
        "trial_key",
        sort=False,
    ):
        group = group.sort_values(
            "analysis_timestamp_ms"
        ).reset_index(
            drop=True
        )

        if len(
            group
        ) == 0:
            continue

        values = group[
            z_cols
        ].to_numpy(
            dtype=np.float32
        )

        # NaN after robust normalization means a feature was unavailable at a
        # given frame. 0 means "at baseline" after normalization.
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        face = (
            pd.to_numeric(
                group.get(
                    "face_detected",
                    pd.Series(
                        1,
                        index=group.index,
                    ),
                ),
                errors="coerce",
            )
            .fillna(0)
            .to_numpy(
                dtype=float
            )
        )

        target = pd.to_numeric(
            group[
                "target"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        timestamp = pd.to_numeric(
            group[
                "analysis_timestamp_ms"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        # Causal warm-start:
        # Every endpoint is eligible, including the first PRE-neutral frames.
        # If less than seq_len history exists, repeat the first observed frame
        # on the LEFT. This uses no future information and avoids deleting most
        # PRE-neutral negatives merely because seq_len ~= PRE duration.
        for end_idx in range(
            0,
            len(group),
            stride,
        ):
            raw_start_idx = (
                end_idx
                - seq_len
                + 1
            )

            start_idx = max(
                0,
                raw_start_idx,
            )

            seq_face = face[
                start_idx:
                end_idx + 1
            ]

            if (
                len(seq_face) == 0
                or np.mean(
                    seq_face
                )
                < min_face_coverage
            ):
                continue

            y = target[
                end_idx
            ]

            if not np.isfinite(
                y
            ):
                continue

            observed = values[
                start_idx:
                end_idx + 1,
                :
            ]

            pad_len = max(
                0,
                seq_len
                - len(
                    observed
                ),
            )

            if pad_len:
                pad = np.repeat(
                    observed[
                        0:1,
                        :
                    ],
                    pad_len,
                    axis=0,
                )

                sequence = np.concatenate(
                    [
                        pad,
                        observed,
                    ],
                    axis=0,
                )
            else:
                sequence = observed

            if len(
                sequence
            ) != seq_len:
                raise RuntimeError(
                    f"{trial_key}: sequence length {len(sequence)} "
                    f"!= expected {seq_len}"
                )

            X_rows.append(
                sequence.T
            )

            y_rows.append(
                int(
                    y
                )
            )

            row = group.iloc[
                end_idx
            ]

            metadata.append({
                "participant": str(
                    row[
                        "participant"
                    ]
                ),
                "protocol": str(
                    row[
                        "protocol"
                    ]
                ),
                "recording_key": str(
                    row[
                        "recording_key"
                    ]
                ),
                "trial_key": str(
                    trial_key
                ),
                "segment_key": str(
                    row[
                        "segment_key"
                    ]
                ),
                "movement_phase": str(
                    row.get(
                        "movement_phase",
                        "",
                    )
                ),
                "target": int(
                    y
                ),
                "sequence_start_ms": float(
                    timestamp[
                        start_idx
                    ]
                ),
                "warmup_padding_frames": int(
                    pad_len
                ),
                "sequence_end_ms": float(
                    timestamp[
                        end_idx
                    ]
                ),
                "frame_idx": int(
                    row[
                        "frame_idx"
                    ]
                ),
            })

    if not X_rows:
        return SequencePack(
            X=np.empty(
                (
                    0,
                    len(
                        feature_cols
                    ),
                    seq_len,
                ),
                dtype=np.float32,
            ),
            y=np.empty(
                (0,),
                dtype=np.int64,
            ),
            meta=pd.DataFrame(),
        )

    return SequencePack(
        X=np.stack(
            X_rows
        ).astype(
            np.float32
        ),
        y=np.asarray(
            y_rows,
            dtype=np.int64,
        ),
        meta=pd.DataFrame(
            metadata
        ),
    )


class SequenceDataset(
    Dataset
):
    def __init__(
        self,
        pack: SequencePack,
    ):
        self.X = torch.from_numpy(
            pack.X
        )

        self.y = torch.from_numpy(
            pack.y.astype(
                np.float32
            )
        )

    def __len__(
        self,
    ):
        return len(
            self.y
        )

    def __getitem__(
        self,
        idx,
    ):
        return (
            self.X[
                idx
            ],
            self.y[
                idx
            ],
        )


class Chomp1d(
    nn.Module
):
    def __init__(
        self,
        chomp_size,
    ):
        super().__init__()
        self.chomp_size = int(
            chomp_size
        )

    def forward(
        self,
        x,
    ):
        if self.chomp_size == 0:
            return x

        return x[
            :,
            :,
            :-self.chomp_size,
        ]


class TemporalBlock(
    nn.Module
):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(
                padding
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(
                padding
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
        )

        self.residual = (
            nn.Identity()
            if in_channels
            == out_channels
            else nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        x,
    ):
        return self.relu(
            self.net(
                x
            )
            + self.residual(
                x
            )
        )


class FacialTCN(
    nn.Module
):
    def __init__(
        self,
        input_channels,
        channels,
        kernel_size,
        dropout,
    ):
        super().__init__()

        blocks = []

        in_channels = input_channels

        for level, out_channels in enumerate(
            channels
        ):
            blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=2 ** level,
                    dropout=dropout,
                )
            )

            in_channels = out_channels

        self.tcn = nn.Sequential(
            *blocks
        )

        self.head = nn.Linear(
            channels[-1],
            1,
        )

    def forward(
        self,
        x,
    ):
        h = self.tcn(
            x
        )

        # Causal prediction for the current/end frame.
        current = h[
            :,
            :,
            -1,
        ]

        return self.head(
            current
        ).squeeze(
            -1
        )


def batch_probabilities(
    model,
    loader,
    device,
):
    model.eval()

    probabilities = []
    targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(
                device
            )

            logits = model(
                X
            )

            probability = torch.sigmoid(
                logits
            )

            probabilities.append(
                probability
                .detach()
                .cpu()
                .numpy()
            )

            targets.append(
                y.numpy()
            )

    if not probabilities:
        return (
            np.empty(
                (0,),
                dtype=float,
            ),
            np.empty(
                (0,),
                dtype=int,
            ),
        )

    return (
        np.concatenate(
            probabilities
        ),
        np.concatenate(
            targets
        ).astype(
            int
        ),
    )


def evaluate_ranking(
    y,
    p,
):
    if len(
        np.unique(
            y
        )
    ) < 2:
        return (
            np.nan,
            np.nan,
        )

    return (
        float(
            roc_auc_score(
                y,
                p,
            )
        ),
        float(
            average_precision_score(
                y,
                p,
            )
        ),
    )


def choose_threshold(
    y,
    p,
):
    if len(
        np.unique(
            y
        )
    ) < 2:
        return 0.5

    thresholds = np.linspace(
        0.02,
        0.98,
        193,
    )

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        pred = (
            p >= threshold
        ).astype(int)

        score = balanced_accuracy_score(
            y,
            pred,
        )

        if score > best_score:
            best_score = float(
                score
            )

            best_threshold = float(
                threshold
            )

    return best_threshold


def classification_metrics(
    y,
    p,
    threshold,
):
    pred = (
        p >= threshold
    ).astype(int)

    negative = (
        y == 0
    )

    specificity = (
        float(
            (
                pred[
                    negative
                ] == 0
            ).mean()
        )
        if negative.any()
        else np.nan
    )

    auroc, auprc = evaluate_ranking(
        y,
        p,
    )

    prevalence = float(
        np.mean(
            y == 1
        )
    )

    return {
        "positive_prevalence": prevalence,
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                pred,
            )
        ),
        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "specificity": specificity,
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift": (
            float(
                auprc
                / prevalence
            )
            if (
                np.isfinite(
                    auprc
                )
                and prevalence > 0
            )
            else np.nan
        ),
    }


def event_metrics(
    meta,
    p,
    threshold,
    min_consecutive=3,
):
    """
    Report both:
      - any-window event recall (legacy, lenient)
      - sustained event recall requiring >= min_consecutive positive endpoints

    With stride=5 at 30 FPS, 3 consecutive positives correspond to roughly
    0.5 s of sustained detection.
    """
    work = meta.copy().reset_index(
        drop=True
    )

    work[
        "probability"
    ] = p

    active = work[
        work[
            "target"
        ].eq(
            1
        )
    ].copy()

    if len(
        active
    ) == 0:
        return {
            "event_recall_any": np.nan,
            "event_recall_run3": np.nan,
        }

    any_detected = []
    sustained_detected = []

    for trial_key, group in active.groupby(
        "trial_key",
        sort=False,
    ):
        group = group.sort_values(
            "sequence_end_ms"
        )

        pred = (
            group[
                "probability"
            ].to_numpy(
                dtype=float
            )
            >= threshold
        ).astype(
            int
        )

        any_detected.append(
            bool(
                pred.any()
            )
        )

        max_run = 0
        current = 0

        for value in pred:
            if value:
                current += 1
                max_run = max(
                    max_run,
                    current,
                )
            else:
                current = 0

        sustained_detected.append(
            max_run
            >= min_consecutive
        )

    return {
        "event_recall_any": float(
            np.mean(
                any_detected
            )
        ),
        "event_recall_run3": float(
            np.mean(
                sustained_detected
            )
        ),
    }


def action_phase_metrics(
    meta,
    p,
    threshold,
):
    """
    Diagnose whether errors concentrate in PRE, POST, ONSET, HOLD, or RELEASE.
    """
    work = meta.copy().reset_index(
        drop=True
    )

    work[
        "probability"
    ] = p

    work[
        "prediction"
    ] = (
        work[
            "probability"
        ]
        >= threshold
    ).astype(
        int
    )

    out = {}

    mapping = {
        "pre_neutral": "pre_neutral_fpr",
        "post_neutral": "post_neutral_fpr",
        "onset": "onset_recall",
        "hold": "hold_recall",
        "release": "release_recall",
    }

    for phase, metric_name in mapping.items():
        subset = work[
            work[
                "movement_phase"
            ].astype(str).eq(
                phase
            )
        ]

        if len(
            subset
        ) == 0:
            out[
                metric_name
            ] = np.nan
        else:
            out[
                metric_name
            ] = float(
                subset[
                    "prediction"
                ].mean()
            )

    return out

def control_far_per_min(
    meta,
    p,
    threshold,
    stride_frames,
    fps,
):
    work = meta.copy().reset_index(
        drop=True
    )

    if len(
        work
    ) == 0:
        return (
            np.nan,
            np.nan,
        )

    work[
        "prediction"
    ] = (
        p >= threshold
    ).astype(int)

    positive_fraction = float(
        work[
            "prediction"
        ].mean()
    )

    event_count = 0
    duration_ms = 0.0

    for recording_key, group in work.groupby(
        "recording_key",
        sort=False,
    ):
        group = group.sort_values(
            "sequence_end_ms"
        )

        if len(
            group
        ) == 0:
            continue

        duration_ms += max(
            0.0,
            float(
                group[
                    "sequence_end_ms"
                ].max()
                - group[
                    "sequence_end_ms"
                ].min()
            ),
        )

        positives = group[
            group[
                "prediction"
            ].eq(
                1
            )
        ]

        if len(
            positives
        ) == 0:
            continue

        end_ms = pd.to_numeric(
            positives[
                "sequence_end_ms"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        event_count += 1

        if len(
            end_ms
        ) >= 2:
            nominal_ms = (
                stride_frames
                / fps
                * 1000.0
            )

            event_count += int(
                (
                    np.diff(
                        end_ms
                    )
                    > nominal_ms * 1.5
                ).sum()
            )

    minutes = (
        duration_ms
        / 60000.0
    )

    far = (
        float(
            event_count
            / minutes
        )
        if minutes > 0
        else np.nan
    )

    return (
        positive_fraction,
        far,
    )


def train_one_fold(
    train_pack,
    val_pack,
    *,
    input_channels,
    args,
    device,
):
    train_ds = SequenceDataset(
        train_pack
    )

    val_ds = SequenceDataset(
        val_pack
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    channels = [
        int(x)
        for x in args.channels.split(
            ","
        )
        if x.strip()
    ]

    model = FacialTCN(
        input_channels=input_channels,
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(
        device
    )

    positives = max(
        1,
        int(
            (
                train_pack.y == 1
            ).sum()
        ),
    )

    negatives = max(
        1,
        int(
            (
                train_pack.y == 0
            ).sum()
        ),
    )

    pos_weight = torch.tensor(
        [
            negatives
            / positives
        ],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_auprc = -np.inf
    best_epoch = -1
    patience_left = args.patience

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        losses = []

        for X, y in train_loader:
            X = X.to(
                device
            )

            y = y.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                X
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        val_probability, val_y = batch_probabilities(
            model,
            val_loader,
            device,
        )

        val_auroc, val_auprc = evaluate_ranking(
            val_y,
            val_probability,
        )

        train_loss = float(
            np.mean(
                losses
            )
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
        })

        score = (
            val_auprc
            if np.isfinite(
                val_auprc
            )
            else -np.inf
        )

        if score > (
            best_auprc
            + args.min_delta
        ):
            best_auprc = score
            best_epoch = epoch
            patience_left = args.patience

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError(
            "TCN never produced a valid validation checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        pd.DataFrame(
            history
        ),
        best_epoch,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Causal TCN baseline for v4 label-free facial-action detection."
        )
    )

    parser.add_argument(
        "--participants",
        nargs="+",
        default=[
            "p1",
            "p2",
            "p3",
        ],
    )

    parser.add_argument(
        "--session",
        default="s01",
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
        "--seq-len",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--validation-repeat",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--min-face-coverage",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--channels",
        default="64,64,64",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
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
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    seed_everything(
        args.seed
    )

    device = choose_device(
        args.device
    )

    print()
    print(
        "============================================================"
    )
    print(
        "V4 CAUSAL TCN BASELINE v2 (PRE-NEUTRAL-PRESERVING)"
    )
    print(
        "============================================================"
    )
    print(
        "Device:",
        device,
    )
    print(
        f"Sequence: {args.seq_len} frames, stride {args.stride}"
    )
    print(
        "Core input: blendshape + geometry + generic motion"
    )
    print(
        "Excluded : DINO + pose/gaze/blink/face_ratio"
    )

    subject_frames = {}

    raw_frames = []

    for participant in args.participants:
        frame = load_subject(
            participant,
            args.session,
            args.calibration_start,
            args.calibration_end,
        )

        frame = make_trial_key(
            frame
        )

        subject_frames[
            participant
        ] = frame

        raw_frames.append(
            frame
        )

    (
        blendshape_cols,
        geometry_cols,
        motion_cols,
        feature_cols,
    ) = discover_core_features(
        raw_frames
    )

    print(
        f"Features: {len(feature_cols)} "
        f"(BS {len(blendshape_cols)} + "
        f"Geom {len(geometry_cols)} + "
        f"Motion {len(motion_cols)})"
    )

    result_rows = []
    history_rows = []
    prediction_rows = []

    for fold_idx, held_out in enumerate(
        args.participants
    ):
        train_subjects = [
            p
            for p in args.participants
            if p != held_out
        ]

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            f"Held out: {held_out}"
        )
        print(
            "Train   : "
            + " + ".join(
                train_subjects
            )
        )

        train_outer = pd.concat(
            [
                subject_frames[p]
                for p in train_subjects
            ],
            ignore_index=True,
            sort=False,
        )

        test_df = (
            subject_frames[
                held_out
            ]
            .copy()
        )

        (
            train_split,
            val_split,
        ) = trial_level_split(
            train_outer,
            args.validation_repeat,
        )

        (
            train_z,
            val_z,
            test_z,
            stats,
        ) = normalize_frames(
            train_outer,
            train_split,
            val_split,
            test_df,
            feature_cols,
        )

        train_pack = build_sequences(
            train_split,
            train_z,
            feature_cols,
            seq_len=args.seq_len,
            stride=args.stride,
            min_face_coverage=args.min_face_coverage,
            include_control=True,
        )

        val_pack = build_sequences(
            val_split,
            val_z,
            feature_cols,
            seq_len=args.seq_len,
            stride=args.stride,
            min_face_coverage=args.min_face_coverage,
            include_control=True,
        )

        test_action_df = test_df[
            test_df[
                "protocol"
            ].isin(
                [
                    "upper",
                    "lower",
                ]
            )
        ].copy()

        test_action_z = test_z.loc[
            test_action_df.index,
            feature_cols,
        ]

        test_action_pack = build_sequences(
            test_action_df,
            test_action_z,
            feature_cols,
            seq_len=args.seq_len,
            stride=args.stride,
            min_face_coverage=args.min_face_coverage,
            include_control=False,
        )

        test_control_df = test_df[
            test_df[
                "protocol"
            ].eq(
                "control"
            )
        ].copy()

        test_control_z = test_z.loc[
            test_control_df.index,
            feature_cols,
        ]

        test_control_pack = build_sequences(
            test_control_df,
            test_control_z,
            feature_cols,
            seq_len=args.seq_len,
            stride=args.stride,
            min_face_coverage=args.min_face_coverage,
            include_control=True,
        )

        if len(
            np.unique(
                train_pack.y
            )
        ) != 2:
            raise RuntimeError(
                f"{held_out}: train sequence set has one class."
            )

        if len(
            np.unique(
                val_pack.y
            )
        ) != 2:
            raise RuntimeError(
                f"{held_out}: validation sequence set has one class."
            )

        if len(
            np.unique(
                test_action_pack.y
            )
        ) != 2:
            raise RuntimeError(
                f"{held_out}: action test sequence set has one class."
            )

        if (
            held_out
            in set(
                train_pack.meta[
                    "participant"
                ]
            )
        ):
            raise RuntimeError(
                f"{held_out}: held-out participant leaked into train sequences."
            )

        fold_seed = (
            args.seed
            + fold_idx
            * 100
        )

        seed_everything(
            fold_seed
        )

        model, history, best_epoch = train_one_fold(
            train_pack,
            val_pack,
            input_channels=len(
                feature_cols
            ),
            args=args,
            device=device,
        )

        history[
            "held_out"
        ] = held_out

        history_rows.append(
            history
        )

        val_loader = DataLoader(
            SequenceDataset(
                val_pack
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        action_loader = DataLoader(
            SequenceDataset(
                test_action_pack
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        control_loader = DataLoader(
            SequenceDataset(
                test_control_pack
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        val_p, val_y = batch_probabilities(
            model,
            val_loader,
            device,
        )

        threshold = choose_threshold(
            val_y,
            val_p,
        )

        action_p, action_y = batch_probabilities(
            model,
            action_loader,
            device,
        )

        control_p, control_y = batch_probabilities(
            model,
            control_loader,
            device,
        )

        metrics = classification_metrics(
            action_y,
            action_p,
            threshold,
        )

        ev_metrics = event_metrics(
            test_action_pack.meta,
            action_p,
            threshold,
            min_consecutive=3,
        )

        phase_metrics = action_phase_metrics(
            test_action_pack.meta,
            action_p,
            threshold,
        )

        control_fraction, control_far = control_far_per_min(
            test_control_pack.meta,
            control_p,
            threshold,
            args.stride,
            fps=30.0,
        )

        result = {
            "held_out": held_out,
            "train_subjects": "+".join(
                train_subjects
            ),
            "best_epoch": int(
                best_epoch
            ),
            "threshold": float(
                threshold
            ),
            "train_sequences": int(
                len(
                    train_pack.y
                )
            ),
            "val_sequences": int(
                len(
                    val_pack.y
                )
            ),
            "test_action_sequences": int(
                len(
                    test_action_pack.y
                )
            ),
            "test_control_sequences": int(
                len(
                    test_control_pack.y
                )
            ),
            **metrics,
            **ev_metrics,
            **phase_metrics,
            "control_positive_fraction": control_fraction,
            "control_false_activations_per_min": control_far,
        }

        result_rows.append(
            result
        )

        print(
            f"  best epoch={best_epoch:>3} "
            f"threshold={threshold:.3f} "
            f"AUPRC={metrics['auprc']:.3f} "
            f"AUROC={metrics['auroc']:.3f} "
            f"BA={metrics['balanced_accuracy']:.3f} "
            f"prev={metrics['positive_prevalence']:.3f} "
            f"eventR3={ev_metrics['event_recall_run3']:.3f} "
            f"control FAR/min={control_far:.2f}"
        )

        action_pred = (
            test_action_pack
            .meta
            .copy()
            .reset_index(
                drop=True
            )
        )

        action_pred[
            "challenge"
        ] = "action"

        action_pred[
            "probability"
        ] = action_p

        action_pred[
            "prediction"
        ] = (
            action_p
            >= threshold
        ).astype(
            int
        )

        action_pred[
            "threshold"
        ] = threshold

        control_pred = (
            test_control_pack
            .meta
            .copy()
            .reset_index(
                drop=True
            )
        )

        control_pred[
            "challenge"
        ] = "control"

        control_pred[
            "probability"
        ] = control_p

        control_pred[
            "prediction"
        ] = (
            control_p
            >= threshold
        ).astype(
            int
        )

        control_pred[
            "threshold"
        ] = threshold

        prediction_rows.extend(
            [
                action_pred,
                control_pred,
            ]
        )

    results = pd.DataFrame(
        result_rows
    )

    history = pd.concat(
        history_rows,
        ignore_index=True,
    )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    summary_metrics = [
        "positive_prevalence",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auroc",
        "auprc",
        "auprc_lift",
        "event_recall_any",
        "event_recall_run3",
        "pre_neutral_fpr",
        "post_neutral_fpr",
        "onset_recall",
        "hold_recall",
        "release_recall",
        "control_positive_fraction",
        "control_false_activations_per_min",
    ]

    summary = (
        results[
            summary_metrics
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "metric",
            }
        )
    )

    out_dir = (
        OUTPUT_ROOT
        / "v4_tcn"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        out_dir
        / "fold_results.csv",
        index=False,
    )

    history.to_csv(
        out_dir
        / "epoch_history.csv",
        index=False,
    )

    predictions.to_csv(
        out_dir
        / "predictions.csv",
        index=False,
    )

    control_phase = (
        predictions[
            predictions[
                "challenge"
            ].eq(
                "control"
            )
        ]
        .groupby(
            [
                "participant",
                "segment_key",
            ],
            as_index=False,
        )
        .agg(
            n_windows=(
                "prediction",
                "size",
            ),
            positive_fraction=(
                "prediction",
                "mean",
            ),
            mean_probability=(
                "probability",
                "mean",
            ),
            max_probability=(
                "probability",
                "max",
            ),
        )
    )

    control_phase.to_csv(
        out_dir
        / "control_phase_summary.csv",
        index=False,
    )

    summary.to_csv(
        out_dir
        / "summary.csv",
        index=False,
    )

    with open(
        out_dir
        / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "participants": args.participants,
                "session": args.session,
                "features": feature_cols,
                "feature_counts": {
                    "blendshape": len(
                        blendshape_cols
                    ),
                    "geometry": len(
                        geometry_cols
                    ),
                    "motion": len(
                        motion_cols
                    ),
                    "total": len(
                        feature_cols
                    ),
                },
                "excluded": [
                    "DINO",
                    "yaw/pitch/roll",
                    "gaze",
                    "blink",
                    "face_ratio",
                    "geom_delta_*",
                    "geom_state_*",
                    "dino_change_*",
                ],
                "sequence": {
                    "causal": True,
                    "seq_len_frames": args.seq_len,
                    "stride_frames": args.stride,
                    "warm_start": (
                        "left-pad by repeating first observed frame in each trial; "
                        "preserves early PRE-neutral endpoints without future leakage"
                    ),
                    "assumed_fps_for_reporting": 30.0,
                },
                "event_metric": {
                    "legacy_any_positive": "event_recall_any",
                    "primary_sustained": (
                        "event_recall_run3; requires >=3 consecutive positive "
                        "sequence endpoints (~0.5 s at stride=5, 30 FPS)"
                    ),
                },
                "validation": {
                    "action_repeat": args.validation_repeat,
                    "control_time_split": "first 70% train / last 30% val",
                    "threshold": (
                        "selected on validation set only by balanced accuracy"
                    ),
                },
                "model": {
                    "channels": args.channels,
                    "kernel_size": args.kernel_size,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "epochs_max": args.epochs,
                    "patience": args.patience,
                },
                "seed": args.seed,
                "device": str(
                    device
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.set_option(
        "display.width",
        220,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "TCN LOSO RESULTS"
    )
    print(
        "============================================================"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "MEAN ± STD"
    )
    print(
        "============================================================"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved to:"
    )
    print(
        out_dir
    )


if __name__ == "__main__":
    main()
