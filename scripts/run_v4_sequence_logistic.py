from __future__ import annotations

"""
Fair sequence-matched logistic baselines for the v4 TCN experiment.

This script uses the EXACT SAME:
- outer LOSO subjects
- train/validation repeat split
- control hard negatives
- 79 input channels
- global robust normalization
- causal 60-frame sequences with PRE-neutral-preserving left padding
- validation threshold policy
- held-out action/control evaluation

as scripts/train_v4_tcn.py.

It compares two non-neural baselines:

1) last_frame
   Logistic regression on only X_t (79 dims).

2) temporal_summary
   Logistic regression on handcrafted causal sequence summaries:
       mean, std, max_abs, end-start
   over the same 60-frame history (79*4 = 316 dims).

This answers:
    Does the TCN itself add value beyond a static linear model or simple
    handcrafted temporal summaries?

Expected local dependency:
    scripts/train_v4_tcn.py
must be the PRE-NEUTRAL-PRESERVING v2-compatible script.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
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

import train_v4_tcn as tcn  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "micro_expression"


def check_tcn_helper():
    required = [
        "load_subject",
        "make_trial_key",
        "discover_core_features",
        "trial_level_split",
        "normalize_frames",
        "build_sequences",
        "classification_metrics",
        "event_metrics",
        "action_phase_metrics",
        "control_far_per_min",
        "choose_threshold",
    ]

    missing = [
        name
        for name in required
        if not hasattr(
            tcn,
            name,
        )
    ]

    if missing:
        raise RuntimeError(
            "scripts/train_v4_tcn.py is not the expected v2-compatible helper. "
            f"Missing: {missing}"
        )


def summarize_sequences(
    X: np.ndarray,
) -> np.ndarray:
    """
    X shape:
        (N, C, T)

    Returns:
        [mean, std, max_abs, end-start]
        shape (N, 4*C)
    """
    if X.ndim != 3:
        raise ValueError(
            f"Expected X (N,C,T), got {X.shape}"
        )

    mean = np.mean(
        X,
        axis=2,
    )

    std = np.std(
        X,
        axis=2,
    )

    max_abs = np.max(
        np.abs(
            X
        ),
        axis=2,
    )

    delta = (
        X[
            :,
            :,
            -1,
        ]
        - X[
            :,
            :,
            0,
        ]
    )

    return np.concatenate(
        [
            mean,
            std,
            max_abs,
            delta,
        ],
        axis=1,
    ).astype(
        np.float32
    )


def representation(
    X: np.ndarray,
    kind: str,
) -> np.ndarray:
    if kind == "last_frame":
        return X[
            :,
            :,
            -1,
        ].astype(
            np.float32
        )

    if kind == "temporal_summary":
        return summarize_sequences(
            X
        )

    raise ValueError(
        kind
    )


def fit_logistic(
    X,
    y,
    C,
):
    classes = np.unique(
        y
    )

    if len(
        classes
    ) != 2:
        raise RuntimeError(
            f"Training set has classes {classes.tolist()}"
        )

    scaler = StandardScaler()

    Xs = scaler.fit_transform(
        X
    )

    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=0,
    )

    model.fit(
        Xs,
        y,
    )

    return (
        scaler,
        model,
    )


def probability(
    scaler,
    model,
    X,
):
    return (
        model
        .predict_proba(
            scaler.transform(
                X
            )
        )[:, 1]
    )


def main():
    check_tcn_helper()

    parser = argparse.ArgumentParser(
        description=(
            "Sequence-matched linear baselines for v4 TCN."
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
        "--C",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    subject_frames = {}
    raw_frames = []

    for participant in args.participants:
        frame = tcn.load_subject(
            participant,
            args.session,
            args.calibration_start,
            args.calibration_end,
        )

        frame = tcn.make_trial_key(
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
    ) = tcn.discover_core_features(
        raw_frames
    )

    print()
    print(
        "============================================================"
    )
    print(
        "V4 SEQUENCE-MATCHED LOGISTIC BASELINES"
    )
    print(
        "============================================================"
    )
    print(
        f"Features: {len(feature_cols)} "
        f"(BS {len(blendshape_cols)} + "
        f"Geom {len(geometry_cols)} + "
        f"Motion {len(motion_cols)})"
    )
    print(
        f"Sequence: {args.seq_len} frames, stride {args.stride}"
    )
    print(
        "Same PRE-neutral-preserving causal windows as TCN."
    )

    rows = []

    for held_out in args.participants:
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
        ) = tcn.trial_level_split(
            train_outer,
            args.validation_repeat,
        )

        (
            train_z,
            val_z,
            test_z,
            _,
        ) = tcn.normalize_frames(
            train_outer,
            train_split,
            val_split,
            test_df,
            feature_cols,
        )

        train_pack = tcn.build_sequences(
            train_split,
            train_z,
            feature_cols,
            seq_len=args.seq_len,
            stride=args.stride,
            min_face_coverage=args.min_face_coverage,
            include_control=True,
        )

        val_combined_pack = tcn.build_sequences(
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

        test_action_pack = tcn.build_sequences(
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

        test_control_pack = tcn.build_sequences(
            test_control_df,
            test_control_z,
            feature_cols,
            seq_len=args.seq_len,
            stride=args.stride,
            min_face_coverage=args.min_face_coverage,
            include_control=True,
        )

        # Outer LOSO safety.
        train_seen = set(
            train_pack
            .meta[
                "participant"
            ]
            .astype(str)
        )

        if held_out in train_seen:
            raise RuntimeError(
                f"{held_out}: held-out participant leaked into train sequences."
            )

        for kind in [
            "last_frame",
            "temporal_summary",
        ]:
            X_train = representation(
                train_pack.X,
                kind,
            )

            X_val = representation(
                val_combined_pack.X,
                kind,
            )

            X_action = representation(
                test_action_pack.X,
                kind,
            )

            X_control = representation(
                test_control_pack.X,
                kind,
            )

            scaler, model = fit_logistic(
                X_train,
                train_pack.y,
                args.C,
            )

            val_p = probability(
                scaler,
                model,
                X_val,
            )

            threshold = tcn.choose_threshold(
                val_combined_pack.y,
                val_p,
            )

            action_p = probability(
                scaler,
                model,
                X_action,
            )

            control_p = probability(
                scaler,
                model,
                X_control,
            )

            metrics = tcn.classification_metrics(
                test_action_pack.y,
                action_p,
                threshold,
            )

            ev = tcn.event_metrics(
                test_action_pack.meta,
                action_p,
                threshold,
                min_consecutive=3,
            )

            phase = tcn.action_phase_metrics(
                test_action_pack.meta,
                action_p,
                threshold,
            )

            (
                control_fraction,
                control_far,
            ) = tcn.control_far_per_min(
                test_control_pack.meta,
                control_p,
                threshold,
                args.stride,
                fps=30.0,
            )

            row = {
                "held_out": held_out,
                "train_subjects": "+".join(
                    train_subjects
                ),
                "baseline": kind,
                "input_dim": int(
                    X_train.shape[1]
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
                        val_combined_pack.y
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
                **ev,
                **phase,
                "control_positive_fraction": (
                    control_fraction
                ),
                "control_false_activations_per_min": (
                    control_far
                ),
            }

            rows.append(
                row
            )

            print(
                f"  {kind:>16} "
                f"AUPRC={metrics['auprc']:.3f} "
                f"AUROC={metrics['auroc']:.3f} "
                f"BA={metrics['balanced_accuracy']:.3f} "
                f"eventR3={ev['event_recall_run3']:.3f} "
                f"FAR/min={control_far:.2f}"
            )

    results = pd.DataFrame(
        rows
    )

    metrics = [
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auroc",
        "auprc",
        "auprc_lift",
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
        results
        .groupby(
            "baseline"
        )[metrics]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    out_dir = (
        OUTPUT_ROOT
        / "v4_sequence_logistic"
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

    summary.to_csv(
        out_dir
        / "summary.csv"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "SEQUENCE-MATCHED LOGISTIC RESULTS"
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
        summary.to_string()
    )

    # Optional direct comparison to the already-generated TCN v2 results.
    tcn_path = (
        OUTPUT_ROOT
        / "v4_tcn"
        / "fold_results.csv"
    )

    if tcn_path.exists():
        tcn_results = pd.read_csv(
            tcn_path
        )

        required_tcn = {
            "held_out",
            "auprc",
            "auroc",
            "balanced_accuracy",
            "event_recall_run3",
            "control_false_activations_per_min",
        }

        if required_tcn.issubset(
            tcn_results.columns
        ):
            compact_tcn = tcn_results[
                [
                    "held_out",
                    "auprc",
                    "auroc",
                    "balanced_accuracy",
                    "event_recall_run3",
                    "control_false_activations_per_min",
                ]
            ].copy()

            compact_tcn[
                "baseline"
            ] = "tcn"

            compact_lr = results[
                [
                    "held_out",
                    "baseline",
                    "auprc",
                    "auroc",
                    "balanced_accuracy",
                    "event_recall_run3",
                    "control_false_activations_per_min",
                ]
            ].copy()

            comparison = pd.concat(
                [
                    compact_lr,
                    compact_tcn,
                ],
                ignore_index=True,
            )

            comparison.to_csv(
                out_dir
                / "comparison_with_tcn.csv",
                index=False,
            )

            print()
            print(
                "============================================================"
            )
            print(
                "DIRECT COMPARISON WITH SAVED TCN"
            )
            print(
                "============================================================"
            )

            print(
                comparison.to_string(
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
