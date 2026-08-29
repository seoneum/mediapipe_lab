from __future__ import annotations

"""
Diagnose why pure nuisance features predict ACTION-active vs ACTION-neutral.

This is NOT another model-development benchmark. It is a confound audit.

It asks two questions:

1) Which individual nuisance variable can predict active vs PRE/POST neutral
   in LOSO action-only evaluation?
   Features:
       face_ratio
       yaw_deg
       pitch_deg
       roll_deg
       blink
       gaze_horizontal
       gaze_vertical

2) Within each instructed action/repeat, how much does each nuisance variable
   actually shift from adjacent PRE/POST neutral to ONSET/HOLD/RELEASE?

Outputs:
    outputs/micro_expression/v4_nuisance_diagnostic/
        single_feature_loso.csv
        single_feature_summary.csv
        trial_nuisance_deltas.csv
        action_nuisance_summary.csv
        config.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )
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
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_v4_modality_ablation as ab  # noqa: E402


OUTPUT_ROOT = ROOT / "outputs" / "micro_expression"
ACTION_PROTOCOLS = ["upper", "lower"]

PURE_NUISANCE = [
    "face_ratio",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "blink",
    "gaze_horizontal",
    "gaze_vertical",
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


def load_action_frames(
    participants,
    session,
    calibration_start,
    calibration_end,
):
    by_subject = {}
    all_frames = []

    for participant in participants:
        pieces = []

        for protocol in ACTION_PROTOCOLS:
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

        subject = pd.concat(
            pieces,
            ignore_index=True,
            sort=False,
        )

        by_subject[
            participant
        ] = subject

        all_frames.append(
            subject
        )

    common = set(
        all_frames[0].columns
    )

    for frame in all_frames[1:]:
        common &= set(
            frame.columns
        )

    nuisance = [
        c
        for c in PURE_NUISANCE
        if c in common
    ]

    missing = [
        c
        for c in PURE_NUISANCE
        if c not in common
    ]

    if missing:
        raise RuntimeError(
            f"Missing pure nuisance columns: {missing}"
        )

    return (
        by_subject,
        nuisance,
    )


def make_action_windows(
    df,
    normalized,
    feature_cols,
    args,
):
    action_df = df[
        df[
            "protocol"
        ].isin(
            ACTION_PROTOCOLS
        )
    ].copy()

    normalized_action = normalized.loc[
        action_df.index,
        feature_cols,
    ]

    return ab.make_windows(
        action_df,
        normalized_action,
        feature_cols,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        min_frames=args.min_window_frames,
        min_face_coverage=args.min_face_coverage,
        min_dino_coverage=0.0,
        require_dino=False,
        normalization_name="global",
    )


def fit_single_feature_loso(
    by_subject,
    participants,
    nuisance_cols,
    args,
):
    rows = []

    feature_sets = {
        feature: [
            feature
        ]
        for feature in nuisance_cols
    }

    feature_sets[
        "ALL_PURE_NUISANCE"
    ] = list(
        nuisance_cols
    )

    for held_out in participants:
        train_subjects = [
            p
            for p in participants
            if p != held_out
        ]

        train_df = pd.concat(
            [
                by_subject[p]
                for p in train_subjects
            ],
            ignore_index=True,
            sort=False,
        )

        test_df = (
            by_subject[
                held_out
            ]
            .copy()
        )

        for feature_name, feature_cols in feature_sets.items():
            global_cal = ab.calibration_frame(
                train_df
            )

            stats = ab.robust_stats(
                global_cal,
                feature_cols,
            )

            train_z = ab.z_transform(
                train_df,
                feature_cols,
                stats,
            )

            test_z = ab.z_transform(
                test_df,
                feature_cols,
                stats,
            )

            X_train, y_train, M_train = make_action_windows(
                train_df,
                train_z,
                feature_cols,
                args,
            )

            X_test, y_test, M_test = make_action_windows(
                test_df,
                test_z,
                feature_cols,
                args,
            )

            if len(
                np.unique(
                    y_train
                )
            ) != 2:
                raise RuntimeError(
                    f"{held_out}/{feature_name}: train has one class."
                )

            if len(
                np.unique(
                    y_test
                )
            ) != 2:
                raise RuntimeError(
                    f"{held_out}/{feature_name}: test has one class."
                )

            scaler = StandardScaler()

            X_train_s = scaler.fit_transform(
                X_train
            )

            X_test_s = scaler.transform(
                X_test
            )

            model = LogisticRegression(
                C=args.C,
                class_weight="balanced",
                max_iter=4000,
                solver="lbfgs",
                random_state=0,
            )

            model.fit(
                X_train_s,
                y_train,
            )

            probability = (
                model.predict_proba(
                    X_test_s
                )[:, 1]
            )

            prevalence = float(
                (
                    y_test == 1
                ).mean()
            )

            auprc = float(
                average_precision_score(
                    y_test,
                    probability,
                )
            )

            rows.append({
                "held_out": held_out,
                "train_subjects": "+".join(
                    train_subjects
                ),
                "feature": feature_name,
                "n_windows": int(
                    len(y_test)
                ),
                "positive_prevalence": prevalence,
                "auroc": float(
                    roc_auc_score(
                        y_test,
                        probability,
                    )
                ),
                "auprc": auprc,
                "auprc_lift": float(
                    auprc / prevalence
                ),
            })

    results = pd.DataFrame(
        rows
    )

    summary = (
        results
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            auroc_mean=(
                "auroc",
                "mean",
            ),
            auroc_std=(
                "auroc",
                "std",
            ),
            auprc_mean=(
                "auprc",
                "mean",
            ),
            auprc_std=(
                "auprc",
                "std",
            ),
            auprc_lift_mean=(
                "auprc_lift",
                "mean",
            ),
            prevalence_mean=(
                "positive_prevalence",
                "mean",
            ),
        )
        .sort_values(
            "auprc_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        results,
        summary,
    )


def central_value(
    series,
    feature,
):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    if feature == "blink":
        # For blink, the mean is an interpretable frame fraction / rate proxy.
        return float(
            values.mean()
        )

    return float(
        values.median()
    )


def trial_deltas(
    by_subject,
    nuisance_cols,
):
    rows = []

    for participant, df in by_subject.items():
        for (
            protocol,
            action,
            repeat_idx,
        ), group in df.groupby(
            [
                "protocol",
                "action",
                "repeat_idx",
            ],
            dropna=True,
            sort=False,
        ):
            phase = (
                group[
                    "movement_phase"
                ]
                .fillna("")
                .astype(str)
            )

            active = group[
                phase.isin(
                    ACTIVE_PHASES
                )
            ]

            neutral = group[
                phase.isin(
                    NEUTRAL_PHASES
                )
            ]

            if (
                len(active) == 0
                or len(neutral) == 0
            ):
                continue

            for feature in nuisance_cols:
                active_value = central_value(
                    active[
                        feature
                    ],
                    feature,
                )

                neutral_value = central_value(
                    neutral[
                        feature
                    ],
                    feature,
                )

                if not (
                    np.isfinite(
                        active_value
                    )
                    and np.isfinite(
                        neutral_value
                    )
                ):
                    continue

                rows.append({
                    "participant": participant,
                    "protocol": protocol,
                    "action": str(
                        action
                    ),
                    "repeat_idx": int(
                        repeat_idx
                    ),
                    "feature": feature,
                    "active_value": active_value,
                    "neutral_value": neutral_value,
                    "delta_active_minus_neutral": (
                        active_value
                        - neutral_value
                    ),
                    "abs_delta": abs(
                        active_value
                        - neutral_value
                    ),
                })

    trial_df = pd.DataFrame(
        rows
    )

    if len(trial_df) == 0:
        raise RuntimeError(
            "No action/repeat nuisance deltas were created."
        )

    def sign_consistency(
        values,
    ):
        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(values) == 0:
            return np.nan

        positive = float(
            (
                values > 0
            ).mean()
        )

        negative = float(
            (
                values < 0
            ).mean()
        )

        return max(
            positive,
            negative,
        )

    summary_rows = []

    for (
        feature,
        action,
    ), group in trial_df.groupby(
        [
            "feature",
            "action",
        ],
        sort=False,
    ):
        delta = group[
            "delta_active_minus_neutral"
        ].to_numpy(
            dtype=float
        )

        summary_rows.append({
            "feature": feature,
            "action": action,
            "n_trials": int(
                len(group)
            ),
            "median_delta": float(
                np.median(
                    delta
                )
            ),
            "median_abs_delta": float(
                np.median(
                    np.abs(
                        delta
                    )
                )
            ),
            "mean_abs_delta": float(
                np.mean(
                    np.abs(
                        delta
                    )
                )
            ),
            "sign_consistency": float(
                sign_consistency(
                    delta
                )
            ),
        })

    action_summary = (
        pd.DataFrame(
            summary_rows
        )
        .sort_values(
            [
                "feature",
                "median_abs_delta",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return (
        trial_df,
        action_summary,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose action-phase coupling of pure nuisance variables."
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
        "--window-ms",
        type=float,
        default=1000.0,
    )

    parser.add_argument(
        "--stride-ms",
        type=float,
        default=250.0,
    )

    parser.add_argument(
        "--min-window-frames",
        type=int,
        default=12,
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

    if len(
        args.participants
    ) < 3:
        raise ValueError(
            "Need at least 3 participants for LOSO."
        )

    out_dir = (
        OUTPUT_ROOT
        / "v4_nuisance_diagnostic"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        by_subject,
        nuisance_cols,
    ) = load_action_frames(
        args.participants,
        args.session,
        args.calibration_start,
        args.calibration_end,
    )

    loso, summary = (
        fit_single_feature_loso(
            by_subject,
            args.participants,
            nuisance_cols,
            args,
        )
    )

    (
        trials,
        action_summary,
    ) = trial_deltas(
        by_subject,
        nuisance_cols,
    )

    loso.to_csv(
        out_dir
        / "single_feature_loso.csv",
        index=False,
    )

    summary.to_csv(
        out_dir
        / "single_feature_summary.csv",
        index=False,
    )

    trials.to_csv(
        out_dir
        / "trial_nuisance_deltas.csv",
        index=False,
    )

    action_summary.to_csv(
        out_dir
        / "action_nuisance_summary.csv",
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
                "features": nuisance_cols,
                "action_test": (
                    "upper/lower active vs pre/post neutral only"
                ),
                "window_ms": args.window_ms,
                "stride_ms": args.stride_ms,
                "calibration_start": args.calibration_start,
                "calibration_end": args.calibration_end,
                "note": (
                    "Exploratory confound diagnostic; no clinical inference."
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

    pd.set_option(
        "display.max_rows",
        200,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "PURE NUISANCE SINGLE-FEATURE LOSO"
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
        "============================================================"
    )
    print(
        "PER-FOLD SINGLE-FEATURE RESULTS"
    )
    print(
        "============================================================"
    )

    print(
        loso.sort_values(
            [
                "held_out",
                "auprc",
            ],
            ascending=[
                True,
                False,
            ],
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "LARGEST WITHIN-ACTION NUISANCE SHIFTS"
    )
    print(
        "============================================================"
    )

    for feature in nuisance_cols:
        subset = (
            action_summary[
                action_summary[
                    "feature"
                ].eq(
                    feature
                )
            ]
            .head(5)
        )

        print()
        print(
            f"[{feature}]"
        )

        print(
            subset.to_string(
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
