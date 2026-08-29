from __future__ import annotations

"""
Stratified LOSO evaluation for v4 facial-action detection.

Why this script exists
----------------------
The previous binary benchmark mixed:
  - action-recording neutral windows (PRE/POST), and
  - a separate control recording containing blink/gaze/head movements

into one negative class.

That is useful as a hard-negative stress test, but it can also let a model learn
recording/protocol identity. This script separates the questions:

A) ACTION DISCRIMINATION
   upper/lower only:
       positive = onset/hold/release
       negative = pre_neutral/post_neutral

B) CONTROL ROBUSTNESS
   control recording only:
       report false-positive fraction and false-activation events/min

It also compares two training regimes:
  1) neutral_only
     train only on upper/lower active vs neutral windows
  2) with_control_hardneg
     add control windows from TRAIN subjects as negative hard examples

Threshold selection
-------------------
The 0.5 logistic-regression threshold is not treated as sacred.
For every outer LOSO fold, the operating threshold is selected using
participant-grouped out-of-fold predictions from TRAIN subjects only.
The held-out subject is never used for threshold tuning.

This script imports the integrity-checked helper functions from:
    scripts/run_v4_modality_ablation.py

Expected helper version:
    v3 motion/nuisance split or compatible later version.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
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
PROTOCOLS = ["upper", "lower", "control"]


def check_helper_api():
    required = [
        "PROTOCOLS",
        "DINO_REGIONS",
        "load_csv",
        "load_npz",
        "label_targets",
        "discover_scalar_groups",
        "preflight_integrity",
        "fit_train_pcas",
        "add_pca_features_to_recording",
        "calibration_frame",
        "robust_stats",
        "z_transform",
        "build_personal_stats_by_recording",
        "personal_z_transform",
        "make_windows",
        "ensure_unique_columns",
        "assert_unique_frame_idx",
    ]

    missing = [
        name
        for name in required
        if not hasattr(ab, name)
    ]

    if missing:
        raise RuntimeError(
            "run_v4_modality_ablation.py is not compatible. "
            f"Missing helpers: {missing}"
        )


def validate_protocol_union(
    pieces,
    participant,
    required_feature_cols,
):
    """
    Upper/lower/control are allowed to have different annotation columns.

    Examples of legitimate differences:
      upper/lower : action, repeat_idx, movement_phase, intended_progress
      control     : label / control-phase metadata

    What must be invariant:
      - each DataFrame has unique column labels
      - frame_idx is unique within each recording
      - all requested model features exist in every protocol
      - structural columns required by the benchmark exist in every protocol
    """
    structural = {
        "frame_idx",
        "analysis_timestamp_ms",
        "participant",
        "protocol",
        "recording_key",
        "target",
        "segment_key",
        "is_calibration",
        "dino_pca_available",
    }

    cleaned = []
    column_sets = {}

    for protocol, frame in pieces:
        frame = ab.ensure_unique_columns(
            frame.copy(),
            f"{participant}/{protocol}/stratified",
        )

        ab.assert_unique_frame_idx(
            frame,
            f"{participant}/{protocol}/stratified",
        )

        missing_structural = sorted(
            structural
            - set(
                frame.columns
            )
        )

        if missing_structural:
            raise RuntimeError(
                f"{participant}/{protocol}: missing structural columns "
                f"{missing_structural}"
            )

        missing_features = [
            c
            for c in required_feature_cols
            if c not in frame.columns
        ]

        if missing_features:
            raise RuntimeError(
                f"{participant}/{protocol}: missing model features "
                f"{missing_features[:20]}"
            )

        column_sets[
            protocol
        ] = set(
            frame.columns
        )

        cleaned.append(
            (
                protocol,
                frame,
            )
        )

    # Report annotation-only schema differences; do not treat them as errors.
    common_columns = set.intersection(
        *[
            cols
            for cols in column_sets.values()
        ]
    )

    schema_only = {
        protocol: sorted(
            cols
            - common_columns
        )
        for protocol, cols in column_sets.items()
    }

    return (
        cleaned,
        schema_only,
    )


def ensure_two_classes(
    y,
    context,
):
    classes = np.unique(
        np.asarray(
            y,
            dtype=int,
        )
    )

    if len(classes) != 2:
        raise RuntimeError(
            f"{context}: expected two classes, got {classes.tolist()}"
        )


def fit_classifier(
    X,
    y,
    C,
):
    ensure_two_classes(
        y,
        "fit_classifier",
    )

    scaler = StandardScaler()

    Xs = scaler.fit_transform(
        X
    )

    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=4000,
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


def predict_probability(
    scaler,
    model,
    X,
):
    if len(X) == 0:
        return np.empty(
            (0,),
            dtype=float,
        )

    return (
        model
        .predict_proba(
            scaler.transform(
                X
            )
        )[:, 1]
    )


def threshold_metric(
    y,
    p,
    threshold,
    objective,
):
    pred = (
        p >= threshold
    ).astype(int)

    if objective == "f1":
        return float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        )

    return float(
        balanced_accuracy_score(
            y,
            pred,
        )
    )


def choose_threshold_group_oof(
    X,
    y,
    groups,
    *,
    C,
    objective,
):
    """
    Grouped OOF threshold tuning using TRAIN participants only.

    With P1/P2/P3 outer LOSO, the training set contains two subjects, so this
    becomes:
        train subject A -> validate B
        train subject B -> validate A
    and the two validation prediction sets are concatenated.
    """
    X = np.asarray(
        X,
        dtype=np.float32,
    )
    y = np.asarray(
        y,
        dtype=np.int64,
    )
    groups = np.asarray(
        groups,
        dtype=object,
    )

    unique_groups = list(
        dict.fromkeys(
            groups.tolist()
        )
    )

    oof_y = []
    oof_p = []

    for validation_group in unique_groups:
        train_mask = (
            groups != validation_group
        )

        val_mask = (
            groups == validation_group
        )

        if (
            train_mask.sum() == 0
            or val_mask.sum() == 0
        ):
            continue

        y_inner_train = y[
            train_mask
        ]

        if len(
            np.unique(
                y_inner_train
            )
        ) < 2:
            continue

        scaler, model = fit_classifier(
            X[
                train_mask
            ],
            y_inner_train,
            C,
        )

        probability = predict_probability(
            scaler,
            model,
            X[
                val_mask
            ],
        )

        oof_y.append(
            y[
                val_mask
            ]
        )

        oof_p.append(
            probability
        )

    if not oof_y:
        return (
            0.5,
            {
                "threshold_source": "fallback_0.5_no_valid_group_oof",
                "oof_n": 0,
                "oof_objective": np.nan,
            },
        )

    oof_y = np.concatenate(
        oof_y
    )

    oof_p = np.concatenate(
        oof_p
    )

    if len(
        np.unique(
            oof_y
        )
    ) < 2:
        return (
            0.5,
            {
                "threshold_source": "fallback_0.5_oof_one_class",
                "oof_n": int(
                    len(oof_y)
                ),
                "oof_objective": np.nan,
            },
        )

    # Fine fixed grid is stable and prevents threshold selection from depending
    # on accidental duplicated probability values.
    candidates = np.linspace(
        0.02,
        0.98,
        193,
    )

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in candidates:
        score = threshold_metric(
            oof_y,
            oof_p,
            float(threshold),
            objective,
        )

        if score > best_score:
            best_score = score
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        {
            "threshold_source": (
                f"train_subject_group_oof_{objective}"
            ),
            "oof_n": int(
                len(oof_y)
            ),
            "oof_objective": float(
                best_score
            ),
        },
    )


def evaluate_binary(
    y,
    probability,
    threshold,
):
    y = np.asarray(
        y,
        dtype=np.int64,
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    ensure_two_classes(
        y,
        "evaluate_binary",
    )

    pred = (
        probability >= threshold
    ).astype(int)

    negative = (
        y == 0
    )

    specificity = float(
        (
            pred[
                negative
            ] == 0
        ).mean()
    )

    prevalence = float(
        (
            y == 1
        ).mean()
    )

    auprc = float(
        average_precision_score(
            y,
            probability,
        )
    )

    return {
        "n_windows": int(
            len(y)
        ),
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
        "false_positive_rate": float(
            1.0 - specificity
        ),
        "f1": float(
            f1_score(
                y,
                pred,
                zero_division=0,
            )
        ),
        "auroc": float(
            roc_auc_score(
                y,
                probability,
            )
        ),
        "auprc": auprc,
        "auprc_lift": float(
            auprc / prevalence
        ) if prevalence > 0 else np.nan,
    }


def trial_id_from_segment(
    recording_key,
    segment_key,
):
    """
    segment_key:
        upper|brows_raise|R1|onset
        lower|smile|R2|hold
    Trial id drops the phase.
    """
    parts = str(
        segment_key
    ).split("|")

    if len(parts) >= 4:
        prefix = "|".join(
            parts[:-1]
        )
    else:
        prefix = str(
            segment_key
        )

    return (
        str(recording_key)
        + "|"
        + prefix
    )


def active_trial_detection_rate(
    metadata,
    y,
    probability,
    threshold,
):
    if len(metadata) == 0:
        return (
            np.nan,
            0,
            0,
        )

    work = metadata.copy().reset_index(
        drop=True
    )

    work["target_eval"] = np.asarray(
        y,
        dtype=int,
    )

    work["probability"] = np.asarray(
        probability,
        dtype=float,
    )

    work = work[
        work[
            "target_eval"
        ].eq(1)
    ].copy()

    if len(work) == 0:
        return (
            np.nan,
            0,
            0,
        )

    work["trial_id"] = [
        trial_id_from_segment(
            recording_key,
            segment_key,
        )
        for recording_key, segment_key
        in zip(
            work["recording_key"],
            work["segment_key"],
        )
    ]

    detected = (
        work
        .groupby(
            "trial_id"
        )["probability"]
        .max()
        .ge(
            threshold
        )
    )

    return (
        float(
            detected.mean()
        ),
        int(
            detected.sum()
        ),
        int(
            len(detected)
        ),
    )


def count_false_activation_events(
    metadata,
    probability,
    threshold,
    stride_ms,
):
    """
    Collapse overlapping positive control windows into contiguous activation runs.
    """
    if len(metadata) == 0:
        return {
            "control_windows": 0,
            "control_positive_window_fraction": np.nan,
            "control_false_activation_events": 0,
            "control_minutes": 0.0,
            "control_false_activations_per_min": np.nan,
        }

    work = metadata.copy().reset_index(
        drop=True
    )

    work["probability"] = np.asarray(
        probability,
        dtype=float,
    )

    work["pred"] = (
        work["probability"]
        >= threshold
    ).astype(int)

    total_events = 0
    total_duration_ms = 0.0

    for recording_key, group in work.groupby(
        "recording_key",
        sort=False,
    ):
        group = group.sort_values(
            "window_start_ms"
        )

        if len(group):
            total_duration_ms += max(
                0.0,
                float(
                    group[
                        "window_end_ms"
                    ].max()
                    - group[
                        "window_start_ms"
                    ].min()
                ),
            )

        positive = group[
            group["pred"].eq(1)
        ]

        if len(positive) == 0:
            continue

        starts = pd.to_numeric(
            positive[
                "window_start_ms"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        # A new event begins if the gap is bigger than roughly one stride.
        total_events += 1

        if len(starts) >= 2:
            total_events += int(
                (
                    np.diff(
                        starts
                    )
                    > stride_ms * 1.5
                ).sum()
            )

    minutes = (
        total_duration_ms
        / 60000.0
    )

    positive_fraction = float(
        work["pred"].mean()
    )

    return {
        "control_windows": int(
            len(work)
        ),
        "control_positive_window_fraction": positive_fraction,
        "control_false_activation_events": int(
            total_events
        ),
        "control_minutes": float(
            minutes
        ),
        "control_false_activations_per_min": (
            float(
                total_events / minutes
            )
            if minutes > 0
            else np.nan
        ),
    }


def feature_qc_from_train(
    train_df,
    feature_cols,
    min_window_frames,
):
    global_cal = ab.calibration_frame(
        train_df
    )

    finite_train = (
        train_df[
            feature_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .notna()
        .sum(axis=0)
    )

    finite_cal = (
        global_cal[
            feature_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .notna()
        .sum(axis=0)
    )

    usable = [
        c
        for c in feature_cols
        if (
            int(
                finite_train[c]
            ) >= min_window_frames
            and int(
                finite_cal[c]
            ) >= 5
        )
    ]

    dropped = [
        c
        for c in feature_cols
        if c not in usable
    ]

    if not usable:
        raise RuntimeError(
            "No usable features survived training-only QC."
        )

    return (
        usable,
        dropped,
    )


def make_mode_windows(
    train_df,
    test_df,
    train_global,
    test_global,
    train_personal,
    test_personal,
    feature_cols,
    args,
):
    """
    Build action/control windows for global/personal/hybrid while asserting that
    window IDs stay identical across normalization views.
    """

    def one(
        df,
        normalized,
        normalization,
        protocols,
    ):
        subset = df[
            df[
                "protocol"
            ].isin(
                protocols
            )
        ].copy()

        normalized_subset = normalized.loc[
            subset.index,
            feature_cols,
        ]

        return ab.make_windows(
            subset,
            normalized_subset,
            feature_cols,
            window_ms=args.window_ms,
            stride_ms=args.stride_ms,
            min_frames=args.min_window_frames,
            min_face_coverage=args.min_face_coverage,
            min_dino_coverage=args.min_dino_coverage,
            require_dino=True,
            normalization_name=normalization,
        )

    outputs = {}

    for split_name, df, zg, zp in [
        (
            "train",
            train_df,
            train_global,
            train_personal,
        ),
        (
            "test",
            test_df,
            test_global,
            test_personal,
        ),
    ]:
        for challenge_name, protocols in [
            (
                "action",
                ["upper", "lower"],
            ),
            (
                "control",
                ["control"],
            ),
        ]:
            Xg, yg, Mg = one(
                df,
                zg,
                "global",
                protocols,
            )

            Xp, yp, Mp = one(
                df,
                zp,
                "personal",
                protocols,
            )

            if (
                len(Xg) != len(Xp)
                or not np.array_equal(
                    yg,
                    yp,
                )
            ):
                raise RuntimeError(
                    f"{split_name}/{challenge_name}: "
                    "global-personal label/window count mismatch"
                )

            ids_g = (
                Mg["window_id"].tolist()
                if len(Mg)
                else []
            )

            ids_p = (
                Mp["window_id"].tolist()
                if len(Mp)
                else []
            )

            if ids_g != ids_p:
                raise RuntimeError(
                    f"{split_name}/{challenge_name}: "
                    "global-personal window IDs differ"
                )

            outputs[
                (
                    split_name,
                    challenge_name,
                    "global",
                )
            ] = (
                Xg,
                yg,
                Mg,
            )

            outputs[
                (
                    split_name,
                    challenge_name,
                    "personal",
                )
            ] = (
                Xp,
                yp,
                Mp,
            )

            outputs[
                (
                    split_name,
                    challenge_name,
                    "hybrid",
                )
            ] = (
                np.concatenate(
                    [
                        Xg,
                        Xp,
                    ],
                    axis=1,
                ),
                yg,
                Mg.copy(),
            )

    return outputs


def main():
    check_helper_api()

    parser = argparse.ArgumentParser(
        description=(
            "Stratified v4 LOSO: action discrimination + control FAR"
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
        "--dino-pca-dim",
        type=int,
        default=8,
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
        "--min-calibration-frames",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--min-face-coverage",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--min-dino-coverage",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--threshold-objective",
        choices=[
            "balanced_accuracy",
            "f1",
        ],
        default="balanced_accuracy",
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
            "Need at least 3 participants for outer LOSO."
        )

    out_dir = (
        OUTPUT_ROOT
        / "v4_stratified_eval"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_map = {}
    npz_map = {}
    raw_frames = []

    for participant in args.participants:
        for protocol in PROTOCOLS:
            df = ab.load_csv(
                participant,
                args.session,
                protocol,
            )

            df = ab.label_targets(
                df,
                args.calibration_start,
                args.calibration_end,
            )

            csv_map[
                (
                    participant,
                    protocol,
                )
            ] = df

            npz_map[
                (
                    participant,
                    protocol,
                )
            ] = ab.load_npz(
                participant,
                args.session,
                protocol,
            )

            raw_frames.append(
                df
            )

    (
        blendshape_cols,
        geometry_cols,
        motion_cols,
        pure_nuisance_cols,
    ) = ab.discover_scalar_groups(
        raw_frames
    )

    preflight = ab.preflight_integrity(
        csv_map,
        npz_map,
        args.participants,
        args.calibration_start,
        args.calibration_end,
        args.min_calibration_frames,
    )

    preflight.to_csv(
        out_dir
        / "preflight_integrity.csv",
        index=False,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "V4 STRATIFIED LOSO EVALUATION v2 (SCHEMA-SAFE)"
    )
    print(
        "============================================================"
    )
    print(
        "ACTION test : upper/lower active vs PRE/POST neutral"
    )
    print(
        "CONTROL test: separate held-out control false activations"
    )
    print(
        "Threshold   : train-subject grouped OOF only"
    )
    print()
    print(
        "PREFLIGHT"
    )
    print(
        preflight.to_string(
            index=False
        )
    )

    result_rows = []
    control_rows = []
    prediction_rows = []
    fold_meta = {}

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

        pcas, pca_meta = (
            ab.fit_train_pcas(
                npz_map,
                train_subjects,
                args.dino_pca_dim,
            )
        )

        fold_meta[
            held_out
        ] = {
            "pca": pca_meta,
        }

        fold_subject_frames = {}
        protocol_schema_report = {}

        # DINO PCA columns are known from the fitted PCA objects and must be
        # present in every protocol after transformation.
        expected_dino_cols = [
            f"dino_pca_{region}_{j:02d}"
            for region in ab.DINO_REGIONS
            for j in range(
                int(
                    pcas[
                        region
                    ].n_components_
                )
            )
        ]

        required_common_features = (
            blendshape_cols
            + geometry_cols
            + motion_cols
            + pure_nuisance_cols
            + expected_dino_cols
        )

        for participant in args.participants:
            labeled_pieces = []

            for protocol in PROTOCOLS:
                transformed = (
                    ab.add_pca_features_to_recording(
                        csv_map[
                            (
                                participant,
                                protocol,
                            )
                        ],
                        npz_map[
                            (
                                participant,
                                protocol,
                            )
                        ],
                        pcas,
                    )
                )

                labeled_pieces.append(
                    (
                        protocol,
                        transformed.reset_index(
                            drop=True
                        ),
                    )
                )

            (
                labeled_pieces,
                schema_only,
            ) = validate_protocol_union(
                labeled_pieces,
                participant,
                required_common_features,
            )

            protocol_schema_report[
                participant
            ] = schema_only

            # IMPORTANT:
            # Annotation schemas legitimately differ by protocol.
            # concat(sort=False) takes the union; model features were already
            # verified common above.
            fold_subject_frames[
                participant
            ] = pd.concat(
                [
                    frame
                    for _, frame in labeled_pieces
                ],
                ignore_index=True,
                sort=False,
            )

            fold_subject_frames[
                participant
            ] = ab.ensure_unique_columns(
                fold_subject_frames[
                    participant
                ],
                f"{participant}/stratified/all_protocols",
            )

            # A recording is allowed to restart frame_idx at zero, so uniqueness
            # is checked per recording rather than on the concatenated subject.
            for recording_key, recording in fold_subject_frames[
                participant
            ].groupby(
                "recording_key",
                sort=False,
            ):
                ab.assert_unique_frame_idx(
                    recording,
                    f"{participant}/{recording_key}",
                )

        fold_meta[
            held_out
        ][
            "protocol_annotation_schema_only"
        ] = protocol_schema_report

        if held_out == args.participants[0]:
            print(
                "  Protocol annotation-schema differences are allowed:"
            )
            for participant, report in protocol_schema_report.items():
                compact = {
                    protocol: columns
                    for protocol, columns in report.items()
                    if columns
                }
                print(
                    f"    {participant}: {compact}"
                )

        dino_cols = sorted(
            c
            for c in fold_subject_frames[
                held_out
            ].columns
            if c.startswith(
                "dino_pca_"
            )
        )

        if dino_cols != sorted(
            expected_dino_cols
        ):
            missing = sorted(
                set(
                    expected_dino_cols
                )
                - set(
                    dino_cols
                )
            )
            extra = sorted(
                set(
                    dino_cols
                )
                - set(
                    expected_dino_cols
                )
            )
            raise RuntimeError(
                f"{held_out}: DINO PCA schema mismatch. "
                f"missing={missing}, extra={extra}"
            )

        modality_sets = {
            "geometry_only": geometry_cols,
            "blendshape_geometry": (
                blendshape_cols
                + geometry_cols
            ),
            "blendshape_geometry_motion": (
                blendshape_cols
                + geometry_cols
                + motion_cols
            ),
            "blendshape_geometry_motion_dino": (
                blendshape_cols
                + geometry_cols
                + motion_cols
                + dino_cols
            ),
            "pure_nuisance_only": (
                pure_nuisance_cols
            ),
            "all_plus_pure_nuisance": (
                blendshape_cols
                + geometry_cols
                + motion_cols
                + dino_cols
                + pure_nuisance_cols
            ),
        }

        train_df = pd.concat(
            [
                fold_subject_frames[p]
                for p in train_subjects
            ],
            ignore_index=True,
            sort=False,
        )

        test_df = (
            fold_subject_frames[
                held_out
            ]
            .copy()
        )

        # Outer LOSO integrity checks.
        train_participants_seen = set(
            train_df[
                "participant"
            ].astype(str)
        )
        test_participants_seen = set(
            test_df[
                "participant"
            ].astype(str)
        )

        if held_out in train_participants_seen:
            raise RuntimeError(
                f"{held_out}: held-out subject leaked into train_df."
            )

        if test_participants_seen != {
            held_out
        }:
            raise RuntimeError(
                f"{held_out}: test_df contains unexpected participants "
                f"{sorted(test_participants_seen)}"
            )

        reference_window_ids = {}

        for modality, requested_features in modality_sets.items():
            (
                feature_cols,
                dropped,
            ) = feature_qc_from_train(
                train_df,
                requested_features,
                args.min_window_frames,
            )

            if dropped:
                print(
                    f"  [{modality}] dropped: "
                    + ", ".join(
                        dropped
                    )
                )

            global_cal = (
                ab.calibration_frame(
                    train_df
                )
            )

            global_stats = (
                ab.robust_stats(
                    global_cal,
                    feature_cols,
                )
            )

            personal_stats = (
                ab.build_personal_stats_by_recording(
                    pd.concat(
                        [
                            train_df,
                            test_df,
                        ],
                        ignore_index=True,
                        sort=False,
                    ),
                    feature_cols,
                    args.min_calibration_frames,
                )
            )

            train_global = ab.z_transform(
                train_df,
                feature_cols,
                global_stats,
            )

            test_global = ab.z_transform(
                test_df,
                feature_cols,
                global_stats,
            )

            train_personal = (
                ab.personal_z_transform(
                    train_df,
                    feature_cols,
                    personal_stats,
                )
            )

            test_personal = (
                ab.personal_z_transform(
                    test_df,
                    feature_cols,
                    personal_stats,
                )
            )

            windows = make_mode_windows(
                train_df,
                test_df,
                train_global,
                test_global,
                train_personal,
                test_personal,
                feature_cols,
                args,
            )

            # Fair modality comparison: exact same held-out action/control windows.
            for split_name in [
                "train",
                "test",
            ]:
                for challenge_name in [
                    "action",
                    "control",
                ]:
                    _, _, meta = windows[
                        (
                            split_name,
                            challenge_name,
                            "global",
                        )
                    ]

                    ids = (
                        meta[
                            "window_id"
                        ].tolist()
                        if len(meta)
                        else []
                    )

                    key = (
                        split_name,
                        challenge_name,
                    )

                    if key not in reference_window_ids:
                        reference_window_ids[
                            key
                        ] = ids
                    elif (
                        ids
                        != reference_window_ids[
                            key
                        ]
                    ):
                        raise RuntimeError(
                            f"{held_out}/{modality}: "
                            f"{split_name}/{challenge_name} windows differ "
                            "across modalities."
                        )

            for normalization in [
                "global",
                "personal",
                "hybrid",
            ]:
                (
                    Xa_train,
                    ya_train,
                    Ma_train,
                ) = windows[
                    (
                        "train",
                        "action",
                        normalization,
                    )
                ]

                (
                    Xa_test,
                    ya_test,
                    Ma_test,
                ) = windows[
                    (
                        "test",
                        "action",
                        normalization,
                    )
                ]

                (
                    Xc_train,
                    yc_train,
                    Mc_train,
                ) = windows[
                    (
                        "train",
                        "control",
                        normalization,
                    )
                ]

                (
                    Xc_test,
                    yc_test,
                    Mc_test,
                ) = windows[
                    (
                        "test",
                        "control",
                        normalization,
                    )
                ]

                ensure_two_classes(
                    ya_train,
                    f"{held_out}/{modality}/{normalization}/action_train",
                )

                ensure_two_classes(
                    ya_test,
                    f"{held_out}/{modality}/{normalization}/action_test",
                )

                # Control should be all-negative by design.
                if len(
                    np.unique(
                        yc_train
                    )
                ) > 1:
                    raise RuntimeError(
                        f"{held_out}: train control unexpectedly has positives."
                    )

                if len(
                    np.unique(
                        yc_test
                    )
                ) > 1:
                    raise RuntimeError(
                        f"{held_out}: test control unexpectedly has positives."
                    )

                for training_regime in [
                    "neutral_only",
                    "with_control_hardneg",
                ]:
                    if training_regime == "neutral_only":
                        X_train = Xa_train
                        y_train = ya_train
                        M_train = Ma_train.copy()
                    else:
                        X_train = np.concatenate(
                            [
                                Xa_train,
                                Xc_train,
                            ],
                            axis=0,
                        )

                        y_train = np.concatenate(
                            [
                                ya_train,
                                yc_train,
                            ],
                            axis=0,
                        )

                        M_train = pd.concat(
                            [
                                Ma_train,
                                Mc_train,
                            ],
                            ignore_index=True,
                        )

                    ensure_two_classes(
                        y_train,
                        f"{held_out}/{modality}/{normalization}/{training_regime}",
                    )

                    groups = (
                        M_train[
                            "participant"
                        ]
                        .astype(str)
                        .to_numpy()
                    )

                    group_set = set(
                        groups.tolist()
                    )

                    if group_set != set(
                        train_subjects
                    ):
                        raise RuntimeError(
                            f"{held_out}/{modality}/{normalization}/"
                            f"{training_regime}: threshold-tuning groups "
                            f"{sorted(group_set)} do not match outer-train "
                            f"subjects {sorted(train_subjects)}"
                        )

                    threshold, threshold_meta = (
                        choose_threshold_group_oof(
                            X_train,
                            y_train,
                            groups,
                            C=args.C,
                            objective=args.threshold_objective,
                        )
                    )

                    scaler, model = fit_classifier(
                        X_train,
                        y_train,
                        args.C,
                    )

                    pa_test = predict_probability(
                        scaler,
                        model,
                        Xa_test,
                    )

                    pc_test = predict_probability(
                        scaler,
                        model,
                        Xc_test,
                    )

                    action_metrics = evaluate_binary(
                        ya_test,
                        pa_test,
                        threshold,
                    )

                    (
                        event_recall,
                        detected_trials,
                        total_trials,
                    ) = active_trial_detection_rate(
                        Ma_test,
                        ya_test,
                        pa_test,
                        threshold,
                    )

                    control_metrics = (
                        count_false_activation_events(
                            Mc_test,
                            pc_test,
                            threshold,
                            args.stride_ms,
                        )
                    )

                    result_rows.append({
                        "held_out": held_out,
                        "train_subjects": "+".join(
                            train_subjects
                        ),
                        "modality": modality,
                        "normalization": normalization,
                        "training_regime": training_regime,
                        "threshold": threshold,
                        **threshold_meta,
                        **action_metrics,
                        "event_recall": event_recall,
                        "detected_active_trials": detected_trials,
                        "active_trials": total_trials,
                    })

                    control_rows.append({
                        "held_out": held_out,
                        "modality": modality,
                        "normalization": normalization,
                        "training_regime": training_regime,
                        "threshold": threshold,
                        **control_metrics,
                    })

                    pred_action = (
                        Ma_test.copy()
                        .reset_index(
                            drop=True
                        )
                    )

                    pred_action[
                        "challenge"
                    ] = "action"

                    pred_action[
                        "modality"
                    ] = modality

                    pred_action[
                        "normalization"
                    ] = normalization

                    pred_action[
                        "training_regime"
                    ] = training_regime

                    pred_action[
                        "threshold"
                    ] = threshold

                    pred_action[
                        "probability"
                    ] = pa_test

                    pred_action[
                        "prediction"
                    ] = (
                        pa_test >= threshold
                    ).astype(int)

                    pred_control = (
                        Mc_test.copy()
                        .reset_index(
                            drop=True
                        )
                    )

                    pred_control[
                        "challenge"
                    ] = "control"

                    pred_control[
                        "modality"
                    ] = modality

                    pred_control[
                        "normalization"
                    ] = normalization

                    pred_control[
                        "training_regime"
                    ] = training_regime

                    pred_control[
                        "threshold"
                    ] = threshold

                    pred_control[
                        "probability"
                    ] = pc_test

                    pred_control[
                        "prediction"
                    ] = (
                        pc_test >= threshold
                    ).astype(int)

                    prediction_rows.extend(
                        [
                            pred_action,
                            pred_control,
                        ]
                    )

    results = pd.DataFrame(
        result_rows
    )

    control = pd.DataFrame(
        control_rows
    )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    summary = (
        results
        .groupby(
            [
                "modality",
                "normalization",
                "training_regime",
            ],
            as_index=False,
        )
        .agg(
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            f1_mean=(
                "f1",
                "mean",
            ),
            auroc_mean=(
                "auroc",
                "mean",
            ),
            auprc_mean=(
                "auprc",
                "mean",
            ),
            specificity_mean=(
                "specificity",
                "mean",
            ),
            event_recall_mean=(
                "event_recall",
                "mean",
            ),
            threshold_mean=(
                "threshold",
                "mean",
            ),
        )
    )

    control_summary = (
        control
        .groupby(
            [
                "modality",
                "normalization",
                "training_regime",
            ],
            as_index=False,
        )
        .agg(
            control_positive_window_fraction_mean=(
                "control_positive_window_fraction",
                "mean",
            ),
            false_activations_per_min_mean=(
                "control_false_activations_per_min",
                "mean",
            ),
        )
    )

    combined = summary.merge(
        control_summary,
        on=[
            "modality",
            "normalization",
            "training_regime",
        ],
        how="left",
        validate="one_to_one",
    )

    combined = combined.sort_values(
        [
            "auprc_mean",
            "false_activations_per_min_mean",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    results.to_csv(
        out_dir
        / "action_results.csv",
        index=False,
    )

    control.to_csv(
        out_dir
        / "control_results.csv",
        index=False,
    )

    combined.to_csv(
        out_dir
        / "stratified_ranking.csv",
        index=False,
    )

    predictions.to_csv(
        out_dir
        / "predictions.csv",
        index=False,
    )

    config = {
        "participants": args.participants,
        "session": args.session,
        "dino_pca_dim": args.dino_pca_dim,
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "calibration_start_s": args.calibration_start,
        "calibration_end_s": args.calibration_end,
        "min_face_coverage": args.min_face_coverage,
        "min_dino_coverage": args.min_dino_coverage,
        "threshold_objective": args.threshold_objective,
        "threshold_policy": (
            "participant-grouped OOF on outer-train subjects only"
        ),
        "protocol_schema_policy": (
            "annotation columns may differ across upper/lower/control; "
            "all model features and structural columns must be common"
        ),
        "window_integrity_policy": (
            "recording_key + segment_key grouping inherited from the "
            "integrity-checked ablation helper"
        ),
        "action_challenge": (
            "upper/lower active vs pre_neutral/post_neutral"
        ),
        "control_challenge": (
            "held-out control recording evaluated separately as all-negative"
        ),
        "training_regimes": [
            "neutral_only",
            "with_control_hardneg",
        ],
        "fold_meta": fold_meta,
    }

    with open(
        out_dir
        / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.set_option(
        "display.width",
        260,
    )

    pd.set_option(
        "display.max_columns",
        40,
    )

    print()
    print(
        "============================================================"
    )
    print(
        "STRATIFIED RANKING"
    )
    print(
        "============================================================"
    )

    print(
        combined.to_string(
            index=False
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "PURE NUISANCE DIAGNOSTIC"
    )
    print(
        "============================================================"
    )

    nuisance = combined[
        combined[
            "modality"
        ].eq(
            "pure_nuisance_only"
        )
    ]

    print(
        nuisance.to_string(
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

    print()
    print(
        "Decision rule:"
    )
    print(
        "  - Choose the core modality by ACTION AUPRC / event recall."
    )
    print(
        "  - Choose hard-negative training only if CONTROL false activations/min drops"
    )
    print(
        "    without materially damaging action AUPRC."
    )
    print(
        "  - If pure_nuisance_only remains strong on ACTION-only evaluation,"
    )
    print(
        "    inspect within-action head/gaze coupling before training a TCN."
    )


if __name__ == "__main__":
    main()
