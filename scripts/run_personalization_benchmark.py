from __future__ import annotations

"""
LOSO personalization benchmark for subtle facial-movement detection.

Compare:
  global   = population neutral baseline estimated from TRAIN subjects only
  personal = each subject's own designated neutral calibration baseline
  hybrid   = global + personal normalized representations

Target:
  0 = neutral / nuisance control
  1 = prompted facial action (onset / hold / release)

Anti-leakage:
  Current action-conditioned geometry / trial-local action-ROI DINO / action-specific
  semantic summary columns are intentionally excluded from this first benchmark.
"""

import argparse
import json
from dataclasses import dataclass
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
        "Install it with:\n"
        "  uv pip install --python .venv/bin/python scikit-learn"
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "outputs" / "micro_expression"

ACTIVE_PHASES = {"onset", "hold", "release"}
NEUTRAL_PHASES = {"pre_neutral", "post_neutral"}

CONTROL_NUISANCE_PREFIXES = (
    "BLINK",
    "GAZE_",
    "HEAD_",
    "CENTER_",
    "FINAL_CENTER",
)

EXPLICITLY_EXCLUDED_PREFIXES = (
    "score_",
    "geom_",
    "geometry_",
    "dino_",
    "semantic_",
    "motion_reference",
)

NUISANCE_COLUMNS = [
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "gaze_horizontal",
    "gaze_vertical",
    "blink",
    "face_ratio",
]

GENERIC_MOTION_COLUMNS = [
    "motion_mean",
    "motion_max",
    "motion_mouth",
    "motion_left_eye",
    "motion_right_eye",
    "motion_left_brow",
    "motion_right_brow",
    "motion_eyes",
    "motion_brow",
    "brow_up_left",
    "brow_up_right",
    "brow_down_left",
    "brow_down_right",
    "brow_vertical_left",
    "brow_vertical_right",
]


@dataclass
class BaselineStats:
    center: pd.Series
    scale: pd.Series
    n_frames: int


def robust_center_scale(
    df: pd.DataFrame,
    feature_cols: list[str],
    floor: float = 1e-3,
) -> BaselineStats:
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    center = x.median(axis=0, skipna=True)
    mad = (x - center).abs().median(axis=0, skipna=True) * 1.4826
    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    iqr = (q75 - q25) / 1.349
    scale = pd.concat([mad, iqr], axis=1).max(axis=1)
    scale = scale.fillna(floor).clip(lower=floor)
    center = center.fillna(0.0)
    return BaselineStats(center=center, scale=scale, n_frames=len(df))


def z_transform(
    df: pd.DataFrame,
    feature_cols: list[str],
    stats: BaselineStats,
) -> pd.DataFrame:
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    z = (x - stats.center[feature_cols]) / stats.scale[feature_cols]
    return z.replace([np.inf, -np.inf], np.nan)


def load_one(
    participant: str,
    session: str,
    protocol: str,
) -> pd.DataFrame | None:
    base = OUTPUT_ROOT / participant / session

    if protocol in {"upper", "lower"}:
        path = base / f"{protocol}_signals_v3.csv"
    else:
        candidates = [
            base / "control_signals_v31.csv",
            base / "control_signals_v3.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return None

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["participant"] = participant
    df["protocol"] = protocol
    df["source_path"] = str(path)

    if "analysis_timestamp_ms" not in df.columns:
        raise RuntimeError(f"analysis_timestamp_ms missing: {path}")

    return df


def add_targets_and_calibration_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = np.nan
    df["is_calibration"] = False
    df["segment_key"] = ""

    protocol = str(df["protocol"].iloc[0])

    if protocol in {"upper", "lower"}:
        phase = (
            df.get("movement_phase", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
        )
        label = (
            df.get("label", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
            .str.upper()
        )

        df.loc[phase.isin(ACTIVE_PHASES), "target"] = 1
        df.loc[phase.isin(NEUTRAL_PHASES), "target"] = 0

        calibration = label.str.contains("INITIAL_NEUTRAL", regex=False)

        if int(calibration.sum()) < 30:
            pre_idx = df.index[phase.eq("pre_neutral")]
            if len(pre_idx):
                first_action = (
                    str(df.loc[pre_idx[0], "action"])
                    if "action" in df.columns
                    else ""
                )
                first_repeat = (
                    pd.to_numeric(
                        pd.Series([df.loc[pre_idx[0], "repeat_idx"]]),
                        errors="coerce",
                    ).iloc[0]
                    if "repeat_idx" in df.columns
                    else np.nan
                )
                fallback = phase.eq("pre_neutral")
                if "action" in df.columns:
                    fallback &= (
                        df["action"].fillna("").astype(str).eq(first_action)
                    )
                if "repeat_idx" in df.columns and np.isfinite(first_repeat):
                    fallback &= pd.to_numeric(
                        df["repeat_idx"],
                        errors="coerce",
                    ).eq(int(first_repeat))
                calibration = fallback

        df.loc[calibration, "is_calibration"] = True

        action = (
            df.get("action", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
        )
        repeat_idx = pd.to_numeric(
            df.get("repeat_idx", pd.Series(np.nan, index=df.index)),
            errors="coerce",
        )

        df["segment_key"] = (
            protocol
            + "|"
            + action
            + "|R"
            + repeat_idx.fillna(-1).astype(int).astype(str)
            + "|"
            + phase
        )

    else:
        label = (
            df.get("label", pd.Series("", index=df.index))
            .fillna("")
            .astype(str)
        )
        upper_label = label.str.upper()

        is_control = upper_label.str.startswith(CONTROL_NUISANCE_PREFIXES)
        df.loc[is_control, "target"] = 0

        calibration = upper_label.eq("CENTER_BASELINE")
        if int(calibration.sum()) == 0:
            calibration = upper_label.str.contains("BASELINE", regex=False)
        df.loc[calibration, "is_calibration"] = True

        df["segment_key"] = "control|" + label

    return df


def select_feature_columns(frames: list[pd.DataFrame]) -> list[str]:
    common = set(frames[0].columns)
    for df in frames[1:]:
        common &= set(df.columns)

    blendshape_cols = sorted(
        c for c in common if c.startswith("bs_")
    )

    generic = [
        c
        for c in GENERIC_MOTION_COLUMNS + NUISANCE_COLUMNS
        if c in common
    ]

    selected = blendshape_cols + generic
    selected = [
        c
        for c in selected
        if not any(
            c.startswith(prefix)
            for prefix in EXPLICITLY_EXCLUDED_PREFIXES
        )
    ]

    if not blendshape_cols:
        raise RuntimeError(
            "No bs_* columns found in all signal CSVs."
        )

    return selected


def build_subject_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_frames: int,
) -> BaselineStats:
    calibration = df[df["is_calibration"].fillna(False)].copy()

    if len(calibration) < min_frames:
        fallback = df[
            (pd.to_numeric(df["target"], errors="coerce") == 0)
            & (df["protocol"].isin(["upper", "lower"]))
        ].copy()
        calibration = pd.concat(
            [calibration, fallback],
            ignore_index=True,
        )

    if len(calibration) < min_frames:
        raise RuntimeError(
            f"{df['participant'].iloc[0]} has only "
            f"{len(calibration)} calibration frames; "
            f"need >= {min_frames}."
        )

    return robust_center_scale(
        calibration,
        feature_cols,
    )


def summarize_window(values: pd.DataFrame) -> np.ndarray:
    x = values.to_numpy(dtype=float)

    with np.errstate(all="ignore"):
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        max_abs = np.nanmax(np.abs(x), axis=0)
        delta = x[-1] - x[0]

    out = np.concatenate(
        [mean, std, max_abs, delta]
    )
    return np.nan_to_num(
        out,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def make_windows(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    feature_cols: list[str],
    window_ms: float,
    stride_ms: float,
    min_frames: int,
    normalization_name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    working = df.copy()

    for c in feature_cols:
        working[f"__z__{c}"] = normalized[c].to_numpy()

    rows_x = []
    rows_y = []
    meta_rows = []

    usable = working[
        working["target"].notna()
        & ~working["is_calibration"].fillna(False)
    ].copy()

    for segment_key, group in usable.groupby(
        "segment_key",
        sort=False,
    ):
        group = group.sort_values(
            "analysis_timestamp_ms"
        )
        t = pd.to_numeric(
            group["analysis_timestamp_ms"],
            errors="coerce",
        ).to_numpy(float)

        if len(t) < min_frames:
            continue

        finite_t = t[np.isfinite(t)]
        if len(finite_t) < min_frames:
            continue

        t0 = float(finite_t.min())
        t1 = float(finite_t.max())

        if t1 - t0 < window_ms * 0.75:
            continue

        start = t0
        while start + window_ms <= t1 + 1e-6:
            end = start + window_ms
            idx = np.flatnonzero(
                (t >= start) & (t < end)
            )

            if len(idx) >= min_frames:
                sub = group.iloc[idx]
                y_values = pd.to_numeric(
                    sub["target"],
                    errors="coerce",
                ).dropna().astype(int)

                if (
                    len(y_values)
                    and y_values.nunique() == 1
                ):
                    zcols = [
                        f"__z__{c}"
                        for c in feature_cols
                    ]
                    vec = summarize_window(
                        sub[zcols]
                    )
                    rows_x.append(vec)
                    rows_y.append(
                        int(y_values.iloc[0])
                    )
                    meta_rows.append(
                        {
                            "participant": str(
                                sub["participant"].iloc[0]
                            ),
                            "protocol": str(
                                sub["protocol"].iloc[0]
                            ),
                            "segment_key": str(segment_key),
                            "target": int(
                                y_values.iloc[0]
                            ),
                            "window_start_ms": float(start),
                            "window_end_ms": float(end),
                            "normalization": normalization_name,
                            "frames": len(sub),
                        }
                    )

            start += stride_ms

    n_features = len(feature_cols) * 4
    if not rows_x:
        return (
            np.empty((0, n_features)),
            np.empty((0,), int),
            pd.DataFrame(),
        )

    return (
        np.vstack(rows_x),
        np.asarray(rows_y, dtype=int),
        pd.DataFrame(meta_rows),
    )


def specificity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    negative = y_true == 0
    if negative.sum() == 0:
        return np.nan
    return float(
        (y_pred[negative] == 0).mean()
    )


def metrics_row(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    pred = (
        probability >= threshold
    ).astype(int)

    specificity = specificity_score(
        y_true,
        pred,
    )

    out = {
        "n_windows": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "specificity": specificity,
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
        "false_positive_rate": (
            float(1.0 - specificity)
            if np.isfinite(specificity)
            else np.nan
        ),
    }

    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )
        out["auprc"] = float(
            average_precision_score(
                y_true,
                probability,
            )
        )
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan

    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LOSO global-vs-personal "
            "facial-movement benchmark"
        )
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=["p1", "p2", "p3"],
    )
    parser.add_argument(
        "--session",
        default="s01",
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
        "--min-calibration-frames",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    if len(args.participants) < 3:
        raise ValueError(
            "LOSO needs at least 3 participants."
        )

    out_dir = (
        OUTPUT_ROOT
        / "personalization_benchmark"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_subject = {}
    all_loaded = []

    for participant in args.participants:
        pieces = []

        for protocol in [
            "upper",
            "lower",
            "control",
        ]:
            df = load_one(
                participant,
                args.session,
                protocol,
            )

            if df is None:
                print(
                    f"[WARN] missing "
                    f"{participant} {protocol}"
                )
                continue

            df = add_targets_and_calibration_flags(
                df
            )
            pieces.append(df)
            all_loaded.append(df)

        if not pieces:
            raise RuntimeError(
                f"No signal files found for "
                f"{participant}"
            )

        per_subject[participant] = pd.concat(
            pieces,
            ignore_index=True,
        )

    feature_cols = select_feature_columns(
        all_loaded
    )

    print(
        "\n"
        "============================================================"
    )
    print("PERSONALIZATION BENCHMARK")
    print(
        "============================================================"
    )
    print(
        "Participants :",
        ", ".join(args.participants),
    )
    print(
        "Features     :",
        len(feature_cols),
    )
    print(
        "  blendshape :",
        sum(
            c.startswith("bs_")
            for c in feature_cols
        ),
    )
    print(
        "  motion/qc  :",
        sum(
            not c.startswith("bs_")
            for c in feature_cols
        ),
    )
    print(
        "Window       :",
        f"{args.window_ms:.0f} ms",
    )
    print(
        "Stride       :",
        f"{args.stride_ms:.0f} ms",
    )
    print()
    print(
        "Excluded to prevent protocol leakage:"
    )
    print(
        "  action-conditioned geometry / "
        "semantic summaries / trial-local DINO"
    )

    personal_stats = {}
    baseline_rows = []

    for participant, df in per_subject.items():
        stats = build_subject_baseline(
            df,
            feature_cols,
            args.min_calibration_frames,
        )
        personal_stats[participant] = stats

        baseline_rows.append(
            {
                "participant": participant,
                "personal_calibration_frames": (
                    stats.n_frames
                ),
            }
        )

    pd.DataFrame(
        baseline_rows
    ).to_csv(
        out_dir
        / "calibration_frame_counts.csv",
        index=False,
    )

    result_rows = []
    prediction_frames = []
    fold_window_counts = []

    for held_out in args.participants:
        train_subjects = [
            p
            for p in args.participants
            if p != held_out
        ]

        train_df = pd.concat(
            [
                per_subject[p]
                for p in train_subjects
            ],
            ignore_index=True,
        )
        test_df = per_subject[
            held_out
        ].copy()

        global_calibration = train_df[
            train_df["is_calibration"]
            .fillna(False)
        ].copy()

        if (
            len(global_calibration)
            < args.min_calibration_frames
        ):
            raise RuntimeError(
                f"held_out={held_out}: "
                "too few train calibration frames"
            )

        global_stats = robust_center_scale(
            global_calibration,
            feature_cols,
        )

        train_raw_parts = [
            per_subject[p]
            for p in train_subjects
        ]
        train_global_parts = [
            z_transform(
                per_subject[p],
                feature_cols,
                global_stats,
            )
            for p in train_subjects
        ]
        train_personal_parts = [
            z_transform(
                per_subject[p],
                feature_cols,
                personal_stats[p],
            )
            for p in train_subjects
        ]

        test_global = z_transform(
            test_df,
            feature_cols,
            global_stats,
        )
        test_personal = z_transform(
            test_df,
            feature_cols,
            personal_stats[held_out],
        )

        xg_train_parts = []
        yg_train_parts = []
        mg_train_parts = []

        for raw, norm in zip(
            train_raw_parts,
            train_global_parts,
        ):
            x, y, meta = make_windows(
                raw,
                norm,
                feature_cols,
                args.window_ms,
                args.stride_ms,
                args.min_window_frames,
                "global",
            )
            if len(x):
                xg_train_parts.append(x)
                yg_train_parts.append(y)
                mg_train_parts.append(meta)

        xg_test, yg_test, mg_test = (
            make_windows(
                test_df,
                test_global,
                feature_cols,
                args.window_ms,
                args.stride_ms,
                args.min_window_frames,
                "global",
            )
        )

        xp_train_parts = []
        yp_train_parts = []
        mp_train_parts = []

        for raw, norm in zip(
            train_raw_parts,
            train_personal_parts,
        ):
            x, y, meta = make_windows(
                raw,
                norm,
                feature_cols,
                args.window_ms,
                args.stride_ms,
                args.min_window_frames,
                "personal",
            )
            if len(x):
                xp_train_parts.append(x)
                yp_train_parts.append(y)
                mp_train_parts.append(meta)

        xp_test, yp_test, mp_test = (
            make_windows(
                test_df,
                test_personal,
                feature_cols,
                args.window_ms,
                args.stride_ms,
                args.min_window_frames,
                "personal",
            )
        )

        if (
            not xg_train_parts
            or not xp_train_parts
            or len(xg_test) == 0
            or len(xp_test) == 0
        ):
            raise RuntimeError(
                f"held_out={held_out}: "
                "not enough windows"
            )

        Xg_train = np.vstack(
            xg_train_parts
        )
        yg_train = np.concatenate(
            yg_train_parts
        )
        Mg_train = pd.concat(
            mg_train_parts,
            ignore_index=True,
        )

        Xp_train = np.vstack(
            xp_train_parts
        )
        yp_train = np.concatenate(
            yp_train_parts
        )
        Mp_train = pd.concat(
            mp_train_parts,
            ignore_index=True,
        )

        if (
            len(Xg_train)
            != len(Xp_train)
            or not np.array_equal(
                yg_train,
                yp_train,
            )
        ):
            raise RuntimeError(
                "Global/personal train "
                "window mismatch."
            )

        if (
            len(xg_test)
            != len(xp_test)
            or not np.array_equal(
                yg_test,
                yp_test,
            )
        ):
            raise RuntimeError(
                "Global/personal test "
                "window mismatch."
            )

        modes = {
            "global": (
                Xg_train,
                yg_train,
                xg_test,
                yg_test,
                Mg_train,
                mg_test,
            ),
            "personal": (
                Xp_train,
                yp_train,
                xp_test,
                yp_test,
                Mp_train,
                mp_test,
            ),
            "hybrid": (
                np.concatenate(
                    [Xg_train, Xp_train],
                    axis=1,
                ),
                yg_train,
                np.concatenate(
                    [xg_test, xp_test],
                    axis=1,
                ),
                yg_test,
                Mg_train.copy(),
                mg_test.copy(),
            ),
        }

        for mode_name, (
            X_train,
            y_train,
            X_test,
            y_test,
            train_meta,
            test_meta,
        ) in modes.items():
            if (
                len(np.unique(y_train))
                < 2
            ):
                raise RuntimeError(
                    f"{held_out}/{mode_name}: "
                    "train has one class only"
                )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(
                X_train
            )
            X_test_s = scaler.transform(
                X_test
            )

            clf = LogisticRegression(
                C=args.C,
                class_weight="balanced",
                max_iter=3000,
                solver="lbfgs",
                random_state=0,
            )
            clf.fit(
                X_train_s,
                y_train,
            )

            probability = clf.predict_proba(
                X_test_s
            )[:, 1]

            metrics = metrics_row(
                y_test,
                probability,
                threshold=0.5,
            )

            result_rows.append(
                {
                    "held_out": held_out,
                    "train_subjects": "+".join(
                        train_subjects
                    ),
                    "normalization": mode_name,
                    "feature_count_frame": (
                        len(feature_cols)
                    ),
                    "feature_count_window": (
                        X_train.shape[1]
                    ),
                    "global_calibration_frames": (
                        global_stats.n_frames
                    ),
                    (
                        "held_out_personal_"
                        "calibration_frames"
                    ): personal_stats[
                        held_out
                    ].n_frames,
                    **metrics,
                }
            )

            pred_meta = (
                test_meta.copy()
                .reset_index(drop=True)
            )
            pred_meta[
                "held_out"
            ] = held_out
            pred_meta[
                "normalization"
            ] = mode_name
            pred_meta[
                "probability"
            ] = probability
            pred_meta[
                "prediction"
            ] = (
                probability >= 0.5
            ).astype(int)

            prediction_frames.append(
                pred_meta
            )

            fold_window_counts.append(
                {
                    "held_out": held_out,
                    "normalization": mode_name,
                    "train_windows": len(
                        X_train
                    ),
                    "train_positive": int(
                        (y_train == 1).sum()
                    ),
                    "train_negative": int(
                        (y_train == 0).sum()
                    ),
                    "test_windows": len(
                        X_test
                    ),
                    "test_positive": int(
                        (y_test == 1).sum()
                    ),
                    "test_negative": int(
                        (y_test == 0).sum()
                    ),
                }
            )

    results = pd.DataFrame(
        result_rows
    )
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    counts = pd.DataFrame(
        fold_window_counts
    )

    results.to_csv(
        out_dir / "loso_results.csv",
        index=False,
    )
    predictions.to_csv(
        out_dir / "loso_predictions.csv",
        index=False,
    )
    counts.to_csv(
        out_dir / "window_counts.csv",
        index=False,
    )

    metric_cols = [
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positive_rate",
        "auroc",
        "auprc",
    ]

    summary = (
        results
        .groupby(
            "normalization"
        )[metric_cols]
        .agg(["mean", "std"])
    )

    summary.to_csv(
        out_dir / "loso_summary.csv"
    )

    config = {
        "participants": args.participants,
        "session": args.session,
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "min_window_frames": (
            args.min_window_frames
        ),
        "min_calibration_frames": (
            args.min_calibration_frames
        ),
        "classifier": (
            "LogisticRegression("
            "class_weight=balanced)"
        ),
        "target": {
            "positive": sorted(
                ACTIVE_PHASES
            ),
            "negative_action": sorted(
                NEUTRAL_PHASES
            ),
            "negative_control": (
                "all control nuisance/"
                "center labels"
            ),
        },
        "feature_columns": feature_cols,
        "excluded_for_leakage": [
            (
                "current action-conditioned "
                "geometry"
            ),
            (
                "current trial-local/"
                "action-ROI DINO"
            ),
            (
                "current action-specific "
                "semantic summary"
            ),
        ],
    }

    with open(
        out_dir / "benchmark_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.set_option(
        "display.max_columns",
        30,
    )
    pd.set_option(
        "display.width",
        220,
    )

    print(
        "\n"
        "============================================================"
    )
    print("LOSO RESULTS")
    print(
        "============================================================"
    )

    show_cols = [
        "held_out",
        "normalization",
        "n_windows",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positive_rate",
        "auroc",
        "auprc",
    ]

    print(
        results[
            show_cols
        ].to_string(index=False)
    )

    print(
        "\n"
        "============================================================"
    )
    print(
        "MEAN ± STD ACROSS HELD-OUT SUBJECTS"
    )
    print(
        "============================================================"
    )
    print(
        summary.to_string()
    )

    print("\nSaved to:")
    print(out_dir)

    print("\nInterpretation:")
    print(
        "  global   : population neutral baseline"
    )
    print(
        "  personal : held subject's own "
        "neutral calibration baseline"
    )
    print(
        "  hybrid   : global + personal "
        "representations"
    )
    print(
        "\nDo NOT interpret this as ASD "
        "clinical validity; it is a "
        "technical benchmark."
    )


if __name__ == "__main__":
    main()
