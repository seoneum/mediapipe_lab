from __future__ import annotations

"""
Leakage-safe LOSO benchmark v4.1 for label-free v4 facial representations.

Compares:
    global   : training-population neutral normalization
    personal : each recording's own neutral calibration normalization
    hybrid   : concatenate global + personal normalized representations

Uses:
    - all MediaPipe blendshapes
    - label-free absolute canonical geometry (geom_abs_*)
    - generic motion + head/gaze/blink nuisance covariates
    - raw pooled DINO region embeddings from *_dino_embeddings_v4.npz

Critical anti-leakage rule:
    DINO PCA is fitted on TRAIN SUBJECTS ONLY inside each LOSO fold.
    The held-out subject is transformed by that training-fitted PCA.

Target:
    1 = onset / hold / release in upper/lower action recordings
    0 = pre_neutral / post_neutral + all non-calibration control frames

Calibration frames:
    baseline_window == 1 from v4.
    They are used only for normalization and are excluded from train/test windows.

This is still a technical benchmark, not ASD clinical validation.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
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
OUTPUT_ROOT = ROOT / "outputs" / "micro_expression"

PROTOCOLS = ["upper", "lower", "control"]
DINO_REGIONS = ["global", "mouth", "eyes", "brow", "nose"]

ACTIVE_PHASES = {"onset", "hold", "release"}
NEUTRAL_PHASES = {"pre_neutral", "post_neutral"}

NUISANCE_CANDIDATES = [
    "face_ratio",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "blink",
    "gaze_horizontal",
    "gaze_vertical",
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
class RobustStats:
    center: pd.Series
    scale: pd.Series
    n_frames: int


def robust_stats(
    frame: pd.DataFrame,
    feature_cols: list[str],
    floor: float = 1e-3,
) -> RobustStats:
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce")

    center = x.median(axis=0, skipna=True)

    mad = (
        (x - center)
        .abs()
        .median(axis=0, skipna=True)
        * 1.4826
    )

    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)
    iqr_scale = (q75 - q25) / 1.349

    scale = pd.concat(
        [mad, iqr_scale],
        axis=1,
    ).max(axis=1)

    scale = (
        scale
        .fillna(floor)
        .clip(lower=floor)
    )

    center = center.fillna(0.0)

    return RobustStats(
        center=center,
        scale=scale,
        n_frames=len(frame),
    )


def z_transform(
    frame: pd.DataFrame,
    feature_cols: list[str],
    stats: RobustStats,
) -> pd.DataFrame:
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce")

    z = (
        x - stats.center[feature_cols]
    ) / stats.scale[feature_cols]

    return z.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def load_csv(
    participant: str,
    session: str,
    protocol: str,
) -> pd.DataFrame:
    path = (
        OUTPUT_ROOT
        / participant
        / session
        / f"{protocol}_signals_v4.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing v4 CSV: {path}"
        )

    df = pd.read_csv(path)
    df["participant"] = participant
    df["protocol"] = protocol
    df["recording_key"] = (
        participant
        + "|"
        + session
        + "|"
        + protocol
    )

    required = [
        "frame_idx",
        "analysis_timestamp_ms",
        "baseline_window",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path} missing columns: {missing}"
        )

    return df


def load_npz(
    participant: str,
    session: str,
    protocol: str,
):
    path = (
        OUTPUT_ROOT
        / participant
        / session
        / f"{protocol}_dino_embeddings_v4.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing v4 DINO NPZ: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    required = [
        "frame_idx",
        *DINO_REGIONS,
    ]

    missing = [
        key
        for key in required
        if key not in data.files
    ]

    if missing:
        raise RuntimeError(
            f"{path} missing arrays: {missing}"
        )

    return data


def label_targets(
    df: pd.DataFrame,
    calibration_start_s: float,
    calibration_end_s: float,
) -> pd.DataFrame:
    """
    Define calibration from absolute video time, not from the extractor's
    baseline_window flag.

    This guarantees that every subject/protocol uses the same 1--7 s (default)
    neutral interval even if v4 files were extracted with different baseline
    settings during development.
    """
    df = df.copy()

    df["target"] = np.nan
    df["segment_key"] = ""

    protocol = str(
        df["protocol"].iloc[0]
    )

    time_s = (
        pd.to_numeric(
            df["analysis_timestamp_ms"],
            errors="coerce",
        )
        / 1000.0
    )

    calibration = (
        time_s.ge(calibration_start_s)
        & time_s.le(calibration_end_s)
    )

    df["is_calibration"] = calibration

    if protocol in {"upper", "lower"}:
        phase = (
            df.get(
                "movement_phase",
                pd.Series("", index=df.index),
            )
            .fillna("")
            .astype(str)
        )

        df.loc[
            phase.isin(ACTIVE_PHASES),
            "target",
        ] = 1

        df.loc[
            phase.isin(NEUTRAL_PHASES),
            "target",
        ] = 0

        action = (
            df.get(
                "action",
                pd.Series("", index=df.index),
            )
            .fillna("")
            .astype(str)
        )

        repeat_idx = pd.to_numeric(
            df.get(
                "repeat_idx",
                pd.Series(np.nan, index=df.index),
            ),
            errors="coerce",
        )

        df["segment_key"] = (
            protocol
            + "|"
            + action
            + "|R"
            + repeat_idx.fillna(-1)
            .astype(int)
            .astype(str)
            + "|"
            + phase
        )

    else:
        # Control is nuisance/neutral by design.
        # Calibration frames will later be excluded from classifier windows.
        df["target"] = 0

        label = (
            df.get(
                "label",
                pd.Series("control", index=df.index),
            )
            .fillna("control")
            .astype(str)
        )

        df["segment_key"] = (
            "control|"
            + label
        )

    return df


def base_scalar_features(
    all_frames: list[pd.DataFrame],
) -> list[str]:
    common = set(
        all_frames[0].columns
    )

    for df in all_frames[1:]:
        common &= set(
            df.columns
        )

    blendshape = sorted(
        c
        for c in common
        if c.startswith("bs_")
    )

    geometry = sorted(
        c
        for c in common
        if c.startswith("geom_abs_")
    )

    nuisance = [
        c
        for c in NUISANCE_CANDIDATES
        if c in common
    ]

    if not blendshape:
        raise RuntimeError(
            "No common bs_* columns found."
        )

    if not geometry:
        raise RuntimeError(
            "No common geom_abs_* columns found. "
            "Make sure v4 extraction finished."
        )

    selected = (
        blendshape
        + geometry
        + nuisance
    )

    # Deliberately exclude personal/session-relative v4 columns:
    # geom_delta_*, geom_state_*, dino_change_*.
    return selected


def collect_train_region_matrix(
    npz_map: dict[tuple[str, str], object],
    train_subjects: list[str],
    protocol_list: list[str],
    region: str,
) -> np.ndarray:
    arrays = []

    for participant in train_subjects:
        for protocol in protocol_list:
            data = npz_map[
                (participant, protocol)
            ]

            arr = np.asarray(
                data[region],
                dtype=np.float32,
            )

            if (
                arr.ndim == 2
                and len(arr)
            ):
                good = np.isfinite(
                    arr
                ).all(axis=1)

                if good.any():
                    arrays.append(
                        arr[good]
                    )

    if not arrays:
        raise RuntimeError(
            f"No finite training DINO embeddings for region={region}"
        )

    return np.vstack(
        arrays
    )


def fit_train_pcas(
    npz_map,
    train_subjects,
    protocol_list,
    requested_dim,
):
    pcas = {}
    actual_dims = {}

    for region in DINO_REGIONS:
        x = collect_train_region_matrix(
            npz_map,
            train_subjects,
            protocol_list,
            region,
        )

        max_dim = min(
            int(requested_dim),
            x.shape[1],
            max(
                1,
                x.shape[0] - 1,
            ),
        )

        if max_dim < 1:
            raise RuntimeError(
                f"Cannot fit PCA for {region}"
            )

        pca = PCA(
            n_components=max_dim,
            svd_solver="auto",
            random_state=0,
        )

        pca.fit(
            x
        )

        pcas[
            region
        ] = pca

        actual_dims[
            region
        ] = max_dim

    return (
        pcas,
        actual_dims,
    )


def add_pca_features_to_recording(
    df: pd.DataFrame,
    data,
    pcas,
) -> pd.DataFrame:
    """
    Transform sparse DINO updates and hold each update forward to video frames.

    PCA is already training-fitted. No fitting occurs here.
    """
    df = df.copy()

    dino_frame_idx = np.asarray(
        data["frame_idx"],
        dtype=np.int64,
    )

    sparse = pd.DataFrame({
        "frame_idx": dino_frame_idx,
    })

    pca_columns = []

    for region in DINO_REGIONS:
        raw = np.asarray(
            data[region],
            dtype=np.float32,
        )

        pca = pcas[
            region
        ]

        transformed = np.full(
            (
                len(raw),
                int(
                    pca.n_components_
                ),
            ),
            np.nan,
            dtype=np.float32,
        )

        if len(raw):
            good = np.isfinite(
                raw
            ).all(axis=1)

            if good.any():
                transformed[
                    good
                ] = pca.transform(
                    raw[
                        good
                    ]
                ).astype(
                    np.float32
                )

        for j in range(
            transformed.shape[1]
        ):
            column = (
                f"dino_pca_{region}_{j:02d}"
            )

            sparse[
                column
            ] = transformed[
                :,
                j,
            ]

            pca_columns.append(
                column
            )

    # Duplicate sparse frame indices should not occur, but make the merge defensive.
    sparse = (
        sparse
        .drop_duplicates(
            subset=["frame_idx"],
            keep="last",
        )
        .sort_values("frame_idx")
    )

    df = df.merge(
        sparse,
        on="frame_idx",
        how="left",
        validate="one_to_one",
    )

    # DINO is intentionally lower-rate than MediaPipe.
    # Hold the latest representation forward within this recording.
    # Expected DINO cadence is sparse (e.g. every 3 video frames).
    # Hold only across short gaps. Never propagate an old DINO vector through
    # a long face-detection failure.
    if len(dino_frame_idx) >= 2:
        gaps = np.diff(
            np.sort(
                np.unique(
                    dino_frame_idx
                )
            )
        )
        gaps = gaps[
            gaps > 0
        ]
        nominal_gap = int(
            round(
                np.median(gaps)
            )
        ) if len(gaps) else 3
    else:
        nominal_gap = 3

    max_hold = max(
        2,
        2 * nominal_gap - 1,
    )

    df[
        pca_columns
    ] = (
        df[
            pca_columns
        ]
        .ffill(limit=max_hold)
        .bfill(limit=min(2, max_hold))
    )

    df["dino_pca_available"] = (
        df[pca_columns]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    return df


def calibration_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    mask = (
        df["is_calibration"]
        .fillna(False)
    )

    if "face_detected" in df.columns:
        mask &= (
            pd.to_numeric(
                df["face_detected"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1)
        )

    return df[
        mask
    ].copy()


def build_personal_stats_by_recording(
    frame: pd.DataFrame,
    feature_cols: list[str],
    min_frames: int,
) -> dict[str, RobustStats]:
    stats = {}

    for recording_key, group in frame.groupby(
        "recording_key",
        sort=False,
    ):
        cal = calibration_frame(
            group
        )

        if len(cal) < min_frames:
            raise RuntimeError(
                f"{recording_key}: only {len(cal)} calibration frames; "
                f"need >= {min_frames}"
            )

        stats[
            recording_key
        ] = robust_stats(
            cal,
            feature_cols,
        )

    return stats


def personal_z_transform(
    frame: pd.DataFrame,
    feature_cols: list[str],
    personal_stats: dict[str, RobustStats],
) -> pd.DataFrame:
    pieces = []

    for recording_key, group in frame.groupby(
        "recording_key",
        sort=False,
    ):
        if recording_key not in personal_stats:
            raise RuntimeError(
                f"No personal baseline for {recording_key}"
            )

        z = z_transform(
            group,
            feature_cols,
            personal_stats[
                recording_key
            ],
        )

        z.index = group.index
        pieces.append(
            z
        )

    if not pieces:
        return pd.DataFrame(
            index=frame.index,
            columns=feature_cols,
        )

    out = pd.concat(
        pieces,
        axis=0,
    ).sort_index()

    return out.loc[
        frame.index,
        feature_cols,
    ]


def summarize_window(
    frame: pd.DataFrame,
) -> np.ndarray:
    """
    Window summary without NumPy all-NaN warnings.

    Missing observations are represented as NaN upstream. Per feature, statistics
    are computed only from finite observations. A feature with zero finite samples
    gets neutral summary values (0 after normalization); the window itself is
    accepted only after face/DINO coverage QC in make_windows().
    """
    x = frame.to_numpy(
        dtype=float
    )

    n_features = x.shape[1]
    mean = np.zeros(
        n_features,
        dtype=float,
    )
    std = np.zeros(
        n_features,
        dtype=float,
    )
    max_abs = np.zeros(
        n_features,
        dtype=float,
    )
    delta = np.zeros(
        n_features,
        dtype=float,
    )

    for j in range(n_features):
        column = x[:, j]
        good_idx = np.flatnonzero(
            np.isfinite(column)
        )

        if len(good_idx) == 0:
            continue

        values = column[
            good_idx
        ]

        mean[j] = float(
            values.mean()
        )
        std[j] = float(
            values.std()
        )
        max_abs[j] = float(
            np.max(
                np.abs(values)
            )
        )

        if len(good_idx) >= 2:
            delta[j] = float(
                column[
                    good_idx[-1]
                ]
                - column[
                    good_idx[0]
                ]
            )

    return np.concatenate(
        [
            mean,
            std,
            max_abs,
            delta,
        ]
    )


def make_windows(
    df: pd.DataFrame,
    normalized: pd.DataFrame,
    feature_cols: list[str],
    *,
    window_ms: float,
    stride_ms: float,
    min_frames: int,
    min_face_coverage: float,
    min_dino_coverage: float,
    normalization_name: str,
):
    working = df.copy()

    z_columns = []

    for feature in feature_cols:
        z_name = (
            "__z__"
            + feature
        )

        working[
            z_name
        ] = normalized[
            feature
        ].to_numpy()

        z_columns.append(
            z_name
        )

    usable = working[
        working["target"].notna()
        & ~working["is_calibration"]
        .fillna(False)
    ].copy()

    x_rows = []
    y_rows = []
    metadata = []

    for segment_key, group in usable.groupby(
        "segment_key",
        sort=False,
    ):
        group = group.sort_values(
            "analysis_timestamp_ms"
        )

        time = pd.to_numeric(
            group[
                "analysis_timestamp_ms"
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite_time = time[
            np.isfinite(
                time
            )
        ]

        if len(
            finite_time
        ) < min_frames:
            continue

        start_time = float(
            finite_time.min()
        )

        end_time = float(
            finite_time.max()
        )

        if (
            end_time - start_time
            < window_ms * 0.75
        ):
            continue

        start = start_time

        while (
            start + window_ms
            <= end_time + 1e-6
        ):
            end = (
                start
                + window_ms
            )

            idx = np.flatnonzero(
                (time >= start)
                & (time < end)
            )

            if len(
                idx
            ) >= min_frames:
                sub = group.iloc[
                    idx
                ]

                if "face_detected" in sub.columns:
                    face_coverage = float(
                        pd.to_numeric(
                            sub["face_detected"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .mean()
                    )
                else:
                    face_coverage = 1.0

                if "dino_pca_available" in sub.columns:
                    dino_coverage = float(
                        pd.to_numeric(
                            sub["dino_pca_available"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .mean()
                    )
                else:
                    dino_coverage = 1.0

                if (
                    face_coverage < min_face_coverage
                    or dino_coverage < min_dino_coverage
                ):
                    start += stride_ms
                    continue

                y = pd.to_numeric(
                    sub["target"],
                    errors="coerce",
                ).dropna().astype(int)

                if (
                    len(y)
                    and y.nunique() == 1
                ):
                    vector = summarize_window(
                        sub[
                            z_columns
                        ]
                    )

                    x_rows.append(
                        vector
                    )

                    y_rows.append(
                        int(
                            y.iloc[0]
                        )
                    )

                    metadata.append({
                        "participant": str(
                            sub[
                                "participant"
                            ].iloc[0]
                        ),
                        "protocol": str(
                            sub[
                                "protocol"
                            ].iloc[0]
                        ),
                        "recording_key": str(
                            sub[
                                "recording_key"
                            ].iloc[0]
                        ),
                        "segment_key": str(
                            segment_key
                        ),
                        "target": int(
                            y.iloc[0]
                        ),
                        "window_start_ms": float(
                            start
                        ),
                        "window_end_ms": float(
                            end
                        ),
                        "frames": int(
                            len(sub)
                        ),
                        "face_coverage": face_coverage,
                        "dino_coverage": dino_coverage,
                        "normalization": normalization_name,
                    })

            start += stride_ms

    n_out = (
        len(feature_cols)
        * 4
    )

    if not x_rows:
        return (
            np.empty(
                (0, n_out),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=np.int64,
            ),
            pd.DataFrame(),
        )

    return (
        np.vstack(
            x_rows
        ).astype(
            np.float32
        ),
        np.asarray(
            y_rows,
            dtype=np.int64,
        ),
        pd.DataFrame(
            metadata
        ),
    )


def specificity(
    y_true,
    y_pred,
):
    neg = (
        y_true == 0
    )

    if not neg.any():
        return np.nan

    return float(
        (
            y_pred[
                neg
            ] == 0
        ).mean()
    )


def evaluate(
    y_true,
    probability,
    threshold=0.5,
):
    pred = (
        probability
        >= threshold
    ).astype(int)

    spec = specificity(
        y_true,
        pred,
    )

    result = {
        "n_windows": int(
            len(y_true)
        ),
        "n_positive": int(
            (y_true == 1).sum()
        ),
        "n_negative": int(
            (y_true == 0).sum()
        ),
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
        "specificity": spec,
        "false_positive_rate": (
            float(
                1.0 - spec
            )
            if np.isfinite(
                spec
            )
            else np.nan
        ),
        "f1": float(
            f1_score(
                y_true,
                pred,
                zero_division=0,
            )
        ),
    }

    if len(
        np.unique(
            y_true
        )
    ) == 2:
        result[
            "auroc"
        ] = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )

        result[
            "auprc"
        ] = float(
            average_precision_score(
                y_true,
                probability,
            )
        )
    else:
        result[
            "auroc"
        ] = np.nan

        result[
            "auprc"
        ] = np.nan

    return result


def concatenate_subjects(
    subject_frames: dict[str, pd.DataFrame],
    subjects: list[str],
) -> pd.DataFrame:
    return pd.concat(
        [
            subject_frames[
                subject
            ]
            for subject in subjects
        ],
        ignore_index=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LOSO v4 benchmark with train-only DINO PCA"
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
        help="PCA components per DINO region; PCA fitted only on training subjects.",
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
        default=30,
    )

    parser.add_argument(
        "--calibration-start",
        type=float,
        default=1.0,
        help="Fixed neutral calibration start in seconds for EVERY recording.",
    )

    parser.add_argument(
        "--calibration-end",
        type=float,
        default=7.0,
        help="Fixed neutral calibration end in seconds for EVERY recording.",
    )

    parser.add_argument(
        "--min-face-coverage",
        type=float,
        default=0.80,
        help="Minimum detected-face fraction required for an evaluation window.",
    )

    parser.add_argument(
        "--min-dino-coverage",
        type=float,
        default=0.70,
        help="Minimum valid DINO-PCA fraction required for an evaluation window.",
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

    if args.dino_pca_dim < 1:
        raise ValueError(
            "--dino-pca-dim must be >= 1"
        )

    out_dir = (
        OUTPUT_ROOT
        / "v4_loso_benchmark"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load v4 CSV + NPZ once.
    # --------------------------------------------------------
    csv_map = {}
    npz_map = {}
    all_csv_frames = []

    print()
    print(
        "============================================================"
    )
    print(
        "V4 LABEL-FREE LOSO BENCHMARK"
    )
    print(
        "============================================================"
    )
    print(
        "Participants:",
        ", ".join(
            args.participants
        ),
    )
    print(
        "DINO PCA   :",
        args.dino_pca_dim,
        "components / region",
    )
    print(
        "PCA policy : TRAIN SUBJECTS ONLY"
    )
    print(
        "Personal baseline scope: EACH RECORDING"
    )
    print(
        "Calibration : "
        f"{args.calibration_start:.1f}-{args.calibration_end:.1f}s "
        "(fixed for every recording)"
    )
    print(
        "Window QC   : "
        f"face>={args.min_face_coverage:.2f}, "
        f"DINO>={args.min_dino_coverage:.2f}"
    )

    for participant in args.participants:
        for protocol in PROTOCOLS:
            df = load_csv(
                participant,
                args.session,
                protocol,
            )

            df = label_targets(
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
            ] = load_npz(
                participant,
                args.session,
                protocol,
            )

            all_csv_frames.append(
                df
            )

    scalar_features = base_scalar_features(
        all_csv_frames
    )

    blendshape_features = [
        c
        for c in scalar_features
        if c.startswith(
            "bs_"
        )
    ]

    geometry_features = [
        c
        for c in scalar_features
        if c.startswith(
            "geom_abs_"
        )
    ]

    nuisance_features = [
        c
        for c in scalar_features
        if c not in blendshape_features
        and c not in geometry_features
    ]

    print()
    print(
        "Raw scalar representation:"
    )
    print(
        f"  blendshape : {len(blendshape_features)}"
    )
    print(
        f"  geometry   : {len(geometry_features)}"
    )
    print(
        f"  nuisance   : {len(nuisance_features)}"
    )

    result_rows = []
    prediction_rows = []
    count_rows = []
    fold_pca_metadata = {}

    # --------------------------------------------------------
    # LOSO
    # --------------------------------------------------------
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
            "Train   :",
            " + ".join(
                train_subjects
            ),
        )

        # Fit DINO PCA on train subjects only.
        pcas, actual_dims = fit_train_pcas(
            npz_map,
            train_subjects,
            PROTOCOLS,
            args.dino_pca_dim,
        )

        fold_pca_metadata[
            held_out
        ] = {
            region: {
                "n_components": int(
                    actual_dims[
                        region
                    ]
                ),
                "explained_variance_ratio_sum": float(
                    np.sum(
                        pcas[
                            region
                        ].explained_variance_ratio_
                    )
                ),
            }
            for region in DINO_REGIONS
        }

        for region in DINO_REGIONS:
            print(
                f"  PCA {region:>6}: "
                f"{actual_dims[region]} dims, "
                f"EVR={fold_pca_metadata[held_out][region]['explained_variance_ratio_sum']:.3f}"
            )

        # Transform every participant with this fold's train-fitted PCA.
        fold_subject_frames = {}

        for participant in args.participants:
            pieces = []

            for protocol in PROTOCOLS:
                transformed = add_pca_features_to_recording(
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

                pieces.append(
                    transformed
                )

            fold_subject_frames[
                participant
            ] = pd.concat(
                pieces,
                ignore_index=True,
            )

        dino_pca_features = sorted(
            c
            for c in fold_subject_frames[
                held_out
            ].columns
            if c.startswith(
                "dino_pca_"
            )
        )

        feature_cols = (
            scalar_features
            + dino_pca_features
        )

        train_df = concatenate_subjects(
            fold_subject_frames,
            train_subjects,
        )

        test_df = fold_subject_frames[
            held_out
        ].copy()

        # ----------------------------------------------------
        # Global baseline uses only TRAIN calibration frames.
        # ----------------------------------------------------
        global_cal = calibration_frame(
            train_df
        )

        if len(
            global_cal
        ) < args.min_calibration_frames:
            raise RuntimeError(
                f"{held_out}: too few global training calibration frames"
            )

        global_stats = robust_stats(
            global_cal,
            feature_cols,
        )

        # ----------------------------------------------------
        # Personal baseline uses each recording's own v4 neutral calibration.
        # Held-out positive/negative labels are never used for this.
        # ----------------------------------------------------
        personal_stats = build_personal_stats_by_recording(
            pd.concat(
                [
                    train_df,
                    test_df,
                ],
                ignore_index=True,
            ),
            feature_cols,
            args.min_calibration_frames,
        )

        train_global = z_transform(
            train_df,
            feature_cols,
            global_stats,
        )

        test_global = z_transform(
            test_df,
            feature_cols,
            global_stats,
        )

        train_personal = personal_z_transform(
            train_df,
            feature_cols,
            personal_stats,
        )

        test_personal = personal_z_transform(
            test_df,
            feature_cols,
            personal_stats,
        )

        # ----------------------------------------------------
        # Make the same temporal windows in each normalization.
        # ----------------------------------------------------
        Xg_train, yg_train, Mg_train = make_windows(
            train_df,
            train_global,
            feature_cols,
            window_ms=args.window_ms,
            stride_ms=args.stride_ms,
            min_frames=args.min_window_frames,
            min_face_coverage=args.min_face_coverage,
            min_dino_coverage=args.min_dino_coverage,
            normalization_name="global",
        )

        Xg_test, yg_test, Mg_test = make_windows(
            test_df,
            test_global,
            feature_cols,
            window_ms=args.window_ms,
            stride_ms=args.stride_ms,
            min_frames=args.min_window_frames,
            min_face_coverage=args.min_face_coverage,
            min_dino_coverage=args.min_dino_coverage,
            normalization_name="global",
        )

        Xp_train, yp_train, Mp_train = make_windows(
            train_df,
            train_personal,
            feature_cols,
            window_ms=args.window_ms,
            stride_ms=args.stride_ms,
            min_frames=args.min_window_frames,
            min_face_coverage=args.min_face_coverage,
            min_dino_coverage=args.min_dino_coverage,
            normalization_name="personal",
        )

        Xp_test, yp_test, Mp_test = make_windows(
            test_df,
            test_personal,
            feature_cols,
            window_ms=args.window_ms,
            stride_ms=args.stride_ms,
            min_frames=args.min_window_frames,
            min_face_coverage=args.min_face_coverage,
            min_dino_coverage=args.min_dino_coverage,
            normalization_name="personal",
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
                f"{held_out}: train global/personal window mismatch"
            )

        if (
            len(Xg_test)
            != len(Xp_test)
            or not np.array_equal(
                yg_test,
                yp_test,
            )
        ):
            raise RuntimeError(
                f"{held_out}: test global/personal window mismatch"
            )

        modes = {
            "global": (
                Xg_train,
                yg_train,
                Xg_test,
                yg_test,
                Mg_test,
            ),
            "personal": (
                Xp_train,
                yp_train,
                Xp_test,
                yp_test,
                Mp_test,
            ),
            "hybrid": (
                np.concatenate(
                    [
                        Xg_train,
                        Xp_train,
                    ],
                    axis=1,
                ),
                yg_train,
                np.concatenate(
                    [
                        Xg_test,
                        Xp_test,
                    ],
                    axis=1,
                ),
                yg_test,
                Mg_test.copy(),
            ),
        }

        for (
            mode_name,
            (
                X_train,
                y_train,
                X_test,
                y_test,
                test_meta,
            ),
        ) in modes.items():
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

            metrics = evaluate(
                y_test,
                probability,
                threshold=0.5,
            )

            result_rows.append({
                "held_out": held_out,
                "train_subjects": "+".join(
                    train_subjects
                ),
                "normalization": mode_name,
                "scalar_features": len(
                    scalar_features
                ),
                "dino_pca_features": len(
                    dino_pca_features
                ),
                "frame_feature_count": len(
                    feature_cols
                ),
                "window_feature_count": int(
                    X_train.shape[1]
                ),
                **metrics,
            })

            pred_meta = (
                test_meta
                .copy()
                .reset_index(
                    drop=True
                )
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

            prediction_rows.append(
                pred_meta
            )

            count_rows.append({
                "held_out": held_out,
                "normalization": mode_name,
                "train_windows": int(
                    len(X_train)
                ),
                "train_positive": int(
                    (y_train == 1).sum()
                ),
                "train_negative": int(
                    (y_train == 0).sum()
                ),
                "test_windows": int(
                    len(X_test)
                ),
                "test_positive": int(
                    (y_test == 1).sum()
                ),
                "test_negative": int(
                    (y_test == 0).sum()
                ),
            })

    results = pd.DataFrame(
        result_rows
    )

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    counts = pd.DataFrame(
        count_rows
    )

    metric_cols = [
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "false_positive_rate",
        "f1",
        "auroc",
        "auprc",
    ]

    summary = (
        results
        .groupby(
            "normalization"
        )[metric_cols]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    results.to_csv(
        out_dir
        / "loso_v4_results.csv",
        index=False,
    )

    predictions.to_csv(
        out_dir
        / "loso_v4_predictions.csv",
        index=False,
    )

    counts.to_csv(
        out_dir
        / "loso_v4_window_counts.csv",
        index=False,
    )

    summary.to_csv(
        out_dir
        / "loso_v4_summary.csv"
    )

    config = {
        "participants": args.participants,
        "session": args.session,
        "protocols": PROTOCOLS,
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "min_window_frames": args.min_window_frames,
        "min_calibration_frames": args.min_calibration_frames,
        "calibration_start_s": args.calibration_start,
        "calibration_end_s": args.calibration_end,
        "min_face_coverage": args.min_face_coverage,
        "min_dino_coverage": args.min_dino_coverage,
        "dino_regions": DINO_REGIONS,
        "dino_pca_requested_dim": args.dino_pca_dim,
        "dino_pca_policy": (
            "fit separately per region on training subjects only"
        ),
        "personal_baseline_scope": (
            "recording; fixed timestamp interval; face_detected==1 only"
        ),
        "target": {
            "positive": sorted(
                ACTIVE_PHASES
            ),
            "negative_action": sorted(
                NEUTRAL_PHASES
            ),
            "negative_control": (
                "all non-calibration control frames"
            ),
        },
        "scalar_features": scalar_features,
        "excluded_from_model": [
            "geom_delta_*",
            "geom_state_*",
            "dino_change_*",
            "action-selected v3 features",
        ],
        "pca_by_fold": fold_pca_metadata,
    }

    with open(
        out_dir
        / "loso_v4_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.set_option(
        "display.max_columns",
        30,
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
        "LOSO V4 RESULTS"
    )
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
        "false_positive_rate",
        "f1",
        "auroc",
        "auprc",
    ]

    print(
        results[
            show_cols
        ].to_string(
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

    print()
    print(
        "Saved to:"
    )
    print(
        out_dir
    )

    print()
    print(
        "Next decision:"
    )
    print(
        "  If v4 all-feature LOSO is stable, move to temporal TCN."
    )
    print(
        "  If not, run modality ablations before increasing model complexity."
    )


if __name__ == "__main__":
    main()
