from __future__ import annotations

"""
V4 modality ablation with leakage-safe LOSO evaluation.

Purpose
-------
Identify which label-free modality helps or hurts cross-subject facial-action
event detection BEFORE moving to a temporal neural network.

Modalities evaluated
--------------------
1) blendshape_only
2) geometry_only
3) dino_only
4) nuisance_only                  # sanity/confound check
5) blendshape_geometry
6) blendshape_dino
7) geometry_dino
8) all_core                       # blendshape + geometry + DINO
9) all_plus_nuisance              # all_core + pose/gaze/blink/motion

For every modality:
    global
    personal
    hybrid

Leakage control
---------------
- LOSO: held-out subject is never used for classifier fitting.
- DINO PCA is fitted on TRAIN SUBJECTS ONLY for each fold and region.
- Personal calibration uses only a fixed neutral time interval (default 1-7 s)
  from that recording.
- Calibration frames are excluded from classifier train/test windows.
- Windows require sufficient face and DINO coverage.
- No action-selected v3 geometry or action-selected DINO ROI is used.

Primary interpretation
----------------------
Use this script to decide:
- whether DINO adds transferable signal,
- whether geometry helps beyond MediaPipe blendshapes,
- whether nuisance variables improve specificity,
- whether nuisance-only performance is suspiciously high (protocol confound),
- whether personal normalization helps consistently.

This is an engineering benchmark, not clinical validation.
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

PURE_NUISANCE_CANDIDATES = [
    "face_ratio",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "blink",
    "gaze_horizontal",
    "gaze_vertical",
]

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


def ensure_unique_columns(
    df: pd.DataFrame,
    context: str,
) -> pd.DataFrame:
    """
    Pandas vertical concat requires unique column labels.

    If duplicate columns are byte-for-byte/equivalent, keep the first.
    If duplicates disagree, fail loudly instead of silently selecting one.
    """
    if df.columns.is_unique:
        return df

    positions_by_name = {}
    for i, name in enumerate(df.columns):
        positions_by_name.setdefault(str(name), []).append(i)

    keep_positions = []
    conflicts = []

    for name, positions in positions_by_name.items():
        keep_positions.append(positions[0])

        if len(positions) <= 1:
            continue

        reference = df.iloc[:, positions[0]]

        for pos in positions[1:]:
            other = df.iloc[:, pos]

            same = (
                reference.equals(other)
                or reference.astype(str).equals(other.astype(str))
            )

            if not same:
                conflicts.append(
                    (
                        name,
                        positions,
                    )
                )
                break

    if conflicts:
        raise RuntimeError(
            f"{context}: conflicting duplicate columns: {conflicts[:10]}"
        )

    cleaned = df.iloc[
        :,
        sorted(keep_positions),
    ].copy()

    if not cleaned.columns.is_unique:
        raise RuntimeError(
            f"{context}: could not make columns unique."
        )

    return cleaned


def assert_unique_frame_idx(
    df: pd.DataFrame,
    context: str,
):
    frame_idx = pd.to_numeric(
        df["frame_idx"],
        errors="coerce",
    )

    if frame_idx.isna().any():
        raise RuntimeError(
            f"{context}: frame_idx contains NaN/non-numeric values."
        )

    duplicated = frame_idx.duplicated(
        keep=False
    )

    if duplicated.any():
        examples = (
            frame_idx[
                duplicated
            ]
            .astype(int)
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{context}: duplicate frame_idx values; examples={examples}"
        )


def validate_npz(
    data,
    context: str,
):
    frame_idx = np.asarray(
        data["frame_idx"]
    )

    if frame_idx.ndim != 1:
        raise RuntimeError(
            f"{context}: NPZ frame_idx must be 1-D, got {frame_idx.shape}"
        )

    if len(
        np.unique(
            frame_idx
        )
    ) != len(frame_idx):
        raise RuntimeError(
            f"{context}: duplicate DINO frame_idx values."
        )

    dims = []

    for region in DINO_REGIONS:
        arr = np.asarray(
            data[region]
        )

        if arr.ndim != 2:
            raise RuntimeError(
                f"{context}: {region} must be 2-D, got {arr.shape}"
            )

        if len(arr) != len(frame_idx):
            raise RuntimeError(
                f"{context}: {region} rows={len(arr)} "
                f"but frame_idx rows={len(frame_idx)}"
            )

        dims.append(
            int(
                arr.shape[1]
            )
        )

    if len(
        set(dims)
    ) != 1:
        raise RuntimeError(
            f"{context}: DINO region dimensions disagree: {dims}"
        )


def preflight_integrity(
    csv_map,
    npz_map,
    participants,
    calibration_start_s,
    calibration_end_s,
    min_calibration_frames,
):
    rows = []

    for participant in participants:
        for protocol in PROTOCOLS:
            df = csv_map[
                (
                    participant,
                    protocol,
                )
            ]

            data = npz_map[
                (
                    participant,
                    protocol,
                )
            ]

            context = (
                f"{participant}/{protocol}"
            )

            if not df.columns.is_unique:
                raise RuntimeError(
                    f"{context}: non-unique columns survived load."
                )

            assert_unique_frame_idx(
                df,
                context,
            )

            validate_npz(
                data,
                context,
            )

            timestamp = pd.to_numeric(
                df[
                    "analysis_timestamp_ms"
                ],
                errors="coerce",
            )

            if timestamp.isna().any():
                raise RuntimeError(
                    f"{context}: invalid analysis_timestamp_ms."
                )

            if not timestamp.is_monotonic_increasing:
                raise RuntimeError(
                    f"{context}: timestamps are not monotonic."
                )

            time_s = (
                timestamp
                / 1000.0
            )

            cal_mask = (
                time_s.ge(
                    calibration_start_s
                )
                & time_s.le(
                    calibration_end_s
                )
            )

            if "face_detected" in df.columns:
                face = (
                    pd.to_numeric(
                        df[
                            "face_detected"
                        ],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(int)
                    .eq(1)
                )
            else:
                face = pd.Series(
                    True,
                    index=df.index,
                )

            cal_face = (
                cal_mask
                & face
            )

            if int(
                cal_face.sum()
            ) < min_calibration_frames:
                raise RuntimeError(
                    f"{context}: only {int(cal_face.sum())} detected-face "
                    f"calibration frames in {calibration_start_s:.1f}-"
                    f"{calibration_end_s:.1f}s; need >= "
                    f"{min_calibration_frames}."
                )

            csv_frames = set(
                pd.to_numeric(
                    df["frame_idx"],
                    errors="coerce",
                )
                .dropna()
                .astype(int)
                .tolist()
            )

            dino_frames = set(
                np.asarray(
                    data["frame_idx"],
                    dtype=int,
                ).tolist()
            )

            overlap = (
                len(
                    csv_frames
                    & dino_frames
                )
                / max(
                    1,
                    len(dino_frames),
                )
            )

            finite_by_region = {}

            for region in DINO_REGIONS:
                arr = np.asarray(
                    data[region],
                    dtype=np.float32,
                )

                finite_by_region[
                    region
                ] = float(
                    np.isfinite(
                        arr
                    ).mean()
                )

            rows.append({
                "participant": participant,
                "protocol": protocol,
                "csv_frames": len(df),
                "face_detection_rate": float(
                    face.mean()
                ),
                "calibration_rows": int(
                    cal_mask.sum()
                ),
                "calibration_face_rows": int(
                    cal_face.sum()
                ),
                "dino_updates": int(
                    len(
                        data["frame_idx"]
                    )
                ),
                "dino_frame_overlap": overlap,
                "dino_dim": int(
                    np.asarray(
                        data["global"]
                    ).shape[1]
                ),
                "dino_finite_min": min(
                    finite_by_region.values()
                ),
            })

            if overlap < 0.999:
                raise RuntimeError(
                    f"{context}: DINO frame_idx overlap with CSV is only "
                    f"{overlap:.3f}."
                )

            if min(
                finite_by_region.values()
            ) < 0.999:
                raise RuntimeError(
                    f"{context}: non-finite DINO embedding values detected: "
                    f"{finite_by_region}"
                )

    report = pd.DataFrame(
        rows
    )

    return report


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

    df = pd.read_csv(
        path
    )

    df = ensure_unique_columns(
        df,
        str(path),
    ).copy()

    # Assignment replaces existing metadata columns instead of creating
    # duplicate labels such as protocol/protocol.
    df = df.assign(
        participant=participant,
        protocol=protocol,
        recording_key=(
            participant
            + "|"
            + session
            + "|"
            + protocol
        ),
    )

    required = [
        "frame_idx",
        "analysis_timestamp_ms",
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

    assert_unique_frame_idx(
        df,
        str(path),
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

    validate_npz(
        data,
        str(path),
    )

    return data


def label_targets(
    df: pd.DataFrame,
    calibration_start_s: float,
    calibration_end_s: float,
) -> pd.DataFrame:
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

    df["is_calibration"] = (
        time_s.ge(calibration_start_s)
        & time_s.le(calibration_end_s)
    )

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


def discover_scalar_groups(
    all_frames: list[pd.DataFrame],
):
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

    pure_nuisance = [
        c
        for c in PURE_NUISANCE_CANDIDATES
        if c in common
    ]

    motion = [
        c
        for c in MOTION_CANDIDATES
        if c in common
    ]

    if not blendshape:
        raise RuntimeError(
            "No common bs_* columns found."
        )

    if not geometry:
        raise RuntimeError(
            "No common geom_abs_* columns found."
        )

    if not pure_nuisance:
        raise RuntimeError(
            "No pure nuisance columns found."
        )

    if not motion:
        raise RuntimeError(
            "No motion columns found."
        )

    return (
        blendshape,
        geometry,
        motion,
        pure_nuisance,
    )

def collect_train_region_matrix(
    npz_map,
    train_subjects,
    region,
):
    arrays = []

    for participant in train_subjects:
        for protocol in PROTOCOLS:
            arr = np.asarray(
                npz_map[
                    (participant, protocol)
                ][region],
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
            f"No finite training DINO data for region={region}"
        )

    return np.vstack(
        arrays
    )


def fit_train_pcas(
    npz_map,
    train_subjects,
    requested_dim,
):
    pcas = {}
    meta = {}

    for region in DINO_REGIONS:
        x = collect_train_region_matrix(
            npz_map,
            train_subjects,
            region,
        )

        n_components = min(
            int(requested_dim),
            x.shape[1],
            max(
                1,
                x.shape[0] - 1,
            ),
        )

        pca = PCA(
            n_components=n_components,
            random_state=0,
        )

        pca.fit(
            x
        )

        pcas[region] = pca
        meta[region] = {
            "n_components": int(
                n_components
            ),
            "explained_variance_ratio_sum": float(
                np.sum(
                    pca.explained_variance_ratio_
                )
            ),
        }

    return pcas, meta


def add_pca_features_to_recording(
    df: pd.DataFrame,
    data,
    pcas,
) -> pd.DataFrame:
    df = ensure_unique_columns(
        df.copy(),
        "add_pca_features_to_recording/input",
    )

    assert_unique_frame_idx(
        df,
        "add_pca_features_to_recording/input",
    )

    validate_npz(
        data,
        "add_pca_features_to_recording/npz",
    )

    dino_frame_idx = np.asarray(
        data["frame_idx"],
        dtype=np.int64,
    )

    sparse_parts = [
        pd.DataFrame(
            {
                "frame_idx": dino_frame_idx,
            }
        )
    ]

    pca_columns = []

    for region in DINO_REGIONS:
        raw = np.asarray(
            data[region],
            dtype=np.float32,
        )

        pca = pcas[region]

        transformed = np.full(
            (
                len(raw),
                int(pca.n_components_),
            ),
            np.nan,
            dtype=np.float32,
        )

        if len(raw):
            good = np.isfinite(
                raw
            ).all(axis=1)

            if good.any():
                transformed[good] = (
                    pca.transform(
                        raw[good]
                    )
                    .astype(np.float32)
                )

        names = [
            f"dino_pca_{region}_{j:02d}"
            for j in range(
                transformed.shape[1]
            )
        ]

        sparse_parts.append(
            pd.DataFrame(
                transformed,
                columns=names,
            )
        )

        pca_columns.extend(
            names
        )

    sparse = pd.concat(
        sparse_parts,
        axis=1,
    )

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

        nominal_gap = (
            int(
                round(
                    np.median(gaps)
                )
            )
            if len(gaps)
            else 3
        )
    else:
        nominal_gap = 3

    max_hold = max(
        2,
        2 * nominal_gap - 1,
    )

    df[pca_columns] = (
        df[pca_columns]
        .ffill(limit=max_hold)
        .bfill(limit=min(2, max_hold))
    )

    df["dino_pca_available"] = (
        df[pca_columns]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    df = ensure_unique_columns(
        df,
        "add_pca_features_to_recording/output",
    )

    assert_unique_frame_idx(
        df,
        "add_pca_features_to_recording/output",
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

    return df[mask].copy()


def build_personal_stats_by_recording(
    frame,
    feature_cols,
    min_frames,
):
    result = {}

    for recording_key, group in frame.groupby(
        "recording_key",
        sort=False,
    ):
        cal = calibration_frame(
            group
        )

        # Require enough rows, and at least some usable observations in every
        # requested modality. Per-feature robust_stats handles occasional NaNs.
        if len(cal) < min_frames:
            raise RuntimeError(
                f"{recording_key}: only {len(cal)} calibration rows; "
                f"need >= {min_frames}"
            )

        finite_counts = (
            cal[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .notna()
            .sum(axis=0)
        )

        insufficient = [
            c
            for c in feature_cols
            if int(finite_counts[c]) < 5
        ]

        if insufficient:
            raise RuntimeError(
                f"{recording_key}: personal calibration has <5 finite "
                f"samples for features: {insufficient[:20]}"
            )

        result[recording_key] = robust_stats(
            cal,
            feature_cols,
        )

    return result


def personal_z_transform(
    frame,
    feature_cols,
    stats_by_recording,
):
    pieces = []

    for recording_key, group in frame.groupby(
        "recording_key",
        sort=False,
    ):
        z = z_transform(
            group,
            feature_cols,
            stats_by_recording[
                recording_key
            ],
        )

        z.index = group.index
        pieces.append(
            z
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

        idx = np.flatnonzero(
            np.isfinite(
                column
            )
        )

        if len(idx) == 0:
            continue

        values = column[idx]

        mean[j] = float(
            values.mean()
        )

        std[j] = float(
            values.std()
        )

        max_abs[j] = float(
            np.max(
                np.abs(
                    values
                )
            )
        )

        if len(idx) >= 2:
            delta[j] = float(
                column[idx[-1]]
                - column[idx[0]]
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
    df,
    normalized,
    feature_cols,
    *,
    window_ms,
    stride_ms,
    min_frames,
    min_face_coverage,
    min_dino_coverage,
    require_dino,
    normalization_name,
):
    working = df.copy()

    z_names = [
        "__z__"
        + c
        for c in feature_cols
    ]

    z_frame = normalized[
        feature_cols
    ].copy()

    z_frame.columns = z_names

    working = pd.concat(
        [
            working.reset_index(drop=True),
            z_frame.reset_index(drop=True),
        ],
        axis=1,
    )

    usable = working[
        working["target"].notna()
        & ~working[
            "is_calibration"
        ].fillna(False)
    ].copy()

    x_rows = []
    y_rows = []
    metadata = []

    for (
        recording_key,
        segment_key,
    ), group in usable.groupby(
        [
            "recording_key",
            "segment_key",
        ],
        sort=False,
    ):
        group = group.sort_values(
            "analysis_timestamp_ms"
        )

        time = pd.to_numeric(
            group["analysis_timestamp_ms"],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite_time = time[
            np.isfinite(time)
        ]

        if len(finite_time) < min_frames:
            continue

        t0 = float(
            finite_time.min()
        )

        t1 = float(
            finite_time.max()
        )

        if (
            t1 - t0
            < window_ms * 0.75
        ):
            continue

        start = t0

        while (
            start + window_ms
            <= t1 + 1e-6
        ):
            end = (
                start + window_ms
            )

            idx = np.flatnonzero(
                (time >= start)
                & (time < end)
            )

            if len(idx) >= min_frames:
                sub = group.iloc[idx]

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

                if require_dino:
                    dino_coverage = float(
                        pd.to_numeric(
                            sub.get(
                                "dino_pca_available",
                                pd.Series(
                                    0,
                                    index=sub.index,
                                ),
                            ),
                            errors="coerce",
                        )
                        .fillna(0)
                        .mean()
                    )
                else:
                    dino_coverage = 1.0

                if (
                    face_coverage >= min_face_coverage
                    and dino_coverage >= min_dino_coverage
                ):
                    y = pd.to_numeric(
                        sub["target"],
                        errors="coerce",
                    ).dropna().astype(int)

                    if (
                        len(y)
                        and y.nunique() == 1
                    ):
                        x_rows.append(
                            summarize_window(
                                sub[z_names]
                            )
                        )

                        y_rows.append(
                            int(y.iloc[0])
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
                                recording_key
                            ),
                            "segment_key": str(
                                segment_key
                            ),
                            "window_id": (
                                str(recording_key)
                                + "|"
                                + str(segment_key)
                                + "|"
                                + f"{float(start):.3f}"
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
                            "face_coverage": face_coverage,
                            "dino_coverage": dino_coverage,
                            "normalization": normalization_name,
                        })

            start += stride_ms

    if not x_rows:
        return (
            np.empty(
                (
                    0,
                    len(feature_cols) * 4,
                ),
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
            y_pred[neg] == 0
        ).mean()
    )


def evaluate(
    y_true,
    probability,
):
    pred = (
        probability >= 0.5
    ).astype(int)

    spec = specificity(
        y_true,
        pred,
    )

    out = {
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
            if np.isfinite(spec)
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

    prevalence = float(
        np.mean(
            y_true == 1
        )
    )

    out["positive_prevalence"] = prevalence
    out["auprc_lift"] = (
        float(
            out["auprc"]
            / prevalence
        )
        if (
            np.isfinite(
                out["auprc"]
            )
            and prevalence > 0
        )
        else np.nan
    )

    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "V4 leakage-safe modality ablation"
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
        "--C",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    if len(args.participants) < 3:
        raise ValueError(
            "Need at least 3 participants."
        )

    out_dir = (
        OUTPUT_ROOT
        / "v4_modality_ablation"
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

            raw_frames.append(
                df
            )

    (
        blendshape_cols,
        geometry_cols,
        motion_cols,
        pure_nuisance_cols,
    ) = discover_scalar_groups(
        raw_frames
    )

    preflight = preflight_integrity(
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
        "V4 MODALITY ABLATION v3 (MOTION / NUISANCE SPLIT)"
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
        f"Blendshape: {len(blendshape_cols)}"
    )
    print(
        f"Geometry  : {len(geometry_cols)}"
    )
    print(
        f"Motion    : {len(motion_cols)}"
    )
    print(
        f"Pure nuis.: {len(pure_nuisance_cols)}"
    )
    print(
        "DINO PCA  : "
        f"{args.dino_pca_dim} / region, TRAIN ONLY"
    )
    print(
        "Calibration: "
        f"{args.calibration_start:.1f}-{args.calibration_end:.1f}s"
    )
    print()
    print(
        "PREFLIGHT INTEGRITY"
    )
    print(
        preflight.to_string(
            index=False
        )
    )

    results = []
    predictions = []
    fold_pca_meta = {}

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

        pcas, pca_meta = fit_train_pcas(
            npz_map,
            train_subjects,
            args.dino_pca_dim,
        )

        fold_pca_meta[
            held_out
        ] = pca_meta

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

                transformed = ensure_unique_columns(
                    transformed,
                    f"{participant}/{protocol}/transformed",
                )

                pieces.append(
                    transformed.reset_index(
                        drop=True
                    )
                )

            # Protocol label schemas may legitimately differ (control vs upper/lower).
            # Vertical concat should take the union of columns; the only hard
            # requirement is that each individual DataFrame has unique labels.
            fold_subject_frames[
                participant
            ] = pd.concat(
                pieces,
                ignore_index=True,
                sort=False,
            )

            fold_subject_frames[
                participant
            ] = ensure_unique_columns(
                fold_subject_frames[
                    participant
                ],
                f"{participant}/all_protocols",
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

        modality_sets = {
            "blendshape_only": blendshape_cols,
            "geometry_only": geometry_cols,
            "dino_only": dino_cols,
            "motion_only": motion_cols,
            "pure_nuisance_only": pure_nuisance_cols,

            "blendshape_geometry": (
                blendshape_cols
                + geometry_cols
            ),

            "blendshape_geometry_motion": (
                blendshape_cols
                + geometry_cols
                + motion_cols
            ),

            "blendshape_geometry_dino": (
                blendshape_cols
                + geometry_cols
                + dino_cols
            ),

            "blendshape_geometry_motion_dino": (
                blendshape_cols
                + geometry_cols
                + motion_cols
                + dino_cols
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
        )

        test_df = (
            fold_subject_frames[
                held_out
            ].copy()
        )

        reference_train_window_ids = None
        reference_test_window_ids = None

        for modality, feature_cols in modality_sets.items():
            # Fair modality comparison requires an identical sample set.
            # Therefore even non-DINO modalities are evaluated only on windows
            # that pass the common DINO availability QC.
            require_dino = True

            global_cal = calibration_frame(
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

            usable_features = [
                c
                for c in feature_cols
                if (
                    int(
                        finite_train[
                            c
                        ]
                    ) >= args.min_window_frames
                    and int(
                        finite_cal[
                            c
                        ]
                    ) >= 5
                )
            ]

            dropped_features = [
                c
                for c in feature_cols
                if c not in usable_features
            ]

            if dropped_features:
                print(
                    f"  [{held_out}/{modality}] drop unusable train features: "
                    + ", ".join(
                        dropped_features
                    )
                )

            feature_cols = usable_features

            if not feature_cols:
                raise RuntimeError(
                    f"{held_out}/{modality}: no usable training features."
                )

            global_stats = robust_stats(
                global_cal,
                feature_cols,
            )

            all_fold = pd.concat(
                [
                    train_df,
                    test_df,
                ],
                ignore_index=True,
            )

            personal_stats = (
                build_personal_stats_by_recording(
                    all_fold,
                    feature_cols,
                    args.min_calibration_frames,
                )
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

            Xg_train, yg_train, Mg_train = make_windows(
                train_df,
                train_global,
                feature_cols,
                window_ms=args.window_ms,
                stride_ms=args.stride_ms,
                min_frames=args.min_window_frames,
                min_face_coverage=args.min_face_coverage,
                min_dino_coverage=args.min_dino_coverage,
                require_dino=require_dino,
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
                require_dino=require_dino,
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
                require_dino=require_dino,
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
                require_dino=require_dino,
                normalization_name="personal",
            )

            train_window_ids = (
                Mg_train[
                    "window_id"
                ].tolist()
                if len(Mg_train)
                else []
            )

            test_window_ids = (
                Mg_test[
                    "window_id"
                ].tolist()
                if len(Mg_test)
                else []
            )

            if reference_train_window_ids is None:
                reference_train_window_ids = train_window_ids
                reference_test_window_ids = test_window_ids
            else:
                if train_window_ids != reference_train_window_ids:
                    raise RuntimeError(
                        f"{held_out}/{modality}: modality ablation train windows differ."
                    )

                if test_window_ids != reference_test_window_ids:
                    raise RuntimeError(
                        f"{held_out}/{modality}: modality ablation test windows differ."
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
                    f"{held_out}/{modality}: train window mismatch"
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
                    f"{held_out}/{modality}: test window mismatch"
                )

            if len(
                np.unique(
                    yg_train
                )
            ) < 2:
                raise RuntimeError(
                    f"{held_out}/{modality}: training windows contain only one class."
                )

            if len(
                np.unique(
                    yg_test
                )
            ) < 2:
                raise RuntimeError(
                    f"{held_out}/{modality}: held-out windows contain only one class."
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

            for normalization, (
                X_train,
                y_train,
                X_test,
                y_test,
                test_meta,
            ) in modes.items():
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
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=0,
                )

                clf.fit(
                    X_train_s,
                    y_train,
                )

                probability = (
                    clf.predict_proba(
                        X_test_s
                    )[:, 1]
                )

                metrics = evaluate(
                    y_test,
                    probability,
                )

                results.append({
                    "held_out": held_out,
                    "train_subjects": "+".join(
                        train_subjects
                    ),
                    "modality": modality,
                    "normalization": normalization,
                    "frame_feature_count": len(
                        feature_cols
                    ),
                    "window_feature_count": int(
                        X_train.shape[1]
                    ),
                    "common_window_qc": 1,
                    **metrics,
                })

                pred = (
                    test_meta
                    .copy()
                    .reset_index(drop=True)
                )

                pred["held_out"] = held_out
                pred["modality"] = modality
                pred["normalization"] = normalization
                pred["probability"] = probability
                pred["prediction"] = (
                    probability >= 0.5
                ).astype(int)

                predictions.append(
                    pred
                )

    results_df = pd.DataFrame(
        results
    )

    predictions_df = pd.concat(
        predictions,
        ignore_index=True,
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
        "positive_prevalence",
        "auprc_lift",
    ]

    summary = (
        results_df
        .groupby(
            [
                "modality",
                "normalization",
            ]
        )[metric_cols]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    ranking = (
        results_df
        .groupby(
            [
                "modality",
                "normalization",
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
            fpr_mean=(
                "false_positive_rate",
                "mean",
            ),
            positive_prevalence_mean=(
                "positive_prevalence",
                "mean",
            ),
            auprc_lift_mean=(
                "auprc_lift",
                "mean",
            ),
        )
        .sort_values(
            [
                "auprc_mean",
                "balanced_accuracy_mean",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    results_df.to_csv(
        out_dir
        / "ablation_results.csv",
        index=False,
    )

    predictions_df.to_csv(
        out_dir
        / "ablation_predictions.csv",
        index=False,
    )

    summary.to_csv(
        out_dir
        / "ablation_summary.csv"
    )

    ranking.to_csv(
        out_dir
        / "ablation_ranking.csv",
        index=False,
    )

    config = {
        "participants": args.participants,
        "session": args.session,
        "calibration_start_s": args.calibration_start,
        "calibration_end_s": args.calibration_end,
        "min_face_coverage": args.min_face_coverage,
        "min_dino_coverage": args.min_dino_coverage,
        "dino_pca_dim": args.dino_pca_dim,
        "dino_pca_policy": (
            "fit on train subjects only, separately by region"
        ),
        "window_ms": args.window_ms,
        "stride_ms": args.stride_ms,
        "ablation_window_policy": (
            "identical windows for all modalities; every modality must pass "
            "the common face and DINO coverage thresholds"
        ),
        "segment_grouping": (
            "recording_key + segment_key; never mix participants/protocol recordings"
        ),
        "modalities": {
            "blendshape_only": len(
                blendshape_cols
            ),
            "geometry_only": len(
                geometry_cols
            ),
            "motion_only": len(
                motion_cols
            ),
            "pure_nuisance_only": len(
                pure_nuisance_cols
            ),
            "dino_only": (
                args.dino_pca_dim
                * len(DINO_REGIONS)
            ),
        },
        "pca_by_fold": fold_pca_meta,
    }

    with open(
        out_dir
        / "ablation_config.json",
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
        "ABLATION RANKING"
    )
    print(
        "============================================================"
    )

    print(
        ranking.to_string(
            index=False
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "BEST NORMALIZATION PER MODALITY (by mean AUPRC)"
    )
    print(
        "============================================================"
    )

    best = (
        ranking
        .sort_values(
            "auprc_mean",
            ascending=False,
        )
        .groupby(
            "modality",
            as_index=False,
        )
        .first()
        .sort_values(
            "auprc_mean",
            ascending=False,
        )
    )

    print(
        best.to_string(
            index=False
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        "KEY PER-FOLD RESULTS"
    )
    print(
        "============================================================"
    )

    key_modalities = [
        "pure_nuisance_only",
        "motion_only",
        "blendshape_only",
        "geometry_only",
        "blendshape_geometry",
        "blendshape_geometry_dino",
        "blendshape_geometry_motion",
        "all_plus_pure_nuisance",
    ]

    key_rows = results_df[
        results_df[
            "modality"
        ].isin(
            key_modalities
        )
    ][
        [
            "held_out",
            "modality",
            "normalization",
            "balanced_accuracy",
            "f1",
            "auroc",
            "auprc",
            "positive_prevalence",
            "auprc_lift",
            "specificity",
            "false_positive_rate",
        ]
    ].sort_values(
        [
            "held_out",
            "modality",
            "normalization",
        ]
    )

    print(
        key_rows.to_string(
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
        "Interpretation checklist:"
    )
    print(
        "  1) pure_nuisance_only AUPRC far above prevalence -> protocol/confound risk"
    )
    print(
        "  2) motion_only can be legitimately predictive because the target is a movement event"
    )
    print(
        "  3) blendshape_geometry_dino > blendshape_geometry -> DINO adds transferable signal"
    )
    print(
        "  4) blendshape_geometry_motion > blendshape_geometry -> generic motion adds useful temporal evidence"
    )
    print(
        "  5) all_plus_pure_nuisance improves specificity/FPR without hurting AUPRC -> nuisance covariates may help"
    )
    print(
        "  6) choose TCN inputs only after this split-confound ablation"
    )


if __name__ == "__main__":
    main()
