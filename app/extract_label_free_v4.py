from __future__ import annotations

"""
Label-free facial representation extractor v4.

Goal
----
Extract the SAME representation on every frame regardless of the instructed action.

This is designed for the eventual setting:
    "We do not know in advance what facial movement the child will make."

Therefore this extractor NEVER uses:
- action name to choose geometry,
- action name to choose a DINO ROI,
- trial PRE segments to construct action-specific baselines.

Instead it always emits:
1) all MediaPipe blendshapes
2) generic canonical landmark geometry for mouth / eyes / brows / nose
3) ALL DINO region representations on every DINO update:
       global / mouth / eyes / brow / nose
4) head pose / gaze / blink / generic frame-to-frame motion

A short initial neutral calibration window is used only for optional
session-relative diagnostic features. Raw label-free features are preserved.

DINO representation
-------------------
The CSV stores compact change scores.
The NPZ stores pooled 384-D DINO embeddings at DINO update frames:
    global, mouth, eyes, brow, nose

These pooled embeddings are NOT reduced with PCA here.
For LOSO experiments, PCA must be fitted on TRAIN subjects only.
This prevents test-subject leakage.

Outputs
-------
outputs/micro_expression/<participant>/<session>/
    <protocol>_signals_v4.csv
    <protocol>_dino_embeddings_v4.npz
    <protocol>_feature_metadata_v4.json

Protocols map to the existing recordings:
    control -> control_gaze.mp4 / control_gaze_labels.csv
    upper   -> upper_face.mp4   / upper_face_labels.csv
    lower   -> lower_face.mp4   / lower_face_labels.csv
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

from micro_expression_signals import (  # noqa: E402
    LEFT_BROW,
    LEFT_EYE,
    MOUTH,
    RIGHT_BROW,
    RIGHT_EYE,
    MicroExpressionSignalExtractor,
)


RECORDING_ROOT = ROOT / "data" / "micro_expression" / "recordings"
OUTPUT_ROOT = ROOT / "outputs" / "micro_expression"
VERSION = "4.0-label-free"

PROTOCOL_FILES = {
    "control": ("control_gaze.mp4", "control_gaze_labels.csv"),
    "upper": ("upper_face.mp4", "upper_face_labels.csv"),
    "lower": ("lower_face.mp4", "lower_face_labels.csv"),
}

# Stable points used for alignment. Defined locally so this script does not depend
# on additional aliases in micro_expression_signals.py.
LEFT_EYE_CENTER = [33, 133, 159, 145]
RIGHT_EYE_CENTER = [362, 263, 386, 374]

NOSE = [
    1, 2, 4, 5, 6, 19, 94, 97,
    98, 129, 168, 195, 197, 326, 327, 358,
]

REGIONS = {
    "mouth": sorted(set(MOUTH)),
    "eyes": sorted(set(LEFT_EYE + RIGHT_EYE)),
    "brow": sorted(set(LEFT_BROW + RIGHT_BROW)),
    "nose": sorted(set(NOSE)),
}

REGION_ORDER = ["mouth", "eyes", "brow", "nose"]

LIP_APERTURE_PAIRS = [
    (13, 14),
    (82, 87),
    (312, 317),
]

EYE_APERTURE_PAIRS = [
    (159, 145),
    (158, 153),
    (386, 374),
    (387, 380),
]


# ============================================================
# Utility
# ============================================================

def sf(value, default=np.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def finite(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def valid_idx(n, indices):
    return [int(i) for i in indices if 0 <= int(i) < n]


def mean_xy(points, indices):
    idx = valid_idx(len(points), indices)
    if not idx:
        return None
    pts = np.asarray(points[idx, :2], dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return None
    return pts.mean(axis=0)


def point_dist(points, i, j):
    if i >= len(points) or j >= len(points):
        return np.nan
    a = np.asarray(points[i, :2], float)
    b = np.asarray(points[j, :2], float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return np.nan
    return float(np.linalg.norm(a - b))


def mean_pair_distance(points, pairs):
    values = [point_dist(points, i, j) for i, j in pairs]
    values = finite(values)
    return float(values.mean()) if len(values) else np.nan


def topk_mean(values, fraction):
    x = finite(values)
    if not len(x):
        return np.nan

    fraction = min(
        max(float(fraction), 1.0 / len(x)),
        1.0,
    )
    k = max(
        1,
        int(math.ceil(len(x) * fraction)),
    )
    top = np.partition(
        x,
        len(x) - k,
    )[-k:]
    return float(np.mean(top))


# ============================================================
# Label loading
# ============================================================

def load_labels(path):
    labels = pd.read_csv(path)

    if "frame_idx" not in labels.columns:
        raise RuntimeError(
            f"frame_idx column missing from label file: {path}"
        )

    labels["frame_idx"] = pd.to_numeric(
        labels["frame_idx"],
        errors="coerce",
    )

    labels = labels[
        labels["frame_idx"].notna()
    ].copy()

    labels["frame_idx"] = (
        labels["frame_idx"]
        .astype(int)
    )

    # If duplicate rows somehow exist, keep the last emitted label row.
    labels = (
        labels
        .drop_duplicates(
            subset=["frame_idx"],
            keep="last",
        )
        .sort_values("frame_idx")
        .reset_index(drop=True)
    )

    return labels


# ============================================================
# Face alignment
# ============================================================

def aligned_face(frame, points, size=256):
    """
    Eye-based affine normalization.

    Returns
    -------
    crop : BGR square image
    aligned_landmarks : Nx2 coordinates normalized to [roughly] 0..1
    """
    h, w = frame.shape[:2]

    ea = mean_xy(
        points,
        LEFT_EYE_CENTER,
    )
    eb = mean_xy(
        points,
        RIGHT_EYE_CENTER,
    )

    if ea is None or eb is None:
        return None, None

    ea = np.asarray(
        [ea[0] * w, ea[1] * h],
        np.float32,
    )
    eb = np.asarray(
        [eb[0] * w, eb[1] * h],
        np.float32,
    )

    left, right = (
        (ea, eb)
        if ea[0] <= eb[0]
        else (eb, ea)
    )

    eye_vector = right - left

    if np.linalg.norm(eye_vector) < 2.0:
        return None, None

    mid = (left + right) / 2.0

    perp = np.asarray(
        [-eye_vector[1], eye_vector[0]],
        np.float32,
    )

    if perp[1] < 0:
        perp *= -1

    src3 = mid + 0.9 * perp

    s = float(size)

    dst_left = np.asarray(
        [0.30 * s, 0.36 * s],
        np.float32,
    )
    dst_right = np.asarray(
        [0.70 * s, 0.36 * s],
        np.float32,
    )

    dst_mid = (
        dst_left + dst_right
    ) / 2.0

    dst_v = (
        dst_right - dst_left
    )

    dst_perp = np.asarray(
        [-dst_v[1], dst_v[0]],
        np.float32,
    )

    if dst_perp[1] < 0:
        dst_perp *= -1

    dst3 = (
        dst_mid
        + 0.9 * dst_perp
    )

    matrix = cv2.getAffineTransform(
        np.float32(
            [left, right, src3]
        ),
        np.float32(
            [dst_left, dst_right, dst3]
        ),
    )

    crop = cv2.warpAffine(
        frame,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    xy = np.asarray(
        points[:, :2],
        np.float32,
    ).copy()

    xy[:, 0] *= w
    xy[:, 1] *= h

    aligned_px = cv2.transform(
        xy[None, :, :],
        matrix,
    )[0]

    aligned_norm = (
        aligned_px
        / float(size)
    )

    return (
        crop,
        aligned_norm.astype(
            np.float32
        ),
    )


def dino_roi_mask(
    grid_h,
    grid_w,
    aligned_points,
    region_indices,
    pad=0.06,
):
    """
    Build a patch-grid ROI from current aligned landmarks.

    No action name is used here. The caller computes ALL regions every update.
    """
    idx = valid_idx(
        len(aligned_points),
        region_indices,
    )

    if not idx:
        return np.ones(
            (grid_h, grid_w),
            dtype=bool,
        )

    pts = np.asarray(
        aligned_points[idx],
        float,
    )

    pts = pts[
        np.isfinite(pts).all(axis=1)
    ]

    if len(pts) < 2:
        return np.ones(
            (grid_h, grid_w),
            dtype=bool,
        )

    x1, y1 = np.min(
        pts,
        axis=0,
    )
    x2, y2 = np.max(
        pts,
        axis=0,
    )

    # Keep small regions large enough to include several patches.
    min_span = 0.16

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    half_w = max(
        (x2 - x1) / 2.0 + pad,
        min_span / 2.0,
    )
    half_h = max(
        (y2 - y1) / 2.0 + pad,
        min_span / 2.0,
    )

    x1 = max(
        0.0,
        cx - half_w,
    )
    x2 = min(
        1.0,
        cx + half_w,
    )
    y1 = max(
        0.0,
        cy - half_h,
    )
    y2 = min(
        1.0,
        cy + half_h,
    )

    xs = (
        np.arange(grid_w)
        + 0.5
    ) / grid_w

    ys = (
        np.arange(grid_h)
        + 0.5
    ) / grid_h

    xx, yy = np.meshgrid(
        xs,
        ys,
    )

    mask = (
        (xx >= x1)
        & (xx <= x2)
        & (yy >= y1)
        & (yy <= y2)
    )

    # Do not silently replace a tiny local ROI with the whole face.
    # Instead slightly expand around the region center.
    if int(mask.sum()) < 4:
        radius_x = max(
            1.5 / grid_w,
            half_w,
        )
        radius_y = max(
            1.5 / grid_h,
            half_h,
        )

        mask = (
            (np.abs(xx - cx) <= radius_x)
            & (np.abs(yy - cy) <= radius_y)
        )

    if int(mask.sum()) < 1:
        return np.ones(
            (grid_h, grid_w),
            dtype=bool,
        )

    return mask


# ============================================================
# Generic geometry
# ============================================================

def region_summary(canonical, indices, prefix):
    idx = valid_idx(
        len(canonical),
        indices,
    )

    if not idx:
        return {
            f"geom_abs_{prefix}_cx": np.nan,
            f"geom_abs_{prefix}_cy": np.nan,
            f"geom_abs_{prefix}_spread": np.nan,
        }

    pts = np.asarray(
        canonical[idx, :2],
        float,
    )

    pts = pts[
        np.isfinite(pts).all(axis=1)
    ]

    if not len(pts):
        return {
            f"geom_abs_{prefix}_cx": np.nan,
            f"geom_abs_{prefix}_cy": np.nan,
            f"geom_abs_{prefix}_spread": np.nan,
        }

    center = pts.mean(axis=0)

    spread = float(
        np.linalg.norm(
            pts - center,
            axis=1,
        ).mean()
    )

    return {
        f"geom_abs_{prefix}_cx": float(center[0]),
        f"geom_abs_{prefix}_cy": float(center[1]),
        f"geom_abs_{prefix}_spread": spread,
    }


def absolute_geometry(canonical):
    """
    Label-free geometry in canonical face coordinates.

    These are the primary geometry features for global-vs-personal normalization.
    They do NOT require any baseline.
    """
    out = {}

    for region in REGION_ORDER:
        out.update(
            region_summary(
                canonical,
                REGIONS[region],
                region,
            )
        )

    out.update({
        "geom_abs_mouth_width": point_dist(
            canonical,
            61,
            291,
        ),
        "geom_abs_lip_aperture": mean_pair_distance(
            canonical,
            LIP_APERTURE_PAIRS,
        ),
        "geom_abs_eye_aperture": mean_pair_distance(
            canonical,
            EYE_APERTURE_PAIRS,
        ),
        "geom_abs_inner_brow_distance": point_dist(
            canonical,
            107,
            336,
        ),
    })

    corners = valid_idx(
        len(canonical),
        [61, 291],
    )

    out["geom_abs_mouth_corner_y"] = (
        float(
            np.mean(
                canonical[
                    corners,
                    1,
                ]
            )
        )
        if corners
        else np.nan
    )

    brow_idx = valid_idx(
        len(canonical),
        REGIONS["brow"],
    )

    out["geom_abs_brow_y"] = (
        float(
            np.mean(
                canonical[
                    brow_idx,
                    1,
                ]
            )
        )
        if brow_idx
        else np.nan
    )

    return out


def region_state(
    canonical,
    template,
    indices,
):
    idx = valid_idx(
        len(canonical),
        indices,
    )

    if not idx:
        return np.nan

    now = np.asarray(
        canonical[idx, :2],
        float,
    )
    ref = np.asarray(
        template[idx, :2],
        float,
    )

    good = (
        np.isfinite(now).all(axis=1)
        & np.isfinite(ref).all(axis=1)
    )

    if not good.any():
        return np.nan

    return float(
        np.linalg.norm(
            now[good] - ref[good],
            axis=1,
        ).mean()
    )


def relative_geometry(
    canonical,
    template,
):
    """
    Session-neutral-relative geometry.

    This is useful for personal monitoring, but raw geom_abs_* features are retained
    separately so global-vs-personal normalization can be evaluated fairly.
    """
    now = absolute_geometry(
        canonical
    )
    ref = absolute_geometry(
        template
    )

    out = {}

    for key, value in now.items():
        suffix = key.removeprefix(
            "geom_abs_"
        )

        ref_value = ref.get(
            key,
            np.nan,
        )

        out[
            f"geom_delta_{suffix}"
        ] = (
            sf(value)
            - sf(ref_value)
            if np.isfinite(sf(value))
            and np.isfinite(sf(ref_value))
            else np.nan
        )

    for region in REGION_ORDER:
        out[
            f"geom_state_{region}"
        ] = region_state(
            canonical,
            template,
            REGIONS[region],
        )

    out[
        "geom_state_global"
    ] = region_state(
        canonical,
        template,
        list(
            range(
                len(canonical)
            )
        ),
    )

    return out


# ============================================================
# DINO label-free region features
# ============================================================

def normalized_feature_mean(features):
    if not features:
        return None

    base = torch.stack(
        [
            x.float()
            for x in features
        ]
    ).mean(
        dim=0
    )

    return F.normalize(
        base,
        dim=-1,
    ).cpu()


def dino_change_map(
    features,
    baseline,
):
    if (
        features is None
        or baseline is None
    ):
        return None

    similarity = (
        features.float()
        * baseline.float()
    ).sum(
        dim=-1
    )

    return (
        torch.clamp(
            1.0 - similarity,
            min=0.0,
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def pooled_embedding(
    features,
    mask,
):
    """
    Mean-pool DINO patch tokens in one anatomical region, then L2 normalize.
    Returned as float32 numpy vector.
    """
    if features is None:
        return None

    feat = features.float()

    mask_t = torch.as_tensor(
        mask,
        dtype=torch.bool,
        device=feat.device,
    )

    selected = feat[
        mask_t
    ]

    if selected.numel() == 0:
        return None

    pooled = selected.mean(
        dim=0
    )

    pooled = F.normalize(
        pooled,
        dim=0,
    )

    return (
        pooled
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def dino_update_payload(
    features,
    aligned_points,
    baseline,
    topk_fraction,
    roi_pad,
):
    """
    Compute EVERY region, independent of action identity.

    Returns
    -------
    scores : scalar session-relative features for CSV
    embeddings : pooled raw DINO embeddings for NPZ
    """
    grid_h = int(
        features.shape[0]
    )
    grid_w = int(
        features.shape[1]
    )

    masks = {
        "global": np.ones(
            (grid_h, grid_w),
            dtype=bool,
        )
    }

    for region in REGION_ORDER:
        masks[region] = dino_roi_mask(
            grid_h,
            grid_w,
            aligned_points,
            REGIONS[region],
            roi_pad,
        )

    embeddings = {}

    for region, mask in masks.items():
        embeddings[region] = pooled_embedding(
            features,
            mask,
        )

    scores = {}

    change = dino_change_map(
        features,
        baseline,
    )

    if change is None:
        for region in ["global"] + REGION_ORDER:
            scores[
                f"dino_change_{region}_mean_update"
            ] = np.nan
            scores[
                f"dino_change_{region}_topk_update"
            ] = np.nan
    else:
        for region, mask in masks.items():
            values = finite(
                change[mask]
            )

            scores[
                f"dino_change_{region}_mean_update"
            ] = (
                float(
                    np.mean(values)
                )
                if len(values)
                else np.nan
            )

            scores[
                f"dino_change_{region}_topk_update"
            ] = topk_mean(
                values,
                topk_fraction,
            )

    return (
        scores,
        embeddings,
    )


# ============================================================
# Existing extractor flattening
# ============================================================

def flatten_signal(signal):
    out = {
        "face_detected": int(
            bool(
                signal.get(
                    "face_detected",
                    False,
                )
            )
        )
    }

    scalar_keys = [
        "face_ratio",
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "blink",
        "motion_mean",
        "motion_max",
        "motion_mouth",
        "motion_left_eye",
        "motion_right_eye",
        "motion_left_brow",
        "motion_right_brow",
        "brow_up_left",
        "brow_up_right",
        "brow_down_left",
        "brow_down_right",
        "brow_vertical_left",
        "brow_vertical_right",
    ]

    for key in scalar_keys:
        out[key] = sf(
            signal.get(
                key,
                np.nan,
            )
        )

    out["motion_eyes"] = np.nanmean(
        [
            out.get(
                "motion_left_eye",
                np.nan,
            ),
            out.get(
                "motion_right_eye",
                np.nan,
            ),
        ]
    )

    out["motion_brow"] = np.nanmean(
        [
            out.get(
                "motion_left_brow",
                np.nan,
            ),
            out.get(
                "motion_right_brow",
                np.nan,
            ),
        ]
    )

    gaze = signal.get(
        "gaze"
    )

    out["gaze_horizontal"] = (
        sf(
            gaze.get(
                "horizontal"
            )
        )
        if gaze
        else np.nan
    )

    out["gaze_vertical"] = (
        sf(
            gaze.get(
                "vertical"
            )
        )
        if gaze
        else np.nan
    )

    for name, value in (
        signal
        .get(
            "blendshapes",
            {},
        )
        .items()
    ):
        out[
            f"bs_{name}"
        ] = sf(
            value
        )

    return out


# ============================================================
# Extraction state
# ============================================================

@dataclass
class ExtractionInfo:
    version: str
    protocol: str
    fps: float
    video_frames: int
    label_rows: int
    extracted_frames: int
    dino_every: int
    dino_update_hz: float
    dino_updates: int
    aligned_crop_size: int
    dino_topk_fraction: float
    dino_roi_pad: float
    baseline_start_s: float
    baseline_end_s: float
    baseline_geometry_frames: int
    baseline_dino_updates: int
    geometry_baseline_ready: bool
    dino_baseline_ready: bool
    dino_embedding_dim: int


def init_relative_nan_columns(row):
    # Absolute geometry is written immediately when a face is visible.
    relative_columns = []

    for region in REGION_ORDER:
        relative_columns.extend([
            f"geom_delta_{region}_cx",
            f"geom_delta_{region}_cy",
            f"geom_delta_{region}_spread",
            f"geom_state_{region}",
        ])

    relative_columns.extend([
        "geom_delta_mouth_width",
        "geom_delta_lip_aperture",
        "geom_delta_eye_aperture",
        "geom_delta_inner_brow_distance",
        "geom_delta_mouth_corner_y",
        "geom_delta_brow_y",
        "geom_state_global",
    ])

    for column in relative_columns:
        row[column] = np.nan

    for region in ["global"] + REGION_ORDER:
        row[
            f"dino_change_{region}_mean_update"
        ] = np.nan
        row[
            f"dino_change_{region}_topk_update"
        ] = np.nan

    row["dino_updated"] = 0
    row["baseline_window"] = 0


def finalize_geometry_baseline(
    baseline_samples,
):
    if not baseline_samples:
        return None

    return (
        np.median(
            np.stack(
                baseline_samples
            ),
            axis=0,
        )
        .astype(
            np.float32
        )
    )


# ============================================================
# Main video extraction
# ============================================================

def extract_video(
    video_path,
    labels_path,
    output_csv,
    output_npz,
    *,
    protocol,
    dino_every,
    crop_size,
    topk_fraction,
    roi_pad,
    baseline_start_s,
    baseline_end_s,
    min_geom_frames,
    min_dino_updates,
):
    labels = load_labels(
        labels_path
    )

    label_lookup = {
        int(row.frame_idx): row
        for _, row
        in labels.iterrows()
    }

    # Prefer the real wall-clock capture timestamps emitted by the
    # recording program. Container FPS can be wrong on AVFoundation
    # (for example, a ~28 FPS capture may be tagged as 15 FPS).
    #
    # The fallback preserves compatibility with older recordings that
    # do not contain capture_timestamp_ms.
    capture_timestamp_lookup = {}

    if "capture_timestamp_ms" in labels.columns:
        capture_times = pd.to_numeric(
            labels["capture_timestamp_ms"],
            errors="coerce",
        )

        for frame_value, timestamp_value in zip(
            labels["frame_idx"],
            capture_times,
        ):
            if np.isfinite(timestamp_value):
                capture_timestamp_lookup[
                    int(frame_value)
                ] = int(
                    round(
                        float(timestamp_value)
                    )
                )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"cannot open video: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if (
        not np.isfinite(fps)
        or fps <= 1
    ):
        fps = 30.0

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        "============================================================"
    )
    print(
        f"LABEL-FREE V4: {protocol.upper()}"
    )
    print(
        "============================================================"
    )
    print(
        f"Video     : {video_path}"
    )
    print(
        f"Labels    : {labels_path}"
    )
    print(
        f"FPS       : {fps:.3f}"
    )
    print(
        f"Frames    : {total}"
    )
    print(
        "Baseline  : "
        f"{baseline_start_s:.2f} - "
        f"{baseline_end_s:.2f} s"
    )
    print(
        f"DINO every: {dino_every} frames "
        f"(~{fps / dino_every:.2f} Hz)"
    )

    if abs(
        total - len(labels)
    ) > 2:
        print(
            "[WARN] video/label frame mismatch: "
            f"{total} vs {len(labels)}"
        )

    # Disable the old extractor's own face-crop DINO path.
    # frame_idx + 1 also prevents an accidental modulo hit at frame zero.
    extractor = MicroExpressionSignalExtractor(
        dino_every=10**9
    )

    rows = []

    # Baseline source samples.
    geom_baseline_samples = []
    dino_baseline_samples = []

    # Frames seen before the baseline was finalized are stored so their
    # session-relative features can be backfilled without a second video pass.
    pending_geom = []
    pending_dino = []

    geom_template = None
    dino_baseline = None

    # DINO pooled embeddings are stored at the sparse DINO update cadence.
    embedding_frames = []
    embedding_times = []
    embedding_regions = {
        region: []
        for region in (
            ["global"]
            + REGION_ORDER
        )
    }

    frame_idx = 0
    dino_update_count = 0
    embedding_dim = 0

    def apply_geom_to_row(
        row_index,
        canonical,
    ):
        if geom_template is None:
            return

        rows[
            row_index
        ].update(
            relative_geometry(
                canonical,
                geom_template,
            )
        )

    def apply_dino_to_row(
        row_index,
        features,
        aligned_points,
    ):
        nonlocal dino_update_count, embedding_dim

        scores, embeddings = (
            dino_update_payload(
                features,
                aligned_points,
                dino_baseline,
                topk_fraction,
                roi_pad,
            )
        )

        rows[
            row_index
        ].update(
            scores
        )

        rows[
            row_index
        ][
            "dino_updated"
        ] = 1

        embedding_frames.append(
            int(
                rows[
                    row_index
                ][
                    "frame_idx"
                ]
            )
        )

        embedding_times.append(
            int(
                rows[
                    row_index
                ][
                    "analysis_timestamp_ms"
                ]
            )
        )

        for region in (
            ["global"]
            + REGION_ORDER
        ):
            vector = embeddings.get(
                region
            )

            if vector is None:
                if embedding_dim <= 0:
                    # We cannot infer dimensionality yet; postpone should be rare.
                    raise RuntimeError(
                        "DINO pooled embedding unexpectedly missing "
                        "before embedding dimensionality was known."
                    )

                vector = np.full(
                    embedding_dim,
                    np.nan,
                    dtype=np.float32,
                )
            else:
                if embedding_dim <= 0:
                    embedding_dim = int(
                        vector.shape[0]
                    )

            embedding_regions[
                region
            ].append(
                vector.astype(
                    np.float16
                )
            )

        dino_update_count += 1

    def maybe_finalize_baselines(
        current_time_s,
    ):
        nonlocal geom_template, dino_baseline, pending_geom, pending_dino

        if (
            current_time_s
            <= baseline_end_s
        ):
            return

        if geom_template is None:
            if (
                len(
                    geom_baseline_samples
                )
                < min_geom_frames
            ):
                raise RuntimeError(
                    "Too few neutral geometry baseline frames: "
                    f"{len(geom_baseline_samples)} "
                    f"(need >= {min_geom_frames})"
                )

            geom_template = (
                finalize_geometry_baseline(
                    geom_baseline_samples
                )
            )

            print(
                "Geometry neutral baseline: "
                f"{len(geom_baseline_samples)} frames"
            )

            for (
                row_index,
                canonical,
            ) in pending_geom:
                apply_geom_to_row(
                    row_index,
                    canonical,
                )

            pending_geom = []

        if dino_baseline is None:
            if (
                len(
                    dino_baseline_samples
                )
                < min_dino_updates
            ):
                raise RuntimeError(
                    "Too few neutral DINO baseline updates: "
                    f"{len(dino_baseline_samples)} "
                    f"(need >= {min_dino_updates})"
                )

            dino_baseline = (
                normalized_feature_mean(
                    dino_baseline_samples
                )
            )

            print(
                "DINO neutral baseline    : "
                f"{len(dino_baseline_samples)} updates"
            )

            for (
                row_index,
                feat,
                aligned_points,
            ) in pending_dino:
                apply_dino_to_row(
                    row_index,
                    feat,
                    aligned_points,
                )

            pending_dino = []

    try:
        while True:
            ok, frame = cap.read()

            if (
                not ok
                or frame is None
            ):
                break

            timestamp_ms = (
                capture_timestamp_lookup.get(
                    frame_idx
                )
            )

            if timestamp_ms is None:
                timestamp_ms = int(
                    round(
                        frame_idx
                        / fps
                        * 1000.0
                    )
                )

            time_s = (
                timestamp_ms
                / 1000.0
            )

            # Freeze neutral references immediately after the predefined
            # initial calibration interval.
            maybe_finalize_baselines(
                time_s
            )

            signal = extractor.extract(
                frame,
                frame_idx + 1,
                timestamp_ms,
            )

            label_row = (
                label_lookup.get(
                    frame_idx
                )
            )

            row = {
                "frame_idx": frame_idx,
                "analysis_timestamp_ms": timestamp_ms,
                "protocol": protocol,
                "extractor_version": VERSION,
            }

            if label_row is not None:
                for column in labels.columns:
                    if column == "frame_idx":
                        continue
                    row[
                        column
                    ] = label_row[
                        column
                    ]

            row.update(
                flatten_signal(
                    signal
                )
            )

            init_relative_nan_columns(
                row
            )

            in_baseline = (
                baseline_start_s
                <= time_s
                <= baseline_end_s
            )

            row[
                "baseline_window"
            ] = int(
                in_baseline
            )

            detected = bool(
                signal.get(
                    "face_detected",
                    False,
                )
            )

            canonical = signal.get(
                "canonical_landmarks"
            )

            points = signal.get(
                "landmarks"
            )

            # ------------------------------------------------
            # Generic geometry: always compute the same set.
            # ------------------------------------------------
            if (
                detected
                and canonical is not None
            ):
                canonical_np = np.asarray(
                    canonical,
                    np.float32,
                ).copy()

                row.update(
                    absolute_geometry(
                        canonical_np
                    )
                )

                # Append row before relative features so backfill can address it.
                current_row_index = len(
                    rows
                )

                if (
                    in_baseline
                    and geom_template is None
                ):
                    geom_baseline_samples.append(
                        canonical_np
                    )

                if geom_template is None:
                    pending_geom.append(
                        (
                            current_row_index,
                            canonical_np,
                        )
                    )
                else:
                    row.update(
                        relative_geometry(
                            canonical_np,
                            geom_template,
                        )
                    )

            rows.append(
                row
            )

            current_row_index = (
                len(rows) - 1
            )

            # ------------------------------------------------
            # Label-free DINO: all anatomical regions, always.
            # ------------------------------------------------
            if (
                detected
                and points is not None
                and frame_idx
                % dino_every
                == 0
            ):
                crop, aligned_points = aligned_face(
                    frame,
                    np.asarray(
                        points,
                        np.float32,
                    ),
                    crop_size,
                )

                if (
                    crop is not None
                    and aligned_points is not None
                ):
                    feat = (
                        extractor
                        .dino
                        .extract(
                            crop
                        )
                        .clone()
                    )

                    if (
                        in_baseline
                        and dino_baseline is None
                    ):
                        dino_baseline_samples.append(
                            feat
                        )

                    if dino_baseline is None:
                        pending_dino.append(
                            (
                                current_row_index,
                                feat,
                                aligned_points.copy(),
                            )
                        )
                    else:
                        apply_dino_to_row(
                            current_row_index,
                            feat,
                            aligned_points,
                        )

            frame_idx += 1

            if frame_idx % 300 == 0:
                print(
                    f"  {frame_idx}/{total}"
                )

    finally:
        cap.release()
        extractor.close()

    # Defensive finalization for a short/cut recording.
    maybe_finalize_baselines(
        baseline_end_s
        + 1.0
    )

    df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # DINO sparse update -> held values for ordinary tabular use.
    # Keep *_update columns intact so models can also respect cadence.
    # --------------------------------------------------------
    dino_update_columns = [
        c
        for c in df.columns
        if (
            c.startswith(
                "dino_change_"
            )
            and c.endswith(
                "_update"
            )
        )
    ]

    for column in dino_update_columns:
        held_name = (
            column
            .removesuffix(
                "_update"
            )
        )

        df[
            held_name
        ] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .ffill()
        )

    # Baseline-relative change should be approximately zero before the first
    # sparse DINO update. Do not backfill from the future.
    held_columns = [
        c
        for c in df.columns
        if (
            c.startswith(
                "dino_change_"
            )
            and not c.endswith(
                "_update"
            )
        )
    ]

    for column in held_columns:
        df[
            column
        ] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .fillna(0.0)
        )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Save raw pooled DINO embeddings at update cadence.
    # Float16 keeps storage manageable. PCA is intentionally deferred.
    # --------------------------------------------------------
    npz_payload = {
        "frame_idx": np.asarray(
            embedding_frames,
            dtype=np.int32,
        ),
        "analysis_timestamp_ms": np.asarray(
            embedding_times,
            dtype=np.int64,
        ),
    }

    for region in (
        ["global"]
        + REGION_ORDER
    ):
        values = embedding_regions[
            region
        ]

        if values:
            npz_payload[
                region
            ] = np.stack(
                values
            ).astype(
                np.float16
            )
        else:
            npz_payload[
                region
            ] = np.empty(
                (
                    0,
                    max(
                        embedding_dim,
                        0,
                    ),
                ),
                dtype=np.float16,
            )

    np.savez_compressed(
        output_npz,
        **npz_payload,
    )

    info = ExtractionInfo(
        version=VERSION,
        protocol=protocol,
        fps=fps,
        video_frames=total,
        label_rows=len(labels),
        extracted_frames=len(df),
        dino_every=dino_every,
        dino_update_hz=fps / dino_every,
        dino_updates=dino_update_count,
        aligned_crop_size=crop_size,
        dino_topk_fraction=topk_fraction,
        dino_roi_pad=roi_pad,
        baseline_start_s=baseline_start_s,
        baseline_end_s=baseline_end_s,
        baseline_geometry_frames=len(
            geom_baseline_samples
        ),
        baseline_dino_updates=len(
            dino_baseline_samples
        ),
        geometry_baseline_ready=(
            geom_template is not None
        ),
        dino_baseline_ready=(
            dino_baseline is not None
        ),
        dino_embedding_dim=embedding_dim,
    )

    print()
    print(
        f"CSV       : {output_csv}"
    )
    print(
        f"DINO NPZ  : {output_npz}"
    )
    print(
        f"Rows      : {len(df)}"
    )
    print(
        f"DINO upd. : {dino_update_count}"
    )
    print(
        f"DINO dim  : {embedding_dim}"
    )
    print(
        "Label-free: yes "
        "(no action-selected geometry/ROI)"
    )

    return (
        df,
        info,
    )


# ============================================================
# Protocol runner
# ============================================================

def run_protocol(
    args,
    protocol,
):
    video_name, label_name = (
        PROTOCOL_FILES[
            protocol
        ]
    )

    input_dir = (
        RECORDING_ROOT
        / args.participant
        / args.session
    )

    video_path = (
        input_dir
        / video_name
    )

    labels_path = (
        input_dir
        / label_name
    )

    if (
        not video_path.exists()
        or not labels_path.exists()
    ):
        print(
            f"[SKIP] missing {protocol}: "
            f"{video_path.name} / "
            f"{labels_path.name}"
        )
        return

    output_dir = (
        OUTPUT_ROOT
        / args.participant
        / args.session
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        output_dir
        / f"{protocol}_signals_v4.csv"
    )

    output_npz = (
        output_dir
        / f"{protocol}_dino_embeddings_v4.npz"
    )

    metadata_path = (
        output_dir
        / f"{protocol}_feature_metadata_v4.json"
    )

    _, info = extract_video(
        video_path,
        labels_path,
        output_csv,
        output_npz,
        protocol=protocol,
        dino_every=args.dino_every,
        crop_size=args.aligned_crop_size,
        topk_fraction=args.dino_topk,
        roi_pad=args.dino_roi_pad,
        baseline_start_s=args.baseline_start,
        baseline_end_s=args.baseline_end,
        min_geom_frames=args.min_baseline_geometry_frames,
        min_dino_updates=args.min_baseline_dino_updates,
    )

    metadata = {
        "extraction": asdict(
            info
        ),
        "design": {
            "action_label_used_for_feature_selection": False,
            "trial_pre_baseline_used": False,
            "neutral_reference": (
                "fixed initial time calibration window"
            ),
            "geometry": (
                "all generic regions + raw absolute canonical geometry"
            ),
            "dino": (
                "all regions every update; pooled embeddings saved before PCA"
            ),
            "pca_policy": (
                "fit PCA on TRAIN subjects only in downstream LOSO"
            ),
            "regions": (
                ["global"]
                + REGION_ORDER
            ),
        },
        "files": {
            "signals_csv": str(
                output_csv
            ),
            "dino_embeddings_npz": str(
                output_npz
            ),
        },
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Metadata  : {metadata_path}"
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Label-free facial representation extractor v4"
        )
    )

    parser.add_argument(
        "--participant",
        required=True,
        help="e.g. p1",
    )

    parser.add_argument(
        "--session",
        default="s01",
    )

    parser.add_argument(
        "--protocol",
        choices=[
            "all",
            "control",
            "upper",
            "lower",
        ],
        default="all",
    )

    parser.add_argument(
        "--dino-every",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--aligned-crop-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dino-topk",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--dino-roi-pad",
        type=float,
        default=0.06,
    )

    # All current recordings begin with a long neutral period.
    # Using a fixed time window makes the representation independent of action labels.
    parser.add_argument(
        "--baseline-start",
        type=float,
        default=1.0,
        help="Seconds from video start.",
    )

    parser.add_argument(
        "--baseline-end",
        type=float,
        default=4.0,
        help="Seconds from video start.",
    )

    parser.add_argument(
        "--min-baseline-geometry-frames",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--min-baseline-dino-updates",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.dino_every < 1:
        raise ValueError(
            "--dino-every must be >= 1"
        )

    if not (
        0.0
        < args.dino_topk
        <= 1.0
    ):
        raise ValueError(
            "--dino-topk must be in (0,1]"
        )

    if not (
        0.0
        <= args.dino_roi_pad
        < 0.5
    ):
        raise ValueError(
            "--dino-roi-pad must be in [0,0.5)"
        )

    if not (
        0.0
        <= args.baseline_start
        < args.baseline_end
    ):
        raise ValueError(
            "baseline-start must be < baseline-end"
        )

    protocols = (
        [
            "control",
            "upper",
            "lower",
        ]
        if args.protocol == "all"
        else [
            args.protocol
        ]
    )

    print()
    print(
        "============================================================"
    )
    print(
        f"Label-free facial extractor {VERSION}"
    )
    print(
        "============================================================"
    )
    print(
        f"Participant : {args.participant}"
    )
    print(
        f"Session     : {args.session}"
    )
    print(
        f"Protocol    : {args.protocol}"
    )
    print(
        "Representation:"
    )
    print(
        "  MediaPipe : all blendshapes"
    )
    print(
        "  Geometry  : all generic regions"
    )
    print(
        "  DINO      : global + mouth + eyes + brow + nose"
    )
    print(
        "  Nuisance  : pose + gaze + blink"
    )
    print(
        "  Action-selected feature logic: NONE"
    )

    for protocol in protocols:
        run_protocol(
            args,
            protocol,
        )

    print()
    print(
        "v4 extraction finished."
    )


if __name__ == "__main__":
    main()
