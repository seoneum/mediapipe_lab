from __future__ import annotations

"""Single-subject subtle facial movement analyzer v3.1.

Protocol assumed for action recordings:
    INITIAL_NEUTRAL -> PRE_NEUTRAL -> ONSET -> HOLD -> RELEASE -> POST_NEUTRAL

v3 design goals
---------------
1. Detect *change from a local neutral*, not only increases in a model score.
2. Build geometry and DINO baselines separately for every action repeat.
3. Use action-specific geometry and action-region DINO evidence.
4. Keep signed deltas for later classification while using magnitude evidence for
   two-sided anomaly detection where appropriate.
5. Avoid DINO/EMA leakage across trial boundaries.
6. Report separability, polarity, repeatability and nuisance-motion QC instead of
   treating a single z-score as model accuracy.

Important terminology
---------------------
`cue_to_detection_ms` starts at the visual instruction cue. It is NOT pure
algorithm latency; it includes participant reaction time and gradual movement.

This analyzer is a technical feasibility / signal-quality tool, not a clinical
validator or diagnostic model.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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
    LEFT_EYE_CENTER,
    MOUTH,
    RIGHT_BROW,
    RIGHT_EYE,
    RIGHT_EYE_CENTER,
    MicroExpressionSignalExtractor,
)


RECORDING_ROOT = ROOT / "data" / "micro_expression" / "recordings"
OUTPUT_ROOT = ROOT / "outputs" / "micro_expression"
VERSION = "3.1"

PROTOCOL_FILES = {
    "control": ("control_gaze.mp4", "control_gaze_labels.csv"),
    "upper": ("upper_face.mp4", "upper_face_labels.csv"),
    "lower": ("lower_face.mp4", "lower_face_labels.csv"),
}

# Nose region used only for coarse state / DINO ROI.
NOSE = [1, 2, 4, 5, 6, 19, 94, 97, 98, 129, 168, 195, 197, 326, 327, 358]

REGIONS = {
    "mouth": sorted(set(MOUTH)),
    "brow": sorted(set(LEFT_BROW + RIGHT_BROW)),
    "eyes": sorted(set(LEFT_EYE + RIGHT_EYE)),
    "nose": sorted(set(NOSE)),
}

# Geometry source:
#   action_signed -> anatomically interpretable signed feature, detected two-sided
#   region_state  -> distance from trial-local neutral template, detected one-sided
ACTION_CONFIG = {
    "brows_raise": {
        "blendshapes": ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
        "semantic_reducer": "mean",
        "region": "brow",
        "geometry_source": "action_signed",
    },
    "brows_frown": {
        "blendshapes": ["browDownLeft", "browDownRight"],
        "semantic_reducer": "mean",
        "region": "brow",
        "geometry_source": "action_signed",
    },
    "eyes_squint": {
        "blendshapes": ["eyeSquintLeft", "eyeSquintRight"],
        "semantic_reducer": "mean",
        "region": "eyes",
        "geometry_source": "action_signed",
    },
    "eyes_wide": {
        "blendshapes": ["eyeWideLeft", "eyeWideRight"],
        "semantic_reducer": "mean",
        "region": "eyes",
        "geometry_source": "action_signed",
    },
    "smile": {
        "blendshapes": ["mouthSmileLeft", "mouthSmileRight"],
        "semantic_reducer": "mean",
        "region": "mouth",
        "geometry_source": "action_signed",
    },
    "mouth_frown": {
        "blendshapes": ["mouthFrownLeft", "mouthFrownRight"],
        "semantic_reducer": "mean",
        "region": "mouth",
        "geometry_source": "action_signed",
    },
    "lip_press": {
        "blendshapes": ["mouthPressLeft", "mouthPressRight"],
        "semantic_reducer": "mean",
        "region": "mouth",
        # A closed neutral mouth may have almost no aperture left to reduce.
        # Region-state is therefore safer than aperture alone.
        "geometry_source": "region_state",
    },
    "lip_pucker": {
        "blendshapes": ["mouthPucker", "mouthFunnel"],
        "semantic_reducer": "max",
        "region": "mouth",
        "geometry_source": "action_signed",
    },
    "jaw_open": {
        "blendshapes": ["jawOpen"],
        "semantic_reducer": "mean",
        "region": "mouth",
        "geometry_source": "action_signed",
    },
    "nose_wrinkle": {
        "blendshapes": ["noseSneerLeft", "noseSneerRight"],
        "semantic_reducer": "mean",
        "region": "nose",
        "geometry_source": "region_state",
    },
}

REGION_MOTION = {
    "mouth": "motion_mouth",
    "brow": "motion_brow",
    "eyes": "motion_eyes",
    "nose": "motion_mean",  # extractor currently has no dedicated nose motion
}


# ============================================================
# Control protocol pairing (v3.1)
# ============================================================
# Each nuisance phase is evaluated against the *stable tail* of the immediately
# preceding center/neutral phase.  This prevents slow resting-face drift from
# being mistaken for a facial action.
CONTROL_TARGET_BASELINES = {
    "BLINK": "CENTER_NEUTRAL_1",
    "GAZE_LEFT": "CENTER_NEUTRAL_2",
    "GAZE_RIGHT": "CENTER_AFTER_GAZE_LEFT",
    "GAZE_UP": "CENTER_AFTER_GAZE_RIGHT",
    "GAZE_DOWN": "CENTER_AFTER_GAZE_UP",
    "HEAD_LEFT": "CENTER_AFTER_GAZE_DOWN",
    "HEAD_RIGHT": "CENTER_AFTER_HEAD_LEFT",
    "HEAD_UP": "CENTER_AFTER_HEAD_RIGHT",
    "HEAD_DOWN": "CENTER_AFTER_HEAD_UP",
    # Recovery sanity check: both phases should be neutral/centered.
    "FINAL_CENTER": "CENTER_AFTER_HEAD_DOWN",
}

CONTROL_TARGET_KIND = {
    "BLINK": "blink",
    "GAZE_LEFT": "gaze",
    "GAZE_RIGHT": "gaze",
    "GAZE_UP": "gaze",
    "GAZE_DOWN": "gaze",
    "HEAD_LEFT": "head",
    "HEAD_RIGHT": "head",
    "HEAD_UP": "head",
    "HEAD_DOWN": "head",
    "FINAL_CENTER": "recovery",
}

CONTROL_BASELINE_TO_TARGET = {
    baseline: target
    for target, baseline in CONTROL_TARGET_BASELINES.items()
}


# ============================================================
# Numeric helpers
# ============================================================


def sf(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def finite(values):
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def med(values):
    x = finite(values)
    return float(np.median(x)) if len(x) else np.nan


def robust_scale(values):
    """Robust spread estimate using max(MAD-scale, IQR-scale)."""
    x = finite(values)
    if len(x) < 5:
        return np.nan
    center = np.median(x)
    mad_scale = 1.4826 * np.median(np.abs(x - center))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr_scale = (q75 - q25) / 1.349
    return float(max(mad_scale, iqr_scale))


def auc_rank(negative, positive):
    """Mann-Whitney/rank ROC AUC, no sklearn dependency."""
    neg = finite(negative)
    pos = finite(positive)
    if len(neg) < 2 or len(pos) < 2:
        return np.nan
    both = np.concatenate([neg, pos])
    ranks = pd.Series(both).rank(method="average").to_numpy(float)
    n0, n1 = len(neg), len(pos)
    u = ranks[n0:].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n0 * n1))


def rank_corr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 4:
        return np.nan
    xr = pd.Series(x[good]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[good]).rank(method="average").to_numpy(float)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def valid_idx(n, indices):
    return [i for i in indices if 0 <= i < n]


def mean_xy(points, indices):
    idx = valid_idx(len(points), indices)
    return points[idx, :2].mean(axis=0) if idx else None


def point_dist(points, i, j):
    if i >= len(points) or j >= len(points):
        return np.nan
    return float(np.linalg.norm(points[i, :2] - points[j, :2]))


def mean_pair_distance(points, pairs):
    values = [point_dist(points, a, b) for a, b in pairs]
    return med(values)


# ============================================================
# Causal smoothing / detection timing
# ============================================================


def causal_ema(values, times_ms, tau_ms):
    """Causal EMA: future frames never influence onset timing."""
    x = np.asarray(values, float)
    t = np.asarray(times_ms, float)
    out = np.full(len(x), np.nan)
    prev = np.nan
    prev_t = np.nan
    tau = max(float(tau_ms), 1e-6)

    for i, value in enumerate(x):
        if not np.isfinite(value):
            continue
        if not np.isfinite(prev):
            prev = value
            prev_t = t[i]
            out[i] = value
            continue
        dt = max(0.0, t[i] - prev_t)
        alpha = 1.0 - math.exp(-dt / tau)
        prev = alpha * value + (1.0 - alpha) * prev
        prev_t = t[i]
        out[i] = prev
    return out


def sparse_ema_hold(values, update_mask, times_ms, tau_ms):
    values = np.asarray(values, float)
    updates = np.asarray(update_mask, bool)
    times = np.asarray(times_ms, float)
    sparse = np.full(len(values), np.nan)
    idx = np.flatnonzero(updates & np.isfinite(values))
    if not len(idx):
        return sparse.copy(), sparse.copy()
    sparse[idx] = causal_ema(values[idx], times[idx], tau_ms)
    held = pd.Series(sparse).ffill().to_numpy(float)
    return sparse, held


def first_sustained(times_ms, detected, duration_ms):
    t = np.asarray(times_ms, float)
    d = np.asarray(detected, bool)
    start = None
    for i, on in enumerate(d):
        if on:
            if start is None:
                start = i
            if t[i] - t[start] >= duration_ms:
                return float(t[start])
        else:
            start = None
    return np.nan


def first_n_updates(times_ms, detected, n):
    t = np.asarray(times_ms, float)
    d = np.asarray(detected, bool)
    run = 0
    start = None
    for i, on in enumerate(d):
        if on:
            if run == 0:
                start = i
            run += 1
            if run >= n:
                return float(t[start])
        else:
            run = 0
            start = None
    return np.nan


# ============================================================
# Labels / trial identity
# ============================================================


def load_labels(path):
    df = pd.read_csv(path)
    if "frame_idx" not in df.columns:
        raise RuntimeError(f"frame_idx missing: {path}")
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="raise").astype(int)
    return df


def label_value(row, key, default=""):
    if row is None or key not in row or pd.isna(row[key]):
        return default
    return row[key]


def phase_elapsed_s(row):
    return sf(label_value(row, "phase_elapsed_ms", 0.0), 0.0) / 1000.0


def is_initial_baseline(row):
    if row is None:
        return False
    label = str(label_value(row, "label", "")).upper()
    return label == "INITIAL_NEUTRAL" or "BASELINE" in label


def trial_key(row):
    if row is None:
        return None
    action = str(label_value(row, "action", ""))
    if action not in ACTION_CONFIG:
        return None
    repeat = sf(label_value(row, "repeat_idx", np.nan))
    if not np.isfinite(repeat) or int(repeat) < 1:
        return None
    return action, int(repeat)


def movement_phase(row):
    return str(label_value(row, "movement_phase", ""))


# ============================================================
# Face alignment + DINO ROI
# ============================================================


def aligned_face(frame, points, size=256):
    """Eye-based affine alignment; returns crop and aligned landmark xy in [0,1]."""
    h, w = frame.shape[:2]
    ea = mean_xy(points, LEFT_EYE_CENTER)
    eb = mean_xy(points, RIGHT_EYE_CENTER)
    if ea is None or eb is None:
        return None, None

    ea = np.asarray([ea[0] * w, ea[1] * h], np.float32)
    eb = np.asarray([eb[0] * w, eb[1] * h], np.float32)
    left, right = (ea, eb) if ea[0] <= eb[0] else (eb, ea)
    eye_vector = right - left
    if np.linalg.norm(eye_vector) < 2.0:
        return None, None

    mid = (left + right) / 2.0
    perp = np.asarray([-eye_vector[1], eye_vector[0]], np.float32)
    if perp[1] < 0:
        perp *= -1
    src3 = mid + 0.9 * perp

    s = float(size)
    dst_left = np.asarray([0.30 * s, 0.36 * s], np.float32)
    dst_right = np.asarray([0.70 * s, 0.36 * s], np.float32)
    dst_mid = (dst_left + dst_right) / 2.0
    dst_v = dst_right - dst_left
    dst_perp = np.asarray([-dst_v[1], dst_v[0]], np.float32)
    if dst_perp[1] < 0:
        dst_perp *= -1
    dst3 = dst_mid + 0.9 * dst_perp

    matrix = cv2.getAffineTransform(
        np.float32([left, right, src3]),
        np.float32([dst_left, dst_right, dst3]),
    )
    crop = cv2.warpAffine(
        frame,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    xy = np.asarray(points[:, :2], np.float32).copy()
    xy[:, 0] *= w
    xy[:, 1] *= h
    aligned_px = cv2.transform(xy[None, :, :], matrix)[0]
    aligned_norm = aligned_px / float(size)
    return crop, aligned_norm.astype(np.float32)


def dino_roi_mask(grid_h, grid_w, aligned_points, region_indices, pad=0.06):
    """Dynamic ROI mask on the DINO patch grid using aligned facial landmarks."""
    idx = valid_idx(len(aligned_points), region_indices)
    if not idx:
        return np.ones((grid_h, grid_w), dtype=bool)

    pts = np.asarray(aligned_points[idx], float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return np.ones((grid_h, grid_w), dtype=bool)

    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)

    # Ensure narrow anatomical regions still cover several patches.
    min_span = 0.16
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max((x2 - x1) / 2.0 + pad, min_span / 2.0)
    half_h = max((y2 - y1) / 2.0 + pad, min_span / 2.0)
    x1, x2 = max(0.0, cx - half_w), min(1.0, cx + half_w)
    y1, y2 = max(0.0, cy - half_h), min(1.0, cy + half_h)

    xs = (np.arange(grid_w) + 0.5) / grid_w
    ys = (np.arange(grid_h) + 0.5) / grid_h
    xx, yy = np.meshgrid(xs, ys)
    mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)

    if mask.sum() < 4:
        return np.ones((grid_h, grid_w), dtype=bool)
    return mask


def normalized_feature_mean(features):
    if len(features) < 1:
        return None
    base = torch.stack(features).mean(dim=0)
    return F.normalize(base.float(), dim=-1).cpu()


def dino_change_map(features, baseline):
    if features is None or baseline is None:
        return None
    similarity = (features.float() * baseline.float()).sum(dim=-1)
    return torch.clamp(1.0 - similarity, min=0.0).cpu().numpy().astype(np.float32)


def topk_mean(values, fraction):
    x = finite(values)
    if not len(x):
        return np.nan
    fraction = min(max(float(fraction), 1.0 / len(x)), 1.0)
    k = max(1, int(math.ceil(len(x) * fraction)))
    top = np.partition(x, len(x) - k)[-k:]
    return float(np.mean(top))


def dino_scores(change_map, roi_mask, topk_fraction):
    if change_map is None:
        return {
            "dino_global_mean_update": np.nan,
            "dino_global_topk_update": np.nan,
            "dino_roi_mean_update": np.nan,
            "dino_roi_topk_update": np.nan,
        }
    whole = finite(np.asarray(change_map, float).reshape(-1))
    roi = finite(np.asarray(change_map, float)[roi_mask])
    return {
        "dino_global_mean_update": float(np.mean(whole)) if len(whole) else np.nan,
        "dino_global_topk_update": topk_mean(whole, topk_fraction),
        "dino_roi_mean_update": float(np.mean(roi)) if len(roi) else np.nan,
        "dino_roi_topk_update": topk_mean(roi, topk_fraction),
    }


# ============================================================
# Trial-local geometry
# ============================================================


LIP_APERTURE_PAIRS = [(13, 14), (82, 87), (312, 317)]
EYE_APERTURE_PAIRS = [(159, 145), (158, 153), (386, 374), (387, 380)]


def region_state(canonical, template, region_indices):
    idx = valid_idx(len(canonical), region_indices)
    if not idx:
        return np.nan
    delta = canonical[idx, :2] - template[idx, :2]
    return float(np.linalg.norm(delta, axis=1).mean())


def geometry_diagnostics(canonical, template):
    mouth_width_now = point_dist(canonical, 61, 291)
    mouth_width_ref = point_dist(template, 61, 291)
    mouth_width_delta = mouth_width_now - mouth_width_ref

    lip_ap_now = mean_pair_distance(canonical, LIP_APERTURE_PAIRS)
    lip_ap_ref = mean_pair_distance(template, LIP_APERTURE_PAIRS)
    lip_aperture_delta = lip_ap_now - lip_ap_ref

    eye_now = mean_pair_distance(canonical, EYE_APERTURE_PAIRS)
    eye_ref = mean_pair_distance(template, EYE_APERTURE_PAIRS)
    eye_aperture_delta = eye_now - eye_ref

    brow_idx = valid_idx(len(canonical), REGIONS["brow"])
    brow_y_now = float(np.mean(canonical[brow_idx, 1])) if brow_idx else np.nan
    brow_y_ref = float(np.mean(template[brow_idx, 1])) if brow_idx else np.nan
    brow_up_delta = brow_y_ref - brow_y_now

    inner_now = point_dist(canonical, 107, 336)
    inner_ref = point_dist(template, 107, 336)
    inner_brow_contraction = inner_ref - inner_now

    corners = valid_idx(len(canonical), [61, 291])
    corner_y_now = float(np.mean(canonical[corners, 1])) if corners else np.nan
    corner_y_ref = float(np.mean(template[corners, 1])) if corners else np.nan
    corner_up_delta = corner_y_ref - corner_y_now

    return {
        "geom_mouth_width_delta_local": sf(mouth_width_delta),
        "geom_lip_aperture_delta_local": sf(lip_aperture_delta),
        "geom_eye_aperture_delta_local": sf(eye_aperture_delta),
        "geom_brow_up_delta_local": sf(brow_up_delta),
        "geom_inner_brow_contraction_local": sf(inner_brow_contraction),
        "geom_mouth_corner_up_delta_local": sf(corner_up_delta),
    }


def action_geometry_score(action, diagnostics, region_state_value):
    """Signed score: positive is the expected anatomical direction when defined."""
    if action == "brows_raise":
        return diagnostics["geom_brow_up_delta_local"]
    if action == "brows_frown":
        down = -diagnostics["geom_brow_up_delta_local"]
        contract = diagnostics["geom_inner_brow_contraction_local"]
        return med([down, contract])
    if action == "eyes_squint":
        return -diagnostics["geom_eye_aperture_delta_local"]
    if action == "eyes_wide":
        return diagnostics["geom_eye_aperture_delta_local"]
    if action == "smile":
        return med([
            diagnostics["geom_mouth_width_delta_local"],
            diagnostics["geom_mouth_corner_up_delta_local"],
        ])
    if action == "mouth_frown":
        return -diagnostics["geom_mouth_corner_up_delta_local"]
    if action == "lip_press":
        # Diagnostic only; primary geometry for lip_press is region_state.
        return -diagnostics["geom_lip_aperture_delta_local"]
    if action == "lip_pucker":
        return -diagnostics["geom_mouth_width_delta_local"]
    if action == "jaw_open":
        return diagnostics["geom_lip_aperture_delta_local"]
    if action == "nose_wrinkle":
        return region_state_value
    return np.nan


def fill_local_geometry(row, canonical, template, action):
    cfg = ACTION_CONFIG[action]
    region = cfg["region"]
    region_value = region_state(canonical, template, REGIONS[region])
    global_value = region_state(canonical, template, list(range(len(canonical))))
    diagnostics = geometry_diagnostics(canonical, template)
    action_signed = action_geometry_score(action, diagnostics, region_value)

    row.update(diagnostics)
    row["geom_region_state_local"] = sf(region_value)
    row["geom_global_state_local"] = sf(global_value)
    row["geom_action_signed_local"] = sf(action_signed)
    row["geometry_raw"] = (
        sf(region_value)
        if cfg["geometry_source"] == "region_state"
        else sf(action_signed)
    )
    row["geometry_detection_mode"] = (
        "one-sided-high"
        if cfg["geometry_source"] == "region_state"
        else "two-sided"
    )


GEOMETRY_COLUMNS = [
    "geom_mouth_width_delta_local",
    "geom_lip_aperture_delta_local",
    "geom_eye_aperture_delta_local",
    "geom_brow_up_delta_local",
    "geom_inner_brow_contraction_local",
    "geom_mouth_corner_up_delta_local",
    "geom_region_state_local",
    "geom_global_state_local",
    "geom_action_signed_local",
    "geometry_raw",
]

DINO_COLUMNS = [
    "dino_global_mean_update",
    "dino_global_topk_update",
    "dino_roi_mean_update",
    "dino_roi_topk_update",
]


def init_local_feature_nans(row):
    for column in GEOMETRY_COLUMNS + DINO_COLUMNS:
        row[column] = np.nan
    row["geometry_detection_mode"] = ""
    row["dino_updated"] = 0


# ============================================================
# Existing extractor flattening
# ============================================================


def flatten_signal(signal):
    out = {"face_detected": int(bool(signal.get("face_detected", False)))}
    scalars = [
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
    for key in scalars:
        out[key] = sf(signal.get(key, np.nan))

    out["motion_eyes"] = med([out.get("motion_left_eye"), out.get("motion_right_eye")])
    out["motion_brow"] = med([out.get("motion_left_brow"), out.get("motion_right_brow")])

    gaze = signal.get("gaze")
    out["gaze_horizontal"] = sf(gaze.get("horizontal")) if gaze else np.nan
    out["gaze_vertical"] = sf(gaze.get("vertical")) if gaze else np.nan

    for name, value in signal.get("blendshapes", {}).items():
        out[f"bs_{name}"] = sf(value)
    return out


# ============================================================
# Extraction state
# ============================================================


@dataclass
class ExtractionInfo:
    fps: float
    video_frames: int
    label_rows: int
    extracted_frames: int
    dino_every: int
    dino_update_hz: float
    aligned_crop_size: int
    dino_topk_fraction: float
    trial_count: int
    trial_geometry_baselines: int
    trial_dino_baselines: int


class TrialBuffer:
    def __init__(self, action, repeat_idx):
        self.action = action
        self.repeat_idx = repeat_idx
        self.geometry_pre: list[tuple[int, np.ndarray]] = []
        self.dino_pre: list[tuple[int, torch.Tensor, np.ndarray]] = []
        self.geometry_template: np.ndarray | None = None
        self.dino_baseline: torch.Tensor | None = None
        self.finalized = False


def finalize_trial_buffer(buffer, rows, topk_fraction, roi_pad, min_geom_frames, min_dino_updates):
    if buffer.finalized:
        return

    action = buffer.action
    region = ACTION_CONFIG[action]["region"]

    if len(buffer.geometry_pre) >= min_geom_frames:
        stack = np.stack([canonical for _, canonical in buffer.geometry_pre])
        buffer.geometry_template = np.median(stack, axis=0).astype(np.float32)
        for row_idx, canonical in buffer.geometry_pre:
            fill_local_geometry(rows[row_idx], canonical, buffer.geometry_template, action)
    else:
        print(
            f"[WARN] {action} R{buffer.repeat_idx}: "
            f"only {len(buffer.geometry_pre)} PRE geometry frames"
        )

    if len(buffer.dino_pre) >= min_dino_updates:
        buffer.dino_baseline = normalized_feature_mean([feat for _, feat, _ in buffer.dino_pre])
        for row_idx, feat, aligned_points in buffer.dino_pre:
            change = dino_change_map(feat, buffer.dino_baseline)
            mask = dino_roi_mask(
                change.shape[0],
                change.shape[1],
                aligned_points,
                REGIONS[region],
                roi_pad,
            )
            rows[row_idx].update(dino_scores(change, mask, topk_fraction))
            rows[row_idx]["dino_updated"] = 1
    else:
        print(
            f"[WARN] {action} R{buffer.repeat_idx}: "
            f"only {len(buffer.dino_pre)} PRE DINO updates"
        )

    buffer.finalized = True


# ============================================================
# Action-video extraction: trial-local geometry + DINO
# ============================================================


def extract_action_video(
    video_path,
    labels_path,
    *,
    dino_every,
    crop_size,
    topk_fraction,
    roi_pad,
    min_geom_frames,
    min_dino_updates,
):
    labels = load_labels(labels_path)
    lookup = {int(row.frame_idx): row for _, row in labels.iterrows()}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video  : {video_path}")
    print(f"Labels : {labels_path}")
    print(f"FPS    : {fps:.3f}")
    print(f"Frames : {total}")
    if abs(total - len(labels)) > 2:
        print(f"[WARN] video/label frame mismatch: {total} vs {len(labels)}")

    # Suppress the extractor's own DINO cadence. Passing frame_idx+1 prevents
    # its internal modulo from firing at frame 0; v3 runs aligned DINO itself.
    extractor = MicroExpressionSignalExtractor(dino_every=10**9)

    rows: list[dict] = []
    buffers: dict[tuple[str, int], TrialBuffer] = {}
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp_ms = int(round(frame_idx / fps * 1000.0))
            signal = extractor.extract(frame, frame_idx + 1, timestamp_ms)
            label_row = lookup.get(frame_idx)

            row = {
                "frame_idx": frame_idx,
                "analysis_timestamp_ms": timestamp_ms,
            }
            if label_row is not None:
                for column in labels.columns:
                    if column != "frame_idx":
                        row[column] = label_row[column]
            row.update(flatten_signal(signal))
            init_local_feature_nans(row)

            key = trial_key(label_row)
            phase = movement_phase(label_row)
            detected = bool(signal.get("face_detected", False))
            canonical = signal.get("canonical_landmarks")
            points = signal.get("landmarks")

            if key is not None:
                action, repeat_idx = key
                buffer = buffers.setdefault(key, TrialBuffer(action, repeat_idx))

                # When PRE ends, freeze that repeat's own neutral references.
                if phase != "pre_neutral" and not buffer.finalized:
                    finalize_trial_buffer(
                        buffer,
                        rows,
                        topk_fraction,
                        roi_pad,
                        min_geom_frames,
                        min_dino_updates,
                    )

                if detected and canonical is not None:
                    canonical_np = np.asarray(canonical, np.float32).copy()
                    if phase == "pre_neutral" and not buffer.finalized:
                        buffer.geometry_pre.append((len(rows), canonical_np))
                    elif buffer.geometry_template is not None:
                        fill_local_geometry(row, canonical_np, buffer.geometry_template, action)

                # DINO only where it matters: PRE/active/POST trial frames.
                if detected and points is not None and frame_idx % dino_every == 0:
                    crop, aligned_points = aligned_face(
                        frame,
                        np.asarray(points, np.float32),
                        crop_size,
                    )
                    if crop is not None and aligned_points is not None:
                        feat = extractor.dino.extract(crop).clone()
                        if phase == "pre_neutral" and not buffer.finalized:
                            buffer.dino_pre.append((len(rows), feat, aligned_points.copy()))
                        elif buffer.dino_baseline is not None:
                            change = dino_change_map(feat, buffer.dino_baseline)
                            mask = dino_roi_mask(
                                change.shape[0],
                                change.shape[1],
                                aligned_points,
                                REGIONS[ACTION_CONFIG[action]["region"]],
                                roi_pad,
                            )
                            row.update(dino_scores(change, mask, topk_fraction))
                            row["dino_updated"] = 1

            rows.append(row)
            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  {frame_idx}/{total}")

    finally:
        cap.release()
        extractor.close()

    # Defensive finalization if a recording ended during PRE.
    for buffer in buffers.values():
        if not buffer.finalized:
            finalize_trial_buffer(
                buffer,
                rows,
                topk_fraction,
                roi_pad,
                min_geom_frames,
                min_dino_updates,
            )

    df = pd.DataFrame(rows)
    geom_ready = sum(b.geometry_template is not None for b in buffers.values())
    dino_ready = sum(b.dino_baseline is not None for b in buffers.values())

    print(f"Extracted {len(df)} frames")
    print(f"Trial-local geometry baselines: {geom_ready}/{len(buffers)}")
    print(f"Trial-local DINO baselines    : {dino_ready}/{len(buffers)}")

    info = ExtractionInfo(
        fps=fps,
        video_frames=total,
        label_rows=len(labels),
        extracted_frames=len(df),
        dino_every=dino_every,
        dino_update_hz=fps / dino_every,
        aligned_crop_size=crop_size,
        dino_topk_fraction=topk_fraction,
        trial_count=len(buffers),
        trial_geometry_baselines=geom_ready,
        trial_dino_baselines=dino_ready,
    )
    return df, info


# ============================================================
# Control extraction: long initial-neutral reference
# ============================================================


def _control_action_columns(row):
    """Initialize action-specific control features."""
    for action in ACTION_CONFIG:
        row[f"control_geom_{action}"] = np.nan
        row[f"control_dino_{action}_update"] = np.nan
    row["control_local_baseline"] = 0
    row["control_reference_target"] = ""
    row["control_reference_label"] = ""
    row["dino_updated"] = 0


def _select_tail(entries, tail_s):
    """Keep only the stable tail of one center/neutral phase."""
    if not entries:
        return []
    max_elapsed_ms = max(float(item[1]) for item in entries)
    cutoff_ms = max(0.0, max_elapsed_ms - float(tail_s) * 1000.0)
    return [item for item in entries if float(item[1]) >= cutoff_ms]


def _control_geometry_value(action, canonical, template):
    cfg = ACTION_CONFIG[action]
    region = cfg["region"]
    region_value = region_state(canonical, template, REGIONS[region])
    diagnostics = geometry_diagnostics(canonical, template)
    signed = action_geometry_score(action, diagnostics, region_value)
    return (
        sf(region_value)
        if cfg["geometry_source"] == "region_state"
        else sf(signed)
    )


def _fill_control_geometry(row, canonical, template):
    for action in ACTION_CONFIG:
        row[f"control_geom_{action}"] = _control_geometry_value(
            action,
            canonical,
            template,
        )


def _fill_control_dino(
    row,
    features,
    aligned_points,
    baseline,
    *,
    topk_fraction,
    roi_pad,
):
    change = dino_change_map(features, baseline)
    if change is None:
        return
    for action, cfg in ACTION_CONFIG.items():
        mask = dino_roi_mask(
            change.shape[0],
            change.shape[1],
            aligned_points,
            REGIONS[cfg["region"]],
            roi_pad,
        )
        score = dino_scores(change, mask, topk_fraction)["dino_roi_topk_update"]
        row[f"control_dino_{action}_update"] = sf(score)
    row["dino_updated"] = 1


class ControlReferenceBuffer:
    def __init__(self, label, target_label):
        self.label = label
        self.target_label = target_label
        # (row_idx, phase_elapsed_ms, canonical)
        self.geometry_frames = []
        # (row_idx, phase_elapsed_ms, features, aligned_points)
        self.dino_updates = []
        self.geometry_template = None
        self.dino_baseline = None
        self.finalized = False
        self.geometry_baseline_rows = []
        self.dino_baseline_rows = []


def finalize_control_reference(
    buffer,
    rows,
    *,
    baseline_tail_s,
    topk_fraction,
    roi_pad,
    min_geom_frames,
    min_dino_updates,
):
    if buffer.finalized:
        return

    geom_tail = _select_tail(buffer.geometry_frames, baseline_tail_s)
    if len(geom_tail) < min_geom_frames:
        # Fall back to the full center phase before giving up.  This is safer
        # than silently constructing a template from only a handful of frames.
        geom_tail = list(buffer.geometry_frames)

    if len(geom_tail) >= min_geom_frames:
        stack = np.stack([canonical for _, _, canonical in geom_tail])
        buffer.geometry_template = np.median(stack, axis=0).astype(np.float32)
        buffer.geometry_baseline_rows = [row_idx for row_idx, _, _ in geom_tail]
        for row_idx, _, canonical in geom_tail:
            rows[row_idx]["control_local_baseline"] = 1
            rows[row_idx]["control_reference_target"] = buffer.target_label
            rows[row_idx]["control_reference_label"] = buffer.label
            _fill_control_geometry(
                rows[row_idx],
                canonical,
                buffer.geometry_template,
            )
    else:
        print(
            f"[WARN] control {buffer.label} -> {buffer.target_label}: "
            f"only {len(buffer.geometry_frames)} geometry frames"
        )

    dino_tail = _select_tail(buffer.dino_updates, baseline_tail_s)
    if len(dino_tail) < min_dino_updates:
        dino_tail = list(buffer.dino_updates)

    if len(dino_tail) >= min_dino_updates:
        buffer.dino_baseline = normalized_feature_mean(
            [features for _, _, features, _ in dino_tail]
        )
        buffer.dino_baseline_rows = [row_idx for row_idx, _, _, _ in dino_tail]
        for row_idx, _, features, aligned_points in dino_tail:
            rows[row_idx]["control_local_baseline"] = 1
            rows[row_idx]["control_reference_target"] = buffer.target_label
            rows[row_idx]["control_reference_label"] = buffer.label
            _fill_control_dino(
                rows[row_idx],
                features,
                aligned_points,
                buffer.dino_baseline,
                topk_fraction=topk_fraction,
                roi_pad=roi_pad,
            )
    else:
        print(
            f"[WARN] control {buffer.label} -> {buffer.target_label}: "
            f"only {len(buffer.dino_updates)} DINO updates"
        )

    buffer.finalized = True


def extract_control_video(
    video_path,
    labels_path,
    *,
    dino_every,
    crop_size,
    topk_fraction,
    roi_pad,
    baseline_tail_s,
    min_geom_frames,
    min_dino_updates,
):
    """Extract control signals using preceding-center local references.

    v3 used one long initial neutral reference for the entire control video.
    v3.1 instead freezes the stable tail of the center phase immediately before
    each BLINK / GAZE / HEAD nuisance phase.  Geometry and DINO are also
    computed separately for every facial-action definition.
    """
    labels = load_labels(labels_path)
    lookup = {int(row.frame_idx): row for _, row in labels.iterrows()}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video  : {video_path}")
    print(f"Labels : {labels_path}")
    print(f"FPS    : {fps:.3f}")
    print(f"Frames : {total}")
    print(f"Control local baseline tail: {baseline_tail_s:.2f} s")

    extractor = MicroExpressionSignalExtractor(dino_every=10**9)
    rows = []
    buffers = {
        baseline: ControlReferenceBuffer(baseline, target)
        for baseline, target in CONTROL_BASELINE_TO_TARGET.items()
    }
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp_ms = int(round(frame_idx / fps * 1000.0))
            signal = extractor.extract(frame, frame_idx + 1, timestamp_ms)
            label_row = lookup.get(frame_idx)

            row = {
                "frame_idx": frame_idx,
                "analysis_timestamp_ms": timestamp_ms,
            }
            if label_row is not None:
                for column in labels.columns:
                    if column != "frame_idx":
                        row[column] = label_row[column]
            row.update(flatten_signal(signal))
            _control_action_columns(row)

            label = str(label_value(label_row, "label", "")).upper()
            elapsed_ms = sf(label_value(label_row, "phase_elapsed_ms", 0.0), 0.0)
            detected = bool(signal.get("face_detected", False))
            canonical = signal.get("canonical_landmarks")
            points = signal.get("landmarks")

            # ------------------------------------------------------------
            # Center/neutral phase: collect reference candidates.
            # ------------------------------------------------------------
            if label in buffers:
                buffer = buffers[label]
                if detected and canonical is not None:
                    buffer.geometry_frames.append(
                        (
                            len(rows),
                            elapsed_ms,
                            np.asarray(canonical, np.float32).copy(),
                        )
                    )

                if detected and points is not None and frame_idx % dino_every == 0:
                    crop, aligned_points = aligned_face(
                        frame,
                        np.asarray(points, np.float32),
                        crop_size,
                    )
                    if crop is not None and aligned_points is not None:
                        feat = extractor.dino.extract(crop).clone()
                        buffer.dino_updates.append(
                            (
                                len(rows),
                                elapsed_ms,
                                feat,
                                aligned_points.copy(),
                            )
                        )

            # ------------------------------------------------------------
            # Nuisance/recovery target: freeze its immediately preceding
            # center reference, then compute action-specific signals.
            # ------------------------------------------------------------
            if label in CONTROL_TARGET_BASELINES:
                baseline_label = CONTROL_TARGET_BASELINES[label]
                buffer = buffers[baseline_label]

                if not buffer.finalized:
                    finalize_control_reference(
                        buffer,
                        rows,
                        baseline_tail_s=baseline_tail_s,
                        topk_fraction=topk_fraction,
                        roi_pad=roi_pad,
                        min_geom_frames=min_geom_frames,
                        min_dino_updates=min_dino_updates,
                    )

                row["control_reference_target"] = label
                row["control_reference_label"] = baseline_label

                if detected and canonical is not None and buffer.geometry_template is not None:
                    _fill_control_geometry(
                        row,
                        np.asarray(canonical, np.float32),
                        buffer.geometry_template,
                    )

                if (
                    detected
                    and points is not None
                    and buffer.dino_baseline is not None
                    and frame_idx % dino_every == 0
                ):
                    crop, aligned_points = aligned_face(
                        frame,
                        np.asarray(points, np.float32),
                        crop_size,
                    )
                    if crop is not None and aligned_points is not None:
                        feat = extractor.dino.extract(crop).clone()
                        _fill_control_dino(
                            row,
                            feat,
                            aligned_points,
                            buffer.dino_baseline,
                            topk_fraction=topk_fraction,
                            roi_pad=roi_pad,
                        )

            rows.append(row)
            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"  {frame_idx}/{total}")

    finally:
        cap.release()
        extractor.close()

    # Finalize any reference phase that was recorded but never followed by its
    # expected target (defensive behavior for interrupted recordings).
    for buffer in buffers.values():
        if buffer.geometry_frames and not buffer.finalized:
            finalize_control_reference(
                buffer,
                rows,
                baseline_tail_s=baseline_tail_s,
                topk_fraction=topk_fraction,
                roi_pad=roi_pad,
                min_geom_frames=min_geom_frames,
                min_dino_updates=min_dino_updates,
            )

    geom_ready = sum(b.geometry_template is not None for b in buffers.values())
    dino_ready = sum(b.dino_baseline is not None for b in buffers.values())
    expected = len(buffers)

    print(f"Control geometry references: {geom_ready}/{expected}")
    print(f"Control DINO references    : {dino_ready}/{expected}")

    df = pd.DataFrame(rows)
    info = ExtractionInfo(
        fps=fps,
        video_frames=total,
        label_rows=len(labels),
        extracted_frames=len(df),
        dino_every=dino_every,
        dino_update_hz=fps / dino_every,
        aligned_crop_size=crop_size,
        dino_topk_fraction=topk_fraction,
        trial_count=len(CONTROL_TARGET_BASELINES),
        trial_geometry_baselines=geom_ready,
        trial_dino_baselines=dino_ready,
    )
    print(f"Extracted {len(df)} frames")
    return df, info


# ============================================================
# Semantic action scores
# ============================================================


def add_semantic_scores(df):
    for action, cfg in ACTION_CONFIG.items():
        columns = [f"bs_{name}" for name in cfg["blendshapes"] if f"bs_{name}" in df.columns]
        target = f"score_{action}"
        if not columns:
            df[target] = np.nan
            continue
        values = df[columns].apply(pd.to_numeric, errors="coerce")
        if cfg["semantic_reducer"] == "max":
            df[target] = values.max(axis=1, skipna=True)
        else:
            df[target] = values.mean(axis=1, skipna=True)
    return df


# ============================================================
# Trial-generic columns + groupwise smoothing
# ============================================================


def init_analysis_columns(df):
    float_columns = [
        "semantic_raw",
        "semantic_smooth",
        "semantic_delta",
        "semantic_signed_evidence",
        "semantic_anomaly_evidence",
        "semantic_threshold_evidence",
        "geometry_smooth",
        "geometry_delta",
        "geometry_signed_evidence",
        "geometry_anomaly_evidence",
        "geometry_threshold_evidence",
        "motion_reference_raw",
        "motion_reference_smooth",
        "motion_reference_delta",
        "motion_reference_evidence",
        "motion_reference_threshold_evidence",
        "dino_smooth_update",
        "dino_smooth",
        "dino_delta",
        "dino_signed_evidence",
        "dino_anomaly_evidence",
        "dino_threshold_evidence",
        "head_delta_deg",
        "face_scale_delta_pct",
        "gaze_delta",
    ]
    for column in float_columns:
        df[column] = np.nan
    for column in ["semantic_detect", "geometry_detect", "motion_reference_detect", "dino_detect"]:
        df[column] = 0
    return df


def trial_masks(df, action, repeat_idx):
    action_mask = df["action"].fillna("").astype(str).eq(action)
    repeat_mask = pd.to_numeric(df["repeat_idx"], errors="coerce").eq(repeat_idx)
    phase = df["movement_phase"].fillna("").astype(str)
    trial = action_mask & repeat_mask
    return {
        "trial": trial,
        "pre": trial & phase.eq("pre_neutral"),
        "onset": trial & phase.eq("onset"),
        "hold": trial & phase.eq("hold"),
        "release": trial & phase.eq("release"),
        "post": trial & phase.eq("post_neutral"),
        "active": trial & phase.isin(["onset", "hold", "release"]),
    }


def add_trial_smoothing(df, ema_ms, dino_ema_ms):
    df = init_analysis_columns(df)
    actions = [a for a in df["action"].dropna().astype(str).unique() if a in ACTION_CONFIG]

    for action in actions:
        cfg = ACTION_CONFIG[action]
        repeats = pd.to_numeric(
            df.loc[df["action"].fillna("").astype(str).eq(action), "repeat_idx"],
            errors="coerce",
        ).dropna().astype(int).unique()

        for repeat_idx in repeats:
            masks = trial_masks(df, action, int(repeat_idx))
            idx = df.index[masks["trial"]]
            if not len(idx):
                continue

            times = pd.to_numeric(df.loc[idx, "analysis_timestamp_ms"], errors="coerce").to_numpy(float)
            semantic_column = f"score_{action}"
            semantic = pd.to_numeric(df.loc[idx, semantic_column], errors="coerce").to_numpy(float)
            geometry = pd.to_numeric(df.loc[idx, "geometry_raw"], errors="coerce").to_numpy(float)
            motion_column = REGION_MOTION[cfg["region"]]
            motion = pd.to_numeric(df.loc[idx, motion_column], errors="coerce").to_numpy(float)

            df.loc[idx, "semantic_raw"] = semantic
            df.loc[idx, "semantic_smooth"] = causal_ema(semantic, times, ema_ms)
            df.loc[idx, "geometry_smooth"] = causal_ema(geometry, times, ema_ms)
            df.loc[idx, "motion_reference_raw"] = motion
            df.loc[idx, "motion_reference_smooth"] = causal_ema(motion, times, ema_ms)

            dino_values = pd.to_numeric(df.loc[idx, "dino_roi_topk_update"], errors="coerce").to_numpy(float)
            dino_updates = pd.to_numeric(df.loc[idx, "dino_updated"], errors="coerce").fillna(0).astype(int).eq(1).to_numpy()
            sparse, held = sparse_ema_hold(dino_values, dino_updates, times, dino_ema_ms)
            df.loc[idx, "dino_smooth_update"] = sparse
            df.loc[idx, "dino_smooth"] = held

    return df


# ============================================================
# Calibration: two-sided vs one-sided evidence
# ============================================================


@dataclass
class Calibration:
    center: float
    scale: float
    threshold_evidence: float
    local_scale: float
    global_scale: float
    floor: float
    q99_evidence: float
    mode: str


def calibrate(local, global_values, floor, k, mode):
    local = finite(local)
    global_values = finite(global_values)
    if len(local) < 5:
        return None

    center = float(np.median(local))
    local_scale = robust_scale(local)
    global_scale = robust_scale(global_values)
    candidates = [float(floor)]
    for value in [local_scale, global_scale]:
        if np.isfinite(value):
            candidates.append(float(value))
    scale = max(candidates)

    signed = (local - center) / scale
    if mode == "two-sided":
        anomaly = np.abs(signed)
    elif mode == "one-sided-high":
        anomaly = signed
    else:
        raise ValueError(f"unknown calibration mode: {mode}")

    q99 = float(np.quantile(anomaly, 0.99))
    threshold_evidence = max(float(k), q99)
    return Calibration(
        center=center,
        scale=scale,
        threshold_evidence=threshold_evidence,
        local_scale=local_scale,
        global_scale=global_scale,
        floor=float(floor),
        q99_evidence=q99,
        mode=mode,
    )


def evidence_arrays(values, calibration):
    values = np.asarray(values, float)
    if calibration is None:
        nan = np.full(len(values), np.nan)
        return nan, nan
    signed = (values - calibration.center) / calibration.scale
    anomaly = np.abs(signed) if calibration.mode == "two-sided" else signed
    return signed, anomaly


def all_pre_mask(df, action=None):
    phase = df["movement_phase"].fillna("").astype(str).eq("pre_neutral")
    if action is None:
        return phase
    return phase & df["action"].fillna("").astype(str).eq(action)


def calibrate_trials(
    df,
    *,
    semantic_floor,
    geometry_floor,
    dino_floor,
    motion_floor,
    threshold_k,
):
    actions = [a for a in df["action"].dropna().astype(str).unique() if a in ACTION_CONFIG]

    for action in actions:
        cfg = ACTION_CONFIG[action]
        same_action_pre = all_pre_mask(df, action)

        semantic_global = pd.to_numeric(df.loc[same_action_pre, "semantic_smooth"], errors="coerce")
        geometry_global = pd.to_numeric(df.loc[same_action_pre, "geometry_smooth"], errors="coerce")
        motion_global = pd.to_numeric(df.loc[same_action_pre, "motion_reference_smooth"], errors="coerce")
        dino_global_mask = same_action_pre & pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
        dino_global = pd.to_numeric(df.loc[dino_global_mask, "dino_smooth_update"], errors="coerce")

        geometry_mode = (
            "one-sided-high" if cfg["geometry_source"] == "region_state" else "two-sided"
        )

        repeats = pd.to_numeric(
            df.loc[df["action"].fillna("").astype(str).eq(action), "repeat_idx"],
            errors="coerce",
        ).dropna().astype(int).unique()

        for repeat_idx in repeats:
            masks = trial_masks(df, action, int(repeat_idx))
            idx = df.index[masks["trial"]]
            pre_idx = df.index[masks["pre"]]
            if len(idx) == 0 or len(pre_idx) < 5:
                continue

            sem_cal = calibrate(
                pd.to_numeric(df.loc[pre_idx, "semantic_smooth"], errors="coerce"),
                semantic_global,
                semantic_floor,
                threshold_k,
                "two-sided",
            )
            geom_cal = calibrate(
                pd.to_numeric(df.loc[pre_idx, "geometry_smooth"], errors="coerce"),
                geometry_global,
                geometry_floor,
                threshold_k,
                geometry_mode,
            )
            mot_cal = calibrate(
                pd.to_numeric(df.loc[pre_idx, "motion_reference_smooth"], errors="coerce"),
                motion_global,
                motion_floor,
                threshold_k,
                "one-sided-high",
            )

            pre_dino = masks["pre"] & pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
            dino_cal = calibrate(
                pd.to_numeric(df.loc[pre_dino, "dino_smooth_update"], errors="coerce"),
                dino_global,
                dino_floor,
                threshold_k,
                "one-sided-high",
            )

            specs = [
                ("semantic", "semantic_smooth", sem_cal),
                ("geometry", "geometry_smooth", geom_cal),
                ("motion_reference", "motion_reference_smooth", mot_cal),
                ("dino", "dino_smooth", dino_cal),
            ]

            face_ok = pd.to_numeric(df.loc[idx, "face_detected"], errors="coerce").fillna(0).astype(int).eq(1).to_numpy()

            for prefix, source, cal in specs:
                values = pd.to_numeric(df.loc[idx, source], errors="coerce").to_numpy(float)
                signed, anomaly = evidence_arrays(values, cal)
                if prefix == "motion_reference":
                    df.loc[idx, "motion_reference_delta"] = values - (cal.center if cal else np.nan)
                    df.loc[idx, "motion_reference_evidence"] = anomaly
                    if cal:
                        df.loc[idx, "motion_reference_threshold_evidence"] = cal.threshold_evidence
                        df.loc[idx, "motion_reference_detect"] = ((anomaly >= cal.threshold_evidence) & face_ok).astype(int)
                    continue

                df.loc[idx, f"{prefix}_delta"] = values - (cal.center if cal else np.nan)
                df.loc[idx, f"{prefix}_signed_evidence"] = signed
                df.loc[idx, f"{prefix}_anomaly_evidence"] = anomaly
                if cal:
                    df.loc[idx, f"{prefix}_threshold_evidence"] = cal.threshold_evidence
                    df.loc[idx, f"{prefix}_detect"] = ((anomaly >= cal.threshold_evidence) & face_ok).astype(int)

            # Trial-local nuisance baselines.
            yaw0 = med(pd.to_numeric(df.loc[pre_idx, "yaw_deg"], errors="coerce"))
            pitch0 = med(pd.to_numeric(df.loc[pre_idx, "pitch_deg"], errors="coerce"))
            roll0 = med(pd.to_numeric(df.loc[pre_idx, "roll_deg"], errors="coerce"))
            yaw = pd.to_numeric(df.loc[idx, "yaw_deg"], errors="coerce").to_numpy(float)
            pitch = pd.to_numeric(df.loc[idx, "pitch_deg"], errors="coerce").to_numpy(float)
            roll = pd.to_numeric(df.loc[idx, "roll_deg"], errors="coerce").to_numpy(float)
            df.loc[idx, "head_delta_deg"] = np.nanmax(
                np.stack([np.abs(yaw - yaw0), np.abs(pitch - pitch0), np.abs(roll - roll0)], axis=1),
                axis=1,
            )

            face0 = med(pd.to_numeric(df.loc[pre_idx, "face_ratio"], errors="coerce"))
            if np.isfinite(face0) and abs(face0) > 1e-9:
                face = pd.to_numeric(df.loc[idx, "face_ratio"], errors="coerce").to_numpy(float)
                df.loc[idx, "face_scale_delta_pct"] = np.abs(face / face0 - 1.0) * 100.0

            gh0 = med(pd.to_numeric(df.loc[pre_idx, "gaze_horizontal"], errors="coerce"))
            gv0 = med(pd.to_numeric(df.loc[pre_idx, "gaze_vertical"], errors="coerce"))
            gh = pd.to_numeric(df.loc[idx, "gaze_horizontal"], errors="coerce").to_numpy(float)
            gv = pd.to_numeric(df.loc[idx, "gaze_vertical"], errors="coerce").to_numpy(float)
            df.loc[idx, "gaze_delta"] = np.sqrt((gh - gh0) ** 2 + (gv - gv0) ** 2)

    return df


# ============================================================
# Metrics
# ============================================================


def hold_auc(df, pre_mask, hold_mask, column, dino=False):
    if dino:
        updates = pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
        pre_mask = pre_mask & updates
        hold_mask = hold_mask & updates
    return auc_rank(
        pd.to_numeric(df.loc[pre_mask, column], errors="coerce"),
        pd.to_numeric(df.loc[hold_mask, column], errors="coerce"),
    )


def hold_delta(df, pre_mask, hold_mask, column, dino=False):
    if dino:
        updates = pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
        pre_mask = pre_mask & updates
        hold_mask = hold_mask & updates
    pre = med(pd.to_numeric(df.loc[pre_mask, column], errors="coerce"))
    hold = med(pd.to_numeric(df.loc[hold_mask, column], errors="coerce"))
    return float(hold - pre) if np.isfinite(pre) and np.isfinite(hold) else np.nan


def build_metrics(
    df,
    *,
    sustain_ms,
    motion_sustain_ms,
    dino_min_updates,
    head_qc_deg,
    face_scale_qc_pct,
    gaze_qc,
):
    rows = []
    actions = [a for a in df["action"].dropna().astype(str).unique() if a in ACTION_CONFIG]

    for action in actions:
        repeats = pd.to_numeric(
            df.loc[df["action"].fillna("").astype(str).eq(action), "repeat_idx"],
            errors="coerce",
        ).dropna().astype(int).unique()

        for repeat_idx in repeats:
            masks = trial_masks(df, action, int(repeat_idx))
            active = df.loc[masks["active"]]
            onset = df.loc[masks["onset"]]
            hold = df.loc[masks["hold"]]
            if active.empty or onset.empty:
                continue

            item = {
                "action": action,
                "repeat_idx": int(repeat_idx),
                "active_frames": len(active),
                "face_detection_rate": float(
                    pd.to_numeric(active["face_detected"], errors="coerce").fillna(0).mean()
                ),
                "head_delta_peak_deg": sf(pd.to_numeric(active["head_delta_deg"], errors="coerce").max()),
                "face_scale_delta_peak_pct": sf(
                    pd.to_numeric(active["face_scale_delta_pct"], errors="coerce").max()
                ),
                "gaze_delta_peak": sf(pd.to_numeric(active["gaze_delta"], errors="coerce").max()),
                "blink_peak": sf(pd.to_numeric(active["blink"], errors="coerce").max()),
                "dino_active_updates": int(
                    pd.to_numeric(active["dino_updated"], errors="coerce").fillna(0).astype(int).sum()
                ),
            }

            item["qc_pass"] = int(
                item["face_detection_rate"] >= 0.98
                and (not np.isfinite(item["head_delta_peak_deg"]) or item["head_delta_peak_deg"] <= head_qc_deg)
                and (
                    not np.isfinite(item["face_scale_delta_peak_pct"])
                    or item["face_scale_delta_peak_pct"] <= face_scale_qc_pct
                )
                and (not np.isfinite(item["gaze_delta_peak"]) or item["gaze_delta_peak"] <= gaze_qc)
            )

            for name, column, dino in [
                ("semantic", "semantic_smooth", False),
                ("geometry", "geometry_smooth", False),
                ("dino", "dino_smooth", True),
            ]:
                auc = hold_auc(df, masks["pre"], masks["hold"], column, dino)
                item[f"{name}_hold_auc"] = auc
                item[f"{name}_hold_auc_separability"] = max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan
                item[f"{name}_hold_polarity"] = (
                    "positive" if np.isfinite(auc) and auc >= 0.5 else ("negative" if np.isfinite(auc) else "")
                )
                item[f"{name}_hold_delta_raw"] = hold_delta(
                    df, masks["pre"], masks["hold"], column, dino
                )

                evidence_column = f"{name}_anomaly_evidence"
                active_evidence = pd.to_numeric(active[evidence_column], errors="coerce")
                item[f"{name}_peak_anomaly_evidence"] = sf(active_evidence.max())

                hold_eval = hold
                if dino:
                    hold_eval = hold[
                        pd.to_numeric(hold["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
                    ]
                item[f"{name}_hold_detection_rate"] = (
                    float(pd.to_numeric(hold_eval[f"{name}_detect"], errors="coerce").fillna(0).mean())
                    if len(hold_eval)
                    else np.nan
                )

                if "intended_progress" in df.columns:
                    progress = pd.to_numeric(df.loc[masks["active"], "intended_progress"], errors="coerce")
                    signed_signal = pd.to_numeric(df.loc[masks["active"], column], errors="coerce")
                    anomaly_signal = pd.to_numeric(
                        df.loc[masks["active"], f"{name}_anomaly_evidence"], errors="coerce"
                    )
                    item[f"{name}_progress_corr_signed"] = rank_corr(progress, signed_signal)
                    item[f"{name}_progress_corr_anomaly"] = rank_corr(progress, anomaly_signal)

            onset_time = pd.to_numeric(onset["phase_elapsed_ms"], errors="coerce").to_numpy(float)
            motion_detect = pd.to_numeric(onset["motion_reference_detect"], errors="coerce").fillna(0).astype(int).to_numpy(bool)
            motion_ms = first_sustained(onset_time, motion_detect, motion_sustain_ms)
            item["motion_reference_cue_ms"] = motion_ms

            for name in ["semantic", "geometry"]:
                detect = pd.to_numeric(onset[f"{name}_detect"], errors="coerce").fillna(0).astype(int).to_numpy(bool)
                cue = first_sustained(onset_time, detect, sustain_ms)
                item[f"{name}_cue_to_detection_ms"] = cue
                item[f"{name}_lag_vs_motion_ms"] = (
                    cue - motion_ms if np.isfinite(cue) and np.isfinite(motion_ms) else np.nan
                )

            dino_onset = onset[
                pd.to_numeric(onset["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
            ]
            if len(dino_onset):
                dino_cue = first_n_updates(
                    pd.to_numeric(dino_onset["phase_elapsed_ms"], errors="coerce"),
                    pd.to_numeric(dino_onset["dino_detect"], errors="coerce").fillna(0).astype(int).to_numpy(bool),
                    dino_min_updates,
                )
            else:
                dino_cue = np.nan
            item["dino_cue_to_detection_ms"] = dino_cue
            item["dino_lag_vs_motion_ms"] = (
                dino_cue - motion_ms if np.isfinite(dino_cue) and np.isfinite(motion_ms) else np.nan
            )

            rows.append(item)

    return pd.DataFrame(rows)


# ============================================================
# Repeat consistency
# ============================================================


def resample_trial_delta(active, pre, column, n=101):
    if column not in active.columns or column not in pre.columns:
        return None
    baseline = med(pd.to_numeric(pre[column], errors="coerce"))
    if not np.isfinite(baseline):
        return None
    times = pd.to_numeric(active["analysis_timestamp_ms"], errors="coerce").to_numpy(float)
    values = pd.to_numeric(active[column], errors="coerce").to_numpy(float) - baseline
    good = np.isfinite(times) & np.isfinite(values)
    if good.sum() < 4:
        return None
    times = times[good]
    values = values[good]
    if times[-1] <= times[0]:
        return None
    u = (times - times[0]) / (times[-1] - times[0])
    return np.interp(np.linspace(0.0, 1.0, n), u, values)


def repeat_consistency(df, metrics):
    if metrics.empty:
        return pd.DataFrame()
    rows = []

    for action in metrics["action"].unique():
        repeats = metrics.loc[metrics["action"].eq(action), "repeat_idx"].astype(int).tolist()
        for name, column in [
            ("semantic", "semantic_smooth"),
            ("geometry", "geometry_smooth"),
            ("dino", "dino_smooth"),
        ]:
            curves = []
            amplitudes = []
            absolute_amplitudes = []

            for repeat_idx in repeats:
                masks = trial_masks(df, action, repeat_idx)
                active = df.loc[masks["active"]]
                pre = df.loc[masks["pre"]]
                hold = df.loc[masks["hold"]]
                curve = resample_trial_delta(active, pre, column)
                if curve is not None:
                    curves.append(curve)
                baseline = med(pd.to_numeric(pre[column], errors="coerce")) if column in pre else np.nan
                hold_value = med(pd.to_numeric(hold[column], errors="coerce")) if column in hold else np.nan
                if np.isfinite(baseline) and np.isfinite(hold_value):
                    amp = hold_value - baseline
                    amplitudes.append(amp)
                    absolute_amplitudes.append(abs(amp))

            correlations = []
            for i in range(len(curves)):
                for j in range(i + 1, len(curves)):
                    if np.std(curves[i]) > 1e-12 and np.std(curves[j]) > 1e-12:
                        correlations.append(float(np.corrcoef(curves[i], curves[j])[0, 1]))

            mean_abs = float(np.mean(absolute_amplitudes)) if absolute_amplitudes else np.nan
            amp_cv = (
                float(np.std(absolute_amplitudes) / mean_abs)
                if absolute_amplitudes and mean_abs > 1e-12
                else np.nan
            )
            sign_consistency = (
                float(abs(np.mean(np.sign(amplitudes)))) if amplitudes else np.nan
            )

            rows.append({
                "action": action,
                "signal": name,
                "repeat_count": len(curves),
                "pair_count": len(correlations),
                "mean_pairwise_corr": float(np.mean(correlations)) if correlations else np.nan,
                "min_pairwise_corr": float(np.min(correlations)) if correlations else np.nan,
                "hold_absolute_amplitude_cv": amp_cv,
                "hold_polarity_consistency": sign_consistency,
            })

    return pd.DataFrame(rows)


# ============================================================
# Control analysis
# ============================================================


def _control_pair_masks(df, target_label):
    labels = df["label"].fillna("").astype(str).str.upper()
    baseline_label = CONTROL_TARGET_BASELINES[target_label]
    ref_target = df.get(
        "control_reference_target",
        pd.Series("", index=df.index),
    ).fillna("").astype(str).str.upper()
    local = pd.to_numeric(
        df.get("control_local_baseline", 0),
        errors="coerce",
    ).fillna(0).astype(int).eq(1)

    baseline = labels.eq(baseline_label) & ref_target.eq(target_label) & local
    target = labels.eq(target_label)
    return baseline, target


def _pair_causal_dense(df, baseline_mask, target_mask, column, tau_ms):
    """Smooth one baseline->target pair without leakage from other phases."""
    idx = df.index[baseline_mask | target_mask]
    if not len(idx):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    idx = idx.sort_values()
    values = pd.to_numeric(df.loc[idx, column], errors="coerce").to_numpy(float)
    times = pd.to_numeric(
        df.loc[idx, "analysis_timestamp_ms"],
        errors="coerce",
    ).to_numpy(float)
    smooth = causal_ema(values, times, tau_ms)
    series = pd.Series(smooth, index=idx, dtype=float)
    return series.loc[df.index[baseline_mask]], series.loc[df.index[target_mask]]


def _pair_causal_sparse(df, baseline_mask, target_mask, column, tau_ms):
    """Causal EMA only on actual DINO update frames for one control pair."""
    update = pd.to_numeric(
        df.get("dino_updated", 0),
        errors="coerce",
    ).fillna(0).astype(int).eq(1)
    b = baseline_mask & update
    t = target_mask & update
    idx = df.index[b | t]
    if not len(idx):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    idx = idx.sort_values()
    values = pd.to_numeric(df.loc[idx, column], errors="coerce").to_numpy(float)
    times = pd.to_numeric(
        df.loc[idx, "analysis_timestamp_ms"],
        errors="coerce",
    ).to_numpy(float)
    smooth = causal_ema(values, times, tau_ms)
    series = pd.Series(smooth, index=idx, dtype=float)
    return series.loc[df.index[b]], series.loc[df.index[t]]


def _rate_and_peak(values, calibration):
    if calibration is None:
        return np.nan, np.nan, np.nan
    _, anomaly = evidence_arrays(values, calibration)
    finite_mask = np.isfinite(anomaly)
    if not finite_mask.any():
        return np.nan, np.nan, np.nan
    rate = float(np.mean(anomaly[finite_mask] >= calibration.threshold_evidence))
    peak = float(np.max(anomaly[finite_mask]))
    median_ev = float(np.median(anomaly[finite_mask]))
    return rate, peak, median_ev


def _phase_basic_summary(group):
    item = {
        "label": str(group["label"].iloc[0]),
        "frames": len(group),
        "face_detection_rate": float(
            pd.to_numeric(group["face_detected"], errors="coerce").fillna(0).mean()
        ),
    }
    for name in [
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "gaze_horizontal",
        "gaze_vertical",
    ]:
        values = pd.to_numeric(group[name], errors="coerce")
        item[f"{name}_range"] = sf(values.max() - values.min())
    item["blink_max"] = sf(pd.to_numeric(group["blink"], errors="coerce").max())
    return item


def control_outputs_v31(
    df,
    *,
    semantic_floor,
    geometry_floor,
    dino_floor,
    threshold_k,
    ema_ms,
    dino_ema_ms,
):
    """Action-specific nuisance false-activation analysis.

    Every nuisance phase is calibrated from the stable tail of its own preceding
    center phase.  MediaPipe semantics, action-specific geometry and
    action-region DINO are evaluated separately.
    """
    if "label" not in df.columns:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    false_rows = []
    phase_rows = []

    # Global neutral spread per action is used only as a scale stabilizer.
    # The *center* of every detector still comes from the pair-local baseline.
    local_baseline = pd.to_numeric(
        df.get("control_local_baseline", 0),
        errors="coerce",
    ).fillna(0).astype(int).eq(1)

    semantic_global = {
        action: pd.to_numeric(
            df.loc[local_baseline, f"score_{action}"],
            errors="coerce",
        )
        for action in ACTION_CONFIG
    }
    geometry_global = {
        action: pd.to_numeric(
            df.loc[local_baseline, f"control_geom_{action}"],
            errors="coerce",
        )
        for action in ACTION_CONFIG
    }
    dino_global = {
        action: pd.to_numeric(
            df.loc[
                local_baseline
                & pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1),
                f"control_dino_{action}_update",
            ],
            errors="coerce",
        )
        for action in ACTION_CONFIG
    }

    for target_label, baseline_label in CONTROL_TARGET_BASELINES.items():
        baseline_mask, target_mask = _control_pair_masks(df, target_label)
        target_group = df.loc[target_mask]
        if target_group.empty:
            print(f"[WARN] control target missing: {target_label}")
            continue

        base_item = _phase_basic_summary(target_group)
        base_item["baseline_label"] = baseline_label
        base_item["control_kind"] = CONTROL_TARGET_KIND.get(target_label, "")
        base_item["baseline_frames"] = int(baseline_mask.sum())
        base_item["dino_target_updates"] = int(
            (
                target_mask
                & pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
            ).sum()
        )

        per_action_rows = []

        for action, cfg in ACTION_CONFIG.items():
            semantic_column = f"score_{action}"
            geometry_column = f"control_geom_{action}"
            dino_column = f"control_dino_{action}_update"

            b_sem, t_sem = _pair_causal_dense(
                df,
                baseline_mask,
                target_mask,
                semantic_column,
                ema_ms,
            )
            b_geom, t_geom = _pair_causal_dense(
                df,
                baseline_mask,
                target_mask,
                geometry_column,
                ema_ms,
            )
            b_dino, t_dino = _pair_causal_sparse(
                df,
                baseline_mask,
                target_mask,
                dino_column,
                dino_ema_ms,
            )

            sem_cal = calibrate(
                b_sem,
                semantic_global[action],
                semantic_floor,
                threshold_k,
                "two-sided",
            )

            geometry_mode = (
                "one-sided-high"
                if cfg["geometry_source"] == "region_state"
                else "two-sided"
            )
            geom_cal = calibrate(
                b_geom,
                geometry_global[action],
                geometry_floor,
                threshold_k,
                geometry_mode,
            )
            dino_cal = calibrate(
                b_dino,
                dino_global[action],
                dino_floor,
                threshold_k,
                "one-sided-high",
            )

            sem_rate, sem_peak, sem_median = _rate_and_peak(t_sem.to_numpy(float), sem_cal)
            geom_rate, geom_peak, geom_median = _rate_and_peak(t_geom.to_numpy(float), geom_cal)
            dino_rate, dino_peak, dino_median = _rate_and_peak(t_dino.to_numpy(float), dino_cal)

            row = {
                "target_label": target_label,
                "baseline_label": baseline_label,
                "control_kind": CONTROL_TARGET_KIND.get(target_label, ""),
                "action": action,
                "target_frames": int(target_mask.sum()),
                "baseline_frames": int(baseline_mask.sum()),
                "dino_target_updates": int(len(t_dino)),
                "dino_baseline_updates": int(len(b_dino)),
                "semantic_false_activation_rate": sem_rate,
                "geometry_false_activation_rate": geom_rate,
                "dino_false_activation_rate": dino_rate,
                "semantic_peak_evidence": sem_peak,
                "geometry_peak_evidence": geom_peak,
                "dino_peak_evidence": dino_peak,
                "semantic_median_evidence": sem_median,
                "geometry_median_evidence": geom_median,
                "dino_median_evidence": dino_median,
                "semantic_threshold_evidence": (
                    sem_cal.threshold_evidence if sem_cal else np.nan
                ),
                "geometry_threshold_evidence": (
                    geom_cal.threshold_evidence if geom_cal else np.nan
                ),
                "dino_threshold_evidence": (
                    dino_cal.threshold_evidence if dino_cal else np.nan
                ),
                "semantic_baseline_scale": sem_cal.scale if sem_cal else np.nan,
                "geometry_baseline_scale": geom_cal.scale if geom_cal else np.nan,
                "dino_baseline_scale": dino_cal.scale if dino_cal else np.nan,
                "geometry_detection_mode": geometry_mode,
            }
            false_rows.append(row)
            per_action_rows.append(row)

        if per_action_rows:
            for modality in ["semantic", "geometry", "dino"]:
                key = f"{modality}_false_activation_rate"
                valid = [r for r in per_action_rows if np.isfinite(sf(r.get(key)))]
                if valid:
                    worst = max(valid, key=lambda r: r[key])
                    base_item[f"max_{modality}_false_activation_rate"] = float(worst[key])
                    base_item[f"worst_{modality}_action"] = worst["action"]
                else:
                    base_item[f"max_{modality}_false_activation_rate"] = np.nan
                    base_item[f"worst_{modality}_action"] = ""

        phase_rows.append(base_item)

    false_df = pd.DataFrame(false_rows)
    phase_df = pd.DataFrame(phase_rows)

    if false_df.empty:
        empty = pd.DataFrame()
        return phase_df, false_df, empty, empty, empty

    def matrix(column):
        return (
            false_df.pivot(
                index="target_label",
                columns="action",
                values=column,
            )
            .reindex(list(CONTROL_TARGET_BASELINES.keys()))
            .reset_index()
        )

    semantic_matrix = matrix("semantic_false_activation_rate")
    geometry_matrix = matrix("geometry_false_activation_rate")
    dino_matrix = matrix("dino_false_activation_rate")

    return phase_df, false_df, semantic_matrix, geometry_matrix, dino_matrix


# ============================================================
# Plots
# ============================================================


def phase_boundaries(subset):
    boundaries = []
    previous = None
    for _, row in subset.iterrows():
        label = str(row.get("label", ""))
        if label != previous:
            boundaries.append(sf(row.get("plot_time_s")))
            previous = label
    return boundaries


def draw_boundaries(ax, boundaries):
    for position in boundaries:
        if np.isfinite(position):
            ax.axvline(position, linestyle="--", linewidth=0.7, alpha=0.25)


def plot_action(df, action, outdir, clip):
    subset = df[df["action"].fillna("").astype(str).eq(action)].copy()
    if subset.empty:
        return

    t0 = sf(subset["analysis_timestamp_ms"].iloc[0], 0.0)
    subset["plot_time_s"] = (
        pd.to_numeric(subset["analysis_timestamp_ms"], errors="coerce") - t0
    ) / 1000.0
    x = subset["plot_time_s"]
    boundaries = phase_boundaries(subset)

    # Raw / locally referenced modalities.
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    raw_specs = [
        ("semantic_smooth", "MediaPipe semantic score"),
        ("geometry_smooth", "trial-local action geometry"),
        ("dino_smooth", "trial-local DINO ROI top-k change"),
    ]
    for ax, (column, ylabel) in zip(axes, raw_specs):
        ax.plot(x, pd.to_numeric(subset[column], errors="coerce"))
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        draw_boundaries(ax, boundaries)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{action} - v3 local raw/smoothed modalities")
    fig.tight_layout()
    fig.savefig(outdir / f"{action}_raw_v3.png", dpi=170)
    plt.close(fig)

    # Signed local deltas: useful for classification/polarity.
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    delta_specs = [
        ("semantic_delta", "MediaPipe signed delta"),
        ("geometry_delta", "geometry signed delta"),
        ("dino_delta", "DINO delta from local PRE noise center"),
    ]
    for ax, (column, ylabel) in zip(axes, delta_specs):
        ax.plot(x, pd.to_numeric(subset[column], errors="coerce"))
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        draw_boundaries(ax, boundaries)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{action} - signed PRE-relative deltas")
    fig.tight_layout()
    fig.savefig(outdir / f"{action}_signed_delta_v3.png", dpi=170)
    plt.close(fig)

    # Detection evidence: semantic is two-sided; geometry mode depends on action;
    # DINO is one-sided distance increase.
    fig, ax = plt.subplots(figsize=(14, 6))
    for column, label in [
        ("semantic_anomaly_evidence", "MediaPipe anomaly evidence"),
        ("geometry_anomaly_evidence", "geometry anomaly evidence"),
        ("dino_anomaly_evidence", "DINO ROI evidence"),
    ]:
        y = pd.to_numeric(subset[column], errors="coerce").clip(-clip, clip)
        ax.plot(x, y, label=label)
    # Thresholds can differ by trial because PRE noise differs. Plot actual curves.
    for column, label in [
        ("semantic_threshold_evidence", "semantic threshold"),
        ("geometry_threshold_evidence", "geometry threshold"),
        ("dino_threshold_evidence", "DINO threshold"),
    ]:
        y = pd.to_numeric(subset[column], errors="coerce").clip(-clip, clip)
        ax.plot(x, y, linestyle=":", linewidth=0.9, alpha=0.65, label=label)
    draw_boundaries(ax, boundaries)
    ax.set_title(f"{action} - v3 trial-local detection evidence")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Evidence [local noise units]")
    ax.set_ylim(0.0, clip)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{action}_evidence_v3.png", dpi=170)
    plt.close(fig)


# ============================================================
# Timing metadata
# ============================================================


def timing_info(df):
    out = {
        "analysis_duration_s": (
            sf(df["analysis_timestamp_ms"].iloc[-1]) / 1000.0 if len(df) else np.nan
        )
    }
    for column, name in [
        ("capture_timestamp_ms", "capture_duration_s"),
        ("video_timestamp_ms", "recorded_video_duration_s"),
    ]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if len(values) >= 2:
                out[name] = float((values.iloc[-1] - values.iloc[0]) / 1000.0)
    if out.get("recorded_video_duration_s", 0.0) > 0 and "capture_duration_s" in out:
        out["capture_to_video_time_ratio"] = (
            out["capture_duration_s"] / out["recorded_video_duration_s"]
        )
    return out


# ============================================================
# Protocol runner
# ============================================================


def run_protocol(args, protocol):
    video_name, label_name = PROTOCOL_FILES[protocol]
    input_dir = RECORDING_ROOT / args.participant / args.session
    video_path = input_dir / video_name
    labels_path = input_dir / label_name

    if not video_path.exists() or not labels_path.exists():
        print(f"[SKIP] missing {protocol} video/labels")
        return

    output_dir = OUTPUT_ROOT / args.participant / args.session
    output_dir.mkdir(parents=True, exist_ok=True)
    signals_tag = "v31" if protocol == "control" else "v3"
    signals_path = output_dir / f"{protocol}_signals_{signals_tag}.csv"

    extraction = None
    if args.reuse_signals and signals_path.exists():
        print(f"Reusing {signals_path}")
        df = pd.read_csv(signals_path)
    else:
        if protocol in {"upper", "lower"}:
            df, extraction = extract_action_video(
                video_path,
                labels_path,
                dino_every=args.dino_every,
                crop_size=args.aligned_crop_size,
                topk_fraction=args.dino_topk,
                roi_pad=args.dino_roi_pad,
                min_geom_frames=args.min_pre_geometry_frames,
                min_dino_updates=args.min_pre_dino_updates,
            )
        else:
            df, extraction = extract_control_video(
                video_path,
                labels_path,
                dino_every=args.dino_every,
                crop_size=args.aligned_crop_size,
                topk_fraction=args.dino_topk,
                roi_pad=args.dino_roi_pad,
                baseline_tail_s=args.control_baseline_tail_s,
                min_geom_frames=args.min_pre_geometry_frames,
                min_dino_updates=args.min_pre_dino_updates,
            )

    df = add_semantic_scores(df)

    metadata = {
        "analyzer_version": VERSION,
        "participant": args.participant,
        "session": args.session,
        "protocol": protocol,
        "parameters": vars(args),
        "timing": timing_info(df),
        "design": {
            "semantic_detection": "two-sided PRE-relative anomaly",
            "geometry_baseline": "trial-local PRE median for action videos",
            "dino_baseline": "trial-local PRE mean normalized patch features for action videos",
            "dino_roi": "dynamic aligned-landmark action region",
            "fusion": "none; modalities reported separately",
            "control_v31": (
                "preceding-center local baseline + action-specific geometry "
                "and action-ROI DINO false-activation analysis"
            ),
        },
    }
    if extraction is not None:
        metadata["extraction"] = asdict(extraction)

    if protocol in {"upper", "lower"}:
        df = add_trial_smoothing(df, args.ema_ms, args.dino_ema_ms)
        df = calibrate_trials(
            df,
            semantic_floor=args.semantic_noise_floor,
            geometry_floor=args.geometry_noise_floor,
            dino_floor=args.dino_noise_floor,
            motion_floor=args.motion_noise_floor,
            threshold_k=args.threshold_k,
        )

        metrics = build_metrics(
            df,
            sustain_ms=args.min_sustain_ms,
            motion_sustain_ms=args.motion_sustain_ms,
            dino_min_updates=args.dino_min_updates,
            head_qc_deg=args.head_qc_deg,
            face_scale_qc_pct=args.face_scale_qc_pct,
            gaze_qc=args.gaze_qc,
        )
        consistency = repeat_consistency(df, metrics)

        metrics_path = output_dir / f"{protocol}_trial_metrics_v3.csv"
        consistency_path = output_dir / f"{protocol}_repeat_consistency_v3.csv"
        metrics.to_csv(metrics_path, index=False)
        consistency.to_csv(consistency_path, index=False)

        plot_dir = output_dir / "plots_v3" / protocol
        plot_dir.mkdir(parents=True, exist_ok=True)
        actions = [
            action
            for action in df["action"].dropna().astype(str).unique()
            if action in ACTION_CONFIG
        ]
        for action in actions:
            plot_action(df, action, plot_dir, args.plot_clip)

        metadata["metrics_path"] = str(metrics_path)
        metadata["repeat_consistency_path"] = str(consistency_path)

        show_columns = [
            column
            for column in [
                "action",
                "repeat_idx",

                # Direction-invariant separability.
                "semantic_hold_auc_separability",
                "geometry_hold_auc_separability",
                "dino_hold_auc_separability",

                # Signed response direction.
                "semantic_hold_polarity",
                "geometry_hold_polarity",
                "dino_hold_polarity",

                # Raw PRE -> HOLD effect.
                "semantic_hold_delta_raw",
                "geometry_hold_delta_raw",
                "dino_hold_delta_raw",

                # Cue-relative detection timing (not pure algorithm latency).
                "semantic_cue_to_detection_ms",
                "geometry_cue_to_detection_ms",
                "dino_cue_to_detection_ms",

                # Nuisance QC.
                "head_delta_peak_deg",
                "face_scale_delta_peak_pct",
                "gaze_delta_peak",
                "blink_peak",
                "qc_pass",
            ]
            if column in metrics.columns
        ]

        print(f"Metrics : {metrics_path}")
        print(f"Repeat  : {consistency_path}")
        if not metrics.empty:
            print("\nNOTE: cue_to_detection_ms is cue-relative, not pure algorithm latency.\n")
            print(metrics[show_columns].to_string(index=False))

    else:
        (
            phase_summary,
            false_activation,
            semantic_matrix,
            geometry_matrix,
            dino_matrix,
        ) = control_outputs_v31(
            df,
            semantic_floor=args.semantic_noise_floor,
            geometry_floor=args.geometry_noise_floor,
            dino_floor=args.dino_noise_floor,
            threshold_k=args.threshold_k,
            ema_ms=args.ema_ms,
            dino_ema_ms=args.dino_ema_ms,
        )

        phase_path = output_dir / "control_phase_summary_v31.csv"
        false_path = output_dir / "control_false_activation_v31.csv"
        semantic_matrix_path = output_dir / "control_false_matrix_semantic_v31.csv"
        geometry_matrix_path = output_dir / "control_false_matrix_geometry_v31.csv"
        dino_matrix_path = output_dir / "control_false_matrix_dino_v31.csv"

        phase_summary.to_csv(phase_path, index=False)
        false_activation.to_csv(false_path, index=False)
        semantic_matrix.to_csv(semantic_matrix_path, index=False)
        geometry_matrix.to_csv(geometry_matrix_path, index=False)
        dino_matrix.to_csv(dino_matrix_path, index=False)

        metadata["control_phase_summary_path"] = str(phase_path)
        metadata["control_false_activation_path"] = str(false_path)
        metadata["control_false_matrix_semantic_path"] = str(semantic_matrix_path)
        metadata["control_false_matrix_geometry_path"] = str(geometry_matrix_path)
        metadata["control_false_matrix_dino_path"] = str(dino_matrix_path)

        print(f"Control phase    : {phase_path}")
        print(f"Control false    : {false_path}")
        print(f"Semantic matrix  : {semantic_matrix_path}")
        print(f"Geometry matrix  : {geometry_matrix_path}")
        print(f"DINO matrix      : {dino_matrix_path}")

        if not phase_summary.empty:
            show = [
                c for c in [
                    "label",
                    "baseline_label",
                    "control_kind",
                    "face_detection_rate",
                    "max_semantic_false_activation_rate",
                    "worst_semantic_action",
                    "max_geometry_false_activation_rate",
                    "worst_geometry_action",
                    "max_dino_false_activation_rate",
                    "worst_dino_action",
                ]
                if c in phase_summary.columns
            ]
            print("\nCONTROL v3.1 worst-case false activation by phase:\n")
            print(phase_summary[show].to_string(index=False))

    df.to_csv(signals_path, index=False)
    metadata["signals_path"] = str(signals_path)

    metadata_tag = "v31" if protocol == "control" else "v3"
    metadata_path = output_dir / f"{protocol}_analysis_metadata_{metadata_tag}.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    face_rate = float(
        pd.to_numeric(df["face_detected"], errors="coerce").fillna(0).mean() * 100.0
    )
    print(f"Signals : {signals_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Face detection: {face_rate:.2f}%")

    ratio = metadata["timing"].get("capture_to_video_time_ratio")
    if ratio is not None and np.isfinite(ratio) and abs(ratio - 1.0) > 0.03:
        print(
            f"[WARN] capture/video timing ratio={ratio:.4f}; "
            "use capture timestamps for timing-sensitive conclusions."
        )


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Single-subject subtle facial movement analyzer v3.1"
    )
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session", default="s01")
    parser.add_argument("--protocol", choices=["all", "control", "upper", "lower"], default="all")

    parser.add_argument("--dino-every", type=int, default=3)
    parser.add_argument("--aligned-crop-size", type=int, default=256)
    parser.add_argument(
        "--dino-topk",
        type=float,
        default=0.25,
        help="Top fraction inside the action-specific DINO ROI.",
    )
    parser.add_argument(
        "--dino-roi-pad",
        type=float,
        default=0.06,
        help="Normalized padding around aligned landmark ROI.",
    )
    parser.add_argument("--min-pre-geometry-frames", type=int, default=20)
    parser.add_argument("--min-pre-dino-updates", type=int, default=4)

    # Only used by control video, whose baseline is the long INITIAL_NEUTRAL phase.
    parser.add_argument("--template-start", type=float, default=1.0)
    parser.add_argument("--template-end", type=float, default=4.0)
    parser.add_argument("--baseline-noise-start", type=float, default=4.5)
    parser.add_argument("--baseline-noise-end", type=float, default=9.0)
    parser.add_argument(
        "--control-baseline-tail-s",
        type=float,
        default=1.0,
        help=(
            "Stable tail length of each center/neutral phase used as the local "
            "reference for the following control nuisance phase."
        ),
    )

    parser.add_argument("--ema-ms", type=float, default=100.0)
    parser.add_argument("--dino-ema-ms", type=float, default=180.0)

    # Floors are deliberately non-zero to stop near-constant neutral channels
    # from producing meaningless 100x-1000x z-like evidence values.
    parser.add_argument("--semantic-noise-floor", type=float, default=0.01)
    parser.add_argument("--geometry-noise-floor", type=float, default=0.001)
    parser.add_argument("--dino-noise-floor", type=float, default=0.001)
    parser.add_argument("--motion-noise-floor", type=float, default=0.0005)
    parser.add_argument("--threshold-k", type=float, default=3.5)

    parser.add_argument("--min-sustain-ms", type=float, default=100.0)
    parser.add_argument("--motion-sustain-ms", type=float, default=67.0)
    parser.add_argument("--dino-min-updates", type=int, default=2)

    parser.add_argument("--head-qc-deg", type=float, default=5.0)
    parser.add_argument("--face-scale-qc-pct", type=float, default=8.0)
    parser.add_argument("--gaze-qc", type=float, default=0.15)
    parser.add_argument("--plot-clip", type=float, default=15.0)

    parser.add_argument(
        "--reuse-signals",
        action="store_true",
        help=(
            "Reuse existing signals (v3 action / v31 control). Only use when extraction parameters "
            "(DINO cadence/ROI/crop and trial-local feature logic) are unchanged."
        ),
    )

    args = parser.parse_args()

    if args.dino_every < 1:
        raise ValueError("--dino-every must be >= 1")
    if not 0.0 < args.dino_topk <= 1.0:
        raise ValueError("--dino-topk must be in (0,1]")
    if not 0.0 <= args.dino_roi_pad < 0.5:
        raise ValueError("--dino-roi-pad must be in [0,0.5)")
    if args.min_pre_geometry_frames < 5:
        raise ValueError("--min-pre-geometry-frames must be >= 5")
    if args.min_pre_dino_updates < 2:
        raise ValueError("--min-pre-dino-updates must be >= 2")
    if not 0.0 <= args.template_start < args.template_end:
        raise ValueError("template-start must be < template-end")
    if args.baseline_noise_start <= args.template_end:
        raise ValueError("baseline-noise-start must be after template-end")
    if args.control_baseline_tail_s <= 0:
        raise ValueError("--control-baseline-tail-s must be > 0")

    protocols = ["control", "upper", "lower"] if args.protocol == "all" else [args.protocol]

    print("\n==============================================")
    print(f"Single-subject facial movement analyzer v{VERSION}")
    print("==============================================")
    print(f"Participant : {args.participant}")
    print(f"Session     : {args.session}")
    print(f"Protocol    : {args.protocol}")
    print(f"DINO every  : {args.dino_every} frames")
    print(f"DINO ROI top: {args.dino_topk:.2f}")
    print("Semantic    : two-sided local-neutral evidence")
    print("Geometry    : trial-local neutral template")
    print("DINO        : trial-local PRE baseline + action ROI")
    print("Control     : preceding-center local baseline + action-specific FPR")
    print("Fusion      : none (modalities evaluated separately)\n")

    for protocol in protocols:
        print("\n----------------------------------------------")
        print(f"Analyzing {protocol.upper()}")
        print("----------------------------------------------")
        run_protocol(args, protocol)

    print("\nAnalysis finished.")


if __name__ == "__main__":
    main()
