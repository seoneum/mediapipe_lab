
from __future__ import annotations

"""Single-subject facial movement analyzer v2.

Designed for the recorded protocol:
    INITIAL_NEUTRAL -> PRE -> ONSET -> HOLD -> RELEASE -> POST

Key changes from v1:
- averaged neutral DINO baseline instead of one frame
- eye-aligned DINO crop to reduce translation/scale/roll nuisance
- DINO top-k patch score for localized subtle changes
- baseline-relative landmark *state* instead of only frame-to-frame motion
- trial-local PRE neutral calibration + long initial-neutral noise estimate
- noise floors to prevent huge z-score explosions
- causal smoothing for timing analysis
- hold-vs-neutral ROC AUC, cue-to-detection, repeat consistency, QC
- 2-of-3 multimodal consensus as a sanity check (not a trained classifier)

Important:
`cue_to_detection_ms` is measured from the instruction cue, not true physical
movement onset. It includes participant reaction time and instructed gradual motion.
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
VERSION = "2.0"

PROTOCOL_FILES = {
    "control": ("control_gaze.mp4", "control_gaze_labels.csv"),
    "upper": ("upper_face.mp4", "upper_face_labels.csv"),
    "lower": ("lower_face.mp4", "lower_face_labels.csv"),
}

NOSE = [1, 2, 4, 5, 6, 19, 94, 97, 98, 129, 168, 195, 197, 326, 327, 358]
REGIONS = {
    "mouth": sorted(set(MOUTH)),
    "brow": sorted(set(LEFT_BROW + RIGHT_BROW)),
    "eyes": sorted(set(LEFT_EYE + RIGHT_EYE)),
    "nose": sorted(set(NOSE)),
}

ACTION_CONFIG = {
    "brows_raise": {
        "blendshapes": ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
        "reducer": "mean", "region": "brow",
    },
    "brows_frown": {
        "blendshapes": ["browDownLeft", "browDownRight"],
        "reducer": "mean", "region": "brow",
    },
    "eyes_squint": {
        "blendshapes": ["eyeSquintLeft", "eyeSquintRight"],
        "reducer": "mean", "region": "eyes",
    },
    "eyes_wide": {
        "blendshapes": ["eyeWideLeft", "eyeWideRight"],
        "reducer": "mean", "region": "eyes",
    },
    "smile": {
        "blendshapes": ["mouthSmileLeft", "mouthSmileRight"],
        "reducer": "mean", "region": "mouth",
    },
    "mouth_frown": {
        "blendshapes": ["mouthFrownLeft", "mouthFrownRight"],
        "reducer": "mean", "region": "mouth",
    },
    "lip_press": {
        "blendshapes": ["mouthPressLeft", "mouthPressRight"],
        "reducer": "mean", "region": "mouth",
    },
    "lip_pucker": {
        # Important v2 change: use either mouthPucker or mouthFunnel.
        "blendshapes": ["mouthPucker", "mouthFunnel"],
        "reducer": "max", "region": "mouth",
    },
    "jaw_open": {
        "blendshapes": ["jawOpen"],
        "reducer": "mean", "region": "mouth",
    },
    "nose_wrinkle": {
        "blendshapes": ["noseSneerLeft", "noseSneerRight"],
        "reducer": "mean", "region": "nose",
    },
}

REGION_MOTION = {
    "mouth": "motion_mouth",
    "brow": "motion_brow",
    "eyes": "motion_eyes",
    "nose": "motion_mean",  # extractor has no dedicated nose frame-motion
}


def sf(x, default=np.nan):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def med(a):
    a = finite(a)
    return float(np.median(a)) if len(a) else np.nan


def valid_idx(n, idx):
    return [i for i in idx if 0 <= i < n]


def mean_xy(points, idx):
    idx = valid_idx(len(points), idx)
    return points[idx, :2].mean(axis=0) if idx else None


def robust_scale(values):
    """Robust scale; takes max(MAD-scale, IQR-scale)."""
    x = finite(values)
    if len(x) < 5:
        return np.nan
    m = np.median(x)
    mad_s = 1.4826 * np.median(np.abs(x - m))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr_s = (q75 - q25) / 1.349
    return float(max(mad_s, iqr_s))


def auc_rank(neg, pos):
    neg, pos = finite(neg), finite(pos)
    if len(neg) < 2 or len(pos) < 2:
        return np.nan
    z = np.concatenate([neg, pos])
    ranks = pd.Series(z).rank(method="average").to_numpy(float)
    n0, n1 = len(neg), len(pos)
    u = ranks[n0:].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n0 * n1))


def rank_corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    xr = pd.Series(x[ok]).rank(method="average").to_numpy(float)
    yr = pd.Series(y[ok]).rank(method="average").to_numpy(float)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def causal_ema(values, times_ms, tau_ms):
    """Causal EMA: no future samples are used for onset estimates."""
    x = np.asarray(values, float)
    t = np.asarray(times_ms, float)
    out = np.full(len(x), np.nan)
    prev = np.nan
    prev_t = np.nan
    tau_ms = max(float(tau_ms), 1e-6)
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        if not np.isfinite(prev):
            prev, prev_t, out[i] = v, t[i], v
            continue
        dt = max(0.0, t[i] - prev_t)
        a = 1.0 - math.exp(-dt / tau_ms)
        prev = a * v + (1.0 - a) * prev
        prev_t = t[i]
        out[i] = prev
    return out


def sparse_ema_hold(values, update_mask, times_ms, tau_ms):
    values = np.asarray(values, float)
    update_mask = np.asarray(update_mask, bool)
    times = np.asarray(times_ms, float)
    out_sparse = np.full(len(values), np.nan)
    idx = np.flatnonzero(update_mask & np.isfinite(values))
    if not len(idx):
        return out_sparse.copy(), out_sparse.copy()
    out_sparse[idx] = causal_ema(values[idx], times[idx], tau_ms)
    held = pd.Series(out_sparse).ffill().to_numpy(float)
    return out_sparse, held


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
            run, start = 0, None
    return np.nan


@dataclass
class Calibration:
    center: float
    scale: float
    threshold: float
    threshold_evidence: float
    q99: float
    local_scale: float
    global_scale: float
    floor: float


def calibrate(local, global_, floor, k):
    local, global_ = finite(local), finite(global_)
    if len(local) < 5:
        return None
    center = float(np.median(local))
    ls, gs = robust_scale(local), robust_scale(global_)
    scale = max([float(floor)] + [x for x in [ls, gs] if np.isfinite(x)])
    q99 = float(np.quantile(local, 0.99))
    threshold = max(center + k * scale, q99)
    return Calibration(center, scale, threshold, (threshold-center)/scale,
                       q99, ls, gs, float(floor))


def evidence(values, cal):
    x = np.asarray(values, float)
    if cal is None:
        return np.full(len(x), np.nan)
    return (x - cal.center) / cal.scale


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


def is_initial_baseline(row):
    if row is None:
        return False
    label = str(label_value(row, "label", "")).upper()
    return label == "INITIAL_NEUTRAL" or "BASELINE" in label


def phase_elapsed_s(row):
    return sf(label_value(row, "phase_elapsed_ms", 0.0), 0.0) / 1000.0

# ============================================================
# Alignment / geometry / extraction
# ============================================================

def aligned_face_crop(frame, points, size=256):
    """Eye-based affine alignment for DINO; removes x/y shift, scale and roll."""
    h, w = frame.shape[:2]
    ea = mean_xy(points, LEFT_EYE_CENTER)
    eb = mean_xy(points, RIGHT_EYE_CENTER)
    if ea is None or eb is None:
        return None
    ea = np.array([ea[0]*w, ea[1]*h], np.float32)
    eb = np.array([eb[0]*w, eb[1]*h], np.float32)
    left, right = (ea, eb) if ea[0] <= eb[0] else (eb, ea)
    v = right - left
    if np.linalg.norm(v) < 2.0:
        return None
    mid = (left + right) / 2.0
    perp = np.array([-v[1], v[0]], np.float32)
    if perp[1] < 0:
        perp *= -1
    src3 = mid + 0.9 * perp

    s = float(size)
    dl = np.array([0.30*s, 0.36*s], np.float32)
    dr = np.array([0.70*s, 0.36*s], np.float32)
    dm = (dl + dr) / 2.0
    dv = dr - dl
    dp = np.array([-dv[1], dv[0]], np.float32)
    if dp[1] < 0:
        dp *= -1
    d3 = dm + 0.9 * dp

    M = cv2.getAffineTransform(
        np.float32([left, right, src3]),
        np.float32([dl, dr, d3]),
    )
    return cv2.warpAffine(
        frame, M, (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def region_state(canonical, template, indices):
    idx = valid_idx(len(canonical), indices)
    if not idx:
        return np.nan
    d = canonical[idx, :2] - template[idx, :2]
    return float(np.linalg.norm(d, axis=1).mean())


def point_dist(points, i, j):
    if i >= len(points) or j >= len(points):
        return np.nan
    return float(np.linalg.norm(points[i, :2] - points[j, :2]))


def geometry_features(canonical, template):
    out = {
        "geom_global_state": float(np.linalg.norm(canonical[:, :2]-template[:, :2], axis=1).mean()),
    }
    for name, idx in REGIONS.items():
        out[f"geom_{name}_state"] = region_state(canonical, template, idx)

    # Interpretable diagnostics; primary geometry evidence still uses region state.
    out["geom_mouth_width_delta"] = point_dist(canonical, 61, 291) - point_dist(template, 61, 291)
    out["geom_mouth_open_delta"] = point_dist(canonical, 13, 14) - point_dist(template, 13, 14)
    eye_now = np.nanmean([point_dist(canonical,159,145), point_dist(canonical,386,374)])
    eye_ref = np.nanmean([point_dist(template,159,145), point_dist(template,386,374)])
    out["geom_eye_aperture_delta"] = float(eye_now - eye_ref)

    bi = valid_idx(len(canonical), REGIONS["brow"])
    out["geom_brow_vertical_delta"] = (
        float(np.mean(canonical[bi,1] - template[bi,1])) if bi else np.nan
    )
    ci = valid_idx(len(canonical), [61, 291])
    out["geom_mouth_corner_vertical_delta"] = (
        float(np.mean(canonical[ci,1] - template[ci,1])) if ci else np.nan
    )
    return out


def flatten_signal(signal):
    out = {"face_detected": int(bool(signal.get("face_detected", False)))}
    scalar = [
        "face_ratio", "yaw_deg", "pitch_deg", "roll_deg", "blink",
        "motion_mean", "motion_max", "motion_mouth",
        "motion_left_eye", "motion_right_eye",
        "motion_left_brow", "motion_right_brow",
        "brow_up_left", "brow_up_right", "brow_down_left", "brow_down_right",
        "brow_vertical_left", "brow_vertical_right",
    ]
    for k in scalar:
        out[k] = sf(signal.get(k, np.nan))
    out["motion_eyes"] = med([out.get("motion_left_eye"), out.get("motion_right_eye")])
    out["motion_brow"] = med([out.get("motion_left_brow"), out.get("motion_right_brow")])
    gaze = signal.get("gaze")
    out["gaze_horizontal"] = sf(gaze.get("horizontal")) if gaze else np.nan
    out["gaze_vertical"] = sf(gaze.get("vertical")) if gaze else np.nan
    for name, value in signal.get("blendshapes", {}).items():
        out[f"bs_{name}"] = sf(value)
    return out


def dino_summary(change_map, topk_fraction):
    if change_map is None:
        return np.nan, np.nan, np.nan
    x = finite(np.asarray(change_map, float).reshape(-1))
    if not len(x):
        return np.nan, np.nan, np.nan
    k = max(1, int(math.ceil(len(x) * min(max(topk_fraction, 1/len(x)), 1.0))))
    top = np.partition(x, len(x)-k)[-k:]
    return float(x.mean()), float(x.max()), float(top.mean())


@dataclass
class ExtractionInfo:
    fps: float
    video_frames: int
    label_rows: int
    extracted_frames: int
    geometry_template_frames: int
    dino_baseline_updates: int
    dino_every: int
    dino_update_hz: float
    aligned_crop_size: int
    dino_topk_fraction: float


def extract_video(video_path, labels_path, *, dino_every, crop_size, topk,
                  template_start, template_end):
    labels = load_labels(labels_path)
    lookup = {int(r.frame_idx): r for _, r in labels.iterrows()}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video  : {video_path}\nLabels : {labels_path}\nFPS    : {fps:.3f}\nFrames : {total}")
    if abs(total - len(labels)) > 2:
        print(f"[WARN] video/label frame mismatch: {total} vs {len(labels)}")

    # Existing extractor is reused for MediaPipe and the already-loaded DINO model.
    # We suppress its normal DINO cadence and run our aligned crop ourselves.
    ext = MicroExpressionSignalExtractor(dino_every=10**9)
    geom_samples = []
    dino_samples = []
    geom_template = None
    dino_ready = False
    last_dino = [np.nan, np.nan, np.nan, np.nan]
    rows = []
    fi = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            ts = int(round(fi / fps * 1000.0))
            sig = ext.extract(frame, fi, ts)
            lr = lookup.get(fi)
            row = {"frame_idx": fi, "analysis_timestamp_ms": ts}
            if lr is not None:
                for c in labels.columns:
                    if c != "frame_idx":
                        row[c] = lr[c]
            row.update(flatten_signal(sig))

            detected = bool(sig.get("face_detected", False))
            initial = is_initial_baseline(lr)
            elapsed = phase_elapsed_s(lr)
            canonical = sig.get("canonical_landmarks")
            points = sig.get("landmarks")

            if detected and canonical is not None and initial and template_start <= elapsed <= template_end:
                geom_samples.append(np.asarray(canonical, np.float32).copy())

            dino_updated = False
            if detected and points is not None and fi % dino_every == 0:
                crop = aligned_face_crop(frame, np.asarray(points, np.float32), crop_size)
                if crop is not None:
                    feat = ext.dino.extract(crop)
                    dino_updated = True
                    if initial and template_start <= elapsed <= template_end:
                        dino_samples.append(feat.clone())

            if geom_template is None and initial and elapsed >= template_end and len(geom_samples) >= 5:
                geom_template = np.median(np.stack(geom_samples), axis=0).astype(np.float32)
                print(f"Geometry neutral template: {len(geom_samples)} frames")

            if not dino_ready and initial and elapsed >= template_end and len(dino_samples) >= 3:
                base = torch.stack(dino_samples).mean(dim=0)
                ext.dino.baseline = F.normalize(base.float(), dim=-1).cpu()
                dino_ready = True
                print(f"DINO averaged baseline: {len(dino_samples)} updates")

            if detected and canonical is not None and geom_template is not None:
                row.update(geometry_features(np.asarray(canonical, np.float32), geom_template))
            else:
                for c in [
                    "geom_global_state", "geom_mouth_state", "geom_brow_state",
                    "geom_eyes_state", "geom_nose_state", "geom_mouth_width_delta",
                    "geom_mouth_open_delta", "geom_eye_aperture_delta",
                    "geom_brow_vertical_delta", "geom_mouth_corner_vertical_delta",
                ]:
                    row[c] = np.nan

            if dino_ready and dino_updated:
                mean_, max_, top_ = dino_summary(ext.dino.change_map(), topk)
                last_dino = [mean_, max_, top_, float(ext.dino.last_inference_ms)]
            row["dino_updated"] = int(dino_ready and dino_updated)
            row["dino_mean"], row["dino_max"], row["dino_topk_mean"], row["dino_inference_ms"] = last_dino
            row["dino_topk_update"] = row["dino_topk_mean"] if row["dino_updated"] else np.nan
            rows.append(row)
            fi += 1
            if fi % 300 == 0:
                print(f"  {fi}/{total}")
    finally:
        cap.release()
        ext.close()

    if geom_template is None:
        raise RuntimeError("Could not create geometry neutral template; check baseline labels.")
    if not dino_ready:
        raise RuntimeError("Could not create averaged DINO baseline; reduce --dino-every or check labels.")

    df = pd.DataFrame(rows)
    info = ExtractionInfo(
        fps=fps, video_frames=total, label_rows=len(labels), extracted_frames=len(df),
        geometry_template_frames=len(geom_samples), dino_baseline_updates=len(dino_samples),
        dino_every=dino_every, dino_update_hz=fps/dino_every,
        aligned_crop_size=crop_size, dino_topk_fraction=topk,
    )
    print(f"Extracted {len(df)} frames")
    return df, info

# ============================================================
# Derived scores / smoothing / baseline masks
# ============================================================

def add_semantic_scores(df):
    for action, cfg in ACTION_CONFIG.items():
        cols = [f"bs_{n}" for n in cfg["blendshapes"] if f"bs_{n}" in df.columns]
        out = f"score_{action}"
        if not cols:
            df[out] = np.nan
            continue
        v = df[cols].apply(pd.to_numeric, errors="coerce")
        df[out] = v.max(axis=1, skipna=True) if cfg["reducer"] == "max" else v.mean(axis=1, skipna=True)
    return df


def add_smoothing(df, ema_ms, dino_ema_ms):
    t = pd.to_numeric(df["analysis_timestamp_ms"], errors="coerce").to_numpy(float)
    cols = [f"score_{a}" for a in ACTION_CONFIG]
    cols += [
        "geom_mouth_state", "geom_brow_state", "geom_eyes_state", "geom_nose_state",
        "motion_mouth", "motion_brow", "motion_eyes", "motion_mean",
    ]
    for c in cols:
        if c in df.columns:
            df[f"{c}_ema"] = causal_ema(pd.to_numeric(df[c], errors="coerce"), t, ema_ms)

    if "dino_topk_update" in df.columns:
        sparse, held = sparse_ema_hold(
            pd.to_numeric(df["dino_topk_update"], errors="coerce"),
            pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1),
            t, dino_ema_ms,
        )
        df["dino_topk_ema_update"] = sparse
        df["dino_topk_ema"] = held
    return df


def global_baseline_mask(df, start_s, end_s):
    labels = df.get("label", pd.Series("", index=df.index)).fillna("").astype(str).str.upper()
    baseline = labels.eq("INITIAL_NEUTRAL") | labels.str.contains("BASELINE", regex=False)
    elapsed = pd.to_numeric(df.get("phase_elapsed_ms", -1), errors="coerce").fillna(-1) / 1000.0
    detected = pd.to_numeric(df.get("face_detected", 0), errors="coerce").fillna(0).astype(int).eq(1)
    return baseline & elapsed.between(start_s, end_s) & detected


def init_trial_columns(df):
    floats = [
        "semantic_raw", "semantic_smooth", "semantic_delta", "semantic_evidence",
        "semantic_threshold_raw", "semantic_threshold_evidence",
        "geometry_raw", "geometry_smooth", "geometry_delta", "geometry_evidence",
        "geometry_threshold_raw", "geometry_threshold_evidence",
        "motion_reference_raw", "motion_reference_smooth", "motion_reference_evidence",
        "motion_reference_threshold_raw", "dino_smooth", "dino_delta", "dino_evidence",
        "dino_threshold_raw", "dino_threshold_evidence", "consensus_votes",
        "head_delta_deg", "face_scale_delta_pct",
    ]
    for c in floats:
        df[c] = np.nan
    for c in ["semantic_detect", "geometry_detect", "motion_reference_detect", "dino_detect", "consensus_detect"]:
        df[c] = 0
    return df


def trial_masks(df, action, repeat_idx):
    a = df["action"].fillna("").astype(str).eq(action)
    r = pd.to_numeric(df["repeat_idx"], errors="coerce").eq(repeat_idx)
    p = df["movement_phase"].fillna("").astype(str)
    trial = a & r
    return {
        "trial": trial,
        "pre": trial & p.eq("pre_neutral"),
        "onset": trial & p.eq("onset"),
        "hold": trial & p.eq("hold"),
        "release": trial & p.eq("release"),
        "post": trial & p.eq("post_neutral"),
        "active": trial & p.isin(["onset", "hold", "release"]),
    }


def calibrate_trials(df, global_mask, *, semantic_floor, geometry_floor, dino_floor,
                     motion_floor, threshold_k):
    if not {"action", "repeat_idx", "movement_phase"}.issubset(df.columns):
        return df
    df = init_trial_columns(df)
    dino_global_mask = global_mask & pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
    dino_global = pd.to_numeric(df.loc[dino_global_mask, "dino_topk_ema_update"], errors="coerce")

    actions = [a for a in df["action"].dropna().astype(str).unique() if a in ACTION_CONFIG]
    for action in actions:
        cfg = ACTION_CONFIG[action]
        sem = f"score_{action}"
        sem_s = f"{sem}_ema"
        geom = f"geom_{cfg['region']}_state"
        geom_s = f"{geom}_ema"
        motion = REGION_MOTION[cfg["region"]]
        motion_s = f"{motion}_ema"

        sem_global = pd.to_numeric(df.loc[global_mask, sem_s], errors="coerce")
        geom_global = pd.to_numeric(df.loc[global_mask, geom_s], errors="coerce")
        motion_global = pd.to_numeric(df.loc[global_mask, motion_s], errors="coerce")

        repeats = pd.to_numeric(
            df.loc[df["action"].fillna("").astype(str).eq(action), "repeat_idx"],
            errors="coerce"
        ).dropna().astype(int).unique()

        for repeat_idx in repeats:
            m = trial_masks(df, action, int(repeat_idx))
            idx, pre = df.index[m["trial"]], df.index[m["pre"]]
            if len(idx) == 0 or len(pre) < 5:
                continue

            sem_cal = calibrate(pd.to_numeric(df.loc[pre, sem_s], errors="coerce"), sem_global, semantic_floor, threshold_k)
            geom_cal = calibrate(pd.to_numeric(df.loc[pre, geom_s], errors="coerce"), geom_global, geometry_floor, threshold_k)
            mot_cal = calibrate(pd.to_numeric(df.loc[pre, motion_s], errors="coerce"), motion_global, motion_floor, threshold_k)
            pre_dino = m["pre"] & pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
            dino_cal = calibrate(
                pd.to_numeric(df.loc[pre_dino, "dino_topk_ema_update"], errors="coerce"),
                dino_global, dino_floor, threshold_k,
            )

            df.loc[idx, "semantic_raw"] = pd.to_numeric(df.loc[idx, sem], errors="coerce")
            df.loc[idx, "semantic_smooth"] = pd.to_numeric(df.loc[idx, sem_s], errors="coerce")
            df.loc[idx, "geometry_raw"] = pd.to_numeric(df.loc[idx, geom], errors="coerce")
            df.loc[idx, "geometry_smooth"] = pd.to_numeric(df.loc[idx, geom_s], errors="coerce")
            df.loc[idx, "motion_reference_raw"] = pd.to_numeric(df.loc[idx, motion], errors="coerce")
            df.loc[idx, "motion_reference_smooth"] = pd.to_numeric(df.loc[idx, motion_s], errors="coerce")
            df.loc[idx, "dino_smooth"] = pd.to_numeric(df.loc[idx, "dino_topk_ema"], errors="coerce")

            specs = [
                ("semantic", sem_s, sem_cal),
                ("geometry", geom_s, geom_cal),
                ("motion_reference", motion_s, mot_cal),
                ("dino", "dino_topk_ema", dino_cal),
            ]
            face_ok = pd.to_numeric(df.loc[idx, "face_detected"], errors="coerce").fillna(0).astype(int).eq(1).to_numpy()
            for prefix, source, cal in specs:
                vals = pd.to_numeric(df.loc[idx, source], errors="coerce").to_numpy(float)
                ev = evidence(vals, cal)
                df.loc[idx, f"{prefix}_evidence"] = ev
                if cal is None:
                    continue
                df.loc[idx, f"{prefix}_threshold_raw"] = cal.threshold
                if prefix != "motion_reference":
                    df.loc[idx, f"{prefix}_threshold_evidence"] = cal.threshold_evidence
                det = (ev >= cal.threshold_evidence) & face_ok
                df.loc[idx, f"{prefix}_detect"] = det.astype(int)
                if prefix in {"semantic", "geometry", "dino"}:
                    df.loc[idx, f"{prefix}_delta"] = vals - cal.center

            # Local PRE-relative head/scale QC. Report only; do not exclude automatically.
            yaw0 = med(pd.to_numeric(df.loc[pre, "yaw_deg"], errors="coerce"))
            pitch0 = med(pd.to_numeric(df.loc[pre, "pitch_deg"], errors="coerce"))
            roll0 = med(pd.to_numeric(df.loc[pre, "roll_deg"], errors="coerce"))
            yaw = pd.to_numeric(df.loc[idx, "yaw_deg"], errors="coerce").to_numpy(float)
            pitch = pd.to_numeric(df.loc[idx, "pitch_deg"], errors="coerce").to_numpy(float)
            roll = pd.to_numeric(df.loc[idx, "roll_deg"], errors="coerce").to_numpy(float)
            df.loc[idx, "head_delta_deg"] = np.nanmax(np.stack([
                np.abs(yaw-yaw0), np.abs(pitch-pitch0), np.abs(roll-roll0)
            ], axis=1), axis=1)
            fr0 = med(pd.to_numeric(df.loc[pre, "face_ratio"], errors="coerce"))
            if np.isfinite(fr0) and abs(fr0) > 1e-9:
                fr = pd.to_numeric(df.loc[idx, "face_ratio"], errors="coerce").to_numpy(float)
                df.loc[idx, "face_scale_delta_pct"] = np.abs(fr/fr0 - 1.0) * 100.0

            votes = sum(
                pd.to_numeric(df.loc[idx, c], errors="coerce").fillna(0).to_numpy(int)
                for c in ["semantic_detect", "geometry_detect", "dino_detect"]
            )
            df.loc[idx, "consensus_votes"] = votes
            df.loc[idx, "consensus_detect"] = (votes >= 2).astype(int)
    return df


def hold_auc(df, pre, hold, column, dino=False):
    if dino:
        upd = pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
        pre, hold = pre & upd, hold & upd
    return auc_rank(
        pd.to_numeric(df.loc[pre, column], errors="coerce"),
        pd.to_numeric(df.loc[hold, column], errors="coerce"),
    )


def hold_delta(df, pre, hold, column, dino=False):
    if dino:
        upd = pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
        pre, hold = pre & upd, hold & upd
    a = med(pd.to_numeric(df.loc[pre, column], errors="coerce"))
    b = med(pd.to_numeric(df.loc[hold, column], errors="coerce"))
    return float(b-a) if np.isfinite(a) and np.isfinite(b) else np.nan


def build_metrics(df, *, sustain_ms, motion_sustain_ms, dino_min_updates,
                  head_qc_deg, face_scale_qc_pct):
    rows = []
    actions = [a for a in df["action"].dropna().astype(str).unique() if a in ACTION_CONFIG]
    for action in actions:
        repeats = pd.to_numeric(
            df.loc[df["action"].fillna("").astype(str).eq(action), "repeat_idx"], errors="coerce"
        ).dropna().astype(int).unique()
        for r in repeats:
            m = trial_masks(df, action, int(r))
            active, onset, hold = df.loc[m["active"]], df.loc[m["onset"]], df.loc[m["hold"]]
            if active.empty or onset.empty:
                continue
            item = {
                "action": action, "repeat_idx": int(r), "active_frames": len(active),
                "face_detection_rate": float(pd.to_numeric(active["face_detected"], errors="coerce").fillna(0).mean()),
                "head_delta_peak_deg": sf(pd.to_numeric(active["head_delta_deg"], errors="coerce").max()),
                "face_scale_delta_peak_pct": sf(pd.to_numeric(active["face_scale_delta_pct"], errors="coerce").max()),
            }
            item["qc_pass"] = int(
                item["face_detection_rate"] >= 0.98
                and (not np.isfinite(item["head_delta_peak_deg"]) or item["head_delta_peak_deg"] <= head_qc_deg)
                and (not np.isfinite(item["face_scale_delta_peak_pct"]) or item["face_scale_delta_peak_pct"] <= face_scale_qc_pct)
            )

            for name, col, dino in [
                ("semantic", "semantic_smooth", False),
                ("geometry", "geometry_smooth", False),
                ("dino", "dino_smooth", True),
            ]:
                auc = hold_auc(df, m["pre"], m["hold"], col, dino)
                item[f"{name}_hold_auc"] = auc
                item[f"{name}_hold_auc_separability"] = max(auc, 1-auc) if np.isfinite(auc) else np.nan
                item[f"{name}_hold_polarity"] = "positive" if np.isfinite(auc) and auc >= .5 else ("negative" if np.isfinite(auc) else "")
                item[f"{name}_hold_delta_raw"] = hold_delta(df, m["pre"], m["hold"], col, dino)
                item[f"{name}_peak_evidence"] = sf(pd.to_numeric(active[f"{name}_evidence"], errors="coerce").max())
                he = hold
                if dino:
                    he = hold[pd.to_numeric(hold["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)]
                item[f"{name}_hold_detection_rate"] = (
                    float(pd.to_numeric(he[f"{name}_detect"], errors="coerce").fillna(0).mean()) if len(he) else np.nan
                )
                item[f"{name}_progress_corr"] = rank_corr(
                    pd.to_numeric(df.loc[m["active"], "intended_progress"], errors="coerce"),
                    pd.to_numeric(df.loc[m["active"], col], errors="coerce"),
                ) if "intended_progress" in df.columns else np.nan

            t = pd.to_numeric(onset["phase_elapsed_ms"], errors="coerce").to_numpy(float)
            motion_ms = first_sustained(t, pd.to_numeric(onset["motion_reference_detect"], errors="coerce").fillna(0).to_numpy(int).astype(bool), motion_sustain_ms)
            item["motion_reference_cue_ms"] = motion_ms
            for name in ["semantic", "geometry", "consensus"]:
                cue = first_sustained(t, pd.to_numeric(onset[f"{name}_detect"], errors="coerce").fillna(0).to_numpy(int).astype(bool), sustain_ms)
                item[f"{name}_cue_to_detection_ms"] = cue
                if name != "consensus":
                    item[f"{name}_lag_vs_motion_ms"] = cue-motion_ms if np.isfinite(cue) and np.isfinite(motion_ms) else np.nan

            d_on = onset[pd.to_numeric(onset["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)]
            dcue = first_n_updates(
                pd.to_numeric(d_on["phase_elapsed_ms"], errors="coerce"),
                pd.to_numeric(d_on["dino_detect"], errors="coerce").fillna(0).to_numpy(int).astype(bool),
                dino_min_updates,
            ) if len(d_on) else np.nan
            item["dino_cue_to_detection_ms"] = dcue
            item["dino_lag_vs_motion_ms"] = dcue-motion_ms if np.isfinite(dcue) and np.isfinite(motion_ms) else np.nan
            item["consensus_hold_detection_rate"] = float(pd.to_numeric(hold["consensus_detect"], errors="coerce").fillna(0).mean()) if len(hold) else np.nan
            rows.append(item)
    return pd.DataFrame(rows)

# ============================================================
# Repeat consistency / control / plots
# ============================================================

def resample_delta(active, pre, column, n=101):
    if column not in active.columns or column not in pre.columns:
        return None
    b = med(pd.to_numeric(pre[column], errors="coerce"))
    if not np.isfinite(b):
        return None
    t = pd.to_numeric(active["analysis_timestamp_ms"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(active[column], errors="coerce").to_numpy(float) - b
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 4:
        return None
    t, y = t[ok], y[ok]
    if t[-1] <= t[0]:
        return None
    u = (t-t[0])/(t[-1]-t[0])
    return np.interp(np.linspace(0,1,n), u, y)


def repeat_consistency(df, metrics):
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    for action in metrics["action"].unique():
        repeats = metrics.loc[metrics["action"].eq(action), "repeat_idx"].astype(int).tolist()
        for name, col in [("semantic","semantic_smooth"),("geometry","geometry_smooth"),("dino","dino_smooth")]:
            curves, amps = [], []
            for r in repeats:
                m = trial_masks(df, action, r)
                active, pre, hold = df.loc[m["active"]], df.loc[m["pre"]], df.loc[m["hold"]]
                c = resample_delta(active, pre, col)
                if c is not None:
                    curves.append(c)
                b = med(pd.to_numeric(pre[col], errors="coerce")) if col in pre else np.nan
                h = med(pd.to_numeric(hold[col], errors="coerce")) if col in hold else np.nan
                if np.isfinite(b) and np.isfinite(h):
                    amps.append(h-b)
            corrs = []
            for i in range(len(curves)):
                for j in range(i+1, len(curves)):
                    if np.std(curves[i]) > 1e-12 and np.std(curves[j]) > 1e-12:
                        corrs.append(float(np.corrcoef(curves[i], curves[j])[0,1]))
            mean_amp = float(np.mean(amps)) if amps else np.nan
            cv = float(np.std(amps)/abs(mean_amp)) if amps and abs(mean_amp) > 1e-12 else np.nan
            rows.append({
                "action": action, "signal": name, "repeat_count": len(curves), "pair_count": len(corrs),
                "mean_pairwise_corr": float(np.mean(corrs)) if corrs else np.nan,
                "min_pairwise_corr": float(np.min(corrs)) if corrs else np.nan,
                "hold_amplitude_cv": cv,
            })
    return pd.DataFrame(rows)


def control_outputs(df, baseline_mask, *, semantic_floor, geometry_floor, dino_floor, threshold_k):
    """Nuisance test: how often expression-like channels activate in gaze/head/blink control phases."""
    if "label" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    sem_cal, geom_cal = {}, {}
    for action, cfg in ACTION_CONFIG.items():
        sc = f"score_{action}_ema"
        gc = f"geom_{cfg['region']}_state_ema"
        vals = pd.to_numeric(df.loc[baseline_mask, sc], errors="coerce")
        sem_cal[action] = calibrate(vals, vals, semantic_floor, threshold_k)
        vals = pd.to_numeric(df.loc[baseline_mask, gc], errors="coerce")
        geom_cal[action] = calibrate(vals, vals, geometry_floor, threshold_k)
    dmask = baseline_mask & pd.to_numeric(df["dino_updated"], errors="coerce").fillna(0).astype(int).eq(1)
    vals = pd.to_numeric(df.loc[dmask, "dino_topk_ema_update"], errors="coerce")
    dcal = calibrate(vals, vals, dino_floor, threshold_k)

    phase_rows, false_rows = [], []
    for label, group in df.groupby(df["label"].fillna("").astype(str)):
        item = {
            "label": label, "frames": len(group),
            "face_detection_rate": float(pd.to_numeric(group["face_detected"], errors="coerce").fillna(0).mean()),
        }
        for n in ["yaw_deg","pitch_deg","roll_deg","gaze_horizontal","gaze_vertical"]:
            v = pd.to_numeric(group[n], errors="coerce")
            item[f"{n}_range"] = sf(v.max()-v.min())
        item["blink_max"] = sf(pd.to_numeric(group["blink"], errors="coerce").max())
        if dcal:
            ev = evidence(pd.to_numeric(group["dino_topk_ema"], errors="coerce"), dcal)
            item["dino_activation_rate"] = float(np.nanmean(ev >= dcal.threshold_evidence))
        else:
            item["dino_activation_rate"] = np.nan
        phase_rows.append(item)

        for action, cfg in ACTION_CONFIG.items():
            s, g = sem_cal[action], geom_cal[action]
            sev = evidence(pd.to_numeric(group[f"score_{action}_ema"], errors="coerce"), s)
            gev = evidence(pd.to_numeric(group[f"geom_{cfg['region']}_state_ema"], errors="coerce"), g)
            false_rows.append({
                "label": label, "action": action,
                "semantic_activation_rate": float(np.nanmean(sev >= s.threshold_evidence)) if s else np.nan,
                "geometry_activation_rate": float(np.nanmean(gev >= g.threshold_evidence)) if g else np.nan,
            })
    return pd.DataFrame(phase_rows), pd.DataFrame(false_rows)


def boundaries(subset):
    out, prev = [], None
    for _, r in subset.iterrows():
        label = str(r.get("label", ""))
        if label != prev:
            out.append(sf(r.get("plot_time_s")))
            prev = label
    return out


def plot_action(df, action, outdir, threshold_k, clip):
    s = df[df["action"].fillna("").astype(str).eq(action)].copy()
    if s.empty:
        return
    t0 = sf(s["analysis_timestamp_ms"].iloc[0], 0)
    s["plot_time_s"] = (pd.to_numeric(s["analysis_timestamp_ms"], errors="coerce") - t0)/1000.0
    x = s["plot_time_s"]
    b = boundaries(s)

    fig, ax = plt.subplots(figsize=(14,6))
    for col, label in [
        ("semantic_evidence","MediaPipe semantic evidence"),
        ("geometry_evidence","baseline-relative geometry evidence"),
        ("dino_evidence","aligned DINO top-k evidence"),
    ]:
        y = pd.to_numeric(s[col], errors="coerce").clip(-clip, clip)
        ax.plot(x, y, label=label)
    ax.axhline(threshold_k, linestyle=":", linewidth=1, label=f"base k={threshold_k:g}")
    for p in b:
        if np.isfinite(p):
            ax.axvline(p, linestyle="--", linewidth=.7, alpha=.25)
    ax.set(title=f"{action} - multimodal evidence", xlabel="Time [s]",
           ylabel="Local-neutral evidence [noise units]", ylim=(-min(3,clip), clip))
    ax.grid(alpha=.2); ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(outdir/f"{action}_evidence.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(3,1,figsize=(14,9),sharex=True)
    for ax, col, label in [
        (axes[0],"semantic_smooth","MediaPipe semantic score"),
        (axes[1],"geometry_smooth","baseline-relative geometry state"),
        (axes[2],"dino_smooth","aligned DINO top-k change"),
    ]:
        ax.plot(x, pd.to_numeric(s[col], errors="coerce")); ax.set_ylabel(label); ax.grid(alpha=.2)
        for p in b:
            if np.isfinite(p): ax.axvline(p, linestyle="--", linewidth=.7, alpha=.25)
    axes[-1].set_xlabel("Time [s]"); fig.suptitle(f"{action} - raw smoothed modalities"); fig.tight_layout()
    fig.savefig(outdir/f"{action}_raw.png", dpi=170); plt.close(fig)


def timing_info(df):
    out = {"analysis_duration_s": sf(df["analysis_timestamp_ms"].iloc[-1])/1000.0 if len(df) else np.nan}
    for c, name in [("capture_timestamp_ms","capture_duration_s"),("video_timestamp_ms","recorded_video_duration_s")]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(x) >= 2: out[name] = float((x.iloc[-1]-x.iloc[0])/1000.0)
    if out.get("recorded_video_duration_s",0) > 0 and "capture_duration_s" in out:
        out["capture_to_video_time_ratio"] = out["capture_duration_s"]/out["recorded_video_duration_s"]
    return out


# ============================================================
# Protocol runner
# ============================================================

def run_protocol(args, protocol):
    video_name, label_name = PROTOCOL_FILES[protocol]
    indir = RECORDING_ROOT / args.participant / args.session
    video, labels = indir/video_name, indir/label_name
    if not video.exists() or not labels.exists():
        print(f"[SKIP] missing {protocol} video/labels")
        return
    outdir = OUTPUT_ROOT / args.participant / args.session
    outdir.mkdir(parents=True, exist_ok=True)
    sig_path = outdir/f"{protocol}_signals_v2.csv"

    if args.reuse_signals and sig_path.exists():
        print(f"Reusing {sig_path}")
        df = pd.read_csv(sig_path); extraction = None
    else:
        df, extraction = extract_video(
            video, labels, dino_every=args.dino_every, crop_size=args.aligned_crop_size,
            topk=args.dino_topk, template_start=args.template_start, template_end=args.template_end,
        )

    df = add_semantic_scores(df)
    df = add_smoothing(df, args.ema_ms, args.dino_ema_ms)
    gmask = global_baseline_mask(df, args.baseline_noise_start, args.baseline_noise_end)
    if int(gmask.sum()) < 10:
        raise RuntimeError(f"Too few stable baseline frames: {int(gmask.sum())}")

    meta = {
        "analyzer_version": VERSION, "participant": args.participant, "session": args.session,
        "protocol": protocol, "parameters": vars(args), "stable_baseline_frames": int(gmask.sum()),
        "timing": timing_info(df),
    }
    if extraction is not None:
        meta["extraction"] = asdict(extraction)

    if protocol in {"upper","lower"}:
        df = calibrate_trials(
            df, gmask, semantic_floor=args.semantic_noise_floor,
            geometry_floor=args.geometry_noise_floor, dino_floor=args.dino_noise_floor,
            motion_floor=args.motion_noise_floor, threshold_k=args.threshold_k,
        )
        metrics = build_metrics(
            df, sustain_ms=args.min_sustain_ms, motion_sustain_ms=args.motion_sustain_ms,
            dino_min_updates=args.dino_min_updates, head_qc_deg=args.head_qc_deg,
            face_scale_qc_pct=args.face_scale_qc_pct,
        )
        cons = repeat_consistency(df, metrics)
        mpath = outdir/f"{protocol}_trial_metrics_v2.csv"
        cpath = outdir/f"{protocol}_repeat_consistency_v2.csv"
        metrics.to_csv(mpath, index=False); cons.to_csv(cpath, index=False)
        pdir = outdir/"plots_v2"/protocol; pdir.mkdir(parents=True, exist_ok=True)
        for action in [a for a in df["action"].dropna().astype(str).unique() if a in ACTION_CONFIG]:
            plot_action(df, action, pdir, args.threshold_k, args.plot_clip)
        meta.update(metrics_path=str(mpath), repeat_consistency_path=str(cpath))

        show = [c for c in [
            "action","repeat_idx","semantic_hold_auc","geometry_hold_auc","dino_hold_auc",
            "semantic_cue_to_detection_ms","geometry_cue_to_detection_ms","dino_cue_to_detection_ms",
            "consensus_cue_to_detection_ms","motion_reference_cue_ms","head_delta_peak_deg","qc_pass"
        ] if c in metrics.columns]
        print(f"Metrics : {mpath}\nRepeat  : {cpath}")
        if not metrics.empty:
            print("\nNOTE: cue_to_detection_ms is cue-relative, not pure algorithm latency.\n")
            print(metrics[show].to_string(index=False))
    else:
        phase, false = control_outputs(
            df, gmask, semantic_floor=args.semantic_noise_floor,
            geometry_floor=args.geometry_noise_floor, dino_floor=args.dino_noise_floor,
            threshold_k=args.threshold_k,
        )
        ppath, fpath = outdir/"control_phase_summary_v2.csv", outdir/"control_false_activation_v2.csv"
        phase.to_csv(ppath,index=False); false.to_csv(fpath,index=False)
        meta.update(control_phase_summary_path=str(ppath), control_false_activation_path=str(fpath))
        print(f"Control phase : {ppath}\nControl false : {fpath}")

    df.to_csv(sig_path, index=False)
    meta["signals_path"] = str(sig_path)
    metapath = outdir/f"{protocol}_analysis_metadata_v2.json"
    with open(metapath,"w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)
    face_rate = float(pd.to_numeric(df["face_detected"],errors="coerce").fillna(0).mean()*100)
    print(f"Signals : {sig_path}\nMetadata: {metapath}\nFace detection: {face_rate:.2f}%")
    ratio = meta["timing"].get("capture_to_video_time_ratio")
    if ratio is not None and np.isfinite(ratio) and abs(ratio-1) > .03:
        print(f"[WARN] capture/video timing ratio={ratio:.4f}; use capture timestamps for timing-sensitive analysis.")


def main():
    p = argparse.ArgumentParser(description="Single-subject subtle facial movement analyzer v2")
    p.add_argument("--participant", required=True)
    p.add_argument("--session", default="s01")
    p.add_argument("--protocol", choices=["all","control","upper","lower"], default="all")
    p.add_argument("--dino-every", type=int, default=3)
    p.add_argument("--aligned-crop-size", type=int, default=256)
    p.add_argument("--dino-topk", type=float, default=.10)
    p.add_argument("--template-start", type=float, default=1.0)
    p.add_argument("--template-end", type=float, default=4.0)
    p.add_argument("--baseline-noise-start", type=float, default=4.5)
    p.add_argument("--baseline-noise-end", type=float, default=9.0)
    p.add_argument("--ema-ms", type=float, default=100.0)
    p.add_argument("--dino-ema-ms", type=float, default=180.0)
    p.add_argument("--semantic-noise-floor", type=float, default=.01)
    p.add_argument("--geometry-noise-floor", type=float, default=.001)
    p.add_argument("--dino-noise-floor", type=float, default=.001)
    p.add_argument("--motion-noise-floor", type=float, default=.0005)
    p.add_argument("--threshold-k", type=float, default=3.5)
    p.add_argument("--min-sustain-ms", type=float, default=100.0)
    p.add_argument("--motion-sustain-ms", type=float, default=67.0)
    p.add_argument("--dino-min-updates", type=int, default=2)
    p.add_argument("--head-qc-deg", type=float, default=5.0)
    p.add_argument("--face-scale-qc-pct", type=float, default=8.0)
    p.add_argument("--plot-clip", type=float, default=15.0)
    p.add_argument("--reuse-signals", action="store_true",
                   help="Reuse *_signals_v2.csv; only if extraction parameters are unchanged.")
    args = p.parse_args()

    if args.dino_every < 1: raise ValueError("--dino-every must be >=1")
    if not 0 < args.dino_topk <= 1: raise ValueError("--dino-topk must be in (0,1]")
    if not (0 <= args.template_start < args.template_end < args.baseline_noise_end):
        raise ValueError("template-start < template-end < baseline-noise-end required")
    if args.baseline_noise_start <= args.template_end:
        raise ValueError("baseline-noise-start must be after template-end")

    protocols = ["control","upper","lower"] if args.protocol == "all" else [args.protocol]
    print("\n==============================================")
    print(f"Single-subject facial movement analyzer v{VERSION}")
    print("==============================================")
    print(f"Participant : {args.participant}\nSession     : {args.session}\nProtocol    : {args.protocol}")
    print(f"DINO every  : {args.dino_every} frames\nDINO top-k  : {args.dino_topk:.2f}")
    print(f"Threshold   : k={args.threshold_k:g} + local q99 guard\n")
    for protocol in protocols:
        print("\n----------------------------------------------")
        print(f"Analyzing {protocol.upper()}")
        print("----------------------------------------------")
        run_protocol(args, protocol)
    print("\nAnalysis finished.")


if __name__ == "__main__":
    main()
