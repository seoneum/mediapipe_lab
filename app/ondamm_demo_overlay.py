"""Reusable live/captured overlay for the ON DAMM submission demo."""
from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from micro_expression_camera import draw_dino_heatmap, draw_landmarks, draw_region
from micro_expression_signals import LEFT_BROW, LEFT_EYE, MOUTH, RIGHT_BROW, RIGHT_EYE


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int] = (245, 245, 245),
    scale: float = 0.52,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render_demo_overlay(
    frame: np.ndarray,
    signal: Mapping[str, Any],
    status: Mapping[str, Any],
) -> np.ndarray:
    """Render landmarks, motion values, recurrence state, and save feedback."""
    preview = np.array(frame, copy=True)
    height, width = preview.shape[:2]
    face_detected = bool(signal.get("face_detected"))
    if face_detected and signal.get("landmarks") is not None:
        points = np.asarray(signal["landmarks"])
        draw_dino_heatmap(preview, signal.get("dino_change_map"), signal.get("bbox"))
        draw_landmarks(preview, points)
        draw_region(preview, points, LEFT_EYE, (255, 120, 0), alpha=0.08)
        draw_region(preview, points, RIGHT_EYE, (255, 120, 0), alpha=0.08)
        draw_region(preview, points, LEFT_BROW, (0, 210, 255), alpha=0.08)
        draw_region(preview, points, RIGHT_BROW, (0, 210, 255), alpha=0.08)
        draw_region(preview, points, MOUTH, (190, 0, 255), alpha=0.10)

    panel_width = min(520, max(360, width - 24))
    panel_height = min(310, max(220, height - 24))
    panel = preview.copy()
    cv2.rectangle(panel, (12, 12), (12 + panel_width, 12 + panel_height), (10, 14, 24), -1)
    cv2.addWeighted(panel, 0.78, preview, 0.22, 0, preview)
    cv2.rectangle(preview, (12, 12), (12 + panel_width, 12 + panel_height), (78, 226, 190), 1)

    _text(preview, "ON:DAMM  LIVE MICRO-MOTION", 28, 40, color=(92, 242, 207), scale=0.64, thickness=2)
    _text(preview, "LOCAL ONLY  |  NO FULL-SESSION VIDEO", 28, 62, color=(175, 190, 205), scale=0.42)
    face_color = (70, 220, 120) if face_detected else (60, 80, 245)
    _text(preview, f"FACE {'TRACKED' if face_detected else 'NOT DETECTED'}", 28, 88, color=face_color, thickness=2)

    eyes = np.mean(
        [
            _number(signal.get("motion_left_eye")),
            _number(signal.get("motion_right_eye")),
        ]
    )
    brow = np.mean(
        [
            _number(signal.get("motion_left_brow")),
            _number(signal.get("motion_right_brow")),
        ]
    )
    _text(preview, f"MOUTH  {_number(signal.get('motion_mouth')):0.5f}", 28, 116, color=(235, 130, 255))
    _text(preview, f"EYES   {eyes:0.5f}", 28, 139, color=(255, 180, 90))
    _text(preview, f"BROW   {brow:0.5f}", 28, 162, color=(80, 225, 255))
    _text(preview, f"PERSONAL MOTION Z  {_number(status.get('motion_score')):0.2f}", 28, 188)

    calibration_remaining = _number(status.get("calibration_remaining"))
    if calibration_remaining > 0:
        state_text = f"CALIBRATING NEUTRAL  {calibration_remaining:0.1f}s"
        state_color = (0, 220, 255)
    elif status.get("temporal_enabled"):
        state_text = "MOTION ACTIVE" if status.get("motion_active") else "READY / OBSERVING"
        state_color = (70, 230, 145)
    else:
        state_text = "TEMPORAL OFF / CHECKPOINT REQUIRED"
        state_color = (70, 160, 255)
    _text(preview, state_text, 28, 217, color=state_color, thickness=2)

    occurrence = int(status.get("occurrence_count") or 0)
    threshold = int(status.get("occurrence_threshold") or 3)
    lifecycle = str(status.get("lifecycle") or "WAITING FOR EPISODE")
    candidate = str(status.get("candidate_id") or status.get("pattern_id") or "-")
    _text(preview, f"{lifecycle}", 28, 244, color=(220, 225, 235), thickness=2)
    _text(preview, f"{candidate}", 28, 267, color=(180, 195, 210))
    _text(preview, f"REPEAT  {occurrence} / {threshold}", 28, 293, color=(92, 242, 207), scale=0.64, thickness=2)

    if status.get("event_saved"):
        banner_height = 72
        top = max(12, height - banner_height - 18)
        banner = preview.copy()
        cv2.rectangle(banner, (12, top), (width - 12, top + banner_height), (20, 120, 70), -1)
        cv2.addWeighted(banner, 0.86, preview, 0.14, 0, preview)
        _text(preview, "REPEATING PATTERN DETECTED", 30, top + 29, color=(255, 255, 255), scale=0.72, thickness=2)
        _text(preview, "EVENT SAVED -> REFRESH ON:DAMM UI", 30, top + 56, color=(110, 255, 205), scale=0.58, thickness=2)

    _text(preview, "ESC/Q: stop", max(20, width - 132), max(24, height - 12), color=(185, 195, 205), scale=0.40)
    return preview
