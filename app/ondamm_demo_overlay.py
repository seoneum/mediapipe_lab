"""Reusable live/captured overlay for the ON DAMM submission demo."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import cv2
import numpy as np

from micro_expression_camera import draw_dino_heatmap, draw_landmarks, draw_region
from micro_expression_signals import LEFT_BROW, LEFT_EYE, MOUTH, RIGHT_BROW, RIGHT_EYE


@dataclass(frozen=True)
class GuidanceTiming:
    """Presentation-only timing for three clearly cued independent motions."""

    stable_seconds: float = 1.0
    announcement_seconds: float = 0.8
    countdown_seconds: int = 3
    move_seconds: float = 1.0
    neutral_seconds: float = 2.0
    repeats: int = 3


class GuidedDemoCountdown:
    """Add large, deterministic action cues without touching analysis frames."""

    def __init__(
        self,
        timing: GuidanceTiming | None = None,
        *,
        action_label: str = "같은 짧은 움직임",
    ) -> None:
        self.timing = timing or GuidanceTiming()
        self.action_label = str(action_label).strip() or "같은 짧은 움직임"
        self._stable_since: float | None = None
        self._sequence_started_at: float | None = None

    def decorate(self, status: Mapping[str, Any], *, timestamp: float) -> dict[str, Any]:
        decorated = dict(status)
        face_lost = bool(decorated.get("face_lost"))
        calibration_ready = bool(decorated.get("calibration_ready"))
        warming_up = bool(decorated.get("warming_up"))

        if face_lost:
            self._reset()
            return self._with_guidance(
                decorated,
                phase="face_lost",
                title="얼굴을 화면에 보여 주세요",
                fallback="SHOW YOUR FACE",
                detail="카메라는 얼굴이 다시 보일 때까지 분석을 멈춥니다",
            )
        if not calibration_ready:
            self._reset()
            return self._with_guidance(
                decorated,
                phase="calibrating",
                title="얼굴을 편하게 유지하세요",
                fallback="RELAX YOUR FACE",
                detail="개인 기준선을 보정하고 있습니다",
            )
        if warming_up:
            self._reset()
            return self._with_guidance(
                decorated,
                phase="warming_up",
                title="잠시 그대로 있어 주세요",
                fallback="PLEASE HOLD STILL",
                detail="시간 흐름 정보를 준비하고 있습니다",
            )

        now = float(timestamp)
        if self._sequence_started_at is None:
            if bool(decorated.get("motion_active")):
                self._stable_since = None
                return self._with_guidance(
                    decorated,
                    phase="settling",
                    title="먼저 중립으로 돌아오세요",
                    fallback="RETURN TO NEUTRAL FIRST",
                    detail="움직임이 멈추면 카운트다운을 시작합니다",
                )
            if self._stable_since is None:
                self._stable_since = now
            stable_elapsed = max(0.0, now - self._stable_since)
            if stable_elapsed < self.timing.stable_seconds:
                return self._with_guidance(
                    decorated,
                    phase="settling",
                    title="잠시 그대로 있어 주세요",
                    fallback="PLEASE HOLD STILL",
                    detail="카운트다운 준비 중",
                )
            self._sequence_started_at = self._stable_since + self.timing.stable_seconds

        cycle_seconds = (
            self.timing.announcement_seconds
            + float(self.timing.countdown_seconds)
            + self.timing.move_seconds
            + self.timing.neutral_seconds
        )
        sequence_elapsed = max(0.0, now - self._sequence_started_at)
        cycle_index = int(sequence_elapsed // cycle_seconds)
        if cycle_index >= self.timing.repeats:
            return self._with_guidance(
                decorated,
                phase="complete",
                title="세 번의 안내가 끝났습니다",
                fallback="THREE CUES COMPLETE",
                detail="화면의 반복 횟수를 확인하세요",
                cycle=self.timing.repeats,
            )

        local = sequence_elapsed - cycle_index * cycle_seconds
        cycle = cycle_index + 1
        if local < self.timing.announcement_seconds:
            return self._with_guidance(
                decorated,
                phase="announcement",
                title="3초 뒤에 시작합니다",
                fallback="STARTING IN 3 SECONDS",
                detail=f"{cycle}번째 · {self.action_label}",
                cycle=cycle,
            )
        local -= self.timing.announcement_seconds
        if local < self.timing.countdown_seconds:
            number = self.timing.countdown_seconds - int(local)
            return self._with_guidance(
                decorated,
                phase="countdown",
                title=str(number),
                fallback=str(number),
                detail=f"{cycle}번째 동작 준비",
                cycle=cycle,
            )
        local -= float(self.timing.countdown_seconds)
        if local < self.timing.move_seconds:
            return self._with_guidance(
                decorated,
                phase="move",
                title="지금 움직이세요!",
                fallback="MOVE NOW!",
                detail=f"{cycle}번째 · {self.action_label}",
                cycle=cycle,
            )
        return self._with_guidance(
            decorated,
            phase="neutral",
            title="중립으로 돌아오세요",
            fallback="RETURN TO NEUTRAL",
            detail="다음 안내까지 얼굴을 편하게 유지하세요",
            cycle=cycle,
        )

    def _reset(self) -> None:
        self._stable_since = None
        self._sequence_started_at = None

    @staticmethod
    def _with_guidance(
        status: dict[str, Any],
        *,
        phase: str,
        title: str,
        fallback: str,
        detail: str,
        cycle: int = 0,
    ) -> dict[str, Any]:
        status.update(
            {
                "guidance_phase": phase,
                "guidance_title": title,
                "guidance_fallback": fallback,
                "guidance_detail": detail,
                "guidance_cycle": cycle,
            }
        )
        return status


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


@lru_cache(maxsize=32)
def _guidance_card(
    width: int,
    height: int,
    title: str,
    detail: str,
    fallback: str,
    phase: str,
) -> np.ndarray:
    colors = {
        "calibrating": (0, 178, 230),
        "warming_up": (0, 178, 230),
        "face_lost": (55, 70, 235),
        "settling": (70, 205, 245),
        "announcement": (92, 242, 207),
        "countdown": (92, 242, 207),
        "move": (70, 230, 145),
        "neutral": (70, 205, 245),
        "complete": (180, 195, 210),
    }
    accent = colors.get(phase, (92, 242, 207))
    card = np.full((height, width, 3), (12, 18, 30), dtype=np.uint8)
    cv2.rectangle(card, (1, 1), (width - 2, height - 2), accent, 3)

    try:
        from PIL import Image, ImageDraw

        from ondamm_video_render import load_font

        title_size = min(112 if phase == "countdown" else 58, max(38, int(height * 0.55)))
        detail_size = min(28, max(20, int(height * 0.17)))
        title_font = load_font(size=title_size)
        detail_font = load_font(size=detail_size)
        rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_box[2] - title_box[0]
        title_height = title_box[3] - title_box[1]
        title_y = 18 if phase == "countdown" else 24
        draw.text(
            ((width - title_width) / 2, title_y - title_box[1]),
            title,
            font=title_font,
            fill=(accent[2], accent[1], accent[0]),
        )
        detail_box = draw.textbbox((0, 0), detail, font=detail_font)
        detail_width = detail_box[2] - detail_box[0]
        detail_y = min(height - detail_size - 12, title_y + title_height + 22)
        draw.text(
            ((width - detail_width) / 2, detail_y - detail_box[1]),
            detail,
            font=detail_font,
            fill=(235, 240, 245),
        )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except Exception:
        # The live runtime remains usable on a machine without a Korean font.
        # The large ASCII fallback is deliberately explicit rather than drawing
        # missing-glyph boxes with cv2.putText.
        title_scale = 3.2 if phase == "countdown" else 1.25
        title_thickness = 6 if phase == "countdown" else 3
        (title_width, title_height), _ = cv2.getTextSize(
            fallback,
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            title_thickness,
        )
        cv2.putText(
            card,
            fallback,
            (max(12, (width - title_width) // 2), max(title_height + 16, height // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_scale,
            accent,
            title_thickness,
            cv2.LINE_AA,
        )
        return card


def _render_large_guidance(preview: np.ndarray, status: Mapping[str, Any]) -> None:
    phase = str(status.get("guidance_phase") or "")
    if not phase:
        return
    height, width = preview.shape[:2]
    card_width = min(width - 32, 940)
    card_height = min(190, max(132, int(height * 0.24)))
    title = str(status.get("guidance_title") or "")
    detail = str(status.get("guidance_detail") or "")
    fallback = str(status.get("guidance_fallback") or title)
    card = _guidance_card(card_width, card_height, title, detail, fallback, phase)
    left = (width - card_width) // 2
    top = max(12, height - card_height - 30)
    region = preview[top : top + card_height, left : left + card_width]
    cv2.addWeighted(card, 0.91, region, 0.09, 0, region)


def render_large_guidance(
    frame: np.ndarray,
    *,
    phase: str,
    title: str,
    detail: str,
    fallback: str,
) -> np.ndarray:
    """Render the shared large Korean cue on a visualization frame in place."""
    _render_large_guidance(
        frame,
        {
            "guidance_phase": phase,
            "guidance_title": title,
            "guidance_detail": detail,
            "guidance_fallback": fallback,
        },
    )
    return frame


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
    quality = int(round(100 * np.clip(_number(status.get("quality_score"), 1.0 if face_detected else 0.0), 0.0, 1.0)))
    fps = _number(status.get("fps"))
    _text(
        preview,
        f"FACE {'TRACKED' if face_detected else 'NOT DETECTED'}  |  QUALITY {quality}%  |  FPS {fps:0.1f}",
        28,
        88,
        color=face_color,
        thickness=2,
    )

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
    calibration_ready = bool(status.get("calibration_ready", calibration_remaining <= 0))
    if status.get("face_lost"):
        state_text = "FACE LOST  |  TEMPORAL HISTORY PAUSED"
        state_color = (60, 80, 245)
    elif not calibration_ready:
        calibration = status.get("calibration_status") or {}
        valid = int(calibration.get("valid_samples", 0))
        required = int(calibration.get("minimum_valid_samples", 0))
        coverage = 100 * _number(calibration.get("face_coverage"))
        state_text = f"CALIBRATING  {calibration_remaining:0.1f}s  |  FACE {valid}/{required} {coverage:.0f}%"
        state_color = (0, 220, 255)
    elif status.get("warming_up"):
        state_text = (
            f"WARMING UP  {int(status.get('warmup_frames') or 0)}"
            f"/{int(status.get('warmup_required_frames') or 60)}"
        )
        state_color = (0, 190, 255)
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
    nearest = status.get("nearest_known_pattern") or "none"
    nearest_distance = _number(status.get("nearest_known_distance"), 1.0)
    _text(preview, f"{lifecycle}", 28, 244, color=(220, 225, 235), thickness=2)
    _text(preview, f"{candidate}  |  nearest {nearest}  d={nearest_distance:0.3f}", 28, 267, color=(180, 195, 210))
    _text(preview, f"PATTERN MATCH  {occurrence} / {threshold}", 28, 293, color=(92, 242, 207), scale=0.64, thickness=2)

    _render_large_guidance(preview, status)

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
