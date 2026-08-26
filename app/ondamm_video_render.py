"""ON DAMM 영상 분석기 렌더러 (todo 5).

PIL(ImageDraw/ImageFont)로 한글 텍스트를 프레임에 새긴다. cv2.putText는
한글 글리프를 그릴 수 없으므로(ASCII 전용 제약) 절대 사용하지 않는다.

한글 폰트 해석 순서(:func:`resolve_kr_font`):
    1. 환경변수 ``ONDAMM_KR_FONT``
    2. macOS: /System/Library/Fonts/AppleSDGothicNeo.ttc
              /System/Library/Fonts/AppleGothic.ttf   (구버전 macOS만 존재)
              /Library/Fonts/Arial Unicode.ttf
    3. Ubuntu: /usr/share/fonts/truetype/nanum/NanumGothic.ttf
    4. 모두 실패하면 시도한 경로 목록을 담은 :class:`RenderError`.

라벨 계약(README "ON DAMM 영상 분석기" 섹션과 동일):
    ``{global_id} · 집중 {attention_pct}% · 흥미 {interest} · {dominant expression}``
dominant expression 우선순위:
    1) metrics_by_gid[gid].expression_timeline 의 마지막 label
    2) fallback_expr_by_gid[gid]  (analyzer가 최신 샘플의 emotion argmax로 채움 —
       타임라인이 아직 생기지 않은 라이브 구간용)
    3) "-"

모든 프레임 하단 중앙에는 반투명 밴드와 함께 비진단 고지 문구가 새겨진다
(burned-in): ``행동 프록시 추정 결과이며 의학적·교육적 진단이 아닙니다``.

:func:`encode` 는 프레임들을 cv2.VideoWriter(mp4v) 임시 파일에 스트리밍 기록한 뒤
ffmpeg으로 libx264/yuv420p/+faststart로 리먹스한다. ffmpeg 부재(FileNotFoundError)
또는 nonzero 종료 코드는 :class:`RenderError` 로 변환되며, 임시 파일은 finally에서
항상 삭제된다(stale-state 가드).
"""

from __future__ import annotations

import colorsys
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "app"))

CAPTION_TEXT = "행동 프록시 추정 결과이며 의학적·교육적 진단이 아닙니다"
FFMPEG_NOT_FOUND_MSG = (
    "ffmpeg not found — brew install ffmpeg (macOS) / sudo apt install ffmpeg (Ubuntu)"
)

# 실제 파일명 검증 완료(증거 파일 참고): 이 Mac에는 AppleSDGothicNeo.ttc와
# /Library/Fonts/Arial Unicode.ttf가 있다. AppleGothic.ttf는 구버전 macOS 전후보.
_FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/AppleGothic.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
)
FONT_ENV_VAR = "ONDAMM_KR_FONT"
DEFAULT_FONT_SIZE = 18
CAPTION_FONT_SIZE_RATIO = 0.85
BOX_THICKNESS = 2


class RenderError(RuntimeError):
    """렌더 단계 실패(폰트 부재, 인코딩 실패, ffmpeg 부재 등)."""


# --------------------------------------------------------------------- fonts


def font_candidate_paths(env_value: str | None = None) -> list[str]:
    """시도할 폰트 경로 후보(환경변수 값을 맨 앞에 포함)."""
    candidates = []
    env = FONT_ENV_VAR if env_value is None else None
    value = env_value if env_value is not None else os.environ.get(env or "", "")
    if value:
        candidates.append(value)
    candidates.extend(_FONT_CANDIDATES)
    return candidates


def resolve_kr_font(env_value: str | None = None) -> str:
    """한글 글리프가 그려지는 첫 번째 실존 폰트 경로를 반환한다.

    어떤 후보도 실존하지 않으면 시도한 경로 전체를 나열하는
    :class:`RenderError` 를 던진다(행동 가능한 오류 메시지 계약).
    """
    tried: list[str] = []
    for candidate in font_candidate_paths(env_value):
        tried.append(candidate)
        path = Path(candidate).expanduser()
        if path.is_file() and path.stat().st_size > 0:
            return str(path)
    raise RenderError(
        "no Korean-capable font found — tried: " + ", ".join(tried)
        + ". Install one (macOS ships AppleSDGothicNeo; Ubuntu: "
        + "sudo apt install fonts-nanum) or set ONDAMM_KR_FONT to a .ttf/.ttc path"
    )


_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def load_font(font_path: str | None = None, size: int = DEFAULT_FONT_SIZE) -> ImageFont.FreeTypeFont:
    """폰트 로드(해석+캐시). ``font_path``가 None이면 resolve_kr_font()."""
    resolved = font_path if font_path is not None else resolve_kr_font()
    key = (resolved, int(size))
    font = _font_cache.get(key)
    if font is None:
        try:
            font = ImageFont.truetype(resolved, int(size))
        except Exception as exc:  # 손상된 폰트 파일 등
            raise RenderError(f"failed to load font '{resolved}': {exc}") from exc
        _font_cache[key] = font
    return font


# --------------------------------------------------------------------- colors


def gid_color_bgr(global_id: str) -> tuple[int, int, int]:
    """global_id에서 결정론적 고채도 색(BGR)을 만든다(사람별 안정적 박스 색).

    md5 해시 → 황금비 스텝 없이 단순 해시 기반 hue, s=0.95/v=1.0 으로 밝고
    서로 구분되는 색을 보장한다. 동일 id는 항상 동일 색.
    """
    digest = hashlib.md5(global_id.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.95, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def argmax_expression(emotion_labels: Iterable[str], emotion_probs: Iterable[float]) -> str:
    """emotion_probs argmax 라벨(라이브 폴백용 순수 함수)."""
    labels = list(emotion_labels)
    probs = [float(p) for p in emotion_probs]
    if not labels or not probs:
        return "-"
    best = max(range(min(len(labels), len(probs))), key=lambda i: probs[i])
    return str(labels[best])


def dominant_expression_of(metrics: Any, fallback: str | None = None) -> str:
    """타임라인 마지막 label → 폴백(argmax) → "-" 순으로 결정."""
    timeline = getattr(metrics, "expression_timeline", None)
    if timeline is None and isinstance(metrics, Mapping):
        timeline = metrics.get("expression_timeline")
    if timeline:
        last = timeline[-1]
        label = last.get("label") if isinstance(last, Mapping) else getattr(last, "label", None)
        if label:
            return str(label)
    return str(fallback) if fallback else "-"


def label_text(
    global_id: str,
    metrics: Any,
    fallback_expr: str | None = None,
) -> str:
    """README 계약과 동일한 라벨 문자열을 만든다."""
    attention = getattr(metrics, "attention_pct", None)
    if attention is None and isinstance(metrics, Mapping):
        attention = metrics.get("attention_pct")
    interest = getattr(metrics, "interest", None)
    if interest is None and isinstance(metrics, Mapping):
        interest = metrics.get("interest")
    attention_pct = float(attention) if attention is not None else 0.0
    interest_label = str(interest) if interest is not None else "-"
    expr = dominant_expression_of(metrics, fallback_expr)
    return f"{global_id} · 집중 {attention_pct:.0f}% · 흥미 {interest_label} · {expr}"


# --------------------------------------------------------------------- drawing


def _clamp(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


def draw_overlays(
    frame_bgr: np.ndarray,
    observations: Iterable[Any],
    metrics_by_gid: Mapping[str, Any],
    font_path: str | None = None,
    *,
    fallback_expr_by_gid: Mapping[str, str] | None = None,
) -> np.ndarray:
    """BGR 프레임에 사람 박스+한글 라벨과 하단 비진단 자막을 새겨 반환한다.

    원본 프레임은 변경하지 않는다(복사본 반환). observations는
    TrackObservation(또는 bbox_xyxy/global_id 속성을 가진 객체)의 목록.
    """
    if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
        raise RenderError("draw_overlays expects an HxWx3 BGR frame")
    height, width = frame_bgr.shape[:2]
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = load_font(font_path, DEFAULT_FONT_SIZE)
    fallbacks = fallback_expr_by_gid or {}

    for obs in observations:
        gid = str(obs.global_id)
        x1, y1, x2, y2 = (float(v) for v in obs.bbox_xyxy)
        xa, ya = _clamp(x1, 0, width - 1), _clamp(y1, 0, height - 1)
        xb, yb = _clamp(x2, 0, width - 1), _clamp(y2, 0, height - 1)
        if xb <= xa or yb <= ya:
            continue
        color_bgr = gid_color_bgr(gid)
        box_rgb = (color_bgr[2], color_bgr[1], color_bgr[0], 255)
        draw.rectangle((xa, ya, xb, yb), outline=box_rgb, width=BOX_THICKNESS)

        metrics = metrics_by_gid.get(gid)
        text = label_text(gid, metrics, fallbacks.get(gid))
        ty = ya - (DEFAULT_FONT_SIZE + 6)
        if ty < 0:
            ty = min(yb + 4, height - DEFAULT_FONT_SIZE - 4)
        tx = max(0, xa)
        _draw_text_with_background(draw, (tx, ty), text, font, box_rgb)

    # 하단 중앙 반투명 자막(burned-in non-diagnostic caption)
    caption_font = load_font(font_path, max(12, int(DEFAULT_FONT_SIZE * CAPTION_FONT_SIZE_RATIO)))
    _draw_caption(draw, pil_image.width, pil_image.height, CAPTION_TEXT, caption_font)

    annotated = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    return annotated


def _draw_text_with_background(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    accent_rgb: tuple[int, int, int, int],
) -> None:
    left, top = xy
    try:
        bbox = draw.textbbox((left, top), text, font=font)
    except ValueError:  # 빈 문자열 등
        return
    pad = 2
    background = (0, 0, 0, 160)
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=background,
    )
    draw.text((left, top), text, font=font, fill=(255, 255, 255, 255))
    # 라벨 좌측에 사람 박스와 같은 색 마커로 귀속 관계를 강화한다.
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[0] + 2, bbox[3] + pad),
        fill=accent_rgb,
    )


def _draw_caption(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except ValueError:
        return
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    band_h = text_h + 10
    top = height - band_h - 4
    draw.rectangle((0, top, width, height), fill=(0, 0, 0, 120))
    tx = max(0, (width - text_w) // 2)
    ty = top + (band_h - text_h) // 2 - bbox[1]
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))


# --------------------------------------------------------------------- encode


def ffmpeg_on_path() -> str | None:
    """PATH 위 ffmpeg 탐색(테스트에서 monkeypatch 대상)."""
    return shutil.which("ffmpeg")


def encode(frames: Iterable[np.ndarray], out_path: str | Path, fps: float) -> Path:
    """프레임 스트림을 MP4(libx264, yuv420p, +faststart)로 인코딩한다.

    프레임들은 먼저 임시 mp4v 컨테이너에 스트리밍 기록된 뒤 ffmpeg으로
    리먹스된다. 임시 파일은 성공/실패 무관하게 finally에서 삭제된다.
    ffmpeg 부재 또는 nonzero 종료 코드는 :class:`RenderError`.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps_safe = float(fps) if fps and float(fps) > 0 else 30.0

    fd, tmp_name = tempfile.mkstemp(prefix=".ondamm_render_", suffix=".mp4", dir=str(out.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    writer: cv2.VideoWriter | None = None
    try:
        for frame in frames:
            if frame is None or frame.ndim != 3:
                continue
            if writer is None:
                height, width = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(tmp), fourcc, fps_safe, (width, height))
                if not writer.isOpened():
                    raise RenderError(f"cv2.VideoWriter failed to open temp file {tmp}")
            writer.write(frame)
        if writer is None:
            raise RenderError("no frames to encode")
        writer.release()
        writer = None

        if ffmpeg_on_path() is None:
            raise RenderError(FFMPEG_NOT_FOUND_MSG)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RenderError(FFMPEG_NOT_FOUND_MSG) from exc
        if proc.returncode != 0 or not out.is_file():
            detail = (proc.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
            try:
                if out.exists():
                    out.unlink()  # ffmpeg 실패 시 부분 출력 잔존 방지(stale-state 가드)
            except OSError:
                pass
            raise RenderError(
                f"ffmpeg remux failed (rc={proc.returncode}): {detail[0]}"
            )
        return out
    finally:
        if writer is not None:
            writer.release()
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
