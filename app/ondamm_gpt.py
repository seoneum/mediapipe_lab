from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def extract_video_frame_data_urls(
    path: Path,
    *,
    max_frames: int = 3,
    max_dimension: int = 768,
    jpeg_quality: int = 82,
) -> list[str]:
    """Extract a small, bounded set of still frames; the whole video is never uploaded."""
    import cv2

    if not 1 <= max_frames <= 8:
        raise ValueError("max_frames must be between 1 and 8")
    if not 64 <= max_dimension <= 1600:
        raise ValueError("max_dimension must be between 64 and 1600")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open local event clip: {path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Local event clip has no readable frames: {path}")
        count = min(max_frames, total_frames)
        indices = [0] if count == 1 else [round(index * (total_frames - 1) / (count - 1)) for index in range(count)]
        data_urls: list[str] = []
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, max_dimension / max(height, width))
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            encoded_ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not encoded_ok:
                continue
            data = base64.b64encode(encoded.tobytes()).decode("ascii")
            data_urls.append(f"data:image/jpeg;base64,{data}")
        if not data_urls:
            raise RuntimeError(f"Could not extract review frames from: {path}")
        return data_urls
    finally:
        capture.release()


def _default_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed ({exc.code}): {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API connection failed: {exc.reason}") from exc


def _output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    parts: list[str] = []
    for output in response.get("output", []):
        if not isinstance(output, dict):
            continue
        for content in output.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if content.get("type") in {"output_text", "text"} and isinstance(text, str):
                parts.append(text.strip())
    result = "\n".join(part for part in parts if part)
    if not result:
        raise RuntimeError("OpenAI response did not contain review text")
    return result


class OpenAIFrameReviewer:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.6",
        transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] = _default_transport,
        timeout_seconds: float = 90.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is required")
        self.api_key = api_key.strip()
        self.model = model.strip() or "gpt-5.6"
        self.transport = transport
        self.timeout_seconds = timeout_seconds

    def review(
        self,
        *,
        frame_data_urls: list[str],
        event_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not frame_data_urls:
            raise ValueError("At least one review frame is required")
        prompt = (
            "당신은 발달장애 아동 지원 기록을 돕는 관찰 보조자입니다. "
            "제공된 이미지는 하나의 로컬 특이 이벤트 영상에서 시간 순서대로 추출한 정지 프레임입니다. "
            "화면에서 직접 확인 가능한 얼굴·시선 방향·자세·움직임 변화만 한국어로 기술하세요. "
            "감정, 의도, 집중도, 순응도, 선호, 진단을 추론하거나 확정하지 마세요. "
            "아동을 평가하거나 점수화하지 말고, 불확실하면 불확실하다고 쓰세요. "
            "출력은 1) 관찰 가능한 변화 2) 확인 불가/한계 3) 교사가 다시 볼 지점의 세 구역으로 간결하게 작성하세요.\n\n"
            f"이벤트 메타데이터: {json.dumps(event_metadata, ensure_ascii=False, sort_keys=True)}"
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(
            {"type": "input_image", "image_url": frame, "detail": "low"}
            for frame in frame_data_urls
        )
        payload = {
            "model": self.model,
            "store": False,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 700,
        }
        response = self.transport(
            OPENAI_RESPONSES_URL,
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            payload,
            self.timeout_seconds,
        )
        return {
            "response_id": response.get("id"),
            "model": self.model,
            "review_text": _output_text(response),
            "remote_frame_count": len(frame_data_urls),
            "whole_video_uploaded": False,
            "dossier_auto_updated": False,
            "non_authoritative_notice": (
                "GPT 결과는 원격 프레임 검토 초안이며 감정·진단·집중도 판정이나 공식 기록 자동 반영에 사용하지 않습니다."
            ),
        }
