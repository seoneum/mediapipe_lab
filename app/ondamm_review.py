from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


REVIEWER_ROLES = ("guardian", "teacher", "institutional_social_worker")
REVIEW_DECISIONS = ("accepted", "rejected", "uncertain")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 80 or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in cleaned):
        raise ValueError(f"{field_name} must contain only letters, numbers, '-' or '_'")
    return cleaned


class EventReviewStore:
    """Append-only, local review records kept separate from the canonical dossier."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self._lock = threading.Lock()

    def get_bundle(self, *, child_id: str, clip: "LocalClip") -> dict[str, Any]:
        path = self._path(child_id, clip.clip_id)
        with self._lock:
            payload = self._read(path) if path.is_file() else self._empty_bundle(child_id, clip)
        return self._with_summary(payload)

    def add_review(
        self,
        *,
        child_id: str,
        clip: "LocalClip",
        reviewer_role: str,
        reviewer_name: str,
        decision: str,
        observed_facts: str,
        context_comment: str = "",
    ) -> dict[str, Any]:
        role = reviewer_role.strip()
        if role not in REVIEWER_ROLES:
            raise ValueError(f"reviewer_role must be one of: {', '.join(REVIEWER_ROLES)}")
        normalized_decision = decision.strip()
        if normalized_decision not in REVIEW_DECISIONS:
            raise ValueError(f"decision must be one of: {', '.join(REVIEW_DECISIONS)}")
        name = self._bounded_text(reviewer_name, field_name="reviewer_name", maximum=120)
        facts = self._bounded_text(observed_facts, field_name="observed_facts", maximum=2_000)
        comment = self._bounded_text(context_comment, field_name="context_comment", maximum=4_000, required=False)
        path = self._path(child_id, clip.clip_id)

        with self._lock:
            payload = self._read(path) if path.is_file() else self._empty_bundle(child_id, clip)
            if payload.get("event_id") != clip.event_id:
                raise RuntimeError("review bundle event identity does not match the selected clip")
            entries = payload.setdefault("reviews", [])
            if len(entries) >= 200:
                raise RuntimeError("review history limit reached for this event clip")
            previous = next((item for item in reversed(entries) if item.get("reviewer_role") == role), None)
            entries.append(
                {
                    "review_id": f"review-{uuid4().hex[:12]}",
                    "reviewer_role": role,
                    "reviewer_name": name,
                    "decision": normalized_decision,
                    "observed_facts": facts,
                    "context_comment": comment,
                    "supersedes_review_id": previous.get("review_id") if previous else None,
                    "created_at": _utc_now(),
                }
            )
            payload["revision"] = int(payload.get("revision", 0)) + 1
            payload["updated_at"] = _utc_now()
            self._write(path, payload)
        return self._with_summary(payload)

    def _path(self, child_id: str, clip_id: str) -> Path:
        safe_child_id = _safe_id(child_id, field_name="child_id")
        safe_clip_id = _safe_id(clip_id, field_name="clip_id")
        return self.root / safe_child_id / f"{safe_clip_id}.json"

    @staticmethod
    def _bounded_text(value: str, *, field_name: str, maximum: int, required: bool = True) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        cleaned = value.strip()
        if required and not cleaned:
            raise ValueError(f"{field_name} is required")
        if len(cleaned) > maximum:
            raise ValueError(f"{field_name} must be at most {maximum} characters")
        return cleaned

    @staticmethod
    def _empty_bundle(child_id: str, clip: "LocalClip") -> dict[str, Any]:
        return {
            "schema_version": 1,
            "child_id": child_id,
            "clip_id": clip.clip_id,
            "event_id": clip.event_id,
            "event_type": clip.event_type,
            "revision": 0,
            "reviews": [],
            "updated_at": None,
            "dossier_auto_updated": False,
        }

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"could not read event review bundle: {path.name}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("reviews"), list):
            raise RuntimeError(f"invalid event review bundle: {path.name}")
        return payload

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _with_summary(payload: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(payload, ensure_ascii=False))
        latest_by_role: dict[str, dict[str, Any]] = {}
        for entry in result.get("reviews", []):
            role = entry.get("reviewer_role")
            if role in REVIEWER_ROLES:
                latest_by_role[role] = entry
        pending_roles = [role for role in REVIEWER_ROLES if role not in latest_by_role]
        decisions = [entry.get("decision") for entry in latest_by_role.values()]
        if not latest_by_role:
            status = "pending"
        elif pending_roles:
            status = "partially_reviewed"
        elif "uncertain" in decisions:
            status = "needs_context"
        elif len(set(decisions)) == 1:
            status = f"consensus_{decisions[0]}"
        else:
            status = "disagreement"
        result["summary"] = {
            "status": status,
            "required_roles": list(REVIEWER_ROLES),
            "completed_roles": [role for role in REVIEWER_ROLES if role in latest_by_role],
            "pending_roles": pending_roles,
            "latest_by_role": latest_by_role,
            "ready_for_human_promotion": status == "consensus_accepted",
            "promotion_requires_separate_approval": True,
        }
        return result


@dataclass(frozen=True)
class LocalClip:
    clip_id: str
    child_id: str
    event_id: str
    event_type: str
    start_timestamp: float
    end_timestamp: float
    trigger_values: dict[str, Any]
    created_at: str
    mode: str
    relative_path: str
    path: Path

    @property
    def duration_seconds(self) -> float:
        return round(max(0.0, self.end_timestamp - self.start_timestamp), 3)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "child_id": self.child_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
            "trigger_values": self.trigger_values,
            "created_at": self.created_at,
            "mode": self.mode,
            "relative_path": self.relative_path,
            "file_size_bytes": self.path.stat().st_size,
            "media_url": f"/media/clips/{self.clip_id}",
        }


class LocalClipCatalog:
    """Indexes event clips that are both metadata-backed and inside the local output root."""

    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = outputs_root.expanduser().resolve()

    def list_clips(self, child_id: str | None = None) -> list[dict[str, Any]]:
        clips = [clip for clip in self._scan() if child_id is None or clip.child_id == child_id]
        clips.sort(key=lambda clip: (clip.created_at, clip.event_id), reverse=True)
        return [clip.to_public_dict() for clip in clips]

    def resolve_clip(self, clip_id: str, *, child_id: str | None = None) -> LocalClip:
        for clip in self._scan():
            if clip.clip_id == clip_id and (child_id is None or clip.child_id == child_id):
                return clip
        raise FileNotFoundError(f"Missing local event clip: {clip_id}")

    def _scan(self) -> list[LocalClip]:
        if not self.outputs_root.exists():
            return []
        clips: list[LocalClip] = []
        for metadata_path in sorted(self.outputs_root.glob("**/event_recording.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            child_id = str(metadata.get("child_id", "")).strip()
            if not child_id:
                continue
            mode = str(metadata.get("mode", "unknown"))
            for event in metadata.get("events", []):
                clip = self._clip_from_event(metadata_path, child_id, mode, event)
                if clip:
                    clips.append(clip)
        return clips

    def _clip_from_event(
        self,
        metadata_path: Path,
        child_id: str,
        mode: str,
        event: object,
    ) -> LocalClip | None:
        if not isinstance(event, dict):
            return None
        raw_path = event.get("clip_path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = metadata_path.parent / candidate
        candidate = candidate.resolve()
        if self.outputs_root not in candidate.parents or not candidate.is_file() or candidate.suffix.lower() != ".mp4":
            return None
        relative_path = candidate.relative_to(self.outputs_root).as_posix()
        event_id = str(event.get("event_id", candidate.stem))
        clip_id = hashlib.sha256(f"{child_id}\n{event_id}\n{relative_path}".encode("utf-8")).hexdigest()[:20]
        try:
            start = float(event.get("start_timestamp", 0.0))
            end = float(event.get("end_timestamp", start))
        except (TypeError, ValueError):
            return None
        trigger_values = event.get("trigger_values", {})
        if not isinstance(trigger_values, dict):
            trigger_values = {}
        return LocalClip(
            clip_id=clip_id,
            child_id=child_id,
            event_id=event_id,
            event_type=str(event.get("event_type", "unknown")),
            start_timestamp=start,
            end_timestamp=end,
            trigger_values=trigger_values,
            created_at=str(event.get("created_at", "")),
            mode=mode,
            relative_path=relative_path,
            path=candidate,
        )


_TRANSCODE_LOCK = threading.Lock()


def probe_video_codec(path: Path) -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required to inspect local event video codecs")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    codec = completed.stdout.strip().lower()
    if completed.returncode != 0 or not codec:
        raise RuntimeError(f"Could not inspect local event clip codec: {completed.stderr.strip()[:300]}")
    return codec


def transcode_to_h264(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to make this local event clip browser-compatible")
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Could not transcode local event clip for browser playback: {completed.stderr.strip()[:500]}")


def ensure_browser_compatible_mp4(
    source: Path,
    *,
    cache_dir: Path,
    probe_codec: Callable[[Path], str] = probe_video_codec,
    transcoder: Callable[[Path, Path], None] = transcode_to_h264,
) -> Path:
    source = source.resolve()
    if probe_codec(source).lower() == "h264":
        return source
    stat = source.stat()
    cache_key = hashlib.sha256(
        f"{source}\n{stat.st_size}\n{stat.st_mtime_ns}\nh264-yuv420p-crf24".encode("utf-8")
    ).hexdigest()[:24]
    cache_root = cache_dir.expanduser().resolve()
    destination = cache_root / f"{cache_key}.mp4"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    with _TRANSCODE_LOCK:
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = cache_root / f".{cache_key}.transcoding.mp4"
        temporary.unlink(missing_ok=True)
        transcoder(source, temporary)
        temporary.replace(destination)
    return destination


def analyze_video_frames(
    path: Path,
    *,
    analyze_frame: Callable[[Any], str | None],
    max_samples: int = 24,
) -> dict[str, Any]:
    import cv2

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open local event clip: {path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        if total_frames <= 0:
            raise RuntimeError(f"Local event clip has no readable frames: {path}")
        sample_count = min(max_samples, total_frames)
        if sample_count == 1:
            indices = [0]
        else:
            indices = sorted({round(index * (total_frames - 1) / (sample_count - 1)) for index in range(sample_count)})
        counts: Counter[str] = Counter()
        unavailable = 0
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                unavailable += 1
                continue
            label = analyze_frame(frame)
            if label:
                counts[label] += 1
            else:
                unavailable += 1
        sampled = len(indices)
        dominant = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if counts else None
        return {
            "sampled_frame_count": sampled,
            "analyzed_frame_count": sum(counts.values()),
            "unavailable_frame_count": unavailable,
            "video_frame_count": total_frames,
            "fps": round(fps, 3),
            "expression_label_counts": dict(sorted(counts.items())),
            "dominant_expression_hint": dominant,
            "non_diagnostic_notice": (
                "MediaPipe 얼굴 blendshape 기반 미세 움직임 힌트이며 감정 상태, 집중도, 진단으로 해석하지 않습니다."
            ),
            "dossier_auto_updated": False,
        }
    finally:
        capture.release()


class MediaPipeExpressionAnalyzer:
    """Local-only face blendshape analyzer. Labels describe movement patterns, not emotions."""

    def __init__(self) -> None:
        self._detector = None
        self._mp = None

    def __enter__(self) -> "MediaPipeExpressionAnalyzer":
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import holistic_landmarker

        from paths import HOLISTIC_MODEL, base_options

        options = holistic_landmarker.HolisticLandmarkerOptions(
            base_options=base_options(HOLISTIC_MODEL),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self._mp = mp
        self._detector = holistic_landmarker.HolisticLandmarker.create_from_options(options)
        return self

    def analyze(self, frame: Any) -> str | None:
        import cv2

        if self._detector is None or self._mp is None:
            raise RuntimeError("MediaPipe analyzer is not open")
        from holistic_camera import estimate_expression

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        if not result.face_blendshapes:
            return None
        label, _ = estimate_expression(result.face_blendshapes, 3)
        return label

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._detector is not None:
            self._detector.close()
        self._detector = None
        self._mp = None
        return False


def analyze_clip_with_mediapipe(
    path: Path,
    *,
    max_samples: int = 24,
    analyzer_factory: Callable[[], Any] = MediaPipeExpressionAnalyzer,
) -> dict[str, Any]:
    with analyzer_factory() as analyzer:
        result = analyze_video_frames(path, analyze_frame=analyzer.analyze, max_samples=max_samples)
    result["analysis_engine"] = "google_mediapipe_holistic_blendshapes"
    return result
