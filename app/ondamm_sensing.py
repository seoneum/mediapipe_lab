from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


@dataclass
class ObservationTally:
    frame_count: int = 0
    face_present_frames: int = 0
    pose_present_frames: int = 0
    gaze_zone_counts: Counter[str] = field(default_factory=Counter)
    posture_proxy_counts: Counter[str] = field(default_factory=Counter)

    def add_frame(
        self,
        *,
        face_present: bool,
        pose_present: bool,
        gaze_zone: str,
        posture_proxy: str,
    ) -> None:
        self.frame_count += 1
        if face_present:
            self.face_present_frames += 1
        if pose_present:
            self.pose_present_frames += 1
        self.gaze_zone_counts[gaze_zone] += 1
        self.posture_proxy_counts[posture_proxy] += 1


@dataclass
class SensingDraft:
    child_id: str
    local_session_id: str
    duration_seconds: float
    frame_count: int
    face_present_ratio: float
    pose_present_ratio: float
    gaze_zone_counts: dict[str, int]
    posture_proxy_counts: dict[str, int]
    optional_audio_presence_note: str | None
    reviewed_note_draft: list[str]
    non_authoritative_notice: str
    storage_policy: dict[str, Any]
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "local_session_id": self.local_session_id,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "face_present_ratio": self.face_present_ratio,
            "pose_present_ratio": self.pose_present_ratio,
            "gaze_zone_counts": self.gaze_zone_counts,
            "posture_proxy_counts": self.posture_proxy_counts,
            "optional_audio_presence_note": self.optional_audio_presence_note,
            "reviewed_note_draft": self.reviewed_note_draft,
            "non_authoritative_notice": self.non_authoritative_notice,
            "storage_policy": self.storage_policy,
            "created_at": self.created_at,
        }


def dominant_key(counts: dict[str, int], fallback: str) -> str:
    if not counts:
        return fallback
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_reviewed_note_draft(
    *,
    face_present_ratio: float,
    pose_present_ratio: float,
    gaze_zone_counts: dict[str, int],
    posture_proxy_counts: dict[str, int],
    optional_audio_presence_note: str | None,
) -> list[str]:
    dominant_gaze = dominant_key(gaze_zone_counts, "unknown")
    dominant_posture = dominant_key(posture_proxy_counts, "unknown")
    lines = [
        "이 초안은 자동 확정 정보가 아니라 교사 검토용 보조 메모 초안입니다.",
        f"얼굴 존재 비율은 약 {face_present_ratio:.0%}, 자세 추정 비율은 약 {pose_present_ratio:.0%}로 관찰되었습니다.",
        f"가장 자주 관찰된 시선 구역은 `{dominant_gaze}` 이고, 자세 proxy는 `{dominant_posture}` 경향이었습니다.",
        "이 결과를 집중도/순응도/진단 점수로 해석하지 말고, 다음 활동 설계의 참고 메모로만 사용하세요.",
    ]
    if optional_audio_presence_note:
        lines.append(f"선택적 오디오 관찰 메모: {optional_audio_presence_note.strip()}")
    return lines


def build_sensing_draft(
    *,
    child_id: str,
    local_session_id: str,
    duration_seconds: float,
    tally: ObservationTally,
    optional_audio_presence_note: str | None = None,
) -> SensingDraft:
    face_present_ratio = safe_ratio(tally.face_present_frames, tally.frame_count)
    pose_present_ratio = safe_ratio(tally.pose_present_frames, tally.frame_count)
    gaze_zone_counts = dict(sorted(tally.gaze_zone_counts.items()))
    posture_proxy_counts = dict(sorted(tally.posture_proxy_counts.items()))
    reviewed_note_draft = build_reviewed_note_draft(
        face_present_ratio=face_present_ratio,
        pose_present_ratio=pose_present_ratio,
        gaze_zone_counts=gaze_zone_counts,
        posture_proxy_counts=posture_proxy_counts,
        optional_audio_presence_note=optional_audio_presence_note,
    )
    return SensingDraft(
        child_id=child_id,
        local_session_id=local_session_id,
        duration_seconds=round(duration_seconds, 2),
        frame_count=tally.frame_count,
        face_present_ratio=face_present_ratio,
        pose_present_ratio=pose_present_ratio,
        gaze_zone_counts=gaze_zone_counts,
        posture_proxy_counts=posture_proxy_counts,
        optional_audio_presence_note=optional_audio_presence_note.strip() if optional_audio_presence_note else None,
        reviewed_note_draft=reviewed_note_draft,
        non_authoritative_notice="센서 출력은 non-authoritative draft이며 dossier에 자동 저장되지 않습니다.",
        storage_policy={
            "raw_media_saved": False,
            "auto_writeback_to_dossier": False,
            "intended_use": "reviewed_note_draft_only",
        },
    )
