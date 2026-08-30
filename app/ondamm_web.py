from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse
from uuid import uuid4

from ondamm_cli import render_handoff_markdown
from ondamm_learning import build_learning_program_plan
from ondamm_gpt import OpenAIFrameReviewer, extract_video_frame_data_urls
from ondamm_models import (
    ConsentGrant,
    Dossier,
    FacialMovementProfile,
    PreSessionRightsCheck,
    SessionSummary,
    unique_preserving_order,
)
from ondamm_pattern_memory import PatternMemoryStore
from ondamm_recommendations import build_baseline_recommendation
from ondamm_review import EventReviewStore, LocalClipCatalog, analyze_clip_with_mediapipe, ensure_browser_compatible_mp4
from ondamm_security import build_export_manifest
from ondamm_sensing import ObservationTally, build_sensing_draft
from ondamm_store import create_dossier, list_dossiers, load_dossier, save_dossier
from ondamm_rights import PURPOSE_LABELS, rights_summary
from ondamm_purge import execute_purge, preview_purge
import ondamm_paths
import ondamm_store


CHILD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class ValidationError(ValueError):
    """Raised when a web request contains invalid user input."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        labels = {
            "child_id": "로컬 아동 ID", "display_name": "표시 이름", "age_band": "연령대",
            "communication_modality": "의사소통 방식", "status": "기록 상태", "reason": "사유",
            "reason_code": "사유 코드", "actor_id": "처리자", "signer_name": "동의 확인자 이름",
            "signature": "확인 서명", "consent_document_id": "동의서 문서 번호", "form_version": "양식 버전",
            "operator_id": "진행 담당자", "confirmation": "삭제 확인 문구", "grant_id": "동의 기록 ID",
            "guardian_cross_checker": "보호자 교차 확인자", "educator_cross_checker": "교육 담당 교차 확인자",
        }
        raise ValidationError(f"{labels.get(key, key)} 항목을 입력해 주세요.")
    return value.strip()


def optional_text(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValidationError(f"{key} must be a string")
    return value.strip()


def text_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValidationError(f"{key} must be a list of strings")
    return unique_preserving_order(value)


def required_confirmation(payload: dict[str, Any], key: str, label: str) -> bool:
    if payload.get(key) is not True:
        raise ValidationError(f"‘{label}’ 항목을 확인해 주세요.")
    return True


def ensure_active(dossier: Dossier, action: str) -> None:
    if dossier.canonical_status != "active":
        raise RuntimeError(f"철회로 잠긴 지원 기록철에서는 ‘{action}’ 작업을 진행할 수 없습니다.")


class OndammWebService:
    """UI-oriented application service over the existing local-first ON DAMM domain."""

    def __init__(
        self,
        *,
        clip_catalog: LocalClipCatalog | None = None,
        event_review_store: EventReviewStore | None = None,
        clip_analyzer: Callable[[Path], dict[str, Any]] = analyze_clip_with_mediapipe,
        gpt_reviewer: Any | None = None,
        frame_extractor: Callable[[Path], list[str]] = extract_video_frame_data_urls,
        pattern_memory_root: Path | None = None,
    ) -> None:
        self.clip_catalog = clip_catalog or LocalClipCatalog(Path(ondamm_paths.ONDAMM_EXPORTS))
        self.event_review_store = event_review_store or EventReviewStore(
            Path(ondamm_paths.ONDAMM_EXPORTS) / "event-reviews"
        )
        self.clip_analyzer = clip_analyzer
        self.frame_extractor = frame_extractor
        self.pattern_memory_root = (
            pattern_memory_root or (Path(ondamm_paths.ONDAMM_EXPORTS) / "pattern-memory")
        ).expanduser().resolve()
        self._browser_media_cache: dict[str, Path] = {}
        if gpt_reviewer is not None:
            self.gpt_reviewer = gpt_reviewer
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            model = os.environ.get("ONDAMM_GPT_MODEL", "gpt-5.6").strip() or "gpt-5.6"
            self.gpt_reviewer = OpenAIFrameReviewer(api_key=api_key, model=model) if api_key else None

    def list_dossiers(self) -> list[dict[str, Any]]:
        return [
            {
                "child_id": dossier.child_id,
                "display_name": dossier.display_name,
                "age_band": dossier.age_band,
                "communication_modality": dossier.communication_modality,
                "canonical_status": dossier.canonical_status,
                "session_count": len(dossier.approved_session_summaries),
                "plan_count": len(dossier.approved_plan_history),
                "updated_at": dossier.updated_at,
            }
            for dossier in list_dossiers()
        ]

    def get_dossier(self, child_id: str) -> dict[str, Any]:
        return load_dossier(self._validate_child_id(child_id)).to_dict()

    def create_dossier(self, payload: dict[str, Any]) -> dict[str, Any]:
        child_id = self._validate_child_id(required_text(payload, "child_id"))
        dossier = Dossier.create(
            child_id=child_id,
            display_name=required_text(payload, "display_name"),
            age_band=required_text(payload, "age_band"),
            communication_modality=required_text(payload, "communication_modality"),
            confirmed_preferences=text_list(payload, "confirmed_preferences"),
            confirmed_avoidances=text_list(payload, "confirmed_avoidances"),
            effective_strategies=text_list(payload, "effective_strategies"),
            triggers_and_calming_supports=text_list(payload, "triggers_and_calming_supports"),
            handoff_notes=text_list(payload, "handoff_notes"),
        )
        create_dossier(dossier)
        return dossier.to_dict()

    def add_session(self, child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "add session summary")
        summary = SessionSummary.create(
            title=required_text(payload, "title"),
            activity_name=required_text(payload, "activity_name"),
            observed_response=required_text(payload, "observed_response"),
            educator_interpretation=required_text(payload, "educator_interpretation"),
            approved_by=required_text(payload, "approved_by"),
            tags=text_list(payload, "tags"),
        )
        dossier.add_session_summary(summary)
        save_dossier(dossier)
        return summary.to_dict()

    def approve_facial_movement_profile(self, child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "approve facial movement profile")
        source_session_ids = text_list(payload, "source_session_ids")
        approved_session_ids = {item.session_id for item in dossier.approved_session_summaries}
        if not source_session_ids or not set(source_session_ids).issubset(approved_session_ids):
            raise ValidationError("source_session_ids must reference approved dossier session summaries")
        raw_threshold = payload.get("activation_threshold")
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, (int, float)):
            raise ValidationError("activation_threshold must be a number")
        raw_priority = payload.get("priority", 80)
        if isinstance(raw_priority, bool) or not isinstance(raw_priority, int):
            raise ValidationError("priority must be an integer")
        try:
            profile = FacialMovementProfile.create(
                label=required_text(payload, "label"),
                display_name=required_text(payload, "display_name"),
                blendshape_names=text_list(payload, "blendshape_names"),
                aggregation=required_text(payload, "aggregation"),
                activation_threshold=float(raw_threshold),
                approved_by=required_text(payload, "approved_by"),
                source_session_ids=source_session_ids,
                priority=raw_priority,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        dossier.add_facial_movement_profile(profile)
        save_dossier(dossier)
        return profile.to_dict()

    def preview_recommendation(self, child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "preview recommendation")
        recommendation = build_baseline_recommendation(
            dossier,
            goal=required_text(payload, "goal"),
            caregiver_input=optional_text(payload, "caregiver_input"),
            drafted_by=optional_text(payload, "drafted_by", "local-operator") or "local-operator",
            approved_by=None,
        )
        return recommendation.to_dict()

    def approve_recommendation(self, child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "approve recommendation")
        recommendation = build_baseline_recommendation(
            dossier,
            goal=required_text(payload, "goal"),
            caregiver_input=optional_text(payload, "caregiver_input"),
            drafted_by=optional_text(payload, "drafted_by", "local-operator") or "local-operator",
            approved_by=required_text(payload, "approved_by"),
        )
        dossier.add_recommendation(recommendation)
        save_dossier(dossier)
        return recommendation.to_dict()

    def preview_learning_plan(self, child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "preview learning plan")
        plan = build_learning_program_plan(
            dossier,
            goal=required_text(payload, "goal"),
            caregiver_input=optional_text(payload, "caregiver_input") or None,
        )
        result = asdict(plan)
        result["total_duration_seconds"] = plan.total_duration_seconds
        result["dossier_auto_updated"] = False
        return result

    def preview_sensing_demo(self, child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "preview sensing draft")
        raw_duration = payload.get("duration_seconds", 8)
        if not isinstance(raw_duration, (int, float)) or isinstance(raw_duration, bool):
            raise ValidationError("duration_seconds must be a number")
        duration = float(raw_duration)
        if not 1 <= duration <= 60:
            raise ValidationError("duration_seconds must be between 1 and 60")
        audio_note = optional_text(payload, "audio_presence_note") or None

        tally = ObservationTally()
        for _ in range(24):
            tally.add_frame(
                face_present=True,
                pose_present=True,
                gaze_zone="center",
                posture_proxy="centered",
                expression_label="neutral",
                facial_movement_labels=[],
                eye_closure_state="open_or_uncertain",
            )
        for _ in range(6):
            tally.add_frame(
                face_present=True,
                pose_present=False,
                gaze_zone="left",
                posture_proxy="unavailable",
                expression_label="mouth_smile",
                facial_movement_labels=["mouth_smile"],
                eye_closure_state="open_or_uncertain",
            )
        return build_sensing_draft(
            child_id=dossier.child_id,
            local_session_id=f"sensing-{uuid4().hex[:8]}",
            duration_seconds=duration,
            tally=tally,
            optional_audio_presence_note=audio_note,
        ).to_dict()

    def list_local_clips(self, child_id: str) -> list[dict[str, Any]]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "list local event clips")
        return self.clip_catalog.list_clips(dossier.child_id)

    def resolve_local_clip_path(self, child_id: str, clip_id: str) -> Path:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "read local event clip")
        return self.clip_catalog.resolve_clip(clip_id, child_id=dossier.child_id).path

    def resolve_media_clip(self, clip_id: str) -> Path:
        clip = self.clip_catalog.resolve_clip(clip_id)
        dossier = load_dossier(self._validate_child_id(clip.child_id))
        ensure_active(dossier, "stream local event clip")
        cached = self._browser_media_cache.get(clip.clip_id)
        if cached and cached.is_file():
            return cached
        browser_path = ensure_browser_compatible_mp4(
            clip.path,
            cache_dir=Path(ondamm_paths.ONDAMM_EXPORTS) / ".web-cache" / dossier.child_id,
        )
        self._browser_media_cache[clip.clip_id] = browser_path
        return browser_path

    def analyze_local_clip(self, child_id: str, clip_id: str) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "analyze local event clip")
        clip = self.clip_catalog.resolve_clip(clip_id, child_id=dossier.child_id)
        result = self.clip_analyzer(clip.path)
        result["clip_id"] = clip.clip_id
        result["child_id"] = dossier.child_id
        result["event_type"] = clip.event_type
        result["dossier_auto_updated"] = False
        return result

    def get_event_reviews(self, child_id: str, clip_id: str) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "read local event reviews")
        clip = self.clip_catalog.resolve_clip(clip_id, child_id=dossier.child_id)
        return self.event_review_store.get_bundle(child_id=dossier.child_id, clip=clip)

    def add_event_review(self, child_id: str, clip_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "add local event review")
        clip = self.clip_catalog.resolve_clip(clip_id, child_id=dossier.child_id)
        try:
            return self.event_review_store.add_review(
                child_id=dossier.child_id,
                clip=clip,
                reviewer_role=required_text(payload, "reviewer_role"),
                reviewer_name=required_text(payload, "reviewer_name"),
                decision=required_text(payload, "decision"),
                observed_facts=required_text(payload, "observed_facts"),
                context_comment=optional_text(payload, "context_comment"),
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def list_temporal_patterns(self, child_id: str) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "read temporal pattern memory")
        try:
            result = self._pattern_memory(dossier.child_id).public_state()
            result["configured"] = True
            return result
        except FileNotFoundError:
            return {
                "schema_version": 1,
                "child_id": dossier.child_id,
                "configured": False,
                "known_patterns": [],
                "candidates": [],
                "suppressed": [],
                "raw_media_saved_for_unpromoted_counts": False,
                "online_tcn_retraining": False,
                "message": "Temporal encoder runtime has not created a local pattern memory yet.",
            }

    def promote_temporal_candidate(
        self,
        child_id: str,
        candidate_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "promote temporal movement candidate")
        store = self._pattern_memory(dossier.child_id)
        clip = self._candidate_clip(dossier.child_id, candidate_id, required_text(payload, "clip_id"))
        review = self.event_review_store.get_bundle(child_id=dossier.child_id, clip=clip)
        if not review["summary"]["ready_for_human_promotion"]:
            raise RuntimeError("three-role consensus_accepted review is required before promotion")
        try:
            pattern = store.promote_candidate(
                candidate_id=candidate_id,
                display_name=required_text(payload, "display_name"),
                approved_by=required_text(payload, "approved_by"),
                source_event_ids=[clip.event_id],
                distance_threshold=payload.get("distance_threshold"),
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        dossier.add_audit_event(
            event_type="temporal_movement_pattern_approved",
            actor_id=pattern["approved_by"],
            details={
                "pattern_id": pattern["pattern_id"],
                "display_name": pattern["display_name"],
                "prototype_digest": pattern["prototype_digest"],
                "encoder_digest": pattern["encoder_digest"],
                "source_event_ids": pattern["source_event_ids"],
                "approved_by": pattern["approved_by"],
                "model_store": "outputs/ondamm/pattern-memory",
            },
        )
        save_dossier(dossier)
        return pattern

    def suppress_temporal_candidate(
        self,
        child_id: str,
        candidate_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "suppress temporal movement candidate")
        clip = self._candidate_clip(dossier.child_id, candidate_id, required_text(payload, "clip_id"))
        review = self.event_review_store.get_bundle(child_id=dossier.child_id, clip=clip)
        if review["summary"]["status"] != "consensus_rejected":
            raise RuntimeError("three-role consensus_rejected review is required before suppression")
        try:
            record = self._pattern_memory(dossier.child_id).suppress_candidate(
                candidate_id=candidate_id,
                approved_by=required_text(payload, "approved_by"),
                reason=required_text(payload, "reason"),
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        dossier.add_audit_event(
            event_type="temporal_movement_pattern_suppressed",
            actor_id=record["approved_by"],
            details={
                "suppression_id": record["suppression_id"],
                "source_candidate_id": candidate_id,
                "prototype_digest": record["prototype_digest"],
            },
        )
        save_dossier(dossier)
        return record

    def watch_temporal_candidate(self, child_id: str, candidate_id: str) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "keep temporal movement candidate under observation")
        return self._pattern_memory(dossier.child_id).mark_watch(candidate_id=candidate_id)

    def review_local_clip_with_gpt(
        self,
        child_id: str,
        clip_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if payload.get("confirm_remote_frame_upload") is not True:
            raise ValidationError("GPT 검토 전에 선택 영상의 추출 프레임 원격 전송에 명시적으로 동의해야 합니다.")
        if self.gpt_reviewer is None:
            raise ValidationError("OPENAI_API_KEY가 설정되지 않아 GPT 검토를 사용할 수 없습니다.")
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "review local event clip with GPT")
        clip = self.clip_catalog.resolve_clip(clip_id, child_id=dossier.child_id)
        frame_data_urls = self.frame_extractor(clip.path)
        result = self.gpt_reviewer.review(
            frame_data_urls=frame_data_urls,
            event_metadata={
                "event_id": clip.event_id,
                "event_type": clip.event_type,
                "start_timestamp": clip.start_timestamp,
                "end_timestamp": clip.end_timestamp,
                "trigger_values": clip.trigger_values,
            },
        )
        result["clip_id"] = clip.clip_id
        result["child_id"] = dossier.child_id
        result["dossier_auto_updated"] = False
        return result

    def integrations_status(self) -> dict[str, Any]:
        return {
            "mediapipe": {
                "configured": True,
                "engine": "google_mediapipe_holistic_blendshapes",
                "local_only": True,
            },
            "openai": {
                "configured": self.gpt_reviewer is not None,
                "model": getattr(self.gpt_reviewer, "model", None),
                "whole_video_uploaded": False,
                "requires_explicit_frame_consent": True,
            },
        }

    def _pattern_memory(self, child_id: str) -> PatternMemoryStore:
        return PatternMemoryStore.open_existing(self.pattern_memory_root, child_id=child_id)

    def _candidate_clip(self, child_id: str, candidate_id: str, clip_id: str):
        clip = self.clip_catalog.resolve_clip(clip_id, child_id=child_id)
        if clip.trigger_values.get("candidate_id") != candidate_id:
            raise ValidationError("clip does not belong to the selected temporal candidate")
        return clip

    def change_status(self, child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        status = required_text(payload, "status")
        if status not in {"active", "withdrawn_locked"}:
            raise ValidationError("지원하지 않는 기록 상태입니다.")
        reason_code = required_text(payload, "reason_code")
        reason = required_text(payload, "reason")
        actor_id = optional_text(payload, "actor_id", "guardian-admin") or "guardian-admin"

        dossier.canonical_status = status
        if status == "withdrawn_locked":
            for grant in dossier.consent_grants:
                grant.revoke(actor_id=actor_id, reason=f"지원 기록철 전체 동의 철회: {reason}")
            dossier.activate_subject_refusal(reason="전체 동의 철회로 모든 촬영·분석 중단")
        dossier.add_audit_event(
            event_type="restoration_approved" if status == "active" else "authoritative_withdrawal",
            actor_id=actor_id,
            details={"reason_code": reason_code, "reason": reason},
        )
        save_dossier(dossier)
        return dossier.to_dict()

    def get_rights(self, child_id: str) -> dict[str, object]:
        dossier = load_dossier(self._validate_child_id(child_id))
        return rights_summary(dossier)

    def grant_consent(self, child_id: str, payload: dict[str, Any]) -> dict[str, object]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "동의 등록")
        purpose = required_text(payload, "purpose")
        try:
            grant = ConsentGrant.create(
                purpose=purpose,
                signer_name=required_text(payload, "signer_name"),
                signature=required_text(payload, "signature"),
                consent_document_id=required_text(payload, "consent_document_id"),
                form_version=required_text(payload, "form_version"),
                guardian_consent_confirmed=required_confirmation(
                    payload, "guardian_consent_confirmed", "보호자 또는 법정대리인 동의"
                ),
                subject_assent_confirmed=required_confirmation(
                    payload, "subject_assent_confirmed", "아동에게 쉬운 설명 후 참여 의사 확인"
                ),
                expires_at=optional_text(payload, "expires_at") or None,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        dossier.add_consent_grant(grant)
        dossier.add_audit_event(
            "consent_granted",
            grant.signer_name,
            {"grant_id": grant.grant_id, "purpose": purpose, "purpose_label": PURPOSE_LABELS[purpose]},
        )
        save_dossier(dossier)
        return rights_summary(dossier)

    def revoke_consent(self, child_id: str, payload: dict[str, Any]) -> dict[str, object]:
        dossier = load_dossier(self._validate_child_id(child_id))
        grant_id = required_text(payload, "grant_id")
        actor = required_text(payload, "actor_id")
        reason = required_text(payload, "reason")
        grant = next((item for item in dossier.consent_grants if item.grant_id == grant_id), None)
        if grant is None:
            raise ValidationError("철회할 동의 기록을 찾을 수 없습니다.")
        grant.revoke(actor_id=actor, reason=reason)
        dossier.add_audit_event("consent_revoked", actor, {"grant_id": grant_id, "reason": reason})
        save_dossier(dossier)
        return rights_summary(dossier)

    def complete_pre_session_check(self, child_id: str, payload: dict[str, Any]) -> dict[str, object]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "교육 전 권리 확인")
        try:
            check = PreSessionRightsCheck.create(
                operator_id=required_text(payload, "operator_id"),
                guardian_cross_checker=required_text(payload, "guardian_cross_checker"),
                educator_cross_checker=required_text(payload, "educator_cross_checker"),
                explanation_confirmed=required_confirmation(payload, "explanation_confirmed", "쉬운 설명"),
                recording_device_recognized=required_confirmation(
                    payload, "recording_device_recognized", "촬영 장치 인지"
                ),
                camera_off_acclimation_completed=required_confirmation(
                    payload, "camera_off_acclimation_completed", "카메라를 끈 적응 시간"
                ),
                stop_control_practiced=required_confirmation(
                    payload, "stop_control_practiced", "아동의 중단 버튼 연습"
                ),
                subject_willing_now=required_confirmation(
                    payload, "subject_willing_now", "현재 참여 의사"
                ),
                valid_minutes=int(payload.get("valid_minutes", 240)),
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(str(exc)) from exc
        dossier.add_pre_session_rights_check(check)
        dossier.add_audit_event(
            "pre_session_rights_check_completed", check.operator_id, {"check_id": check.check_id}
        )
        save_dossier(dossier)
        return rights_summary(dossier)

    def child_stop(self, child_id: str, payload: dict[str, Any]) -> dict[str, object]:
        dossier = load_dossier(self._validate_child_id(child_id))
        reason = optional_text(payload, "reason", "아동이 ‘촬영 싫어요·중단’ 버튼을 누름")
        dossier.activate_subject_refusal(reason=reason)
        dossier.add_audit_event(
            "subject_refusal_activated", "child-stop-control", {"reason": dossier.subject_refusal_reason}
        )
        save_dossier(dossier)
        return rights_summary(dossier)

    def preview_purge(self, child_id: str) -> dict[str, object]:
        return preview_purge(self._validate_child_id(child_id))

    def execute_purge(self, child_id: str, payload: dict[str, Any]) -> dict[str, object]:
        try:
            return execute_purge(
                self._validate_child_id(child_id),
                confirmation=required_text(payload, "confirmation"),
                actor_id=required_text(payload, "actor_id"),
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def export_handoff(self, child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        dossier = load_dossier(self._validate_child_id(child_id))
        ensure_active(dossier, "export handoff")
        actor_id = optional_text(payload, "actor_id", "local-operator") or "local-operator"
        markdown = render_handoff_markdown(dossier)
        issuance_time = utc_now()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        export_dir = Path(ondamm_store.ONDAMM_EXPORTS)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / f"handoff-{dossier.child_id}-{stamp}.md"
        manifest_path = export_dir / f"handoff-{dossier.child_id}-{stamp}.manifest.json"
        export_path.write_text(markdown, encoding="utf-8")
        manifest = build_export_manifest(
            child_id=dossier.child_id,
            issuance_time=issuance_time,
            markdown=markdown,
            export_path=str(export_path),
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        dossier.add_audit_event(
            event_type="signed_export_generated",
            actor_id=actor_id,
            details={
                "output": str(export_path),
                "manifest_output": str(manifest_path),
                "artifact_id": manifest["artifact_id"],
            },
        )
        save_dossier(dossier)
        return {
            "export_path": str(export_path),
            "manifest_path": str(manifest_path),
            "markdown": markdown,
            "manifest": manifest,
        }

    @staticmethod
    def _validate_child_id(child_id: str) -> str:
        if not CHILD_ID_PATTERN.fullmatch(child_id):
            raise ValidationError("로컬 아동 ID에는 영문, 숫자, 하이픈(-), 밑줄(_)만 사용할 수 있습니다.")
        return child_id


class ApiRouter:
    """Small framework-free router kept separate so API behavior is directly testable."""

    def __init__(self, service: OndammWebService) -> None:
        self.service = service

    def dispatch(
        self,
        method: str,
        raw_path: str,
        payload: dict[str, Any] | None,
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]]]:
        path = urlparse(raw_path).path.rstrip("/") or "/"
        body = payload or {}
        if method == "GET" and path == "/api/health":
            return 200, {"status": "ok", "service": "ondamm-local-ui"}
        if method == "GET" and path == "/api/integrations":
            return 200, self.service.integrations_status()
        if path == "/api/dossiers":
            if method == "GET":
                return 200, self.service.list_dossiers()
            if method == "POST":
                return 201, self.service.create_dossier(body)

        segments = [unquote(segment) for segment in path.split("/") if segment]
        if len(segments) >= 3 and segments[:2] == ["api", "dossiers"]:
            child_id = segments[2]
            if len(segments) == 3 and method == "GET":
                return 200, self.service.get_dossier(child_id)
            if len(segments) == 4 and segments[3] == "clips" and method == "GET":
                return 200, self.service.list_local_clips(child_id)
            if len(segments) == 4 and segments[3] == "patterns" and method == "GET":
                return 200, self.service.list_temporal_patterns(child_id)
            if len(segments) == 7 and segments[3] == "patterns" and segments[4] == "candidates" and method == "POST":
                candidate_id = segments[5]
                action = segments[6]
                if action == "promote":
                    return 201, self.service.promote_temporal_candidate(child_id, candidate_id, body)
                if action == "suppress":
                    return 201, self.service.suppress_temporal_candidate(child_id, candidate_id, body)
                if action == "watch":
                    return 200, self.service.watch_temporal_candidate(child_id, candidate_id)
            if len(segments) == 6 and segments[3] == "clips" and segments[5] == "mediapipe" and method == "POST":
                return 200, self.service.analyze_local_clip(child_id, segments[4])
            if len(segments) == 6 and segments[3] == "clips" and segments[5] == "gpt-review" and method == "POST":
                return 200, self.service.review_local_clip_with_gpt(child_id, segments[4], body)
            if len(segments) == 6 and segments[3] == "clips" and segments[5] == "reviews":
                if method == "GET":
                    return 200, self.service.get_event_reviews(child_id, segments[4])
                if method == "POST":
                    return 201, self.service.add_event_review(child_id, segments[4], body)
            if len(segments) == 4 and segments[3] == "sessions" and method == "POST":
                return 201, self.service.add_session(child_id, body)
            if len(segments) == 5 and segments[3:] == ["facial-movement-profiles", "approve"] and method == "POST":
                return 201, self.service.approve_facial_movement_profile(child_id, body)
            if len(segments) == 5 and segments[3:] == ["recommendations", "preview"] and method == "POST":
                return 200, self.service.preview_recommendation(child_id, body)
            if len(segments) == 5 and segments[3:] == ["recommendations", "approve"] and method == "POST":
                return 201, self.service.approve_recommendation(child_id, body)
            if len(segments) == 5 and segments[3:] == ["learning-plan", "preview"] and method == "POST":
                return 200, self.service.preview_learning_plan(child_id, body)
            if len(segments) == 5 and segments[3:] == ["sensing", "demo"] and method == "POST":
                return 200, self.service.preview_sensing_demo(child_id, body)
            if len(segments) == 5 and segments[3:] == ["handoff", "export"] and method == "POST":
                return 201, self.service.export_handoff(child_id, body)
            if len(segments) == 4 and segments[3] == "status" and method == "POST":
                return 200, self.service.change_status(child_id, body)
            if len(segments) == 4 and segments[3] == "rights" and method == "GET":
                return 200, self.service.get_rights(child_id)
            if len(segments) == 5 and segments[3:] == ["rights", "consents"] and method == "POST":
                return 201, self.service.grant_consent(child_id, body)
            if len(segments) == 6 and segments[3:5] == ["rights", "consents"] and segments[5] == "revoke" and method == "POST":
                return 200, self.service.revoke_consent(child_id, body)
            if len(segments) == 5 and segments[3:] == ["rights", "pre-session"] and method == "POST":
                return 201, self.service.complete_pre_session_check(child_id, body)
            if len(segments) == 5 and segments[3:] == ["rights", "child-stop"] and method == "POST":
                return 200, self.service.child_stop(child_id, body)
            if len(segments) == 5 and segments[3:] == ["purge", "preview"] and method == "POST":
                return 200, self.service.preview_purge(child_id)
            if len(segments) == 5 and segments[3:] == ["purge", "execute"] and method == "POST":
                return 200, self.service.execute_purge(child_id, body)

        return 404, {"error": "not_found", "message": "요청한 기능을 찾을 수 없습니다."}


def make_http_handler(router: ApiRouter, ui_dir: Path) -> type[BaseHTTPRequestHandler]:
    static_root = ui_dir.resolve()

    class OndammHttpHandler(BaseHTTPRequestHandler):
        server_version = "ONDAMM/1.0"

        def do_GET(self) -> None:  # noqa: N802
            request_path = urlparse(self.path).path
            if request_path.startswith("/media/clips/"):
                self._serve_clip_media(unquote(request_path.removeprefix("/media/clips/")))
                return
            if request_path.startswith("/api/"):
                self._serve_api("GET")
                return
            self._serve_static()

        def do_POST(self) -> None:  # noqa: N802
            self._serve_api("POST")

        def _serve_api(self, method: str) -> None:
            try:
                payload = self._read_json() if method == "POST" else None
                status, response = router.dispatch(method, self.path, payload)
                self._send_json(status, response)
            except ValidationError as exc:
                self._send_json(400, {"error": "validation_error", "message": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(404, {"error": "not_found", "message": str(exc)})
            except FileExistsError as exc:
                self._send_json(409, {"error": "already_exists", "message": str(exc)})
            except RuntimeError as exc:
                self._send_json(409, {"error": "blocked", "message": str(exc)})
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "invalid_json", "message": "JSON 요청을 읽을 수 없습니다."})
            except Exception as exc:  # pragma: no cover - final HTTP safety net
                self._send_json(500, {"error": "internal_error", "message": str(exc)})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValidationError("request body is too large")
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValidationError("request body must be a JSON object")
            return payload

        def _send_json(self, status: int, payload: object) -> None:
            content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _serve_clip_media(self, clip_id: str) -> None:
            try:
                target = router.service.resolve_media_clip(clip_id)
                if target is None or not target.is_file():
                    raise FileNotFoundError(clip_id)
                size = target.stat().st_size
                start, end = 0, max(0, size - 1)
                status = 200
                range_header = self.headers.get("Range")
                if range_header:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                    if not match:
                        self.send_error(416)
                        return
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = int(last) if last else end
                    elif last:
                        length = int(last)
                        start = max(0, size - length)
                    if start >= size or end < start:
                        self.send_error(416)
                        return
                    end = min(end, size - 1)
                    status = 206
                content_length = end - start + 1
                self.send_response(status)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(content_length))
                self.send_header("Cache-Control", "private, no-store")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                with target.open("rb") as handle:
                    handle.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = handle.read(min(65_536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (FileNotFoundError, RuntimeError):
                self.send_error(404)

        def _serve_static(self) -> None:
            request_path = unquote(urlparse(self.path).path)
            relative = "index.html" if request_path == "/" else request_path.lstrip("/")
            target = (static_root / relative).resolve()
            if static_root not in target.parents and target != static_root:
                self.send_error(403)
                return
            if not target.is_file():
                target = static_root / "index.html"
            if not target.is_file():
                self.send_error(404)
                return
            content = target.read_bytes()
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
                content_type += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            print(f"[ondamm-web] {self.address_string()} {format % args}")

    return OndammHttpHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="ON DAMM local-first web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ui-dir")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    ui_dir = Path(args.ui_dir).expanduser().resolve() if args.ui_dir else project_root / "ui"
    handler = make_http_handler(ApiRouter(OndammWebService()), ui_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"ON DAMM UI: http://{args.host}:{args.port}")
    print("Local-first server. Stop with Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
