from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 2
CONSENT_PURPOSES = (
    "camera_capture",
    "research_metrics",
    "model_training",
    "remote_review",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


@dataclass
class SessionSummary:
    session_id: str
    title: str
    activity_name: str
    observed_response: str
    educator_interpretation: str
    approved_by: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        title: str,
        activity_name: str,
        observed_response: str,
        educator_interpretation: str,
        approved_by: str,
        tags: list[str] | None = None,
    ) -> "SessionSummary":
        return cls(
            session_id=f"session-{uuid4().hex[:8]}",
            title=title.strip(),
            activity_name=activity_name.strip(),
            observed_response=observed_response.strip(),
            educator_interpretation=educator_interpretation.strip(),
            approved_by=approved_by.strip(),
            tags=unique_preserving_order(tags or []),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionSummary":
        return cls(
            session_id=data["session_id"],
            title=data["title"],
            activity_name=data["activity_name"],
            observed_response=data["observed_response"],
            educator_interpretation=data["educator_interpretation"],
            approved_by=data["approved_by"],
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", utc_now()),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = unique_preserving_order(self.tags)
        return payload


@dataclass
class RecommendationEntry:
    recommendation_id: str
    goal: str
    summary: str
    suggested_activities: list[str]
    rationale_lines: list[str]
    drafted_by: str
    approved_by: str | None = None
    status: str = "draft"
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        goal: str,
        summary: str,
        suggested_activities: list[str],
        rationale_lines: list[str],
        drafted_by: str,
        approved_by: str | None = None,
    ) -> "RecommendationEntry":
        return cls(
            recommendation_id=f"plan-{uuid4().hex[:8]}",
            goal=goal.strip(),
            summary=summary.strip(),
            suggested_activities=unique_preserving_order(suggested_activities),
            rationale_lines=unique_preserving_order(rationale_lines),
            drafted_by=drafted_by.strip(),
            approved_by=approved_by.strip() if approved_by else None,
            status="approved" if approved_by else "draft",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecommendationEntry":
        return cls(
            recommendation_id=data["recommendation_id"],
            goal=data["goal"],
            summary=data["summary"],
            suggested_activities=list(data.get("suggested_activities", [])),
            rationale_lines=list(data.get("rationale_lines", [])),
            drafted_by=data["drafted_by"],
            approved_by=data.get("approved_by"),
            status=data.get("status", "draft"),
            created_at=data.get("created_at", utc_now()),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["suggested_activities"] = unique_preserving_order(self.suggested_activities)
        payload["rationale_lines"] = unique_preserving_order(self.rationale_lines)
        return payload


@dataclass
class FacialMovementProfile:
    profile_id: str
    label: str
    display_name: str
    blendshape_names: list[str]
    aggregation: str
    activation_threshold: float
    approved_by: str
    source_session_ids: list[str]
    priority: int = 80
    status: str = "approved"
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        label: str,
        display_name: str,
        blendshape_names: list[str],
        aggregation: str,
        activation_threshold: float,
        approved_by: str,
        source_session_ids: list[str],
        priority: int = 80,
    ) -> "FacialMovementProfile":
        # Keep MediaPipe outside the dossier domain; this only validates names
        # and thresholds used by the pure movement analyzer.
        from ondamm_facial_movement import MovementRule

        rule = MovementRule(
            label=label,
            display_name=display_name,
            blendshape_names=tuple(blendshape_names),
            aggregation=aggregation,
            activation_threshold=activation_threshold,
            priority=priority,
        )
        approver = approved_by.strip() if isinstance(approved_by, str) else ""
        source_ids = unique_preserving_order(source_session_ids)
        if not approver or not source_ids:
            raise ValueError("facial movement profile requires explicit approval and source sessions")
        return cls(
            profile_id=f"facial-profile-{uuid4().hex[:10]}",
            label=rule.label,
            display_name=rule.display_name,
            blendshape_names=list(rule.blendshape_names),
            aggregation=rule.aggregation,
            activation_threshold=rule.activation_threshold,
            approved_by=approver,
            source_session_ids=source_ids,
            priority=rule.priority,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FacialMovementProfile":
        profile = cls(
            profile_id=data["profile_id"],
            label=data["label"],
            display_name=data["display_name"],
            blendshape_names=list(data["blendshape_names"]),
            aggregation=data["aggregation"],
            activation_threshold=float(data["activation_threshold"]),
            approved_by=data["approved_by"],
            source_session_ids=list(data["source_session_ids"]),
            priority=int(data.get("priority", 80)),
            status=data.get("status", "approved"),
            created_at=data.get("created_at", utc_now()),
        )
        from ondamm_facial_movement import rules_from_approved_profiles

        rules_from_approved_profiles([profile.to_dict()])
        return profile

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConsentGrant:
    grant_id: str
    purpose: str
    signer_name: str
    signature_digest: str
    consent_document_id: str
    form_version: str
    guardian_consent_confirmed: bool
    subject_assent: str
    granted_at: str = field(default_factory=utc_now)
    expires_at: str | None = None
    revoked_at: str | None = None
    revoked_by: str | None = None
    revocation_reason: str | None = None
    signature_method: str = "typed_attestation_digest"

    @classmethod
    def create(
        cls,
        *,
        purpose: str,
        signer_name: str,
        signature: str,
        consent_document_id: str,
        form_version: str,
        guardian_consent_confirmed: bool,
        subject_assent_confirmed: bool,
        expires_at: str | None = None,
    ) -> "ConsentGrant":
        normalized_purpose = purpose.strip() if isinstance(purpose, str) else ""
        if normalized_purpose not in CONSENT_PURPOSES:
            raise ValueError(f"지원하지 않는 동의 목적입니다: {normalized_purpose or '비어 있음'}")
        cleaned_signer = signer_name.strip() if isinstance(signer_name, str) else ""
        cleaned_signature = signature.strip() if isinstance(signature, str) else ""
        cleaned_document_id = consent_document_id.strip() if isinstance(consent_document_id, str) else ""
        cleaned_form_version = form_version.strip() if isinstance(form_version, str) else ""
        if not cleaned_signer:
            raise ValueError("동의 확인자의 이름을 입력해 주세요.")
        if not cleaned_signature:
            raise ValueError("동의 확인 서명을 입력해 주세요.")
        if not cleaned_document_id or not cleaned_form_version:
            raise ValueError("동의서 문서 번호와 양식 버전을 입력해 주세요.")
        if guardian_consent_confirmed is not True:
            raise ValueError("보호자 또는 법정대리인의 동의 확인이 필요합니다.")
        if subject_assent_confirmed is not True:
            raise ValueError("아동에게 쉬운 방식으로 설명하고 본인의 참여 의사를 확인해 주세요.")
        if expires_at:
            try:
                datetime.fromisoformat(expires_at)
            except ValueError as exc:
                raise ValueError("동의 만료 시각은 ISO 형식이어야 합니다.") from exc
        return cls(
            grant_id=f"consent-{uuid4().hex[:12]}",
            purpose=normalized_purpose,
            signer_name=cleaned_signer,
            signature_digest=hashlib.sha256(cleaned_signature.encode("utf-8")).hexdigest(),
            consent_document_id=cleaned_document_id,
            form_version=cleaned_form_version,
            guardian_consent_confirmed=True,
            subject_assent="affirmed",
            expires_at=expires_at,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConsentGrant":
        return cls(
            grant_id=data["grant_id"],
            purpose=data["purpose"],
            signer_name=data["signer_name"],
            signature_digest=data["signature_digest"],
            consent_document_id=data["consent_document_id"],
            form_version=data.get("form_version", "unknown"),
            guardian_consent_confirmed=bool(data.get("guardian_consent_confirmed", False)),
            subject_assent=data.get("subject_assent", "not_confirmed"),
            granted_at=data.get("granted_at", utc_now()),
            expires_at=data.get("expires_at"),
            revoked_at=data.get("revoked_at"),
            revoked_by=data.get("revoked_by"),
            revocation_reason=data.get("revocation_reason"),
            signature_method=data.get("signature_method", "typed_attestation_digest"),
        )

    def revoke(self, *, actor_id: str, reason: str) -> None:
        if self.revoked_at is None:
            self.revoked_at = utc_now()
            self.revoked_by = actor_id.strip()
            self.revocation_reason = reason.strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreSessionRightsCheck:
    check_id: str
    operator_id: str
    explanation_confirmed: bool
    recording_device_recognized: bool
    camera_off_acclimation_completed: bool
    stop_control_practiced: bool
    subject_willing_now: bool
    created_at: str
    expires_at: str
    guardian_cross_checker: str = ""
    educator_cross_checker: str = ""

    @classmethod
    def create(
        cls,
        *,
        operator_id: str,
        guardian_cross_checker: str,
        educator_cross_checker: str,
        explanation_confirmed: bool,
        recording_device_recognized: bool,
        camera_off_acclimation_completed: bool,
        stop_control_practiced: bool,
        subject_willing_now: bool,
        valid_minutes: int = 240,
    ) -> "PreSessionRightsCheck":
        cleaned_operator = operator_id.strip() if isinstance(operator_id, str) else ""
        if not cleaned_operator:
            raise ValueError("교육 전 확인을 진행한 담당자를 입력해 주세요.")
        guardian_name = guardian_cross_checker.strip() if isinstance(guardian_cross_checker, str) else ""
        educator_name = educator_cross_checker.strip() if isinstance(educator_cross_checker, str) else ""
        if not guardian_name or not educator_name:
            raise ValueError("보호자와 교육 담당자의 교차 확인 이름을 모두 입력해 주세요.")
        if guardian_name == educator_name:
            raise ValueError("교차 확인은 서로 다른 두 사람이 각각 확인해야 합니다.")
        checks = {
            "촬영 목적을 쉬운 방식으로 설명했는지": explanation_confirmed,
            "카메라 또는 녹화 인형이 촬영 장치임을 확인했는지": recording_device_recognized,
            "카메라를 끈 상태에서 적응 시간을 가졌는지": camera_off_acclimation_completed,
            "아동이 중단 버튼을 직접 눌러 연습했는지": stop_control_practiced,
            "아동이 지금 참여하겠다는 의사를 보였는지": subject_willing_now,
        }
        missing = [label for label, confirmed in checks.items() if confirmed is not True]
        if missing:
            raise ValueError("교육 전 확인이 완료되지 않았습니다: " + ", ".join(missing))
        if not 5 <= int(valid_minutes) <= 480:
            raise ValueError("교육 전 확인 유효시간은 5분에서 480분 사이여야 합니다.")
        created = datetime.now(timezone.utc)
        return cls(
            check_id=f"rights-check-{uuid4().hex[:12]}",
            operator_id=cleaned_operator,
            explanation_confirmed=True,
            recording_device_recognized=True,
            camera_off_acclimation_completed=True,
            stop_control_practiced=True,
            subject_willing_now=True,
            created_at=created.isoformat(),
            expires_at=(created + timedelta(minutes=int(valid_minutes))).isoformat(),
            guardian_cross_checker=guardian_name,
            educator_cross_checker=educator_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreSessionRightsCheck":
        return cls(
            check_id=data["check_id"],
            operator_id=data["operator_id"],
            explanation_confirmed=bool(data.get("explanation_confirmed", False)),
            recording_device_recognized=bool(data.get("recording_device_recognized", False)),
            camera_off_acclimation_completed=bool(data.get("camera_off_acclimation_completed", False)),
            stop_control_practiced=bool(data.get("stop_control_practiced", False)),
            subject_willing_now=bool(data.get("subject_willing_now", False)),
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            guardian_cross_checker=data.get("guardian_cross_checker", data.get("operator_id", "")),
            educator_cross_checker=data.get("educator_cross_checker", data.get("operator_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Dossier:
    child_id: str
    display_name: str
    age_band: str
    communication_modality: str
    local_canonical_id: str = field(default_factory=lambda: f"local-{uuid4().hex[:10]}")
    canonical_status: str = "active"
    confirmed_preferences: list[str] = field(default_factory=list)
    confirmed_avoidances: list[str] = field(default_factory=list)
    effective_strategies: list[str] = field(default_factory=list)
    triggers_and_calming_supports: list[str] = field(default_factory=list)
    handoff_notes: list[str] = field(default_factory=list)
    approved_session_summaries: list[SessionSummary] = field(default_factory=list)
    approved_plan_history: list[RecommendationEntry] = field(default_factory=list)
    approved_facial_movement_profiles: list[FacialMovementProfile] = field(default_factory=list)
    consent_grants: list[ConsentGrant] = field(default_factory=list)
    pre_session_rights_checks: list[PreSessionRightsCheck] = field(default_factory=list)
    subject_refusal_active: bool = False
    subject_refusal_at: str | None = None
    subject_refusal_reason: str | None = None
    access_audit_records: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        child_id: str,
        display_name: str,
        age_band: str,
        communication_modality: str,
        confirmed_preferences: list[str] | None = None,
        confirmed_avoidances: list[str] | None = None,
        effective_strategies: list[str] | None = None,
        triggers_and_calming_supports: list[str] | None = None,
        handoff_notes: list[str] | None = None,
    ) -> "Dossier":
        return cls(
            child_id=child_id.strip(),
            display_name=display_name.strip(),
            age_band=age_band.strip(),
            communication_modality=communication_modality.strip(),
            confirmed_preferences=unique_preserving_order(confirmed_preferences or []),
            confirmed_avoidances=unique_preserving_order(confirmed_avoidances or []),
            effective_strategies=unique_preserving_order(effective_strategies or []),
            triggers_and_calming_supports=unique_preserving_order(triggers_and_calming_supports or []),
            handoff_notes=unique_preserving_order(handoff_notes or []),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Dossier":
        return cls(
            child_id=data["child_id"],
            display_name=data["display_name"],
            age_band=data["age_band"],
            communication_modality=data["communication_modality"],
            local_canonical_id=data.get("local_canonical_id", f"local-{uuid4().hex[:10]}"),
            canonical_status=data.get("canonical_status", "active"),
            confirmed_preferences=list(data.get("confirmed_preferences", [])),
            confirmed_avoidances=list(data.get("confirmed_avoidances", [])),
            effective_strategies=list(data.get("effective_strategies", [])),
            triggers_and_calming_supports=list(data.get("triggers_and_calming_supports", [])),
            handoff_notes=list(data.get("handoff_notes", [])),
            approved_session_summaries=[
                SessionSummary.from_dict(item)
                for item in data.get("approved_session_summaries", [])
            ],
            approved_plan_history=[
                RecommendationEntry.from_dict(item)
                for item in data.get("approved_plan_history", [])
            ],
            approved_facial_movement_profiles=[
                FacialMovementProfile.from_dict(item)
                for item in data.get("approved_facial_movement_profiles", [])
            ],
            consent_grants=[
                ConsentGrant.from_dict(item)
                for item in data.get("consent_grants", [])
            ],
            pre_session_rights_checks=[
                PreSessionRightsCheck.from_dict(item)
                for item in data.get("pre_session_rights_checks", [])
            ],
            subject_refusal_active=bool(data.get("subject_refusal_active", False)),
            subject_refusal_at=data.get("subject_refusal_at"),
            subject_refusal_reason=data.get("subject_refusal_reason"),
            access_audit_records=list(data.get("access_audit_records", [])),
            schema_version=SCHEMA_VERSION,
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )

    def touch(self) -> None:
        self.updated_at = utc_now()

    def add_session_summary(self, summary: SessionSummary) -> None:
        self.approved_session_summaries.append(summary)
        self.touch()

    def add_recommendation(self, recommendation: RecommendationEntry) -> None:
        self.approved_plan_history.append(recommendation)
        self.touch()

    def add_facial_movement_profile(self, profile: FacialMovementProfile) -> None:
        existing = {item.label: item for item in self.approved_facial_movement_profiles}
        existing[profile.label] = profile
        self.approved_facial_movement_profiles = list(existing.values())
        self.add_audit_event(
            event_type="facial_movement_profile_approved",
            actor_id=profile.approved_by,
            details={
                "profile_id": profile.profile_id,
                "label": profile.label,
                "source_session_ids": profile.source_session_ids,
            },
        )

    def add_consent_grant(self, grant: ConsentGrant) -> None:
        for existing in self.consent_grants:
            if existing.purpose == grant.purpose and existing.revoked_at is None:
                existing.revoke(actor_id=grant.signer_name, reason="새 동의 확인으로 대체됨")
        self.consent_grants.append(grant)
        self.touch()

    def add_pre_session_rights_check(self, check: PreSessionRightsCheck) -> None:
        self.pre_session_rights_checks.append(check)
        self.pre_session_rights_checks = self.pre_session_rights_checks[-50:]
        self.subject_refusal_active = False
        self.subject_refusal_at = None
        self.subject_refusal_reason = None
        self.touch()

    def activate_subject_refusal(self, *, reason: str = "아동이 중단 버튼을 누름") -> None:
        self.subject_refusal_active = True
        self.subject_refusal_at = utc_now()
        self.subject_refusal_reason = reason.strip() or "아동이 중단 의사를 표시함"
        self.touch()

    def add_audit_event(self, event_type: str, actor_id: str, details: dict[str, Any]) -> None:
        self.access_audit_records.append(
            {
                "event_type": event_type,
                "actor_id": actor_id,
                "timestamp": utc_now(),
                "details": details,
            }
        )
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "child_id": self.child_id,
            "display_name": self.display_name,
            "age_band": self.age_band,
            "communication_modality": self.communication_modality,
            "local_canonical_id": self.local_canonical_id,
            "canonical_status": self.canonical_status,
            "confirmed_preferences": unique_preserving_order(self.confirmed_preferences),
            "confirmed_avoidances": unique_preserving_order(self.confirmed_avoidances),
            "effective_strategies": unique_preserving_order(self.effective_strategies),
            "triggers_and_calming_supports": unique_preserving_order(self.triggers_and_calming_supports),
            "handoff_notes": unique_preserving_order(self.handoff_notes),
            "approved_session_summaries": [item.to_dict() for item in self.approved_session_summaries],
            "approved_plan_history": [item.to_dict() for item in self.approved_plan_history],
            "approved_facial_movement_profiles": [
                item.to_dict() for item in self.approved_facial_movement_profiles
            ],
            "consent_grants": [item.to_dict() for item in self.consent_grants],
            "pre_session_rights_checks": [
                item.to_dict() for item in self.pre_session_rights_checks
            ],
            "subject_refusal_active": self.subject_refusal_active,
            "subject_refusal_at": self.subject_refusal_at,
            "subject_refusal_reason": self.subject_refusal_reason,
            "access_audit_records": self.access_audit_records,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
