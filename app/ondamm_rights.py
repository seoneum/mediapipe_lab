from __future__ import annotations

from datetime import datetime, timezone

from ondamm_models import ConsentGrant, Dossier, PreSessionRightsCheck
from ondamm_store import load_dossier


PURPOSE_LABELS = {
    "camera_capture": "카메라 촬영 및 현장 관찰",
    "research_metrics": "연구 전용 표정·흥미·주의 지표 분석",
    "model_training": "연구용 모델 학습",
    "remote_review": "외부 또는 원격 교차 검토",
}


class RightsBlockedError(RuntimeError):
    """아동의 권리 또는 유효한 동의가 확인되지 않아 작업을 막을 때 사용한다."""


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def consent_is_active(grant: ConsentGrant, *, now: datetime | None = None) -> bool:
    current = now or datetime.now(timezone.utc)
    if (
        grant.revoked_at
        or not grant.guardian_consent_confirmed
        or grant.subject_assent != "affirmed"
        or not grant.signer_name.strip()
        or not grant.consent_document_id.strip()
        or len(grant.signature_digest) != 64
    ):
        return False
    if not grant.expires_at:
        return True
    try:
        return _parse_time(grant.expires_at) > current
    except (TypeError, ValueError):
        return False


def latest_active_consent(
    dossier: Dossier, purpose: str, *, now: datetime | None = None
) -> ConsentGrant | None:
    matches = [
        grant
        for grant in dossier.consent_grants
        if grant.purpose == purpose and consent_is_active(grant, now=now)
    ]
    return matches[-1] if matches else None


def latest_valid_pre_session_check(
    dossier: Dossier, *, now: datetime | None = None
) -> PreSessionRightsCheck | None:
    current = now or datetime.now(timezone.utc)
    valid = []
    for check in dossier.pre_session_rights_checks:
        try:
            not_expired = _parse_time(check.expires_at) > current
        except (TypeError, ValueError):
            not_expired = False
        if (
            not_expired
            and check.explanation_confirmed
            and check.recording_device_recognized
            and check.camera_off_acclimation_completed
            and check.stop_control_practiced
            and check.subject_willing_now
            and bool(check.guardian_cross_checker.strip())
            and bool(check.educator_cross_checker.strip())
            and check.guardian_cross_checker.strip() != check.educator_cross_checker.strip()
        ):
            valid.append(check)
    return valid[-1] if valid else None


def require_purpose_consent(dossier: Dossier, purpose: str) -> ConsentGrant:
    if dossier.canonical_status != "active":
        raise RightsBlockedError("철회되거나 잠긴 기록철입니다. 새로운 촬영·분석을 시작할 수 없습니다.")
    grant = latest_active_consent(dossier, purpose)
    if grant is None:
        label = PURPOSE_LABELS.get(purpose, purpose)
        raise RightsBlockedError(f"‘{label}’에 대한 유효한 서명 동의가 없어 실행을 중단했습니다.")
    return grant


def require_camera_session(dossier: Dossier) -> PreSessionRightsCheck:
    require_purpose_consent(dossier, "camera_capture")
    if dossier.subject_refusal_active:
        raise RightsBlockedError(
            "아동이 중단 의사를 표시했습니다. 촬영을 재개하지 마세요. 충분히 쉰 뒤, "
            "아동이 다시 원할 때 교육 전 확인을 처음부터 진행해야 합니다."
        )
    check = latest_valid_pre_session_check(dossier)
    if check is None:
        raise RightsBlockedError(
            "유효한 교육 전 권리 확인이 없습니다. 웹에서 카메라를 끈 적응 시간, "
            "촬영 장치 설명, 중단 버튼 연습과 현재 참여 의사를 먼저 확인해 주세요."
        )
    return check


def require_research_metrics_consent(child_id: str) -> ConsentGrant:
    try:
        dossier = load_dossier(child_id)
    except FileNotFoundError as exc:
        raise RightsBlockedError("해당 아동의 로컬 기록철을 찾을 수 없습니다.") from exc
    return require_purpose_consent(dossier, "research_metrics")


def rights_summary(dossier: Dossier) -> dict[str, object]:
    current_check = latest_valid_pre_session_check(dossier)
    purposes = {}
    for purpose, label in PURPOSE_LABELS.items():
        grant = latest_active_consent(dossier, purpose)
        purposes[purpose] = {
            "label": label,
            "active": grant is not None,
            "grant_id": grant.grant_id if grant else None,
            "granted_at": grant.granted_at if grant else None,
            "signer_name": grant.signer_name if grant else None,
        }
    return {
        "record_status_label": "사용 가능" if dossier.canonical_status == "active" else "철회로 잠김",
        "child_stop_active": dossier.subject_refusal_active,
        "child_stop_message": (
            "아동이 중단을 요청했습니다. 촬영과 분석을 시작할 수 없습니다."
            if dossier.subject_refusal_active
            else "현재 중단 요청 없음"
        ),
        "pre_session_ready": current_check is not None and not dossier.subject_refusal_active,
        "pre_session_check": current_check.to_dict() if current_check else None,
        "purposes": purposes,
    }
