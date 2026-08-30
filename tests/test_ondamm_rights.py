from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import ondamm_paths
import ondamm_store
from ondamm_models import ConsentGrant, Dossier, PreSessionRightsCheck
from ondamm_purge import execute_purge, preview_purge
from ondamm_rights import RightsBlockedError, require_camera_session, require_purpose_consent


def dossier() -> Dossier:
    return Dossier.create(
        child_id="rights-child",
        display_name="권리 확인 아동",
        age_band="초등",
        communication_modality="쉬운 말과 그림",
    )


def grant_camera(target: Dossier) -> None:
    target.add_consent_grant(
        ConsentGrant.create(
            purpose="camera_capture",
            signer_name="보호자",
            signature="보호자",
            consent_document_id="CAM-001",
            form_version="1.0",
            guardian_consent_confirmed=True,
            subject_assent_confirmed=True,
        )
    )


def complete_check(target: Dossier) -> None:
    target.add_pre_session_rights_check(
        PreSessionRightsCheck.create(
            operator_id="교사",
            guardian_cross_checker="보호자",
            educator_cross_checker="교사",
            explanation_confirmed=True,
            recording_device_recognized=True,
            camera_off_acclimation_completed=True,
            stop_control_practiced=True,
            subject_willing_now=True,
        )
    )


def test_camera_requires_separate_consent_and_pre_session_check() -> None:
    target = dossier()
    with pytest.raises(RightsBlockedError, match="서명 동의"):
        require_camera_session(target)
    grant_camera(target)
    with pytest.raises(RightsBlockedError, match="교육 전 권리 확인"):
        require_camera_session(target)
    complete_check(target)
    assert require_camera_session(target).operator_id == "교사"


def test_child_stop_blocks_until_new_full_check() -> None:
    target = dossier()
    grant_camera(target)
    complete_check(target)
    target.activate_subject_refusal()
    with pytest.raises(RightsBlockedError, match="중단 의사"):
        require_camera_session(target)
    complete_check(target)
    assert require_camera_session(target).operator_id == "교사"


def test_research_metrics_require_own_signed_purpose() -> None:
    target = dossier()
    grant_camera(target)
    with pytest.raises(RightsBlockedError, match="연구 전용"):
        require_purpose_consent(target, "research_metrics")


def test_purge_preview_and_execute_delete_only_exact_child(tmp_path, monkeypatch) -> None:
    dossiers = tmp_path / "data" / "dossiers"
    exports = tmp_path / "outputs" / "ondamm"
    monkeypatch.setattr(ondamm_store, "ONDAMM_DOSSIERS", dossiers)
    monkeypatch.setattr(ondamm_store, "ONDAMM_EXPORTS", exports)
    monkeypatch.setattr(ondamm_paths, "ONDAMM_DOSSIERS", dossiers)
    monkeypatch.setattr(ondamm_paths, "ONDAMM_EXPORTS", exports)

    target = dossier()
    target.canonical_status = "withdrawn_locked"
    ondamm_store.save_dossier(target)
    other = Dossier.create(child_id="rights-child-2", display_name="다른 아동", age_band="초등", communication_modality="말")
    ondamm_store.save_dossier(other)
    memory = exports / "pattern-memory" / target.child_id
    memory.mkdir(parents=True)
    (memory / "vectors.npz").write_bytes(b"private")
    cache = exports / ".web-cache" / target.child_id
    cache.mkdir(parents=True)
    (cache / "preview.mp4").write_bytes(b"private")

    preview = preview_purge(target.child_id)
    assert preview["target_count"] == 3
    assert all(target.child_id in item["path"] for item in preview["targets"])

    result = execute_purge(target.child_id, confirmation=f"삭제 {target.child_id}", actor_id="보호자")
    assert result["receipt"]["처리_결과"] == "철회 후 로컬 자료 삭제 완료"
    assert not ondamm_store.dossier_path(target.child_id).exists()
    assert ondamm_store.dossier_path(other.child_id).exists()
    assert target.child_id not in result["receipt_path"]
