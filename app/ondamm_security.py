from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Any
from uuid import uuid4

from ondamm_paths import ONDAMM_SECRETS

ARTIFACT_VERSION = "ondamm-export-v1"
VALIDITY_DISCLAIMER = (
    "이 export는 발급 시점 snapshot이며 recipient-side import/promotion 용도가 아니다. "
    "새 환경에서는 이 artifact를 참고해 수동으로 continuity dossier를 다시 작성해야 한다."
)
SIGNER_MODE = "household_single_operator_demo"
SIGNER_KEY_ID = "local-demo-key"


def ensure_secret_file() -> Path:
    ONDAMM_SECRETS.mkdir(parents=True, exist_ok=True)
    path = ONDAMM_SECRETS / "operator_secret.key"
    if not path.exists():
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return path


def read_secret() -> bytes:
    return ensure_secret_file().read_text(encoding="utf-8").strip().encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sign_text(value: str) -> str:
    return hmac.new(read_secret(), value.encode("utf-8"), hashlib.sha256).hexdigest()


def build_export_manifest(*, child_id: str, issuance_time: str, markdown: str, export_path: str) -> dict[str, Any]:
    payload_hash = sha256_text(markdown)
    signed_payload = "\n".join([child_id, ARTIFACT_VERSION, issuance_time, payload_hash, SIGNER_MODE, SIGNER_KEY_ID, VALIDITY_DISCLAIMER])
    return {
        "artifact_id": f"artifact-{payload_hash[:12]}",
        "child_id": child_id,
        "artifact_version": ARTIFACT_VERSION,
        "issuance_time": issuance_time,
        "dossier_payload_hash": payload_hash,
        "signer_mode": SIGNER_MODE,
        "signer_key_id": SIGNER_KEY_ID,
        "validity_disclaimer": VALIDITY_DISCLAIMER,
        "signature": sign_text(signed_payload),
        "export_path": export_path,
    }


def build_reestablishment_template(*, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": "continuity_reestablished_from_reference",
        "new_local_canonical_id": f"local-{uuid4().hex[:10]}",
        "source_artifact_id": manifest["artifact_id"],
        "source_artifact_hash": manifest["dossier_payload_hash"],
        "source_issuance_time": manifest["issuance_time"],
        "manual_transfer_confirmed": False,
        "notice_acknowledged": False,
        "manual_reestablishment_notice": "이 연속성은 imported state가 아니라 외부 handoff artifact를 참고한 수동 재구성입니다.",
        "notes": "",
    }
