from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import ondamm_cli  # noqa: E402
import ondamm_paths  # noqa: E402
import ondamm_security  # noqa: E402
import ondamm_store  # noqa: E402
from ondamm_models import Dossier  # noqa: E402


class OnDammContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ondamm-continuity-"))
        self.dossiers_dir = self.temp_dir / "dossiers"
        self.exports_dir = self.temp_dir / "exports"
        self.secrets_dir = self.temp_dir / "secrets"

        ondamm_paths.ONDAMM_DATA = self.temp_dir
        ondamm_paths.ONDAMM_DOSSIERS = self.dossiers_dir
        ondamm_paths.ONDAMM_EXPORTS = self.exports_dir
        ondamm_paths.ONDAMM_SECRETS = self.secrets_dir
        ondamm_store.ONDAMM_DOSSIERS = self.dossiers_dir
        ondamm_store.ONDAMM_EXPORTS = self.exports_dir
        ondamm_security.ONDAMM_SECRETS = self.secrets_dir

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def seed_dossier(self, child_id: str = "child-x") -> None:
        dossier = Dossier.create(
            child_id=child_id,
            display_name="Demo Child",
            age_band="초등 저학년",
            communication_modality="시각 단서",
            confirmed_preferences=["동물 카드"],
            effective_strategies=["짧은 단계 제시"],
        )
        ondamm_store.create_dossier(dossier)

    def test_export_manifest_contains_integrity_fields(self) -> None:
        self.seed_dossier("child-export")
        export_output = self.exports_dir / "export-child-export.md"
        manifest_output = self.exports_dir / "export-child-export.md.manifest.json"
        ondamm_cli.command_export_handoff(
            SimpleNamespace(
                child_id="child-export",
                output=str(export_output),
                manifest_output=str(manifest_output),
                actor_id="local-operator",
            )
        )
        manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
        self.assertEqual(manifest["child_id"], "child-export")
        self.assertIn("artifact_id", manifest)
        self.assertIn("dossier_payload_hash", manifest)
        self.assertIn("signature", manifest)
        self.assertIn("snapshot", manifest["validity_disclaimer"])

    def test_prepare_reestablishment_template_records_manual_notice(self) -> None:
        manifest = {
            "artifact_id": "artifact-demo",
            "child_id": "child-export",
            "dossier_payload_hash": "abc123",
            "issuance_time": "2026-06-22T00:00:00+00:00",
        }
        manifest_path = self.exports_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        output = self.exports_dir / "reestablish.json"
        ondamm_cli.command_prepare_reestablishment(SimpleNamespace(manifest=str(manifest_path), output=str(output)))
        template = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(template["source_artifact_id"], "artifact-demo")
        self.assertFalse(template["manual_transfer_confirmed"])
        self.assertIn("수동 재구성", template["manual_reestablishment_notice"])

    def test_withdrawn_dossier_blocks_handoff_generation(self) -> None:
        self.seed_dossier("child-withdraw")
        ondamm_cli.command_withdraw_dossier(
            SimpleNamespace(
                child_id="child-withdraw",
                actor_id="guardian-admin",
                reason_code="consent_withdrawn",
                reason="보호자 요청",
            )
        )
        with self.assertRaises(RuntimeError):
            ondamm_cli.command_handoff_brief(
                SimpleNamespace(child_id="child-withdraw", output=str(self.exports_dir / "handoff.md"), actor_id="local-operator")
            )


if __name__ == "__main__":
    unittest.main()
