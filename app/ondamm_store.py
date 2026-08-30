from __future__ import annotations

import json
from pathlib import Path

from ondamm_models import Dossier
from ondamm_paths import ONDAMM_DOSSIERS, ONDAMM_EXPORTS


def ensure_ondamm_dirs() -> None:
    ONDAMM_DOSSIERS.mkdir(parents=True, exist_ok=True)
    ONDAMM_EXPORTS.mkdir(parents=True, exist_ok=True)


def dossier_path(child_id: str) -> Path:
    ensure_ondamm_dirs()
    return ONDAMM_DOSSIERS / f"{child_id}.json"


def export_path(filename: str) -> Path:
    ensure_ondamm_dirs()
    return ONDAMM_EXPORTS / filename


def create_dossier(dossier: Dossier) -> Path:
    path = dossier_path(dossier.child_id)
    if path.exists():
        raise FileExistsError(f"Dossier already exists: {path}")
    save_dossier(dossier)
    return path


def save_dossier(dossier: Dossier) -> Path:
    path = dossier_path(dossier.child_id)
    ensure_ondamm_dirs()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(dossier.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_dossier(child_id: str) -> Dossier:
    path = dossier_path(child_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing dossier: {path}")
    return Dossier.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_dossiers() -> list[Dossier]:
    ensure_ondamm_dirs()
    dossiers: list[Dossier] = []
    for path in sorted(ONDAMM_DOSSIERS.glob("*.json")):
        dossiers.append(Dossier.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    return dossiers
