from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence, Type

from app.ondamm_smirk_manifest import ManifestRecord, load_manifest


def records_for_split(manifest_path: str | Path, split: str) -> list[ManifestRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split}")
    records = [record for record in load_manifest(manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"no records for split {split!r} in {manifest_path}")
    return records


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def make_ondamm_dataset_class(base_dataset_class: Type[object]):
    """Build a SMIRK BaseDataset subclass without importing SMIRK on the Mac."""

    class ONDAMMDataset(base_dataset_class):
        def __init__(self, records: Sequence[ManifestRecord], config, test: bool = False):
            super().__init__(list(records), config, test=test)
            self.name = "ONDAMM"
            self.root = Path(config.dataset.root)

        def __getitem_aux__(self, index: int):
            import cv2
            import numpy as np

            record = self.data_list[index]
            image_path = _resolve(self.root, record.image_path)
            fan_path = _resolve(self.root, record.fan_landmarks_path)
            mediapipe_path = _resolve(self.root, record.mediapipe_landmarks_path)

            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"could not decode image: {image_path}")

            landmarks_fan = np.load(fan_path, allow_pickle=False)
            landmarks_mediapipe = np.load(mediapipe_path, allow_pickle=False)
            if landmarks_fan.ndim == 3:
                if len(landmarks_fan) != 1:
                    raise ValueError(
                        f"{record.image_id}: expected one FAN face, got {len(landmarks_fan)}"
                    )
                landmarks_fan = landmarks_fan[0]
            if landmarks_fan.shape[0] != 68 or landmarks_fan.shape[1] < 2:
                raise ValueError(
                    f"{record.image_id}: FAN landmarks must have shape (68, >=2), "
                    f"got {landmarks_fan.shape}"
                )
            if landmarks_mediapipe.ndim != 2 or landmarks_mediapipe.shape[1] < 2:
                raise ValueError(
                    f"{record.image_id}: invalid MediaPipe landmarks "
                    f"{landmarks_mediapipe.shape}"
                )

            sample = self.prepare_data(
                image=image,
                landmarks_fan=landmarks_fan,
                landmarks_mediapipe=landmarks_mediapipe,
            )
            sample["manifest_index"] = index
            return sample

    ONDAMMDataset.__name__ = "ONDAMMDataset"
    return ONDAMMDataset


def load_smirk_base_dataset(smirk_root: str | Path):
    root = Path(smirk_root).expanduser().resolve()
    path = root / "datasets" / "base_dataset.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing SMIRK BaseDataset: {path}")
    spec = importlib.util.spec_from_file_location("_ondamm_smirk_base_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load SMIRK BaseDataset from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BaseDataset


def build_datasets(config, smirk_root: str | Path):
    base_dataset_class = load_smirk_base_dataset(smirk_root)
    dataset_class = make_ondamm_dataset_class(base_dataset_class)
    train_records = records_for_split(config.dataset.manifest, "train")
    val_records = records_for_split(config.dataset.manifest, "val")
    return (
        dataset_class(train_records, config, test=False),
        dataset_class(val_records, config, test=True),
    )
