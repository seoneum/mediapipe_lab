from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


MEASUREMENT_KEYS = (
    "expression_params",
    "eyelid_params",
    "jaw_params",
    "pose_params",
    "cam",
    "shape_params",
)


def encoder_state_dict(checkpoint: Mapping[str, object]) -> dict[str, object]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a state dictionary")
    selected = {
        key.removeprefix("smirk_encoder."): value
        for key, value in state.items()
        if key.startswith("smirk_encoder.")
    }
    if selected:
        return selected
    return dict(state)


def _first_batch_item(value):
    data = value.detach().cpu().tolist()
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


def serialize_encoder_outputs(outputs: Mapping[str, object]) -> dict[str, object]:
    missing = [key for key in MEASUREMENT_KEYS if key not in outputs]
    if missing:
        raise ValueError(f"SMIRK encoder output is missing: {', '.join(missing)}")
    return {key: _first_batch_item(outputs[key]) for key in MEASUREMENT_KEYS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_paths(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export SMIRK encoder measurements as JSONL.")
    parser.add_argument("--smirk-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--already-cropped", action="store_true")
    args = parser.parse_args(argv)

    smirk_root = Path(args.smirk_root).expanduser().resolve()
    if not (smirk_root / "src" / "smirk_encoder.py").is_file():
        raise FileNotFoundError(f"invalid SMIRK checkout: {smirk_root}")
    sys.path.insert(0, str(smirk_root))

    import cv2
    import numpy as np
    import torch
    from skimage.transform import warp
    from src.smirk_encoder import SmirkEncoder

    model = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(encoder_state_dict(checkpoint))
    model.eval()

    run_mediapipe = None
    crop_face = None
    if not args.already_cropped:
        from datasets.base_dataset import BaseDataset
        from utils.mediapipe_utils import run_mediapipe as smirk_run_mediapipe

        run_mediapipe = smirk_run_mediapipe
        crop_face = BaseDataset.crop_face

    input_root = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in _image_paths(input_root):
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                continue
            if args.already_cropped:
                crop = cv2.resize(bgr, (224, 224))
            else:
                landmarks = run_mediapipe(bgr)
                if landmarks is None:
                    continue
                transform = crop_face(bgr, landmarks[..., :2], scale=1.4, image_size=224)
                crop = warp(
                    bgr,
                    transform.inverse,
                    output_shape=(224, 224),
                    preserve_range=True,
                ).astype(np.uint8)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .div(255.0)
                .to(args.device)
            )
            with torch.inference_mode():
                measurements = serialize_encoder_outputs(model(tensor))
            payload = {
                "source_path": str(image_path.relative_to(input_root)),
                "source_sha256": _sha256(image_path),
                "checkpoint": str(Path(args.checkpoint).name),
                "measurements": measurements,
                "notice": "Observable geometry only; not emotion, attention, preference, or diagnosis.",
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
