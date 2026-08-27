from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Sequence

from app.ondamm_smirk_manifest import load_manifest, validate_manifest
from smirk_ondamm.dataset import build_datasets


REQUIRED_SMIRK_FILES = (
    "src/smirk_trainer.py",
    "datasets/base_dataset.py",
    "configs/config_train.yaml",
)


def validate_smirk_checkout(smirk_root: str | Path) -> Path:
    root = Path(smirk_root).expanduser().resolve()
    missing = [relative for relative in REQUIRED_SMIRK_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"SMIRK checkout is incomplete at {root}; missing: {', '.join(missing)}"
        )
    return root


def _collate_without_failed_samples(batch):
    import torch

    accepted = [sample for sample in batch if sample is not None]
    if not accepted:
        return None
    return torch.utils.data.dataloader.default_collate(accepted)


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch, device: str):
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune SMIRK on a leakage-audited ON DAMM manifest.")
    parser.add_argument("--smirk-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--resume",
        help="Optional checkpoint. Omit for encoder pretraining from initialization.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--max-images-per-person", type=int, default=5000)
    args = parser.parse_args(argv)

    smirk_root = validate_smirk_checkout(args.smirk_root)
    manifest_records = load_manifest(args.manifest)
    summary = validate_manifest(
        manifest_records,
        split_mode="global",
        max_images_per_person=args.max_images_per_person,
        require_files=True,
        root=args.dataset_root,
    )

    # SMIRK resolves FLAME, renderer, and expression-template assets relative
    # to the repository root, so running from the caller's directory is unsafe.
    os.chdir(smirk_root)
    sys.path.insert(0, str(smirk_root))
    import torch
    from omegaconf import OmegaConf
    from src.smirk_trainer import SmirkTrainer

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    config = OmegaConf.load(args.config)
    config.device = args.device
    config.resume = (
        str(Path(args.resume).expanduser().resolve())
        if args.resume
        else config.get("resume")
    )
    config.train.log_path = str(Path(args.run_dir).expanduser().resolve())
    config.dataset.manifest = str(Path(args.manifest).expanduser().resolve())
    config.dataset.root = str(Path(args.dataset_root).expanduser().resolve())

    run_dir = Path(config.train.log_path)
    train_images_dir = run_dir / "train_images"
    val_images_dir = run_dir / "val_images"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, run_dir / "config.yaml")
    (run_dir / "manifest-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _seed_everything(args.seed)
    train_dataset, val_dataset = build_datasets(config, smirk_root)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=_collate_without_failed_samples,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
        collate_fn=_collate_without_failed_samples,
    )

    trainer = SmirkTrainer(config).to(config.device)
    if config.resume:
        trainer.load_model(
            config.resume,
            load_fuse_generator=config.load_fuse_generator,
            load_encoder=config.load_encoder,
            device=config.device,
        )
    trainer.create_base_encoder()

    for epoch in range(config.train.resume_epoch, config.train.num_epochs):
        trainer.configure_optimizers(len(train_loader))
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            for batch_index, batch in enumerate(loader):
                if batch is None:
                    continue
                trainer.set_freeze_status(config, batch_index, epoch)
                batch = _move_batch(batch, config.device)
                outputs = trainer.step(batch, batch_index, phase=phase)
                if batch_index % config.train.visualize_every == 0:
                    with torch.no_grad():
                        visualizations = trainer.create_visualizations(batch, outputs)
                        trainer.save_visualizations(
                            visualizations,
                            str(run_dir / f"{phase}_images" / f"{epoch}_{batch_index}.jpg"),
                            show_landmarks=True,
                        )
        if epoch % config.train.save_every == 0:
            trainer.save_model(
                trainer.state_dict(),
                str(run_dir / f"model_{epoch}.pt"),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
