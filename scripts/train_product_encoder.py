from __future__ import annotations

"""Train and export the runtime product TCN on all development participants."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
APP_DIR = ROOT / "app"
for directory in (SCRIPTS_DIR, APP_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_v4_modality_ablation as ab  # noqa: E402
import train_v4_tcn as tcn  # noqa: E402
from ondamm_temporal_encoder import TemporalEncoderSpec, export_temporal_encoder_checkpoint  # noqa: E402


def train_full_development(pack: tcn.SequencePack, *, args: argparse.Namespace, device: torch.device, epochs: int):
    """Fit final weights on every development sequence after epoch selection."""
    tcn.seed_everything(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        tcn.SequenceDataset(pack),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    channels = [int(value) for value in args.channels.split(",") if value.strip()]
    model = tcn.FacialTCN(
        input_channels=pack.X.shape[1],
        channels=channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    positives = max(1, int((pack.y == 1).sum()))
    negatives = max(1, int((pack.y == 0).sum()))
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": float(sum(losses) / len(losses))})
    return model, pd.DataFrame(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train encoder_product.pt on all development participants")
    parser.add_argument("--participants", nargs="+", default=["p1", "p2", "p3"])
    parser.add_argument("--session", default="s01")
    parser.add_argument("--calibration-start", type=float, default=1.0)
    parser.add_argument("--calibration-end", type=float, default=7.0)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--validation-repeat", type=int, default=3)
    parser.add_argument("--min-face-coverage", type=float, default=0.80)
    parser.add_argument("--channels", default="64,64,64")
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "micro_expression" / "v4_tcn"
    )
    args = parser.parse_args()

    tcn.seed_everything(args.seed)
    device = tcn.choose_device(args.device)
    subject_frames = {}
    raw_frames = []
    for participant in args.participants:
        frame = tcn.make_trial_key(
            tcn.load_subject(participant, args.session, args.calibration_start, args.calibration_end)
        )
        subject_frames[participant] = frame
        raw_frames.append(frame)
    blendshape, geometry, motion, features = tcn.discover_core_features(raw_frames)
    development = pd.concat(raw_frames, ignore_index=True, sort=False)
    train_split, val_split = tcn.trial_level_split(development, args.validation_repeat)
    stats = ab.robust_stats(ab.calibration_frame(development), features)
    train_z = ab.z_transform(train_split, features, stats)
    val_z = ab.z_transform(val_split, features, stats)
    train_pack = tcn.build_sequences(
        train_split,
        train_z,
        features,
        seq_len=args.seq_len,
        stride=args.stride,
        min_face_coverage=args.min_face_coverage,
        include_control=True,
    )
    val_pack = tcn.build_sequences(
        val_split,
        val_z,
        features,
        seq_len=args.seq_len,
        stride=args.stride,
        min_face_coverage=args.min_face_coverage,
        include_control=True,
    )
    _, selection_history, best_epoch = tcn.train_one_fold(
        train_pack, val_pack, input_channels=len(features), args=args, device=device
    )
    development_z = ab.z_transform(development, features, stats)
    development_pack = tcn.build_sequences(
        development,
        development_z,
        features,
        seq_len=args.seq_len,
        stride=args.stride,
        min_face_coverage=args.min_face_coverage,
        include_control=True,
    )
    model, product_history = train_full_development(
        development_pack,
        args=args,
        device=device,
        epochs=best_epoch,
    )
    channels = tuple(int(value) for value in args.channels.split(",") if value.strip())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "encoder_product.pt"
    digest = export_temporal_encoder_checkpoint(
        checkpoint,
        spec=TemporalEncoderSpec(
            feature_names=tuple(features),
            sequence_length=args.seq_len,
            stride_frames=args.stride,
            channels=channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
            embedding_dim=channels[-1],
        ),
        model_state_dict=model.state_dict(),
        normalization_mean=[float(stats.center[name]) for name in features],
        normalization_std=[float(stats.scale[name]) for name in features],
        metadata={
            "checkpoint_role": "product",
            "development_participants": list(args.participants),
            "train_participants": list(args.participants),
            "validation_repeat": int(args.validation_repeat),
            "best_epoch": int(best_epoch),
            "epoch_selection": "repeat-held-out validation, then fixed-epoch refit on all development sequences",
            "normalization": "all-development calibration robust center/scale",
        },
    )

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config.update(
        {
            "participants": list(args.participants),
            "session": args.session,
            "features": list(features),
            "feature_counts": {
                "blendshape": len(blendshape),
                "geometry": len(geometry),
                "motion": len(motion),
                "total": len(features),
            },
        }
    )
    config.setdefault("sequence", {}).update(
        {"causal": True, "seq_len_frames": args.seq_len, "stride_frames": args.stride}
    )
    config.setdefault("model", {}).update(
        {"channels": args.channels, "kernel_size": args.kernel_size, "dropout": args.dropout}
    )
    config.setdefault("encoder_checkpoints", {})["product"] = {
        "path": str(checkpoint.relative_to(ROOT)),
        "sha256": digest,
        "role": "runtime-product",
    }
    config["runtime_default_checkpoint"] = str(checkpoint.relative_to(ROOT))
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selection_history.to_csv(output_dir / "product_epoch_selection_history.csv", index=False)
    product_history.to_csv(output_dir / "product_epoch_history.csv", index=False)
    print(f"device: {device}")
    print(f"train_sequences: {len(train_pack.y)}")
    print(f"validation_sequences: {len(val_pack.y)}")
    print(f"full_development_sequences: {len(development_pack.y)}")
    print(f"best_epoch: {best_epoch}")
    print(f"encoder_product: {checkpoint}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
