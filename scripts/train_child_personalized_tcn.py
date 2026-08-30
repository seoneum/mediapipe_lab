from __future__ import annotations

"""Fine-tune the temporal encoder for one child and test on a later session.

P1/P2/P3 may provide the pretrained initialization, but they are never part of
the objective or the final evaluation here.  Every train/validation row belongs
to ``--child-id`` and ``--future-session`` is loaded only after model fitting,
normalization, and threshold selection are complete.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
SCRIPTS_DIR = ROOT / "scripts"
for directory in (APP_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_v4_modality_ablation as ab  # noqa: E402
import train_v4_tcn as tcn  # noqa: E402
from ondamm_temporal_encoder import (  # noqa: E402
    TemporalEncoder,
    TemporalEncoderSpec,
    export_temporal_encoder_checkpoint,
)


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


@dataclass(frozen=True)
class SessionPartition:
    child_id: str
    train_sessions: tuple[str, ...]
    future_session: str


def validate_session_partition(
    child_id: str,
    train_sessions: Sequence[str],
    future_session: str,
) -> SessionPartition:
    child = str(child_id).strip()
    if not SAFE_ID.fullmatch(child):
        raise ValueError("child_id contains unsupported characters")
    sessions = tuple(str(value).strip() for value in train_sessions)
    if not sessions or any(not value for value in sessions):
        raise ValueError("at least one non-empty training session is required")
    if len(set(sessions)) != len(sessions):
        raise ValueError("training sessions must be unique")
    future = str(future_session).strip()
    if not future:
        raise ValueError("future_session is required")
    if future in sessions:
        raise ValueError("future_session must be held out from training sessions")
    return SessionPartition(child, sessions, future)


def _checkpoint_payload(path: Path) -> tuple[TemporalEncoder, Mapping[str, Any]]:
    source = path.expanduser().resolve()
    # Strict validation prevents accidental initialization from a random or
    # incompatible research checkpoint.
    encoder = TemporalEncoder.from_checkpoint(source, device="cpu", product_contract=True)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RuntimeError("pretrained checkpoint must be a mapping")
    state = payload.get("encoder_state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("pretrained checkpoint is missing encoder_state_dict")
    return encoder, payload


def initialized_child_model(
    spec: TemporalEncoderSpec,
    pretrained_state: Mapping[str, Any],
    *,
    device: torch.device,
) -> tcn.FacialTCN:
    model = tcn.FacialTCN(
        input_channels=len(spec.feature_names),
        channels=list(spec.channels),
        kernel_size=spec.kernel_size,
        dropout=spec.dropout,
    ).to(device)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in pretrained_state.items()
        if key.startswith("tcn.")
        and key in model_state
        and getattr(value, "shape", None) == model_state[key].shape
    }
    missing = sorted(
        key for key in model_state if key.startswith("tcn.") and key not in compatible
    )
    if missing:
        raise RuntimeError(f"pretrained checkpoint is missing compatible TCN weight: {missing[0]}")
    model.load_state_dict(compatible, strict=False)
    return model


def _criterion(pack: tcn.SequencePack, device: torch.device) -> nn.Module:
    positives = max(1, int((pack.y == 1).sum()))
    negatives = max(1, int((pack.y == 0).sum()))
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    )


def _loader(
    pack: tcn.SequencePack,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        tcn.SequenceDataset(pack),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("child fine-tuning received no training batches")
    return float(np.mean(losses))


def select_fine_tuning_epoch(
    spec: TemporalEncoderSpec,
    pretrained_state: Mapping[str, Any],
    train_pack: tcn.SequencePack,
    val_pack: tcn.SequencePack,
    *,
    args: argparse.Namespace | SimpleNamespace,
    device: torch.device,
) -> tuple[int, pd.DataFrame]:
    if len(np.unique(train_pack.y)) != 2 or len(np.unique(val_pack.y)) != 2:
        raise RuntimeError("child train and validation sequences must both contain two classes")
    tcn.seed_everything(args.seed)
    model = initialized_child_model(spec, pretrained_state, device=device)
    train_loader = _loader(
        train_pack,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = _loader(
        val_pack,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    criterion = _criterion(train_pack, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_epoch = -1
    best_auprc = -np.inf
    patience_left = args.patience
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        probability, labels = tcn.batch_probabilities(model, val_loader, device)
        auroc, auprc = tcn.evaluate_ranking(labels, probability)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_auroc": auroc,
                "val_auprc": auprc,
            }
        )
        score = auprc if np.isfinite(auprc) else -np.inf
        if score > best_auprc + args.min_delta:
            best_auprc = score
            best_epoch = epoch
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
    if best_epoch < 1:
        raise RuntimeError("child fine-tuning could not select an epoch")
    return best_epoch, pd.DataFrame(history)


def fit_child_model(
    spec: TemporalEncoderSpec,
    pretrained_state: Mapping[str, Any],
    pack: tcn.SequencePack,
    *,
    epochs: int,
    args: argparse.Namespace | SimpleNamespace,
    device: torch.device,
) -> tuple[tcn.FacialTCN, pd.DataFrame]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if len(np.unique(pack.y)) != 2:
        raise RuntimeError("full child training sequences must contain two classes")
    tcn.seed_everything(args.seed)
    model = initialized_child_model(spec, pretrained_state, device=device)
    loader = _loader(pack, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    criterion = _criterion(pack, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    for epoch in range(1, epochs + 1):
        history.append(
            {
                "epoch": epoch,
                "train_loss": _train_epoch(
                    model,
                    loader,
                    criterion,
                    optimizer,
                    device,
                ),
            }
        )
    return model, pd.DataFrame(history)


def _load_child_sessions(
    child_id: str,
    sessions: Sequence[str],
    *,
    calibration_start: float,
    calibration_end: float,
) -> pd.DataFrame:
    frames = []
    for session in sessions:
        frame = tcn.load_subject(
            child_id,
            session,
            calibration_start,
            calibration_end,
        )
        frame = tcn.make_trial_key(frame)
        frame["session_id"] = session
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _pack(
    frame: pd.DataFrame,
    normalized: pd.DataFrame,
    features: Sequence[str],
    *,
    spec: TemporalEncoderSpec,
    min_face_coverage: float,
    include_control: bool,
) -> tcn.SequencePack:
    return tcn.build_sequences(
        frame,
        normalized,
        list(features),
        seq_len=spec.sequence_length,
        stride=spec.stride_frames,
        min_face_coverage=min_face_coverage,
        include_control=include_control,
    )


def _probabilities(
    model: nn.Module,
    pack: tcn.SequencePack,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    return tcn.batch_probabilities(
        model,
        _loader(pack, batch_size=batch_size, shuffle=False, seed=0),
        device,
    )


def _future_movement_metrics(
    model: nn.Module,
    future: pd.DataFrame,
    future_z: pd.DataFrame,
    features: Sequence[str],
    *,
    spec: TemporalEncoderSpec,
    min_face_coverage: float,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    action = future[future["protocol"].isin(["upper", "lower"])].copy()
    control = future[future["protocol"].eq("control")].copy()
    action_pack = _pack(
        action,
        future_z.loc[action.index, list(features)],
        features,
        spec=spec,
        min_face_coverage=min_face_coverage,
        include_control=False,
    )
    control_pack = _pack(
        control,
        future_z.loc[control.index, list(features)],
        features,
        spec=spec,
        min_face_coverage=min_face_coverage,
        include_control=True,
    )
    action_probability, action_y = _probabilities(
        model,
        action_pack,
        batch_size=batch_size,
        device=device,
    )
    control_probability, _ = _probabilities(
        model,
        control_pack,
        batch_size=batch_size,
        device=device,
    )
    metrics = {
        **tcn.classification_metrics(action_y, action_probability, threshold),
        **tcn.event_metrics(action_pack.meta, action_probability, threshold, min_consecutive=3),
        **tcn.action_phase_metrics(action_pack.meta, action_probability, threshold),
    }
    control_fraction, control_far = tcn.control_far_per_min(
        control_pack.meta,
        control_probability,
        threshold,
        spec.stride_frames,
        fps=30.0,
    )
    metrics.update(
        {
            "control_positive_fraction": control_fraction,
            "false_activations_per_min": control_far,
            "threshold": threshold,
            "action_sequences": int(len(action_pack.y)),
            "control_sequences": int(len(control_pack.y)),
            "metric_scope": "same-child held-out future session; scripted movement proxy",
        }
    )
    action_predictions = action_pack.meta.copy().reset_index(drop=True)
    action_predictions["challenge"] = "action"
    action_predictions["probability"] = action_probability
    control_predictions = control_pack.meta.copy().reset_index(drop=True)
    control_predictions["challenge"] = "control"
    control_predictions["probability"] = control_probability
    predictions = pd.concat([action_predictions, control_predictions], ignore_index=True)
    predictions["prediction"] = (predictions["probability"] >= threshold).astype(int)
    predictions["threshold"] = threshold
    return metrics, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strong within-child TCN fine-tuning with future-session testing",
    )
    parser.add_argument("--child-id", required=True)
    parser.add_argument("--train-sessions", nargs="+", required=True)
    parser.add_argument("--future-session", required=True)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=ROOT / "outputs" / "micro_expression" / "v4_tcn" / "encoder_product.pt",
    )
    parser.add_argument("--calibration-start", type=float, default=1.0)
    parser.add_argument("--calibration-end", type=float, default=7.0)
    parser.add_argument("--validation-repeat", type=int, default=3)
    parser.add_argument("--min-face-coverage", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partition = validate_session_partition(
        args.child_id,
        args.train_sessions,
        args.future_session,
    )
    if not 0 < args.min_face_coverage <= 1:
        raise ValueError("--min-face-coverage must be in (0, 1]")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch size must be positive")
    device = tcn.choose_device(args.device)
    pretrained_encoder, payload = _checkpoint_payload(args.pretrained_checkpoint)
    spec = pretrained_encoder.spec
    pretrained_state = payload["encoder_state_dict"]

    # Only historical sessions are loaded while fitting normalization/model.
    historical = _load_child_sessions(
        partition.child_id,
        partition.train_sessions,
        calibration_start=args.calibration_start,
        calibration_end=args.calibration_end,
    )
    _, _, _, discovered_features = tcn.discover_core_features([historical])
    if tuple(discovered_features) != spec.feature_names:
        raise RuntimeError("target-child feature order does not match pretrained encoder")
    train_split, val_split = tcn.trial_level_split(historical, args.validation_repeat)
    stats = ab.robust_stats(ab.calibration_frame(historical), list(spec.feature_names))
    train_z = ab.z_transform(train_split, list(spec.feature_names), stats)
    val_z = ab.z_transform(val_split, list(spec.feature_names), stats)
    historical_z = ab.z_transform(historical, list(spec.feature_names), stats)
    train_pack = _pack(
        train_split,
        train_z,
        spec.feature_names,
        spec=spec,
        min_face_coverage=args.min_face_coverage,
        include_control=True,
    )
    val_pack = _pack(
        val_split,
        val_z,
        spec.feature_names,
        spec=spec,
        min_face_coverage=args.min_face_coverage,
        include_control=True,
    )
    historical_pack = _pack(
        historical,
        historical_z,
        spec.feature_names,
        spec=spec,
        min_face_coverage=args.min_face_coverage,
        include_control=True,
    )
    best_epoch, selection_history = select_fine_tuning_epoch(
        spec,
        pretrained_state,
        train_pack,
        val_pack,
        args=args,
        device=device,
    )
    model, fine_tune_history = fit_child_model(
        spec,
        pretrained_state,
        historical_pack,
        epochs=best_epoch,
        args=args,
        device=device,
    )
    historical_probability, historical_y = _probabilities(
        model,
        historical_pack,
        batch_size=args.batch_size,
        device=device,
    )
    threshold = tcn.choose_threshold(historical_y, historical_probability)

    # The future session is first loaded after all train-time choices are fixed.
    future = _load_child_sessions(
        partition.child_id,
        [partition.future_session],
        calibration_start=args.calibration_start,
        calibration_end=args.calibration_end,
    )
    future_z = ab.z_transform(future, list(spec.feature_names), stats)
    movement_metrics, future_predictions = _future_movement_metrics(
        model,
        future,
        future_z,
        spec.feature_names,
        spec=spec,
        min_face_coverage=args.min_face_coverage,
        batch_size=args.batch_size,
        device=device,
        threshold=threshold,
    )

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            ROOT
            / "outputs"
            / "micro_expression"
            / "children"
            / partition.child_id
            / "temporal"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "encoder_child_personalized.pt"
    digest = export_temporal_encoder_checkpoint(
        checkpoint,
        spec=spec,
        model_state_dict=model.state_dict(),
        normalization_mean=[float(stats.center[name]) for name in spec.feature_names],
        normalization_std=[float(stats.scale[name]) for name in spec.feature_names],
        metadata={
            "checkpoint_role": "child-personalized",
            "child_id": partition.child_id,
            "development_participants": list(
                payload.get("metadata", {}).get("development_participants", [])
            ),
            "train_participants": [partition.child_id],
            "training_sessions": list(partition.train_sessions),
            "future_session_excluded_from_training": partition.future_session,
            "pretrained_checkpoint_sha256": pretrained_encoder.encoder_digest,
            "best_epoch": int(best_epoch),
            "epoch_selection": (
                "target-child held-out repeat, then fixed-epoch refit on all historical sessions"
            ),
            "normalization": "target-child historical-session calibration robust center/scale",
            "catastrophic_forgetting_of_development_participants_allowed": True,
        },
    )
    config = {
        "objective": "strong within-child personalization",
        "child_id": partition.child_id,
        "training_sessions": list(partition.train_sessions),
        "future_session": partition.future_session,
        "future_session_role": "held-out test only",
        "cross_person_generalization_metric": False,
        "development_participant_retention_required": False,
        "features": list(spec.feature_names),
        "feature_counts": {
            "blendshape": sum(name.startswith("bs_") for name in spec.feature_names),
            "geometry": sum(name.startswith("geom_abs_") for name in spec.feature_names),
            "motion": sum(name.startswith("motion_") for name in spec.feature_names),
            "total": len(spec.feature_names),
        },
        "sequence": {
            "causal": True,
            "seq_len_frames": spec.sequence_length,
            "stride_frames": spec.stride_frames,
        },
        "model": {
            "channels": ",".join(str(value) for value in spec.channels),
            "kernel_size": spec.kernel_size,
            "dropout": spec.dropout,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "all_tcn_layers_trainable": True,
        },
        "threshold": {
            "value": threshold,
            "source": "target-child historical training sessions after final refit",
        },
        "encoder_checkpoints": {
            "child_personalized": {
                "path": str(checkpoint),
                "sha256": digest,
                "role": "runtime-child-personalized",
            }
        },
        "runtime_activation": (
            "explicit only; changing encoder invalidates old prototype vectors, so rebuild or "
            "re-embed child memory before switching"
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "future_session_movement_metrics.json").write_text(
        json.dumps(movement_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selection_history.to_csv(output_dir / "epoch_selection_history.csv", index=False)
    fine_tune_history.to_csv(output_dir / "fine_tune_history.csv", index=False)
    future_predictions.to_csv(output_dir / "future_session_predictions.csv", index=False)
    print(f"objective: strong within-child personalization")
    print(f"child_id: {partition.child_id}")
    print(f"training_sessions: {', '.join(partition.train_sessions)}")
    print(f"held_out_future_session: {partition.future_session}")
    print(f"best_epoch: {best_epoch}")
    print(f"checkpoint: {checkpoint}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
