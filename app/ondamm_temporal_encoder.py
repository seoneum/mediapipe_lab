"""Frozen causal TCN encoder for personal temporal movement memory.

The product runtime consumes a deliberately narrow feature contract:
MediaPipe blendshapes, canonical geometry, and generic facial motion.  DINO and
pose/gaze/blink nuisance columns are not accepted by the first encoder.

This module does not train online.  It loads an explicitly exported checkpoint,
returns an L2-normalized embedding, and exposes the checkpoint digest used by the
pattern-memory provenance records.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FORBIDDEN_FEATURE_PREFIXES = ("dino_", "geom_delta_", "geom_state_")
FORBIDDEN_FEATURES = {
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "blink",
    "face_ratio",
    "gaze_horizontal",
    "gaze_vertical",
    "dino_pca_available",
    "face_detected",
    "is_calibration",
    "target",
}


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def validate_feature_names(names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(name).strip() for name in names)
    if not normalized or any(not name for name in normalized):
        raise ValueError("feature_names must contain non-empty names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("feature_names must be unique")
    for name in normalized:
        if name in FORBIDDEN_FEATURES or name.startswith(FORBIDDEN_FEATURE_PREFIXES):
            raise ValueError(f"forbidden nuisance/QC feature in temporal encoder: {name}")
        if not (name.startswith("bs_") or name.startswith("geom_abs_") or name.startswith("motion_")):
            raise ValueError(f"unsupported temporal encoder feature: {name}")
    return normalized


@dataclass(frozen=True)
class TemporalEncoderSpec:
    feature_names: tuple[str, ...]
    sequence_length: int = 60
    stride_frames: int = 5
    channels: tuple[int, ...] = (64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.0
    embedding_dim: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", validate_feature_names(self.feature_names))
        if self.sequence_length <= 1:
            raise ValueError("sequence_length must be greater than one")
        if self.stride_frames <= 0:
            raise ValueError("stride_frames must be positive")
        if not self.channels or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must contain positive values")
        if self.kernel_size <= 1:
            raise ValueError("kernel_size must be greater than one")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "sequence_length": self.sequence_length,
            "stride_frames": self.stride_frames,
            "channels": list(self.channels),
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "embedding_dim": self.embedding_dim,
        }


class _Chomp1d:
    """Factory wrapper that keeps torch optional until a real checkpoint is loaded."""

    @staticmethod
    def build(torch: Any, size: int) -> Any:
        class Chomp(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.chomp_size = size

            def forward(self, value: Any) -> Any:
                return value if self.chomp_size == 0 else value[:, :, :-self.chomp_size]

        return Chomp()


def build_torch_encoder(spec: TemporalEncoderSpec) -> Any:
    import torch

    class TemporalBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            padding = (spec.kernel_size - 1) * dilation
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, spec.kernel_size, padding=padding, dilation=dilation),
                _Chomp1d.build(torch, padding),
                torch.nn.ReLU(),
                torch.nn.Dropout(spec.dropout),
                torch.nn.Conv1d(out_channels, out_channels, spec.kernel_size, padding=padding, dilation=dilation),
                _Chomp1d.build(torch, padding),
                torch.nn.ReLU(),
                torch.nn.Dropout(spec.dropout),
            )
            self.residual = (
                torch.nn.Identity()
                if in_channels == out_channels
                else torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )
            self.relu = torch.nn.ReLU()

        def forward(self, value: Any) -> Any:
            return self.relu(self.net(value) + self.residual(value))

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            in_channels = len(spec.feature_names)
            for level, out_channels in enumerate(spec.channels):
                blocks.append(TemporalBlock(in_channels, out_channels, dilation=2**level))
                in_channels = out_channels
            # Name matches the research FacialTCN so its tcn.* weights can be imported.
            self.tcn = torch.nn.Sequential(*blocks)
            self.embedding_head = (
                torch.nn.Identity()
                if spec.channels[-1] == spec.embedding_dim
                else torch.nn.Linear(spec.channels[-1], spec.embedding_dim)
            )

        def forward(self, value: Any) -> Any:
            hidden = self.tcn(value)[:, :, -1]
            embedding = self.embedding_head(hidden)
            return torch.nn.functional.normalize(embedding, p=2, dim=1, eps=1e-8)

    return Encoder()


class TemporalEncoder:
    def __init__(
        self,
        *,
        spec: TemporalEncoderSpec,
        encode_batch: Callable[[np.ndarray], np.ndarray],
        encoder_digest: str,
        normalization_mean: Sequence[float] | None = None,
        normalization_std: Sequence[float] | None = None,
    ) -> None:
        self.spec = spec
        self._encode_batch = encode_batch
        self.encoder_digest = encoder_digest
        feature_count = len(spec.feature_names)
        self.mean = (
            np.zeros(feature_count, dtype=np.float32)
            if normalization_mean is None
            else _finite_vector(normalization_mean, name="normalization_mean")
        )
        self.std = (
            np.ones(feature_count, dtype=np.float32)
            if normalization_std is None
            else _finite_vector(normalization_std, name="normalization_std")
        )
        if self.mean.size != feature_count or self.std.size != feature_count:
            raise ValueError("normalization vectors must match feature_names")
        if np.any(self.std <= 0):
            raise ValueError("normalization_std must be positive")

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        values = np.asarray(sequence, dtype=np.float32)
        expected = (self.spec.sequence_length, len(self.spec.feature_names))
        if values.shape != expected:
            raise ValueError(f"sequence shape must be {expected}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("sequence must contain only finite values")
        normalized = (values - self.mean[None, :]) / self.std[None, :]
        # Torch Conv1d contract: batch, channels, time.
        batch = np.transpose(normalized[None, :, :], (0, 2, 1))
        result = np.asarray(self._encode_batch(batch), dtype=np.float32)
        if result.shape == (1, self.spec.embedding_dim):
            result = result[0]
        if result.shape != (self.spec.embedding_dim,) or not np.isfinite(result).all():
            raise RuntimeError("temporal encoder returned an invalid embedding")
        norm = float(np.linalg.norm(result))
        if norm <= 1e-8 or not math.isfinite(norm):
            raise RuntimeError("temporal encoder returned a zero embedding")
        return result / norm

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: str = "cpu") -> "TemporalEncoder":
        import torch

        checkpoint_path = path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing temporal encoder checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("temporal encoder checkpoint must be a mapping")
        raw_spec = checkpoint.get("encoder_spec")
        if not isinstance(raw_spec, dict):
            raise RuntimeError("checkpoint is missing encoder_spec")
        spec = TemporalEncoderSpec(
            feature_names=tuple(raw_spec.get("feature_names", ())),
            sequence_length=int(raw_spec.get("sequence_length", 60)),
            stride_frames=int(raw_spec.get("stride_frames", 5)),
            channels=tuple(int(value) for value in raw_spec.get("channels", (64, 64, 64))),
            kernel_size=int(raw_spec.get("kernel_size", 3)),
            dropout=float(raw_spec.get("dropout", 0.0)),
            embedding_dim=int(raw_spec.get("embedding_dim", 64)),
        )
        model = build_torch_encoder(spec).to(device)
        state = checkpoint.get("encoder_state_dict") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("checkpoint is missing an encoder state_dict")
        model_state = model.state_dict()
        compatible = {key: value for key, value in state.items() if key in model_state and model_state[key].shape == value.shape}
        missing_tcn = sorted(key for key in model_state if key.startswith("tcn.") and key not in compatible)
        if missing_tcn:
            raise RuntimeError(f"checkpoint is missing compatible TCN weights: {missing_tcn[0]}")
        model.load_state_dict(compatible, strict=False)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        def encode_batch(batch: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
                return model(tensor).detach().cpu().numpy()

        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        return cls(
            spec=spec,
            encode_batch=encode_batch,
            encoder_digest=digest,
            normalization_mean=checkpoint.get("normalization_mean"),
            normalization_std=checkpoint.get("normalization_std"),
        )


def export_temporal_encoder_checkpoint(
    path: Path,
    *,
    spec: TemporalEncoderSpec,
    model_state_dict: Mapping[str, Any],
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Export trained ``tcn.*`` weights into the strict runtime contract.

    A research classifier may contain an additional classification head.  It is
    intentionally discarded: product inference uses the last causal TCN hidden
    state as its temporal embedding.  A missing trained TCN weight is an error;
    this function never creates a random/untrained product fallback.
    """
    import torch

    mean = _finite_vector(normalization_mean, name="normalization_mean")
    std = _finite_vector(normalization_std, name="normalization_std")
    feature_count = len(spec.feature_names)
    if mean.size != feature_count or std.size != feature_count:
        raise ValueError("normalization vectors must match feature_names")
    if np.any(std <= 0):
        raise ValueError("normalization_std must be positive")

    expected = build_torch_encoder(spec).state_dict()
    compatible = {
        key: value.detach().cpu() if hasattr(value, "detach") else value
        for key, value in model_state_dict.items()
        if key in expected and getattr(value, "shape", None) == expected[key].shape
    }
    missing_tcn = sorted(key for key in expected if key.startswith("tcn.") and key not in compatible)
    if missing_tcn:
        raise ValueError(f"model_state_dict is missing compatible TCN weights: {missing_tcn[0]}")
    missing_embedding = sorted(
        key for key in expected if key.startswith("embedding_head.") and key not in compatible
    )
    if missing_embedding:
        raise ValueError(f"model_state_dict is missing compatible embedding weights: {missing_embedding[0]}")

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    payload = {
        "schema_version": 1,
        "encoder_spec": spec.to_dict(),
        "encoder_state_dict": compatible,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def checkpoint_manifest(spec: TemporalEncoderSpec, *, encoder_digest: str) -> str:
    """Stable, human-readable checkpoint contract for manifests and tests."""
    return json.dumps(
        {"encoder_spec": spec.to_dict(), "encoder_digest": encoder_digest},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
