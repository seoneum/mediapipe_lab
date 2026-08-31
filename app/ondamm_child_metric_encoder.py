"""Child-specific metric projection over the frozen temporal encoder."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ondamm_temporal_encoder import TemporalEncoder


class ChildMetricEncoder:
    """
    Runtime-compatible temporal encoder wrapper.

    79D sequence
        -> frozen base TCN
        -> 64D normalized embedding
        -> child-specific linear metric projection
        -> 32D normalized embedding
    """

    def __init__(
        self,
        *,
        base_encoder: TemporalEncoder,
        projection_weight: np.ndarray,
        child_id: str,
        metric_checkpoint_digest: str,
        candidate_distance_threshold: float,
    ) -> None:
        weight = np.asarray(
            projection_weight,
            dtype=np.float32,
        )

        if weight.ndim != 2:
            raise ValueError(
                "metric projection weight must be 2D"
            )

        if (
            weight.shape[1]
            != base_encoder.spec.embedding_dim
        ):
            raise ValueError(
                "metric projection input dimension "
                "does not match base encoder"
            )

        if not np.isfinite(weight).all():
            raise ValueError(
                "metric projection contains non-finite values"
            )

        if not 0 < candidate_distance_threshold <= 2:
            raise ValueError(
                "candidate distance threshold must be in (0, 2]"
            )

        self.base_encoder = base_encoder
        self.projection_weight = weight

        self.child_id = str(
            child_id
        ).strip()

        self.candidate_distance_threshold = float(
            candidate_distance_threshold
        )

        # Same 79D input contract / sequence contract,
        # but output embedding dimension becomes 32D.
        self.spec = replace(
            base_encoder.spec,
            embedding_dim=int(
                weight.shape[0]
            ),
        )

        provenance = {
            "schema": "ondamm-child-metric-runtime-v1",
            "child_id": self.child_id,
            "base_encoder_sha256":
                base_encoder.encoder_digest,
            "metric_checkpoint_sha256":
                metric_checkpoint_digest,
            "embedding_dimension":
                self.spec.embedding_dim,
        }

        canonical = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        # PatternMemory provenance identity.
        #
        # Any base encoder OR metric-head change creates
        # an entirely different memory space.
        self.encoder_digest = hashlib.sha256(
            canonical
        ).hexdigest()

    def encode(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:
        base = self.base_encoder.encode(
            sequence
        )

        projected = (
            self.projection_weight
            @ base
        )

        if (
            projected.ndim != 1
            or projected.size
            != self.spec.embedding_dim
            or not np.isfinite(
                projected
            ).all()
        ):
            raise RuntimeError(
                "metric head produced invalid embedding"
            )

        norm = float(
            np.linalg.norm(
                projected
            )
        )

        if norm <= 1e-8:
            raise RuntimeError(
                "metric head produced zero embedding"
            )

        return (
            projected / norm
        ).astype(
            np.float32
        )


def load_child_metric_encoder(
    *,
    base_checkpoint_path: Path,
    metric_checkpoint_path: Path,
    child_id: str,
) -> ChildMetricEncoder:
    base_path = (
        base_checkpoint_path
        .expanduser()
        .resolve()
    )

    metric_path = (
        metric_checkpoint_path
        .expanduser()
        .resolve()
    )

    if not base_path.is_file():
        raise FileNotFoundError(
            f"missing base temporal checkpoint: {base_path}"
        )

    if not metric_path.is_file():
        raise FileNotFoundError(
            f"missing child metric checkpoint: {metric_path}"
        )

    base_encoder = TemporalEncoder.from_checkpoint(
        base_path,
        device="cpu",
    )

    metric_digest = hashlib.sha256(
        metric_path.read_bytes()
    ).hexdigest()

    payload = torch.load(
        metric_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise RuntimeError(
            "child metric checkpoint must be a mapping"
        )

    if payload.get(
        "schema_version"
    ) != 1:
        raise RuntimeError(
            "unsupported child metric checkpoint schema"
        )

    if payload.get(
        "checkpoint_role"
    ) != "child-metric-head":
        raise RuntimeError(
            "checkpoint is not a child metric head"
        )

    checkpoint_child = str(
        payload.get(
            "child_id",
            "",
        )
    ).strip()

    if checkpoint_child != str(
        child_id
    ).strip():
        raise RuntimeError(
            "metric checkpoint child_id "
            "does not match active child"
        )

    source_digest = str(
        payload.get(
            "source_encoder_sha256",
            "",
        )
    ).lower()

    if (
        source_digest
        != base_encoder.encoder_digest
    ):
        raise RuntimeError(
            "metric checkpoint was trained from "
            "a different base temporal encoder"
        )

    if bool(
        payload.get(
            "future_session_used",
            True,
        )
    ):
        raise RuntimeError(
            "metric checkpoint reports future-session leakage"
        )

    expected_features = tuple(
        base_encoder.spec.feature_names
    )

    metric_features = tuple(
        payload.get(
            "feature_names",
            (),
        )
    )

    if (
        metric_features
        != expected_features
    ):
        raise RuntimeError(
            "metric checkpoint feature order "
            "does not match base encoder"
        )

    child_mean = np.asarray(
        payload.get(
            "child_normalization_mean",
            (),
        ),
        dtype=np.float32,
    )

    child_std = np.asarray(
        payload.get(
            "child_normalization_std",
            (),
        ),
        dtype=np.float32,
    )

    feature_count = len(
        expected_features
    )

    if (
        child_mean.shape
        != (feature_count,)
        or child_std.shape
        != (feature_count,)
        or not np.isfinite(
            child_mean
        ).all()
        or not np.isfinite(
            child_std
        ).all()
        or np.any(
            child_std <= 0
        )
    ):
        raise RuntimeError(
            "invalid child normalization in metric checkpoint"
        )

    # IMPORTANT:
    # Metric-head training used child-specific normalization.
    # Runtime must reproduce the exact same normalization.
    base_encoder.mean = (
        child_mean.copy()
    )

    base_encoder.std = (
        child_std.copy()
    )

    state = payload.get(
        "metric_head_state_dict"
    )

    if not isinstance(
        state,
        dict,
    ):
        raise RuntimeError(
            "metric checkpoint is missing metric_head_state_dict"
        )

    weight = state.get(
        "projection.weight"
    )

    if weight is None:
        raise RuntimeError(
            "metric checkpoint is missing projection.weight"
        )

    if hasattr(
        weight,
        "detach",
    ):
        weight = (
            weight.detach()
            .cpu()
            .numpy()
        )

    weight = np.asarray(
        weight,
        dtype=np.float32,
    )

    source_dim = int(
        payload.get(
            "source_embedding_dim",
            -1,
        )
    )

    projection_dim = int(
        payload.get(
            "projection_dim",
            -1,
        )
    )

    if (
        source_dim
        != base_encoder.spec.embedding_dim
    ):
        raise RuntimeError(
            "metric source embedding dimension mismatch"
        )

    if weight.shape != (
        projection_dim,
        source_dim,
    ):
        raise RuntimeError(
            "metric projection matrix shape mismatch"
        )

    threshold = float(
        payload.get(
            "validation_selected_threshold",
            0.0,
        )
    )

    if not 0 < threshold <= 2:
        raise RuntimeError(
            "metric checkpoint contains invalid threshold"
        )

    return ChildMetricEncoder(
        base_encoder=base_encoder,
        projection_weight=weight,
        child_id=checkpoint_child,
        metric_checkpoint_digest=
            metric_digest,
        candidate_distance_threshold=
            threshold,
    )
