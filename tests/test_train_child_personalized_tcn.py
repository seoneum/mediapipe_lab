from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
SCRIPTS_DIR = ROOT / "scripts"
for directory in (APP_DIR, SCRIPTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import train_v4_tcn as tcn  # noqa: E402
from ondamm_temporal_encoder import (  # noqa: E402
    TemporalEncoder,
    TemporalEncoderSpec,
    build_torch_encoder,
    export_temporal_encoder_checkpoint,
)
from train_child_personalized_tcn import (  # noqa: E402
    fit_child_model,
    initialized_child_model,
    select_fine_tuning_epoch,
    validate_session_partition,
)


def product_feature_names() -> tuple[str, ...]:
    return tuple(
        [f"bs_fixture_{index:02d}" for index in range(52)]
        + [f"geom_abs_fixture_{index:02d}" for index in range(18)]
        + [f"motion_fixture_{index:02d}" for index in range(9)]
    )


class ChildPersonalizedTcnTests(unittest.TestCase):
    def test_future_session_cannot_leak_into_training(self) -> None:
        with self.assertRaisesRegex(ValueError, "held out"):
            validate_session_partition("child-a", ["s01", "s02"], "s02")
        partition = validate_session_partition("child-a", ["s01", "s02"], "s03")
        self.assertEqual(partition.train_sessions, ("s01", "s02"))
        self.assertEqual(partition.future_session, "s03")

    def test_child_model_copies_all_pretrained_tcn_weights(self) -> None:
        spec = TemporalEncoderSpec(
            feature_names=("bs_a", "geom_abs_b", "motion_c"),
            sequence_length=4,
            stride_frames=1,
            channels=(4,),
            kernel_size=3,
            dropout=0.2,
            embedding_dim=4,
        )
        pretrained = build_torch_encoder(spec)
        for value in pretrained.state_dict().values():
            value.fill_(0.125)
        child = initialized_child_model(
            spec,
            pretrained.state_dict(),
            device=torch.device("cpu"),
        )
        for key, value in child.state_dict().items():
            if key.startswith("tcn."):
                self.assertTrue(torch.equal(value, pretrained.state_dict()[key]))
        self.assertIn("head.weight", child.state_dict())

    def test_child_fine_tuning_runs_from_pretrained_initialization(self) -> None:
        spec = TemporalEncoderSpec(
            feature_names=("bs_a", "geom_abs_b", "motion_c"),
            sequence_length=4,
            stride_frames=1,
            channels=(4,),
            kernel_size=3,
            dropout=0.0,
            embedding_dim=4,
        )
        pretrained = build_torch_encoder(spec)
        rng = np.random.default_rng(7)
        pack = tcn.SequencePack(
            X=rng.normal(size=(12, 3, 4)).astype(np.float32),
            y=np.asarray([0, 1] * 6, dtype=np.int64),
            meta=pd.DataFrame(),
        )
        args = SimpleNamespace(
            seed=3,
            batch_size=4,
            lr=1e-3,
            weight_decay=0.0,
            epochs=2,
            patience=2,
            min_delta=0.0,
        )
        best_epoch, selection_history = select_fine_tuning_epoch(
            spec,
            pretrained.state_dict(),
            pack,
            pack,
            args=args,
            device=torch.device("cpu"),
        )
        model, history = fit_child_model(
            spec,
            pretrained.state_dict(),
            pack,
            epochs=best_epoch,
            args=args,
            device=torch.device("cpu"),
        )
        self.assertGreaterEqual(best_epoch, 1)
        self.assertFalse(selection_history.empty)
        self.assertEqual(len(history), best_epoch)
        self.assertEqual(model(torch.from_numpy(pack.X[:1])).shape, (1,))

    def test_runtime_accepts_child_personalized_checkpoint_contract(self) -> None:
        spec = TemporalEncoderSpec(
            feature_names=product_feature_names(),
            sequence_length=60,
            stride_frames=5,
            channels=(64, 64, 64),
            kernel_size=3,
            dropout=0.2,
            embedding_dim=64,
        )
        model = build_torch_encoder(spec)
        with tempfile.TemporaryDirectory(prefix="ondamm-child-checkpoint-") as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "encoder_child_personalized.pt"
            digest = export_temporal_encoder_checkpoint(
                checkpoint,
                spec=spec,
                model_state_dict=model.state_dict(),
                normalization_mean=np.zeros(79),
                normalization_std=np.ones(79),
                metadata={
                    "checkpoint_role": "child-personalized",
                    "child_id": "child-a",
                    "train_participants": ["child-a"],
                    "training_sessions": ["s01", "s02"],
                    "future_session_excluded_from_training": "s03",
                    "best_epoch": 2,
                    "normalization": "target-child historical calibration",
                },
            )
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "features": list(spec.feature_names),
                        "feature_counts": {
                            "blendshape": 52,
                            "geometry": 18,
                            "motion": 9,
                            "total": 79,
                        },
                        "encoder_checkpoints": {
                            "child_personalized": {
                                "path": str(checkpoint),
                                "sha256": digest,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            encoder = TemporalEncoder.from_checkpoint(checkpoint)
        self.assertEqual(encoder.spec.feature_names, spec.feature_names)

    def test_runtime_rejects_future_session_leakage_in_checkpoint(self) -> None:
        spec = TemporalEncoderSpec(
            feature_names=product_feature_names(),
            sequence_length=60,
            stride_frames=5,
            channels=(64, 64, 64),
            kernel_size=3,
            dropout=0.2,
            embedding_dim=64,
        )
        model = build_torch_encoder(spec)
        with tempfile.TemporaryDirectory(prefix="ondamm-child-leak-") as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "encoder_child_personalized.pt"
            digest = export_temporal_encoder_checkpoint(
                checkpoint,
                spec=spec,
                model_state_dict=model.state_dict(),
                normalization_mean=np.zeros(79),
                normalization_std=np.ones(79),
                metadata={
                    "checkpoint_role": "child-personalized",
                    "child_id": "child-a",
                    "train_participants": ["child-a"],
                    "training_sessions": ["s01", "s03"],
                    "future_session_excluded_from_training": "s03",
                    "best_epoch": 2,
                    "normalization": "target-child historical calibration",
                },
            )
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "features": list(spec.feature_names),
                        "feature_counts": {
                            "blendshape": 52,
                            "geometry": 18,
                            "motion": 9,
                            "total": 79,
                        },
                        "encoder_checkpoints": {
                            "child_personalized": {
                                "path": str(checkpoint),
                                "sha256": digest,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "leaked"):
                TemporalEncoder.from_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
