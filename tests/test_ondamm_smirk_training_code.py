from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.ondamm_smirk_manifest import ManifestRecord, write_manifest
from smirk_ondamm.dataset import make_ondamm_dataset_class, records_for_split
from smirk_ondamm.export_features import encoder_state_dict, serialize_encoder_outputs
from smirk_ondamm.train_ondamm import validate_smirk_checkout


class FakeBaseDataset:
    def __init__(self, data_list, config, test=False):
        self.data_list = data_list
        self.config = config
        self.test = test

    def prepare_data(self, *, image, landmarks_fan, landmarks_mediapipe):
        return {
            "image": image,
            "fan": landmarks_fan,
            "mediapipe": landmarks_mediapipe,
        }


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class ONDAMMSMIRKTrainingCodeTests(unittest.TestCase):
    def make_record(self, image_id: str, split: str) -> ManifestRecord:
        return ManifestRecord(
            image_id=image_id,
            image_path=f"images/{image_id}.jpg",
            fan_landmarks_path=f"fan/{image_id}.npy",
            mediapipe_landmarks_path=f"mediapipe/{image_id}.npy",
            person_id=f"person-{split}",
            session_id=f"session-{split}",
            capture_date=f"2026-08-{1 if split == 'train' else 2:02d}",
            source_video_id=f"video-{split}",
            frame_timestamp_ms=0,
            split=split,
            image_sha256=f"sha-{image_id}",
        )

    def test_records_for_split_loads_only_requested_split(self) -> None:
        records = [self.make_record("train-a", "train"), self.make_record("val-a", "val")]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            write_manifest(path, records)

            selected = records_for_split(path, "train")

        self.assertEqual(["train-a"], [record.image_id for record in selected])

    def test_records_for_split_rejects_empty_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            write_manifest(path, [self.make_record("train-a", "train")])

            with self.assertRaisesRegex(ValueError, "no records for split"):
                records_for_split(path, "test")

    def test_dataset_class_preserves_manifest_paths_and_test_mode(self) -> None:
        dataset_class = make_ondamm_dataset_class(FakeBaseDataset)
        record = self.make_record("sample-a", "val")
        config = SimpleNamespace(dataset=SimpleNamespace(root="/srv/ondamm"))

        dataset = dataset_class([record], config, test=True)

        self.assertEqual("ONDAMM", dataset.name)
        self.assertTrue(dataset.test)
        self.assertEqual(record, dataset.data_list[0])

    def test_encoder_state_dict_removes_smirk_encoder_prefix(self) -> None:
        checkpoint = {
            "smirk_encoder.pose.weight": "pose",
            "smirk_encoder.expression.weight": "expression",
            "smirk_generator.weight": "generator",
        }

        selected = encoder_state_dict(checkpoint)

        self.assertEqual(
            {"pose.weight": "pose", "expression.weight": "expression"},
            selected,
        )

    def test_serialize_encoder_outputs_keeps_measurement_parameters(self) -> None:
        outputs = {
            "expression_params": FakeTensor([[0.1, 0.2]]),
            "eyelid_params": FakeTensor([[0.3, 0.4]]),
            "jaw_params": FakeTensor([[0.5, 0.0, 0.0]]),
            "pose_params": FakeTensor([[0.1, 0.0, -0.1]]),
            "cam": FakeTensor([[7.0, 0.0, 0.0]]),
            "shape_params": FakeTensor([[0.01, 0.02]]),
            "ignored": FakeTensor([[99]]),
        }

        payload = serialize_encoder_outputs(outputs)

        self.assertEqual([0.1, 0.2], payload["expression_params"])
        self.assertEqual([0.3, 0.4], payload["eyelid_params"])
        self.assertNotIn("ignored", payload)

    def test_validate_smirk_checkout_requires_training_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(FileNotFoundError, "SMIRK checkout is incomplete"):
                validate_smirk_checkout(root)

            for relative in (
                "src/smirk_trainer.py",
                "datasets/base_dataset.py",
                "configs/config_train.yaml",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# fixture\n", encoding="utf-8")

            validate_smirk_checkout(root)


if __name__ == "__main__":
    unittest.main()
