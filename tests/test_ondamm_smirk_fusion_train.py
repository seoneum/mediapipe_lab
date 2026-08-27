from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.ondamm_smirk_fusion_train import (
    FusionError,
    FeatureSchema,
    SoftmaxModel,
    fit_softmax,
    validate_rows,
)


class ONDAMMSMIRKFusionTests(unittest.TestCase):
    def row(
        self,
        *,
        sample_id: str,
        person_id: str,
        split: str,
        label: str,
        blink: float,
        eyelid: float,
    ) -> dict[str, object]:
        return {
            "sample_id": sample_id,
            "person_id": person_id,
            "session_id": f"session-{person_id}",
            "source_video_id": f"video-{person_id}",
            "split": split,
            "movement_label": label,
            "mediapipe": {"eyeBlinkLeft": blink, "eyeBlinkRight": blink},
            "smirk": {
                "expression": [blink, 1.0 - blink],
                "eyelid": [eyelid, eyelid],
                "jaw": [0.0, 0.0, 0.0],
                "pose": [0.0, 0.0, 0.0],
                "quality": {"face_score": 1.0},
            },
        }

    def test_feature_schema_has_namespaced_deterministic_features(self) -> None:
        row = self.row(
            sample_id="a",
            person_id="p-train",
            split="train",
            label="eyes_closed",
            blink=0.9,
            eyelid=0.8,
        )

        schema = FeatureSchema.from_rows([row])
        vector = schema.transform(row)

        self.assertEqual(len(schema.feature_names), vector.shape[0])
        self.assertEqual("mediapipe.eyeBlinkLeft", schema.feature_names[0])
        self.assertIn("smirk.expression.0", schema.feature_names)
        self.assertIn("smirk.quality.face_score", schema.feature_names)
        self.assertTrue(np.isfinite(vector).all())

    def test_forbidden_or_non_observable_label_is_rejected(self) -> None:
        for label in ("happy", "attention_high", "autism", "진단"):
            with self.subTest(label=label):
                row = self.row(
                    sample_id="bad",
                    person_id="p1",
                    split="train",
                    label=label,
                    blink=0.1,
                    eyelid=0.1,
                )
                with self.assertRaises(FusionError):
                    validate_rows([row])

    def test_person_leakage_between_splits_is_rejected(self) -> None:
        train = self.row(
            sample_id="train",
            person_id="same-person",
            split="train",
            label="eyes_closed",
            blink=0.9,
            eyelid=0.8,
        )
        test = self.row(
            sample_id="test",
            person_id="same-person",
            split="test",
            label="open_or_uncertain",
            blink=0.1,
            eyelid=0.1,
        )
        test["session_id"] = "later-session"
        test["source_video_id"] = "later-video"

        with self.assertRaisesRegex(FusionError, "person_id.*multiple splits"):
            validate_rows([train, test])

    def test_softmax_training_learns_observable_movement_and_abstains(self) -> None:
        rows = []
        for index, blink in enumerate((0.0, 0.1, 0.2, 0.8, 0.9, 1.0)):
            label = "open_or_uncertain" if blink < 0.5 else "eyes_closed"
            rows.append(
                self.row(
                    sample_id=f"train-{index}",
                    person_id=f"train-person-{index}",
                    split="train",
                    label=label,
                    blink=blink,
                    eyelid=blink,
                )
            )
        schema = FeatureSchema.from_rows(rows)
        x = np.stack([schema.transform(row) for row in rows])
        labels = [str(row["movement_label"]) for row in rows]

        model, history = fit_softmax(
            x,
            labels,
            feature_names=schema.feature_names,
            epochs=400,
            learning_rate=0.1,
            l2=1e-4,
            seed=7,
        )

        self.assertLess(history[-1], history[0])
        predictions = model.predict(x, abstain_threshold=0.6)
        self.assertEqual(labels, predictions)
        uncertain = np.zeros((1, x.shape[1]), dtype=np.float64)
        self.assertEqual(["abstain"], model.predict(uncertain, abstain_threshold=1.0))

    def test_model_json_round_trip(self) -> None:
        model = SoftmaxModel(
            feature_names=("mediapipe.eyeBlinkLeft",),
            classes=("eyes_closed", "open_or_uncertain"),
            mean=np.array([0.5]),
            scale=np.array([0.2]),
            weights=np.array([[1.0, -1.0]]),
            bias=np.array([0.0, 0.0]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)
            loaded = SoftmaxModel.load(path)

        self.assertEqual(model.feature_names, loaded.feature_names)
        np.testing.assert_allclose(model.weights, loaded.weights)


if __name__ == "__main__":
    unittest.main()
