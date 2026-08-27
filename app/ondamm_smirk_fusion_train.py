from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from app.ondamm_facial_movement import DEFAULT_MOVEMENT_RULES


VALID_SPLITS = frozenset({"train", "val", "test"})
ALLOWED_MOVEMENT_LABELS = frozenset(
    {rule.label for rule in DEFAULT_MOVEMENT_RULES} | {"open_or_uncertain"}
)
FORBIDDEN_LABEL_PARTS = (
    "emotion",
    "happy",
    "sad",
    "angry",
    "fear",
    "attention",
    "concentration",
    "preference",
    "diagnos",
    "autism",
    "asd",
    "compliance",
    "감정",
    "행복",
    "슬픔",
    "분노",
    "공포",
    "집중",
    "선호",
    "진단",
    "자폐",
    "순응",
)
SMIRK_VECTOR_FIELDS = ("expression", "eyelid", "jaw", "pose")


class FusionError(ValueError):
    """Raised when fusion data violates schema, safety, or leakage rules."""


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FusionError(f"{name} must be an object")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise FusionError(f"{name} must be numeric")
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise FusionError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise FusionError(f"{name} must be finite")
    return result


def _validate_label(value: object) -> str:
    label = str(value).strip()
    lowered = label.lower()
    if any(part in lowered for part in FORBIDDEN_LABEL_PARTS):
        raise FusionError(f"forbidden non-observable target label: {label}")
    if label not in ALLOWED_MOVEMENT_LABELS:
        raise FusionError(
            f"movement_label must be an approved observable label, got {label!r}"
        )
    return label


def _validate_group_exclusivity(rows: Sequence[Mapping[str, object]], field: str) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row[field])].add(str(row["split"]))
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        raise FusionError(f"{field} appears in multiple splits: {', '.join(leaking[:5])}")


def validate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise FusionError("fusion dataset is empty")
    required = (
        "sample_id",
        "person_id",
        "session_id",
        "source_video_id",
        "split",
        "movement_label",
        "mediapipe",
        "smirk",
    )
    sample_ids: set[str] = set()
    split_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        missing = [field for field in required if field not in row]
        if missing:
            raise FusionError(f"fusion row is missing: {', '.join(missing)}")
        for field in ("sample_id", "person_id", "session_id", "source_video_id"):
            if not str(row[field]).strip():
                raise FusionError(f"{field} must not be empty")
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise FusionError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        split = str(row["split"])
        if split not in VALID_SPLITS:
            raise FusionError(f"invalid split: {split}")
        label = _validate_label(row["movement_label"])
        _mapping(row["mediapipe"], "mediapipe")
        _mapping(row["smirk"], "smirk")
        split_counts[split] += 1
        label_counts[label] += 1

    _validate_group_exclusivity(rows, "person_id")
    _validate_group_exclusivity(rows, "session_id")
    _validate_group_exclusivity(rows, "source_video_id")
    return {
        "rows": len(rows),
        "splits": dict(sorted(split_counts.items())),
        "labels": dict(sorted(label_counts.items())),
    }


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    mediapipe_keys: tuple[str, ...]
    quality_keys: tuple[str, ...]
    smirk_dimensions: tuple[tuple[str, int], ...]

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, object]]) -> "FeatureSchema":
        if not rows:
            raise FusionError("cannot build feature schema from no rows")
        mediapipe_keys: set[str] = set()
        quality_keys: set[str] = set()
        dimensions: dict[str, int] = {}
        for row in rows:
            mediapipe = _mapping(row.get("mediapipe"), "mediapipe")
            smirk = _mapping(row.get("smirk"), "smirk")
            mediapipe_keys.update(str(key) for key in mediapipe)
            quality = _mapping(smirk.get("quality", {}), "smirk.quality")
            quality_keys.update(str(key) for key in quality)
            for field in SMIRK_VECTOR_FIELDS:
                value = smirk.get(field)
                if not isinstance(value, (list, tuple)):
                    raise FusionError(f"smirk.{field} must be an array")
                size = len(value)
                if size < 1:
                    raise FusionError(f"smirk.{field} must not be empty")
                if field in dimensions and dimensions[field] != size:
                    raise FusionError(f"inconsistent smirk.{field} dimensions")
                dimensions[field] = size
        return cls(
            mediapipe_keys=tuple(sorted(mediapipe_keys)),
            quality_keys=tuple(sorted(quality_keys)),
            smirk_dimensions=tuple((field, dimensions[field]) for field in SMIRK_VECTOR_FIELDS),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = [f"mediapipe.{key}" for key in self.mediapipe_keys]
        for field, size in self.smirk_dimensions:
            names.extend(f"smirk.{field}.{index}" for index in range(size))
        names.extend(f"smirk.quality.{key}" for key in self.quality_keys)
        return tuple(names)

    def transform(self, row: Mapping[str, object]) -> np.ndarray:
        mediapipe = _mapping(row.get("mediapipe"), "mediapipe")
        smirk = _mapping(row.get("smirk"), "smirk")
        quality = _mapping(smirk.get("quality", {}), "smirk.quality")
        values: list[float] = []
        for key in self.mediapipe_keys:
            if key not in mediapipe:
                raise FusionError(f"missing mediapipe feature: {key}")
            values.append(_finite_float(mediapipe[key], f"mediapipe.{key}"))
        for field, expected_size in self.smirk_dimensions:
            vector = smirk.get(field)
            if not isinstance(vector, (list, tuple)) or len(vector) != expected_size:
                raise FusionError(f"smirk.{field} must have length {expected_size}")
            values.extend(
                _finite_float(value, f"smirk.{field}.{index}")
                for index, value in enumerate(vector)
            )
        for key in self.quality_keys:
            if key not in quality:
                raise FusionError(f"missing smirk quality feature: {key}")
            values.append(_finite_float(quality[key], f"smirk.quality.{key}"))
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SoftmaxModel:
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        standardized = (values - self.mean) / self.scale
        logits = standardized @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray, *, abstain_threshold: float = 0.7) -> list[str]:
        if not 0 < abstain_threshold <= 1:
            raise FusionError("abstain_threshold must be in (0, 1]")
        probabilities = self.predict_proba(x)
        indices = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        return [
            self.classes[index] if score >= abstain_threshold else "abstain"
            for index, score in zip(indices, confidence)
        ]

    def save(self, path: str | Path) -> None:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "standardized-softmax-linear",
            "target_contract": "observable facial movement only",
            "feature_names": list(self.feature_names),
            "classes": list(self.classes),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
        model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SoftmaxModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(payload["feature_names"]),
            classes=tuple(payload["classes"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=np.asarray(payload["bias"], dtype=np.float64),
        )


def fit_softmax(
    x: np.ndarray,
    labels: Sequence[str],
    *,
    feature_names: Sequence[str],
    epochs: int = 500,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    seed: int = 20260814,
) -> tuple[SoftmaxModel, list[float]]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise FusionError("x must be 2D with one label per row")
    if values.shape[1] != len(feature_names):
        raise FusionError("feature_names length does not match x")
    classes = tuple(sorted({_validate_label(label) for label in labels}))
    if len(classes) < 2:
        raise FusionError("training requires at least two movement classes")
    class_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([class_index[label] for label in labels], dtype=np.int64)

    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - mean) / scale
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(values.shape[1], len(classes)))
    bias = np.zeros(len(classes), dtype=np.float64)
    history: list[float] = []

    for _ in range(epochs):
        logits = standardized @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        loss = -np.log(probabilities[np.arange(len(y)), y] + 1e-12).mean()
        loss += 0.5 * l2 * np.square(weights).sum()
        history.append(float(loss))

        gradient = probabilities
        gradient[np.arange(len(y)), y] -= 1.0
        gradient /= len(y)
        weights -= learning_rate * (standardized.T @ gradient + l2 * weights)
        bias -= learning_rate * gradient.sum(axis=0)

    return (
        SoftmaxModel(
            feature_names=tuple(feature_names),
            classes=classes,
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
        ),
        history,
    )


def load_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FusionError(f"expected JSON object at line {line_number}")
            rows.append(value)
    return rows


def _metrics(model: SoftmaxModel, x: np.ndarray, labels: Sequence[str], threshold: float):
    predictions = model.predict(x, abstain_threshold=threshold)
    accepted = [index for index, value in enumerate(predictions) if value != "abstain"]
    correct = sum(predictions[index] == labels[index] for index in accepted)
    return {
        "rows": len(labels),
        "coverage": len(accepted) / len(labels) if labels else 0.0,
        "conditional_accuracy": correct / len(accepted) if accepted else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a MediaPipe+SMIRK observable-movement fusion baseline."
    )
    parser.add_argument("--input", required=True, help="JSONL rows with train/val/test splits")
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--abstain-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args(argv)

    rows = load_rows(args.input)
    summary = validate_rows(rows)
    by_split = {split: [row for row in rows if row["split"] == split] for split in VALID_SPLITS}
    for split, split_rows in by_split.items():
        if not split_rows:
            raise FusionError(f"fusion dataset has no {split} rows")

    schema = FeatureSchema.from_rows(by_split["train"])
    arrays = {
        split: np.stack([schema.transform(row) for row in split_rows])
        for split, split_rows in by_split.items()
    }
    labels = {
        split: [str(row["movement_label"]) for row in split_rows]
        for split, split_rows in by_split.items()
    }
    model, history = fit_softmax(
        arrays["train"],
        labels["train"],
        feature_names=schema.feature_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    model.save(args.output_model)
    metrics = {
        "dataset": summary,
        "training_loss_first": history[0],
        "training_loss_last": history[-1],
        "abstain_threshold": args.abstain_threshold,
        "per_split": {
            split: _metrics(model, arrays[split], labels[split], args.abstain_threshold)
            for split in ("train", "val", "test")
        },
        "notice": "Observable movement only; not emotion, attention, preference, compliance, autism, or diagnosis.",
    }
    metrics_path = Path(args.output_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
