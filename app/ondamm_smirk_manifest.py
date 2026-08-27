from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VALID_SPLITS = frozenset({"train", "val", "test"})
UNASSIGNED_SPLIT = "unassigned"
DEFAULT_MAX_IMAGES_PER_PERSON = 5_000


class ManifestError(ValueError):
    """Raised when a SMIRK training manifest violates a safety contract."""


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One accepted SMIRK training image and its leakage/provenance groups."""

    image_id: str
    image_path: str
    fan_landmarks_path: str
    mediapipe_landmarks_path: str
    person_id: str
    session_id: str
    capture_date: str
    source_video_id: str
    frame_timestamp_ms: int
    split: str = UNASSIGNED_SPLIT
    consent_training: bool = True
    approval_state: str = "approved"
    deletion_state: str = "active"
    image_sha256: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ManifestRecord":
        required = {
            "image_id",
            "image_path",
            "fan_landmarks_path",
            "mediapipe_landmarks_path",
            "person_id",
            "session_id",
            "capture_date",
            "source_video_id",
            "frame_timestamp_ms",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ManifestError(f"manifest record is missing fields: {', '.join(missing)}")
        try:
            return cls(
                image_id=str(value["image_id"]),
                image_path=str(value["image_path"]),
                fan_landmarks_path=str(value["fan_landmarks_path"]),
                mediapipe_landmarks_path=str(value["mediapipe_landmarks_path"]),
                person_id=str(value["person_id"]),
                session_id=str(value["session_id"]),
                capture_date=str(value["capture_date"]),
                source_video_id=str(value["source_video_id"]),
                frame_timestamp_ms=int(str(value["frame_timestamp_ms"])),
                split=str(value.get("split", UNASSIGNED_SPLIT)),
                consent_training=_parse_bool(value.get("consent_training", True)),
                approval_state=str(value.get("approval_state", "approved")),
                deletion_state=str(value.get("deletion_state", "active")),
                image_sha256=str(value.get("image_sha256", "")),
            )
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"invalid manifest record {value.get('image_id', '<unknown>')}: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def with_split(self, split: str) -> "ManifestRecord":
        return replace(self, split=split)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ManifestError(f"expected boolean value, got {value!r}")


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    manifest_path = Path(path)
    records: list[ManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"invalid JSON at {manifest_path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"expected JSON object at {manifest_path}:{line_number}")
            records.append(ManifestRecord.from_mapping(value))
    if not records:
        raise ManifestError(f"manifest is empty: {manifest_path}")
    return records


def load_csv(path: str | Path) -> list[ManifestRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestError(f"CSV has no header: {csv_path}")
        return [ManifestRecord.from_mapping(row) for row in reader]


def write_manifest(path: str | Path, records: Iterable[ManifestRecord]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(records)
    if not materialized:
        raise ManifestError("refusing to write an empty manifest")
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _require_text(record: ManifestRecord, field_name: str) -> None:
    if not str(getattr(record, field_name)).strip():
        raise ManifestError(f"{record.image_id or '<unknown>'}: {field_name} must not be empty")


def _validate_group_exclusivity(records: Sequence[ManifestRecord], field_name: str) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.split in VALID_SPLITS:
            group_splits[str(getattr(record, field_name))].add(record.split)
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        preview = ", ".join(leaking[:5])
        raise ManifestError(f"{field_name} appears in multiple splits: {preview}")


def _validate_person_capture_date_exclusivity(records: Sequence[ManifestRecord]) -> None:
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        if record.split in VALID_SPLITS:
            group_splits[(record.person_id, record.capture_date)].add(record.split)
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking:
        preview = ", ".join(f"{person}@{date}" for person, date in leaking[:5])
        raise ManifestError(f"person_capture_date appears in multiple splits: {preview}")


def validate_manifest(
    records: Sequence[ManifestRecord],
    *,
    split_mode: str = "global",
    max_images_per_person: int = DEFAULT_MAX_IMAGES_PER_PERSON,
    allow_unassigned: bool = False,
    require_files: bool = False,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Validate consent, accepted-image cap, paths, and split leakage.

    split_mode="global" makes people exclusive across train/val/test.
    split_mode="calibration" allows one person in multiple splits but still keeps
    each session, capture date, and source video entirely inside one split.
    """

    if split_mode not in {"global", "calibration"}:
        raise ManifestError(f"unsupported split_mode: {split_mode}")
    if max_images_per_person < 1:
        raise ManifestError("max_images_per_person must be positive")
    if not records:
        raise ManifestError("manifest must contain at least one record")

    root_path = Path(root) if root is not None else None
    image_ids: set[str] = set()
    frame_keys: set[tuple[str, int]] = set()
    person_counts: Counter[str] = Counter()

    text_fields = (
        "image_id",
        "image_path",
        "fan_landmarks_path",
        "mediapipe_landmarks_path",
        "person_id",
        "session_id",
        "capture_date",
        "source_video_id",
    )
    for record in records:
        for field_name in text_fields:
            _require_text(record, field_name)
        if record.image_id in image_ids:
            raise ManifestError(f"duplicate image_id: {record.image_id}")
        image_ids.add(record.image_id)

        frame_key = (record.source_video_id, record.frame_timestamp_ms)
        if frame_key in frame_keys:
            raise ManifestError(
                "duplicate source-video frame: "
                f"{record.source_video_id}@{record.frame_timestamp_ms}"
            )
        frame_keys.add(frame_key)

        if record.frame_timestamp_ms < 0:
            raise ManifestError(f"{record.image_id}: frame_timestamp_ms must be non-negative")
        valid_split_values = VALID_SPLITS | ({UNASSIGNED_SPLIT} if allow_unassigned else set())
        if record.split not in valid_split_values:
            raise ManifestError(f"{record.image_id}: invalid split {record.split!r}")
        if not record.consent_training:
            raise ManifestError(f"{record.image_id}: training consent is not active")
        if record.approval_state != "approved":
            raise ManifestError(f"{record.image_id}: approval_state must be 'approved'")
        if record.deletion_state != "active":
            raise ManifestError(f"{record.image_id}: deletion_state must be 'active'")

        person_counts[record.person_id] += 1
        if person_counts[record.person_id] > max_images_per_person:
            raise ManifestError(
                f"{record.person_id}: accepted-image cap exceeded "
                f"({max_images_per_person})"
            )

        if require_files:
            for field_name in ("image_path", "fan_landmarks_path", "mediapipe_landmarks_path"):
                candidate = Path(str(getattr(record, field_name)))
                if root_path is not None and not candidate.is_absolute():
                    candidate = root_path / candidate
                if not candidate.is_file():
                    raise ManifestError(f"{record.image_id}: missing {field_name}: {candidate}")

    if split_mode == "global":
        _validate_group_exclusivity(records, "person_id")
    _validate_group_exclusivity(records, "session_id")
    _validate_group_exclusivity(records, "source_video_id")
    _validate_person_capture_date_exclusivity(records)

    split_counts = Counter(record.split for record in records)
    return {
        "records": len(records),
        "people": len(person_counts),
        "per_split": dict(sorted(split_counts.items())),
        "max_images_for_one_person": max(person_counts.values()),
        "split_mode": split_mode,
    }


def assign_person_splits(
    records: Sequence[ManifestRecord],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 20260814,
) -> list[ManifestRecord]:
    """Assign every person to exactly one deterministic global-model split."""

    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1:
        raise ManifestError("val_fraction and test_fraction must be between 0 and 1")
    if val_fraction + test_fraction >= 1:
        raise ManifestError("val_fraction + test_fraction must be less than 1")

    validate_manifest(
        records,
        split_mode="calibration",
        allow_unassigned=True,
    )
    people = sorted({record.person_id for record in records})
    if len(people) < 3:
        raise ManifestError("global person split requires at least three people")

    random.Random(seed).shuffle(people)
    val_count = max(1, round(len(people) * val_fraction))
    test_count = max(1, round(len(people) * test_fraction))
    if val_count + test_count >= len(people):
        raise ManifestError("split fractions leave no people for training")

    test_people = set(people[:test_count])
    val_people = set(people[test_count : test_count + val_count])
    assignments = {
        person_id: (
            "test" if person_id in test_people else "val" if person_id in val_people else "train"
        )
        for person_id in people
    }
    assigned = [record.with_split(assignments[record.person_id]) for record in records]
    validate_manifest(assigned, split_mode="global")
    return assigned


def _print_summary(summary: Mapping[str, object]) -> None:
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build, split, and audit leakage-safe ON DAMM SMIRK manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("from-csv", help="Convert metadata CSV to JSONL")
    convert_parser.add_argument("--input", required=True)
    convert_parser.add_argument("--output", required=True)

    split_parser = subparsers.add_parser("split", help="Assign deterministic person holdout splits")
    split_parser.add_argument("--input", required=True)
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--val-fraction", type=float, default=0.15)
    split_parser.add_argument("--test-fraction", type=float, default=0.15)
    split_parser.add_argument("--seed", type=int, default=20260814)

    audit_parser = subparsers.add_parser("audit", help="Validate consent, files, and leakage")
    audit_parser.add_argument("--input", required=True)
    audit_parser.add_argument("--split-mode", choices=("global", "calibration"), default="global")
    audit_parser.add_argument("--require-files", action="store_true")
    audit_parser.add_argument("--root")
    audit_parser.add_argument("--max-images-per-person", type=int, default=DEFAULT_MAX_IMAGES_PER_PERSON)

    args = parser.parse_args(argv)
    if args.command == "from-csv":
        records = load_csv(args.input)
        validate_manifest(records, split_mode="calibration", allow_unassigned=True)
        write_manifest(args.output, records)
        _print_summary({"written": len(records), "output": args.output})
        return 0
    if args.command == "split":
        records = load_manifest(args.input)
        assigned = assign_person_splits(
            records,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        write_manifest(args.output, assigned)
        _print_summary(validate_manifest(assigned, split_mode="global"))
        return 0

    records = load_manifest(args.input)
    _print_summary(
        validate_manifest(
            records,
            split_mode=args.split_mode,
            max_images_per_person=args.max_images_per_person,
            require_files=args.require_files,
            root=args.root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
