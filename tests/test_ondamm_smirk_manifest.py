from __future__ import annotations

import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from app.ondamm_smirk_manifest import (
    ManifestError,
    ManifestRecord,
    assign_person_splits,
    load_manifest,
    validate_manifest,
    write_manifest,
)


class ONDAMMSMIRKManifestTests(unittest.TestCase):
    def make_records(
        self,
        *,
        people: int = 5,
        images_per_person: int = 2,
        split: str = "unassigned",
    ) -> list[ManifestRecord]:
        records: list[ManifestRecord] = []
        for person_index in range(people):
            person_id = f"person-{person_index:02d}"
            for image_index in range(images_per_person):
                session_id = f"{person_id}-session-{image_index:02d}"
                source_video_id = f"{session_id}-video"
                records.append(
                    ManifestRecord(
                        image_id=f"{person_id}-image-{image_index:04d}",
                        image_path=f"images/{person_id}/{image_index:04d}.jpg",
                        fan_landmarks_path=f"fan/{person_id}/{image_index:04d}.npy",
                        mediapipe_landmarks_path=f"mediapipe/{person_id}/{image_index:04d}.npy",
                        person_id=person_id,
                        session_id=session_id,
                        capture_date=f"2026-08-{person_index + 1:02d}",
                        source_video_id=source_video_id,
                        frame_timestamp_ms=image_index * 1000,
                        split=split,
                        consent_training=True,
                        approval_state="approved",
                        deletion_state="active",
                        image_sha256=f"digest-{person_index}-{image_index}",
                    )
                )
        return records

    def test_assign_person_splits_is_deterministic_and_person_exclusive(self) -> None:
        records = self.make_records(people=5, images_per_person=3)

        first = assign_person_splits(records, val_fraction=0.2, test_fraction=0.2, seed=17)
        second = assign_person_splits(records, val_fraction=0.2, test_fraction=0.2, seed=17)

        self.assertEqual(first, second)
        by_person: dict[str, set[str]] = defaultdict(set)
        for record in first:
            by_person[record.person_id].add(record.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_person.values()))
        self.assertEqual({"train", "val", "test"}, {record.split for record in first})
        validate_manifest(first, split_mode="global")

    def test_global_validation_rejects_person_leakage(self) -> None:
        records = self.make_records(people=3, images_per_person=1, split="train")
        leaked = records[0].with_split("test")
        duplicate_person = ManifestRecord(
            **{
                **records[0].to_dict(),
                "image_id": "leaked-image",
                "session_id": "leaked-session",
                "source_video_id": "leaked-video",
                "frame_timestamp_ms": 9999,
                "split": "train",
            }
        )

        with self.assertRaisesRegex(ManifestError, "person_id.*multiple splits"):
            validate_manifest([leaked, duplicate_person, *records[1:]], split_mode="global")

    def test_validation_rejects_source_video_leakage_in_calibration_mode(self) -> None:
        records = self.make_records(people=1, images_per_person=2, split="train")
        second = ManifestRecord(
            **{
                **records[1].to_dict(),
                "source_video_id": records[0].source_video_id,
                "split": "test",
            }
        )

        with self.assertRaisesRegex(ManifestError, "source_video_id.*multiple splits"):
            validate_manifest([records[0], second], split_mode="calibration")

    def test_same_capture_date_is_allowed_for_different_people(self) -> None:
        records = self.make_records(people=2, images_per_person=1, split="train")
        second = ManifestRecord(
            **{
                **records[1].to_dict(),
                "capture_date": records[0].capture_date,
                "split": "test",
            }
        )

        validate_manifest([records[0], second], split_mode="global")

    def test_same_person_capture_date_cannot_cross_calibration_splits(self) -> None:
        records = self.make_records(people=1, images_per_person=2, split="train")
        second = ManifestRecord(
            **{
                **records[1].to_dict(),
                "capture_date": records[0].capture_date,
                "split": "test",
            }
        )

        with self.assertRaisesRegex(ManifestError, "person_capture_date.*multiple splits"):
            validate_manifest([records[0], second], split_mode="calibration")

    def test_validation_rejects_unapproved_or_withdrawn_samples(self) -> None:
        record = self.make_records(people=1, images_per_person=1, split="train")[0]
        unapproved = ManifestRecord(**{**record.to_dict(), "approval_state": "pending"})
        withdrawn = ManifestRecord(**{**record.to_dict(), "deletion_state": "withdrawn"})
        no_consent = ManifestRecord(**{**record.to_dict(), "consent_training": False})

        for invalid in (unapproved, withdrawn, no_consent):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ManifestError):
                    validate_manifest([invalid], split_mode="calibration")

    def test_validation_enforces_per_person_accepted_image_cap(self) -> None:
        records = self.make_records(people=1, images_per_person=4, split="train")

        with self.assertRaisesRegex(ManifestError, "accepted-image cap"):
            validate_manifest(records, split_mode="calibration", max_images_per_person=3)

    def test_manifest_jsonl_round_trip(self) -> None:
        records = assign_person_splits(
            self.make_records(people=3, images_per_person=2),
            val_fraction=0.2,
            test_fraction=0.2,
            seed=3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            write_manifest(path, records)

            loaded = load_manifest(path)

        self.assertEqual(records, loaded)
        for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
            json.loads(line)


if __name__ == "__main__":
    unittest.main()
