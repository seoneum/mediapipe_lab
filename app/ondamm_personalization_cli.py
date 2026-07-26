"""Deterministic JSON CLI for the offline personalization model.

The command accepts only event summaries and teacher labels.  It deliberately
has no camera, media, network, UI, or diagnostic inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from app.ondamm_personalization import (
        EventFeatureRow,
        InsufficientEvidenceError,
        PersonalizationModel,
        TeacherLabel,
        generate_recommendations,
    )
except ModuleNotFoundError:  # direct ``python app/..._cli.py`` invocation
    from ondamm_personalization import (
        EventFeatureRow,
        InsufficientEvidenceError,
        PersonalizationModel,
        TeacherLabel,
        generate_recommendations,
    )

SCHEMA_VERSION = 1
MODEL_CONFIG_VERSION = "centroid-baseline-v1"
NON_DIAGNOSTIC_NOTICE = (
    "Offline teacher-approved support hint only; this is non-diagnostic and does not infer emotion, attention, preference, or condition."
)


def _load_array(path: Path, *, name: str) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read {name} JSON: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {name} JSON: {exc.msg}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{name} JSON root must be an array")
    return payload


def _parse_row(payload: Any) -> EventFeatureRow:
    if not isinstance(payload, Mapping):
        raise ValueError("each row must be an event summary object")
    # Both representations are public core APIs.  Summaries are the normal
    # input; serialized feature rows remain useful for deterministic exports.
    if "features" in payload:
        return EventFeatureRow.from_dict(payload)
    return EventFeatureRow.from_summary(payload)


def _parse_rows(payloads: Sequence[Any]) -> list[EventFeatureRow]:
    rows = [_parse_row(payload) for payload in payloads]
    event_ids = [row.event_id for row in rows]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("rows contain duplicate event_id values")
    return rows


def _parse_labels(payloads: Sequence[Any]) -> list[TeacherLabel]:
    labels = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("each label must be a label object")
        labels.append(TeacherLabel.from_dict(payload))
    event_ids = [label.event_id for label in labels]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("labels contain duplicate event_id values")
    return labels


def _demo_inputs(person_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    labels = []
    for index in range(1, 4):
        event_id = f"demo-{index:02d}"
        rows.append(
            {
                "person_id": person_id,
                "event_id": event_id,
                "event_type": "task_start",
                "duration_seconds": 10.0,
                "quality_flags": {
                    "quality_ok": True,
                    "quality_complete": True,
                    "teacher_reviewed": True,
                },
                "quality_score": 0.9,
                "zone_transition": "none",
                "teacher_context": "visual_schedule",
            }
        )
        labels.append(
            {
                "schema_version": 1,
                "event_id": event_id,
                "support_label": "visual_schedule",
                "outcome_label": "helpful",
                "teacher_approved": True,
            }
        )
    return rows, labels


def _eligible_rows(rows: Sequence[EventFeatureRow], labels: Sequence[TeacherLabel], person_id: str) -> list[EventFeatureRow]:
    by_event = {label.event_id: label for label in labels}
    return sorted(
        (
            row
            for row in rows
            if row.person_id == person_id
            and (label := by_event.get(row.event_id)) is not None
            and label.teacher_approved is True
            and label.outcome_label in {"helpful", "effective"}
            and row.quality_ok
            and row.quality_complete
            and row.teacher_reviewed
            and row.quality_score >= 0.5
        ),
        key=lambda row: row.event_id,
    )


def _abstention_reason(
    rows: Sequence[EventFeatureRow], labels: Sequence[TeacherLabel], person_id: str, eligible_count: int, min_samples: int
) -> str:
    target_rows = [row for row in rows if row.person_id == person_id]
    if not target_rows:
        return "mismatched_person"
    by_event = {label.event_id: label for label in labels}
    matched = [row for row in target_rows if row.event_id in by_event]
    if not matched:
        return "mismatched_labels"
    if not any(by_event[row.event_id].teacher_approved for row in matched):
        return "unapproved"
    if not any(
        by_event[row.event_id].teacher_approved is True
        and by_event[row.event_id].outcome_label in {"helpful", "effective"}
        for row in matched
    ):
        return "insufficient_positive_evidence"
    if not any(
        by_event[row.event_id].teacher_approved is True
        and by_event[row.event_id].outcome_label in {"helpful", "effective"}
        and row.quality_ok
        and row.quality_complete
        and row.teacher_reviewed
        and row.quality_score >= 0.5
        for row in matched
    ):
        return "low_quality"
    if eligible_count < min_samples:
        return "sparse"
    return "low_confidence"


def run(
    *,
    rows: Sequence[EventFeatureRow],
    labels: Sequence[TeacherLabel],
    person_id: str,
    min_samples: int,
    min_confidence: float,
    recommend: bool,
    synthetic: bool = False,
) -> dict[str, Any]:
    if not isinstance(person_id, str) or not person_id.strip():
        raise ValueError("person_id must be a non-empty string")
    person_id = person_id.strip()
    if any(not isinstance(row, EventFeatureRow) for row in rows):
        raise ValueError("rows must contain EventFeatureRow values")
    if any(not isinstance(label, TeacherLabel) for label in labels):
        raise ValueError("labels must contain TeacherLabel values")
    label_ids = [label.event_id for label in labels]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("labels contain duplicate event_id values")
    row_ids = [row.event_id for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("rows contain duplicate event_id values")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)) or not 0.0 < min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    min_confidence = float(min_confidence)
    eligible = _eligible_rows(rows, labels, person_id)
    positive_event_ids = {
        label.event_id
        for label in labels
        if label.teacher_approved is True and label.outcome_label in {"helpful", "effective"}
    }
    prediction_rows = sorted(
        (row for row in eligible if row.event_id in positive_event_ids),
        key=lambda row: row.event_id,
    )
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_config_version": MODEL_CONFIG_VERSION,
        "person_id": person_id,
        "synthetic": synthetic,
        "sample_count": len(eligible),
        "min_samples": min_samples,
        "min_confidence": min_confidence,
        "confidence": None,
        "abstained": True,
        "abstention_reason": None,
        "support_label": None,
        "evidence_ids": [],
        "recommendations": [],
        "approved_manifest_digest": None,
        "non_diagnostic_notice": NON_DIAGNOSTIC_NOTICE,
    }
    if not eligible:
        base["abstention_reason"] = _abstention_reason(rows, labels, person_id, 0, min_samples)
        return base
    try:
        model = PersonalizationModel.fit(
            rows,
            labels,
            target_person_id=person_id,
            min_samples=min_samples,
            min_confidence=min_confidence,
        )
    except InsufficientEvidenceError:
        base["abstention_reason"] = _abstention_reason(rows, labels, person_id, len(eligible), min_samples)
        return base

    base["sample_count"] = model.sample_count
    base["approved_manifest_digest"] = model.approved_manifest_digest
    predictions = tuple(
        prediction
        for row in prediction_rows
        if (prediction := model.predict(row)) is not None
    )
    if not predictions:
        base["abstention_reason"] = "low_confidence"
        return base

    predictions_by_label: dict[str, list[Any]] = {}
    for prediction in predictions:
        predictions_by_label.setdefault(prediction.support_label, []).append(prediction)
    selected_label = min(
        predictions_by_label,
        key=lambda label: (
            -len(predictions_by_label[label]),
            -sum(item.confidence for item in predictions_by_label[label]) / len(predictions_by_label[label]),
            label,
        ),
    )
    selected_predictions = predictions_by_label[selected_label]
    confidence = round(
        sum(item.confidence for item in selected_predictions) / len(selected_predictions),
        6,
    )
    evidence_ids = sorted(
        {
            evidence_id
            for prediction in selected_predictions
            for evidence_id in prediction.evidence_ids
        }
    )
    base.update(
        {
            "confidence": confidence,
            "abstained": False,
            "abstention_reason": None,
            "support_label": selected_label,
            "evidence_ids": evidence_ids,
        }
    )
    if recommend:
        recommendations = generate_recommendations(
            max(
                selected_predictions,
                key=lambda item: (item.confidence, item.sample_count, item.evidence_ids[0]),
            ),
            teacher_approved=True,
            evidence_ids=evidence_ids,
        )
        base["recommendations"] = [
            {
                "support_label": item.support_label,
                "hint": item.hint,
                "teacher_approved": item.teacher_approved,
                "evidence_ids": sorted(item.evidence_ids),
                "provenance": item.provenance,
            }
            for item in recommendations
        ]
    return base


def _dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline teacher-approved personalization demo")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="use deterministic synthetic, camera-free rows")
    source.add_argument("--rows-json", type=Path, help="JSON array of event summaries")
    parser.add_argument("--labels-json", type=Path, help="JSON array of teacher labels")
    parser.add_argument("--person-id", default="demo-person")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--recommend", "--teacher-approved", dest="recommend", action="store_true")
    args = parser.parse_args(argv)

    if not args.demo and args.labels_json is None:
        parser.error("--labels-json is required with --rows-json")
    if args.demo and args.labels_json is not None:
        parser.error("--labels-json cannot be used with --demo")
    try:
        if args.demo:
            row_payloads, label_payloads = _demo_inputs(args.person_id)
        else:
            row_payloads = _load_array(args.rows_json, name="rows")
            label_payloads = _load_array(args.labels_json, name="labels")
        rows = _parse_rows(row_payloads)
        labels = _parse_labels(label_payloads)
        payload = run(
            rows=rows,
            labels=labels,
            person_id=args.person_id,
            min_samples=args.min_samples,
            min_confidence=args.min_confidence,
            recommend=args.recommend,
            synthetic=args.demo,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    encoded = _dump(payload)
    print(encoded)
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write output: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
