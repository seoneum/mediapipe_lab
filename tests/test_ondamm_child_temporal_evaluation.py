from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ondamm_child_temporal_evaluation import (  # noqa: E402
    GroundTruthEvent,
    TemporalDetection,
    evaluate_future_session,
)


CHILD = "child-a"
SESSION = "s03"


def truth(
    event_id: str,
    pattern_id: str,
    start: float,
    *,
    known: bool,
) -> GroundTruthEvent:
    return GroundTruthEvent(
        child_id=CHILD,
        session_id=SESSION,
        event_id=event_id,
        pattern_id=pattern_id,
        start_timestamp=start,
        end_timestamp=start + 1.0,
        known_at_session_start=known,
    )


def detection(
    detection_id: str,
    start: float,
    lifecycle: str,
    *,
    pattern_id: str | None = None,
    candidate_id: str | None = None,
    occurrence_count: int = 0,
    eventized_timestamp: float | None = None,
) -> TemporalDetection:
    return TemporalDetection(
        child_id=CHILD,
        session_id=SESSION,
        detection_id=detection_id,
        start_timestamp=start,
        end_timestamp=start + 1.0,
        lifecycle=lifecycle,
        pattern_id=pattern_id,
        candidate_id=candidate_id,
        occurrence_count=occurrence_count,
        eventized_timestamp=eventized_timestamp,
    )


class ChildTemporalEvaluationTests(unittest.TestCase):
    def test_perfect_known_and_unknown_future_session_metrics(self) -> None:
        ground_truth = [
            truth("known-1", "known-wave", 10.0, known=True),
            truth("unknown-1", "unknown-tap", 20.0, known=False),
            truth("unknown-2", "unknown-tap", 30.0, known=False),
            truth("unknown-3", "unknown-tap", 40.0, known=False),
        ]
        detections = [
            detection(
                "known-detection",
                10.0,
                "KNOWN_OCCURRENCE",
                pattern_id="known-wave",
            ),
            detection(
                "unknown-detection-1",
                20.0,
                "UNKNOWN_OCCURRENCE",
                candidate_id="candidate-tap",
                occurrence_count=1,
            ),
            detection(
                "unknown-detection-2",
                30.0,
                "UNKNOWN_OCCURRENCE",
                candidate_id="candidate-tap",
                occurrence_count=2,
            ),
            detection(
                "unknown-detection-3",
                40.0,
                "REPEATING_CANDIDATE",
                candidate_id="candidate-tap",
                occurrence_count=3,
                eventized_timestamp=41.0,
            ),
            detection("false-activation", 80.0, "UNKNOWN_OCCURRENCE"),
        ]

        metrics = evaluate_future_session(
            ground_truth,
            detections,
            child_id=CHILD,
            future_session=SESSION,
            session_duration_seconds=120.0,
        )

        self.assertEqual(metrics["objective"], "within-child-future-session")
        self.assertEqual(metrics["known_pattern_event_recall"], 1.0)
        self.assertEqual(metrics["known_pattern_precision"], 1.0)
        self.assertEqual(metrics["false_activations_per_min"], 0.5)
        self.assertEqual(metrics["unknown_repeated_pattern_discovery_precision"], 1.0)
        self.assertEqual(metrics["duplicate_cluster_rate"], 0.0)
        self.assertEqual(metrics["false_merge_rate"], 0.0)
        self.assertEqual(metrics["occurrences_required_until_discovery_mean"], 3.0)
        self.assertEqual(
            metrics["first_observation_to_eventization_latency_seconds_mean"],
            21.0,
        )
        self.assertEqual(metrics["future_session_stability"], 1.0)

    def test_duplicate_clusters_and_false_merges_are_reported(self) -> None:
        ground_truth = [
            truth("a-1", "unknown-a", 10.0, known=False),
            truth("a-2", "unknown-a", 20.0, known=False),
            truth("a-3", "unknown-a", 30.0, known=False),
            truth("a-4", "unknown-a", 35.0, known=False),
            truth("a-5", "unknown-a", 37.0, known=False),
            truth("b-1", "unknown-b", 40.0, known=False),
            truth("b-2", "unknown-b", 50.0, known=False),
            truth("b-3", "unknown-b", 60.0, known=False),
        ]
        detections = [
            detection("a-c1-1", 10.0, "UNKNOWN_OCCURRENCE", candidate_id="c1", occurrence_count=1),
            detection("a-c1-2", 20.0, "UNKNOWN_OCCURRENCE", candidate_id="c1", occurrence_count=2),
            detection("a-c1-3", 30.0, "REPEATING_CANDIDATE", candidate_id="c1", occurrence_count=3),
            detection("a-c2-1", 35.0, "REPEATING_CANDIDATE", candidate_id="c2", occurrence_count=3),
            detection("merge-a", 37.0, "REPEATING_CANDIDATE", candidate_id="merge", occurrence_count=3),
            detection("merge-b", 40.0, "REPEATING_CANDIDATE", candidate_id="merge", occurrence_count=4),
        ]

        metrics = evaluate_future_session(
            ground_truth,
            detections,
            child_id=CHILD,
            future_session=SESSION,
            session_duration_seconds=120.0,
        )

        self.assertAlmostEqual(metrics["duplicate_cluster_rate"], 0.5)
        self.assertAlmostEqual(metrics["false_merge_rate"], 1 / 3)
        self.assertAlmostEqual(
            metrics["unknown_repeated_pattern_discovery_precision"],
            2 / 3,
        )

    def test_rejects_cross_child_or_non_future_session_rows(self) -> None:
        wrong_child = GroundTruthEvent(
            child_id="child-b",
            session_id=SESSION,
            event_id="event",
            pattern_id="pattern",
            start_timestamp=0.0,
            end_timestamp=1.0,
            known_at_session_start=True,
        )
        with self.assertRaisesRegex(ValueError, "target child"):
            evaluate_future_session(
                [wrong_child],
                [],
                child_id=CHILD,
                future_session=SESSION,
                session_duration_seconds=60.0,
            )


if __name__ == "__main__":
    unittest.main()
