import hashlib
import tempfile
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock
from dataclasses import replace

from app.ondamm_face_event_learning import (
    NOTICE, MAX_EVENT_SPAN_SECONDS, MAX_CONTAINER_BREADTH, MAX_GRAPH_NODES, MAX_INPUT_BYTES, MAX_SAFE_COLLECTION,
    EventCandidate, LabelRecord, ObservationSample, PrototypeModel,
    dumps, extract_event_candidates, main, match_reviewable_candidates, train_model, _check_forbidden,
    loads_sample, loads_candidate, loads_label, loads_model,
)


class FaceEventLearningTests(unittest.TestCase):
    def sample(self, i, *, person="p", quality=.95, movement=None, gaze="center", dwell=.5, head=0, flags=()):
        return ObservationSample(float(i), person, "s", "goal", movement or ((.1, .1) if i < 2 else (.4, .4)), gaze, dwell, "level", head, quality, flags)

    def test_proxy_and_forbidden_validation(self):
        with self.assertRaises(ValueError):
            ObservationSample.from_dict({"timestamp": 1, "person_id": "p", "session_id": "s", "context_id": "c", "facial_movement_proxy_values": [.2], "gaze_zone": "center", "gaze_dwell_seconds": 0, "head_orientation_zone": "level", "head_transition_count": 0, "quality_score": .9, "emotion": "happy"})
        with self.assertRaises(ValueError):
            ObservationSample.from_dict({"timestamp": 1, "person_id": "p", "session_id": "s", "context_id": "c", "facial_movement_proxy_values": [.2], "gaze_zone": "center", "gaze_dwell_seconds": 0, "head_orientation_zone": "level", "head_transition_count": 0, "quality_score": .9, "raw_video": "x"})
        with self.assertRaises(ValueError):
            self.sample(0, movement=(1.1,))
        with self.assertRaises(TypeError):
            self.sample(0).facial_movement_proxy_values[0] = .3

    def test_direct_constructor_bounds_movement_before_materialization(self):
        def movement_generator():
            if False:
                yield 0.1

        with self.assertRaises(ValueError):
            ObservationSample(0, "p", "s", "goal", movement_generator(), "center", 0, "level", 0, .9)
        with self.assertRaises(ValueError):
            ObservationSample(0, "p", "s", "goal", [0.1] * 33, "center", 0, "level", 0, .9)
        self.assertEqual(
            ObservationSample(0, "p", "s", "goal", [.1, .2], "center", 0, "level", 0, .9).facial_movement_proxy_values,
            (.1, .2),
        )
    def test_deterministic_extraction_and_threshold_boundaries(self):
        rows = [self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2, head=2)]
        first, second = extract_event_candidates(rows), extract_event_candidates(rows)
        self.assertEqual([x.to_dict() for x in first], [x.to_dict() for x in second])
        self.assertTrue(first)
        self.assertTrue(all(x.requires_human_review and "non-diagnostic" in x.notice for x in first))
        self.assertEqual(extract_event_candidates([self.sample(0, quality=.6)]), [])
        self.assertTrue(any(x.event_type == "quality" for x in extract_event_candidates([self.sample(0, quality=.59)])))
        self.assertTrue(any(x.event_type == "quality" for x in extract_event_candidates([self.sample(0, flags=("unavailable",))])))
    def test_extraction_cap_precedes_candidate_materialization(self):
        samples = [self.sample(i, quality=.5, dwell=2, head=2) for i in range(1400)]
        from app import ondamm_face_event_learning as learning
        with mock.patch.object(learning, "_make_candidate", wraps=learning._make_candidate) as make_candidate:
            with self.assertRaises(ValueError):
                extract_event_candidates(samples)
        self.assertLessEqual(make_candidate.call_count, MAX_SAFE_COLLECTION)


    def test_json_round_trips(self):
        sample = self.sample(0)
        self.assertEqual(ObservationSample.from_dict(json.loads(dumps(sample))), sample)
        candidate = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])[0]
        from app.ondamm_face_event_learning import loads_candidate
        self.assertEqual(loads_candidate(dumps(candidate)).to_dict(), candidate.to_dict())
        label = LabelRecord("teacher", "t", candidate.candidate_id, "pause", "goal", "helpful", True)
        from app.ondamm_face_event_learning import loads_label
        self.assertEqual(loads_label(dumps(label)), label)
    def test_direct_loaders_enforce_bounded_utf8_input(self):
        oversized = b" " * (MAX_INPUT_BYTES + 1)
        for loader in (loads_sample, loads_candidate, loads_label, loads_model):
            with self.subTest(loader=loader.__name__):
                with self.assertRaises(ValueError):
                    loader(oversized)
        with self.assertRaises(ValueError):
            loads_sample(b"\xff")

    def test_role_approval_and_disagreement_gates(self):
        candidate = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])[0]
        with self.assertRaises(ValueError):
            train_model([candidate], [LabelRecord("teacher", "t", candidate.candidate_id, "pause", "goal", "helpful", True)], source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        labels = [LabelRecord("expert", "e", candidate.candidate_id, "pause", "goal", "helpful", True), LabelRecord("teacher", "t", candidate.candidate_id, "break", "goal", "helpful", True)]
        with self.assertRaises(ValueError): train_model([candidate], labels, source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        labels = [LabelRecord("expert", "e", candidate.candidate_id, "pause", "goal", "uncertain", True), LabelRecord("teacher", "t", candidate.candidate_id, "pause", "goal", "helpful", True)]
        with self.assertRaises(ValueError): train_model([candidate], labels, source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])

    def test_per_person_leakage_and_tamper_rejection(self):
        cands = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        c_other = extract_event_candidates([self.sample(0, person="other"), self.sample(1, person="other"), self.sample(2, person="other", gaze="left", dwell=2)])
        labels = [LabelRecord("expert", "e", cands[0].candidate_id, "pause", "goal", "helpful", True), LabelRecord("teacher", "t", cands[0].candidate_id, "pause", "goal", "helpful", True)]
        with self.assertRaises(ValueError): train_model(cands + c_other, labels, source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        model = train_model(cands, labels, source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        tampered = model.to_dict(); tampered["training_candidate_ids"][0] = "tampered"
        with self.assertRaises(ValueError): match_reviewable_candidates(tampered, [])

    def test_candidate_only_matching_and_confidence_abstention(self):
        train_samples = [self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)]
        cands = extract_event_candidates(train_samples)
        labels = [LabelRecord("expert", "e", cands[0].candidate_id, "pause", "goal", "helpful", True), LabelRecord("teacher", "t", cands[0].candidate_id, "pause", "goal", "helpful", True)]
        model = train_model(cands, labels, source_samples=train_samples)
        matches = match_reviewable_candidates(model, [self.sample(20), self.sample(21), self.sample(22, gaze="left", dwell=2)])
        self.assertTrue(all(x.requires_human_review and x.source_model_digest == model.model_digest for x in matches))
        self.assertTrue(all("support_strategy_candidate" in x.feature_summary for x in matches))
        strict = model.to_dict()
        strict["abstention_threshold"] = 1.0
        digest_payload = {key: value for key, value in strict.items() if key != "model_digest"}
        strict["model_digest"] = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(match_reviewable_candidates(strict, train_samples), [])
        self.assertEqual(match_reviewable_candidates(model, [self.sample(0, person="different")]), [])

    def test_cycle_and_depth_fail_closed(self):
        cyclic = {}
        cyclic["nested"] = cyclic
        with self.assertRaises(ValueError):
            ObservationSample.from_dict(cyclic)
        deep = value = {}
        for _ in range(40):
            value["nested"] = {}
            value = value["nested"]
        with self.assertRaises(ValueError):
            ObservationSample.from_dict(deep)

    def test_value_policy_and_candidate_schema(self):
        candidate = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])[0]
        raw = candidate.to_dict()
        raw["feature_summary"]["nested"] = {"emotion": "happy"}
        with self.assertRaises(ValueError):
            from app.ondamm_face_event_learning import loads_candidate
            loads_candidate(json.dumps(raw))
        with self.assertRaises(ValueError):
            LabelRecord("teacher", "t", candidate.candidate_id, "unsupported_strategy", "goal", "helpful", True)
        with self.assertRaises(ValueError):
            LabelRecord("teacher", "t", candidate.candidate_id, "pause", "emotion goal", "helpful", True)
        raw.pop("evidence_ids")
        with self.assertRaises(ValueError):
            from app.ondamm_face_event_learning import loads_candidate
            loads_candidate(json.dumps(raw))
        with self.assertRaises(ValueError):
            ObservationSample(
                1, "p", "s", "emotion", (.2,), "center", 0, "level", 0, .9
            )
        with self.assertRaises(ValueError):
            ObservationSample(
                1, "p", "s", "goal", (.2,), "center", 0, "level", 0, .9, ("raw_video",)
            )
        tampered = candidate.to_dict()
        tampered["provenance"] = list(candidate.provenance[:-1])
        tampered["candidate_id"] = hashlib.sha256(
            json.dumps(
                {key: value for key, value in tampered.items() if key != "candidate_id"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]
        with self.assertRaises(ValueError):
            loads_candidate(json.dumps(tampered))
    def test_generic_dumps_rejects_non_finite_and_unsafe_numbers(self):
        for value in (float("nan"), float("inf"), float("-inf"), 10 ** 1000):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    dumps({"value": value})
    def test_strict_notice_and_scalar_serialization_policy(self):
        candidate = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])[0]
        for forbidden in ("emotion=happy", "diagnosis=none", "raw_video=clip"):
            raw = candidate.to_dict()
            raw["notice"] = f"{NOTICE} {forbidden}"
            with self.assertRaises(ValueError):
                from app.ondamm_face_event_learning import loads_candidate
                loads_candidate(json.dumps(raw))
        with self.assertRaises(ValueError):
            dumps({"nested": ["raw_video"]})
        with self.assertRaises(ValueError):
            dumps({"nested": ["diagnosis"]})

    def test_candidate_bounds_and_evidence_cardinality(self):
        candidate = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])[0]
        with self.assertRaises(ValueError):
            replace(candidate, start_timestamp=-1)
        with self.assertRaises(ValueError):
            replace(candidate, end_timestamp=candidate.start_timestamp + MAX_EVENT_SPAN_SECONDS + 1)
    def test_collection_bounds_reject_before_copying(self):
        candidate = extract_event_candidates(
            [self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)]
        )[0]
        summary = dict(candidate.feature_summary)
        summary["gaze_zones"] = ["center", "left", "other", "extra"]
        with self.assertRaises(ValueError):
            replace(candidate, feature_summary=summary)
        with self.assertRaises(ValueError):
            replace(candidate, quality_flags=tuple(f"flag-{index}" for index in range(33)))
        self.assertGreater(candidate.feature_summary["sample_count"], 1)
        summary = dict(candidate.feature_summary)
        summary["sample_count"] -= 1
        with self.assertRaises(ValueError):
            replace(candidate, feature_summary=summary)

    def test_main_rejects_non_object_and_unknown_root_fields(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with mock.patch("sys.stdin", io.StringIO("[]")):
                rc = main([])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(output.getvalue())["error"], "input JSON root must be an object")
        output = io.StringIO()
        with redirect_stdout(output):
            with mock.patch("sys.stdin", io.StringIO(json.dumps({"samples": [], "ignored": True}))):
                rc = main([])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(output.getvalue())["error"], "unknown root fields: ['ignored']")
    def test_main_rejects_local_input_read_errors(self):
        for path_kind in ("missing", "directory", "unicode"):
            with self.subTest(path_kind=path_kind):
                with tempfile.TemporaryDirectory() as directory:
                    if path_kind == "missing":
                        path = f"{directory}/missing.json"
                    elif path_kind == "directory":
                        path = directory
                    else:
                        path = f"{directory}/invalid.json"
                        with open(path, "wb") as handle:
                            handle.write(b"\xff")
                    output = io.StringIO()
                    with redirect_stdout(output):
                        rc = main(["--input", path])
                    self.assertEqual(rc, 2)
                    payload = json.loads(output.getvalue())
                    self.assertEqual(payload["notice"], NOTICE)

    def test_main_rejects_oversized_stdin_and_file_input(self):
        oversized = " " * (MAX_INPUT_BYTES + 1)
        output = io.StringIO()
        with redirect_stdout(output):
            with mock.patch("sys.stdin", io.StringIO(oversized)):
                rc = main([])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(output.getvalue())["notice"], NOTICE)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/oversized.json"
            with open(path, "wb") as handle:
                handle.write(b" " * (MAX_INPUT_BYTES + 1))
            output = io.StringIO()
            with redirect_stdout(output):
                rc = main(["--input", path])
            self.assertEqual(rc, 2)
            self.assertEqual(json.loads(output.getvalue())["notice"], NOTICE)
    def test_main_rejects_deep_json_recursion(self):
        nested = "[" * 1100 + "]" * 1100
        self.assertLess(len(nested.encode("utf-8")), MAX_INPUT_BYTES)
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            with mock.patch("sys.stdin", io.StringIO(nested)):
                rc = main([])
        self.assertEqual(rc, 2)
        payload = json.loads(output.getvalue())
        self.assertIsInstance(payload, dict)
        self.assertIn("error", payload)
        self.assertIsInstance(payload["error"], str)
        self.assertEqual(payload["notice"], NOTICE)
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn("Traceback", output.getvalue())

    def test_main_rejects_malformed_argv_without_argparse_output(self):
        for args in (["--unknown"], ["--input"], ["--demo", "--bogus"]):
            with self.subTest(args=args):
                output, errors = io.StringIO(), io.StringIO()
                with redirect_stdout(output), redirect_stderr(errors):
                    rc = main(args)
                self.assertEqual(rc, 2)
                self.assertEqual(json.loads(output.getvalue())["notice"], NOTICE)
                self.assertEqual(errors.getvalue(), "")
    def test_main_rejects_conflicting_modes_deterministically(self):
        output = io.StringIO()
        with redirect_stdout(output):
            rc = main(["--demo", "--train"])
        self.assertEqual(rc, 2)
        self.assertEqual(
            json.loads(output.getvalue()),
            {"error": "modes are mutually exclusive", "notice": NOTICE},
        )

    def test_model_training_provenance_normalizes(self):
        cands = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        labels = [
            LabelRecord("expert", "e1", cands[0].candidate_id, "pause", "goal", "helpful", True),
            LabelRecord("teacher", "t1", cands[0].candidate_id, "pause", "goal", "helpful", True),
            LabelRecord("expert", "e2", cands[1].candidate_id, "pause", "goal", "helpful", True),
            LabelRecord("teacher", "t2", cands[1].candidate_id, "pause", "goal", "helpful", True),
        ]
        model = train_model(cands, labels, source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        equivalent = PrototypeModel(
            model.person_id,
            model.strategy_prototypes,
            tuple(reversed(model.training_candidate_ids)),
            dict(reversed(list(model.training_fingerprints.items()))),
            tuple(reversed(model.label_provenance)),
            model.model_digest,
            model.min_quality,
            model.abstention_threshold,
        )
        self.assertEqual(equivalent.to_dict(), model.to_dict())

    def test_per_candidate_approval_and_context_gates(self):
        cands = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        first = cands[0]
        labels = [LabelRecord("expert", "e", first.candidate_id, "pause", "goal", "helpful", True)]
        with self.assertRaises(ValueError):
            train_model(cands, labels, source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        labels.append(LabelRecord("teacher", "t", first.candidate_id, "pause", "other-goal", "helpful", True))
        with self.assertRaises(ValueError):
            train_model(cands, labels, source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])

    def test_prototype_manifest_validation_and_recomputed_tamper(self):
        cands = extract_event_candidates([self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        labels = [LabelRecord("expert", "e", cands[0].candidate_id, "pause", "goal", "helpful", True), LabelRecord("teacher", "t", cands[0].candidate_id, "pause", "goal", "helpful", True)]
        model = train_model(cands, labels, source_samples=[self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)])
        manifest = model.to_dict()
        manifest["strategy_prototypes"]["pause"] = []
        with self.assertRaises(ValueError):
            PrototypeModel.from_dict(manifest)
        manifest = model.to_dict()
        manifest["training_candidate_ids"] = []
        manifest["model_digest"] = hashlib.sha256(json.dumps({key: value for key, value in manifest.items() if key != "model_digest"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.assertRaises(ValueError):
            match_reviewable_candidates(manifest, [])

    def test_nested_feature_freeze_and_json_lists(self):
        samples = [self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)]
        candidate = extract_event_candidates(samples)[0]
        with self.assertRaises(TypeError):
            candidate.feature_summary["feature_vector"][0] = 99
        with self.assertRaises(TypeError):
            candidate.feature_summary["gaze_zones"][0] = "mutated"
        with self.assertRaises(TypeError):
            candidate.feature_summary["head_zones"][0] = "mutated"
        payload = candidate.to_dict()
        self.assertIsInstance(payload["feature_summary"]["feature_vector"], list)
        self.assertIsInstance(payload["feature_summary"]["gaze_zones"], list)
        self.assertIsInstance(payload["feature_summary"]["head_zones"], list)

    def test_direct_model_wrong_digest_rejected(self):
        samples = [self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)]
        candidates = extract_event_candidates(samples)
        labels = [
            LabelRecord("expert", "e", candidates[0].candidate_id, "pause", "goal", "helpful", True),
            LabelRecord("teacher", "t", candidates[0].candidate_id, "pause", "goal", "helpful", True),
        ]
        model = train_model(candidates, labels, source_samples=samples)
        with self.assertRaises(ValueError):
            PrototypeModel(
                model.person_id,
                model.strategy_prototypes,
                model.training_candidate_ids,
                model.training_fingerprints,
                model.label_provenance,
                "0" * 64,
                model.min_quality,
                model.abstention_threshold,
            )

    def test_source_bound_training_rejects_recomputed_candidate(self):
        samples = [self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)]
        candidate = extract_event_candidates(samples)[0]
        payload = candidate.to_dict()
        payload["feature_summary"]["reason"] = "altered_proxy_evidence"
        unsigned = {key: value for key, value in payload.items() if key != "candidate_id"}
        payload["candidate_id"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
        from app.ondamm_face_event_learning import loads_candidate
        altered = loads_candidate(json.dumps(payload))
        labels = [
            LabelRecord("expert", "e", altered.candidate_id, "pause", "goal", "helpful", True),
            LabelRecord("teacher", "t", altered.candidate_id, "pause", "goal", "helpful", True),
        ]
        with self.assertRaises(ValueError):
            train_model([altered], labels, source_samples=samples)

    def test_source_digest_mismatch_rejected(self):
        samples = [self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)]
        candidate = extract_event_candidates(samples)[0]
        payload = candidate.to_dict()
        payload["source_sample_digest"] = "0" * 64
        unsigned = {key: value for key, value in payload.items() if key != "candidate_id"}
        payload["candidate_id"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
        from app.ondamm_face_event_learning import loads_candidate
        altered = loads_candidate(json.dumps(payload))
        labels = [
            LabelRecord("expert", "e", altered.candidate_id, "pause", "goal", "helpful", True),
            LabelRecord("teacher", "t", altered.candidate_id, "pause", "goal", "helpful", True),
        ]
        with self.assertRaises(ValueError):
            train_model([altered], labels, source_samples=samples)

    def test_source_bound_training_valid(self):
        samples = [self.sample(0), self.sample(1), self.sample(2, gaze="left", dwell=2)]
        candidates = extract_event_candidates(samples)
        labels = [
            LabelRecord("expert", "e", candidates[0].candidate_id, "pause", "goal", "helpful", True),
            LabelRecord("teacher", "t", candidates[0].candidate_id, "pause", "goal", "helpful", True),
        ]
        model = train_model(candidates, labels, source_samples=samples)
        self.assertIn(candidates[0].candidate_id, model.training_candidate_ids)
    def test_graph_bounds_and_finite_json_integer(self):
        bounded = {"values": [0] * MAX_CONTAINER_BREADTH}
        _check_forbidden(bounded)
        nested = value = 0
        for _ in range(32):
            value = {"nested": value}
        _check_forbidden(nested)
        with self.assertRaises(ValueError):
            _check_forbidden({"branches": [[0] * MAX_CONTAINER_BREADTH for _ in range(MAX_GRAPH_NODES // (MAX_CONTAINER_BREADTH + 1) + 1)]})
        for field, value in (
            ("timestamp", 10 ** 1000),
            ("facial_movement_proxy_values", [10 ** 1000]),
        ):
            sample = self.sample(0).to_dict()
            sample[field] = value
            output = io.StringIO()
            with redirect_stdout(output):
                with mock.patch("sys.stdin", io.StringIO(json.dumps({"samples": [sample]}))):
                    rc = main([])
            self.assertEqual(rc, 2)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["notice"], NOTICE)
            self.assertEqual(payload["error"], f"{field} must be finite numeric")

    def test_cli_rejects_oversized_root_and_nested_containers(self):
        cases = (
            ([], {"samples": [None] * (MAX_CONTAINER_BREADTH + 1)}),
            (["--train"], {"samples": [], "labels": [None] * (MAX_CONTAINER_BREADTH + 1)}),
            ([], {"samples": [{"nested": [0] * (MAX_CONTAINER_BREADTH + 1)}]}),
        )
        for args, data in cases:
            with self.subTest(args=args):
                output = io.StringIO()
                with redirect_stdout(output):
                    with mock.patch("sys.stdin", io.StringIO(json.dumps(data))):
                        rc = main(args)
                self.assertEqual(rc, 2)
                self.assertEqual(json.loads(output.getvalue())["notice"], NOTICE)
    def test_cli_demo_is_deterministic_and_non_diagnostic(self):
        cmd = [sys.executable, "-m", "app.ondamm_face_event_learning_cli", "--demo"]
        a = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        b = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        self.assertEqual(a, b)
        payload = json.loads(a)
        self.assertIn("non-diagnostic", payload["notice"])
        self.assertNotIn("emotion", a.lower()); self.assertNotIn("raw_image", a.lower())


if __name__ == "__main__":
    unittest.main()
