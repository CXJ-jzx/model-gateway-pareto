"""Synthetic run artifacts test independent validation and tamper detection."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from bp_slo.dataset import file_hash, iter_references, normalize_request, write_json, write_jsonl
from bp_slo.statistics import analyze
from bp_slo.validation import REQUIRED, validate_run


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run = self.base / "run"
        (self.run / "dataset").mkdir(parents=True)
        self.source = self.base / "source"
        self.source.mkdir()
        self.config = dict(model_id="BP", minimum_group_samples=2,
                           input_bin_upper_bounds=[32, 128], output_bin_upper_bounds=[4, 16],
                           train_fraction=.6, calibration_fraction=.2, routing_contract={"slots": 3})
        self.raw = []
        start = datetime(2026, 9, 21, tzinfo=timezone.utc)
        for index in range(5):
            self.raw.append(dict(request_id=f"r{index}", target_model="BP", is_stream=True,
                                 request_arrival_at=(start + timedelta(minutes=index)).isoformat(),
                                 arrival_at_ms=100, finished_at_ms=600, elapsed_ms=500,
                                 gateway_result="completed", http_status=200, input_tokens=20,
                                 output_tokens=4, final_endpoint_id="E", attempts=[], quality_flags=[]))
        self.raw[0]["historical_performance"] = {
            "endpoint_histories": [{"endpoint_id": "E", "endpoint_model": {"3d": "s1"}}],
            "model_all_endpoints": {"7d": "s2"},
        }
        historical = start - timedelta(minutes=1)
        self.snapshots = [
            dict(snapshot_id="s1", scope="endpoint_model", endpoint_id="E", model_id="BP", window_days=3,
                 as_of_exclusive=historical.isoformat(), as_of_epoch=historical.timestamp(),
                 last_observation_at=(historical-timedelta(seconds=1)).isoformat()),
            dict(snapshot_id="s2", scope="model_all_endpoints", endpoint_id=None, model_id="BP", window_days=7,
                 as_of_epoch=historical.timestamp(), included_end_exclusive=historical.isoformat()),
        ]
        source_model = dict(model_id="BP", price_config={"input": 1}, capacity_config={"rpm": 60})
        source_endpoint = dict(endpoint_id="E", deployment_type="direct", models=[source_model])
        self.offerings = [dict(endpoint_id="E", model_id="BP", deployment_type="direct", source_line=1,
                               price_config=source_model["price_config"], capacity_config=source_model["capacity_config"],
                               source_model_entry=source_model)]
        source_traffic = [dict(request_id="other", target_model="OTHER")] + self.raw
        write_jsonl(self.source / "traffic.jsonl", source_traffic)
        write_jsonl(self.source / "offerings.jsonl", [source_endpoint])
        write_jsonl(self.source / "snapshots.jsonl", self.snapshots)
        self.rows = [normalize_request(raw, i+2) for i, raw in enumerate(self.raw)]
        self.refs = [ref for raw in self.raw for ref in iter_references(raw)]
        self.sources = {role: {"path": str(self.source / f"{role}.jsonl"),
                               "sha256": file_hash(self.source / f"{role}.jsonl"),
                               "bytes": (self.source / f"{role}.jsonl").stat().st_size}
                        for role in ("traffic", "offerings", "snapshots")}
        self.extraction = dict(model_id="BP", source_request_count=6, selected_request_count=5,
                               endpoint_offering_count=1, historical_reference_count=2,
                               source_order_arrival_reversals=0, unique_snapshot_count=2,
                               snapshot_scopes=dict(Counter(s["scope"] for s in self.snapshots)), sources=self.sources)
        write_json(self.run / "config.json", self.config)
        for name, values in (("raw_requests", self.raw), ("requests", self.rows),
                             ("endpoint_offerings", self.offerings), ("historical_references", self.refs),
                             ("historical_snapshots", self.snapshots)):
            write_jsonl(self.run / "dataset" / f"{name}.jsonl", values)
        self.analysis, self.splits = analyze(self.rows, self.offerings, self.config, self.extraction)
        write_json(self.run / "analysis.json", self.analysis)
        write_jsonl(self.run / "dataset/diagnostic_splits.jsonl", self.splits)
        self.refresh_manifest()

    def refresh_manifest(self):
        self.manifest = dict(sources=self.sources, extraction=self.extraction,
                             artifacts={relative: dict(sha256=file_hash(self.run / relative),
                                                        bytes=(self.run / relative).stat().st_size)
                                        for relative in REQUIRED if (self.run / relative).is_file()})
        write_json(self.run / "manifest.json", self.manifest)

    def assert_check_fails(self, name):
        result = validate_run(self.run)
        self.assertFalse(result["passed"], result)
        self.assertTrue(any(check["name"] == name and not check["passed"] for check in result["checks"]), result)
        return result

    def test_valid_run(self):
        result = validate_run(self.run)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["warnings"], [])

    def test_absent_source_is_warning_not_failure(self):
        for role in self.sources:
            self.sources[role]["path"] = str(self.base / "unavailable" / f"{role}.jsonl")
        self.analysis, self.splits = analyze(self.rows, self.offerings, self.config, self.extraction)
        write_json(self.run / "analysis.json", self.analysis)
        self.refresh_manifest()
        result = validate_run(self.run)
        self.assertTrue(result["passed"], result)
        self.assertEqual(len(result["warnings"]), 3)

    def test_missing_and_malformed_artifacts_never_raise(self):
        result = validate_run(self.base / "missing")
        self.assertFalse(result["passed"])
        for content in ("{bad", "[]", '{"bad": NaN}'):
            with self.subTest(content=content):
                (self.run / "manifest.json").write_text(content, encoding="utf-8")
                self.assert_check_fails("read:manifest.json")
        self.refresh_manifest()
        (self.run / "dataset/requests.jsonl").write_text("null\n", encoding="utf-8")
        self.assert_check_fails("read:dataset/requests.jsonl")

    def test_artifact_hash_tampering_is_detected(self):
        changed = deepcopy(self.analysis)
        changed["total"] = 999
        write_json(self.run / "analysis.json", changed)
        self.assert_check_fails("manifest_integrity")

    def test_rehashed_normalization_tampering_is_detected(self):
        self.rows[0]["predicted_output_tokens"] = self.rows[0]["actual_output_tokens"]
        write_jsonl(self.run / "dataset/requests.jsonl", self.rows)
        self.refresh_manifest()
        self.assert_check_fails("request_identity_normalization_order")

    def test_duplicate_request_ids_are_detected(self):
        write_jsonl(self.run / "dataset/requests.jsonl", self.rows + [self.rows[0]])
        self.refresh_manifest()
        self.assert_check_fails("request_identity_normalization_order")

    def test_chronological_order_is_checked(self):
        write_jsonl(self.run / "dataset/requests.jsonl", list(reversed(self.rows)))
        self.refresh_manifest()
        self.assert_check_fails("request_identity_normalization_order")

    def test_missing_and_scope_mismatched_snapshots_are_detected(self):
        variants = [self.snapshots[:1], self.snapshots + [self.snapshots[0]]]
        for field, value in (("scope", "endpoint"), ("model_id", "OTHER"),
                             ("endpoint_id", "OTHER"), ("window_days", 7)):
            changed = deepcopy(self.snapshots)
            changed[0][field] = value
            variants.append(changed)
        for snapshots in variants:
            with self.subTest(snapshots=snapshots):
                write_jsonl(self.run / "dataset/historical_snapshots.jsonl", snapshots)
                self.refresh_manifest()
                self.assert_check_fails("historical_reference_closure_and_no_leakage")

    def test_references_must_equal_raw_references(self):
        write_jsonl(self.run / "dataset/historical_references.jsonl", self.refs[:1])
        self.refresh_manifest()
        self.assert_check_fails("historical_reference_closure_and_no_leakage")

    def test_snapshot_asof_must_not_be_after_request_arrival(self):
        self.snapshots[1]["as_of_epoch"] = datetime(2026, 9, 21, 0, 1, tzinfo=timezone.utc).timestamp()
        write_jsonl(self.run / "dataset/historical_snapshots.jsonl", self.snapshots)
        self.refresh_manifest()
        self.assert_check_fails("historical_reference_closure_and_no_leakage")

    def test_last_observation_must_be_strictly_before_asof(self):
        self.snapshots[0]["last_observation_at"] = self.snapshots[0]["as_of_exclusive"]
        write_jsonl(self.run / "dataset/historical_snapshots.jsonl", self.snapshots)
        self.refresh_manifest()
        self.assert_check_fails("historical_reference_closure_and_no_leakage")

    def test_rehashed_analysis_and_split_tampering_is_detected(self):
        self.analysis["modes"]["stream"]["total"] = 999
        write_json(self.run / "analysis.json", self.analysis)
        self.refresh_manifest()
        self.assert_check_fails("analysis_and_splits_reproducibility")
        self.analysis, _ = analyze(self.rows, self.offerings, self.config, self.extraction)
        write_json(self.run / "analysis.json", self.analysis)
        self.splits[-1]["training_outcome_visible"] = True
        write_jsonl(self.run / "dataset/diagnostic_splits.jsonl", self.splits)
        self.refresh_manifest()
        self.assert_check_fails("analysis_and_splits_reproducibility")

    def test_modified_available_source_fails(self):
        write_jsonl(self.source / "traffic.jsonl", self.raw)
        self.assert_check_fails("source:traffic")

    def test_extraction_completeness_survives_rehashed_partial_run(self):
        write_jsonl(self.run / "dataset/raw_requests.jsonl", self.raw[1:])
        write_jsonl(self.run / "dataset/requests.jsonl", self.rows[1:])
        self.refresh_manifest()
        self.assert_check_fails("external_source_projection_and_completeness")

    def test_offering_projection_must_match_external_source(self):
        self.offerings[0]["price_config"] = {"input": 999}
        self.offerings[0]["source_model_entry"]["price_config"] = {"input": 999}
        write_jsonl(self.run / "dataset/endpoint_offerings.jsonl", self.offerings)
        self.refresh_manifest()
        self.assert_check_fails("external_source_projection_and_completeness")

    def test_manifest_cannot_reference_files_outside_run(self):
        self.manifest["artifacts"]["../source/traffic.jsonl"] = self.sources["traffic"]
        write_json(self.run / "manifest.json", self.manifest)
        self.assert_check_fails("manifest_integrity")


if __name__ == "__main__":
    unittest.main()
