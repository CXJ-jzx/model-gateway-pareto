import copy
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from endpoint_routing_strategy.data.normalize import normalize_request
from endpoint_routing_strategy.data.pipeline import dump_jsonl, prepare_dataset
from endpoint_routing_strategy.data.profiles import build_incoming_request, enrich_request as enrich_incoming, fit_profiles, urgency_at, validate_config
from endpoint_routing_strategy.data.validation import validate_dataset
from endpoint_routing_strategy.data_pipeline import load_observations_and_latest_history


CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiments/configs/data_preparation.json"


def enrich_request(request, profiles, config, split):
    return enrich_incoming(build_incoming_request(request, profiles, config, split), profiles, config)


def raw_request(index=0, stream=True):
    return {
        "request_id": f"r{index}", "target_model": "m", "is_stream": stream,
        "historical_performance": {"request_arrival_at": f"2026-10-01T00:00:{index:02d}+08:00"},
        "arrival_at_ms": index * 1000, "input_tokens": 100, "output_tokens": 11,
        "usable_for_full_attempt_timing": True, "gateway_result": "completed", "final_endpoint_id": "e",
        "attempts": [{"attempt_id": f"a{index}", "endpoint_id": "e", "result": "success",
                      "http_status": 200, "sent_at_ms": index * 1000 + 10,
                      "finished_at_ms": index * 1000 + 510, "first_token_at_ms": 100,
                      "output_tokens": 11}],
    }


class NormalizationTests(unittest.TestCase):
    def test_latency_proxy_and_completed_visibility(self):
        request = normalize_request(raw_request())
        attempt = request.attempts[0]
        self.assertAlmostEqual(attempt.e2e_ms, 500)
        self.assertAlmostEqual(attempt.ttft_ms, 100)
        self.assertAlmostEqual(attempt.tpot_ms, 40)
        self.assertEqual(attempt.observation(True).occurred_at, attempt.finished_at)

    def test_incomplete_attempt_never_becomes_arrival_observation(self):
        raw = raw_request()
        raw["attempts"][0]["finished_at_ms"] = None
        self.assertIsNone(normalize_request(raw).attempts[0].observation(True))

    def test_invalid_time_and_ttft_are_flagged(self):
        raw = raw_request()
        raw["attempts"][0]["first_token_at_ms"] = 999
        attempt = normalize_request(raw).attempts[0]
        self.assertIsNone(attempt.ttft_ms)
        self.assertIn("invalid_ttft", attempt.flags)
        raw["attempts"][0]["finished_at_ms"] = 1
        self.assertIsNone(normalize_request(raw).attempts[0].finished_at)

    def test_missing_actual_usage_is_not_zero(self):
        raw = raw_request()
        raw["input_tokens"] = None
        self.assertIsNone(normalize_request(raw).actual_input_tokens)

    def test_predictions_separate_and_zero_preserved(self):
        raw = raw_request()
        raw["token_predictions"] = {"input_tokens": 0, "output_tokens": 5000}
        request = normalize_request(raw)
        self.assertEqual(request.predicted_input_tokens, 0)
        self.assertEqual(request.predicted_output_tokens, 5000)
        self.assertEqual(request.actual_output_tokens, 11)

    def test_parser_shared_by_demo_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traffic.jsonl"
            dump_jsonl(path, [raw_request()])
            legacy, _, _ = load_observations_and_latest_history(path, "m")
            expected = normalize_request(raw_request()).attempts[0].observation(True)
            self.assertEqual(legacy, [expected])
            # Historical tools are optional and excluded from the GitHub release.
            legacy_path = Path(__file__).resolve().parents[1] / "experiments/legacy/replay_requests.py"
            if legacy_path.is_file():
                from endpoint_routing_strategy.experiments.legacy.replay_requests import load_replay_data
                _, replay, _ = load_replay_data(path, "m")
                self.assertEqual(legacy, replay)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.config["minimum_group_samples"] = 1
        self.training = [normalize_request(raw_request(i)) for i in range(3)]
        self.cutoff = normalize_request(raw_request(5)).arrived_at
        self.profiles = fit_profiles(self.training, self.cutoff, self.config)

    def test_future_training_completion_excluded(self):
        raw = raw_request(2)
        raw["output_tokens"] = 900000
        raw["attempts"][0]["finished_at_ms"] = 60000
        profiles = fit_profiles([normalize_request(raw)], self.cutoff, self.config)
        self.assertEqual(profiles["groups"]["m|stream"]["output"]["source"], "configured_fallback")
        self.assertIsNone(profiles["latest_outcome_at"])

    def test_heldout_actual_outcome_cannot_change_online_fields(self):
        raw = raw_request(8)
        first = enrich_request(normalize_request(raw), self.profiles, self.config, "test")
        raw["output_tokens"] = 999999
        raw["attempts"][0]["output_tokens"] = 999999
        second = enrich_request(normalize_request(raw), self.profiles, self.config, "test")
        # Quality flags may change in other examples; predictions and policy cannot.
        for field in ("token_predictions", "priority", "slo", "workload", "urgency"):
            self.assertEqual(first[field], second[field])

    def test_priority_heaviness_independent_and_three_slo_levels(self):
        raw = raw_request(8)
        raw.update(predicted_input_tokens=10000, predicted_output_tokens=10000)
        budgets = []
        for priority, tier in (("high", "strict"), ("normal", "standard"), ("low", "relaxed")):
            raw["priority"] = priority
            row = enrich_request(normalize_request(raw), self.profiles, self.config, "test")
            self.assertEqual(row["slo"]["tier"], tier)
            self.assertEqual(row["workload"]["level"], "mixed_heavy")
            self.assertIsNone(row["workload"]["classification_heavy"])
            self.assertLess(row["slo"]["max_wait_ms"], row["slo"]["ttft_ms"])
            budgets.append(row["slo"]["e2e_ms"])
        self.assertEqual(budgets, sorted(set(budgets)))

    def test_priority_slo_conflict_rejected(self):
        raw = raw_request(8)
        raw.update(priority="high", slo_tier="relaxed")
        with self.assertRaisesRegex(ValueError, "conflict"):
            enrich_request(normalize_request(raw), self.profiles, self.config, "test")

    def test_completed_log_cannot_enter_gateway_enrichment(self):
        with self.assertRaises(TypeError):
            enrich_incoming(normalize_request(raw_request(8)), self.profiles, self.config)

    def test_initial_request_independent_of_result_endpoint_and_quality(self):
        raw = raw_request(8)
        first = build_incoming_request(normalize_request(raw), self.profiles, self.config, "test")
        raw.update(input_tokens=999999, output_tokens=999999, final_endpoint_id="other", gateway_result="failed",
                   quality_flags=["post_execution_error"])
        raw["attempts"][0].update(http_status=500, result="server_error", finished_at_ms=999999)
        second = build_incoming_request(normalize_request(raw), self.profiles, self.config, "test")
        self.assertEqual(first, second)

    def test_warmup_cannot_use_future_training_statistics(self):
        facts = normalize_request(raw_request(0))
        initial = build_incoming_request(facts, self.profiles, self.config, "train")
        self.assertEqual(initial.output_prediction_source, "configured_warmup_mock")
        self.assertEqual(initial.predicted_output_tokens, self.config["fallback"]["output_tokens"])

    def test_urgency_increases_with_wait_and_ttft_deadline(self):
        request = normalize_request(raw_request(8))
        row = enrich_request(request, self.profiles, self.config, "test")
        now = request.arrived_at + timedelta(milliseconds=row["slo"]["ttft_ms"])
        urgent = urgency_at(row["slo"], now, row["estimated_service_ms"], self.config, row["estimated_ttft_ms"])
        self.assertEqual(urgent["level"], "critical")
        self.assertEqual(urgent["limiting_constraint"], "ttft")

    def test_invalid_config_rejected(self):
        invalid = copy.deepcopy(self.config)
        invalid["slo_tiers"]["strict"]["latency_multiplier"] = 5
        with self.assertRaises(ValueError):
            validate_config(invalid)


class DatasetTests(unittest.TestCase):
    def test_pipeline_reproducible_and_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traffic = root / "raw.jsonl"
            dump_jsonl(traffic, [raw_request(i) for i in range(20)])
            a, b = root / "a", root / "b"
            self.assertTrue(prepare_dataset(traffic, CONFIG_PATH, a)["passed"])
            self.assertTrue(prepare_dataset(traffic, CONFIG_PATH, b)["passed"])
            self.assertEqual((a / "requests.jsonl").read_bytes(), (b / "requests.jsonl").read_bytes())
            initial_rows = [json.loads(x) for x in (a / "incoming_requests.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(initial_rows), 20)
            forbidden = {"attempts", "http_status", "actual_input_tokens", "actual_output_tokens", "final_endpoint_id", "flags"}
            self.assertTrue(all(not (forbidden & row.keys()) for row in initial_rows))
            rows = [json.loads(x) for x in (a / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["urgency"]["level"] = "critical"
            dump_jsonl(a / "requests.jsonl", rows)
            report = validate_dataset(a)
            self.assertFalse(report["passed"])
            self.assertIn("artifact_hash_mismatch", report["errors"])
            self.assertIn("enrichment_not_reproducible", report["errors"])

    def test_duplicates_and_bad_rows_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traffic = root / "raw.jsonl"
            dump_jsonl(traffic, [raw_request(i) for i in range(20)] + [raw_request(0), {"request_id": "bad"}])
            output = root / "output"
            result = prepare_dataset(traffic, CONFIG_PATH, output)
            self.assertTrue(result["passed"])
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["duplicate_requests"], 1)
            self.assertEqual(audit["rejected_lines"], 1)


if __name__ == "__main__":
    unittest.main()
