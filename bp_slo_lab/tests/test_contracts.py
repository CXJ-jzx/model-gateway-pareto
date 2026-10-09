"""Public-contract checks with synthetic facts; no legacy project imports."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import unittest

from bp_slo.dataset import normalize_request
from bp_slo.pareto import Candidate, rank_frontier
from bp_slo.statistics import assign_time_splits, describe


def request_fact(**overrides):
    """Times are marker offsets; first_token_at_ms is already a duration."""
    raw = {
        "request_id": "r1",
        "target_model": "BP",
        "request_arrival_at": "2026-09-21T12:00:00+08:00",
        "arrival_at_ms": 10_000,
        "finished_at_ms": 10_500,
        "elapsed_ms": 500,
        "is_stream": True,
        "gateway_result": "completed",
        "http_status": 200,
        "input_tokens": 20,
        "output_tokens": 4,
        "first_token_at_ms": 60,
        "final_endpoint_id": "C",
        "usable_for_full_attempt_timing": True,
        "quality_flags": ["source_flag"],
        "attempts": [{
            "endpoint_id": "C",
            "sent_at_ms": 10_100,
            "finished_at_ms": 10_400,
            "first_token_at_ms": 50,
            "result": "success",
            "http_status": 200,
        }],
    }
    raw.update(overrides)
    return raw


class RequestNormalizationTests(unittest.TestCase):
    def test_keeps_request_and_attempt_scopes_distinct(self):
        raw = request_fact()
        original = deepcopy(raw)
        row = normalize_request(raw, source_line=17)
        self.assertEqual(raw, original)
        self.assertEqual(row["source_line"], 17)
        self.assertEqual(row["record_kind"], "completed_request_fact")
        self.assertEqual(row["request_e2e_ms"], 500)
        self.assertAlmostEqual(row["attempt_e2e_ms"], 300)
        self.assertEqual(row["request_ttft_proxy_ms"], 60)
        self.assertEqual(row["ttft_proxy_ms"], 50)
        self.assertAlmostEqual(row["tpot_proxy_ms"], 250 / 3)
        self.assertEqual(row["attempt_count"], 1)
        self.assertEqual(datetime.fromisoformat(row["arrived_at"]),
                         datetime(2026, 9, 21, 4, tzinfo=timezone.utc))
        self.assertEqual(datetime.fromisoformat(row["finished_at"]),
                         datetime(2026, 9, 21, 4, tzinfo=timezone.utc)
                         + timedelta(milliseconds=500))
        self.assertIn("source_flag", row["quality_flags"])

    def test_actual_lengths_never_become_predictions_or_slo(self):
        row = normalize_request(request_fact())
        self.assertEqual(row["actual_input_tokens"], 20)
        self.assertEqual(row["actual_output_tokens"], 4)
        self.assertIsNone(row["predicted_input_tokens"])
        self.assertIsNone(row["predicted_output_tokens"])
        self.assertEqual(row["prediction_source"], "missing")
        self.assertNotIn("slo", row)
        self.assertNotIn("priority", row)

    def test_preserves_provided_predictions_and_partial_availability(self):
        complete = normalize_request(request_fact(predicted_input_tokens=24,
                                                   predicted_output_tokens=6))
        self.assertEqual(complete["prediction_source"], "provided")
        self.assertEqual(complete["predicted_input_tokens"], 24)
        self.assertEqual(complete["predicted_output_tokens"], 6)
        partial = normalize_request(request_fact(predicted_output_tokens=6))
        self.assertEqual(partial["prediction_source"], "partial")
        self.assertIsNone(partial["predicted_input_tokens"])

    def test_failed_requests_have_no_success_performance(self):
        row = normalize_request(request_fact(gateway_result="failed", http_status=429))
        self.assertFalse(row["success"])
        self.assertFalse(row["eligible_length_e2e"])
        self.assertEqual(row["request_e2e_ms"], 500)
        for field in ("request_ttft_proxy_ms", "attempt_e2e_ms", "ttft_proxy_ms", "tpot_proxy_ms"):
            self.assertIsNone(row[field], field)

    def test_failed_or_mismatched_final_attempt_cannot_supply_metrics(self):
        for change in ({"result": "failed"}, {"http_status": 502}, {"endpoint_id": "D"}):
            with self.subTest(change=change):
                raw = request_fact()
                raw["attempts"][-1].update(change)
                row = normalize_request(raw)
                self.assertIsNone(row["attempt_e2e_ms"])
                self.assertIsNone(row["ttft_proxy_ms"])
                self.assertIsNone(row["tpot_proxy_ms"])

    def test_timing_evidence_is_required(self):
        for value in (False, None, 1):
            with self.subTest(value=value):
                row = normalize_request(request_fact(usable_for_full_attempt_timing=value))
                self.assertIsNone(row["attempt_e2e_ms"])
                self.assertIsNone(row["tpot_proxy_ms"])

    def test_output_zero_or_one_is_valid_length_but_has_no_tpot(self):
        for value in (0, 1):
            with self.subTest(value=value):
                row = normalize_request(request_fact(output_tokens=value))
                self.assertEqual(row["actual_output_tokens"], value)
                self.assertTrue(row["eligible_length_e2e"])
                self.assertIsNone(row["tpot_proxy_ms"])

    def test_invalid_tokens_are_not_silently_coerced(self):
        for value in (None, True, -1, 1.5, "4", float("nan")):
            with self.subTest(value=value):
                row = normalize_request(request_fact(output_tokens=value))
                self.assertIsNone(row["actual_output_tokens"])
                self.assertFalse(row["eligible_length_e2e"])
                self.assertIsNone(row["tpot_proxy_ms"])

    def test_nonstream_has_e2e_but_no_stream_metrics(self):
        row = normalize_request(request_fact(is_stream=False))
        self.assertEqual(row["stream_type"], "nonstream")
        self.assertTrue(row["eligible_length_e2e"])
        self.assertIsNone(row["request_ttft_proxy_ms"])
        self.assertIsNone(row["ttft_proxy_ms"])
        self.assertIsNone(row["tpot_proxy_ms"])

    def test_stream_type_must_be_boolean(self):
        for value in (None, 0, 1, "false", "true"):
            with self.subTest(value=value):
                row = normalize_request(request_fact(is_stream=value))
                self.assertEqual(row["stream_type"], "unknown")
                self.assertFalse(row["eligible_length_e2e"])

    def test_historical_arrival_fallback_and_naive_time_rejection(self):
        row = normalize_request(request_fact(request_arrival_at=None,
                                             historical_performance={
                                                 "request_arrival_at": "2026-09-21T12:00:00+08:00"}))
        self.assertIsNotNone(row["arrived_at"])
        invalid = normalize_request(request_fact(request_arrival_at="2026-09-21T12:00:00"))
        self.assertIsNone(invalid["arrived_at"])
        self.assertIsNone(invalid["finished_at"])
        self.assertFalse(invalid["eligible_length_e2e"])
        self.assertIn("invalid_arrival_time", invalid["quality_flags"])

    def test_reversed_attempt_times_do_not_produce_negative_latency(self):
        raw = request_fact()
        raw["attempts"][-1]["finished_at_ms"] = 10_050
        row = normalize_request(raw)
        self.assertIsNone(row["attempt_e2e_ms"])
        self.assertIsNone(row["tpot_proxy_ms"])
        self.assertIn("invalid_final_attempt_time", row["quality_flags"])
        invalid_end = normalize_request(request_fact(finished_at_ms=9_999))
        self.assertIsNone(invalid_end["finished_at"])
        self.assertIn("invalid_request_finished_time", invalid_end["quality_flags"])

    def test_proxy_later_than_its_own_scope_is_rejected(self):
        raw = request_fact(first_token_at_ms=501)
        raw["attempts"][-1]["first_token_at_ms"] = 301
        row = normalize_request(raw)
        self.assertIsNone(row["request_ttft_proxy_ms"])
        self.assertIsNone(row["ttft_proxy_ms"])
        self.assertIsNone(row["tpot_proxy_ms"])

    def test_elapsed_mismatch_is_flagged_without_rewriting_source(self):
        row = normalize_request(request_fact(elapsed_ms=450))
        self.assertEqual(row["request_e2e_ms"], 450)
        self.assertIn("request_elapsed_mismatch_gt_1ms", row["quality_flags"])


class DescriptiveStatisticsTests(unittest.TestCase):
    def test_sample_variance_and_interpolated_quantiles(self):
        result = describe([1, 2, 3, 4])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["variance"], 5 / 3)
        self.assertAlmostEqual(result["std"], math.sqrt(5 / 3))
        for field, expected in {"p50": 2.5, "p75": 3.25, "p90": 3.7, "p95": 3.85}.items():
            self.assertAlmostEqual(result[field], expected)

    def test_invalid_values_are_excluded_but_zero_is_kept(self):
        result = describe([None, False, True, float("nan"), float("inf"), -float("inf"), 0, 2])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["variance"], 2)

    def test_empty_and_singleton_variance_are_unknown(self):
        for values, n in (([], 0), ([5], 1)):
            with self.subTest(values=values):
                result = describe(values)
                self.assertEqual(result["n"], n)
                self.assertIsNone(result["variance"])
                self.assertIsNone(result["std"])
        self.assertIsNone(describe([])["p95"])
        self.assertEqual(describe([5])["p95"], 5)

    def test_negative_values_are_valid_for_residual_statistics(self):
        result = describe([-1, 0, 1])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["variance"], 1)


class TemporalSplitTests(unittest.TestCase):
    @staticmethod
    def row(request_id, minute, mode="stream", finish_minutes=1):
        start = datetime(2026, 9, 21, tzinfo=timezone.utc) + timedelta(minutes=minute)
        return {"request_id": request_id, "stream_type": mode,
                "arrived_at": start.isoformat(),
                "finished_at": (start + timedelta(minutes=finish_minutes)).isoformat()}

    def test_modes_are_split_independently_even_when_their_dates_differ(self):
        rows = [self.row(f"s{i}", i) for i in range(10)]
        rows += [self.row(f"n{i}", 100 + i, "nonstream") for i in range(10)]
        assignments, metadata = assign_time_splits(list(reversed(rows)))
        for prefix, mode in (("s", "stream"), ("n", "nonstream")):
            labels = [assignments[f"{prefix}{i}"] for i in range(10)]
            self.assertEqual(labels, ["train"] * 6 + ["calibration"] * 2 + ["test"] * 2)
            self.assertIn("train_cutoff", metadata[mode])
            self.assertIn("calibration_cutoff", metadata[mode])

    def test_equal_arrivals_are_never_split(self):
        minutes = [0, 1, 2, 3, 4, 5, 5, 5, 6, 7]
        rows = [self.row(f"r{i}", minute) for i, minute in enumerate(minutes)]
        assignments, _ = assign_time_splits(rows)
        self.assertEqual(len({assignments[f"r{i}"] for i in (5, 6, 7)}), 1)
        order = {"train": 0, "calibration": 1, "test": 2}
        labels = [order[assignments[f"r{i}"]] for i in range(10)]
        self.assertEqual(labels, sorted(labels))

    def test_unknown_or_invalid_arrival_is_unassigned(self):
        rows = [self.row(f"r{i}", i) for i in range(10)]
        unknown = self.row("unknown", 3, "unknown")
        missing = self.row("missing", 3)
        missing["arrived_at"] = None
        assignments, _ = assign_time_splits(rows + [unknown, missing])
        self.assertEqual(assignments["unknown"], "unassigned")
        self.assertEqual(assignments["missing"], "unassigned")

    def test_arrival_split_does_not_claim_future_completion_is_visible(self):
        rows = [self.row(f"r{i}", i) for i in range(10)]
        rows[0]["finished_at"] = self.row("future", 1_000)["finished_at"]
        assignments, metadata = assign_time_splits(rows)
        self.assertEqual(assignments["r0"], "train")
        self.assertGreater(datetime.fromisoformat(rows[0]["finished_at"]),
                           datetime.fromisoformat(metadata["stream"]["train_cutoff"]))


class ParetoContractTests(unittest.TestCase):
    def test_dominated_candidates_cannot_fill_empty_slots(self):
        result = rank_frontier([Candidate("best", 1, 1), Candidate("dominated", 2, 2)])
        self.assertEqual(result["rule"], "pareto_only_top3")
        self.assertEqual(result["frontier_ranked"], ["best"])
        self.assertEqual(result["slots"], ["best", None, None])

    def test_frontier_of_four_keeps_only_first_three_slots(self):
        candidates = [Candidate("A", 1, 4), Candidate("B", 2, 3),
                      Candidate("C", 3, 2), Candidate("D", 4, 1)]
        result = rank_frontier(candidates, lambda_cost=1)
        self.assertEqual(result["frontier_ranked"], ["A", "B", "C", "D"])
        self.assertEqual(result["slots"], ["A", "B", "C"])
        self.assertEqual(result["cost_reference"], 2.5)
        fastest = rank_frontier(candidates, lambda_cost=0)
        self.assertEqual(fastest["slots"], ["D", "C", "B"])

    def test_unavailable_candidate_neither_dominates_nor_changes_reference(self):
        result = rank_frontier([Candidate("unavailable", 0, 0, eligible=False),
                                Candidate("A", 2, 2), Candidate("B", 4, 1)])
        self.assertEqual(result["cost_reference"], 3)
        self.assertEqual(set(result["frontier_ranked"]), {"A", "B"})
        self.assertNotIn("unavailable", result["slots"])

    def test_equal_points_remain_frontier_with_deterministic_ties(self):
        result = rank_frontier([Candidate("B", 1, 1), Candidate("A", 1, 1)])
        self.assertEqual(result["slots"], ["A", "B", None])

    def test_empty_and_zero_cost_are_defined(self):
        empty = rank_frontier([])
        self.assertEqual(empty["slots"], [None, None, None])
        self.assertIsNone(empty["cost_reference"])
        zero = rank_frontier([Candidate("A", 0, 1), Candidate("B", 0, 2)])
        self.assertEqual(zero["cost_reference"], 1)
        self.assertEqual(zero["slots"], ["A", None, None])

    def test_duplicate_endpoints_and_mixed_currencies_are_rejected(self):
        for candidates in ([Candidate("A", 1, 1), Candidate("A", 2, 2)],
                           [Candidate("A", 1, 2, currency="CNY"), Candidate("B", 2, 1, currency="USD")]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                rank_frontier(candidates)

    def test_invalid_values_and_lambda_are_rejected(self):
        for value in (-1, True, float("nan"), float("inf")):
            for field in ("cost", "performance"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    Candidate("A", **{"cost": 1, "performance": 1, field: value})
        for value in (-0.1, 1.1, True, float("nan"), float("inf")):
            with self.subTest(lambda_cost=value), self.assertRaises(ValueError):
                rank_frontier([], lambda_cost=value)


if __name__ == "__main__":
    unittest.main()
