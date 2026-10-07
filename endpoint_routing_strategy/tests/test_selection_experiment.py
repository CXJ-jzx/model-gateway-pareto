import json
import csv
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from xml.etree import ElementTree

from endpoint_routing_strategy import InMemoryRoutingState, ModelRoutingPolicy, Observation, RoutingEngine
from endpoint_routing_strategy.experiments.selection.runner import DEFAULT_CONFIG, check_decision, run_suite
from endpoint_routing_strategy.experiments.selection.scenarios import load_cases


class SelectionExperimentTests(unittest.TestCase):
    @staticmethod
    def case(case_id):
        return next(case for case in load_cases(DEFAULT_CONFIG) if case.case_id == case_id)

    @staticmethod
    def route(case, **overrides):
        return RoutingEngine(case.state).route_request(
            case.request, cutoff=case.cutoff, **{**case.parameter_overrides, **overrides}
        )

    def test_config_covers_three_models_and_both_modes(self):
        cases = load_cases(DEFAULT_CONFIG)
        groups = {(c.request.model_id, c.request.is_stream) for c in cases}
        configured = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))["cases"]
        self.assertEqual(len(cases), len(configured))
        self.assertGreaterEqual(len(cases), 25)
        self.assertEqual(len(groups), 6)
        for case in cases:
            with self.subTest(case=case.case_id):
                self.assertEqual(case.request.slo.tier, {"high": "strict", "normal": "standard", "low": "relaxed"}[case.request.priority])
                decision = RoutingEngine(case.state).route_request(case.request, cutoff=case.cutoff, **case.parameter_overrides)
                self.assertEqual(check_decision(decision), [])

    def test_suite_outputs_audit_graph_and_timing_for_every_case(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "suite"
            summary = run_suite(DEFAULT_CONFIG, directory, benchmark_repeats=2)
            self.assertTrue(summary["passed"])
            self.assertEqual(summary["no_candidate_count"], 2)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["other_strategies_connected"])
            self.assertFalse(manifest["endpoint_execution_simulated"])
            self.assertFalse(manifest["backup_dispatch_performed"])
            self.assertIn("failover.py", manifest["code_sha256"])
            self.assertGreater(len(manifest["code_sha256"]), 3)
            for case in summary["cases"]:
                case_dir = directory / case["case_id"]
                with self.subTest(case=case["case_id"]):
                    self.assertTrue((case_dir / "snapshot.json").is_file())
                    self.assertTrue((case_dir / "benchmark.csv").is_file())
                    self.assertEqual(ElementTree.parse(case_dir / "pareto.svg").getroot().tag,
                                     "{http://www.w3.org/2000/svg}svg")
                    payload = json.loads((case_dir / "decision.json").read_text(encoding="utf-8"))
                    self.assertTrue(payload["validation"]["passed"])
                    self.assertEqual(case["ordered_endpoints"], payload["ordered_endpoints"])
                    self.assertEqual(case["backup_endpoints"], payload["backup_endpoints"])
                    self.assertIn("selected_estimated_request_cost", case)
                    self.assertIn("selected_stability_lower_bound", case)
                    if payload["status"] == "no_candidate":
                        self.assertEqual(payload["no_candidate_action"], "handoff_to_caller")
                        self.assertIn("slo_infeasible", payload["no_candidate_categories"])
            with (directory / "decisions.csv").open(encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), summary["case_count"])
            for csv_row, case in zip(csv_rows, summary["cases"]):
                self.assertEqual(json.loads(csv_row["ordered_endpoints"]), case["ordered_endpoints"])
            with self.assertRaises(FileExistsError):
                run_suite(DEFAULT_CONFIG, directory, benchmark_repeats=1)

    def test_default_request_cost_changes_with_input_output_length(self):
        input_long = self.route(self.case("request_input_long"))
        output_long = self.route(self.case("request_output_long"))
        self.assertEqual(input_long.cost_mode, "predicted_request")
        self.assertEqual(input_long.selected_endpoint, "input_cheap")
        self.assertEqual(output_long.selected_endpoint, "output_cheap")
        candidates = {c.endpoint_id: c for c in input_long.candidates}
        self.assertAlmostEqual(candidates["input_cheap"].estimated_request_cost, 0.0058)
        self.assertAlmostEqual(candidates["output_cheap"].estimated_request_cost, 0.0302)
        self.assertEqual(check_decision(input_long), [])
        self.assertEqual(check_decision(output_long), [])

    def test_predicted_billing_does_not_add_rho_preference(self):
        case = self.case("request_input_long")
        left, right = self.route(case, rho_input_price=0), self.route(case, rho_input_price=1)
        self.assertEqual(left.selected_endpoint, right.selected_endpoint)
        self.assertEqual([c.cost_raw for c in left.candidates], [c.cost_raw for c in right.candidates])

    def test_weighted_request_objective_is_distinct_from_total_bill(self):
        decision = self.route(self.case("request_input_long"), cost_mode="weighted_predicted_request", rho_input_price=0.75)
        self.assertEqual(check_decision(decision), [])
        candidate = next(c for c in decision.candidates if c.endpoint_id == "input_cheap")
        self.assertAlmostEqual(candidate.cost_raw, 0.75 * 0.005 + 0.25 * 0.0008)
        self.assertAlmostEqual(candidate.estimated_request_cost, 0.0058)
        self.assertNotEqual(candidate.cost_raw, candidate.estimated_request_cost)

    def test_legacy_rho_cases_explicitly_select_unit_price_mode(self):
        left, right = self.route(self.case("price_rho_0")), self.route(self.case("price_rho_1"))
        self.assertEqual(left.cost_mode, "unit_price")
        self.assertEqual(left.selected_endpoint, "output_cheap")
        self.assertEqual(right.selected_endpoint, "input_cheap")

    def test_top_three_returns_one_primary_and_two_unique_backups(self):
        decision = self.route(self.case("top_three_order"))
        self.assertEqual(len(decision.ordered_endpoints), 3)
        self.assertEqual(len(set(decision.ordered_endpoints)), 3)
        self.assertEqual(decision.ordered_endpoints[0], decision.selected_endpoint)
        self.assertEqual(decision.backup_endpoints, decision.ordered_endpoints[1:])
        self.assertEqual(check_decision(decision), [])

    def test_stable_backup_can_be_dominated_and_fill_single_frontier(self):
        decision = self.route(self.case("stable_dominated_backup"))
        candidates = {c.endpoint_id: c for c in decision.candidates}
        self.assertEqual(sum(c.pareto for c in candidates.values()), 1)
        self.assertEqual(decision.ordered_endpoints, ["primary", "stable_expensive", "reliable_second"])
        self.assertFalse(candidates["stable_expensive"].pareto)
        self.assertIsNone(candidates["stable_expensive"].score)
        self.assertIsNotNone(candidates["stable_expensive"].routing_score)
        self.assertEqual(candidates["stable_expensive"].route_role, "backup")
        self.assertEqual(check_decision(decision), [])

    def test_many_95_percent_samples_rank_ahead_of_one_perfect_sample(self):
        decision = self.route(self.case("sparse_success_vs_large_sample"))
        candidates = {c.endpoint_id: c for c in decision.candidates}
        self.assertAlmostEqual(candidates["sparse_perfect"].stability_success_rate, 1)
        self.assertAlmostEqual(candidates["many_95_percent"].stability_success_rate, 0.95)
        self.assertGreater(candidates["many_95_percent"].stability_lower_bound,
                           candidates["sparse_perfect"].stability_lower_bound)
        self.assertEqual(decision.backup_endpoints, ["many_95_percent", "sparse_perfect"])

    def test_disabled_backup_is_filtered_before_stability_sort(self):
        decision = self.route(self.case("disabled_backup"))
        self.assertNotIn("stable_expensive", decision.ordered_endpoints)
        self.assertEqual(decision.backup_endpoints, ["reliable_second", "sparse_perfect"])
        self.assertIn("disabled", decision.endpoint_checks["stable_expensive"]["reasons"])

    def test_pareto_only_backup_pool_does_not_force_three_candidates(self):
        decision = self.route(self.case("pareto_only_backup_pool"))
        self.assertEqual(decision.ordered_endpoints, ["primary"])
        self.assertEqual(decision.backup_endpoints, [])
        self.assertEqual(check_decision(decision), [])

    def test_second_phase_selects_stability_not_cheapest_remaining(self):
        decision = self.route(self.case("backup_phase_stability_first"))
        self.assertEqual(decision.selected_endpoint, "stable_expensive")
        self.assertEqual(decision.selection_phase, "backup")
        self.assertNotIn("primary", decision.ordered_endpoints)
        self.assertEqual(check_decision(decision), [])

    def test_independent_checker_detects_cost_and_order_tampering(self):
        case = self.case("stable_dominated_backup")
        decision = self.route(case)
        decision.candidates[0].estimated_input_cost += 1
        self.assertTrue(any(error.startswith("cost:") for error in check_decision(decision)))
        decision = self.route(case)
        decision.ordered_endpoints[1], decision.ordered_endpoints[2] = decision.ordered_endpoints[2], decision.ordered_endpoints[1]
        self.assertIn("backup_stability_order_inconsistent", check_decision(decision))

    def test_fixture_workload_and_urgency_match_data_pipeline_vocabulary(self):
        cases = load_cases(DEFAULT_CONFIG)
        by_case = {case.case_id: case for case in cases}
        self.assertEqual(by_case["request_input_long"].request.workload["level"], "input_heavy")
        self.assertEqual(by_case["request_output_long"].request.workload["level"], "output_heavy")
        for case in cases:
            self.assertIn(case.request.urgency["level"], {"normal", "elevated", "critical"})
            self.assertIn(case.request.workload["level"], {"light", "input_heavy", "output_heavy", "mixed_heavy"})
            self.assertEqual(case.provenance["predictions_source"], "explicit_fixture_not_observed_outcome")

    def test_invalid_experiment_config_and_repeats_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps({"schema_version": "bad", "cases": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)
            for repeats in (0, -1, True):
                with self.assertRaises(ValueError):
                    run_suite(DEFAULT_CONFIG, Path(temp) / "never_created", benchmark_repeats=repeats)

    def test_unknown_capacity_policy_is_explicit(self):
        case = load_cases(DEFAULT_CONFIG)[0]
        offering = case.state.catalog.candidates(case.request.model_id, include_ineligible=True)[0]
        case.state.register_endpoint(replace(offering, capacity_config={}))
        decision = RoutingEngine(case.state).route_request(case.request, unknown_capacity_policy="block", **case.parameter_overrides)
        reasons = decision.endpoint_checks[offering.endpoint_id]["reasons"]
        self.assertIn("capacity_unknown_rpm", reasons)
        self.assertIn("capacity_unknown_tpm", reasons)
        self.assertIn("capacity_unknown_concurrency", reasons)
        self.assertNotEqual(decision.selected_endpoint, offering.endpoint_id)
        with self.assertRaises(ValueError):
            RoutingEngine(case.state).route_request(case.request, unknown_capacity_policy="invented")

    def test_optional_stream_e2e_constraint_also_requires_enough_evidence(self):
        case = next(c for c in load_cases(DEFAULT_CONFIG) if c.case_id == "ttft_weight_0")
        state = InMemoryRoutingState()
        state.policies.upsert(ModelRoutingPolicy(model_id=case.request.model_id))
        offering = case.state.catalog.candidates(case.request.model_id)[0]
        state.register_endpoint(offering)
        for i in range(2):
            state.windows.record(case.request.model_id, Observation(
                occurred_at=case.request.arrived_at - timedelta(seconds=1), endpoint_id=offering.endpoint_id,
                is_stream=True, success=True, result="success", http_status=200,
                e2e_ms=200 if i == 0 else None, ttft_ms=10, tpot_ms=2,
            ))
        decision = RoutingEngine(state).route_request(case.request, min_samples=2, target_samples=2, prior_strength=0)
        self.assertIsNone(decision.selected_endpoint)
        self.assertIn("insufficient_samples_e2e", decision.endpoint_checks[offering.endpoint_id]["reasons"])


if __name__ == "__main__":
    unittest.main()
