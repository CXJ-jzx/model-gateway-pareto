"""Predicted-request costs and stability-first backup plan regression tests.

These fixtures exercise only our routing policy.  They do not simulate an
endpoint executor or treat a backup recommendation as a reserved capacity slot.
"""

from datetime import datetime, timedelta, timezone
import json
import unittest

from endpoint_routing_strategy import (
    EndpointOffering,
    InMemoryRoutingState,
    ModelRoutingPolicy,
    Observation,
    RequestSLO,
    RoutingEngine,
    RoutingRequest,
)


class CostAndBackupTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
        self.state = InMemoryRoutingState(max_window_minutes=360)
        self.state.policies.upsert(ModelRoutingPolicy("model-cost"))
        self.engine = RoutingEngine(self.state)

    def add(self, endpoint, *, pin=1.0, pout=1.0, e2e=300.0,
            successes=10, failures=0, capacity=None, status=500):
        self.state.register_endpoint(EndpointOffering(
            endpoint_id=endpoint,
            model_id="model-cost",
            deployment_type="regression_snapshot",
            price_config={
                "pricing_status": "configured", "currency": "CNY",
                "price_tiers": [{
                    "currency": "CNY", "billing_unit": "per_million_tokens",
                    "input_per_million": pin, "output_per_million": pout,
                    "conditions": {},
                }],
            },
            capacity_config=capacity or {},
        ))
        for i in range(successes + failures):
            success = i < successes
            self.state.record_observation("model-cost", Observation(
                occurred_at=self.now - timedelta(seconds=1),
                endpoint_id=endpoint, is_stream=False, success=success,
                result="success" if success else "error",
                http_status=200 if success else status,
                e2e_ms=e2e if success else None, ttft_ms=None, tpot_ms=None,
            ))

    def request(self, tin=100, tout=100):
        return RoutingRequest(
            request_id="cost-backup-request", model_id="model-cost",
            arrived_at=self.now, is_stream=False,
            predicted_input_tokens=tin, predicted_output_tokens=tout,
            slo=RequestSLO(e2e_ms=1000),
        )

    def route(self, request=None, **kwargs):
        parameters = dict(target_samples=1, min_samples=1, prior_strength=0)
        parameters.update(kwargs)
        return self.engine.route_request(request or self.request(), **parameters)

    @staticmethod
    def candidates(decision):
        return {candidate.endpoint_id: candidate for candidate in decision.candidates}

    def three(self):
        # The primary is strictly cheaper and faster.  Backups deliberately are
        # dominated in the two-dimensional cost/performance objective.
        self.add("primary", pin=1, pout=1, e2e=100, successes=10)
        self.add("reliable", pin=3, pout=3, e2e=400, successes=99, failures=1)
        self.add("tiny-perfect", pin=2, pout=2, e2e=300, successes=1)

    def test_default_cost_uses_predicted_lengths_and_changes_primary(self):
        self.add("cheap-input", pin=1, pout=10)
        self.add("cheap-output", pin=10, pout=1)
        input_heavy = self.route(self.request(1000, 10), lambda_cost=1)
        output_heavy = self.route(self.request(10, 1000), lambda_cost=1)
        self.assertEqual(input_heavy.cost_mode, "predicted_request")
        self.assertEqual(input_heavy.selected_endpoint, "cheap-input")
        self.assertEqual(output_heavy.selected_endpoint, "cheap-output")

    def test_true_request_bill_contains_unweighted_cost_components(self):
        self.add("priced", pin=2, pout=8)
        candidate = self.route(self.request(2000, 500), rho_input_price=0.1).candidates[0]
        self.assertAlmostEqual(candidate.estimated_input_cost, 0.004)
        self.assertAlmostEqual(candidate.estimated_output_cost, 0.004)
        self.assertAlmostEqual(candidate.estimated_request_cost, 0.008)
        self.assertAlmostEqual(candidate.cost_raw, 0.008)
        self.assertEqual(candidate.cost_mode, "predicted_request")

    def test_rho_does_not_change_true_predicted_request_bill(self):
        self.add("input", pin=1, pout=10)
        self.add("output", pin=10, pout=1)
        first = self.route(self.request(1000, 10), rho_input_price=0, lambda_cost=1)
        second = self.route(self.request(1000, 10), rho_input_price=1, lambda_cost=1)
        self.assertEqual(first.selected_endpoint, second.selected_endpoint)
        for ep, candidate in self.candidates(first).items():
            self.assertAlmostEqual(candidate.cost_raw, self.candidates(second)[ep].cost_raw)

    def test_weighted_predicted_mode_exposes_preference_not_bill(self):
        self.add("input", pin=1, pout=10)
        self.add("output", pin=10, pout=1)
        first = self.route(cost_mode="weighted_predicted_request", rho_input_price=1, lambda_cost=1)
        second = self.route(cost_mode="weighted_predicted_request", rho_input_price=0, lambda_cost=1)
        self.assertEqual(first.selected_endpoint, "input")
        self.assertEqual(second.selected_endpoint, "output")
        candidate = self.candidates(first)["input"]
        self.assertAlmostEqual(candidate.cost_raw, 100 / 1_000_000)
        self.assertAlmostEqual(candidate.estimated_request_cost, 1100 / 1_000_000)

    def test_unit_price_mode_retains_previous_length_independent_cost(self):
        self.add("priced", pin=2, pout=8)
        short = self.route(self.request(1, 1), cost_mode="unit_price", rho_input_price=0.25)
        long = self.route(self.request(10000, 5000), cost_mode="unit_price", rho_input_price=0.25)
        self.assertAlmostEqual(short.candidates[0].cost_raw, 6.5)
        self.assertEqual(short.candidates[0].cost_raw, long.candidates[0].cost_raw)
        self.assertAlmostEqual(short.candidates[0].unit_price_cost, 6.5)

    def test_zero_token_request_has_zero_finite_cost(self):
        self.add("free-prediction", pin=2, pout=8)
        decision = self.route(self.request(0, 0))
        self.assertEqual(decision.selected_endpoint, "free-prediction")
        candidate = decision.candidates[0]
        self.assertEqual(candidate.cost_raw, 0)
        self.assertEqual(candidate.estimated_request_cost, 0)
        self.assertEqual(candidate.cost_normalized, 0)
        self.assertEqual(decision.cost_reference, 1)
        json.dumps(decision.to_dict(), allow_nan=False)

    def test_top_three_has_unique_primary_then_stability_ordered_backups(self):
        self.three()
        decision = self.route()
        self.assertEqual(decision.ordered_endpoints, ["primary", "reliable", "tiny-perfect"])
        self.assertEqual(decision.backup_endpoints, ["reliable", "tiny-perfect"])
        self.assertEqual(len(decision.ordered_endpoints), len(set(decision.ordered_endpoints)))
        self.assertEqual(decision.selected_endpoint, decision.ordered_endpoints[0])
        points = self.candidates(decision)
        self.assertEqual(points["primary"].route_rank, 1)
        self.assertEqual(points["primary"].route_role, "primary")
        self.assertEqual(points["reliable"].route_rank, 2)
        self.assertEqual(points["reliable"].route_role, "backup")

    def test_backup_confidence_prefers_99_of_100_over_one_of_one(self):
        self.three()
        decision = self.route()
        points = self.candidates(decision)
        self.assertLess(points["reliable"].stability_success_rate, points["tiny-perfect"].stability_success_rate)
        self.assertGreater(points["reliable"].stability_lower_bound, points["tiny-perfect"].stability_lower_bound)
        self.assertGreater(points["reliable"].stability_sample_count, points["tiny-perfect"].stability_sample_count)
        self.assertEqual(decision.backup_endpoints[0], "reliable")

    def test_dominated_endpoint_can_be_stable_backup_without_pareto_label(self):
        self.three()
        decision = self.route()
        points = self.candidates(decision)
        self.assertFalse(points["reliable"].pareto)
        self.assertIsNone(points["reliable"].score)
        self.assertIsNotNone(points["reliable"].routing_score)
        self.assertIn("reliable", decision.backup_endpoints)

    def test_pareto_only_backup_pool_does_not_invent_three_frontier_points(self):
        self.three()
        decision = self.route(backup_pool="pareto")
        self.assertEqual(decision.ordered_endpoints, ["primary"])
        self.assertEqual(decision.backup_endpoints, [])

    def test_top_k_does_not_change_primary_choice(self):
        self.three()
        one = self.route(top_k=1)
        three = self.route(top_k=3)
        self.assertEqual(one.selected_endpoint, three.selected_endpoint)
        self.assertEqual(one.ordered_endpoints, ["primary"])
        self.assertEqual(one.backup_endpoints, [])

    def test_fewer_available_candidates_returns_shorter_plan(self):
        self.add("only")
        decision = self.route(top_k=3)
        self.assertEqual(decision.ordered_endpoints, ["only"])
        self.assertEqual(decision.backup_endpoints, [])

    def test_disabled_slo_and_projected_capacity_failures_are_not_backups(self):
        self.add("primary", pin=1, pout=1, e2e=100)
        self.add("disabled", pin=2, pout=2)
        self.add("slo-failure", pin=2, pout=2, e2e=1500)
        self.add("full", pin=2, pout=2, capacity={"concurrency": 1})
        self.state.catalog.set_enabled("model-cost", "disabled", False)
        self.state.catalog.update_model_endpoint_load("model-cost", "full", current_concurrency=1)
        decision = self.route()
        self.assertEqual(decision.ordered_endpoints, ["primary"])
        for ep in ("disabled", "slo-failure", "full"):
            candidate = self.candidates(decision)[ep]
            self.assertFalse(candidate.feasible)
            self.assertIsNone(candidate.route_rank)
            self.assertEqual(candidate.route_role, "none")

    def test_no_candidate_produces_empty_plan_and_json_trace(self):
        self.add("disabled")
        self.state.catalog.set_enabled("model-cost", "disabled", False)
        decision = self.route()
        self.assertEqual(decision.status, "no_candidate")
        self.assertIsNone(decision.selected_endpoint)
        self.assertEqual(decision.ordered_endpoints, [])
        self.assertEqual(decision.backup_endpoints, [])
        json.dumps(decision.to_dict(), allow_nan=False)

    def test_explicit_exclusion_removes_previous_primary_from_plan(self):
        self.three()
        decision = self.route(exclude_endpoints=("primary",))
        self.assertNotIn("primary", decision.ordered_endpoints)
        self.assertNotEqual(decision.selected_endpoint, "primary")

    def test_allowed_pool_is_revalidated_and_exclusion_wins(self):
        self.three()
        decision = self.route(allowed_endpoints=("primary", "reliable"), exclude_endpoints=("primary",))
        self.assertEqual(decision.ordered_endpoints, ["reliable"])
        self.state.catalog.set_enabled("model-cost", "reliable", False)
        stale = self.route(allowed_endpoints=("reliable",))
        self.assertEqual(stale.ordered_endpoints, [])

    def test_empty_allowed_pool_does_not_mean_all_endpoints(self):
        self.three()
        self.assertEqual(self.route(allowed_endpoints=()).ordered_endpoints, [])

    def test_plan_trace_repeats_without_stale_ranks_and_is_json_serializable(self):
        self.three()
        first = self.route(top_k=3)
        second = self.route(top_k=1)
        self.assertEqual(first.ordered_endpoints, ["primary", "reliable", "tiny-perfect"])
        self.assertEqual(second.ordered_endpoints, ["primary"])
        for candidate in second.candidates:
            if candidate.endpoint_id != "primary":
                self.assertIsNone(candidate.route_rank)
                self.assertEqual(candidate.route_role, "none")
        trace = second.to_dict()
        self.assertEqual(trace["cost_mode"], "predicted_request")
        self.assertEqual(trace["ordered_endpoints"], ["primary"])
        self.assertEqual(trace["backup_endpoints"], [])
        json.dumps(trace, allow_nan=False)

    def test_stability_combines_stream_and_nonstream_for_the_same_pair(self):
        self.add("primary", pin=1, pout=1, e2e=100)
        self.add("stream-unreliable", pin=2, pout=2, e2e=300)
        self.add("pair-reliable", pin=3, pout=3, e2e=400)
        for _ in range(90):
            self.state.record_observation("model-cost", Observation(
                occurred_at=self.now - timedelta(seconds=1),
                endpoint_id="stream-unreliable", is_stream=True,
                success=False, result="rate_limit", http_status=429,
                e2e_ms=None, ttft_ms=None, tpot_ms=None,
            ))
        decision = self.route()
        points = self.candidates(decision)
        self.assertTrue(points["stream-unreliable"].feasible)
        self.assertAlmostEqual(points["stream-unreliable"].stability_success_rate, 0.1)
        self.assertAlmostEqual(points["stream-unreliable"].stability_429_rate, 0.9)
        self.assertEqual(points["stream-unreliable"].stability_5xx_rate, 0)
        self.assertEqual(decision.backup_endpoints, ["pair-reliable", "stream-unreliable"])

    def test_future_completion_does_not_change_stability(self):
        self.three()
        before = self.route()
        for _ in range(100):
            self.state.record_observation("model-cost", Observation(
                occurred_at=self.now + timedelta(seconds=1),
                endpoint_id="reliable", is_stream=False,
                success=False, result="error", http_status=500,
                e2e_ms=None, ttft_ms=None, tpot_ms=None,
            ))
        after = self.route()
        old, new = self.candidates(before)["reliable"], self.candidates(after)["reliable"]
        self.assertEqual(old.stability_sample_count, new.stability_sample_count)
        self.assertEqual(old.stability_success_rate, new.stability_success_rate)
        self.assertEqual(old.stability_lower_bound, new.stability_lower_bound)
        self.assertEqual(before.ordered_endpoints, after.ordered_endpoints)

    def test_backup_phase_selects_stability_over_pareto_score(self):
        self.three()
        initial = self.route()
        retry = self.route(
            selection_phase="backup", allowed_endpoints=initial.backup_endpoints,
            exclude_endpoints=(initial.selected_endpoint,),
        )
        self.assertEqual(retry.selected_endpoint, "reliable")
        self.assertFalse(self.candidates(retry)["reliable"].pareto)
        self.assertEqual(retry.ordered_endpoints[0], "reliable")

    def test_none_options_inherit_model_policy_defaults(self):
        self.three()
        decision = self.route(cost_mode=None, top_k=None, backup_pool=None, stability_z=None)
        self.assertEqual(decision.cost_mode, "predicted_request")
        self.assertEqual(decision.top_k, 3)
        self.assertEqual(decision.backup_pool, "all_feasible")
        self.assertEqual(decision.stability_z, 1.96)

    def test_invalid_cost_modes_rejected(self):
        for mode in ("total_tokens_price", ""):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.route(cost_mode=mode)

    def test_invalid_top_k_rejected(self):
        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.route(top_k=top_k)

    def test_invalid_backup_pool_rejected(self):
        for pool in ("random", ""):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                self.route(backup_pool=pool)

    def test_invalid_stability_z_rejected(self):
        for z in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(z=z), self.assertRaises(ValueError):
                self.route(stability_z=z)


if __name__ == "__main__":
    unittest.main()
