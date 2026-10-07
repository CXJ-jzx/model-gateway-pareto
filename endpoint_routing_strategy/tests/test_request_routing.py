"""Request-aware routing contract and rule-consistency tests.

These are deterministic regression fixtures, not claims about real endpoint
performance and not a comparison against another routing strategy.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest

from endpoint_routing_strategy import (
    EndpointOffering,
    InMemoryRoutingState,
    ModelRoutingPolicy,
    Observation,
    Prior,
    RequestSLO,
    RoutingEngine,
    RoutingRequest,
)
from endpoint_routing_strategy.routing_engine import finalize_selection


class RequestRoutingTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
        self.state = InMemoryRoutingState(max_window_minutes=360)
        self.state.policies.upsert(ModelRoutingPolicy("model-x"))
        self.engine = RoutingEngine(self.state)

    def add_endpoint(
        self,
        endpoint,
        *,
        model="model-x",
        input_price=1.0,
        output_price=1.0,
        capacity=None,
        currency="CNY",
    ):
        self.state.register_endpoint(
            EndpointOffering(
                endpoint_id=endpoint,
                model_id=model,
                deployment_type="mock_snapshot",
                price_config={
                    "pricing_status": "configured",
                    "currency": currency,
                    "price_tiers": [
                        {
                            "currency": currency,
                            "billing_unit": "per_million_tokens",
                            "input_per_million": input_price,
                            "output_per_million": output_price,
                            "conditions": {},
                        }
                    ],
                },
                capacity_config=capacity or {},
            )
        )

    def observe(
        self,
        endpoint,
        *,
        model="model-x",
        stream=False,
        e2e=500.0,
        ttft=100.0,
        tpot=10.0,
        at=None,
        success=True,
    ):
        self.state.record_observation(
            model,
            Observation(
                occurred_at=at or self.now - timedelta(seconds=1),
                endpoint_id=endpoint,
                is_stream=stream,
                success=success,
                result="success" if success else "error",
                http_status=200 if success else 500,
                e2e_ms=e2e,
                ttft_ms=ttft if stream else None,
                tpot_ms=tpot if stream else None,
            ),
        )

    def request(self, *, stream=False, input_tokens=100, output_tokens=100, slo=None):
        return RoutingRequest(
            request_id="request-fixture",
            model_id="model-x",
            arrived_at=self.now,
            is_stream=stream,
            predicted_input_tokens=input_tokens,
            predicted_output_tokens=output_tokens,
            slo=slo or (
                RequestSLO(ttft_ms=1000, tpot_ms=100)
                if stream
                else RequestSLO(e2e_ms=1000)
            ),
            priority="normal",
            workload={"level": "light"},
            urgency={"level": "normal"},
        )

    def route(self, request=None, **overrides):
        parameters = {"target_samples": 1, "min_samples": 1, "prior_strength": 0}
        parameters.update(overrides)
        return self.engine.route_request(request or self.request(), **parameters)

    @staticmethod
    def by_endpoint(decision):
        return {candidate.endpoint_id: candidate for candidate in decision.candidates}

    def test_lambda_changes_cost_performance_preference(self):
        self.add_endpoint("cheap", input_price=1, output_price=1)
        self.add_endpoint("fast", input_price=3, output_price=3)
        self.observe("cheap", e2e=900)
        self.observe("fast", e2e=200)
        self.assertEqual(self.route(lambda_cost=0).selected_endpoint, "fast")
        self.assertEqual(self.route(lambda_cost=1).selected_endpoint, "cheap")

    def test_eta_changes_ttft_tpot_preference(self):
        self.add_endpoint("fast_first_token")
        self.add_endpoint("fast_generation")
        self.observe("fast_first_token", stream=True, ttft=100, tpot=80)
        self.observe("fast_generation", stream=True, ttft=800, tpot=10)
        request = self.request(stream=True)
        self.assertEqual(
            self.route(request, eta_ttft=1, lambda_cost=0).selected_endpoint,
            "fast_first_token",
        )
        self.assertEqual(
            self.route(request, eta_ttft=0, lambda_cost=0).selected_endpoint,
            "fast_generation",
        )

    def test_rho_changes_input_output_unit_price_preference(self):
        self.add_endpoint("cheap_input", input_price=1, output_price=10)
        self.add_endpoint("cheap_output", input_price=10, output_price=1)
        self.observe("cheap_input")
        self.observe("cheap_output")
        self.assertEqual(
            self.route(cost_mode="unit_price", rho_input_price=1, lambda_cost=1).selected_endpoint, "cheap_input"
        )
        self.assertEqual(
            self.route(cost_mode="unit_price", rho_input_price=0, lambda_cost=1).selected_endpoint, "cheap_output"
        )

    def test_pareto_dominated_endpoint_is_not_selectable(self):
        self.add_endpoint("good", input_price=1, output_price=1)
        self.add_endpoint("dominated", input_price=2, output_price=2)
        self.observe("good", e2e=200)
        self.observe("dominated", e2e=700)
        decision = self.route()
        candidates = self.by_endpoint(decision)
        self.assertEqual(decision.selected_endpoint, "good")
        self.assertTrue(candidates["good"].pareto)
        self.assertFalse(candidates["dominated"].pareto)
        self.assertIsNone(candidates["dominated"].score)

    def test_reusing_candidates_clears_previous_selection_and_scores(self):
        self.add_endpoint("cheap", input_price=1, output_price=1)
        self.add_endpoint("fast", input_price=3, output_price=3)
        self.observe("cheap", e2e=900)
        self.observe("fast", e2e=200)
        decision = self.route(lambda_cost=0)
        candidates = self.by_endpoint(decision)
        self.assertTrue(candidates["fast"].selected)
        second = finalize_selection(decision.candidates, replace(decision.parameters, lambda_cost=1))
        self.assertEqual(second.endpoint_id, "cheap")
        self.assertTrue(candidates["cheap"].selected)
        self.assertFalse(candidates["fast"].selected)
        candidates["cheap"].feasible = False
        third = finalize_selection(decision.candidates, decision.parameters)
        self.assertEqual(third.endpoint_id, "fast")
        self.assertFalse(candidates["cheap"].pareto)
        self.assertIsNone(candidates["cheap"].score)
        self.assertEqual(candidates["cheap"].selection_probability, 0)

    def test_score_and_cost_normalization_have_explicit_units(self):
        self.add_endpoint("cheap", input_price=1, output_price=1)
        self.add_endpoint("fast", input_price=3, output_price=3)
        self.observe("cheap", e2e=900)
        self.observe("fast", e2e=200)
        decision = self.route(cost_mode="unit_price", lambda_cost=0.25)
        self.assertAlmostEqual(decision.cost_reference, 2)
        for candidate in decision.candidates:
            self.assertAlmostEqual(candidate.cost_normalized, candidate.cost_raw / 2)
            if candidate.pareto:
                self.assertAlmostEqual(
                    candidate.score,
                    0.25 * candidate.cost_normalized + 0.75 * candidate.performance,
                )

    def test_predicted_token_length_is_not_the_unit_price_cost(self):
        self.add_endpoint("available", input_price=2, output_price=8)
        self.observe("available")
        short = self.route(self.request(input_tokens=1, output_tokens=1), cost_mode="unit_price", rho_input_price=0.25)
        long = self.route(self.request(input_tokens=10000, output_tokens=5000), cost_mode="unit_price", rho_input_price=0.25)
        self.assertAlmostEqual(short.candidates[0].cost_raw, 6.5)
        self.assertEqual(short.candidates[0].cost_raw, long.candidates[0].cost_raw)

    def test_each_projected_capacity_accepts_equality_and_rejects_excess(self):
        for metric, limit, increment in (("rpm", 10, 1), ("tpm", 1000, 200), ("concurrency", 5, 1)):
            with self.subTest(metric=metric):
                state = InMemoryRoutingState()
                state.policies.upsert(ModelRoutingPolicy("model-x"))
                original_state, original_engine = self.state, self.engine
                self.state, self.engine = state, RoutingEngine(state)
                try:
                    self.add_endpoint("limited", capacity={metric: limit})
                    self.observe("limited")
                    self.state.catalog.update_model_endpoint_load(
                        "model-x", "limited", **{f"current_{metric}": limit - increment}
                    )
                    boundary = self.route()
                    self.assertEqual(boundary.selected_endpoint, "limited")
                    self.assertEqual(boundary.endpoint_checks["limited"]["capacity"][metric]["projected"], limit)
                    self.state.catalog.update_model_endpoint_load(
                        "model-x", "limited", **{f"current_{metric}": limit - increment + 1}
                    )
                    over = self.route()
                    self.assertIsNone(over.selected_endpoint)
                    self.assertIn(f"capacity_{metric}", over.endpoint_checks["limited"]["reasons"])
                finally:
                    self.state, self.engine = original_state, original_engine

    def test_projected_tpm_counts_predicted_input_plus_output(self):
        self.add_endpoint("limited", capacity={"tpm": 1000})
        self.observe("limited")
        self.state.catalog.update_model_endpoint_load("model-x", "limited", current_tpm=850)
        decision = self.route(self.request(input_tokens=100, output_tokens=100))
        capacity = decision.endpoint_checks["limited"]["capacity"]["tpm"]
        self.assertEqual(capacity["increment"], 200)
        self.assertEqual(capacity["projected"], 1050)
        self.assertIn("capacity_tpm", decision.endpoint_checks["limited"]["reasons"])

    def test_zero_configured_capacity_is_unavailable_not_unknown(self):
        self.add_endpoint("zero", capacity={"rpm": 0})
        self.observe("zero")
        decision = self.route()
        self.assertIsNone(decision.selected_endpoint)
        self.assertIn("capacity_rpm", decision.endpoint_checks["zero"]["reasons"])

    def test_endpoint_aggregate_load_does_not_replace_pair_capacity(self):
        self.add_endpoint("available", capacity={"rpm": 10, "tpm": 1000, "concurrency": 5})
        self.observe("available")
        self.state.catalog.update_endpoint_runtime(
            "available", current_rpm=99999, current_tpm=99999, current_concurrency=99999
        )
        self.assertEqual(self.route().selected_endpoint, "available")

    def test_disabled_cooldown_and_unhealthy_are_explained(self):
        for endpoint in ("disabled", "cooling", "unhealthy", "healthy"):
            self.add_endpoint(endpoint)
            self.observe(endpoint)
        self.state.catalog.set_enabled("model-x", "disabled", False)
        self.state.catalog.set_cooldown("model-x", "cooling", self.now + timedelta(seconds=10))
        self.state.catalog.update_endpoint_runtime("unhealthy", health_status="down")
        decision = self.route()
        self.assertEqual(decision.selected_endpoint, "healthy")
        for endpoint, reason in (("disabled", "disabled"), ("cooling", "cooldown"), ("unhealthy", "endpoint_unhealthy")):
            self.assertIn(reason, decision.endpoint_checks[endpoint]["reasons"])

    def test_cooldown_deadline_itself_is_available(self):
        self.add_endpoint("available")
        self.observe("available")
        self.state.catalog.set_cooldown("model-x", "available", self.now)
        self.assertEqual(self.route().selected_endpoint, "available")

    def test_nonstream_slo_filter_runs_before_cost_choice(self):
        self.add_endpoint("cheap_slow", input_price=0.1, output_price=0.1)
        self.add_endpoint("within_slo", input_price=10, output_price=10)
        self.observe("cheap_slow", e2e=1100)
        self.observe("within_slo", e2e=900)
        decision = self.route(lambda_cost=1)
        self.assertEqual(decision.selected_endpoint, "within_slo")
        self.assertIn("slo_e2e", decision.endpoint_checks["cheap_slow"]["reasons"])

    def test_stream_checks_each_metric_even_when_score_weight_is_zero(self):
        self.add_endpoint("bad_tpot")
        self.observe("bad_tpot", stream=True, ttft=10, tpot=101)
        decision = self.route(self.request(stream=True), eta_ttft=1)
        self.assertIsNone(decision.selected_endpoint)
        self.assertIn("slo_tpot", decision.endpoint_checks["bad_tpot"]["reasons"])
        self.assertLess(self.by_endpoint(decision)["bad_tpot"].performance, 1)

    def test_stream_ttft_slo_rejected_independently(self):
        self.add_endpoint("bad_ttft")
        self.observe("bad_ttft", stream=True, ttft=1001, tpot=1)
        decision = self.route(self.request(stream=True), eta_ttft=0)
        self.assertIn("slo_ttft", decision.endpoint_checks["bad_ttft"]["reasons"])
        self.assertIsNone(decision.selected_endpoint)

    def test_explicit_slo_filter_opt_out_keeps_metrics_visible(self):
        self.add_endpoint("slow")
        self.observe("slow", e2e=1500)
        decision = self.route(require_slo=False)
        self.assertEqual(decision.selected_endpoint, "slow")
        self.assertGreater(decision.endpoint_checks["slow"]["slo"]["e2e"]["ratio"], 1)

    def test_prior_only_has_estimated_metrics_for_slo(self):
        self.add_endpoint("prior")
        self.state.catalog.set_prior(
            "model-x", "prior", Prior("prior", 600, 100, 10, 0.99, 100, "fixture")
        )
        decision = self.route(prior_strength=30)
        self.assertEqual(decision.selected_endpoint, "prior")
        self.assertEqual(decision.candidates[0].sample_count, 0)
        self.assertAlmostEqual(decision.candidates[0].estimated_e2e_ms, 600)
        self.assertIsNone(decision.candidates[0].p95_e2e_ms)

    def test_missing_stream_metric_is_rejected(self):
        self.add_endpoint("missing")
        self.observe("missing", stream=True, ttft=100, tpot=None)
        decision = self.route(self.request(stream=True))
        self.assertIsNone(decision.selected_endpoint)
        self.assertIn("missing_tpot", decision.endpoint_checks["missing"]["reasons"])

    def test_disabling_slo_filter_does_not_hide_missing_metrics(self):
        self.add_endpoint("missing")
        self.observe("missing", stream=True, ttft=100, tpot=None)
        decision = self.route(self.request(stream=True), require_slo=False, eta_ttft=1)
        self.assertIsNone(decision.selected_endpoint)
        self.assertIn("missing_tpot", decision.endpoint_checks["missing"]["reasons"])

    def test_per_metric_prior_fusion_is_used_for_slo_guard(self):
        self.add_endpoint("blended")
        self.observe("blended", e2e=900)
        self.state.catalog.set_prior(
            "model-x", "blended", Prior("blended", 1500, None, None, 0.99, 100, "fixture")
        )
        decision = self.route(prior_strength=1)
        candidate = self.by_endpoint(decision)["blended"]
        self.assertEqual(candidate.p95_e2e_ms, 900)
        self.assertAlmostEqual(candidate.estimated_e2e_ms, 1200)
        self.assertIsNone(decision.selected_endpoint)
        self.assertIn("slo_e2e", decision.endpoint_checks["blended"]["reasons"])

    def test_future_sample_does_not_enter_arrival_decision(self):
        self.add_endpoint("available")
        self.observe("available", e2e=500)
        self.observe("available", e2e=9000, at=self.now + timedelta(seconds=1))
        decision = self.route()
        self.assertEqual(decision.selected_endpoint, "available")
        self.assertEqual(decision.candidates[0].sample_count, 1)
        self.assertEqual(decision.candidates[0].p95_e2e_ms, 500)

    def test_model_and_stream_windows_are_isolated(self):
        self.add_endpoint("shared")
        self.add_endpoint("shared", model="model-y")
        self.add_endpoint("only_other_model", model="model-y")
        self.observe("shared", model="model-y", e2e=9000)
        self.observe("shared", stream=True, e2e=9000, ttft=9000, tpot=9000)
        self.observe("shared", e2e=500)
        decision = self.route()
        self.assertEqual(set(decision.endpoint_checks), {"shared"})
        self.assertEqual(decision.selected_endpoint, "shared")
        self.assertEqual(decision.candidates[0].sample_count, 1)
        self.assertEqual(decision.candidates[0].p95_e2e_ms, 500)

    def test_no_candidate_is_a_structured_result(self):
        decision = self.route()
        self.assertEqual(decision.status, "no_candidate")
        self.assertIsNone(decision.selected_endpoint)
        self.assertEqual(decision.candidates, [])
        self.assertIsNone(decision.cost_reference)
        json.dumps(decision.to_dict(), allow_nan=False)

    def test_missing_or_other_currency_price_is_explained(self):
        self.add_endpoint("usd", currency="USD")
        self.observe("usd")
        decision = self.route(currency="CNY")
        self.assertIsNone(decision.selected_endpoint)
        self.assertIn("missing_price", decision.endpoint_checks["usd"]["reasons"])

    def test_cutoff_cannot_precede_arrival(self):
        with self.assertRaises(ValueError):
            self.route(cutoff=self.now - timedelta(microseconds=1))

    def test_wait_consumes_e2e_and_ttft_but_not_tpot_budget(self):
        self.add_endpoint("available")
        self.observe("available", stream=True, ttft=600, tpot=10)
        decision = self.route(self.request(stream=True), cutoff=self.now + timedelta(milliseconds=500))
        self.assertIn("slo_ttft", decision.endpoint_checks["available"]["reasons"])
        self.assertEqual(decision.endpoint_checks["available"]["slo"]["ttft"]["budget_ms"], 500)
        self.assertEqual(decision.endpoint_checks["available"]["slo"]["tpot"]["budget_ms"], 100)

    def test_invalid_preferences_and_window_parameters_fail_fast(self):
        for overrides in (
            {"lambda_cost": -0.1}, {"eta_ttft": 1.1}, {"rho_input_price": float("nan")},
            {"half_life_minutes": 0}, {"target_samples": 0}, {"min_samples": -1},
            {"window_candidates_minutes": ()}, {"prior_strength": -1},
            {"temperature": 100}, {"stream": True}, {"slo_e2e_ms": 2000},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.route(**overrides)

    def test_invalid_request_tokens_and_timezone_fail_fast(self):
        for changes in (
            {"predicted_input_tokens": -1}, {"predicted_output_tokens": -1},
            {"predicted_input_tokens": 1.5}, {"predicted_output_tokens": True},
            {"arrived_at": self.now.replace(tzinfo=None)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.route(replace(self.request(), **changes))

    def test_missing_or_invalid_required_slo_fails_fast(self):
        for stream, slo_fields in (
            (False, {}), (False, {"e2e_ms": 0}),
            (False, {"e2e_ms": float("nan")}),
            (True, {"ttft_ms": 100}), (True, {"tpot_ms": 10}),
        ):
            with self.subTest(stream=stream, slo=slo_fields), self.assertRaises(ValueError):
                self.route(self.request(stream=stream, slo=RequestSLO(**slo_fields)))

    def test_prepared_request_adapter_ignores_completed_result_fields(self):
        row = {
            "schema_version": "1.0", "request_id": "prepared-request", "model_id": "model-x",
            "arrival_time": self.now.isoformat(), "stream_type": "stream", "priority": "high",
            "token_predictions": {"input_tokens": 200, "output_tokens": 400,
                                  "input_source": "mock", "output_source": "mock"},
            "slo": {"tier": "strict", "e2e_ms": 3000, "ttft_ms": 1000, "tpot_ms": 100},
            "workload": {"level": "light"}, "urgency": {"level": "normal"},
            "actual_input_tokens": 99999, "actual_output_tokens": 99999,
            "final_endpoint_id": "must_not_influence_routing",
        }
        request = RoutingRequest.from_prepared(row)
        self.assertEqual(request.predicted_input_tokens, 200)
        self.assertEqual(request.predicted_output_tokens, 400)
        self.assertTrue(request.is_stream)
        self.assertEqual(request.slo.tier, "strict")
        self.assertEqual(request.priority, "high")
        self.add_endpoint("available")
        self.observe("available", stream=True)
        self.assertEqual(self.route(request).selected_endpoint, "available")

    def test_decision_is_deterministic_json_safe_and_does_not_reserve_load(self):
        for endpoint in ("a", "b"):
            self.add_endpoint(endpoint)
            self.observe(endpoint)
        before = self.state.catalog.runtime_state("model-x", "a")
        first, second = self.route(), self.route()
        self.assertEqual(first.selected_endpoint, "a")
        self.assertEqual(second.selected_endpoint, "a")
        self.assertEqual(first.parameters.temperature, 0)
        self.assertEqual(before, self.state.catalog.runtime_state("model-x", "a"))
        self.assertTrue(first.timings_us)
        self.assertTrue(all(value >= 0 for value in first.timings_us.values()))
        json.dumps(first.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
