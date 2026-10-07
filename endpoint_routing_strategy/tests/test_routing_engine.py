"""Core statistics, Pareto selection and in-memory state regression tests."""
import unittest
from datetime import datetime, timedelta, timezone

from endpoint_routing_strategy.memory_state import (
    InMemoryRoutingState,
    ModelEndpointRegistry,
    ModelEndpointStateIndex,
)
from endpoint_routing_strategy.models import (
    Candidate,
    EndpointOffering,
    ModelRoutingPolicy,
    Observation,
    Price,
    Prior,
    StrategyParameters,
)
from endpoint_routing_strategy.routing_engine import (
    RoutingEngine,
    combine_stream_performance,
    compute_candidate,
    effective_sample_size,
    finalize_selection,
    pareto_mask,
    weighted_quantile,
)


def make_candidate(endpoint: str, cost: float, performance: float) -> Candidate:
    return Candidate(
        endpoint_id=endpoint,
        currency="CNY",
        input_per_million=cost,
        output_per_million=cost,
        cost_raw=cost,
        cost_normalized=None,
        performance=performance,
        online_performance=performance,
        prior_performance=None,
        p95_e2e_ms=performance * 1000,
        p95_ttft_ms=None,
        p95_tpot_ms=None,
        success_rate=0.99,
        online_success_rate=0.99,
        sample_count=100,
        effective_sample_count=100,
        window_minutes=30,
        prior_weight=0,
        feasible=True,
        exclusion_reason="",
    )


def make_offering(endpoint: str, model: str = "model-x", tiers: int = 2) -> EndpointOffering:
    price_tiers = [
        {
            "currency": "CNY",
            "billing_unit": "per_million_tokens",
            "input_per_million": float(index + 1),
            "output_per_million": float((index + 1) * 2),
            "conditions": {"tier": index},
        }
        for index in range(tiers)
    ]
    return EndpointOffering(
        endpoint_id=endpoint,
        model_id=model,
        deployment_type="External API",
        price_config={
            "pricing_status": "configured",
            "currency": "CNY",
            "price_tiers": price_tiers,
        },
        capacity_config={"rpm": 100, "tpm": 10_000, "concurrency": 5},
    )


def make_observation(
    endpoint: str,
    occurred_at: datetime,
    *,
    stream: bool = False,
    status: int = 200,
    e2e_ms: float | None = 1000,
) -> Observation:
    success = status == 200
    return Observation(
        occurred_at=occurred_at,
        endpoint_id=endpoint,
        is_stream=stream,
        success=success,
        result="success" if success else "error",
        http_status=status,
        e2e_ms=e2e_ms if success else None,
        ttft_ms=100 if stream and success else None,
        tpot_ms=10 if stream and success else None,
    )


class StatisticsTests(unittest.TestCase):
    def test_weighted_quantile(self):
        self.assertEqual(weighted_quantile([(1, 1), (2, 1), (10, 8)], 0.5), 10)
        self.assertEqual(weighted_quantile([(1, 1), (2, 1), (10, 8)], 0.1), 1)

    def test_effective_sample_size(self):
        self.assertAlmostEqual(effective_sample_size([1, 1, 1, 1]), 4.0)
        self.assertLess(effective_sample_size([1, 0.01, 0.01]), 2.0)

    def test_stream_performance_is_normalized(self):
        value = combine_stream_performance(5000, 100, 0.5, 5000, 100)
        self.assertAlmostEqual(value, 1.0)


class ParetoTests(unittest.TestCase):
    def test_pareto_removes_dominated_point(self):
        candidates = [
            make_candidate("A", 1, 1),
            make_candidate("B", 2, 2),
            make_candidate("C", 0.5, 3),
        ]
        self.assertEqual(pareto_mask(candidates), {"A", "C"})

    def test_linear_selection_changes_with_lambda(self):
        performance_first = [make_candidate("cheap", 1, 3), make_candidate("fast", 3, 1)]
        selected = finalize_selection(performance_first, StrategyParameters(stream=False, lambda_cost=0.1))
        self.assertEqual(selected.endpoint_id, "fast")

        cost_first = [make_candidate("cheap", 1, 3), make_candidate("fast", 3, 1)]
        selected = finalize_selection(cost_first, StrategyParameters(stream=False, lambda_cost=0.9))
        self.assertEqual(selected.endpoint_id, "cheap")


class WindowTests(unittest.TestCase):
    def test_stream_tpot_and_adaptive_window(self):
        cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        observations = [
            make_observation("A", cutoff - timedelta(minutes=index), stream=True, e2e_ms=2000)
            for index in range(40)
        ]
        params = StrategyParameters(
            stream=True,
            target_samples=30,
            min_samples=20,
            window_candidates_minutes=(5, 15, 30, 60),
            half_life_minutes=30,
            prior_strength=0,
            min_success_rate=0,
            slo_ttft_ms=100,
            slo_tpot_ms=10,
        )
        candidate = compute_candidate(Price("A", "M", "CNY", 1, 1, 0, {}), observations, cutoff, None, params)
        self.assertEqual(candidate.window_minutes, 30)
        self.assertAlmostEqual(candidate.performance, 1.0)
        self.assertTrue(candidate.feasible)

    def test_index_isolated_by_model_endpoint_and_stream_type(self):
        index = ModelEndpointStateIndex(max_age_minutes=60)
        now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        observation = make_observation("endpoint-a", now, stream=True)
        index.record("model-x", observation)
        index.record("model-y", observation)
        self.assertEqual(len(index.recent("model-x", "endpoint-a", True, now, 60)), 1)
        self.assertEqual(index.recent("model-x", "endpoint-a", False, now, 60), [])
        self.assertEqual(index.recent("model-z", "endpoint-a", True, now, 60), [])


class InMemoryStateTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        self.state = InMemoryRoutingState(max_window_minutes=60)
        self.state.register_endpoint(make_offering("endpoint-a"))
        self.state.register_endpoint(make_offering("endpoint-b"))
        self.state.register_endpoint(make_offering("endpoint-a", model="model-y"))

    def test_model_policy_is_maintained_by_model_id(self):
        policy = ModelRoutingPolicy(
            model_id="model-x",
            slo_e2e_ms=2000,
            lambda_cost=0.7,
            eta_ttft=0.6,
            rho_input_price=0.4,
        )
        self.state.policies.upsert(policy)
        self.assertEqual(self.state.policies.get("model-x"), policy)
        self.assertEqual(policy.to_parameters(False).lambda_cost, 0.7)

    def test_model_endpoint_state_has_tier_capacity_prior_and_switches(self):
        prior = Prior("endpoint-a", 900, 100, 10, 0.98, 30, "test")
        self.state.catalog.set_price_tier("model-x", "endpoint-a", 1)
        self.state.catalog.set_prior("model-x", "endpoint-a", prior)
        self.state.catalog.set_cooldown("model-x", "endpoint-a", self.now + timedelta(minutes=5))
        self.state.catalog.update_quality("model-x", "endpoint-a", 0.9, 0.05, 0.02, 100)
        self.state.catalog.update_model_endpoint_load(
            "model-x",
            "endpoint-a",
            current_concurrency=2,
            current_rpm=40,
            current_tpm=2000,
        )
        runtime = self.state.catalog.runtime_state("model-x", "endpoint-a")
        self.assertEqual(runtime.price_tier_index, 1)
        self.assertEqual(runtime.capacity_rpm, 100)
        self.assertEqual(runtime.capacity_tpm, 10_000)
        self.assertEqual(runtime.current_concurrency, 2)
        self.assertEqual(runtime.current_rpm, 40)
        self.assertEqual(runtime.current_tpm, 2000)
        self.assertEqual(runtime.historical_prior, prior)
        self.assertEqual(runtime.rate_limit_429_rate, 0.05)
        self.assertNotIn(
            "endpoint-a",
            {item.endpoint_id for item in self.state.catalog.candidates("model-x", self.now)},
        )

    def test_endpoint_state_has_load_health_and_supported_models(self):
        self.state.catalog.update_endpoint_runtime(
            "endpoint-a",
            current_concurrency=3,
            current_rpm=40,
            current_tpm=2000,
            health_status="healthy",
        )
        endpoint = self.state.catalog.endpoint_state("endpoint-a")
        self.assertEqual(endpoint.current_concurrency, 3)
        self.assertEqual(endpoint.supported_models, ("model-x", "model-y"))
        self.assertIn(
            "endpoint-a",
            {item.endpoint_id for item in self.state.catalog.candidates("model-x", self.now)},
        )
        self.state.catalog.update_endpoint_runtime("endpoint-a", health_status="unhealthy")
        self.assertNotIn(
            "endpoint-a",
            {item.endpoint_id for item in self.state.catalog.candidates("model-x", self.now)},
        )

    def test_capacity_and_dynamic_remove_affect_candidate_index(self):
        self.state.catalog.update_endpoint_runtime("endpoint-a", current_rpm=100)
        candidates = {item.endpoint_id for item in self.state.catalog.candidates("model-x", self.now)}
        self.assertEqual(candidates, {"endpoint-a", "endpoint-b"})

        self.state.catalog.update_model_endpoint_load("model-x", "endpoint-a", current_rpm=100)
        candidates = {item.endpoint_id for item in self.state.catalog.candidates("model-x", self.now)}
        self.assertEqual(candidates, {"endpoint-b"})
        self.assertEqual(
            {item.endpoint_id for item in self.state.catalog.candidates("model-y", self.now)},
            {"endpoint-a"},
        )
        self.assertTrue(self.state.remove_endpoint("model-x", "endpoint-b"))
        self.assertEqual(self.state.catalog.candidates("model-x", self.now), ())

    def test_concurrency_counter_is_maintained_per_model_endpoint(self):
        self.assertEqual(
            self.state.catalog.adjust_model_endpoint_concurrency("model-x", "endpoint-a", 1),
            1,
        )
        self.assertEqual(
            self.state.catalog.adjust_model_endpoint_concurrency("model-x", "endpoint-a", -1),
            0,
        )
        self.assertEqual(
            self.state.catalog.runtime_state("model-y", "endpoint-a").current_concurrency,
            0,
        )

    def test_observations_update_success_429_and_5xx_rates(self):
        observations = [
            make_observation("endpoint-a", self.now - timedelta(minutes=2), status=200),
            make_observation("endpoint-a", self.now - timedelta(minutes=1), status=429),
            make_observation("endpoint-a", self.now, status=503),
        ]
        self.state.bulk_record("model-x", observations)
        runtime = self.state.catalog.runtime_state("model-x", "endpoint-a")
        self.assertAlmostEqual(runtime.success_rate, 1 / 3)
        self.assertAlmostEqual(runtime.rate_limit_429_rate, 1 / 3)
        self.assertAlmostEqual(runtime.server_error_5xx_rate, 1 / 3)

    def test_routing_engine_uses_memory_indexes_and_cooldown(self):
        self.state.policies.upsert(ModelRoutingPolicy("model-x", slo_e2e_ms=1000))
        observations = [
            make_observation("endpoint-a", self.now, e2e_ms=900),
            make_observation("endpoint-b", self.now, e2e_ms=1100),
        ]
        self.state.bulk_record("model-x", observations)
        self.state.catalog.set_cooldown("model-x", "endpoint-a", self.now + timedelta(minutes=5))
        candidates, selected, _ = RoutingEngine(self.state).route(
            "model-x",
            False,
            self.now,
            target_samples=1,
            min_samples=1,
            min_success_rate=0,
            prior_strength=0,
            window_candidates_minutes=(5,),
        )
        self.assertEqual([item.endpoint_id for item in candidates], ["endpoint-b"])
        self.assertEqual(selected.endpoint_id, "endpoint-b")


class RepositoryContractTests(unittest.TestCase):
    def test_registry_can_bootstrap_from_future_repository_adapter(self):
        class FakeRepository:
            def load_all(self):
                return [make_offering("endpoint-a")]

            def upsert(self, offering):
                raise NotImplementedError

            def remove(self, model_id, endpoint_id):
                raise NotImplementedError

        registry = ModelEndpointRegistry.from_repository(FakeRepository())
        self.assertEqual(registry.models_for_endpoint("endpoint-a"), ("model-x",))


if __name__ == "__main__":
    unittest.main()
