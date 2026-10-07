"""Integration checks for token-tier resolution in the formal request router."""
from copy import deepcopy
from datetime import datetime, timezone
import unittest

from endpoint_routing_strategy import (
    EndpointOffering, InMemoryRoutingState, ModelRoutingPolicy, Prior,
    RequestSLO, RoutingEngine, RoutingRequest,
)


class RequestPricingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
        self.state = InMemoryRoutingState(max_window_minutes=360)
        self.state.policies.upsert(ModelRoutingPolicy("pricing-model", slo_e2e_ms=1000))
        self.engine = RoutingEngine(self.state)

    @staticmethod
    def tier(pin=1, pout=4, **overrides):
        result = dict(currency="CNY", billing_unit="per_million_tokens",
                      input_per_million=pin, output_per_million=pout, conditions={})
        result.update(overrides)
        return result

    def add(self, endpoint, tiers, configured_index=0):
        self.state.register_endpoint(EndpointOffering(
            endpoint_id=endpoint, model_id="pricing-model", deployment_type="pricing_regression",
            price_config=dict(pricing_status="configured", currency="CNY", price_tiers=tiers),
            capacity_config=dict(rpm=1000, tpm=2_000_000, concurrency=10),
        ))
        self.state.catalog.set_prior("pricing-model", endpoint, Prior(
            endpoint_id=endpoint, e2e_p95_ms=100, ttft_p95_ms=None,
            tpot_proxy_ms=None, success_rate=1, sample_count=100, source="pricing_regression",
        ))
        if configured_index:
            self.state.catalog.set_price_tier("pricing-model", endpoint, configured_index)

    def request(self, tin=100, tout=10):
        return RoutingRequest(
            request_id="request-pricing", model_id="pricing-model", arrived_at=self.now,
            is_stream=False, predicted_input_tokens=tin, predicted_output_tokens=tout,
            slo=RequestSLO(e2e_ms=1000),
        )

    def two_tiers(self):
        return [self.tier(1, 4, input_tokens_min_exclusive=0, input_tokens_max_inclusive=100),
                self.tier(2, 8, input_tokens_min_exclusive=100, input_tokens_max_inclusive=None)]

    def test_request_length_selects_auto_tier_and_changes_primary(self):
        self.add("variable", self.two_tiers())
        self.add("fixed", [self.tier(1.5, 6)])
        short = self.engine.route_request(self.request(tin=100), lambda_cost=1)
        long = self.engine.route_request(self.request(tin=101), lambda_cost=1)
        self.assertEqual(short.selected_endpoint, "variable")
        self.assertEqual(long.selected_endpoint, "fixed")
        for decision, index in ((short, 0), (long, 1)):
            check = decision.endpoint_checks["variable"]
            self.assertEqual(check["price_tier_index"], index)
            self.assertEqual(check["configured_price_tier_index"], 0)
            self.assertEqual(check["selected_price"]["tier_index"], index)
            self.assertEqual(check["price_resolution"]["selection_source"], "predicted_input_token_rules")
            self.assertIsNone(check["price_resolution"]["reason"])

    def test_boundary_prices_produce_correct_input_and_output_costs(self):
        self.add("variable", self.two_tiers())
        for tin, index, pin, pout in ((100, 0, 1, 4), (101, 1, 2, 8)):
            with self.subTest(tin=tin):
                decision = self.engine.route_request(self.request(tin=tin, tout=20))
                candidate = decision.candidates[0]
                self.assertEqual(decision.endpoint_checks["variable"]["price_tier_index"], index)
                self.assertAlmostEqual(candidate.estimated_input_cost, tin * pin / 1_000_000)
                self.assertAlmostEqual(candidate.estimated_output_cost, 20 * pout / 1_000_000)
                self.assertAlmostEqual(candidate.estimated_request_cost, (tin * pin + 20 * pout) / 1_000_000)
                self.assertAlmostEqual(candidate.cost_raw, candidate.estimated_request_cost)

    def test_configured_tier_mismatch_excludes_endpoint_not_silent_underpricing(self):
        self.add("variable", self.two_tiers())
        self.add("fixed", [self.tier(10, 20)])
        decision = self.engine.route_request(self.request(tin=101), price_tier_mode="configured")
        self.assertEqual(decision.selected_endpoint, "fixed")
        resolution = decision.endpoint_checks["variable"]["price_resolution"]
        self.assertEqual(resolution["reason"], "configured_token_tier_mismatch")
        self.assertEqual(resolution["tier_index"], 0)
        self.assertIsNone(decision.endpoint_checks["variable"]["selected_price"])

    def test_configured_matching_tier_can_be_used_explicitly(self):
        self.add("variable", self.two_tiers(), configured_index=1)
        decision = self.engine.route_request(self.request(tin=101), price_tier_mode="configured")
        self.assertEqual(decision.selected_endpoint, "variable")
        check = decision.endpoint_checks["variable"]
        self.assertEqual(check["configured_price_tier_index"], 1)
        self.assertEqual(check["price_tier_index"], 1)
        self.assertEqual(check["price_resolution"]["selection_source"], "configured_explicit")

    def test_unknown_token_conditions_fail_closed(self):
        self.add("unknown", [self.tier(0.01, 0.01, conditions={"output_token_band": "lte_100k"})])
        self.add("fixed", [self.tier()])
        decision = self.engine.route_request(self.request())
        self.assertEqual(decision.selected_endpoint, "fixed")
        resolution = decision.endpoint_checks["unknown"]["price_resolution"]
        self.assertEqual(resolution["reason"], "unsupported_token_conditions")
        self.assertEqual(resolution["token_rule_evaluations"][0]["unsupported_conditions"], ["conditions.output_token_band"])

    def test_overlapping_tiers_and_no_matching_tier_return_no_candidate(self):
        self.add("overlap", [self.tier(input_tokens_min_exclusive=0, input_tokens_max_inclusive=100),
                             self.tier(input_tokens_min_exclusive=50, input_tokens_max_inclusive=200)])
        for tin, reason in ((75, "ambiguous_price_tiers"), (201, "no_matching_price_tier")):
            with self.subTest(tin=tin):
                decision = self.engine.route_request(self.request(tin=tin))
                self.assertEqual(decision.status, "no_candidate")
                self.assertIsNone(decision.selected_endpoint)
                self.assertEqual(decision.endpoint_checks["overlap"]["price_resolution"]["reason"], reason)

    def test_currency_mismatch_is_not_implicitly_converted(self):
        self.add("usd", [self.tier(2, 8, currency="USD")])
        blocked = self.engine.route_request(self.request(), currency="CNY")
        self.assertEqual(blocked.status, "no_candidate")
        self.assertEqual(blocked.endpoint_checks["usd"]["price_resolution"]["reason"], "currency_mismatch")
        accepted = self.engine.route_request(self.request(), currency="USD")
        self.assertEqual(accepted.selected_endpoint, "usd")
        self.assertEqual(accepted.candidates[0].currency, "USD")
        self.assertAlmostEqual(accepted.candidates[0].cost_raw, 0.00028)

    def test_512k_band_is_decimal_and_resolution_is_recorded(self):
        self.add("band", [self.tier(0.63, 2.52, conditions={"input_token_band": "lte_512k"}),
                          self.tier(1.26, 5.04, conditions={"input_token_band": "gt_512k"})])
        for tin, index in ((512000, 0), (512001, 1), (524288, 1)):
            with self.subTest(tin=tin):
                decision = self.engine.route_request(self.request(tin=tin))
                resolution = decision.endpoint_checks["band"]["price_resolution"]
                self.assertEqual(decision.selected_endpoint, "band")
                self.assertEqual(resolution["tier_index"], index)
                self.assertEqual(resolution["band_k_multiplier"], 1000)
                self.assertEqual(resolution["token_rule_evaluations"][index]["rules"]["band_threshold_tokens"], 512000)

    def test_plain_tier_metadata_keeps_configured_pricing(self):
        self.add("metadata", [self.tier(1, 4, conditions={"tier": 0}),
                              self.tier(2, 8, conditions={"tier": 1})], configured_index=1)
        decision = self.engine.route_request(self.request())
        self.assertEqual(decision.selected_endpoint, "metadata")
        resolution = decision.endpoint_checks["metadata"]["price_resolution"]
        self.assertEqual(resolution["selection_source"], "configured_no_token_rules")
        self.assertEqual(resolution["tier_index"], 1)
        self.assertAlmostEqual(decision.candidates[0].cost_raw, 0.00028)

    def test_auto_price_resolution_does_not_modify_configuration_or_runtime_index(self):
        self.add("variable", self.two_tiers())
        prior_offering = self.state.catalog.candidates("pricing-model", include_ineligible=True)[0]
        original = deepcopy(prior_offering.price_config)
        version = self.state.catalog.version
        decision = self.engine.route_request(self.request(tin=101))
        current_offering = self.state.catalog.candidates("pricing-model", include_ineligible=True)[0]
        self.assertEqual(current_offering.price_config, original)
        self.assertEqual(self.state.catalog.runtime_state("pricing-model", "variable").price_tier_index, 0)
        self.assertEqual(self.state.catalog.version, version)
        self.assertEqual(decision.endpoint_checks["variable"]["price_tier_index"], 1)

    def test_legacy_route_retains_configured_unit_price_contract(self):
        self.add("variable", self.two_tiers(), configured_index=1)
        candidates, selected, _ = self.engine.route("pricing-model", False, self.now, rho_input_price=0.5)
        self.assertEqual(selected.endpoint_id, "variable")
        self.assertAlmostEqual(candidates[0].cost_raw, 5)
        self.assertEqual(candidates[0].cost_mode, "unit_price")
        self.assertIsNone(candidates[0].estimated_request_cost)

    def test_invalid_price_tier_mode_rejected(self):
        self.add("variable", self.two_tiers())
        with self.assertRaises(ValueError):
            self.engine.route_request(self.request(), price_tier_mode="guess")


if __name__ == "__main__":
    unittest.main()
