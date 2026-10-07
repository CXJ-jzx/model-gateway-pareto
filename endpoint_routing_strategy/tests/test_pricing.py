"""Pricing tier regression checks, including examples from the source dataset."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from endpoint_routing_strategy.models import EndpointOffering
from endpoint_routing_strategy.pricing import resolve_request_price


def offering(tiers, *, currency="CNY", status="configured"):
    return EndpointOffering(
        endpoint_id="pricing-endpoint", model_id="pricing-model", deployment_type="test",
        price_config={"pricing_status": status, "currency": currency, "price_tiers": tiers},
        capacity_config={},
    )


def tier(pin=1, pout=4, **overrides):
    result = dict(currency="CNY", billing_unit="per_million_tokens",
                  input_per_million=pin, output_per_million=pout, conditions={})
    result.update(overrides)
    return result


class RequestPricingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[2] / "历史性能数据包" / "端点配置.jsonl"
        cls.source_offerings = {}
        if source.exists():
            with source.open(encoding="utf-8-sig") as handle:
                for line in handle:
                    endpoint = json.loads(line)
                    for model in endpoint["models"]:
                        cls.source_offerings[(endpoint["endpoint_id"], model["model_id"])] = EndpointOffering(
                            endpoint_id=endpoint["endpoint_id"], model_id=model["model_id"],
                            deployment_type=endpoint["deployment_type"],
                            price_config=model["price_config"], capacity_config=model["capacity_config"],
                        )

    def resolve(self, record, length=100, index=0, currency="CNY", mode="auto"):
        return resolve_request_price(record, index, length, currency, mode)

    def test_fixed_price_and_metadata_keep_configured_tier(self):
        record = offering([tier(1, 2, conditions={"tier": 0}), tier(3, 4, conditions={"tier": 1})])
        price, trace = self.resolve(record, index=1)
        self.assertEqual(price.input_per_million, 3)
        self.assertEqual(trace["selection_source"], "configured_no_token_rules")
        self.assertIsNone(trace["reason"])
        self.assertFalse(trace["has_token_rules"])

    def test_single_fixed_price_accepts_zero_tokens(self):
        price, trace = self.resolve(offering([tier()]), length=0)
        self.assertEqual(price.tier_index, 0)
        self.assertIsNone(trace["reason"])

    def test_null_bounds_are_not_effective_token_rules(self):
        record = offering([tier(input_tokens_min_exclusive=None, input_tokens_max_inclusive=None),
                           tier(3, 8, input_tokens_min_exclusive=None, input_tokens_max_inclusive=None)])
        price, trace = self.resolve(record, index=1)
        self.assertEqual(price.tier_index, 1)
        self.assertFalse(trace["has_token_rules"])
        self.assertEqual(trace["selection_source"], "configured_no_token_rules")

    def test_ranges_use_exclusive_lower_inclusive_upper(self):
        record = offering([
            tier(0.75, 3, input_tokens_min_exclusive=0, input_tokens_max_inclusive=32000),
            tier(1.125, 4.5, input_tokens_min_exclusive=32000, input_tokens_max_inclusive=128000),
            tier(1.875, 7.5, input_tokens_min_exclusive=128000, input_tokens_max_inclusive=200000),
        ])
        for length, index in ((1, 0), (32000, 0), (32001, 1), (128000, 1), (128001, 2), (200000, 2)):
            with self.subTest(length=length):
                price, trace = self.resolve(record, length=length, index=2)
                self.assertEqual(price.tier_index, index)
                self.assertEqual(trace["matched_tier_indexes"], [index])
                self.assertEqual(trace["selection_source"], "predicted_input_token_rules")
        for length in (0, 200001):
            self.assertEqual(self.resolve(record, length=length)[1]["reason"], "no_matching_price_tier")

    def test_actual_endpoint_c_model_av_boundaries(self):
        record = self.source_offerings.get(("端点C", "模型AV"))
        if record is None:
            self.skipTest("Original endpoint dataset is not packaged")
        for length, index, pin, pout in (
            (32000, 0, 0.75, 3), (32001, 1, 1.125, 4.5),
            (128000, 1, 1.125, 4.5), (128001, 2, 1.875, 7.5),
        ):
            with self.subTest(length=length):
                price, trace = self.resolve(record, length=length)
                self.assertEqual((price.tier_index, price.input_per_million, price.output_per_million), (index, pin, pout))
                self.assertEqual(trace["request_input_tokens"], length)

    def test_actual_endpoint_d_model_ak_decimal_512k(self):
        record = self.source_offerings.get(("端点D", "模型AK"))
        if record is None:
            self.skipTest("Original endpoint dataset is not packaged")
        for length, index, pin, pout in ((512000, 0, 0.63, 2.52), (512001, 1, 1.26, 5.04), (524288, 1, 1.26, 5.04)):
            with self.subTest(length=length):
                price, trace = self.resolve(record, length=length)
                self.assertEqual((price.tier_index, price.input_per_million, price.output_per_million), (index, pin, pout))
                self.assertEqual(trace["token_rule_evaluations"][index]["rules"]["band_threshold_tokens"], 512000)

    def test_auto_refuses_overlapping_or_missing_matches(self):
        record = offering([tier(input_tokens_min_exclusive=0, input_tokens_max_inclusive=100),
                           tier(input_tokens_min_exclusive=50, input_tokens_max_inclusive=200)])
        price, trace = self.resolve(record, length=75)
        self.assertIsNone(price)
        self.assertEqual(trace["reason"], "ambiguous_price_tiers")
        self.assertEqual(trace["matched_tier_indexes"], [0, 1])
        self.assertEqual(self.resolve(record, length=201)[1]["reason"], "no_matching_price_tier")

    def test_unbounded_catchall_and_matching_band_is_ambiguous(self):
        record = offering([tier(), tier(conditions={"input_token_band": "lte_512k"})])
        self.assertEqual(self.resolve(record)[1]["reason"], "ambiguous_price_tiers")

    def test_unknown_token_condition_refuses_auto_even_if_another_tier_matches(self):
        for conditions in ({"input_token_band": "medium"}, {"output_token_band": "lte_1k"}, {"context_length": 1000}):
            with self.subTest(conditions=conditions):
                record = offering([tier(), tier(conditions=conditions)])
                self.assertEqual(self.resolve(record)[1]["reason"], "unsupported_token_conditions")

    def test_unknown_top_level_token_condition_is_not_ignored(self):
        record = offering([tier(max_output_tokens=500)])
        self.assertEqual(self.resolve(record)[1]["reason"], "unsupported_token_conditions")

    def test_known_numeric_and_band_conditions_must_both_match(self):
        record = offering([tier(input_tokens_min_exclusive=100, input_tokens_max_inclusive=600000,
                                conditions={"input_token_band": "lte_512k"})])
        self.assertEqual(self.resolve(record, length=100)[1]["reason"], "no_matching_price_tier")
        self.assertIsNotNone(self.resolve(record, length=101)[0])
        self.assertEqual(self.resolve(record, length=512001)[1]["reason"], "no_matching_price_tier")

    def test_configured_mode_checks_selected_token_rule(self):
        record = offering([tier(conditions={"input_token_band": "lte_512k"}),
                           tier(conditions={"input_token_band": "gt_512k"})])
        price, trace = self.resolve(record, length=512001, index=0, mode="configured")
        self.assertIsNone(price)
        self.assertEqual(trace["reason"], "configured_token_tier_mismatch")
        self.assertEqual(trace["tier_index"], 0)
        price, trace = self.resolve(record, length=512001, index=1, mode="configured")
        self.assertEqual(price.tier_index, 1)
        self.assertEqual(trace["selection_source"], "configured_explicit")

    def test_explicit_mode_disambiguates_overlap_with_warning(self):
        record = offering([tier(input_tokens_max_inclusive=100), tier(input_tokens_max_inclusive=200)])
        price, trace = self.resolve(record, length=50, index=1, mode="configured")
        self.assertEqual(price.tier_index, 1)
        self.assertIn("explicit_tier_resolves_overlapping_token_ranges", trace["warnings"])

    def test_explicit_mode_does_not_need_other_tiers_unknown_rules(self):
        record = offering([tier(), tier(conditions={"output_token_band": "unknown"})])
        self.assertIsNotNone(self.resolve(record, mode="configured")[0])

    def test_no_currency_conversion_and_tier_currency_wins(self):
        record = offering([tier(currency="USD")])
        self.assertEqual(self.resolve(record)[1]["reason"], "currency_mismatch")
        price, trace = self.resolve(record, currency="USD")
        self.assertEqual(price.currency, "USD")
        self.assertEqual(trace["selected_currency"], "USD")

    def test_missing_currency_falls_back_to_config_currency(self):
        price, trace = self.resolve(offering([tier(currency=None)]))
        self.assertEqual(price.currency, "CNY")
        self.assertIsNone(trace["reason"])

    def test_missing_price_or_unsupported_billing_unit(self):
        for record, reason in (
            (offering([], status="missing"), "pricing_not_configured"),
            (offering([]), "missing_price_tiers"),
            (offering([tier(input_per_million=None)]), "missing_unit_prices"),
            (offering([tier(billing_unit="per_image")]), "unsupported_billing_unit"),
        ):
            with self.subTest(reason=reason):
                price, trace = self.resolve(record)
                self.assertIsNone(price)
                self.assertEqual(trace["reason"], reason)

    def test_invalid_unit_prices_fail_closed(self):
        for value in (True, -1, float("nan"), float("inf"), "unknown"):
            with self.subTest(value=value):
                self.assertEqual(self.resolve(offering([tier(input_per_million=value)]))[1]["reason"], "invalid_unit_prices")

    def test_zero_unit_prices_are_valid(self):
        price, _ = self.resolve(offering([tier(0, 0)]))
        self.assertEqual(price.input_per_million, 0)

    def test_invalid_ranges_and_condition_shape_fail_closed(self):
        for values in (
            {"input_tokens_min_exclusive": -1}, {"input_tokens_max_inclusive": True},
            {"input_tokens_max_inclusive": 1.5},
            {"input_tokens_min_exclusive": 10, "input_tokens_max_inclusive": 10},
            {"conditions": ["unknown"]}, {"conditions": []},
        ):
            with self.subTest(values=values):
                price, trace = self.resolve(offering([tier(**values)]))
                self.assertIsNone(price)
                self.assertEqual(trace["reason"], "invalid_token_rules")

    def test_invalid_configured_index_is_not_used_by_auto_numeric_rules(self):
        record = offering([tier(input_tokens_min_exclusive=0)])
        self.assertIsNotNone(self.resolve(record, index=100)[0])
        self.assertEqual(self.resolve(record, index=100, mode="configured")[1]["reason"], "invalid_price_tier_index")
        self.assertEqual(self.resolve(offering([tier()]), index=100)[1]["reason"], "invalid_price_tier_index")

    def test_resolver_does_not_mutate_source_configuration(self):
        record = offering([tier(input_tokens_min_exclusive=0, conditions={"tier": 1, "metadata": {"version": 1}})])
        previous = deepcopy(record.price_config)
        price, trace = self.resolve(record)
        price.tier_conditions["tier"] = 99
        price.tier_conditions["metadata"]["version"] = 99
        trace["tier_conditions"]["metadata"]["version"] = 55
        trace["token_rule_evaluations"][0]["rules"]["input_tokens_min_exclusive"] = 999
        self.assertEqual(record.price_config, previous)

    def test_invalid_call_arguments_raise(self):
        record = offering([tier()])
        for arguments in (
            (record, True, 100, "CNY", "auto"), (record, 0, False, "CNY", "auto"),
            (record, 0, -1, "CNY", "auto"), (record, 0, 100.5, "CNY", "auto"),
            (record, 0, 100, "", "auto"), (record, 0, 100, "CNY", "unknown"),
        ):
            with self.subTest(arguments=arguments[1:]):
                with self.assertRaises(ValueError):
                    resolve_request_price(*arguments)


if __name__ == "__main__":
    unittest.main()
