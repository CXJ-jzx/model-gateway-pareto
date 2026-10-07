"""Pressure detection only: no routing, queue, prediction or provider simulation."""
import json
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from endpoint_routing_strategy import BusyThresholds, EndpointOffering, InMemoryRoutingState
from endpoint_routing_strategy.experiments.busy_check import run_checks


class BusyTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
        self.state = InMemoryRoutingState()

    def add(self, ep="a", model="m", capacity=None):
        self.state.register_endpoint(EndpointOffering(
            ep, model, "fixture", {},
            {"rpm": 1000, "tpm": 10000, "concurrency": 10} if capacity is None else capacity,
        ))

    def assess(self, model="m", **kwargs):
        return self.state.assess_busy(model, at_time=self.now, **kwargs)

    def test_each_dimension_enters_at_threshold(self):
        for field, value in (("current_rpm", 800), ("current_tpm", 8000), ("current_concurrency", 8)):
            with self.subTest(field=field):
                self.state = InMemoryRoutingState()
                self.add()
                self.state.catalog.update_model_endpoint_load("m", "a", **{field: value})
                result = self.assess()
                self.assertTrue(result.is_busy)
                self.assertAlmostEqual(result.endpoints[0].pressure, 0.8)
                self.assertFalse(result.endpoints[0].saturated_dimensions)

    def test_hysteresis_and_exit_boundary(self):
        self.add()
        for rpm, status in ((700, "idle"), (800, "busy"), (700, "busy"), (600, "idle"), (700, "idle")):
            with self.subTest(rpm=rpm, status=status):
                self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=rpm)
                self.assertEqual(self.assess().status, status)

    def test_full_and_overfull_are_not_lost_or_clipped(self):
        self.add()
        self.state.catalog.update_model_endpoint_load("m", "a", current_concurrency=12)
        self.assertEqual(self.state.catalog.candidates("m", at_time=self.now), ())
        result = self.assess()
        self.assertEqual(result.status, "busy")
        self.assertEqual(result.endpoints[0].pressure, 1.2)
        self.assertEqual(result.endpoints[0].saturated_dimensions, ("concurrency",))

    def test_idle_alternative_prevents_model_busy(self):
        self.add("a")
        self.add("b")
        self.state.catalog.update_model_endpoint_load("m", "a", current_tpm=10000)
        result = self.assess()
        self.assertFalse(result.is_busy)
        self.assertEqual(result.endpoint_ids("busy"), ("a",))
        self.assertEqual(result.endpoint_ids("idle"), ("b",))
        self.assertEqual(result.to_dict()["busy_fraction"], 0.5)

    def test_subset_allows_caller_to_exclude_slo_incompatible_alternative(self):
        self.add("a")
        self.add("b")
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=900)
        self.assertEqual(self.assess().status, "idle")
        result = self.assess(endpoint_ids=["a"])
        self.assertTrue(result.is_busy)
        self.assertEqual(result.scope, "endpoint_subset")
        self.assertEqual(len(result.endpoints), 1)
        self.assertEqual(self.assess().endpoint_ids("busy"), ("a",))

    def test_all_busy(self):
        self.add("a")
        self.add("b")
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=900)
        self.state.catalog.update_model_endpoint_load("m", "b", current_tpm=9000)
        self.assertTrue(self.assess().is_busy)

    def test_pair_load_not_endpoint_aggregate_or_other_model(self):
        self.add("a", "m")
        self.add("a", "other")
        self.state.catalog.update_model_endpoint_load("m", "a", current_concurrency=9)
        self.state.catalog.update_endpoint_runtime("a", current_concurrency=100, current_rpm=100000)
        self.assertTrue(self.assess("m").is_busy)
        self.assertFalse(self.assess("other").is_busy)

    def test_unknown_capacity_not_assumed_idle(self):
        self.add(capacity={})
        result = self.assess()
        self.assertEqual(result.status, "unknown")
        self.assertIsNone(result.is_busy)
        self.assertIsNone(result.endpoints[0].pressure)
        self.assertEqual(result.endpoints[0].unknown_dimensions, ("rpm", "tpm", "concurrency"))

    def test_known_high_pressure_is_sufficient_even_with_unknown_limits(self):
        self.add(capacity={"rpm": 1000})
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=800)
        self.assertTrue(self.assess().is_busy)
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=500)
        self.assertEqual(self.assess().status, "unknown")
        self.state.register_endpoint(replace(self.state.catalog.candidates("m", include_ineligible=True)[0],
                                            capacity_config={"rpm": 1000, "tpm": 10000, "concurrency": 10}))
        self.assertFalse(self.assess().is_busy)

    def test_busy_plus_unknown_is_unknown_not_all_busy(self):
        self.add("a")
        self.add("b", capacity={})
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=900)
        self.assertEqual(self.assess().status, "unknown")

    def test_idle_plus_unknown_has_known_idle_alternative(self):
        self.add("a")
        self.add("b", capacity={})
        self.assertFalse(self.assess().is_busy)

    def test_zero_capacity_unavailable_without_division_by_zero(self):
        for dimension in ("rpm", "tpm", "concurrency"):
            with self.subTest(dimension=dimension):
                self.state = InMemoryRoutingState()
                self.add(capacity={dimension: 0})
                result = self.assess()
                self.assertEqual(result.status, "unavailable")
                self.assertIsNone(result.is_busy)
                self.assertIn(f"zero_capacity_{dimension}", result.endpoints[0].reasons)
                json.dumps(result.to_dict(), allow_nan=False)

    def test_disabled_cooldown_and_unhealthy_are_not_busy(self):
        self.add("a")
        self.add("b")
        self.add("c")
        for ep in ("a", "b", "c"):
            self.state.catalog.update_model_endpoint_load("m", ep, current_rpm=1000)
        self.state.catalog.set_enabled("m", "a", False)
        self.state.catalog.set_cooldown("m", "b", self.now + timedelta(seconds=1))
        self.state.catalog.update_endpoint_runtime("c", health_status="down")
        result = self.assess()
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(len(result.endpoint_ids("unavailable")), 3)
        self.assertIsNone(result.to_dict()["busy_fraction"])

    def test_cooldown_end_and_health_recovery(self):
        self.add()
        self.state.catalog.set_cooldown("m", "a", self.now)
        self.assertEqual(self.assess().status, "idle")
        self.state.catalog.update_endpoint_runtime("a", health_status="unhealthy")
        self.assertEqual(self.assess().status, "unavailable")
        self.state.catalog.update_endpoint_runtime("a", health_status="healthy")
        self.assertEqual(self.assess().status, "idle")

    def test_empty_model_and_empty_subset_are_not_busy(self):
        self.assertEqual(self.assess().status, "unavailable")
        self.add()
        self.assertEqual(self.assess(endpoint_ids=[]).status, "unavailable")

    def test_remove_and_readd_clears_hysteresis(self):
        self.add()
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=900)
        self.assertTrue(self.assess().is_busy)
        self.assertTrue(self.state.remove_endpoint("m", "a"))
        self.assertEqual(self.assess().status, "unavailable")
        self.add()
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=700)
        self.assertFalse(self.assess().is_busy)

    def test_assessment_does_not_mutate_load_or_compute_metrics(self):
        self.add()
        before = self.state.catalog.capacity_snapshot("m")
        with patch.object(self.state.windows, "health_summary", side_effect=AssertionError("No metric scans")):
            result = self.assess()
        self.assertEqual(before, self.state.catalog.capacity_snapshot("m"))
        self.assertFalse(result.to_dict()["capacity_reserved"])
        self.assertTrue(result.to_dict()["capacity_only"])

    def test_snapshot_keeps_immutable_previous_values(self):
        self.add()
        version, runtimes, health = self.state.catalog.capacity_snapshot("m")
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=900)
        self.assertEqual(runtimes[0].current_rpm, 0)
        self.assertEqual(health["a"], "unknown")
        self.assertGreater(self.assess().catalog_version, version)

    def test_chronology_and_input_validation(self):
        self.add()
        self.assess()
        with self.assertRaises(ValueError):
            self.state.assess_busy("m", at_time=self.now - timedelta(seconds=1))
        for arguments in ({"at_time": datetime(2026, 1, 1)}, {"endpoint_ids": "a"},
                          {"endpoint_ids": ["not-registered"]}, {"endpoint_ids": [1]}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.state.assess_busy("m", **arguments)
        with self.assertRaises(ValueError):
            self.state.assess_busy("")

    def test_threshold_validation_and_custom_configuration(self):
        for enter, exit in ((0.8, 0.8), (0, 0), (1.1, 0.6), (0.8, -0.1),
                            (float("nan"), 0.6), (0.8, float("inf")), (True, 0.6), ("0.8", 0.6)):
            with self.subTest(enter=enter, exit=exit), self.assertRaises(ValueError):
                BusyThresholds(enter, exit)
        self.state = InMemoryRoutingState(busy_thresholds=BusyThresholds(0.9, 0.7))
        self.add()
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=800)
        self.assertFalse(self.assess().is_busy)
        self.state.catalog.update_model_endpoint_load("m", "a", current_rpm=900)
        self.assertTrue(self.assess().is_busy)
        with self.assertRaises(ValueError):
            InMemoryRoutingState(busy_thresholds={"enter": 0.8})

    def test_experiment_rules_with_default_and_boundary_thresholds(self):
        for thresholds in (BusyThresholds(), BusyThresholds(1, 0), BusyThresholds(0.9, 0.7)):
            with self.subTest(thresholds=thresholds):
                report = run_checks(thresholds)
                self.assertEqual(report["cases"], 10)
                self.assertEqual(report["passed"], 10)
                json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
