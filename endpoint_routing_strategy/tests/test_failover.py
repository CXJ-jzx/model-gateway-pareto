import json
import unittest
from datetime import datetime, timedelta, timezone

from endpoint_routing_strategy import (
    AttemptResult, EndpointOffering, InMemoryRoutingState, ModelRoutingPolicy,
    Observation, RequestSLO, RetryPolicy, RoutingEngine, RoutingRequest,
)


class FailoverTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 5, tzinfo=timezone.utc)
        self.state = InMemoryRoutingState()
        self.state.policies.upsert(ModelRoutingPolicy("model", lambda_cost=1))
        for ep, price, failures in (("primary", 1, 5), ("stable", 5, 1), ("third", 3, 10)):
            self.state.register_endpoint(EndpointOffering(
                ep, "model", "mock_snapshot",
                dict(pricing_status="configured", currency="CNY", price_tiers=[dict(
                    currency="CNY", billing_unit="per_million_tokens", input_per_million=price,
                    output_per_million=price, conditions={})]),
                dict(rpm=1000, tpm=100000, concurrency=10),
            ))
            for i in range(100):
                success = i >= failures
                self.state.windows.record("model", Observation(
                    self.now - timedelta(seconds=1), ep, False, success,
                    "success" if success else "error", 200 if success else 500,
                    200 if success else None, None, None,
                ))
        self.request = RoutingRequest("request", "model", self.now, False, 100, 50, RequestSLO(e2e_ms=1000))
        self.engine = RoutingEngine(self.state)
        self.options = dict(target_samples=1, min_samples=1, prior_strength=0)

    def session(self, **overrides):
        return self.engine.start_failover(self.request, retry_safe=True, **self.options, **overrides)

    def fail_first(self, session, status=503, milliseconds=10):
        first = session.next_attempt(cutoff=self.now)
        self.assertEqual(first.endpoint_id, "primary")
        session.complete_attempt(AttemptResult(False, status), cutoff=self.now + timedelta(milliseconds=milliseconds))

    def test_primary_failure_uses_stable_backup_and_records_feedback(self):
        session = self.session()
        self.assertEqual(session.route_plan, ("primary", "stable", "third"))
        before = self.state.catalog.runtime_state("model", "primary")
        self.fail_first(session)
        second = session.next_attempt(cutoff=self.now + timedelta(milliseconds=20))
        self.assertEqual(second.endpoint_id, "stable")
        self.assertEqual(second.decision.selection_phase, "backup")
        self.assertFalse(next(c for c in second.decision.candidates if c.selected).pareto)
        session.complete_attempt(AttemptResult(True, 200, e2e_ms=210), cutoff=self.now + timedelta(milliseconds=230))
        self.assertEqual(session.status, "succeeded")
        self.assertIsNone(session.next_attempt())
        after = self.state.catalog.runtime_state("model", "primary")
        self.assertEqual(after.current_concurrency, before.current_concurrency)
        self.assertGreater(after.cooldown_until, self.now)
        self.assertEqual(after.health_sample_count, 101)
        self.assertAlmostEqual(session.estimated_cumulative_cost, .0009)
        json.dumps(session.to_dict(), allow_nan=False)

    def test_three_distinct_attempts_then_stop(self):
        session = self.session()
        self.fail_first(session)
        second = session.next_attempt()
        self.assertEqual(second.endpoint_id, "stable")
        session.complete_attempt(AttemptResult(False, 429, retry_after_seconds=12))
        third = session.next_attempt()
        self.assertEqual(third.endpoint_id, "third")
        session.complete_attempt(AttemptResult(False, 502))
        self.assertEqual(session.stop_reason, "attempts_exhausted")
        self.assertIsNone(session.next_attempt())
        self.assertEqual(len({r["endpoint_id"] for r in session.attempts}), 3)

    def test_non_retryable_client_errors_stop(self):
        for status in (400, 401, 403, 404):
            with self.subTest(status=status):
                session = self.session()
                self.fail_first(session, status)
                self.assertEqual(session.stop_reason, "non_retryable_error")
                self.assertIsNone(session.next_attempt())

    def test_retry_must_be_explicitly_safe(self):
        session = self.engine.start_failover(self.request, **self.options)
        self.fail_first(session)
        self.assertEqual(session.stop_reason, "request_not_retry_safe")
        self.assertIsNone(session.next_attempt())

    def test_client_response_started_never_retries(self):
        session = self.session()
        session.next_attempt()
        session.complete_attempt(AttemptResult(False, 503, response_started=True))
        self.assertEqual(session.stop_reason, "response_already_started")
        self.assertIsNone(session.next_attempt())

    def test_transport_failure_can_retry_before_output(self):
        session = self.session()
        session.next_attempt()
        session.complete_attempt(AttemptResult(False, error_kind="timeout"))
        self.assertEqual(session.next_attempt().endpoint_id, "stable")

    def test_revalidates_backup_health_and_capacity(self):
        session = self.session()
        self.fail_first(session)
        self.state.catalog.set_enabled("model", "stable", False)
        self.assertEqual(session.next_attempt().endpoint_id, "third")
        session.complete_attempt(AttemptResult(False, 500))
        self.state.catalog.update_model_endpoint_load("model", "stable", current_concurrency=10)
        self.assertIsNone(session.next_attempt())
        self.assertEqual(session.stop_reason, "no_feasible_remaining_endpoint")

    def test_retry_consumes_original_deadline_budget(self):
        session = self.session()
        self.fail_first(session, milliseconds=1200)
        self.assertIsNone(session.next_attempt())
        self.assertIn("slo_infeasible", session.last_decision.to_dict()["no_candidate_categories"])

    def test_max_attempts_includes_primary(self):
        session = self.session(retry_policy=RetryPolicy(max_attempts=1))
        self.fail_first(session)
        self.assertEqual(session.stop_reason, "attempts_exhausted")
        self.assertIsNone(session.next_attempt())

    def test_predicted_retry_cost_budget_stops_unaffordable_attempt(self):
        session = self.session(retry_policy=RetryPolicy(max_estimated_total_cost=.0002))
        self.fail_first(session)
        self.assertIsNone(session.next_attempt())
        self.assertEqual(session.stop_reason, "estimated_cost_budget_exhausted")
        self.assertEqual(len(session.attempts), 1)

    def test_live_clock_is_injectable(self):
        current = [self.now]
        session = self.session(clock=lambda: current[0])
        session.next_attempt()
        current[0] += timedelta(milliseconds=1100)
        session.complete_attempt(AttemptResult(False, 503))
        self.assertIsNone(session.next_attempt())

    def test_removed_offering_does_not_lose_attempt_feedback_record(self):
        session = self.session()
        session.next_attempt()
        self.state.remove_endpoint("model", "primary")
        session.complete_attempt(AttemptResult(False, 503))
        self.assertEqual(session.attempts[0]["feedback_skipped_reason"], "offering_removed")
        self.assertEqual(session.next_attempt().endpoint_id, "stable")

    def test_primary_removed_before_dispatch_uses_stable_backup(self):
        session = self.session()
        self.state.remove_endpoint("model", "primary")
        self.assertEqual(session.next_attempt().endpoint_id, "stable")

    def test_session_enforces_attempt_lifecycle_and_time(self):
        session = self.session()
        with self.assertRaises(RuntimeError):
            session.complete_attempt(AttemptResult(True, 200))
        session.next_attempt()
        with self.assertRaises(RuntimeError):
            session.next_attempt()
        with self.assertRaises(ValueError):
            session.complete_attempt(AttemptResult(False, 500), cutoff=self.now - timedelta(seconds=1))

    def test_invalid_retry_configuration_and_results(self):
        for fields in ({"max_attempts": 0}, {"max_attempts": True}, {"rate_limit_cooldown_seconds": -1},
                       {"retryable_statuses": (200,)}, {"max_estimated_total_cost": float("nan")}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                RetryPolicy(**fields)
        with self.assertRaises(ValueError):
            AttemptResult(True, 500)
        with self.assertRaises(ValueError):
            AttemptResult(False, 503, retry_after_seconds=-1)


if __name__ == "__main__":
    unittest.main()
