"""Guarded failover planning and attempt feedback, without an HTTP simulator.

The caller sends requests, measures outcomes and maintains load counters. This
planner rechecks routing constraints before each distinct endpoint attempt.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from .models import Observation
from .request_routing import RoutingDecision, RoutingRequest, _aware


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # Includes the first attempt, not three extra retries.
    retryable_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    retryable_transport_errors: tuple[str, ...] = ("timeout", "connection_error")
    rate_limit_cooldown_seconds: float = 30.0
    server_error_cooldown_seconds: float = 10.0
    max_estimated_total_cost: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        for status in self.retryable_statuses:
            if isinstance(status, bool) or not isinstance(status, int) or not 400 <= status <= 599:
                raise ValueError("retryable_statuses must contain HTTP error codes")
        for value in (self.rate_limit_cooldown_seconds, self.server_error_cooldown_seconds, self.max_estimated_total_cost):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError("Retry budgets/cooldown durations must be finite and nonnegative")


@dataclass(frozen=True)
class AttemptResult:
    success: bool
    http_status: int | None = None
    error_kind: str | None = None
    response_started: bool = False  # Any bytes already delivered to the client.
    e2e_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    retry_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool) or not isinstance(self.response_started, bool):
            raise ValueError("success/response_started must be booleans")
        if self.http_status is not None and (isinstance(self.http_status, bool)
                                             or not isinstance(self.http_status, int) or not 100 <= self.http_status <= 599):
            raise ValueError("Invalid HTTP status")
        if self.success and (self.error_kind is not None or (self.http_status is not None and self.http_status >= 400)):
            raise ValueError("Successful results cannot contain error status/kind")
        for value in (self.e2e_ms, self.ttft_ms, self.tpot_ms, self.retry_after_seconds):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError("Attempt metrics must be finite and nonnegative")


@dataclass(frozen=True)
class AttemptPlan:
    endpoint_id: str
    attempt_number: int
    decision: RoutingDecision
    estimated_request_cost: float


class FailoverSession:
    def __init__(self, engine, request: RoutingRequest, *, retry_safe: bool = False,
                 retry_policy: RetryPolicy | None = None, cutoff: datetime | None = None, clock=None, **route_options) -> None:
        if not isinstance(retry_safe, bool):
            raise ValueError("retry_safe must be a boolean")
        if "selection_phase" in route_options:
            raise ValueError("The failover planner controls selection_phase")
        if clock is not None and not callable(clock):
            raise ValueError("clock must be callable, or None")
        self.clock = clock
        self.engine, self.request = engine, request
        self.retry_safe = retry_safe
        self.retry_policy = retry_policy or RetryPolicy()
        if not isinstance(self.retry_policy, RetryPolicy):
            raise ValueError("retry_policy must be RetryPolicy")
        self.route_options = dict(route_options)
        self._last_time = cutoff or (clock() if clock else request.arrived_at)
        self.initial_decision = engine.route_request(request, cutoff=self._last_time, **route_options)
        self.route_plan = tuple(self.initial_decision.ordered_endpoints)
        self.attempts: list[dict] = []
        self._inflight: AttemptPlan | None = None
        self.estimated_cumulative_cost = 0.0
        self.status = "ready" if self.route_plan else "stopped"
        self.stop_reason = None if self.route_plan else "no_candidate"
        self.last_decision = self.initial_decision

    def _time(self, cutoff: datetime | None) -> datetime:
        now = cutoff or (self.clock() if self.clock else self._last_time)
        _aware(now, "cutoff")
        if now < self._last_time:
            raise ValueError("Attempt time cannot move backwards")
        return now

    def next_attempt(self, *, cutoff: datetime | None = None) -> AttemptPlan | None:
        if self._inflight is not None:
            raise RuntimeError("Complete the current attempt before requesting another")
        if self.status != "ready":
            return None
        now = self._time(cutoff)
        self._last_time = now
        if len(self.attempts) >= min(self.retry_policy.max_attempts, len(self.route_plan)):
            self.status, self.stop_reason = "stopped", "attempts_exhausted"
            return None
        attempted = [row["endpoint_id"] for row in self.attempts]
        options = {k: v for k, v in self.route_options.items() if k not in {"allowed_endpoints", "exclude_endpoints"}}
        # The first endpoint is kept if still feasible. If unavailable, switch to
        # stable backups instead of choosing the cheapest remaining point.
        decision = None
        if not self.attempts:
            decision = self.engine.route_request(self.request, cutoff=now, allowed_endpoints=self.route_plan[:1],
                                                  exclude_endpoints=attempted, **options)
        if decision is None or decision.selected_endpoint is None:
            decision = self.engine.route_request(self.request, cutoff=now, allowed_endpoints=self.route_plan,
                                                  exclude_endpoints=attempted, selection_phase="backup", **options)
        self.last_decision = decision
        if decision.selected_endpoint is None:
            self.status, self.stop_reason = "stopped", "no_feasible_remaining_endpoint"
            return None
        candidate = next(c for c in decision.candidates if c.selected)
        cost = candidate.estimated_request_cost
        budget = self.retry_policy.max_estimated_total_cost
        if budget is not None and self.estimated_cumulative_cost + cost > budget:
            self.status, self.stop_reason = "stopped", "estimated_cost_budget_exhausted"
            return None
        plan = AttemptPlan(candidate.endpoint_id, len(self.attempts) + 1, decision, cost)
        self._inflight = plan
        self.estimated_cumulative_cost += cost
        self.attempts.append(dict(endpoint_id=plan.endpoint_id, attempt_number=plan.attempt_number,
                                  planned_at=now.isoformat(), finished_at=None, result=None,
                                  estimated_request_cost=cost, decision=decision.to_dict()))
        self.status = "awaiting_result"
        return plan

    def complete_attempt(self, result: AttemptResult, *, cutoff: datetime | None = None) -> None:
        if self._inflight is None:
            raise RuntimeError("No outstanding attempt")
        if not isinstance(result, AttemptResult):
            raise ValueError("result must be AttemptResult")
        now = self._time(cutoff)
        endpoint_id = self._inflight.endpoint_id
        observation = Observation(
            occurred_at=now, endpoint_id=endpoint_id, is_stream=self.request.is_stream,
            success=result.success, result="success" if result.success else (result.error_kind or "error"),
            http_status=result.http_status, e2e_ms=result.e2e_ms,
            ttft_ms=result.ttft_ms if self.request.is_stream else None,
            tpot_ms=result.tpot_ms if self.request.is_stream else None,
        )
        try:
            runtime = self.engine.state.catalog.runtime_state(self.request.model_id, endpoint_id)
        except KeyError:
            runtime = None
        if runtime is not None:
            self.engine.state.record_observation(self.request.model_id, observation)
        cooldown = 0.0
        if not result.success and result.http_status == 429:
            cooldown = (result.retry_after_seconds if result.retry_after_seconds is not None
                        else self.retry_policy.rate_limit_cooldown_seconds)
        elif not result.success and result.http_status is not None and 500 <= result.http_status <= 599:
            cooldown = self.retry_policy.server_error_cooldown_seconds
        if cooldown > 0 and runtime is not None:
            old = self.engine.state.catalog.runtime_state(self.request.model_id, endpoint_id).cooldown_until
            deadline = now + timedelta(seconds=cooldown)
            self.engine.state.catalog.set_cooldown(self.request.model_id, endpoint_id,
                                                   max(old, deadline) if old is not None else deadline)
        self._last_time = now
        self.attempts[-1].update(finished_at=now.isoformat(), result=asdict(result),
                                  feedback_skipped_reason="offering_removed" if runtime is None else None)
        self._inflight = None
        if result.success:
            self.status, self.stop_reason = "succeeded", "success"
        elif result.response_started:
            self.status, self.stop_reason = "stopped", "response_already_started"
        elif not self.retry_safe:
            self.status, self.stop_reason = "stopped", "request_not_retry_safe"
        elif (result.http_status in self.retry_policy.retryable_statuses
              or (result.http_status is None and result.error_kind in self.retry_policy.retryable_transport_errors)):
            self.status, self.stop_reason = "ready", None
        else:
            self.status, self.stop_reason = "stopped", "non_retryable_error"
        if self.status == "ready" and len(self.attempts) >= min(self.retry_policy.max_attempts, len(self.route_plan)):
            self.status, self.stop_reason = "stopped", "attempts_exhausted"

    def to_dict(self) -> dict:
        return dict(
            request_id=self.request.request_id, status=self.status, stop_reason=self.stop_reason,
            route_plan=list(self.route_plan), retry_safe=self.retry_safe, retry_policy=asdict(self.retry_policy),
            estimated_cumulative_cost=self.estimated_cumulative_cost, attempts=self.attempts,
            initial_decision=self.initial_decision.to_dict(), last_decision=self.last_decision.to_dict(),
            capacity_reserved=False, http_dispatched_by_planner=False,
            cost_note="Sum of full predicted attempt costs; not an actual invoice",
        )
