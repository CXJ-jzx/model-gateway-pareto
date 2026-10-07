"""Request-aware admission, metric constraints and auditable endpoint decisions.

The shared statistics/Pareto/score implementation lives in routing_engine.py.
This module consumes arrival-time inputs only; it neither executes nor simulates
an endpoint, and never inserts the request's eventual result into its window.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import median
from time import perf_counter_ns
from typing import Any

from .memory_state import InMemoryRoutingState
from .models import Candidate, StrategyParameters
from .pricing import resolve_request_price
from .routing_engine import (
    apply_request_cost, backup_sort_key, build_route_order, combine_stream_performance,
    compute_candidate, estimate_stability, finalize_selection,
)


def _aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


@dataclass(frozen=True)
class RequestSLO:
    e2e_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    tier: str = "standard"

    def __post_init__(self) -> None:
        if self.tier not in {"strict", "standard", "relaxed"}:
            raise ValueError("SLO tier must be strict, standard or relaxed")
        for name in ("e2e_ms", "ttft_ms", "tpot_ms"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive, or None")


@dataclass(frozen=True)
class RoutingRequest:
    request_id: str
    model_id: str
    arrived_at: datetime
    is_stream: bool
    predicted_input_tokens: int
    predicted_output_tokens: int
    slo: RequestSLO
    priority: str = "normal"
    workload: dict = field(default_factory=dict)
    urgency: dict = field(default_factory=dict)
    input_provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id or not self.model_id:
            raise ValueError("request_id and model_id are required")
        _aware(self.arrived_at, "arrived_at")
        if not isinstance(self.is_stream, bool) or not isinstance(self.slo, RequestSLO):
            raise ValueError("is_stream must be bool and slo must be RequestSLO")
        for name in ("predicted_input_tokens", "predicted_output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.priority not in {"high", "normal", "low"}:
            raise ValueError("priority must be high, normal or low")
        if self.is_stream and (self.slo.ttft_ms is None or self.slo.tpot_ms is None):
            raise ValueError("stream requests require TTFT and TPOT SLOs")
        if not self.is_stream and self.slo.e2e_ms is None:
            raise ValueError("nonstream requests require an E2E SLO")

    @classmethod
    def from_prepared(cls, row: dict) -> "RoutingRequest":
        """Read the existing data pipeline's requests.jsonl, not completion logs."""
        try:
            mode = row["stream_type"]
            if mode not in {"stream", "nonstream"}:
                raise ValueError("stream_type must be stream or nonstream")
            slo = row["slo"]
            return cls(
                request_id=row["request_id"], model_id=row["model_id"],
                arrived_at=datetime.fromisoformat(row["arrival_time"]), is_stream=mode == "stream",
                predicted_input_tokens=row["token_predictions"]["input_tokens"],
                predicted_output_tokens=row["token_predictions"]["output_tokens"],
                slo=RequestSLO(**{k: slo[k] for k in ("e2e_ms", "ttft_ms", "tpot_ms", "tier") if k in slo}),
                priority=row["priority"], workload=dict(row.get("workload", {})),
                urgency=dict(row.get("urgency", {})),
                input_provenance=dict(
                    slo_source=slo.get("source"), slo_profile_key=slo.get("profile_key"),
                    slo_version=slo.get("version"), priority_source=row.get("priority_source"),
                    input_prediction_source=row["token_predictions"].get("input_source"),
                    output_prediction_source=row["token_predictions"].get("output_source"),
                    split=row.get("split"), source_line=row.get("source_line"),
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Invalid prepared request: {exc}") from exc


@dataclass
class RoutingDecision:
    request: RoutingRequest
    cutoff: datetime
    status: str
    selected_endpoint: str | None
    parameters: StrategyParameters
    candidates: list[Candidate]
    endpoint_checks: dict[str, dict]
    timings_us: dict[str, float]
    cost_reference: float | None
    require_slo: bool
    catalog_version_before: int
    catalog_version_after: int
    unknown_capacity_policy: str = "allow"
    cost_mode: str = "predicted_request"
    top_k: int = 3
    backup_pool: str = "all_feasible"
    stability_z: float = 1.96
    ordered_endpoints: list[str] = field(default_factory=list)
    selection_phase: str = "primary"
    price_tier_mode: str = "auto"

    @property
    def backup_endpoints(self) -> list[str]:
        return self.ordered_endpoints[1:]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["request"]["arrived_at"] = self.request.arrived_at.isoformat()
        payload["cutoff"] = self.cutoff.isoformat()
        positive = [c.cost_raw for c in self.candidates if c.cost_raw > 0]
        plot_reference = median(positive) if positive else 1.0
        payload.update(
            schema_version="2.0", strategy="request_aware_pareto_linear_with_stable_backups",
            selection_mode="deterministic_argmin" if self.selection_phase == "primary" else "deterministic_stability_first",
            cost_mode=self.cost_mode, backup_endpoints=self.backup_endpoints,
            cost_unit="currency_per_million_tokens" if self.cost_mode == "unit_price" else "currency_per_request",
            rho_active=self.cost_mode != "predicted_request",
            tpm_mode="predicted_input_plus_output_reservation", capacity_reserved=False,
            catalog_stable=self.catalog_version_before == self.catalog_version_after,
            no_candidate_action="handoff_to_caller" if self.status == "no_candidate" else None,
            no_candidate_categories=_failure_categories(self.endpoint_checks) if self.status == "no_candidate" else [],
            active_features=["model_id", "stream_type", "token_predictions", "slo", "endpoint_state", "recent_performance", "historical_prior", "lambda", "eta"]
                            + (["rho"] if self.cost_mode != "predicted_request" else []),
            metadata_only=["priority", "workload", "urgency"],
            cost_normalization=dict(
                method="divide_by_median_positive_cost",
                reference=self.cost_reference if self.cost_reference is not None else plot_reference,
                scope="feasible_candidates" if self.cost_reference is not None else "priced_candidates_for_plot_only",
            ),
            tie_break_order=(["score_ascending", "success_rate_descending", "effective_samples_descending", "endpoint_id_ascending"]
                             if self.selection_phase == "primary" else
                             ["stability_lower_bound_descending", "stability_success_rate_descending",
                              "stability_sample_count_descending", "routing_score_ascending", "endpoint_id_ascending"]),
            backup_order=["stability_lower_bound_descending", "stability_success_rate_descending",
                          "stability_sample_count_descending", "routing_score_ascending", "endpoint_id_ascending"],
            stability_method="weighted_pair_success_Wilson_lower_proxy_not_calibrated_confidence",
            stability_window_minutes=60,
        )
        return payload


def _failure_categories(checks: dict[str, dict]) -> list[str]:
    if not checks:
        return ["no_offering"]
    reasons = [reason for check in checks.values() for reason in check["reasons"]]
    categories = set()
    for reason in reasons:
        if reason.startswith("capacity_unknown_") or reason.startswith(("missing_", "insufficient_samples_")):
            categories.add("missing_evidence_or_configuration")
        elif reason.startswith("capacity_"):
            categories.add("temporary_capacity")
        elif reason == "cooldown":
            categories.add("temporary_cooldown")
        elif reason.startswith("slo_"):
            categories.add("slo_infeasible")
        elif reason in {"disabled", "endpoint_unhealthy"}:
            categories.add("unavailable_endpoint")
        else:
            categories.add("insufficient_quality_or_evidence")
    return sorted(categories)


def _capacity_check(runtime, request: RoutingRequest, unknown_policy: str) -> tuple[dict, list[str]]:
    checks, reasons = {}, []
    for name, increment in (("rpm", 1), ("tpm", request.predicted_input_tokens + request.predicted_output_tokens),
                            ("concurrency", 1)):
        current, limit = getattr(runtime, f"current_{name}"), getattr(runtime, f"capacity_{name}")
        projected = current + increment
        checks[name] = dict(current=current, increment=increment, projected=projected, limit=limit,
                            remaining=None if limit is None else limit - projected)
        # Zero explicitly means unavailable; None means no known configured limit.
        if limit is not None and (limit == 0 or projected > limit):
            reasons.append(f"capacity_{name}")
        if limit is None and unknown_policy == "block":
            reasons.append(f"capacity_unknown_{name}")
    return checks, reasons


def route_request(
    state: InMemoryRoutingState, request: RoutingRequest, *, cutoff: datetime | None = None,
    currency: str = "CNY", require_slo: bool = True, unknown_capacity_policy: str = "allow",
    cost_mode: str | None = None, top_k: int | None = None, backup_pool: str | None = None,
    stability_z: float | None = None, exclude_endpoints=(), allowed_endpoints=None,
    selection_phase: str = "primary", price_tier_mode: str = "auto", **parameter_overrides,
) -> RoutingDecision:
    started = perf_counter_ns()
    if not isinstance(request, RoutingRequest):
        raise ValueError("request must be RoutingRequest; use from_prepared for requests.jsonl")
    cutoff = cutoff or request.arrived_at
    _aware(cutoff, "cutoff")
    if cutoff < request.arrived_at:
        raise ValueError("cutoff cannot precede arrival")
    if not isinstance(require_slo, bool) or not isinstance(currency, str) or not currency:
        raise ValueError("Invalid require_slo/currency")
    if unknown_capacity_policy not in {"allow", "block"}:
        raise ValueError("unknown_capacity_policy must be allow or block")
    if price_tier_mode not in {"auto", "configured"}:
        raise ValueError("price_tier_mode must be auto or configured")
    fixed = {"stream", "slo_e2e_ms", "slo_ttft_ms", "slo_tpot_ms", "temperature"}
    if fixed.intersection(parameter_overrides):
        raise ValueError("Request SLO/mode and deterministic selection cannot be overridden; edit the request")
    policy = state.policies.get(request.model_id)
    cost_mode = policy.cost_mode if cost_mode is None else cost_mode
    top_k = policy.top_k if top_k is None else top_k
    backup_pool = policy.backup_pool if backup_pool is None else backup_pool
    stability_z = policy.stability_z if stability_z is None else stability_z
    if cost_mode not in {"predicted_request", "weighted_predicted_request", "unit_price"}:
        raise ValueError("Invalid cost_mode")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if backup_pool not in {"all_feasible", "pareto"} or selection_phase not in {"primary", "backup"}:
        raise ValueError("Invalid backup_pool/selection_phase")
    if isinstance(stability_z, bool) or not isinstance(stability_z, (int, float)) or not math.isfinite(stability_z) or stability_z <= 0:
        raise ValueError("stability_z must be finite and positive")
    endpoint_sets = {}
    for name, endpoints in (("exclude_endpoints", exclude_endpoints), ("allowed_endpoints", allowed_endpoints)):
        if endpoints is None:
            endpoint_sets[name] = None
            continue
        if isinstance(endpoints, (str, bytes)):
            raise ValueError(f"{name} must be a collection of endpoint IDs")
        try:
            ids = tuple(endpoints)
        except TypeError as exc:
            raise ValueError(f"{name} must be a collection of endpoint IDs") from exc
        if any(not isinstance(ep, str) for ep in ids):
            raise ValueError(f"{name} must be a collection of endpoint IDs")
        endpoint_sets[name] = set(ids)
    excluded = endpoint_sets["exclude_endpoints"] or set()
    allowed = endpoint_sets["allowed_endpoints"]
    overrides = dict(parameter_overrides, temperature=0.0)
    for name in ("e2e", "ttft", "tpot"):
        value = getattr(request.slo, f"{name}_ms")
        if value is not None:
            overrides[f"slo_{name}_ms"] = value
    params = policy.to_parameters(request.is_stream, **overrides)
    if max(params.window_candidates_minutes) > state.windows.max_age.total_seconds() / 60:
        raise ValueError("Requested window exceeds the in-memory retention period")
    version_before = state.catalog.version
    offerings = state.catalog.candidates(request.model_id, at_time=cutoff, include_ineligible=True)
    indexed = perf_counter_ns()
    checks, candidates = {}, []
    admission_ns, metrics_ns = 0, 0
    elapsed_ms = (cutoff - request.arrived_at).total_seconds() * 1000

    for offering in sorted(offerings, key=lambda item: item.endpoint_id):
        ep = offering.endpoint_id
        phase = perf_counter_ns()
        runtime = state.catalog.runtime_state(request.model_id, ep)
        health = state.catalog.endpoint_state(ep).health_status
        capacity, reasons = _capacity_check(runtime, request, unknown_capacity_policy)
        if ep in excluded:
            reasons.append("previous_attempt")
        if allowed is not None and ep not in allowed:
            reasons.append("not_in_route_plan")
        if not runtime.enabled:
            reasons.append("disabled")
        if runtime.cooldown_until is not None and cutoff < runtime.cooldown_until:
            reasons.append("cooldown")
        if health.lower() in {"unhealthy", "down", "disabled"}:
            reasons.append("endpoint_unhealthy")
        price, price_resolution = resolve_request_price(
            offering, runtime.price_tier_index, request.predicted_input_tokens, currency,
            tier_mode=price_tier_mode,
        )
        if price is None:
            reasons.append("missing_price")
        check = checks[ep] = dict(
            reasons=reasons, capacity=capacity, slo={}, enabled=runtime.enabled, health_status=health,
            cooldown_until=runtime.cooldown_until.isoformat() if runtime.cooldown_until else None,
            price_tier_index=price.tier_index if price is not None else None,
            configured_price_tier_index=runtime.price_tier_index,
            price_resolution=price_resolution, configured_price_currency=offering.price_config.get("currency"),
            selected_price=None if price is None else asdict(price),
        )
        admission_ns += perf_counter_ns() - phase
        # Still calculate priced excluded points for the explanatory graph.
        if price is None:
            continue
        phase = perf_counter_ns()
        observations = state.windows.recent(request.model_id, ep, request.is_stream, cutoff,
                                            max(params.window_candidates_minutes))
        candidate = compute_candidate(price, observations, cutoff, runtime.historical_prior, params)
        apply_request_cost(candidate, request.predicted_input_tokens, request.predicted_output_tokens,
                           cost_mode, params.rho_input_price)
        health_observations = []
        for mode in (False, True):
            health_observations.extend(state.windows.recent(request.model_id, ep, mode, cutoff, 60))
        estimate_stability(candidate, health_observations, cutoff, runtime.historical_prior, params, stability_z)
        prior = runtime.historical_prior
        relevant = ("ttft", "tpot") if request.is_stream else ("e2e",)
        if request.is_stream and require_slo and request.slo.e2e_ms is not None:
            relevant += ("e2e",)
        # Never silently drop an active stream metric, even if its score weight is zero.
        for name in relevant:
            estimate = getattr(candidate, f"estimated_{name}_ms")
            if estimate is None:
                reasons.append(f"missing_{name}")
            historical = (getattr(prior, {"e2e": "e2e_p95_ms", "ttft": "ttft_p95_ms", "tpot": "tpot_proxy_ms"}[name])
                          if prior else None)
            if estimate is not None and historical is None and candidate.metric_effective_samples[name] < params.min_samples:
                reasons.append(f"insufficient_samples_{name}")
        if request.is_stream:
            candidate.performance = combine_stream_performance(candidate.estimated_ttft_ms, candidate.estimated_tpot_ms,
                                                               params.eta_ttft, params.slo_ttft_ms, params.slo_tpot_ms)
        else:
            candidate.performance = (candidate.estimated_e2e_ms / params.slo_e2e_ms
                                     if candidate.estimated_e2e_ms is not None else None)
        for name in (("e2e", "ttft", "tpot") if request.is_stream else ("e2e",)):
            budget = getattr(request.slo, f"{name}_ms")
            if budget is None:
                continue
            if name in {"e2e", "ttft"}:
                budget -= elapsed_ms
            estimate = getattr(candidate, f"estimated_{name}_ms")
            check["slo"][name] = dict(estimated_ms=estimate, budget_ms=budget,
                                      ratio=estimate / budget if estimate is not None and budget > 0 else None)
            if require_slo:
                if estimate is None:
                    reasons.append(f"missing_{name}")
                elif budget <= 0 or estimate > budget:
                    reasons.append(f"slo_{name}")
        if candidate.exclusion_reason:
            reasons.append(candidate.exclusion_reason)
        reasons[:] = list(dict.fromkeys(reasons))
        candidate.feasible = not reasons
        candidate.exclusion_reason = "; ".join(reasons)
        check["metrics_source"] = "per_metric_online_prior_blend"
        check["prior_source"] = prior.source if prior else None
        candidates.append(candidate)
        metrics_ns += perf_counter_ns() - phase

    before_selection = perf_counter_ns()
    feasible = [c for c in candidates if c.feasible and c.performance is not None]
    reference = None
    selected = None
    ordered = []
    if feasible:
        reference = median([c.cost_raw for c in feasible if c.cost_raw > 0]) if any(c.cost_raw > 0 for c in feasible) else 1.0
        selected = finalize_selection(candidates, params)
        if selection_phase == "backup":
            pool = [c for c in feasible if backup_pool == "all_feasible" or c.pareto]
            selected.selected = False
            selected.selection_probability = 0.0
            selected = min(pool, key=backup_sort_key)
            selected.selected = True
            selected.selection_probability = 1.0
        ordered = build_route_order(candidates, selected, top_k, backup_pool)
        if selection_phase == "backup":
            selected.route_role = "retry_backup"
    else:
        # Plotting-only reference; not represented as a selectable cost baseline.
        positive = [c.cost_raw for c in candidates if c.cost_raw > 0]
        plot_reference = median(positive) if positive else 1.0
        for candidate in candidates:
            candidate.cost_normalized = candidate.cost_raw / plot_reference
    ended = perf_counter_ns()
    return RoutingDecision(
        request=request, cutoff=cutoff, status="selected" if selected else "no_candidate",
        selected_endpoint=selected.endpoint_id if selected else None, parameters=params,
        candidates=candidates, endpoint_checks=checks, cost_reference=reference, require_slo=require_slo,
        catalog_version_before=version_before, catalog_version_after=state.catalog.version,
        unknown_capacity_policy=unknown_capacity_policy,
        cost_mode=cost_mode, top_k=top_k, backup_pool=backup_pool, stability_z=stability_z,
        ordered_endpoints=ordered, selection_phase=selection_phase, price_tier_mode=price_tier_mode,
        timings_us=dict(index_lookup=(indexed - started) / 1000, admission_checks=admission_ns / 1000,
                        metric_estimation=metrics_ns / 1000, pareto_and_score=(ended - before_selection) / 1000,
                        route_total=(ended - started) / 1000),
    )
