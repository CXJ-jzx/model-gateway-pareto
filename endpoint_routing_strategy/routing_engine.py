from __future__ import annotations

import math
from datetime import datetime, timedelta
from statistics import median
from typing import Sequence

try:
    from .memory_state import InMemoryRoutingState, ModelEndpointStateIndex
    from .models import Candidate, Observation, Price, Prior, StrategyParameters
except ImportError:  # 兼容直接运行目录内脚本
    from memory_state import InMemoryRoutingState, ModelEndpointStateIndex
    from models import Candidate, Observation, Price, Prior, StrategyParameters


def weighted_average(values: Sequence[tuple[float, float]]) -> float | None:
    total_weight = sum(weight for _, weight in values if weight > 0)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in values if weight > 0) / total_weight


def weighted_quantile(values: Sequence[tuple[float, float]], quantile: float) -> float | None:
    usable = sorted((float(value), float(weight)) for value, weight in values if weight > 0)
    if not usable:
        return None
    target = min(1.0, max(0.0, quantile)) * sum(weight for _, weight in usable)
    cumulative = 0.0
    for value, weight in usable:
        cumulative += weight
        if cumulative >= target:
            return value
    return usable[-1][0]


def effective_sample_size(weights: Sequence[float]) -> float:
    positive = [weight for weight in weights if weight > 0]
    if not positive:
        return 0.0
    total = sum(positive)
    squared = sum(weight * weight for weight in positive)
    return (total * total) / squared if squared > 0 else 0.0


def combine_stream_performance(
    ttft_ms: float | None,
    tpot_ms: float | None,
    eta_ttft: float,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
) -> float | None:
    components = []
    if ttft_ms is not None:
        components.append((ttft_ms / slo_ttft_ms, eta_ttft))
    if tpot_ms is not None:
        components.append((tpot_ms / slo_tpot_ms, 1.0 - eta_ttft))
    return weighted_average(components)


def select_window(
    observations: Sequence[Observation],
    cutoff: datetime,
    target_samples: int,
    candidates_minutes: Sequence[int],
) -> tuple[int, list[Observation]]:
    if not candidates_minutes:
        raise ValueError("窗口候选列表不能为空")
    selected = []
    for minutes in candidates_minutes:
        start = cutoff - timedelta(minutes=minutes)
        selected = [item for item in observations if start <= item.occurred_at <= cutoff]
        if len(selected) >= target_samples:
            return int(minutes), selected
    return int(candidates_minutes[-1]), selected


def _metric_neff(values: Sequence[tuple[float, float]]) -> float:
    return effective_sample_size([weight for _, weight in values])


def _blend_metric(online: float | None, prior: float | None, neff: float, strength: float) -> tuple[float | None, float]:
    """Per-metric estimate; raw online P95 remains separately available for audit."""
    if online is None:
        return prior, 1.0 if prior is not None else 0.0
    if prior is None:
        return online, 0.0
    weight = strength / (neff + strength) if neff + strength > 0 else 0.0
    return (1 - weight) * online + weight * prior, weight


def compute_candidate(
    price: Price,
    endpoint_observations: Sequence[Observation],
    cutoff: datetime,
    prior: Prior | None,
    params: StrategyParameters,
) -> Candidate:
    matching = [item for item in endpoint_observations if item.is_stream == params.stream]
    window_minutes, window = select_window(
        matching,
        cutoff,
        params.target_samples,
        params.window_candidates_minutes,
    )
    weighted = []
    for item in window:
        age_minutes = max(0.0, (cutoff - item.occurred_at).total_seconds() / 60.0)
        weight = math.pow(2.0, -age_minutes / params.half_life_minutes)
        weighted.append((item, weight))

    attempt_weights = [weight for _, weight in weighted]
    n_eff = effective_sample_size(attempt_weights)
    online_success = weighted_average(
        [(1.0 if item.success else 0.0, weight) for item, weight in weighted]
    )
    e2e_values = [(item.e2e_ms, weight) for item, weight in weighted if item.success and item.e2e_ms is not None]
    ttft_values = [(item.ttft_ms, weight) for item, weight in weighted if item.success and item.ttft_ms is not None]
    tpot_values = [(item.tpot_ms, weight) for item, weight in weighted if item.success and item.tpot_ms is not None]
    p95_e2e = weighted_quantile(e2e_values, 0.95)
    p95_ttft = weighted_quantile(ttft_values, 0.95)
    p95_tpot = weighted_quantile(tpot_values, 0.95)
    metric_neffs = {"e2e": _metric_neff(e2e_values), "ttft": _metric_neff(ttft_values), "tpot": _metric_neff(tpot_values)}
    estimates, prior_weights = {}, {}
    for name, online, historical in (
        ("e2e", p95_e2e, prior.e2e_p95_ms if prior else None),
        ("ttft", p95_ttft, prior.ttft_p95_ms if prior else None),
        ("tpot", p95_tpot, prior.tpot_proxy_ms if prior else None),
    ):
        estimates[name], prior_weights[name] = _blend_metric(online, historical, metric_neffs[name], params.prior_strength)

    if params.stream:
        online_performance = combine_stream_performance(
            p95_ttft,
            p95_tpot,
            params.eta_ttft,
            params.slo_ttft_ms,
            params.slo_tpot_ms,
        )
        metric_neff = min(
            [value for value in (_metric_neff(ttft_values), _metric_neff(tpot_values)) if value > 0]
            or [0.0]
        )
        prior_performance = (
            combine_stream_performance(
                prior.ttft_p95_ms,
                prior.tpot_proxy_ms,
                params.eta_ttft,
                params.slo_ttft_ms,
                params.slo_tpot_ms,
            )
            if prior
            else None
        )
    else:
        online_performance = p95_e2e / params.slo_e2e_ms if p95_e2e is not None else None
        metric_neff = _metric_neff(e2e_values)
        prior_performance = (
            prior.e2e_p95_ms / params.slo_e2e_ms
            if prior and prior.e2e_p95_ms is not None
            else None
        )

    if online_performance is not None and prior_performance is not None:
        online_weight = metric_neff / (metric_neff + params.prior_strength)
        performance = online_weight * online_performance + (1.0 - online_weight) * prior_performance
        prior_weight = 1.0 - online_weight
    elif online_performance is not None:
        performance = online_performance
        prior_weight = 0.0
    else:
        performance = prior_performance
        prior_weight = 1.0 if prior_performance is not None else 0.0

    if online_success is not None and prior and prior.success_rate is not None:
        online_weight = n_eff / (n_eff + params.prior_strength)
        success_rate = online_weight * online_success + (1.0 - online_weight) * prior.success_rate
    elif online_success is not None:
        success_rate = online_success
    elif prior:
        success_rate = prior.success_rate
    else:
        success_rate = None

    cost_raw = (
        params.rho_input_price * price.input_per_million
        + (1.0 - params.rho_input_price) * price.output_per_million
    )
    reasons = []
    if performance is None:
        reasons.append("缺少性能数据")
    elif prior_performance is None and metric_neff < params.min_samples:
        reasons.append(f"有效样本少于{params.min_samples}且无历史先验")
    if success_rate is None:
        reasons.append("缺少成功率")
    elif success_rate < params.min_success_rate:
        reasons.append(f"成功率低于{params.min_success_rate:.0%}")

    return Candidate(
        endpoint_id=price.endpoint_id,
        currency=price.currency,
        input_per_million=price.input_per_million,
        output_per_million=price.output_per_million,
        cost_raw=cost_raw,
        cost_normalized=None,
        performance=performance,
        online_performance=online_performance,
        prior_performance=prior_performance,
        p95_e2e_ms=p95_e2e,
        p95_ttft_ms=p95_ttft,
        p95_tpot_ms=p95_tpot,
        success_rate=success_rate,
        online_success_rate=online_success,
        sample_count=len(window),
        effective_sample_count=n_eff,
        window_minutes=window_minutes,
        prior_weight=prior_weight,
        feasible=not reasons,
        exclusion_reason="；".join(reasons),
        estimated_e2e_ms=estimates["e2e"],
        estimated_ttft_ms=estimates["ttft"],
        estimated_tpot_ms=estimates["tpot"],
        metric_prior_weights=prior_weights,
        metric_effective_samples=metric_neffs,
        unit_price_cost=cost_raw,
    )


def apply_request_cost(candidate: Candidate, input_tokens: int, output_tokens: int, mode: str, rho: float) -> None:
    """Keep estimated billing separate from an optional price-preference objective."""
    candidate.cost_mode = mode
    candidate.estimated_input_cost = input_tokens * candidate.input_per_million / 1_000_000
    candidate.estimated_output_cost = output_tokens * candidate.output_per_million / 1_000_000
    candidate.estimated_request_cost = candidate.estimated_input_cost + candidate.estimated_output_cost
    candidate.unit_price_cost = rho * candidate.input_per_million + (1 - rho) * candidate.output_per_million
    if mode == "predicted_request":
        candidate.cost_raw = candidate.estimated_request_cost
    elif mode == "weighted_predicted_request":
        candidate.cost_raw = rho * candidate.estimated_input_cost + (1 - rho) * candidate.estimated_output_cost
    elif mode == "unit_price":
        candidate.cost_raw = candidate.unit_price_cost
    else:
        raise ValueError("Invalid cost_mode")
    if not all(math.isfinite(v) for v in (candidate.estimated_input_cost, candidate.estimated_output_cost,
                                          candidate.estimated_request_cost, candidate.cost_raw)):
        raise ValueError("Predicted request cost must be finite")


def estimate_stability(candidate: Candidate, observations: Sequence[Observation], cutoff: datetime,
                       prior: Prior | None, params: StrategyParameters, z: float) -> None:
    """Conservative weighted-success ranking proxy, not a calibrated confidence guarantee."""
    weighted = [(item, 2 ** (-max(0.0, (cutoff - item.occurred_at).total_seconds() / 60) / params.half_life_minutes))
                for item in observations if item.occurred_at <= cutoff]
    neff = effective_sample_size([w for _, w in weighted])
    online = weighted_average([(float(o.success), w) for o, w in weighted])
    prior_rate = prior.success_rate if prior else None
    prior_mass = min(prior.sample_count, params.prior_strength) if prior and prior_rate is not None else 0.0
    if online is not None and prior_rate is not None and prior_mass > 0:
        rate = (neff * online + prior_mass * prior_rate) / (neff + prior_mass)
        source = "pair_recent_both_modes_plus_prior"
    elif online is not None:
        rate, prior_mass, source = online, 0.0, "pair_recent_both_modes"
    else:
        rate, source = prior_rate, "historical_prior_only" if prior_rate is not None else "unknown"
    n = neff + prior_mass
    if rate is None or n <= 0:
        lower = 0.0
    else:
        z2 = z * z
        lower = (rate + z2 / (2 * n) - z * math.sqrt(rate * (1 - rate) / n + z2 / (4 * n * n))) / (1 + z2 / n)
    candidate.stability_success_rate = rate
    candidate.stability_sample_count = n
    candidate.stability_lower_bound = max(0.0, min(1.0, lower))
    candidate.stability_429_rate = weighted_average([(float(o.http_status == 429), w) for o, w in weighted])
    candidate.stability_5xx_rate = weighted_average([(float(o.http_status is not None and 500 <= o.http_status <= 599), w) for o, w in weighted])
    candidate.stability_source = source


def backup_sort_key(candidate: Candidate) -> tuple:
    return (-candidate.stability_lower_bound, -(candidate.stability_success_rate or 0.0),
            -candidate.stability_sample_count,
            candidate.routing_score if candidate.routing_score is not None else math.inf, candidate.endpoint_id)


def build_route_order(candidates: Sequence[Candidate], selected: Candidate, top_k: int,
                      backup_pool: str) -> list[str]:
    """Primary optimizes cost/performance; fallbacks optimize reliability under the same hard constraints."""
    remaining = [c for c in candidates if c.feasible and c.endpoint_id != selected.endpoint_id
                 and (backup_pool == "all_feasible" or c.pareto)]
    ordered = [selected] + sorted(remaining, key=backup_sort_key)[:top_k - 1]
    for rank, candidate in enumerate(ordered, 1):
        candidate.route_rank = rank
        candidate.route_role = "primary" if rank == 1 else "backup"
    return [c.endpoint_id for c in ordered]


def pareto_mask(candidates: Sequence[Candidate]) -> set[str]:
    usable = [item for item in candidates if item.feasible and item.performance is not None]
    frontier = set()
    for candidate in usable:
        dominated = any(
            other.endpoint_id != candidate.endpoint_id
            and other.cost_raw <= candidate.cost_raw
            and other.performance <= candidate.performance
            and (other.cost_raw < candidate.cost_raw or other.performance < candidate.performance)
            for other in usable
        )
        if not dominated:
            frontier.add(candidate.endpoint_id)
    return frontier


def finalize_selection(candidates: list[Candidate], params: StrategyParameters) -> Candidate:
    for item in candidates:
        item.pareto = False
        item.score = None
        item.selected = False
        item.selection_probability = 0.0
        item.cost_normalized = None
        item.routing_score = None
        item.route_rank = None
        item.route_role = "none"
    feasible = [item for item in candidates if item.feasible and item.performance is not None]
    if not feasible:
        details = "; ".join(f"{item.endpoint_id}: {item.exclusion_reason}" for item in candidates)
        raise ValueError(f"没有可用候选 Endpoint。{details}")
    positive_costs = [item.cost_raw for item in feasible if item.cost_raw > 0]
    cost_reference = median(positive_costs) if positive_costs else 1.0
    for item in candidates:
        item.cost_normalized = item.cost_raw / cost_reference
    frontier = pareto_mask(candidates)
    for item in candidates:
        item.pareto = item.endpoint_id in frontier
        if item.feasible and item.performance is not None:
            item.routing_score = params.lambda_cost * item.cost_normalized + (1 - params.lambda_cost) * item.performance
        if item.pareto and item.performance is not None:
            item.score = (
                params.lambda_cost * (item.cost_normalized or 0.0)
                + (1.0 - params.lambda_cost) * item.performance
            )
    selectable = [item for item in candidates if item.pareto and item.score is not None]
    selected = min(
        selectable,
        key=lambda item: (
            item.score,
            -(item.success_rate or 0.0),
            -item.effective_sample_count,
            item.endpoint_id,
        ),
    )
    selected.selected = True
    if params.temperature <= 0:
        selected.selection_probability = 1.0
    else:
        minimum = min(item.score for item in selectable if item.score is not None)
        weights = {
            item.endpoint_id: math.exp(-(item.score - minimum) / params.temperature)
            for item in selectable
            if item.score is not None
        }
        total = sum(weights.values())
        for item in selectable:
            item.selection_probability = weights[item.endpoint_id] / total
    return selected


def build_candidates_indexed(
    model_id: str,
    prices: dict[str, Price],
    state_index: ModelEndpointStateIndex,
    priors: dict[str, Prior],
    cutoff: datetime,
    params: StrategyParameters,
) -> list[Candidate]:
    max_window = max(params.window_candidates_minutes)
    return [
        compute_candidate(
            prices[endpoint_id],
            state_index.recent(
                model_id,
                endpoint_id,
                params.stream,
                cutoff,
                max_window,
            ),
            cutoff,
            priors.get(endpoint_id),
            params,
        )
        for endpoint_id in sorted(prices)
    ]


class RoutingEngine:
    def __init__(self, state: InMemoryRoutingState) -> None:
        self.state = state

    def route_request(self, request, *, cutoff=None, currency="CNY", require_slo=True, **parameter_overrides):
        """Request-aware routing. Does not dispatch a request or reserve capacity."""
        from .request_routing import route_request
        return route_request(self.state, request, cutoff=cutoff, currency=currency,
                             require_slo=require_slo, **parameter_overrides)

    def start_failover(self, request, **options):
        """Create a guarded attempt planner; actual HTTP/SDK calls remain with the caller."""
        from .failover import FailoverSession
        return FailoverSession(self, request, **options)

    def route(
        self,
        model_id: str,
        stream: bool,
        cutoff: datetime,
        currency: str = "CNY",
        tier_override: int | None = None,
        **parameter_overrides,
    ) -> tuple[list[Candidate], Candidate, StrategyParameters]:
        policy = self.state.policies.get(model_id)
        params = policy.to_parameters(stream, **parameter_overrides)
        prices = self.state.catalog.prices_for(
            model_id,
            currency,
            at_time=cutoff,
            tier_override=tier_override,
        )
        if not prices:
            raise ValueError(f"模型 {model_id} 没有可用的 {currency} 候选Endpoint")
        candidates = build_candidates_indexed(
            model_id,
            prices,
            self.state.windows,
            self.state.catalog.priors_for(model_id),
            cutoff,
            params,
        )
        selected = finalize_selection(candidates, params)
        return candidates, selected, params
