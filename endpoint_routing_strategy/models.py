from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Price:
    endpoint_id: str
    model_id: str
    currency: str
    input_per_million: float
    output_per_million: float
    tier_index: int
    tier_conditions: dict

    def __post_init__(self) -> None:
        for value in (self.input_per_million, self.output_per_million):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Unit prices must be finite and nonnegative")


@dataclass(frozen=True)
class Prior:
    endpoint_id: str
    e2e_p95_ms: float | None
    ttft_p95_ms: float | None
    tpot_proxy_ms: float | None
    success_rate: float | None
    sample_count: float
    source: str

    def __post_init__(self) -> None:
        for value in (self.e2e_p95_ms, self.ttft_p95_ms, self.tpot_proxy_ms, self.sample_count):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Prior metrics/count must be finite and nonnegative")
        if self.success_rate is not None and (not math.isfinite(self.success_rate) or not 0 <= self.success_rate <= 1):
            raise ValueError("Prior success_rate must be in [0, 1]")


@dataclass(frozen=True)
class Observation:
    occurred_at: datetime
    endpoint_id: str
    is_stream: bool
    success: bool
    result: str
    http_status: int | None
    e2e_ms: float | None
    ttft_ms: float | None
    tpot_ms: float | None

    def __post_init__(self) -> None:
        for value in (self.e2e_ms, self.ttft_ms, self.tpot_ms):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Observed metrics must be finite and nonnegative")


@dataclass
class Candidate:
    endpoint_id: str
    currency: str
    input_per_million: float
    output_per_million: float
    cost_raw: float
    cost_normalized: float | None
    performance: float | None
    online_performance: float | None
    prior_performance: float | None
    p95_e2e_ms: float | None
    p95_ttft_ms: float | None
    p95_tpot_ms: float | None
    success_rate: float | None
    online_success_rate: float | None
    sample_count: int
    effective_sample_count: float
    window_minutes: int
    prior_weight: float
    feasible: bool
    exclusion_reason: str
    pareto: bool = False
    score: float | None = None
    selection_probability: float = 0.0
    selected: bool = False
    estimated_e2e_ms: float | None = None
    estimated_ttft_ms: float | None = None
    estimated_tpot_ms: float | None = None
    metric_prior_weights: dict[str, float] = field(default_factory=dict)
    metric_effective_samples: dict[str, float] = field(default_factory=dict)
    cost_mode: str = "unit_price"
    unit_price_cost: float | None = None
    estimated_input_cost: float | None = None
    estimated_output_cost: float | None = None
    estimated_request_cost: float | None = None
    routing_score: float | None = None
    stability_success_rate: float | None = None
    stability_sample_count: float = 0.0
    stability_lower_bound: float = 0.0
    stability_429_rate: float | None = None
    stability_5xx_rate: float | None = None
    stability_source: str = "unknown"
    route_rank: int | None = None
    route_role: str = "none"


@dataclass(frozen=True)
class StrategyParameters:
    stream: bool
    eta_ttft: float = 0.5
    rho_input_price: float = 0.5
    lambda_cost: float = 0.5
    slo_e2e_ms: float = 60_000.0
    slo_ttft_ms: float = 5_000.0
    slo_tpot_ms: float = 100.0
    target_samples: int = 100
    min_samples: int = 30
    window_candidates_minutes: tuple[int, ...] = (5, 15, 30, 60, 180, 360)
    half_life_minutes: float = 30.0
    prior_strength: float = 30.0
    min_success_rate: float = 0.8
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")
        for name in ("eta_ttft", "rho_input_price", "lambda_cost", "min_success_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("slo_e2e_ms", "slo_ttft_ms", "slo_tpot_ms", "half_life_minutes"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("prior_strength", "temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("target_samples", "min_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        windows = self.window_candidates_minutes
        if (not windows or any(isinstance(w, bool) or not isinstance(w, int) or w <= 0 for w in windows)
                or tuple(windows) != tuple(sorted(set(windows)))):
            raise ValueError("windows must be positive, unique and increasing integers")


@dataclass(frozen=True)
class ModelRoutingPolicy:
    model_id: str
    slo_e2e_ms: float = 60_000.0
    slo_ttft_ms: float = 5_000.0
    slo_tpot_ms: float = 100.0
    lambda_cost: float = 0.5
    eta_ttft: float = 0.5
    rho_input_price: float = 0.5
    cost_mode: str = "predicted_request"
    top_k: int = 3
    backup_pool: str = "all_feasible"
    stability_z: float = 1.96

    def __post_init__(self) -> None:
        if self.cost_mode not in {"predicted_request", "weighted_predicted_request", "unit_price"}:
            raise ValueError("Invalid cost_mode")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.backup_pool not in {"all_feasible", "pareto"}:
            raise ValueError("backup_pool must be all_feasible or pareto")
        if isinstance(self.stability_z, bool) or not isinstance(self.stability_z, (int, float)) or not math.isfinite(self.stability_z) or self.stability_z <= 0:
            raise ValueError("stability_z must be finite and positive")

    def to_parameters(self, stream: bool, **overrides) -> StrategyParameters:
        values = dict(
            stream=stream, slo_e2e_ms=self.slo_e2e_ms,
            slo_ttft_ms=self.slo_ttft_ms, slo_tpot_ms=self.slo_tpot_ms,
            lambda_cost=self.lambda_cost, eta_ttft=self.eta_ttft,
            rho_input_price=self.rho_input_price,
        )
        values.update(overrides)
        return StrategyParameters(**values)


@dataclass(frozen=True)
class EndpointOffering:
    endpoint_id: str
    model_id: str
    deployment_type: str
    price_config: dict
    capacity_config: dict


@dataclass(frozen=True)
class ModelEndpointRuntimeState:
    model_id: str
    endpoint_id: str
    price_tier_index: int = 0
    capacity_rpm: float | None = None
    capacity_tpm: float | None = None
    capacity_concurrency: float | None = None
    current_rpm: float = 0.0
    current_tpm: float = 0.0
    current_concurrency: float = 0.0
    historical_prior: Prior | None = None
    enabled: bool = True
    cooldown_until: datetime | None = None
    success_rate: float | None = None
    rate_limit_429_rate: float | None = None
    server_error_5xx_rate: float | None = None
    health_sample_count: int = 0
    updated_at: datetime | None = None


@dataclass(frozen=True)
class EndpointRuntimeState:
    endpoint_id: str
    current_concurrency: float = 0.0
    current_rpm: float = 0.0
    current_tpm: float = 0.0
    health_status: str = "unknown"
    supported_models: tuple[str, ...] = ()
    updated_at: datetime | None = None


class ModelEndpointRepository(Protocol):
    """Future database adapter contract. The current runtime is memory-only."""

    def load_all(self) -> Sequence[EndpointOffering]: ...

    def upsert(self, offering: EndpointOffering) -> None: ...

    def remove(self, model_id: str, endpoint_id: str) -> bool: ...
