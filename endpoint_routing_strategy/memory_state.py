from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Sequence

try:
    from .busy import BusyAssessment, BusyDetector, BusyThresholds
    from .models import (
        EndpointOffering,
        EndpointRuntimeState,
        ModelEndpointRepository,
        ModelEndpointRuntimeState,
        ModelRoutingPolicy,
        Observation,
        Price,
        Prior,
    )
except ImportError:  # 兼容直接运行目录内脚本
    from busy import BusyAssessment, BusyDetector, BusyThresholds
    from models import (
        EndpointOffering,
        EndpointRuntimeState,
        ModelEndpointRepository,
        ModelEndpointRuntimeState,
        ModelRoutingPolicy,
        Observation,
        Price,
        Prior,
    )


class ModelPolicyRegistry:
    """model_id -> SLO and preference parameters."""

    def __init__(self) -> None:
        self._policies: dict[str, ModelRoutingPolicy] = {}
        self._lock = threading.RLock()

    def upsert(self, policy: ModelRoutingPolicy) -> None:
        policy.to_parameters(False)  # Validate all numeric fields, including NaN/Inf.
        for name in ("lambda_cost", "eta_ttft", "rho_input_price"):
            if not 0 <= getattr(policy, name) <= 1:
                raise ValueError(f"{name} 必须在0到1之间")
        if min(policy.slo_e2e_ms, policy.slo_ttft_ms, policy.slo_tpot_ms) <= 0:
            raise ValueError("SLO必须大于0")
        with self._lock:
            self._policies[policy.model_id] = policy

    def get(self, model_id: str) -> ModelRoutingPolicy:
        with self._lock:
            policy = self._policies.get(model_id)
            if policy is None:
                raise KeyError(f"模型 {model_id} 尚未配置路由策略")
            return policy

    def remove(self, model_id: str) -> bool:
        with self._lock:
            return self._policies.pop(model_id, None) is not None


class ModelEndpointRegistry:
    """model -> endpoints plus model-endpoint and endpoint runtime state."""

    def __init__(self) -> None:
        self._offerings: dict[str, dict[str, EndpointOffering]] = {}
        self._runtime: dict[tuple[str, str], ModelEndpointRuntimeState] = {}
        self._models_by_endpoint: dict[str, set[str]] = {}
        self._endpoint_runtime: dict[str, EndpointRuntimeState] = {}
        self._lock = threading.RLock()
        self._version = 0

    @classmethod
    def from_offerings(cls, offerings: Sequence[EndpointOffering]) -> "ModelEndpointRegistry":
        registry = cls()
        for offering in offerings:
            registry.upsert(offering)
        return registry

    @classmethod
    def from_repository(cls, repository: ModelEndpointRepository) -> "ModelEndpointRegistry":
        return cls.from_offerings(repository.load_all())

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def upsert(self, offering: EndpointOffering) -> None:
        now = datetime.now().astimezone()
        key = (offering.model_id, offering.endpoint_id)
        capacity = offering.capacity_config
        for name in ("rpm", "tpm", "concurrency"):
            value = capacity.get(name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"capacity {name} must be finite and nonnegative, or None")
        with self._lock:
            self._offerings.setdefault(offering.model_id, {})[offering.endpoint_id] = offering
            self._models_by_endpoint.setdefault(offering.endpoint_id, set()).add(offering.model_id)
            old = self._runtime.get(key)
            if old is None:
                self._runtime[key] = ModelEndpointRuntimeState(
                    model_id=offering.model_id,
                    endpoint_id=offering.endpoint_id,
                    capacity_rpm=capacity.get("rpm"),
                    capacity_tpm=capacity.get("tpm"),
                    capacity_concurrency=capacity.get("concurrency"),
                    updated_at=now,
                )
            else:
                self._runtime[key] = replace(
                    old,
                    capacity_rpm=capacity.get("rpm"),
                    capacity_tpm=capacity.get("tpm"),
                    capacity_concurrency=capacity.get("concurrency"),
                    updated_at=now,
                )
            self._refresh_supported_models(offering.endpoint_id, now)
            self._version += 1

    def remove(self, model_id: str, endpoint_id: str) -> bool:
        with self._lock:
            endpoints = self._offerings.get(model_id)
            if not endpoints or endpoint_id not in endpoints:
                return False
            del endpoints[endpoint_id]
            if not endpoints:
                del self._offerings[model_id]
            self._runtime.pop((model_id, endpoint_id), None)
            models = self._models_by_endpoint.get(endpoint_id)
            if models:
                models.discard(model_id)
                if not models:
                    del self._models_by_endpoint[endpoint_id]
            self._refresh_supported_models(endpoint_id, datetime.now().astimezone())
            self._version += 1
            return True

    def candidates(
        self,
        model_id: str,
        at_time: datetime | None = None,
        include_ineligible: bool = False,
    ) -> tuple[EndpointOffering, ...]:
        reference_time = at_time or datetime.now().astimezone()
        with self._lock:
            result = []
            for endpoint_id, offering in self._offerings.get(model_id, {}).items():
                runtime = self._runtime[(model_id, endpoint_id)]
                cooling = (
                    runtime.cooldown_until is not None
                    and reference_time < runtime.cooldown_until
                )
                endpoint_runtime = self._endpoint_runtime.get(
                    endpoint_id,
                    EndpointRuntimeState(endpoint_id),
                )
                unhealthy = endpoint_runtime.health_status.lower() in {
                    "unhealthy", "down", "disabled",
                }
                at_capacity = any(
                    limit is not None and current >= limit
                    for current, limit in (
                        (runtime.current_concurrency, runtime.capacity_concurrency),
                        (runtime.current_rpm, runtime.capacity_rpm),
                        (runtime.current_tpm, runtime.capacity_tpm),
                    )
                )
                if include_ineligible or (runtime.enabled and not cooling and not unhealthy and not at_capacity):
                    result.append(offering)
            return tuple(result)

    def models_for_endpoint(self, endpoint_id: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._models_by_endpoint.get(endpoint_id, set())))

    def capacity_snapshot(self, model_id: str) -> tuple[
        int, tuple[ModelEndpointRuntimeState, ...], dict[str, str]
    ]:
        """Read this model's immutable pair states and global health under one lock.

        Include full/disabled pairs: otherwise pressure assessment would silently
        lose the very endpoints whose load it needs to inspect.
        """
        with self._lock:
            ids = self._offerings.get(model_id, {})
            runtimes = tuple(self._runtime[(model_id, ep)] for ep in ids)
            health = {ep: self._endpoint_runtime[ep].health_status for ep in ids}
            return self._version, runtimes, health

    def set_enabled(self, model_id: str, endpoint_id: str, enabled: bool) -> None:
        self._update_runtime(model_id, endpoint_id, enabled=enabled)

    def set_price_tier(self, model_id: str, endpoint_id: str, tier_index: int) -> None:
        if tier_index < 0:
            raise ValueError("price tier index 不能小于0")
        self._update_runtime(model_id, endpoint_id, price_tier_index=tier_index)

    def set_prior(self, model_id: str, endpoint_id: str, prior: Prior | None) -> None:
        self._update_runtime(model_id, endpoint_id, historical_prior=prior)

    def set_cooldown(
        self,
        model_id: str,
        endpoint_id: str,
        cooldown_until: datetime | None,
    ) -> None:
        self._update_runtime(model_id, endpoint_id, cooldown_until=cooldown_until)

    def update_quality(
        self,
        model_id: str,
        endpoint_id: str,
        success_rate: float | None,
        rate_limit_429_rate: float | None,
        server_error_5xx_rate: float | None,
        sample_count: int,
    ) -> None:
        for name, value in (
            ("success_rate", success_rate),
            ("rate_limit_429_rate", rate_limit_429_rate),
            ("server_error_5xx_rate", server_error_5xx_rate),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在0到1之间")
        if sample_count < 0:
            raise ValueError("sample_count不能小于0")
        self._update_runtime(
            model_id,
            endpoint_id,
            success_rate=success_rate,
            rate_limit_429_rate=rate_limit_429_rate,
            server_error_5xx_rate=server_error_5xx_rate,
            health_sample_count=sample_count,
        )

    def update_model_endpoint_load(
        self,
        model_id: str,
        endpoint_id: str,
        *,
        current_concurrency: float | None = None,
        current_rpm: float | None = None,
        current_tpm: float | None = None,
    ) -> None:
        """更新某个模型在某个Endpoint上的实时用量。"""
        for name, value in (
            ("current_concurrency", current_concurrency),
            ("current_rpm", current_rpm),
            ("current_tpm", current_tpm),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name}不能小于0")
        with self._lock:
            key = (model_id, endpoint_id)
            old = self._runtime.get(key)
            if old is None:
                raise KeyError(f"不存在组合 {model_id} × {endpoint_id}")
            self._runtime[key] = replace(
                old,
                current_concurrency=(
                    old.current_concurrency
                    if current_concurrency is None
                    else current_concurrency
                ),
                current_rpm=old.current_rpm if current_rpm is None else current_rpm,
                current_tpm=old.current_tpm if current_tpm is None else current_tpm,
                updated_at=datetime.now().astimezone(),
            )
            self._version += 1

    def adjust_model_endpoint_concurrency(
        self,
        model_id: str,
        endpoint_id: str,
        delta: float,
    ) -> float:
        """请求开始时传+1，结束时传-1；更新与读取在同一把锁内完成。"""
        if not math.isfinite(delta):
            raise ValueError("concurrency delta must be finite")
        with self._lock:
            key = (model_id, endpoint_id)
            old = self._runtime.get(key)
            if old is None:
                raise KeyError(f"不存在组合 {model_id} × {endpoint_id}")
            value = old.current_concurrency + delta
            if value < 0:
                raise ValueError("current_concurrency不能小于0")
            self._runtime[key] = replace(
                old,
                current_concurrency=value,
                updated_at=datetime.now().astimezone(),
            )
            self._version += 1
            return value

    def runtime_state(self, model_id: str, endpoint_id: str) -> ModelEndpointRuntimeState:
        with self._lock:
            state = self._runtime.get((model_id, endpoint_id))
            if state is None:
                raise KeyError(f"不存在组合 {model_id} × {endpoint_id}")
            return state

    def priors_for(self, model_id: str) -> dict[str, Prior]:
        with self._lock:
            result = {}
            for endpoint_id in self._offerings.get(model_id, {}):
                prior = self._runtime[(model_id, endpoint_id)].historical_prior
                if prior is not None:
                    result[endpoint_id] = prior
            return result

    def prices_for(
        self,
        model_id: str,
        currency: str,
        at_time: datetime | None = None,
        tier_override: int | None = None,
        include_ineligible: bool = False,
    ) -> dict[str, Price]:
        if tier_override is not None and (isinstance(tier_override, bool) or not isinstance(tier_override, int) or tier_override < 0):
            raise ValueError("tier_override must be a nonnegative integer")
        prices: dict[str, Price] = {}
        for offering in self.candidates(model_id, at_time=at_time, include_ineligible=include_ineligible):
            runtime = self.runtime_state(model_id, offering.endpoint_id)
            tier_index = runtime.price_tier_index if tier_override is None else tier_override
            config = offering.price_config
            tiers = config.get("price_tiers", [])
            if config.get("pricing_status") != "configured" or tier_index >= len(tiers):
                continue
            tier = tiers[tier_index]
            if tier.get("billing_unit") != "per_million_tokens":
                continue
            tier_currency = tier.get("currency") or config.get("currency")
            if tier_currency != currency:
                continue
            input_price = tier.get("input_per_million")
            output_price = tier.get("output_per_million")
            if input_price is None or output_price is None:
                continue
            if any(not math.isfinite(float(value)) or float(value) < 0 for value in (input_price, output_price)):
                raise ValueError(f"Invalid unit price for {model_id}/{offering.endpoint_id}")
            prices[offering.endpoint_id] = Price(
                endpoint_id=offering.endpoint_id,
                model_id=model_id,
                currency=tier_currency,
                input_per_million=float(input_price),
                output_per_million=float(output_price),
                tier_index=tier_index,
                tier_conditions=dict(tier.get("conditions", {})),
            )
        return prices

    def update_endpoint_runtime(
        self,
        endpoint_id: str,
        *,
        current_concurrency: float | None = None,
        current_rpm: float | None = None,
        current_tpm: float | None = None,
        health_status: str | None = None,
    ) -> None:
        """保留Endpoint聚合观测；其中负载字段不参与模型容量判断。"""
        for name, value in (
            ("current_concurrency", current_concurrency),
            ("current_rpm", current_rpm),
            ("current_tpm", current_tpm),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name}不能小于0")
        with self._lock:
            old = self._endpoint_runtime.get(endpoint_id, EndpointRuntimeState(endpoint_id))
            self._endpoint_runtime[endpoint_id] = replace(
                old,
                current_concurrency=old.current_concurrency if current_concurrency is None else current_concurrency,
                current_rpm=old.current_rpm if current_rpm is None else current_rpm,
                current_tpm=old.current_tpm if current_tpm is None else current_tpm,
                health_status=old.health_status if health_status is None else health_status,
                supported_models=tuple(sorted(self._models_by_endpoint.get(endpoint_id, set()))),
                updated_at=datetime.now().astimezone(),
            )
            self._version += 1

    def endpoint_state(self, endpoint_id: str) -> EndpointRuntimeState:
        with self._lock:
            return self._endpoint_runtime.get(
                endpoint_id,
                EndpointRuntimeState(
                    endpoint_id=endpoint_id,
                    supported_models=tuple(sorted(self._models_by_endpoint.get(endpoint_id, set()))),
                ),
            )

    def _update_runtime(self, model_id: str, endpoint_id: str, **changes) -> None:
        with self._lock:
            key = (model_id, endpoint_id)
            old = self._runtime.get(key)
            if old is None:
                raise KeyError(f"不存在组合 {model_id} × {endpoint_id}")
            self._runtime[key] = replace(
                old,
                updated_at=datetime.now().astimezone(),
                **changes,
            )
            self._version += 1

    def _refresh_supported_models(self, endpoint_id: str, now: datetime) -> None:
        models = tuple(sorted(self._models_by_endpoint.get(endpoint_id, set())))
        old = self._endpoint_runtime.get(endpoint_id, EndpointRuntimeState(endpoint_id))
        self._endpoint_runtime[endpoint_id] = replace(
            old,
            supported_models=models,
            updated_at=now,
        )


class ModelEndpointStateIndex:
    """(model, endpoint, stream_type) -> bounded recent observation window."""

    def __init__(self, max_age_minutes: int = 360, max_samples_per_key: int = 20_000) -> None:
        self.max_age = timedelta(minutes=max_age_minutes)
        self.max_samples_per_key = max_samples_per_key
        self._observations: dict[tuple[str, str, bool], deque[Observation]] = defaultdict(deque)
        self._lock = threading.RLock()

    def record(self, model_id: str, observation: Observation) -> None:
        key = (model_id, observation.endpoint_id, observation.is_stream)
        with self._lock:
            bucket = self._observations[key]
            if bucket and observation.occurred_at < bucket[-1].occurred_at:
                ordered = sorted((*bucket, observation), key=lambda item: item.occurred_at)
                bucket.clear()
                bucket.extend(ordered)
            else:
                bucket.append(observation)
            oldest_allowed = bucket[-1].occurred_at - self.max_age
            while bucket and (
                bucket[0].occurred_at < oldest_allowed
                or len(bucket) > self.max_samples_per_key
            ):
                bucket.popleft()

    def bulk_record(self, model_id: str, observations: Sequence[Observation]) -> None:
        for observation in observations:
            self.record(model_id, observation)

    def recent(
        self,
        model_id: str,
        endpoint_id: str,
        is_stream: bool,
        cutoff: datetime,
        max_window_minutes: int,
    ) -> list[Observation]:
        start = cutoff - timedelta(minutes=max_window_minutes)
        with self._lock:
            bucket = self._observations.get((model_id, endpoint_id, is_stream), ())
            return [item for item in bucket if start <= item.occurred_at <= cutoff]

    def health_summary(
        self,
        model_id: str,
        endpoint_id: str,
        cutoff: datetime,
        window_minutes: int = 60,
    ) -> tuple[float | None, float | None, float | None, int]:
        observations = []
        for mode in (False, True):
            observations.extend(self.recent(model_id, endpoint_id, mode, cutoff, window_minutes))
        count = len(observations)
        if not count:
            return None, None, None, 0
        success_rate = sum(item.success for item in observations) / count
        rate_429 = sum(item.http_status == 429 for item in observations) / count
        rate_5xx = sum(item.http_status is not None and 500 <= item.http_status <= 599 for item in observations) / count
        return success_rate, rate_429, rate_5xx, count

    def remove_endpoint(self, model_id: str, endpoint_id: str) -> int:
        removed = 0
        with self._lock:
            for mode in (False, True):
                bucket = self._observations.pop((model_id, endpoint_id, mode), None)
                if bucket is not None:
                    removed += len(bucket)
        return removed

    def key_count(self) -> int:
        with self._lock:
            return len(self._observations)

    def observation_count(self) -> int:
        with self._lock:
            return sum(len(bucket) for bucket in self._observations.values())


class InMemoryRoutingState:
    """Single memory-only entry point used by the gateway."""

    def __init__(self, max_window_minutes: int = 360, *, busy_thresholds: BusyThresholds | None = None) -> None:
        self.policies = ModelPolicyRegistry()
        self.catalog = ModelEndpointRegistry()
        self.windows = ModelEndpointStateIndex(max_age_minutes=max_window_minutes)
        self.busy = BusyDetector(self.catalog, busy_thresholds)

    def assess_busy(
        self, model_id: str, *, at_time: datetime | None = None, endpoint_ids: Sequence[str] | None = None,
    ) -> BusyAssessment:
        """Capacity pressure only; does not route, reserve, dispatch or check SLO."""
        return self.busy.assess(model_id, at_time=at_time, endpoint_ids=endpoint_ids)

    def register_endpoint(self, offering: EndpointOffering) -> None:
        self.catalog.upsert(offering)

    def remove_endpoint(self, model_id: str, endpoint_id: str) -> bool:
        removed = self.catalog.remove(model_id, endpoint_id)
        if removed:
            self.windows.remove_endpoint(model_id, endpoint_id)
            self.busy.forget_endpoint(model_id, endpoint_id)
        return removed

    def record_observation(
        self,
        model_id: str,
        observation: Observation,
        health_window_minutes: int = 60,
    ) -> None:
        self.windows.record(model_id, observation)
        quality = self.windows.health_summary(
            model_id,
            observation.endpoint_id,
            observation.occurred_at,
            health_window_minutes,
        )
        self.catalog.update_quality(model_id, observation.endpoint_id, *quality)

    def bulk_record(self, model_id: str, observations: Sequence[Observation]) -> None:
        self.windows.bulk_record(model_id, observations)
        endpoints = {item.endpoint_id for item in observations}
        if not observations:
            return
        cutoff = max(item.occurred_at for item in observations)
        registered = {
            item.endpoint_id
            for item in self.catalog.candidates(model_id, include_ineligible=True)
        }
        for endpoint_id in endpoints:
            if endpoint_id not in registered:
                continue
            self.catalog.update_quality(
                model_id,
                endpoint_id,
                *self.windows.health_summary(model_id, endpoint_id, cutoff),
            )
