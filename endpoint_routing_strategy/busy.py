"""Cheap capacity-pressure assessment, not admission, scheduling or SLO checking."""
from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Protocol, Sequence

try:
    from .models import ModelEndpointRuntimeState
except ImportError:  # Support the retained direct-script entry points.
    from models import ModelEndpointRuntimeState


@dataclass(frozen=True)
class BusyThresholds:
    """Experimental tunable thresholds; neither value is a provider limit."""

    enter: float = 0.8
    exit: float = 0.6

    def __post_init__(self) -> None:
        for value in (self.enter, self.exit):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Busy thresholds must be finite numbers")
        if not 0 <= self.exit < self.enter <= 1:
            raise ValueError("Busy thresholds must satisfy 0 <= exit < enter <= 1")


class CapacitySnapshotProvider(Protocol):
    def capacity_snapshot(self, model_id: str) -> tuple[
        int, tuple[ModelEndpointRuntimeState, ...], dict[str, str]
    ]: ...


@dataclass(frozen=True)
class EndpointBusyAssessment:
    endpoint_id: str
    status: str
    pressure: float | None
    utilization: dict[str, float | None]
    unknown_dimensions: tuple[str, ...]
    saturated_dimensions: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BusyAssessment:
    model_id: str
    assessed_at: datetime
    status: str
    reason: str
    catalog_version: int
    thresholds: BusyThresholds
    scope: str
    endpoints: tuple[EndpointBusyAssessment, ...]

    @property
    def is_busy(self) -> bool | None:
        return {"busy": True, "idle": False}.get(self.status)

    def endpoint_ids(self, status: str) -> tuple[str, ...]:
        return tuple(item.endpoint_id for item in self.endpoints if item.status == status)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["assessed_at"] = self.assessed_at.isoformat()
        result["is_busy"] = self.is_busy
        for status in ("busy", "idle", "unknown", "unavailable"):
            result[f"{status}_endpoint_ids"] = self.endpoint_ids(status)
        eligible = sum(item.status != "unavailable" for item in self.endpoints)
        result["busy_fraction"] = len(self.endpoint_ids("busy")) / eligible if eligible else None
        result["capacity_only"] = True
        result["capacity_reserved"] = False
        return result


class BusyDetector:
    """One detector per state; stores hysteresis bits, never duplicates load counters.

    A model is busy only if every non-excluded endpoint in the requested scope is
    busy. A known idle alternative prevents a model-wide busy verdict. Missing
    limits do not establish idleness; a known overloaded dimension establishes
    pressure even when other limits are unknown.
    """

    def __init__(self, catalog: CapacitySnapshotProvider, thresholds: BusyThresholds | None = None) -> None:
        if thresholds is not None and not isinstance(thresholds, BusyThresholds):
            raise ValueError("thresholds must be BusyThresholds")
        self.catalog = catalog
        self.thresholds = thresholds or BusyThresholds()
        self._busy_by_model: dict[str, dict[str, bool]] = {}
        self._last_assessed_at: dict[str, datetime] = {}
        self._lock = threading.RLock()

    def forget_endpoint(self, model_id: str, endpoint_id: str) -> None:
        with self._lock:
            states = self._busy_by_model.get(model_id)
            if states is not None:
                states.pop(endpoint_id, None)
                if not states:
                    self._busy_by_model.pop(model_id, None)
                    self._last_assessed_at.pop(model_id, None)

    def assess(
        self, model_id: str, *, at_time: datetime | None = None,
        endpoint_ids: Sequence[str] | None = None,
    ) -> BusyAssessment:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a nonempty string")
        now = at_time if at_time is not None else datetime.now().astimezone()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("at_time must be a timezone-aware datetime")
        selected = None
        if endpoint_ids is not None:
            if isinstance(endpoint_ids, (str, bytes)):
                raise ValueError("endpoint_ids must be a collection of nonempty strings")
            selected = set(endpoint_ids)
            if any(not isinstance(value, str) or not value for value in selected):
                raise ValueError("endpoint_ids must be a collection of nonempty strings")
        with self._lock:
            previous_time = self._last_assessed_at.get(model_id)
            if previous_time is not None and now < previous_time:
                raise ValueError("Assessments must be chronological per model; use a fresh state for a new replay")
            version, runtimes, health = self.catalog.capacity_snapshot(model_id)
            registered = {item.endpoint_id for item in runtimes}
            if selected is not None and selected - registered:
                raise ValueError(f"Endpoints not registered for {model_id}: {sorted(selected - registered)}")
            remembered = self._busy_by_model.get(model_id, {})
            remembered = {ep: value for ep, value in remembered.items() if ep in registered}
            results = []
            for runtime in runtimes:
                if selected is not None and runtime.endpoint_id not in selected:
                    continue
                result, retain_busy = self._assess_endpoint(
                    runtime, health[runtime.endpoint_id], now, remembered.get(runtime.endpoint_id, False)
                )
                remembered[runtime.endpoint_id] = retain_busy
                results.append(result)
            if registered:
                self._busy_by_model[model_id] = remembered
                self._last_assessed_at[model_id] = now
            else:
                self._busy_by_model.pop(model_id, None)
                self._last_assessed_at.pop(model_id, None)
            eligible = [item for item in results if item.status != "unavailable"]
            if not eligible:
                status, reason = "unavailable", "no_capacity_candidates"
            elif any(item.status == "idle" for item in eligible):
                status, reason = "idle", "idle_alternative_exists"
            elif any(item.status == "unknown" for item in eligible):
                status, reason = "unknown", "insufficient_capacity_information"
            else:
                status, reason = "busy", "all_capacity_candidates_busy"
            return BusyAssessment(
                model_id, now, status, reason, version, self.thresholds,
                "model" if selected is None else "endpoint_subset", tuple(results),
            )

    def _assess_endpoint(
        self, runtime: ModelEndpointRuntimeState, health: str, now: datetime, was_busy: bool,
    ) -> tuple[EndpointBusyAssessment, bool]:
        utilization, unknown, saturated, exclusions = {}, [], [], []
        for name in ("rpm", "tpm", "concurrency"):
            current, limit = getattr(runtime, f"current_{name}"), getattr(runtime, f"capacity_{name}")
            if limit is None:
                utilization[name] = None
                unknown.append(name)
            elif limit == 0:
                utilization[name] = None
                exclusions.append(f"zero_capacity_{name}")
            else:
                utilization[name] = current / limit
                if current >= limit:
                    saturated.append(name)
        known = [value for value in utilization.values() if value is not None]
        pressure = max(known) if known else None
        if not runtime.enabled:
            exclusions.append("disabled")
        if runtime.cooldown_until is not None and now < runtime.cooldown_until:
            exclusions.append("cooldown")
        if health.lower() in {"unhealthy", "down", "disabled"}:
            exclusions.append("endpoint_unhealthy")
        retained = False
        if exclusions:
            status, reasons = "unavailable", tuple(exclusions)
        elif pressure is not None and pressure >= self.thresholds.enter:
            status, retained = "busy", True
            reasons = tuple(f"high_utilization_{name}" for name, value in utilization.items()
                            if value is not None and value >= self.thresholds.enter)
        elif was_busy and pressure is not None and pressure > self.thresholds.exit:
            status, retained, reasons = "busy", True, ("hysteresis_hold",)
        elif unknown:
            status, retained, reasons = "unknown", was_busy, ("capacity_unknown",)
        else:
            status, reasons = "idle", ("below_enter_threshold" if not was_busy else "recovered_below_exit",)
        return EndpointBusyAssessment(
            runtime.endpoint_id, status, pressure, utilization, tuple(unknown), tuple(saturated), reasons,
        ), retained
