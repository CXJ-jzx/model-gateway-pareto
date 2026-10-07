"""Build editable, isolated state snapshots; no endpoint execution simulator.

The optional prepared dataset supplies arrival metadata only. All endpoint state
and request-field changes used to exercise a rule are explicit fixture inputs.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

from ... import (
    EndpointOffering, InMemoryRoutingState, ModelRoutingPolicy, Observation,
    Prior, RequestSLO, RoutingRequest,
)


@dataclass
class ScenarioCase:
    case_id: str
    description: str
    state: InMemoryRoutingState
    request: RoutingRequest
    parameter_overrides: dict
    cutoff: datetime | None = None
    provenance: dict = field(default_factory=dict)


def _merge(base: dict, overrides: dict) -> dict:
    """Recursive dict merge; lists are replaced, not concatenated."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _finite(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _prepared_inputs(dataset_dir: Path | None, scenario_models: list[str]) -> tuple[dict, dict, list]:
    if dataset_dir is None:
        return {}, {}, []
    path = dataset_dir / "requests.jsonl"
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row.get("model_id"), row.get("stream_type"))
            if key not in result:
                # Deserialize to validate the arrival-only contract.
                result[key] = RoutingRequest.from_prepared(row)
    dual_mode = sorted({model for model, mode in result
                        if (model, "stream") in result and (model, "nonstream") in result})
    if not dual_mode:
        raise ValueError("Prepared arrival requests contain no model with both stream and nonstream samples")
    preferred = [model for model in scenario_models if model in dual_mode]
    available = preferred + [model for model in dual_mode if model not in preferred]
    # Keep preferred valid model identities before filling insufficient slots.
    mapping = {model: model for model in preferred}
    unused = [model for model in available if model not in preferred]
    for model in scenario_models:
        if model not in mapping and unused:
            mapping[model] = unused.pop(0)
    return result, mapping, [model for model in scenario_models if model not in dual_mode]


def _request(config: dict, case: dict, source: RoutingRequest | None) -> tuple[RoutingRequest, dict]:
    specification = _merge(config.get("request_defaults", {}), case.get("request", {}))
    if not isinstance(specification.get("model_id"), str) or not specification["model_id"]:
        raise ValueError("Fixture model_id must be a nonempty string")
    mode = specification.get("stream_type", "nonstream")
    if mode not in {"stream", "nonstream"}:
        raise ValueError("stream_type must be stream or nonstream")
    arrived = source.arrived_at if source else datetime.fromisoformat(config["synthetic_arrival_time"])
    priority = specification.get("priority", "normal")
    tier_by_priority = {"high": "strict", "normal": "standard", "low": "relaxed"}
    if priority not in tier_by_priority:
        raise ValueError("priority must be high, normal or low")
    slo_spec = dict(specification["slo"])
    tier = tier_by_priority[priority]
    if "tier" in slo_spec and slo_spec["tier"] != tier:
        raise ValueError("Fixture priority and SLO tier disagree")
    slo_spec["tier"] = tier
    if mode == "nonstream":
        slo_spec["ttft_ms"] = slo_spec["tpot_ms"] = None
    slo = RequestSLO(**slo_spec)
    input_tokens = specification.get("predicted_input_tokens", 100)
    output_tokens = specification.get("predicted_output_tokens", 50)
    for name, value in (("predicted_input_tokens", input_tokens), ("predicted_output_tokens", output_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    input_threshold = specification.get("input_heavy_threshold", 4096)
    output_threshold = specification.get("output_heavy_threshold", 2048)
    _finite(input_threshold, "input_heavy_threshold", positive=True)
    _finite(output_threshold, "output_heavy_threshold", positive=True)
    # Workload/urgency must be recomputed after modifying predictions/SLO.
    input_heavy, output_heavy = input_tokens >= input_threshold, output_tokens >= output_threshold
    workload_level = ("mixed_heavy" if input_heavy and output_heavy else "input_heavy"
                      if input_heavy else "output_heavy" if output_heavy else "light")
    workload = dict(level=workload_level,
                    input_heavy=input_heavy, output_heavy=output_heavy,
                    classification_heavy=None, task_type="unknown",
                    input_threshold=input_threshold, output_threshold=output_threshold,
                    source="explicit_fixture_token_thresholds")
    estimated_e2e = _finite(specification.get("estimated_service_ms", 400), "estimated_service_ms")
    estimated_ttft = _finite(specification.get("estimated_ttft_ms", 40), "estimated_ttft_ms")
    budgets = []
    if slo.e2e_ms is not None:
        budgets.append(("e2e", slo.e2e_ms, estimated_e2e))
    if mode == "stream":
        budgets.append(("ttft", slo.ttft_ms, estimated_ttft))
    constraint, budget, estimate = min(budgets, key=lambda item: (item[1] - item[2]) / item[1])
    fraction = (budget - estimate) / budget
    urgency = dict(level="critical" if fraction <= 0.1 else "elevated" if fraction <= 0.3 else "normal",
                   limiting_constraint=constraint, remaining_budget_ms=budget, slack_ms=budget - estimate,
                   slack_fraction=fraction, as_of=arrived.isoformat(), source="explicit_fixture_service_estimate")
    request = RoutingRequest(
        request_id=f"{source.request_id if source else 'synthetic'}__{case['case_id']}",
        model_id=specification["model_id"], arrived_at=arrived, is_stream=mode == "stream",
        predicted_input_tokens=input_tokens, predicted_output_tokens=output_tokens,
        slo=slo, priority=priority, workload=workload, urgency=urgency,
        input_provenance=dict(
            slo_source="explicit_scenario_fixture", slo_version=config["schema_version"],
            slo_profile_key=case.get("snapshot_template"),
            priority_source="explicit_scenario_fixture",
            input_prediction_source="explicit_scenario_fixture",
            output_prediction_source="explicit_scenario_fixture",
            source_prepared_profile=copy.deepcopy(source.input_provenance) if source else None,
        ),
    )
    provenance = dict(
        input_source="prepared_arrival_metadata" if source else "synthetic_arrival_metadata",
        source_request_id=source.request_id if source else None,
        modeled_fields=["predicted_input_tokens", "predicted_output_tokens", "slo", "priority", "workload", "urgency", "endpoint_state"],
        predictions_source="explicit_fixture_not_observed_outcome",
        cost_inputs_source="explicit_token_predictions_and_endpoint_unit_prices",
        stability_inputs_source="explicit_pair_recent_success_failure_batches_or_limited_prior",
        urgency_estimate_ms={"e2e": estimated_e2e, "ttft": estimated_ttft if request.is_stream else None},
        endpoint_execution_simulated=False,
    )
    return request, provenance


def _state(config: dict, case: dict, request: RoutingRequest, cutoff: datetime, model_mapping: dict) -> InMemoryRoutingState:
    templates = config.get("snapshot_templates", {})
    template_name = case.get("snapshot_template")
    if template_name is not None and template_name not in templates:
        raise ValueError(f"Unknown snapshot template {template_name}")
    snapshot = _merge(templates.get(template_name, {}), case.get("snapshot", {}))
    endpoints = snapshot.get("endpoints", [])
    if not isinstance(endpoints, list):
        raise ValueError("snapshot.endpoints must be a list")
    patches = case.get("endpoint_overrides", {})
    scenario_model = _merge(config.get("request_defaults", {}), case.get("request", {}))["model_id"]
    def mapped_model(logical_model: str) -> str:
        if logical_model == scenario_model:
            return request.model_id
        actual = model_mapping.get(logical_model, logical_model)
        # Preserve isolation even if a custom fixture names a foreign logical
        # model identical to the substituted real request-model identifier.
        return f"{actual}__fixture_foreign_{logical_model}" if actual == request.model_id else actual
    known_ids = {ep["endpoint_id"] for ep in endpoints}
    if patches.keys() - known_ids:
        raise ValueError(f"Unknown endpoint_overrides keys: {sorted(patches.keys() - known_ids)}")
    state = InMemoryRoutingState()
    state.policies.upsert(ModelRoutingPolicy(model_id=request.model_id))
    seen = set()
    for original in endpoints:
        endpoint = _merge(original, patches.get(original["endpoint_id"], {}))
        endpoint_id = endpoint["endpoint_id"]
        model_id = mapped_model(endpoint["model_id"]) if "model_id" in endpoint else request.model_id
        if (model_id, endpoint_id) in seen:
            raise ValueError(f"Duplicate offering {model_id}/{endpoint_id}")
        seen.add((model_id, endpoint_id))
        prices = endpoint.get("price_tiers", [dict(input_per_million=2, output_per_million=2)])
        price_tiers = []
        for tier in prices:
            tier = _merge(dict(currency="CNY", billing_unit="per_million_tokens", conditions={}), tier)
            for name in ("input_per_million", "output_per_million"):
                _finite(tier[name], name)
            price_tiers.append(tier)
        state.register_endpoint(EndpointOffering(
            endpoint_id=endpoint_id, model_id=model_id, deployment_type="fixed_snapshot",
            price_config=dict(pricing_status="configured", currency="CNY", price_tiers=price_tiers),
            capacity_config=_merge(dict(rpm=60, tpm=100_000, concurrency=10), endpoint.get("capacity", {})),
        ))
        load = _merge(dict(current_rpm=0, current_tpm=0, current_concurrency=0), endpoint.get("load", {}))
        state.catalog.update_model_endpoint_load(model_id, endpoint_id, **load)
        enabled = endpoint.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be bool")
        state.catalog.set_enabled(model_id, endpoint_id, enabled)
        tier_index = endpoint.get("price_tier_index", 0)
        if isinstance(tier_index, bool) or not isinstance(tier_index, int) or tier_index < 0:
            raise ValueError("price_tier_index must be a nonnegative integer")
        state.catalog.set_price_tier(model_id, endpoint_id, tier_index)
        state.catalog.update_endpoint_runtime(endpoint_id, health_status=endpoint.get("health_status", "healthy"))
        if endpoint.get("cooldown_remaining_seconds") is not None:
            duration = _finite(endpoint["cooldown_remaining_seconds"], "cooldown_remaining_seconds")
            state.catalog.set_cooldown(model_id, endpoint_id, cutoff + timedelta(seconds=duration))
        if endpoint.get("prior") is not None:
            prior = _merge(dict(e2e_p95_ms=None, ttft_p95_ms=None, tpot_proxy_ms=None,
                                success_rate=1, sample_count=100, source="explicit_fixture_historical_prior"), endpoint["prior"])
            _finite(prior["sample_count"], "prior sample_count", positive=True)
            state.catalog.set_prior(model_id, endpoint_id, Prior(endpoint_id=endpoint_id, **prior))
        for batch in endpoint.get("observations", []):
            age = _finite(batch.get("age_minutes", 1), "observation age_minutes")
            count = batch.get("count", 8)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("observation count must be a positive integer")
            batch_mode = batch.get("stream_type", "stream" if request.is_stream else "nonstream")
            if batch_mode not in {"stream", "nonstream"}:
                raise ValueError("observation stream_type must be stream or nonstream")
            success = batch.get("success", True)
            if not isinstance(success, bool):
                raise ValueError("observation success must be bool")
            http_status = batch.get("http_status", 200 if success else 500)
            if http_status is not None and (
                isinstance(http_status, bool) or not isinstance(http_status, int)
                or not 100 <= http_status <= 599
            ):
                raise ValueError("observation http_status must be a valid integer status or None")
            if success and http_status is not None and not 200 <= http_status <= 299:
                raise ValueError("A successful fixture observation cannot have an error HTTP status")
            for name in ("e2e_ms", "ttft_ms", "tpot_ms"):
                if batch.get(name) is not None:
                    _finite(batch[name], name, positive=True)
            for _ in range(count):
                observation_model = mapped_model(batch["model_id"]) if "model_id" in batch else model_id
                state.windows.record(observation_model, Observation(
                    occurred_at=cutoff - timedelta(minutes=age), endpoint_id=endpoint_id,
                    is_stream=batch_mode == "stream", success=success,
                    result="success" if success else "error", http_status=http_status,
                    e2e_ms=batch.get("e2e_ms"), ttft_ms=batch.get("ttft_ms"), tpot_ms=batch.get("tpot_ms"),
                ))
    return state


def load_cases(config_path: Path, dataset_dir: Path | None = None) -> list[ScenarioCase]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "1.0" or not isinstance(config.get("cases"), list):
        raise ValueError("Scenario config requires schema_version 1.0 and a cases list")
    scenario_models = []
    ids = set()
    for case in config["cases"]:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in ids or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in case_id):
            raise ValueError("case_id must be a unique safe lowercase identifier")
        ids.add(case_id)
        request = _merge(config.get("request_defaults", {}), case.get("request", {}))
        if request["model_id"] not in scenario_models:
            scenario_models.append(request["model_id"])
    sources, model_mapping, skipped_models = _prepared_inputs(Path(dataset_dir) if dataset_dir is not None else None, scenario_models)
    result = []
    for case in config["cases"]:
        specification = _merge(config.get("request_defaults", {}), case.get("request", {}))
        scenario_model = specification["model_id"]
        if dataset_dir is not None and scenario_model not in model_mapping:
            # Insufficient real-data groups are not silently synthesized.
            continue
        actual_model = model_mapping.get(scenario_model, scenario_model)
        key = (actual_model, specification.get("stream_type", "nonstream"))
        request, provenance = _request(config, case, sources.get(key))
        if actual_model != scenario_model:
            request = replace(request, model_id=actual_model)
        delay = _finite(case.get("routing_delay_ms", 0), "routing_delay_ms")
        cutoff = request.arrived_at + timedelta(milliseconds=delay)
        overrides = _merge(config.get("parameter_defaults", {}), case.get("parameters", {}))
        if "window_candidates_minutes" in overrides:
            overrides["window_candidates_minutes"] = tuple(overrides["window_candidates_minutes"])
        state = _state(config, case, request, cutoff, model_mapping)
        provenance["fixture_config"] = str(Path(config_path).resolve())
        provenance["prepared_dataset"] = str(Path(dataset_dir).resolve()) if dataset_dir is not None else None
        provenance.update(scenario_model_id=scenario_model, actual_model_id=actual_model,
                          model_substitution=actual_model != scenario_model, skipped_models=skipped_models,
                          skipped_model_groups=[f"{model}/both" for model in skipped_models],
                          skipped_case_ids=[item["case_id"] for item in config["cases"]
                                            if dataset_dir is not None and _merge(config.get("request_defaults", {}), item.get("request", {}))["model_id"] not in model_mapping],
                          model_mapping=model_mapping, distinct_dataset_models=len(set(model_mapping.values())) if model_mapping else 0,
                          endpoint_state_source="explicit_fixture_not_real_endpoint_configuration")
        provenance["snapshot_template"] = case.get("snapshot_template")
        result.append(ScenarioCase(case["case_id"], case.get("description", ""), state, request, overrides, cutoff, provenance))
    return result
