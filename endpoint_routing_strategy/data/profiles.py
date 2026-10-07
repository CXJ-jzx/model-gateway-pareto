"""Frozen training-only statistics and configurable request policy enrichment."""

import hashlib
import math
import statistics
from collections import defaultdict
from datetime import timedelta

from .schema import IncomingRequest


def quantile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


def validate_config(config):
    def positive(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0

    if config.get("schema_version") != "1.0":
        raise ValueError("Unsupported schema_version")
    if not isinstance(config.get("seed"), int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    train, calibration = config["train_fraction"], config["calibration_fraction"]
    if not positive(train) or not positive(calibration) or train + calibration >= 1:
        raise ValueError("Split fractions must be positive and sum to less than 1")
    minimum = config["minimum_group_samples"]
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("minimum_group_samples must be a positive integer")
    probabilities = config["priority_distribution"]
    if set(probabilities) != {"high", "normal", "low"} or not all(positive(v) for v in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1):
        raise ValueError("Exactly three priorities with probabilities summing to 1 are required")
    if config["priority_to_slo"] != {"high": "strict", "normal": "standard", "low": "relaxed"}:
        raise ValueError("Priority/SLO mapping must be high→strict, normal→standard, low→relaxed")
    tiers = config["slo_tiers"]
    if set(tiers) != {"strict", "standard", "relaxed"}:
        raise ValueError("Exactly three SLO tiers are required")
    multipliers, queues = [], []
    for name in ("strict", "standard", "relaxed"):
        multiplier, queue = tiers[name]["latency_multiplier"], tiers[name]["queue_fraction"]
        if not positive(multiplier) or multiplier <= 1 or not positive(queue) or queue >= 1 - 1 / multiplier:
            raise ValueError("SLO multiplier must exceed 1 and queue must leave enough service budget")
        multipliers.append(multiplier)
        queues.append(queue)
    if multipliers != sorted(set(multipliers)) or queues != sorted(set(queues)):
        raise ValueError("SLO and queue budgets must increase strictly from strict to relaxed")
    if not positive(config["heavy_quantile"]) or config["heavy_quantile"] >= 1:
        raise ValueError("heavy_quantile must be between 0 and 1")
    for key in ("heavy_min_input_tokens", "heavy_min_output_tokens"):
        if not positive(config[key]):
            raise ValueError(f"{key} must be positive")
    for key in ("input_tokens", "output_tokens", "ttft_ms", "tpot_ms", "e2e_ms"):
        if not positive(config["fallback"][key]):
            raise ValueError(f"fallback.{key} must be positive")
    critical = config["urgency"]["critical_slack_fraction"]
    elevated = config["urgency"]["elevated_slack_fraction"]
    if not positive(critical) or not positive(elevated) or not critical < elevated < 1:
        raise ValueError("urgency thresholds must satisfy 0 < critical < elevated < 1")


def group_key(request):
    return f"{request.model_id}|{'stream' if request.is_stream else 'nonstream'}"


def fit_profiles(training, cutoff, config):
    """Only outcomes actually completed strictly before the frozen cutoff are used."""
    groups = defaultdict(lambda: defaultdict(list))
    latest_outcome = None
    for request in training:
        completed = [a for a in request.attempts if a.finished_at and a.finished_at < cutoff]
        if not completed:
            continue
        for a in completed:
            latest_outcome = max(latest_outcome, a.finished_at) if latest_outcome else a.finished_at
            for key, value in (("e2e", a.e2e_ms), ("ttft", a.ttft_ms), ("tpot", a.tpot_ms)):
                if value is not None and value > 0:
                    groups[group_key(request)][key].append(value)
                    groups["__global__"][key].append(value)
        # Request-level actual tokens become available only after its final attempt.
        if request.attempts[-1].finished_at is None or request.attempts[-1].finished_at >= cutoff:
            continue
        for key, value in (("input", request.actual_input_tokens), ("output", request.actual_output_tokens)):
            if value is not None:
                groups[group_key(request)][key].append(value)
                groups["__global__"][key].append(value)
    result = {}
    for key in sorted({group_key(r) for r in training} | {"__global__"}):
        stats = {}
        for name, fallback_key in (("input", "input_tokens"), ("output", "output_tokens"), ("e2e", "e2e_ms"), ("ttft", "ttft_ms"), ("tpot", "tpot_ms")):
            local = groups[key][name]
            global_values = groups["__global__"][name]
            if len(local) >= config["minimum_group_samples"]:
                values, source = local, "training_group"
            elif len(global_values) >= config["minimum_group_samples"]:
                values, source = global_values, "training_global"
            else:
                values, source = [config["fallback"][fallback_key]], "configured_fallback"
            stats[name] = {
                "median": statistics.median(values), "p75": quantile(values, 0.75),
                "heavy_threshold": quantile(values, config["heavy_quantile"]),
                "local_samples": len(local), "source": source,
            }
        result[key] = stats
    return {"fit_cutoff": cutoff.isoformat(), "latest_outcome_at": latest_outcome.isoformat() if latest_outcome else None,
            "groups": result, "scope": "training_completed_outcomes_only"}


def urgency_at(slo, now, estimated_remaining_service_ms, config, estimated_remaining_ttft_ms=None):
    """Dynamic slack, not a second fixed business-priority label."""
    from .normalize import parse_time

    remaining = (parse_time(slo["deadline"]) - now).total_seconds() * 1000
    slack = remaining - estimated_remaining_service_ms
    fraction = slack / slo["e2e_ms"]
    constraint = "e2e"
    if slo.get("ttft_deadline") and estimated_remaining_ttft_ms is not None:
        first_remaining = (parse_time(slo["ttft_deadline"]) - now).total_seconds() * 1000
        first_slack = first_remaining - estimated_remaining_ttft_ms
        if first_slack / slo["ttft_ms"] < fraction:
            remaining, slack = first_remaining, first_slack
            fraction, constraint = first_slack / slo["ttft_ms"], "ttft"
    if remaining <= 0 or fraction <= config["urgency"]["critical_slack_fraction"]:
        level = "critical"
    elif fraction <= config["urgency"]["elevated_slack_fraction"]:
        level = "elevated"
    else:
        level = "normal"
    return {"level": level, "limiting_constraint": constraint, "remaining_budget_ms": round(remaining, 3),
            "slack_ms": round(slack, 3), "slack_fraction": round(fraction, 6), "as_of": now.isoformat()}


def build_incoming_request(facts, profiles, config, split):
    """Reconstruct only arrival metadata; missing historic predictions are explicit mocks."""
    from .normalize import parse_time

    stats = profiles["groups"].get(group_key(facts), profiles["groups"]["__global__"])
    predictions, sources = {}, {}
    for axis in ("input", "output"):
        supplied = getattr(facts, f"predicted_{axis}_tokens")
        if supplied is not None:
            predictions[axis], sources[axis] = supplied, "request_prediction"
        elif facts.arrived_at < parse_time(profiles["fit_cutoff"]):
            predictions[axis] = round(config["fallback"][f"{axis}_tokens"])
            sources[axis] = "configured_warmup_mock"
        else:
            predictions[axis] = max(0, round(stats[axis]["median"]))
            sources[axis] = f"mock_{stats[axis]['source']}"
    return IncomingRequest(
        config["schema_version"], facts.request_id, facts.user_id, facts.model_id,
        facts.arrived_at, facts.is_stream, predictions["input"], predictions["output"],
        sources["input"], sources["output"], facts.task_type, facts.priority,
        facts.slo_tier, split, facts.source_line,
    )


def enrich_request(request: IncomingRequest, profiles, config, split=None):
    if not isinstance(request, IncomingRequest):
        raise TypeError("Gateway enrichment accepts IncomingRequest, not completed traffic logs")
    split = split or request.split
    key = group_key(request)
    stats = profiles["groups"].get(key, profiles["groups"]["__global__"])
    prediction = {}
    for name, raw in (("input", request.predicted_input_tokens), ("output", request.predicted_output_tokens)):
        prediction[f"{name}_tokens"] = raw
        prediction[f"{name}_source"] = getattr(request, f"{name}_prediction_source")
    pi, po = prediction["input_tokens"], prediction["output_tokens"]
    priority = request.priority
    if priority is not None and priority not in config["priority_distribution"]:
        raise ValueError("invalid_priority")
    if priority is None:
        digest = hashlib.sha256(f"{config['seed']}:{request.request_id}".encode()).digest()
        sample = int.from_bytes(digest[:8], "big") / 2**64
        cumulative = 0
        for level in ("high", "normal", "low"):
            cumulative += config["priority_distribution"][level]
            if sample < cumulative:
                priority = level
                break
    tier = config["priority_to_slo"][priority]
    if request.slo_tier is not None and request.slo_tier != tier:
        raise ValueError("priority_slo_conflict")
    input_ratio = pi / max(stats["input"]["median"], 1)
    output_ratio = po / max(stats["output"]["median"], 1)
    ttft = max(1, stats["ttft"]["p75"] * math.sqrt(max(0.25, input_ratio)))
    tpot = max(0.001, stats["tpot"]["p75"])
    if request.is_stream:
        service_ms = ttft + max(po - 1, 0) * tpot
    else:
        service_ms = max(1, stats["e2e"]["p75"] * max(0.25, 0.3 * input_ratio + 0.7 * output_ratio))
    tier_config = config["slo_tiers"][tier]
    multiplier = tier_config["latency_multiplier"]
    budget = service_ms * multiplier
    # TTFT includes gateway queueing, so streaming wait must fit both budgets.
    max_wait = budget * tier_config["queue_fraction"]
    if request.is_stream:
        max_wait = min(max_wait, ttft * multiplier * tier_config["queue_fraction"])
    slo = {"tier": tier, "e2e_ms": round(budget, 3),
           "ttft_ms": round(ttft * multiplier, 3) if request.is_stream else None,
           "tpot_ms": round(tpot * multiplier, 3) if request.is_stream else None,
           "max_wait_ms": round(max_wait, 3),
           "deadline": (request.arrived_at + timedelta(milliseconds=budget)).isoformat(),
           "ttft_deadline": (request.arrived_at + timedelta(milliseconds=ttft * multiplier)).isoformat() if request.is_stream else None,
           "source": "synthetic_policy_from_training", "profile_key": key}
    hi = max(config["heavy_min_input_tokens"], stats["input"]["heavy_threshold"])
    ho = max(config["heavy_min_output_tokens"], stats["output"]["heavy_threshold"])
    input_heavy, output_heavy = pi >= hi, po >= ho
    workload = "mixed_heavy" if input_heavy and output_heavy else "input_heavy" if input_heavy else "output_heavy" if output_heavy else "light"
    return {
        "schema_version": config["schema_version"], "request_id": request.request_id,
        "user_id": request.user_id,
        "model_id": request.model_id, "arrival_time": request.arrived_at.isoformat(),
        "stream_type": "stream" if request.is_stream else "nonstream", "split": split,
        "token_predictions": prediction, "priority": priority,
        "priority_source": "request" if request.priority is not None else "seeded_simulation",
        "slo": slo, "estimated_service_ms": round(service_ms, 3),
        "estimated_ttft_ms": round(ttft, 3) if request.is_stream else None,
        "workload": {"level": workload, "input_heavy": input_heavy, "output_heavy": output_heavy,
                     "classification_heavy": None, "task_type": request.task_type or "unknown",
                     "input_threshold": hi, "output_threshold": ho,
                     "source": "token_predictions_and_training_thresholds"},
        "urgency": urgency_at(slo, request.arrived_at, service_ms, config, ttft if request.is_stream else None),
        "source_line": request.source_line,
    }
