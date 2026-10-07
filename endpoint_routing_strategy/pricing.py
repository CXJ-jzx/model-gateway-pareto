"""Resolve token-length pricing tiers without guessing unknown conditions.

The source configuration uses ``input_tokens_min_exclusive`` /
``input_tokens_max_inclusive`` and, for some offerings, input-token bands.
``k`` in a band is explicitly interpreted as decimal 1,000 tokens; the dataset
does not define a binary convention. Currency conversion is out of scope.
"""
from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

from .models import EndpointOffering, Price


_BOUND_KEYS = ("input_tokens_min_exclusive", "input_tokens_max_inclusive")
_BAND_PATTERN = re.compile(r"(lte|gt)_(\d+)(k)?", re.IGNORECASE)


def _token_related(key: str) -> bool:
    """Do not mistake ordinary metadata (e.g. 'tier') for token conditions."""
    lowered = key.lower()
    return "token" in lowered or "length" in lowered or "context_window" in lowered


def _tier_conditions(tier: dict) -> dict:
    result = deepcopy(tier.get("conditions") or {})
    result.update({key: tier[key] for key in _BOUND_KEYS if key in tier})
    return result


def _evaluate_token_rules(tier: dict, index: int, input_tokens: int) -> dict:
    conditions = tier.get("conditions")
    if conditions is None:
        conditions = {}
    evaluation = dict(
        tier_index=index, has_token_rules=False, matches=True, rules={},
        unsupported_conditions=[], invalid_rules=[],
    )
    if not isinstance(conditions, dict):
        evaluation["invalid_rules"].append("conditions_not_an_object")
        evaluation["matches"] = None
        return evaluation
    for key in tier:
        if key not in _BOUND_KEYS and _token_related(key):
            evaluation["unsupported_conditions"].append(key)
    for key in conditions:
        if key != "input_token_band" and _token_related(key):
            evaluation["unsupported_conditions"].append(f"conditions.{key}")

    lower = upper = None
    for key in _BOUND_KEYS:
        if key not in tier:
            continue
        value = tier[key]
        evaluation["has_token_rules"] |= value is not None
        evaluation["rules"][key] = value
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            evaluation["invalid_rules"].append(key)
        elif key == "input_tokens_min_exclusive":
            lower = value
        else:
            upper = value
    if lower is not None and upper is not None and lower >= upper:
        evaluation["invalid_rules"].append("empty_or_reversed_input_token_range")
    if lower is not None and input_tokens <= lower:
        evaluation["matches"] = False
    if upper is not None and input_tokens > upper:
        evaluation["matches"] = False

    if "input_token_band" in conditions:
        evaluation["has_token_rules"] = True
        band = conditions["input_token_band"]
        evaluation["rules"]["input_token_band"] = band
        parsed = _BAND_PATTERN.fullmatch(band) if isinstance(band, str) else None
        if parsed is None:
            evaluation["unsupported_conditions"].append("conditions.input_token_band")
        else:
            comparison, number, suffix = parsed.groups()
            threshold = int(number) * (1000 if suffix else 1)
            evaluation["rules"]["band_threshold_tokens"] = threshold
            evaluation["rules"]["band_k_multiplier"] = 1000
            match = input_tokens <= threshold if comparison.lower() == "lte" else input_tokens > threshold
            evaluation["matches"] = evaluation["matches"] and match

    if evaluation["unsupported_conditions"]:
        evaluation["has_token_rules"] = True
    if evaluation["unsupported_conditions"] or evaluation["invalid_rules"]:
        # Even an apparently unmatched known condition cannot safely establish
        # uniqueness when another part of the token rule is unknown.
        evaluation["matches"] = None
    return evaluation


def resolve_request_price(
    offering: EndpointOffering, configured_tier_index: int, predicted_input_tokens: int,
    currency: str, tier_mode: str = "auto",
) -> tuple[Price | None, dict[str, Any]]:
    """Return one justified price plus an auditable tier-resolution record.

    ``auto`` requires exactly one token-rule match. With no token rules anywhere
    in the offering, the configured tier remains authoritative (non-token
    conditions such as modality/time bands must be configured by the caller).
    ``configured`` explicitly chooses a tier but still refuses a token-rule
    mismatch or an unsupported token condition. Failure returns ``None`` and a
    machine-readable ``reason``; success has ``reason=None``.
    """
    if not isinstance(offering, EndpointOffering):
        raise ValueError("offering must be EndpointOffering")
    for name, value in (("configured_tier_index", configured_tier_index),
                        ("predicted_input_tokens", predicted_input_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(currency, str) or not currency:
        raise ValueError("currency is required")
    if tier_mode not in {"auto", "configured"}:
        raise ValueError("tier_mode must be auto or configured")

    trace: dict[str, Any] = dict(
        reason=None, selection_source=None, tier_index=None, tier_mode=tier_mode,
        configured_tier_index=configured_tier_index, request_input_tokens=predicted_input_tokens,
        requested_currency=currency, tier_conditions={}, has_token_rules=False,
        matched_tier_indexes=[], token_rule_evaluations=[],
        band_k_multiplier=1000, warnings=[],
    )
    config = offering.price_config
    if not isinstance(config, dict):
        trace["reason"] = "invalid_price_config"
        return None, trace
    if config.get("pricing_status") != "configured":
        trace["reason"] = "pricing_not_configured"
        return None, trace
    tiers = config.get("price_tiers")
    if not isinstance(tiers, (list, tuple)) or not tiers:
        trace["reason"] = "missing_price_tiers"
        return None, trace
    if any(not isinstance(tier, dict) for tier in tiers):
        trace["reason"] = "invalid_price_tier_config"
        return None, trace

    evaluations = [_evaluate_token_rules(tier, index, predicted_input_tokens)
                   for index, tier in enumerate(tiers)]
    trace["token_rule_evaluations"] = evaluations
    trace["has_token_rules"] = any(item["has_token_rules"] for item in evaluations)
    trace["matched_tier_indexes"] = [item["tier_index"] for item in evaluations if item["matches"] is True]
    if tier_mode == "auto" and trace["has_token_rules"]:
        trace["selection_source"] = "predicted_input_token_rules"
        if any(item["invalid_rules"] for item in evaluations):
            trace["reason"] = "invalid_token_rules"
            return None, trace
        if any(item["unsupported_conditions"] for item in evaluations):
            trace["reason"] = "unsupported_token_conditions"
            return None, trace
        matches = trace["matched_tier_indexes"]
        if not matches:
            trace["reason"] = "no_matching_price_tier"
            return None, trace
        if len(matches) > 1:
            trace["reason"] = "ambiguous_price_tiers"
            return None, trace
        selected_index = matches[0]
    else:
        trace["selection_source"] = ("configured_explicit" if tier_mode == "configured"
                                     else "configured_no_token_rules")
        if configured_tier_index >= len(tiers):
            trace["reason"] = "invalid_price_tier_index"
            return None, trace
        selected_index = configured_tier_index

    trace["tier_index"] = selected_index
    selected = tiers[selected_index]
    evaluation = evaluations[selected_index]
    if evaluation["invalid_rules"]:
        trace["reason"] = "invalid_token_rules"
        return None, trace
    if evaluation["unsupported_conditions"]:
        trace["reason"] = "unsupported_token_conditions"
        return None, trace
    trace["tier_conditions"] = _tier_conditions(selected)
    if evaluation["matches"] is not True:
        trace["reason"] = "configured_token_tier_mismatch"
        return None, trace
    if tier_mode == "configured" and len(trace["matched_tier_indexes"]) > 1 and trace["has_token_rules"]:
        trace["warnings"].append("explicit_tier_resolves_overlapping_token_ranges")
    if any(key not in {"input_token_band"} for key in (selected.get("conditions") or {})):
        trace["warnings"].append("non_token_conditions_require_caller_configuration")
    billing_unit = selected.get("billing_unit") or config.get("billing_unit")
    if billing_unit != "per_million_tokens":
        trace["reason"] = "unsupported_billing_unit"
        return None, trace
    selected_currency = selected.get("currency") or config.get("currency")
    trace["selected_currency"] = selected_currency
    if selected_currency != currency:
        trace["reason"] = "currency_mismatch"
        return None, trace
    pin, pout = selected.get("input_per_million"), selected.get("output_per_million")
    if pin is None or pout is None:
        trace["reason"] = "missing_unit_prices"
        return None, trace
    try:
        if isinstance(pin, bool) or isinstance(pout, bool):
            raise ValueError("Boolean price")
        pin, pout = float(pin), float(pout)
        if not all(math.isfinite(value) and value >= 0 for value in (pin, pout)):
            raise ValueError("Invalid price")
    except (TypeError, ValueError, OverflowError):
        trace["reason"] = "invalid_unit_prices"
        return None, trace
    return Price(
        endpoint_id=offering.endpoint_id, model_id=offering.model_id, currency=selected_currency,
        input_per_million=pin, output_per_million=pout, tier_index=selected_index,
        tier_conditions=deepcopy(trace["tier_conditions"]),
    ), trace
