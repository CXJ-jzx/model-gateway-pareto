"""One parser for preparation, routing demos and factual replay."""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .schema import CanonicalAttempt, CanonicalRequest


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Absolute timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Expected a JSON object")
                    yield value
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON object") from exc


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def tokens(value, name, flags):
    parsed = number(value)
    if parsed is None or not parsed.is_integer():
        flags.append(f"{name}_missing" if value is None else f"{name}_invalid")
        return None
    return int(parsed)


def absolute_time(arrived_at, arrival_marker, marker):
    parsed = number(marker)
    return None if parsed is None else arrived_at + timedelta(milliseconds=parsed - arrival_marker)


def normalize_request(raw: dict, source_line: int = 0) -> CanonicalRequest:
    request_id, model = raw.get("request_id"), raw.get("target_model")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("missing_request_id")
    if not isinstance(model, str) or not model:
        raise ValueError("missing_model_id")
    if not isinstance(raw.get("is_stream"), bool):
        raise ValueError("invalid_stream_type")
    historical = raw.get("historical_performance") or {}
    arrival = parse_time(raw.get("request_arrival_at") or historical.get("request_arrival_at"))
    if arrival is None:
        raise ValueError("missing_absolute_arrival")
    marker = number(raw.get("arrival_at_ms"))
    if marker is None:
        raise ValueError("missing_or_invalid_arrival_marker")
    flags = list(raw.get("quality_flags") or [])
    actual_input = tokens(raw.get("input_tokens"), "actual_input_tokens", flags)
    actual_output = tokens(raw.get("output_tokens"), "actual_output_tokens", flags)
    predicted = raw.get("token_predictions") or {}
    pi = tokens(raw.get("predicted_input_tokens", predicted.get("input_tokens")), "predicted_input_tokens", flags)
    po = tokens(raw.get("predicted_output_tokens", predicted.get("output_tokens")), "predicted_output_tokens", flags)
    attempts, seen = [], set()
    for index, attempt in enumerate(raw.get("attempts") or []):
        af = list(attempt.get("quality_flags") or [])
        endpoint = attempt.get("endpoint_id")
        if not isinstance(endpoint, str) or not endpoint:
            flags.append("attempt_missing_endpoint")
            continue
        aid = str(attempt.get("attempt_id") or f"{request_id}:attempt:{index}")
        if aid in seen:
            flags.append("duplicate_attempt_id")
            continue
        seen.add(aid)
        if attempt.get("request_id", request_id) != request_id:
            flags.append("attempt_request_mismatch")
            continue
        sent = absolute_time(arrival, marker, attempt.get("sent_at_ms"))
        end = absolute_time(arrival, marker, attempt.get("finished_at_ms"))
        if sent is not None and sent < arrival:
            af.append("sent_before_arrival")
            sent = None
        if end is not None and (end < arrival or (sent is not None and end < sent)):
            af.append("invalid_attempt_time_order")
            end = None
        raw_output = attempt.get("output_tokens")
        # Only the final attempt can inherit request-level output usage.
        if raw_output is None and index == len(raw.get("attempts") or []) - 1 and endpoint == raw.get("final_endpoint_id"):
            raw_output = raw.get("output_tokens")
        output = tokens(raw_output, "output_tokens", af)
        status = attempt.get("http_status")
        if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
            status = None
            af.append("http_status_missing_or_invalid")
        result = str(attempt.get("result") or "unknown")
        success = result == "success" and status == 200
        timing = raw.get("usable_for_full_attempt_timing") is True
        e2e = (end - sent).total_seconds() * 1000 if success and timing and sent and end else None
        ttft = number(attempt.get("first_token_at_ms")) if raw["is_stream"] and success and timing else None
        # Source first_token_at_ms is a latency proxy, NOT a timestamp marker.
        if ttft is not None and (e2e is None or ttft > e2e + 0.01):
            af.append("invalid_ttft")
            ttft = None
        if ttft is not None and e2e is not None:
            ttft = min(ttft, e2e)
        tpot = (e2e - ttft) / (output - 1) if e2e is not None and ttft is not None and output and output > 1 else None
        if success and e2e is None:
            af.append("performance_timing_unusable")
        if raw["is_stream"] and success and ttft is None:
            af.append("ttft_unavailable")
        if end is None:
            af.append("completion_unavailable")
        attempts.append(CanonicalAttempt(aid, endpoint, sent, end, result, status,
                                         actual_input, output, e2e, ttft, tpot, tuple(sorted(set(af)))))
    return CanonicalRequest(
        request_id, model, arrival, raw["is_stream"], actual_input, actual_output, pi, po,
        raw.get("task_type"), raw.get("priority"), raw.get("slo_tier"),
        raw.get("final_endpoint_id"), str(raw.get("gateway_result") or "unknown"),
        tuple(attempts), tuple(sorted(set(flags))), source_line,
        raw.get("user_id"),
    )
