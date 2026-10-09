"""Extract source facts and make metric scopes explicit without inventing arrivals."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def token(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def timestamp(value):
    try:
        result = datetime.fromisoformat(value)
        return result.astimezone(timezone.utc) if result.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def iso(value):
    return value.isoformat() if value is not None else None


def event_time(arrival, arrival_marker, marker):
    marker = number(marker)
    if arrival is None or arrival_marker is None or marker is None or marker < arrival_marker:
        return None
    try:
        return arrival + timedelta(milliseconds=marker - arrival_marker)
    except OverflowError:
        return None


def prediction_values(raw):
    nested = raw.get("token_predictions") or {}
    nested = nested if isinstance(nested, dict) else {}
    values = [token(raw.get(f"predicted_{kind}_tokens", nested.get(f"{kind}_tokens"))) for kind in ("input", "output")]
    source = "provided" if all(v is not None for v in values) else "missing" if all(v is None for v in values) else "partial"
    return *values, source


def normalize_request(raw, source_line=1):
    """One completed-request fact; actual lengths never become predictions.

    TTFT/TPOT metrics use the final matched successful attempt. Request E2E uses
    elapsed_ms; request and attempt timing scopes remain separate.
    """
    flags = set(raw.get("quality_flags") or [])
    historical = raw.get("historical_performance") or {}
    arrival = timestamp(raw.get("request_arrival_at") or historical.get("request_arrival_at"))
    marker = number(raw.get("arrival_at_ms"))
    end = event_time(arrival, marker, raw.get("finished_at_ms"))
    if arrival is None:
        flags.add("invalid_arrival_time")
    if end is None:
        flags.add("invalid_request_finished_time")
    mode = "stream" if raw.get("is_stream") is True else "nonstream" if raw.get("is_stream") is False else "unknown"
    if mode == "unknown":
        flags.add("unknown_stream_type")
    success = raw.get("gateway_result") == "completed" and raw.get("http_status") == 200
    e2e = number(raw.get("elapsed_ms"))
    if e2e is None:
        flags.add("invalid_request_e2e")
    if e2e is not None and arrival is not None and end is not None and abs((end-arrival).total_seconds()*1000-e2e) > 1:
        flags.add("request_elapsed_mismatch_gt_1ms")
    tin, tout, cached = (token(raw.get(k)) for k in ("input_tokens", "output_tokens", "cache_hit_tokens"))
    for name, value in (("input", tin), ("output", tout)):
        if value is None:
            flags.add(f"missing_or_invalid_actual_{name}_tokens")
    pi, po, prediction_source = prediction_values(raw)
    request_ttft = number(raw.get("first_token_at_ms")) if success and mode == "stream" else None
    if request_ttft is not None and (e2e is None or request_ttft > e2e):
        flags.add("invalid_request_ttft_proxy")
        request_ttft = None
    attempts = raw.get("attempts") or []
    final = attempts[-1] if attempts else None
    attempt_e2e = ttft = tpot = None
    attempt_output = None
    if final and final.get("endpoint_id") == raw.get("final_endpoint_id"):
        start = event_time(arrival, marker, final.get("sent_at_ms"))
        finish = event_time(arrival, marker, final.get("finished_at_ms"))
        valid_time = start is not None and finish is not None and finish >= start
        if not valid_time:
            flags.add("invalid_final_attempt_time")
        if success and final.get("result") == "success" and final.get("http_status") == 200 and raw.get("usable_for_full_attempt_timing") is True and valid_time:
            attempt_e2e = (finish - start).total_seconds() * 1000
            if mode == "stream":
                ttft = number(final.get("first_token_at_ms"))
                if ttft is not None and ttft > attempt_e2e + 0.01:
                    flags.add("invalid_attempt_ttft_proxy")
                    ttft = None
                elif ttft is not None:
                    ttft = min(ttft, attempt_e2e)
                attempt_output = token(final.get("output_tokens"))
                if final.get("output_tokens") is None:
                    attempt_output = tout
                if ttft is not None and attempt_output is not None and attempt_output > 1:
                    tpot = (attempt_e2e - ttft) / (attempt_output - 1)
    if success and mode == "stream" and ttft is None:
        flags.add("ttft_proxy_unavailable")
    if success and mode == "stream" and tpot is None:
        flags.add("tpot_proxy_unavailable")
    refs = list(iter_references(raw))
    return dict(
        schema_version="1.0", record_kind="completed_request_fact", source_line=source_line,
        request_id=raw.get("request_id"), model_id=raw.get("target_model"), user_id=raw.get("user_id"),
        stream_type=mode, arrived_at=iso(arrival), finished_at=iso(end), success=success,
        gateway_result=raw.get("gateway_result"), http_status=raw.get("http_status"),
        final_endpoint_id=raw.get("final_endpoint_id"), actual_input_tokens=tin, actual_output_tokens=tout,
        actual_cache_hit_tokens=cached, predicted_input_tokens=pi, predicted_output_tokens=po,
        prediction_source=prediction_source, request_e2e_ms=e2e, request_ttft_proxy_ms=request_ttft,
        attempt_e2e_ms=attempt_e2e, ttft_proxy_ms=ttft, tpot_proxy_ms=tpot,
        final_attempt_output_tokens=attempt_output, attempt_count=len(attempts),
        source_attempt_count=raw.get("attempt_count"),
        source_usable_for_token_workload=raw.get("usable_for_token_workload"),
        source_usable_for_full_attempt_timing=raw.get("usable_for_full_attempt_timing"),
        eligible_length_e2e=success and mode != "unknown" and tout is not None and e2e is not None and arrival is not None,
        eligible_ttft_input=success and mode == "stream" and tin is not None and ttft is not None,
        eligible_tpot_output=success and mode == "stream" and tout is not None and tpot is not None,
        quality_flags=sorted(flags), historical_snapshot_ids=sorted({x["snapshot_id"] for x in refs}),
        metric_scope={"request_e2e_ms": "gateway_arrival_to_completion", "ttft_proxy_ms": "final_attempt_first_response_proxy", "tpot_proxy_ms": "final_attempt_derived_average"},
    )


def iter_references(raw):
    history = raw.get("historical_performance") or {}
    for endpoint in history.get("endpoint_histories") or []:
        for field, scope in (("endpoint_model", "endpoint_model"), ("endpoint_all_models", "endpoint")):
            for window, sid in (endpoint.get(field) or {}).items():
                if sid:
                    yield dict(request_id=raw["request_id"], snapshot_id=sid, scope=scope, window=window,
                               endpoint_id=endpoint.get("endpoint_id"), model_id=raw["target_model"] if scope == "endpoint_model" else None)
    for window, sid in (history.get("model_all_endpoints") or {}).items():
        if sid:
            yield dict(request_id=raw["request_id"], snapshot_id=sid, scope="model_all_endpoints", window=window,
                       endpoint_id=None, model_id=raw["target_model"])


def extract(source_dir, dataset_dir, model_id):
    """Keep every model request (including errors/unknown mode) and reference closure."""
    source_dir, dataset_dir = Path(source_dir), Path(dataset_dir)
    files = {"traffic": source_dir / "流量记录_历史性能.jsonl", "offerings": source_dir / "端点配置.jsonl", "snapshots": source_dir / "历史性能快照.jsonl"}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    selected, rows, references = [], [], []
    seen, total = set(), 0
    for line_number, raw in read_jsonl(files["traffic"]):
        total += 1
        if raw.get("target_model") != model_id:
            continue
        rid = raw.get("request_id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError(f"Invalid/duplicate request_id on source line {line_number}")
        seen.add(rid)
        selected.append(raw)
        rows.append(normalize_request(raw, line_number))
        references.extend(iter_references(raw))
    if not rows:
        raise ValueError(f"No requests for {model_id}")
    arrival_reversals = sum(a["arrived_at"] is not None and b["arrived_at"] is not None and a["arrived_at"] > b["arrived_at"] for a, b in zip(rows, rows[1:]))
    rows.sort(key=lambda row: (row["arrived_at"] or "9999", row["request_id"]))
    offerings = []
    for line_number, endpoint in read_jsonl(files["offerings"]):
        for entry in endpoint.get("models") or []:
            if entry.get("model_id") == model_id:
                offerings.append(dict(endpoint_id=endpoint["endpoint_id"], model_id=model_id,
                                      deployment_type=endpoint.get("deployment_type"), source_line=line_number,
                                      price_config=entry.get("price_config"), capacity_config=entry.get("capacity_config"),
                                      source_model_entry=entry))
    wanted = {r["snapshot_id"] for r in references}
    snapshots = {}
    for _, item in read_jsonl(files["snapshots"]):
        sid = item.get("snapshot_id")
        if sid in wanted:
            if sid in snapshots:
                raise ValueError(f"Duplicate referenced snapshot: {sid}")
            snapshots[sid] = item
    if wanted != snapshots.keys():
        raise ValueError(f"Unresolved snapshots: {sorted(wanted-snapshots.keys())[:5]}")
    dataset_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(dataset_dir / "raw_requests.jsonl", selected)
    write_jsonl(dataset_dir / "requests.jsonl", rows)
    write_jsonl(dataset_dir / "endpoint_offerings.jsonl", offerings)
    write_jsonl(dataset_dir / "historical_references.jsonl", references)
    write_jsonl(dataset_dir / "historical_snapshots.jsonl", snapshots.values())
    return rows, offerings, dict(
        model_id=model_id, source_request_count=total, selected_request_count=len(rows),
        endpoint_offering_count=len(offerings), historical_reference_count=len(references),
        source_order_arrival_reversals=arrival_reversals,
        unique_snapshot_count=len(snapshots), snapshot_scopes=dict(Counter(x["scope"] for x in snapshots.values())),
        sources={role: {"path": str(path.resolve()), "sha256": file_hash(path), "bytes": path.stat().st_size} for role, path in files.items()},
    )
