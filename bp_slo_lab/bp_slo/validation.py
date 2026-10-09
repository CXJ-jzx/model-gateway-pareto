"""Read-only artifact and provenance checks, including historical no-leakage."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import re
from pathlib import Path

from .dataset import file_hash, iter_references, normalize_request, read_jsonl, timestamp
from .statistics import analyze


REQUIRED = (
    "config.json", "analysis.json", "dataset/raw_requests.jsonl",
    "dataset/requests.jsonl", "dataset/endpoint_offerings.jsonl",
    "dataset/historical_references.jsonl", "dataset/historical_snapshots.jsonl",
    "dataset/diagnostic_splits.jsonl",
)


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _equal(actual, expected, detail):
    _require(_canonical(actual) == _canonical(expected), detail)


def _index(rows, field):
    result = {}
    for row in rows:
        key = row.get(field)
        _require(isinstance(key, str) and bool(key), f"Invalid {field}: {key!r}")
        _require(key not in result, f"Duplicate {field}: {key}")
        result[key] = row
    return result


def _json_object(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "Expected a JSON object")
    _canonical(value)  # Reject NaN/Infinity, which Python's JSON reader accepts.
    return value


def _jsonl_objects(path):
    rows = [row for _, row in read_jsonl(path)]
    _require(all(isinstance(row, dict) for row in rows), "Expected JSON object rows")
    _canonical(rows)
    return rows


def _digest_record(path, entry):
    _require(isinstance(entry, dict), f"Invalid digest record: {path}")
    _require(type(entry.get("bytes")) is int and entry["bytes"] >= 0,
             f"Invalid byte count: {path}")
    _require(isinstance(entry.get("sha256"), str)
             and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]),
             f"Invalid SHA-256: {path}")


def _digest_matches(path, entry):
    _digest_record(path, entry)
    _require(path.is_file(), f"Missing file: {path}")
    _require(path.stat().st_size == entry["bytes"], f"Byte count mismatch: {path}")
    _require(file_hash(path) == entry["sha256"], f"SHA-256 mismatch: {path}")


def _offering_projection(source, model):
    projected = []
    for line, endpoint in read_jsonl(source):
        for entry in endpoint.get("models") or []:
            if entry.get("model_id") == model:
                projected.append(dict(endpoint_id=endpoint["endpoint_id"], model_id=model,
                                      deployment_type=endpoint.get("deployment_type"), source_line=line,
                                      price_config=entry.get("price_config"),
                                      capacity_config=entry.get("capacity_config"), source_model_entry=entry))
    return projected


def validate_run(run_dir):
    """Return all possible checks; missing/malformed artifacts become failures.

    External source files are optional for portable runs. Missing sources are
    warnings, whereas an available source with different bytes is a failure.
    No artifact is changed by validation.
    """
    checks, warnings, data = [], [], {}

    def check(name, action):
        try:
            detail = action()
            checks.append(dict(name=name, passed=True, detail=detail or "OK"))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, detail=f"{type(exc).__name__}: {exc}"))

    def finish():
        return dict(schema_version="1.0", passed=bool(checks) and all(c["passed"] for c in checks),
                    checks=checks, warnings=warnings)

    try:
        root = Path(run_dir).resolve()
    except Exception as exc:
        checks.append(dict(name="run_directory", passed=False, detail=f"{type(exc).__name__}: {exc}"))
        return finish()

    for relative in ("manifest.json",) + REQUIRED:
        def load(relative=relative):
            path = root / relative
            data[relative] = _jsonl_objects(path) if relative.endswith(".jsonl") else _json_object(path)
        check(f"read:{relative}", load)

    def manifest_integrity():
        manifest = data["manifest.json"]
        artifacts = manifest["artifacts"]
        _require(isinstance(artifacts, dict), "artifacts must be an object")
        _require(set(REQUIRED).issubset(artifacts), "Manifest does not cover all required artifacts")
        for relative, entry in artifacts.items():
            _require(isinstance(relative, str) and relative, "Invalid artifact path")
            candidate = Path(relative)
            path = (root / candidate).resolve()
            _require(not candidate.is_absolute() and root in path.parents,
                     f"Artifact escapes run directory: {relative}")
            _require(relative not in ("manifest.json", "validation.json"), "Manifest includes its own output")
            _digest_matches(path, entry)
        _require(isinstance(manifest["sources"], dict), "sources must be an object")
        _equal(manifest["sources"], manifest["extraction"]["sources"],
               "Manifest and extraction source records differ")
        _require(set(manifest["sources"]) == {"traffic", "offerings", "snapshots"},
                 "Expected traffic, offerings, and snapshots sources")
        return f"Verified {len(artifacts)} artifact hashes"
    check("manifest_integrity", manifest_integrity)

    source_paths = {}
    for role in ("traffic", "offerings", "snapshots"):
        def source_check(role=role):
            entry = data["manifest.json"]["sources"][role]
            _require(isinstance(entry, dict), f"Invalid source entry: {role}")
            _require(isinstance(entry.get("path"), str) and bool(entry["path"]), "Invalid source path")
            path = Path(entry["path"])
            if not path.is_absolute():
                path = root / path
            _digest_record(path, entry)
            if not path.exists():
                warning = f"Optional {role} source unavailable: {path}; external verification skipped"
                warnings.append(warning)
                return warning
            _digest_matches(path, entry)
            source_paths[role] = path
            return "Source hash and byte count verified"
        check(f"source:{role}", source_check)

    def requests_check():
        raw = data["dataset/raw_requests.jsonl"]
        rows = data["dataset/requests.jsonl"]
        model = data["config.json"]["model_id"]
        _require(isinstance(model, str) and bool(model), "Invalid configured model")
        _require(bool(raw), "No selected requests")
        raw_index, row_index = _index(raw, "request_id"), _index(rows, "request_id")
        _require(raw_index.keys() == row_index.keys(), "Raw and normalized request IDs differ")
        _require(all(r.get("target_model") == model for r in raw), "Raw requests contain another model")
        lines = []
        for rid, item in raw_index.items():
            row = row_index[rid]
            source_line = row.get("source_line")
            _require(type(source_line) is int and source_line > 0, f"Invalid source line: {rid}")
            lines.append(source_line)
            _equal(row, normalize_request(item, source_line), f"Normalization mismatch: {rid}")
        _require(lines == sorted(set(lines)), "Raw requests do not preserve unique source-line order")
        _equal(rows, sorted(rows, key=lambda r: (r["arrived_at"] or "9999", r["request_id"])),
               "Normalized requests are not in chronological order")
        return f"Recomputed {len(rows)} normalized requests"
    check("request_identity_normalization_order", requests_check)

    def references_check():
        raw = data["dataset/raw_requests.jsonl"]
        references = data["dataset/historical_references.jsonl"]
        snapshots = data["dataset/historical_snapshots.jsonl"]
        _equal(references, [ref for item in raw for ref in iter_references(item)],
               "Historical references do not equal raw request references")
        snapshot_index = _index(snapshots, "snapshot_id")
        _require(set(snapshot_index) == {r["snapshot_id"] for r in references},
                 "Snapshot closure has missing or unreferenced snapshots")
        rows = _index(data["dataset/requests.jsonl"], "request_id")
        for ref in references:
            snapshot = snapshot_index[ref["snapshot_id"]]
            label = ref["snapshot_id"]
            for field in ("scope", "model_id", "endpoint_id"):
                _equal(snapshot.get(field), ref[field], f"Snapshot {field} mismatch: {label}")
            window = re.fullmatch(r"([1-9][0-9]*)d", ref["window"])
            _require(window is not None, f"Invalid reference window: {label}")
            _require(type(snapshot.get("window_days")) is int
                     and snapshot["window_days"] == int(window.group(1)), f"Window mismatch: {label}")
            as_of = timestamp(snapshot.get("as_of_exclusive"))
            epoch = snapshot.get("as_of_epoch")
            if epoch is not None:
                _require(type(epoch) in (int, float), f"Invalid as_of_epoch: {label}")
                epoch_time = datetime.fromtimestamp(epoch, timezone.utc)
                _require(as_of is None or as_of == epoch_time, f"Conflicting as-of timestamps: {label}")
                as_of = epoch_time
            _require(as_of is not None, f"Missing/invalid snapshot as-of time: {label}")
            if snapshot.get("as_of_exclusive") is not None:
                _require(timestamp(snapshot["as_of_exclusive"]) is not None, f"Invalid as_of_exclusive: {label}")
            arrival = timestamp(rows[ref["request_id"]].get("arrived_at"))
            _require(arrival is not None and as_of <= arrival, f"Historical snapshot leaks future data: {label}")
            if snapshot.get("included_end_exclusive") is not None:
                included_end = timestamp(snapshot["included_end_exclusive"])
                _require(included_end is not None and included_end <= as_of,
                         f"Included window ends after snapshot as-of: {label}")
            if snapshot.get("last_observation_at") is not None:
                last = timestamp(snapshot["last_observation_at"])
                _require(last is not None and last < as_of, f"Last observation is not strictly historical: {label}")
        return f"Verified closure and no-leakage for {len(references)} references"
    check("historical_reference_closure_and_no_leakage", references_check)

    def offerings_check():
        offerings = data["dataset/endpoint_offerings.jsonl"]
        model = data["config.json"]["model_id"]
        keys = set()
        for entry in offerings:
            key = (entry["endpoint_id"], entry["model_id"])
            _require(isinstance(key[0], str) and bool(key[0]), "Invalid offering endpoint")
            _require(key[1] == model, "Offering model mismatch")
            _require(key not in keys, f"Duplicate endpoint offering: {key}")
            keys.add(key)
            _require(type(entry["source_line"]) is int and entry["source_line"] > 0, "Invalid offering source line")
            source = entry["source_model_entry"]
            _require(source["model_id"] == model, "Offering source model mismatch")
            for field in ("price_config", "capacity_config"):
                _equal(entry.get(field), source.get(field), f"Offering {field} projection mismatch")
        return f"Verified {len(offerings)} unique model offerings"
    check("endpoint_offerings", offerings_check)

    def extraction_check():
        extraction = data["manifest.json"]["extraction"]
        rows = data["dataset/requests.jsonl"]
        raw = data["dataset/raw_requests.jsonl"]
        snapshots = data["dataset/historical_snapshots.jsonl"]
        indexed = _index(rows, "request_id")
        arrivals = [indexed[r["request_id"]]["arrived_at"] for r in raw]
        expected = dict(model_id=data["config.json"]["model_id"], selected_request_count=len(rows),
                        endpoint_offering_count=len(data["dataset/endpoint_offerings.jsonl"]),
                        historical_reference_count=len(data["dataset/historical_references.jsonl"]),
                        unique_snapshot_count=len(snapshots),
                        snapshot_scopes=dict(Counter(s["scope"] for s in snapshots)),
                        source_order_arrival_reversals=sum(a is not None and b is not None and a > b
                                                          for a, b in zip(arrivals, arrivals[1:])))
        for key, value in expected.items():
            _equal(extraction[key], value, f"Extraction metadata mismatch: {key}")
    check("extraction_metadata", extraction_check)

    def external_data_check():
        model = data["config.json"]["model_id"]
        if "traffic" in source_paths:
            selected, source_lines, total = [], {}, 0
            for line, raw in read_jsonl(source_paths["traffic"]):
                total += 1
                if raw.get("target_model") == model:
                    selected.append(raw)
                    _require(raw["request_id"] not in source_lines, "Duplicate source request ID")
                    source_lines[raw["request_id"]] = line
            _equal(data["dataset/raw_requests.jsonl"], selected, "Selected raw requests are not complete/source ordered")
            _equal(data["manifest.json"]["extraction"]["source_request_count"], total, "Source request count mismatch")
            for row in data["dataset/requests.jsonl"]:
                _require(row["source_line"] == source_lines[row["request_id"]], "Request source line mismatch")
        if "offerings" in source_paths:
            _equal(data["dataset/endpoint_offerings.jsonl"], _offering_projection(source_paths["offerings"], model),
                   "Offerings do not match source projection")
        if "snapshots" in source_paths:
            saved = _index(data["dataset/historical_snapshots.jsonl"], "snapshot_id")
            original = {}
            for _, snapshot in read_jsonl(source_paths["snapshots"]):
                sid = snapshot.get("snapshot_id")
                if sid in saved:
                    _require(sid not in original, f"Duplicate source snapshot: {sid}")
                    original[sid] = snapshot
            _equal(saved, original, "Historical snapshots differ from source")
        return f"Compared extracted facts against {len(source_paths)} available sources"
    check("external_source_projection_and_completeness", external_data_check)

    def recompute_check():
        expected, splits = analyze(data["dataset/requests.jsonl"], data["dataset/endpoint_offerings.jsonl"],
                                   data["config.json"], data["manifest.json"]["extraction"])
        _equal(data["analysis.json"], expected, "Analysis differs from recomputed descriptive statistics")
        _equal(data["dataset/diagnostic_splits.jsonl"], splits, "Diagnostic splits/outcome visibility differ from recomputation")
        return "Analysis and temporal diagnostic splits exactly reproduced"
    check("analysis_and_splits_reproducibility", recompute_check)
    return finish()
