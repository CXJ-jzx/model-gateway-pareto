"""Independent artifact checks, including lineage, policy consistency and leakage."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .normalize import iter_jsonl, parse_time
from .profiles import build_incoming_request, enrich_request, fit_profiles, validate_config
from .schema import CanonicalRequest, IncomingRequest


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    profiles = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
    validate_config(config)
    errors = Counter()
    source = Path(manifest["source_file"])
    if source.exists() and file_hash(source) != manifest["source_sha256"]:
        errors["source_hash_mismatch"] += 1
    for name, expected in manifest["artifact_hashes"].items():
        if not (directory / name).is_file() or file_hash(directory / name) != expected:
            errors["artifact_hash_mismatch"] += 1
    facts_list = list(iter_jsonl(directory / "canonical_requests.jsonl"))
    canonical = {r["request_id"]: r for r in facts_list}
    if len(canonical) != len(facts_list):
        errors["duplicate_canonical_request"] += 1
    prepared = list(iter_jsonl(directory / "requests.jsonl"))
    incoming_list = list(iter_jsonl(directory / "incoming_requests.jsonl"))
    incoming = {r["request_id"]: IncomingRequest.from_dict(r) for r in incoming_list}
    if len(incoming) != len(incoming_list) or set(incoming) != set(canonical):
        errors["incoming_count_or_lineage_mismatch"] += 1
    seen = set()
    cutoff = parse_time(profiles["fit_cutoff"])
    training = [CanonicalRequest.from_dict(r) for r in facts_list if parse_time(r["arrived_at"]) < cutoff]
    if fit_profiles(training, cutoff, config) != profiles:
        errors["training_profiles_not_reproducible"] += 1
    latest = parse_time(profiles["latest_outcome_at"])
    if latest and latest >= cutoff:
        errors["training_outcome_after_cutoff"] += 1
    for rid, initial in incoming.items():
        facts = canonical.get(rid)
        if facts is None:
            errors["incoming_missing_lineage"] += 1
            continue
        arrival = initial.arrived_at
        split = "train" if arrival < cutoff else "calibration" if arrival < parse_time(manifest["test_cutoff"]) else "test"
        if initial != build_incoming_request(CanonicalRequest.from_dict(facts), profiles, config, split):
            errors["initial_request_not_reproducible"] += 1
    for row in prepared:
        rid = row["request_id"]
        if rid in seen:
            errors["duplicate_request"] += 1
        seen.add(rid)
        facts = canonical.get(rid)
        if facts is None:
            errors["missing_lineage"] += 1
            continue
        expected = enrich_request(incoming[rid], profiles, config, row["split"])
        if row != expected:
            errors["enrichment_not_reproducible"] += 1
        arrival = parse_time(row["arrival_time"])
        if arrival < cutoff or row["split"] not in {"calibration", "test"}:
            errors["evaluation_before_fit_cutoff"] += 1
        if row["split"] != ("calibration" if arrival < parse_time(manifest["test_cutoff"]) else "test"):
            errors["split_boundary_mismatch"] += 1
        if row["priority"] not in config["priority_distribution"] or row["slo"]["tier"] != config["priority_to_slo"].get(row["priority"]):
            errors["priority_slo_conflict"] += 1
        for name in ("input", "output"):
            value = row["token_predictions"][f"{name}_tokens"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors["invalid_prediction"] += 1
            if row["token_predictions"][f"{name}_source"] == "request_prediction" and value != facts[f"predicted_{name}_tokens"]:
                errors["prediction_not_preserved"] += 1
        forbidden = {"actual_output_tokens", "actual_input_tokens", "final_endpoint_id", "attempts", "http_status", "gateway_result", "data_quality_flags", "finished_at", "flags"}
        if forbidden & row.keys():
            errors["outcomes_in_online_request"] += 1
        slo = row["slo"]
        remaining = (parse_time(slo["deadline"]) - arrival).total_seconds() * 1000
        if abs(remaining - slo["e2e_ms"]) > 0.02:
            errors["deadline_budget_inconsistent"] += 1
        if slo["max_wait_ms"] + row["estimated_service_ms"] > slo["e2e_ms"] + 0.02:
            errors["queue_exceeds_service_budget"] += 1
        if row["stream_type"] == "stream":
            if not slo["ttft_ms"] or not slo["tpot_ms"] or slo["max_wait_ms"] >= slo["ttft_ms"]:
                errors["stream_budget_invalid"] += 1
            if slo["max_wait_ms"] + row["estimated_ttft_ms"] > slo["ttft_ms"] + 0.02:
                errors["queue_exceeds_ttft_budget"] += 1
        elif slo["ttft_ms"] is not None or slo["tpot_ms"] is not None:
            errors["nonstream_has_stream_slo"] += 1
        workload = row["workload"]
        for name in ("input", "output"):
            expected = row["token_predictions"][f"{name}_tokens"] >= workload[f"{name}_threshold"]
            if expected != workload[f"{name}_heavy"]:
                errors["heavy_prediction_inconsistent"] += 1
        if row["urgency"]["as_of"] != row["arrival_time"]:
            errors["initial_urgency_time_invalid"] += 1
    observations = list(iter_jsonl(directory / "observations.jsonl"))
    for row in observations:
        if row["occurred_at"] != row["finished_at"]:
            errors["observation_published_before_completion"] += 1
        if parse_time(row["occurred_at"]) < parse_time(row["arrival_time"]):
            errors["observation_before_arrival"] += 1
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit["duplicate_conflicts"]:
        errors["conflicting_duplicates"] += audit["duplicate_conflicts"]
    if audit["input_nonblank_lines"] != audit["accepted_requests"] + audit["rejected_lines"] + audit["duplicate_requests"]:
        errors["count_conservation_failed"] += 1
    if len(prepared) != manifest["prepared_requests"] or len(observations) != manifest["observations"] or len(incoming) != manifest["incoming_requests"]:
        errors["manifest_count_mismatch"] += 1
    if len(prepared) + audit["enrichment_rejected"] + audit["training_requests"] != audit["accepted_requests"]:
        errors["enrichment_count_conservation_failed"] += 1
    return {"passed": not errors, "errors": dict(errors), "checks": [
        "artifact_hashes", "source_hash_when_available", "request_uniqueness", "lineage", "training_cutoff",
        "prediction_preservation", "online_outcome_separation", "priority_slo_mapping",
        "deadline_queue_budget", "heavy_consistency", "completion_visibility", "count_conservation",
        "training_profiles_recomputed", "enrichment_recomputed", "split_boundaries",
        "initial_request_whitelist", "initial_request_recomputed",
    ], "incoming_requests": len(incoming), "prepared_requests": len(prepared), "observations": len(observations)}
