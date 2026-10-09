"""Descriptive statistics only: variance is never a routing penalty."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta, timezone
import math
import statistics as st

from .dataset import timestamp


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*q
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right]-values[left])*(position-left)


def describe(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)]
    n = len(values)
    return dict(n=n, mean=st.mean(values) if n else None,
                variance=st.variance(values) if n > 1 else None, std=st.stdev(values) if n > 1 else None,
                min=min(values) if n else None, max=max(values) if n else None,
                **{f"p{int(q*100)}": quantile(values, q) for q in (.5, .75, .9, .95)})


def _ranks(values):
    ranked = [0.0]*len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for index in ordered[i:j]:
            ranked[index] = (i+j-1)/2 + 1
        i = j
    return ranked


def correlations(rows, x, y):
    pairs = [(r[x], r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None]
    if len(pairs) < 3:
        return dict(n=len(pairs), pearson=None, spearman=None)
    a, b = zip(*pairs)
    if len(set(a)) < 2 or len(set(b)) < 2:
        return dict(n=len(pairs), pearson=None, spearman=None)
    return dict(n=len(pairs), pearson=st.correlation(a, b), spearman=st.correlation(_ranks(a), _ranks(b)))


def bin_label(value, bounds):
    if value is None:
        return None
    lower = 0
    for upper in bounds:
        if value <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def group_statistics(rows, length, bounds, metric, minimum):
    groups = defaultdict(list)
    for row in rows:
        if row.get(length) is not None and row.get(metric) is not None:
            groups[bin_label(row[length], bounds)].append(row[metric])
    labels = [bin_label(v, bounds) for v in [0]+[b+1 for b in bounds]]
    return [{"range": label, "sample_status": "enough_for_initial_screen" if len(groups[label]) >= minimum else "sparse",
             **describe(groups[label])} for label in labels]


def assign_time_splits(rows, train_fraction=.6, calibration_fraction=.2):
    if (isinstance(train_fraction, bool) or isinstance(calibration_fraction, bool)
            or not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1
            or train_fraction + calibration_fraction >= 1):
        raise ValueError("Fractions must be positive and leave a test interval")
    assignments = {r["request_id"]: "unassigned" for r in rows}
    metadata = {}
    for mode in ("nonstream", "stream"):
        ordered = sorted((r for r in rows if r["stream_type"] == mode and timestamp(r.get("arrived_at")) is not None),
                         key=lambda r: (timestamp(r["arrived_at"]), r["request_id"]))
        if not ordered:
            continue
        n = len(ordered)
        train_cutoff = timestamp(ordered[min(n-1, int(n*train_fraction))]["arrived_at"])
        calibration_cutoff = timestamp(ordered[min(n-1, int(n*(train_fraction+calibration_fraction)))]["arrived_at"])
        for row in ordered:
            arrived = timestamp(row["arrived_at"])
            assignments[row["request_id"]] = "train" if arrived < train_cutoff else "calibration" if arrived < calibration_cutoff else "test"
        metadata[mode] = dict(train_cutoff=train_cutoff.isoformat(), calibration_cutoff=calibration_cutoff.isoformat(),
                              counts=dict(Counter(assignments[r["request_id"]] for r in ordered)))
    return assignments, metadata


def analyze(rows, offerings, config, extraction):
    modes = {}
    for mode in ("nonstream", "stream", "unknown"):
        group = [r for r in rows if r["stream_type"] == mode]
        success = [r for r in group if r["success"]]
        modes[mode] = dict(total=len(group), success=len(success), failures=len(group)-len(success),
                           success_rate=len(success)/len(group) if group else None,
                           length_e2e_eligible=sum(r["eligible_length_e2e"] for r in group),
                           ttft_input_eligible=sum(r["eligible_ttft_input"] for r in group),
                           tpot_output_eligible=sum(r["eligible_tpot_output"] for r in group),
                           http_statuses=dict(Counter(str(r["http_status"]) for r in group)),
                           prediction_sources=dict(Counter(r["prediction_source"] for r in group)),
                           missing_input_success=sum(r["actual_input_tokens"] is None for r in success),
                           missing_output_success=sum(r["actual_output_tokens"] is None for r in success),
                           metrics={metric: describe(r[metric] for r in success) for metric in (
                               "request_e2e_ms", "attempt_e2e_ms", "request_ttft_proxy_ms", "ttft_proxy_ms", "tpot_proxy_ms",
                               "actual_input_tokens", "actual_output_tokens", "actual_cache_hit_tokens")})
    non = [r for r in rows if r["stream_type"] == "nonstream" and r["success"]]
    stream = [r for r in rows if r["stream_type"] == "stream" and r["success"]]
    minimum = config["minimum_group_samples"]
    tables = {
        "nonstream_e2e_by_output": group_statistics([r for r in non if r["eligible_length_e2e"]], "actual_output_tokens", config["output_bin_upper_bounds"], "request_e2e_ms", minimum),
        "nonstream_e2e_by_input": group_statistics(non, "actual_input_tokens", config["input_bin_upper_bounds"], "request_e2e_ms", minimum),
        "stream_ttft_by_input": group_statistics(stream, "actual_input_tokens", config["input_bin_upper_bounds"], "ttft_proxy_ms", minimum),
        "stream_tpot_by_output": group_statistics(stream, "actual_output_tokens", config["output_bin_upper_bounds"], "tpot_proxy_ms", minimum),
        "stream_tpot_by_input": group_statistics(stream, "actual_input_tokens", config["input_bin_upper_bounds"], "tpot_proxy_ms", minimum),
    }
    associations = {name: correlations(group, x, y) for name, group, x, y in (
        ("nonstream_output_e2e", non, "actual_output_tokens", "request_e2e_ms"),
        ("nonstream_input_e2e", non, "actual_input_tokens", "request_e2e_ms"),
        ("stream_input_ttft", stream, "actual_input_tokens", "ttft_proxy_ms"),
        ("stream_output_tpot", stream, "actual_output_tokens", "tpot_proxy_ms"),
        ("stream_input_tpot", stream, "actual_input_tokens", "tpot_proxy_ms"),
    )}
    assignments, split_meta = assign_time_splits(rows, config["train_fraction"], config["calibration_fraction"])
    split_rows, coverage = [], []
    for r in rows:
        split = assignments[r["request_id"]]
        cutoff = timestamp(split_meta.get(r["stream_type"], {}).get("train_cutoff"))
        finished = timestamp(r["finished_at"])
        split_rows.append(dict(request_id=r["request_id"], stream_type=r["stream_type"], split=split,
                               arrived_at=r["arrived_at"], finished_at=r["finished_at"],
                               training_outcome_visible=split == "train" and cutoff is not None and finished is not None and finished < cutoff))
    for mode in ("nonstream", "stream"):
        for split in ("train", "calibration", "test"):
            part = [r for r in rows if r["stream_type"] == mode and assignments[r["request_id"]] == split]
            eligible = [r for r in part if r["eligible_length_e2e"]]
            valid_metric = "ttft_proxy_ms" if mode == "stream" else "request_e2e_ms"
            coverage.append(dict(mode=mode, split=split, total=len(part), valid=len(eligible),
                                 output_bins=dict(Counter(bin_label(r["actual_output_tokens"], config["output_bin_upper_bounds"]) for r in eligible)),
                                 input_bins_ttft=dict(Counter(bin_label(r["actual_input_tokens"], config["input_bin_upper_bounds"]) for r in part if r["eligible_ttft_input"])),
                                 primary_metric=describe(r[valid_metric] for r in part if r["success"]),
                                 tpot=describe(r["tpot_proxy_ms"] for r in part if r["success"])))
    hourly = defaultdict(list)
    for r in rows:
        dt = timestamp(r["arrived_at"])
        if dt is not None and r["stream_type"] != "unknown":
            hourly[(r["stream_type"], dt.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:00"))].append(r)
    hourly_stats = [dict(mode=mode, hour=hour, total=len(items), success=sum(r["success"] for r in items),
                         request_e2e_ms=describe(r["request_e2e_ms"] for r in items if r["success"]),
                         ttft_proxy_ms=describe(r["ttft_proxy_ms"] for r in items if r["success"]),
                         tpot_proxy_ms=describe(r["tpot_proxy_ms"] for r in items if r["success"]))
                    for (mode, hour), items in sorted(hourly.items())]
    endpoint_counts = []
    for ep in sorted({r["final_endpoint_id"] for r in rows if r["final_endpoint_id"] is not None}):
        for mode in ("nonstream", "stream"):
            part = [r for r in rows if r["final_endpoint_id"] == ep and r["stream_type"] == mode]
            endpoint_counts.append(dict(endpoint_id=ep, mode=mode, total=len(part), success=sum(r["success"] for r in part),
                                        length_e2e_eligible=sum(r["eligible_length_e2e"] for r in part),
                                        ttft_input_eligible=sum(r["eligible_ttft_input"] for r in part),
                                        tpot_output_eligible=sum(r["eligible_tpot_output"] for r in part)))
    dates = [r["arrived_at"] for r in rows if r["arrived_at"]]
    result = dict(schema_version="1.0", model_id=config["model_id"], total=len(rows), modes=modes,
                  time_range_utc=[min(dates), max(dates)] if dates else None, extraction=extraction,
                  groups=tables, correlations=associations, endpoint_counts=endpoint_counts,
                  attempt_counts=dict(Counter(str(r["attempt_count"]) for r in rows)),
                  quality_flags=dict(Counter(flag for r in rows for flag in r["quality_flags"])),
                  raw_prediction_count=sum(r["prediction_source"] != "missing" for r in rows),
                  split_method="diagnostic_per_mode_arrival_time_not_approved_for_calibration",
                  split_metadata=split_meta, split_coverage=coverage, hourly=hourly_stats,
                  quantile_method="linear interpolation: position=(n-1)*q", variance_method="sample variance, denominator n-1; n<2 -> null",
                  units={"latency": "ms", "tpot": "ms/token", "latency_variance": "ms^2", "tpot_variance": "(ms/token)^2"},
                  slo_thresholds=None, routing_contract=config["routing_contract"])
    return result, split_rows
