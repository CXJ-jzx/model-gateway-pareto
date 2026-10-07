"""Prepare ordinary historical traffic; scenario injection belongs to later experiments."""

import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .normalize import normalize_request
from .profiles import build_incoming_request, enrich_request, fit_profiles, validate_config
from .validation import file_hash, validate_dataset


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)}")


def dump_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default, allow_nan=False) + "\n")


def prepare_dataset(traffic_path: Path, config_path: Path, output: Path):
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    validate_config(config)
    # Keep previous runs recoverable. Caller selects a new output directory.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")
    rows, rejections, signatures = {}, [], {}
    input_count, duplicates, conflicts = 0, 0, 0
    with traffic_path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            input_count += 1
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("not_json_object")
                row = normalize_request(raw, line_number)
                signature = json.dumps(raw, sort_keys=True, ensure_ascii=True)
                if row.request_id in rows:
                    duplicates += 1
                    if signatures[row.request_id] != signature:
                        conflicts += 1
                    continue
                rows[row.request_id] = row
                signatures[row.request_id] = signature
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
                rejections.append({"source_line": line_number, "reason": str(exc)})
    ordered = sorted(rows.values(), key=lambda r: (r.arrived_at, r.request_id))
    if len(ordered) < 5:
        raise ValueError("At least five valid requests are required for chronological splitting")
    train_index = max(1, int(len(ordered) * config["train_fraction"]))
    calibration_index = int(len(ordered) * (config["train_fraction"] + config["calibration_fraction"]))
    if not train_index < calibration_index < len(ordered):
        raise ValueError("Data too small for configured split fractions")
    cutoff, test_cutoff = ordered[train_index].arrived_at, ordered[calibration_index].arrived_at
    training = [r for r in ordered if r.arrived_at < cutoff]
    if not training or test_cutoff <= cutoff:
        raise ValueError("Arrival timestamps do not support three distinct time ranges")
    profiles = fit_profiles(training, cutoff, config)
    incoming, prepared, enrich_rejections = [], [], []
    for request in ordered:
        split = "train" if request.arrived_at < cutoff else "calibration" if request.arrived_at < test_cutoff else "test"
        initial = build_incoming_request(request, profiles, config, split)
        incoming.append(initial)
        if split == "train":
            continue
        try:
            prepared.append(enrich_request(initial, profiles, config))
        except ValueError as exc:
            enrich_rejections.append({"request_id": request.request_id, "source_line": request.source_line, "reason": str(exc)})
    observations, attempt_flags = [], Counter()
    for request in ordered:
        for attempt in request.attempts:
            attempt_flags.update(attempt.flags)
            observation = attempt.observation(request.is_stream)
            if observation is not None:
                observations.append({**asdict(observation), "model_id": request.model_id,
                                     "request_id": request.request_id, "attempt_id": attempt.attempt_id,
                                     "finished_at": attempt.finished_at, "arrival_time": request.arrived_at})
    observations.sort(key=lambda r: (r["occurred_at"], r["attempt_id"]))
    coverage = {}
    for field in ("actual_input_tokens", "actual_output_tokens", "predicted_input_tokens", "predicted_output_tokens", "task_type"):
        count = sum(getattr(r, field) is not None for r in ordered)
        coverage[field] = {"known": count, "missing": len(ordered) - count, "ratio": count / len(ordered)}
    audit = {"input_nonblank_lines": input_count, "accepted_requests": len(ordered),
             "rejected_lines": len(rejections), "duplicate_requests": duplicates,
             "duplicate_conflicts": conflicts, "enrichment_rejected": len(enrich_rejections),
             "training_requests": len(training), "prepared_requests": len(prepared),
             "incoming_requests": len(incoming),
             "field_coverage": coverage, "request_flags": dict(Counter(flag for r in ordered for flag in r.flags)),
             "attempt_flags": dict(attempt_flags), "priority_distribution": dict(Counter(r["priority"] for r in prepared)),
             "slo_distribution": dict(Counter(r["slo"]["tier"] for r in prepared)),
             "workload_distribution": dict(Counter(r["workload"]["level"] for r in prepared)),
             "urgency_distribution": dict(Counter(r["urgency"]["level"] for r in prepared)),
             "prediction_sources": dict(Counter(r["token_predictions"][f"{axis}_source"] for r in prepared for axis in ("input", "output"))),
             "by_model_stream": dict(Counter(f"{r['model_id']}|{r['stream_type']}" for r in prepared))}
    output.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output / "canonical_requests.jsonl", (asdict(r) for r in ordered))
    dump_jsonl(output / "incoming_requests.jsonl", (asdict(r) for r in incoming))
    dump_jsonl(output / "requests.jsonl", prepared)
    dump_jsonl(output / "observations.jsonl", observations)
    dump_jsonl(output / "rejections.jsonl", rejections)
    dump_jsonl(output / "enrichment_rejections.jsonl", enrich_rejections)
    dump_json(output / "profiles.json", profiles)
    dump_json(output / "resolved_config.json", config)
    dump_json(output / "audit.json", audit)
    names = ["canonical_requests.jsonl", "incoming_requests.jsonl", "requests.jsonl", "observations.jsonl", "rejections.jsonl", "enrichment_rejections.jsonl", "profiles.json", "resolved_config.json", "audit.json"]
    manifest = {"schema_version": "1.0", "source_file": str(traffic_path.resolve()), "source_sha256": file_hash(traffic_path),
                "seed": config["seed"], "fit_cutoff": cutoff.isoformat(), "test_cutoff": test_cutoff.isoformat(),
                "prepared_requests": len(prepared), "observations": len(observations),
                "incoming_requests": len(incoming),
                "artifact_hashes": {name: file_hash(output / name) for name in names},
                "limitations": ["SLO and priority are experimental policies, not observed business labels",
                                "Missing predictions use frozen historical statistics, not per-request actual usage",
                                "Classification heaviness is unknown without task features",
                                "Performance metrics preserve source latency-proxy semantics",
                                "Original messages/prompts are absent; arrival metadata is reconstructed, not original payload",
                                "Warmup predictions use configured mocks; held-out missing predictions use training statistics"]}
    dump_json(output / "manifest.json", manifest)
    validation = validate_dataset(output)
    dump_json(output / "validation.json", validation)
    summary = f"""# 数据处理结果

- 原始非空行：{input_count}
- 接受请求：{len(ordered)}；拒绝：{len(rejections)}；重复：{duplicates}
- 训练请求：{len(training)}；增强请求：{len(prepared)}
- 初始请求：{len(incoming)}
- 增强拒绝：{len(enrich_rejections)}
- 完成后可见的观察：{len(observations)}
- 训练截止：{cutoff.isoformat()}
- 测试起点：{test_cutoff.isoformat()}
- 验证通过：{validation['passed']}

incoming_requests.jsonl是请求刚到达时的输入元数据，覆盖全部有效请求，不含Endpoint、状态码、耗时、重试和实际Token。
requests.jsonl是校准和测试区间的初始请求经过网关规则补充后的调度输入。
执行结果仅保留在canonical_requests.jsonl与observations.jsonl。
原日志没有messages或prompt，不能恢复原始请求正文；没有真实到达时预测的字段明确标为mock。
训练区间初始请求的缺失预测使用固定配置，不读取训练区间未来统计。
旧日志缺失预测时，使用截止前已完成训练样本的统计值；不会复制本请求实际输出。
三级SLO和优先级为可配置模拟标签。轻重来自预测Token，紧急程度来自时间余量。
本阶段不注入突发、故障等特殊场景。详细分布与缺失率见audit.json。
"""
    (output / "summary.md").write_text(summary, encoding="utf-8")
    return validation
