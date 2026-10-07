"""Run only our production selector against configurable, fixed state fixtures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from ... import RoutingEngine
from ...data_pipeline import write_candidates_csv, write_svg
from .scenarios import ScenarioCase, load_cases


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "selection_scenarios.json"


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                         if isinstance(value, (list, dict, tuple)) else value
                         for key, value in row.items()} for row in rows)


def snapshot(case: ScenarioCase) -> dict:
    """Capture the model-scoped decision inputs, not future request outcomes."""
    cutoff = case.cutoff or case.request.arrived_at
    offerings = case.state.catalog.candidates(case.request.model_id, at_time=cutoff, include_ineligible=True)
    return _jsonable(dict(
        case_id=case.case_id, description=case.description, provenance=case.provenance,
        request=asdict(case.request), cutoff=cutoff,
        policy=asdict(case.state.policies.get(case.request.model_id)),
        parameter_overrides=case.parameter_overrides,
        catalog_version=case.state.catalog.version,
        windows_retention_minutes=case.state.windows.max_age.total_seconds() / 60,
        endpoints=[dict(
            offering=asdict(offering),
            model_endpoint_state=asdict(case.state.catalog.runtime_state(case.request.model_id, offering.endpoint_id)),
            endpoint_health=case.state.catalog.endpoint_state(offering.endpoint_id).health_status,
            observations={"stream" if mode else "nonstream": [asdict(o) for o in case.state.windows.recent(
                case.request.model_id, offering.endpoint_id, mode, cutoff,
                int(case.state.windows.max_age.total_seconds() / 60))] for mode in (False, True)},
        ) for offering in sorted(offerings, key=lambda o: o.endpoint_id)],
    ))


def check_decision(decision) -> list[str]:
    """Recalculate objective/order invariants without production selection helpers."""
    errors = []

    def close(value, expected):
        return value is not None and math.isfinite(value) and math.isclose(
            value, expected, rel_tol=1e-10, abs_tol=1e-12
        )

    def stable_key(candidate):
        return (-candidate.stability_lower_bound, -(candidate.stability_success_rate or 0),
                -candidate.stability_sample_count,
                candidate.routing_score if candidate.routing_score is not None else math.inf,
                candidate.endpoint_id)

    by_id = {c.endpoint_id: c for c in decision.candidates}
    if len(by_id) != len(decision.candidates):
        errors.append("duplicate_candidate_endpoint")
    feasible = [c for c in decision.candidates if c.feasible and c.performance is not None]
    selected = [c for c in decision.candidates if c.selected]
    rho = decision.parameters.rho_input_price
    for candidate in decision.candidates:
        endpoint = candidate.endpoint_id
        input_cost = decision.request.predicted_input_tokens * candidate.input_per_million / 1_000_000
        output_cost = decision.request.predicted_output_tokens * candidate.output_per_million / 1_000_000
        total_cost = input_cost + output_cost
        unit_cost = rho * candidate.input_per_million + (1 - rho) * candidate.output_per_million
        objective_cost = {
            "predicted_request": total_cost,
            "weighted_predicted_request": rho * input_cost + (1 - rho) * output_cost,
            "unit_price": unit_cost,
        }.get(decision.cost_mode)
        for field_name, expected in (("estimated_input_cost", input_cost), ("estimated_output_cost", output_cost),
                                     ("estimated_request_cost", total_cost), ("unit_price_cost", unit_cost),
                                     ("cost_raw", objective_cost)):
            if expected is None or not close(getattr(candidate, field_name), expected):
                errors.append(f"cost:{endpoint}:{field_name}")
        if candidate.cost_mode != decision.cost_mode:
            errors.append(f"cost_mode:{endpoint}")
        rate, samples, z = candidate.stability_success_rate, candidate.stability_sample_count, decision.stability_z
        if samples < 0 or not math.isfinite(samples) or (rate is not None and not 0 <= rate <= 1):
            errors.append(f"stability_inputs:{endpoint}")
        elif rate is None or samples == 0:
            if not close(candidate.stability_lower_bound, 0):
                errors.append(f"stability_lower_bound:{endpoint}")
        else:
            z2 = z * z
            lower = (rate + z2 / (2 * samples)
                     - z * math.sqrt(rate * (1 - rate) / samples + z2 / (4 * samples * samples))) / (1 + z2 / samples)
            if not close(candidate.stability_lower_bound, max(0, min(1, lower))):
                errors.append(f"stability_lower_bound:{endpoint}")
        if candidate.feasible and decision.endpoint_checks[endpoint]["reasons"]:
            errors.append(f"feasible_has_exclusion_reason:{endpoint}")

    ordered = decision.ordered_endpoints
    if len(ordered) > decision.top_k or len(ordered) != len(set(ordered)):
        errors.append("ordered_endpoints_invalid")
    if decision.backup_endpoints != ordered[1:]:
        errors.append("backup_list_inconsistent")
    if decision.status == "no_candidate":
        if feasible or selected or decision.selected_endpoint is not None or ordered:
            errors.append("no_candidate_inconsistent")
        if decision.cost_reference is not None:
            errors.append("no_candidate_cost_reference")
        return errors
    if len(selected) != 1 or selected[0].endpoint_id != decision.selected_endpoint:
        return errors + ["selection_marker_inconsistent"]
    chosen = selected[0]
    if not ordered or ordered[0] != chosen.endpoint_id:
        errors.append("ordered_primary_inconsistent")
    positive = [c.cost_raw for c in feasible if c.cost_raw > 0]
    reference = median(positive) if positive else 1.0
    if not close(decision.cost_reference, reference):
        errors.append("cost_reference_inconsistent")
    frontier = []
    for candidate in decision.candidates:
        if not close(candidate.cost_normalized, candidate.cost_raw / reference):
            errors.append(f"normalization:{candidate.endpoint_id}")
        if not candidate.feasible:
            if candidate.pareto or candidate.routing_score is not None:
                errors.append(f"ineligible_scored:{candidate.endpoint_id}")
            continue
        score = (decision.parameters.lambda_cost * candidate.cost_raw / reference
                 + (1 - decision.parameters.lambda_cost) * candidate.performance)
        if not close(candidate.routing_score, score):
            errors.append(f"routing_score:{candidate.endpoint_id}")
        dominated = any(other.endpoint_id != candidate.endpoint_id and other.cost_raw <= candidate.cost_raw
                        and other.performance <= candidate.performance
                        and (other.cost_raw < candidate.cost_raw or other.performance < candidate.performance)
                        for other in feasible)
        if candidate.pareto == dominated:
            errors.append(f"pareto:{candidate.endpoint_id}")
        if not dominated:
            frontier.append(candidate)
            if not close(candidate.score, score):
                errors.append(f"score:{candidate.endpoint_id}")
        elif candidate.score is not None:
            errors.append(f"dominated_primary_score:{candidate.endpoint_id}")

    if decision.selection_phase == "primary":
        expected_primary = min(frontier, key=lambda c: (
            c.routing_score, -(c.success_rate or 0), -c.effective_sample_count, c.endpoint_id
        ))
        if chosen.endpoint_id != expected_primary.endpoint_id:
            errors.append("not_minimum_frontier_score")
    else:
        pool = feasible if decision.backup_pool == "all_feasible" else frontier
        if chosen.endpoint_id != min(pool, key=stable_key).endpoint_id:
            errors.append("not_stability_first_backup")
    remaining = [c for c in feasible if c.endpoint_id != chosen.endpoint_id
                 and (decision.backup_pool == "all_feasible" or c.pareto)]
    expected_order = [chosen.endpoint_id] + [c.endpoint_id for c in sorted(remaining, key=stable_key)[:decision.top_k - 1]]
    if ordered != expected_order:
        errors.append("backup_stability_order_inconsistent")
    for candidate in decision.candidates:
        expected_rank = ordered.index(candidate.endpoint_id) + 1 if candidate.endpoint_id in ordered else None
        expected_role = ("retry_backup" if decision.selection_phase == "backup" else "primary") if expected_rank == 1 else "backup" if expected_rank is not None else "none"
        if candidate.route_rank != expected_rank or candidate.route_role != expected_role:
            errors.append(f"route_rank_or_role:{candidate.endpoint_id}")
    for endpoint in ordered:
        candidate = by_id.get(endpoint)
        if candidate is None or not candidate.feasible or decision.endpoint_checks[endpoint]["reasons"]:
            errors.append(f"ordered_ineligible:{endpoint}")
            continue
        for name, capacity in decision.endpoint_checks[endpoint]["capacity"].items():
            increment = decision.request.predicted_input_tokens + decision.request.predicted_output_tokens if name == "tpm" else 1
            if not close(capacity["increment"], increment) or not close(capacity["projected"], capacity["current"] + increment):
                errors.append(f"capacity_arithmetic:{endpoint}:{name}")
            limit = capacity["limit"]
            if ((limit is None and decision.unknown_capacity_policy == "block")
                    or (limit is not None and (limit == 0 or capacity["projected"] > limit))):
                errors.append(f"ordered_capacity_violation:{endpoint}:{name}")
        for name, metric in decision.endpoint_checks[endpoint]["slo"].items():
            if decision.require_slo and (metric["estimated_ms"] is None or metric["budget_ms"] <= 0
                                         or metric["estimated_ms"] > metric["budget_ms"]):
                errors.append(f"ordered_slo_violation:{endpoint}:{name}")
    return errors


def _quantile(values: list[float], q: float) -> float:
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def run_suite(config_path: Path, output_dir: Path, *, dataset_dir: Path | None = None,
              benchmark_repeats: int = 30) -> dict:
    if isinstance(benchmark_repeats, bool) or not isinstance(benchmark_repeats, int) or benchmark_repeats < 1:
        raise ValueError("benchmark_repeats must be a positive integer")
    dataset_metadata = None
    requests_hash = _digest(dataset_dir / "requests.jsonl") if dataset_dir else None
    if dataset_dir and (dataset_dir / "manifest.json").exists():
        dataset_metadata = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        expected_hash = dataset_metadata.get("artifact_hashes", {}).get("requests.jsonl")
        if expected_hash is not None and expected_hash != requests_hash:
            raise ValueError("Prepared requests.jsonl hash differs from its manifest; regenerate or validate the dataset")
    cases = load_cases(config_path, dataset_dir)
    if not cases:
        raise ValueError("No eligible scenario cases")
    # Refuse overwrites so the original evidence and old experiments remain intact.
    output_dir.mkdir(parents=True, exist_ok=False)
    files = [ROOT / "models.py", ROOT / "memory_state.py", ROOT / "routing_engine.py", ROOT / "request_routing.py",
             ROOT / "pricing.py", ROOT / "failover.py", ROOT / "data_pipeline.py",
             Path(__file__), Path(__file__).with_name("scenarios.py")]
    manifest = dict(
        schema_version="1.0", started_at=datetime.now(timezone.utc).isoformat(),
        strategy="request_aware_pareto_linear", other_strategies_connected=False,
        endpoint_execution_simulated=False, config_file=str(config_path.resolve()), config_sha256=_digest(config_path),
        backup_dispatch_performed=False, backup_ranking="stability_first_after_primary",
        cost_modes=sorted({case.parameter_overrides.get("cost_mode", "predicted_request") for case in cases}),
        dataset_dir=str(dataset_dir.resolve()) if dataset_dir else None,
        requests_sha256=requests_hash,
        dataset_manifest_sha256=_digest(dataset_dir / "manifest.json") if dataset_metadata else None,
        dataset_requests_hash_verified=bool(dataset_metadata and dataset_metadata.get("artifact_hashes", {}).get("requests.jsonl") == requests_hash),
        python=platform.python_version(), platform=platform.platform(), benchmark_repeats=benchmark_repeats,
        code_sha256={path.name: _digest(path) for path in files}, cases=[case.case_id for case in cases],
        timing_scope="in-memory production routing only; excludes fixture loading, verification, SVG and file I/O",
    )
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "resolved_config.json", json.loads(config_path.read_text(encoding="utf-8")))
    if dataset_metadata:
        _write_json(output_dir / "source_dataset_manifest.json", dataset_metadata)
    rows, validations = [], []
    for case in cases:
        case_dir = output_dir / case.case_id
        case_dir.mkdir()
        before = snapshot(case)
        engine = RoutingEngine(case.state)
        def route():
            return engine.route_request(case.request, cutoff=case.cutoff, **case.parameter_overrides)
        decision = route()
        errors = check_decision(decision)
        # Warmups are excluded. Repeats do not dispatch or cumulatively consume capacity.
        for _ in range(3):
            route()
        timings = []
        behavior = _fingerprint(dict(selected=decision.selected_endpoint, ordered=decision.ordered_endpoints,
                                     backups=decision.backup_endpoints, candidates=[asdict(c) for c in decision.candidates],
                                     checks=decision.endpoint_checks))
        for i in range(benchmark_repeats):
            repeated = route()
            if _fingerprint(dict(selected=repeated.selected_endpoint, ordered=repeated.ordered_endpoints,
                                 backups=repeated.backup_endpoints, candidates=[asdict(c) for c in repeated.candidates],
                                 checks=repeated.endpoint_checks)) != behavior:
                errors.append("nondeterministic_repeat")
            timings.append(dict(repeat=i + 1, **repeated.timings_us))
        if _fingerprint(before) != _fingerprint(snapshot(case)):
            errors.append("routing_mutated_state")
        payload = decision.to_dict()
        payload.update(case_id=case.case_id, description=case.description, provenance=case.provenance,
                       validation=dict(passed=not errors, errors=sorted(set(errors))), input_snapshot_sha256=_fingerprint(before))
        _write_json(case_dir / "snapshot.json", before)
        _write_json(case_dir / "decision.json", payload)
        _write_csv(case_dir / "benchmark.csv", timings)
        write_candidates_csv(case_dir / "candidates.csv", decision.candidates)
        chosen = next((c for c in decision.candidates if c.selected), None)
        write_svg(case_dir / "pareto.svg", case.request.model_id, decision.candidates, chosen, decision.parameters)
        latencies = [t["route_total"] for t in timings]
        rows.append(dict(
            case_id=case.case_id, model_id=case.request.model_id, mode="stream" if case.request.is_stream else "nonstream",
            priority=case.request.priority, slo_tier=case.request.slo.tier, workload=case.request.workload.get("level"),
            lambda_cost=decision.parameters.lambda_cost, eta_ttft=decision.parameters.eta_ttft,
            rho_input_price=decision.parameters.rho_input_price, half_life_minutes=decision.parameters.half_life_minutes,
            cost_mode=decision.cost_mode, top_k=decision.top_k, backup_pool=decision.backup_pool,
            selection_phase=decision.selection_phase, ordered_endpoints=decision.ordered_endpoints,
            backup_endpoints=decision.backup_endpoints,
            status=decision.status, selected_endpoint=decision.selected_endpoint,
            candidate_count=len(decision.endpoint_checks), feasible_count=sum(c.feasible for c in decision.candidates),
            pareto_count=sum(c.pareto for c in decision.candidates), selected_score=chosen.routing_score if chosen else None,
            selected_frontier_score=chosen.score if chosen else None,
            selected_estimated_input_cost=chosen.estimated_input_cost if chosen else None,
            selected_estimated_output_cost=chosen.estimated_output_cost if chosen else None,
            selected_estimated_request_cost=chosen.estimated_request_cost if chosen else None,
            selected_currency=chosen.currency if chosen else None,
            selected_stability_lower_bound=chosen.stability_lower_bound if chosen else None,
            selected_stability_success_rate=chosen.stability_success_rate if chosen else None,
            selected_stability_samples=chosen.stability_sample_count if chosen else None,
            ordered_endpoint_metrics={c.endpoint_id: dict(
                route_rank=c.route_rank, route_role=c.route_role, estimated_request_cost=c.estimated_request_cost,
                stability_lower_bound=c.stability_lower_bound, stability_success_rate=c.stability_success_rate,
                stability_sample_count=c.stability_sample_count, routing_score=c.routing_score,
            ) for c in decision.candidates if c.route_rank is not None},
            route_once_us=decision.timings_us["route_total"], benchmark_mean_us=mean(latencies),
            benchmark_p50_us=_quantile(latencies, .5), benchmark_p95_us=_quantile(latencies, .95),
            benchmark_p99_us=_quantile(latencies, .99), validation_passed=not errors,
        ))
        validations.append(dict(case_id=case.case_id, passed=not errors, errors=sorted(set(errors))))
    summary = dict(
        passed=all(v["passed"] for v in validations), case_count=len(rows),
        selected_count=sum(r["status"] == "selected" for r in rows),
        no_candidate_count=sum(r["status"] == "no_candidate" for r in rows),
        validations=validations, cases=rows,
        cost_modes=sorted({r["cost_mode"] for r in rows}),
        decisions_with_backup_count=sum(bool(r["backup_endpoints"]) for r in rows),
        scope="Rule-following and overhead only; no baselines, no causal claims about latency or savings",
    )
    _write_csv(output_dir / "decisions.csv", rows)
    _write_json(output_dir / "summary.json", summary)
    markdown = ["# 我们的 Endpoint 选择策略：场景验证", "",
                f"验证{'通过' if summary['passed'] else '失败'}：{len(rows)} 个场景；{summary['selected_count']} 个选中，{summary['no_candidate_count']} 个无可用候选。", "",
                "仅运行正式 RoutingEngine.route_request，无其他策略，无 Endpoint 执行模拟器。状态固定，重复调用不累积请求负载。", "",
                "首选按成本/性能评分；至多两个备用按稳定性代理排序。序列只是路由计划，不表示已经发送或累积支付备用费用。", "",
                "| 场景 | 模型 / 模式 | 成本模式 | 有序候选（首项为本阶段选择） | 选择点预测总费用 | 稳定性代理下界 | 状态 | 开销 P50 / P95 (μs) | 验证 |",
                "|---|---|---|---|---:|---:|---|---:|---|"]
    for row in rows:
        cost_display = f"{row['selected_estimated_request_cost']:.8g} {row['selected_currency']}" if row["selected_estimated_request_cost"] is not None else "—"
        stability_display = f"{row['selected_stability_lower_bound']:.4f}" if row["selected_stability_lower_bound"] is not None else "—"
        route_display = " → ".join(row["ordered_endpoints"]) or "—"
        markdown.append(f"| [{row['case_id']}]({row['case_id']}/decision.json) | {row['model_id']} / {row['mode']} | {row['cost_mode']} | {route_display} | {cost_display} | {stability_display} | {row['status']} | {row['benchmark_p50_us']:.1f} / {row['benchmark_p95_us']:.1f} | {'通过' if row['validation_passed'] else '失败'} |")
    markdown += ["", "检查范围：全部有序候选的硬容量与SLO约束、成本各分项和三种成本模式、二维Pareto、首选最小评分、备用稳定性排序与rank/role、重复序列一致、状态不被路由修改。", "",
                 "默认predicted_request费用=(预测输入Tokens×输入单价+预测输出Tokens×输出单价)/100万；unit_price和weighted_predicted_request是显式偏好目标，后者带ρ而不是实际账单。各候选预测总费用始终另外记录，不能把所有备用费用相加当成已执行请求费用。", "",
                 "稳定性下界是基于加权有效样本的Wilson式保守排序代理，不是经校准的置信保证；备用可以被成本/性能二维支配，但必须满足同样硬约束。", "",
                 "每个场景保存输入状态、决策明细、候选表、Pareto图和分阶段开销。没有价格或性能数据的点不能绘制，其原因仍记录在decision.json。", "",
                 "这是人工状态下的规则验证，不是请求真实SLO达标率、实际账单节省或相对其他策略优势的证明。P99在少量重复下仅作参考；绘图和文件读写不计入路由开销。"]
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Only our strategy: request-aware endpoint selection scenarios")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, help="Prepared dataset containing arrival-only requests.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    args = parser.parse_args(argv)
    output = args.output_dir or DEFAULT_CONFIG.parent.parent / "output" / f"selection_{datetime.now():%Y%m%d_%H%M%S_%f}"
    try:
        summary = run_suite(args.config, output, dataset_dir=args.dataset_dir, benchmark_repeats=args.benchmark_repeats)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Scenario experiment failed: {exc}\n")
    print(json.dumps(dict(passed=summary["passed"], case_count=summary["case_count"], output_dir=str(output.resolve())), ensure_ascii=False))
    return 0 if summary["passed"] else 1
