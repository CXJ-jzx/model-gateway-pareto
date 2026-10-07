"""A tiny failure-feedback example, not an endpoint execution simulator."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter_ns

from ... import AttemptResult, RoutingEngine
from ...data_pipeline import write_svg
from .runner import DEFAULT_CONFIG, ROOT, _digest, _write_json, check_decision, snapshot
from .scenarios import load_cases


def run_demo(config_path: Path, output_dir: Path, *, dataset_dir: Path | None = None,
             case_id: str = "stable_dominated_backup") -> dict:
    cases = load_cases(config_path, dataset_dir)
    case = next((item for item in cases if item.case_id == case_id), None)
    if case is None:
        raise ValueError(f"No eligible scenario {case_id}")
    before = snapshot(case)
    engine = RoutingEngine(case.state)
    options = {key: value for key, value in case.parameter_overrides.items()
               if key not in {"selection_phase", "exclude_endpoints", "allowed_endpoints"}}
    now = case.cutoff or case.request.arrived_at
    timings = {}
    started = perf_counter_ns()
    session = engine.start_failover(case.request, cutoff=now, retry_safe=True, **options)
    timings["initial_plan_us"] = (perf_counter_ns() - started) / 1000
    if len(session.route_plan) < 2:
        raise ValueError("The demo requires at least one feasible backup")
    started = perf_counter_ns()
    first = session.next_attempt(cutoff=now)
    timings["first_recheck_us"] = (perf_counter_ns() - started) / 1000
    if first is None:
        raise ValueError("The primary became infeasible")
    failure_time = now + timedelta(milliseconds=10)
    started = perf_counter_ns()
    session.complete_attempt(AttemptResult(False, 503, e2e_ms=10), cutoff=failure_time)
    timings["failure_feedback_us"] = (perf_counter_ns() - started) / 1000
    started = perf_counter_ns()
    second = session.next_attempt(cutoff=now + timedelta(milliseconds=20))
    timings["backup_recheck_and_select_us"] = (perf_counter_ns() - started) / 1000
    if second is None:
        raise ValueError(f"No backup after the failure: {session.stop_reason}")
    candidate = next(item for item in second.decision.candidates if item.selected)
    # Mock outcome values are explicitly copied from the estimate only to show
    # the feedback API. They are NOT independent measurements of SLO success.
    duration = candidate.estimated_e2e_ms or 100.0
    finish_time = now + timedelta(milliseconds=20 + duration)
    started = perf_counter_ns()
    session.complete_attempt(AttemptResult(
        True, 200, e2e_ms=duration,
        ttft_ms=candidate.estimated_ttft_ms if case.request.is_stream else None,
        tpot_ms=candidate.estimated_tpot_ms if case.request.is_stream else None,
    ), cutoff=finish_time)
    timings["success_feedback_us"] = (perf_counter_ns() - started) / 1000
    errors = check_decision(session.initial_decision) + check_decision(second.decision)
    after = snapshot(replace(case, cutoff=finish_time))
    before_runtime = {item["offering"]["endpoint_id"]: item["model_endpoint_state"] for item in before["endpoints"]}
    after_runtime = {item["offering"]["endpoint_id"]: item["model_endpoint_state"] for item in after["endpoints"]}
    for endpoint, state in before_runtime.items():
        for field in ("current_rpm", "current_tpm", "current_concurrency"):
            if state[field] != after_runtime[endpoint][field]:
                errors.append(f"Load unexpectedly mutated: {endpoint}/{field}")
    if first.endpoint_id != session.route_plan[0]:
        errors.append("First attempt did not match the planned primary")
    if second.endpoint_id not in session.route_plan[1:] or second.decision.selection_phase != "backup":
        errors.append("Failure did not switch to a planned stability-first backup")
    if session.status != "succeeded":
        errors.append("Session did not reach the success terminal state")
    report = dict(
        passed=not errors, errors=errors, case_id=case_id,
        model_id=case.request.model_id, stream_type="stream" if case.request.is_stream else "nonstream",
        route_plan=list(session.route_plan), attempted_endpoints=[first.endpoint_id, second.endpoint_id],
        status=session.status, stop_reason=session.stop_reason,
        estimated_cumulative_cost=session.estimated_cumulative_cost,
        timings_us=timings, actual_http_requests_sent=0,
        outcome_source="explicit_mock_503_then_200_not_a_real_performance_measurement",
        capacity_reserved=False,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "input_snapshot.json", before)
    _write_json(output_dir / "final_snapshot.json", after)
    _write_json(output_dir / "session.json", session.to_dict())
    _write_json(output_dir / "summary.json", report)
    _write_json(output_dir / "manifest.json", dict(
        created_at=datetime.now(timezone.utc).isoformat(), config_sha256=_digest(config_path),
        case_id=case_id, provenance=case.provenance,
        code_sha256={name: _digest(ROOT / name) for name in
                     ("models.py", "memory_state.py", "routing_engine.py", "request_routing.py", "pricing.py", "failover.py", "data_pipeline.py")},
        demo_sha256=_digest(Path(__file__)), actual_http_requests_sent=0,
    ))
    write_svg(output_dir / "primary_pareto.svg", case.request.model_id, session.initial_decision.candidates,
              next(item for item in session.initial_decision.candidates if item.selected),
              session.initial_decision.parameters)
    write_svg(output_dir / "backup_pareto.svg", case.request.model_id, second.decision.candidates, candidate, second.decision.parameters)
    (output_dir / "summary.md").write_text(
        "# 失败切换规则演示\n\n"
        f"- 模型：{case.request.model_id}；验证通过：{report['passed']}\n"
        f"- 初始计划：{' → '.join(session.route_plan)}\n"
        f"- 实际调用规划：{first.endpoint_id}（Mock 503）→ {second.endpoint_id}（Mock 200）\n"
        f"- 最终状态：{session.status}；预测尝试费用合计：{session.estimated_cumulative_cost:g} CNY\n\n"
        "首选失败后，写入失败记录及组合冷却时间；扣除已耗时预算，重新检查容量、健康和SLO，再按稳定性选择备用。\n\n"
        "本实验仅注入两条结果来验证控制逻辑，没有发送HTTP，不建设Endpoint模拟器。成功延迟是明确Mock，不用于证明真实SLO达标率。"
        "费用是已规划尝试的完整预测费用之和，不是实际账单；未使用的第三候选不计费。"
        "容量没有原子预占，实际调用和负载计数由网关负责。分阶段开销见summary.json；两张SVG展示首选与重选状态。\n",
        encoding="utf-8",
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mock one failure and validate stability-first failover")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--case-id", default="stable_dominated_backup")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or DEFAULT_CONFIG.parent.parent / "output" / f"failover_{datetime.now():%Y%m%d_%H%M%S_%f}"
    try:
        report = run_demo(args.config, output, dataset_dir=args.dataset_dir, case_id=args.case_id)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Failover demo failed: {exc}\n")
    print(json.dumps(dict(passed=report["passed"], route_plan=report["route_plan"],
                          attempted_endpoints=report["attempted_endpoints"], output_dir=str(output.resolve())), ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
