"""Shared experiment I/O and historical loaders; arrival data preparation lives in data/pipeline.py."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Sequence

try:
    from .data.normalize import iter_jsonl, normalize_request, parse_time
    from .memory_state import InMemoryRoutingState
    from .models import Candidate, EndpointOffering, Observation, Price, Prior, StrategyParameters
    from .routing_engine import combine_stream_performance, compute_candidate, finalize_selection, weighted_quantile
except ImportError:  # 兼容直接运行目录内脚本
    from endpoint_routing_strategy.data.normalize import iter_jsonl, normalize_request, parse_time
    from memory_state import InMemoryRoutingState
    from models import Candidate, EndpointOffering, Observation, Price, Prior, StrategyParameters
    from routing_engine import combine_stream_performance, compute_candidate, finalize_selection, weighted_quantile


def resolve_data_files(data_dir: Path) -> tuple[Path, Path, Path]:
    roles = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        first = next(iter_jsonl(path), {})
        role = ("config" if "models" in first and "endpoint_id" in first else
                "snapshot" if "snapshot_id" in first else
                "traffic" if "request_id" in first and "target_model" in first else None)
        if role:
            if role in roles:
                raise ValueError(f"Ambiguous {role} files: {roles[role]}, {path}")
            roles[role] = path
    missing = {"config", "traffic", "snapshot"} - roles.keys()
    if missing:
        raise FileNotFoundError(f"{data_dir}: missing data roles {sorted(missing)}")
    return roles["config"], roles["traffic"], roles["snapshot"]


def load_offerings(config_path: Path) -> list[EndpointOffering]:
    offerings = []
    for endpoint in iter_jsonl(config_path):
        for model in endpoint.get("models", []):
            offerings.append(
                EndpointOffering(
                    endpoint_id=endpoint["endpoint_id"],
                    model_id=model["model_id"],
                    deployment_type=endpoint.get("deployment_type", "unknown"),
                    price_config=dict(model.get("price_config", {})),
                    capacity_config=dict(model.get("capacity_config", {})),
                )
            )
    return offerings


def bootstrap_memory_state(config_path: Path, max_window_minutes: int = 360) -> InMemoryRoutingState:
    state = InMemoryRoutingState(max_window_minutes=max_window_minutes)
    for offering in load_offerings(config_path):
        state.register_endpoint(offering)
    return state


def load_observations_and_latest_history(
    traffic_path: Path,
    model_id: str,
    history_cutoff: datetime | None = None,
) -> tuple[list[Observation], datetime, dict[str, dict[str, str]]]:
    observations = []
    latest_time = None
    latest_history_time = None
    latest_refs = {}
    for request in iter_jsonl(traffic_path):
        if request.get("target_model") != model_id:
            continue
        historical = request.get("historical_performance", {})
        occurred_at = parse_time(historical.get("request_arrival_at"))
        if occurred_at is None:
            continue
        if latest_time is None or occurred_at > latest_time:
            latest_time = occurred_at
        if (
            (history_cutoff is None or occurred_at <= history_cutoff)
            and (latest_history_time is None or occurred_at > latest_history_time)
        ):
            latest_history_time = occurred_at
            latest_refs = {
                item["endpoint_id"]: dict(item.get("endpoint_model", {}))
                for item in historical.get("endpoint_histories", [])
            }
        try:
            canonical = normalize_request(request)
        except (ValueError, TypeError):
            continue
        for attempt in canonical.attempts:
            observation = attempt.observation(canonical.is_stream)
            if observation is not None:
                observations.append(observation)
                latest_time = max(latest_time, observation.occurred_at)
    if latest_time is None:
        raise ValueError(f"流量文件中没有模型{model_id}的请求")
    observations.sort(key=lambda item: item.occurred_at)
    return observations, latest_time, latest_refs


def load_priors(snapshot_path: Path, refs: dict[str, dict[str, str]]) -> dict[str, Prior]:
    needed = {}
    for endpoint_id, windows in refs.items():
        for window_name in ("3d", "7d"):
            snapshot_id = windows.get(window_name)
            if snapshot_id:
                needed[snapshot_id] = (endpoint_id, window_name)
    if not needed:
        return {}
    found = {}
    for snapshot in iter_jsonl(snapshot_path):
        snapshot_id = snapshot.get("snapshot_id")
        if snapshot_id not in needed:
            continue
        endpoint_id, window_name = needed[snapshot_id]
        found.setdefault(endpoint_id, {})[window_name] = snapshot
        if sum(len(value) for value in found.values()) == len(needed):
            break
    weights = {"3d": 0.7, "7d": 0.3}
    priors = {}
    for endpoint_id, windows in found.items():
        def metric(name: str) -> float | None:
            values = [
                (float(snapshot["metrics"][name]), weights[window])
                for window, snapshot in windows.items()
                if snapshot.get("has_observations") and snapshot.get("metrics", {}).get(name) is not None
            ]
            if not values:
                return None
            total = sum(weight for _, weight in values)
            return sum(value * weight for value, weight in values) / total

        tps = metric("output_tps_e2e_success")
        counts = [
            (float(snapshot.get("counts", {}).get("log_records", 0)), weights[window])
            for window, snapshot in windows.items() if snapshot.get("has_observations")
        ]
        sample_count = (
            sum(value * weight for value, weight in counts) / sum(weight for _, weight in counts)
            if counts else 0.0
        )
        priors[endpoint_id] = Prior(
            endpoint_id=endpoint_id,
            e2e_p95_ms=metric("p95_success_latency_ms"),
            ttft_p95_ms=metric("p95_success_stream_first_response_ms"),
            tpot_proxy_ms=1000.0 / tps if tps and tps > 0 else None,
            success_rate=metric("success_rate"),
            sample_count=sample_count,
            source="0.7×3d + 0.3×7d",
        )
    return priors


def write_candidates_csv(path: Path, candidates: Sequence[Candidate]) -> None:
    fields = list(asdict(candidates[0]).keys()) if candidates else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in candidates)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_svg(
    path: Path,
    model_id: str,
    candidates: Sequence[Candidate],
    selected: Candidate | None,
    params: StrategyParameters,
) -> None:
    width, height = 1000, 680
    cost_mode = candidates[0].cost_mode if candidates else "predicted_request"
    cost_label = {"unit_price": "归一化单位价格", "predicted_request": "预计单次请求费用 / 参考费用",
                  "weighted_predicted_request": "偏好加权请求成本 / 参考成本"}.get(cost_mode, "归一化成本")
    left, right, top, bottom = 110, 70, 90, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    plotted = [item for item in candidates if item.cost_normalized is not None and item.performance is not None]
    x_max = max([item.cost_normalized for item in plotted] + [1.0]) * 1.15
    y_max = max([item.performance for item in plotted] + [1.0]) * 1.15
    sx = lambda value: left + value / x_max * plot_w
    sy = lambda value: top + plot_h - value / y_max * plot_h
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Microsoft YaHei",sans-serif;fill:#202124}.axis{stroke:#374151;stroke-width:2}.grid{stroke:#e5e7eb}.frontier{stroke:#2563eb;stroke-width:3;fill:none}.decision{stroke:#7c3aed;stroke-width:2;stroke-dasharray:8 6}.small{font-size:12px;fill:#4b5563}</style>',
        f'<text x="{left}" y="42" font-size="24" font-weight="700">{escape(model_id)} 成本-性能 Pareto 路由</text>',
        f'<text x="{left}" y="67" class="small">橙色：选中；绿色：备用；蓝色：前沿；灰色：其他；红色：不可用。λ={params.lambda_cost:g}，η={params.eta_ttft:g}；{escape(cost_mode)}</text>',
    ]
    for tick in range(6):
        xv, yv = x_max * tick / 5, y_max * tick / 5
        x, y = sx(xv), sy(yv)
        lines += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>',
            f'<text x="{x:.1f}" y="{top+plot_h+28}" text-anchor="middle" class="small">{xv:.2f}</text>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" class="small">{yv:.2f}</text>',
        ]
    lines += [
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
        f'<text x="{left+plot_w/2}" y="{height-35}" text-anchor="middle">{cost_label}</text>',
        f'<text x="28" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 28 {top+plot_h/2})">性能惩罚 / SLO</text>',
    ]
    frontier = sorted([item for item in plotted if item.pareto], key=lambda item: item.cost_normalized or 0)
    if len(frontier) > 1:
        points = " ".join(f"{sx(item.cost_normalized or 0):.1f},{sy(item.performance or 0):.1f}" for item in frontier)
        lines.append(f'<polyline points="{points}" class="frontier"/>')
    if selected is not None and selected.score is not None:
        lam = params.lambda_cost
        if 0 < lam < 1:
            intersections = []
            for x in (0.0, x_max):
                y = (selected.score - lam * x) / (1 - lam)
                if 0 <= y <= y_max:
                    intersections.append((x, y))
            for y in (0.0, y_max):
                x = (selected.score - (1 - lam) * y) / lam
                if 0 <= x <= x_max:
                    intersections.append((x, y))
            if len(intersections) >= 2:
                a, b = intersections[:2]
                lines.append(f'<line x1="{sx(a[0]):.1f}" y1="{sy(a[1]):.1f}" x2="{sx(b[0]):.1f}" y2="{sy(b[1]):.1f}" class="decision"/>')
        elif lam == 0:
            lines.append(f'<line x1="{left}" y1="{sy(selected.performance or 0):.1f}" x2="{left+plot_w}" y2="{sy(selected.performance or 0):.1f}" class="decision"/>')
        elif lam == 1:
            lines.append(f'<line x1="{sx(selected.cost_normalized or 0):.1f}" y1="{top}" x2="{sx(selected.cost_normalized or 0):.1f}" y2="{top+plot_h}" class="decision"/>')
    if selected is None:
        lines.append(f'<text x="{left}" y="{height-10}" class="small">没有满足约束的 Endpoint；交给后续调度模块，不自动放宽 SLO。</text>')
    for item in plotted:
        x, y = sx(item.cost_normalized or 0), sy(item.performance or 0)
        color, radius = (("#ef4444", 7) if not item.feasible else
                         (("#f59e0b", 11) if item.selected else
                          (("#16a34a", 10) if item.route_rank is not None else (("#3b82f6", 9) if item.pareto else ("#9ca3af", 7)))))
        rank_label = f" [#{item.route_rank}]" if item.route_rank is not None else ""
        lines += [
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"><title>{escape(item.exclusion_reason or "可用")}</title></circle>',
            f'<text x="{x+12:.1f}" y="{y-8:.1f}">{escape(item.endpoint_id + rank_label)}</text>',
            f'<text x="{x+12:.1f}" y="{y+10:.1f}" class="small">C={item.cost_raw:.4g}, P={item.performance:.3f}</text>',
        ]
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def write_route_outputs(
    output_dir: Path,
    model_id: str,
    cutoff: datetime,
    params: StrategyParameters,
    candidates: Sequence[Candidate],
    selected: Candidate,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_candidates_csv(output_dir / "candidates.csv", candidates)
    payload = {
        "model_id": model_id,
        "mode": "stream" if params.stream else "nonstream",
        "cutoff": cutoff.isoformat(),
        "parameters": asdict(params),
        "selected_endpoint": selected.endpoint_id,
        "selected_score": selected.score,
        "candidates": [asdict(item) for item in candidates],
    }
    (output_dir / "routing_decision.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(output_dir / "cost_performance.svg", model_id, candidates, selected, params)
    rows = sorted(candidates, key=lambda item: (not item.selected, not item.pareto, item.endpoint_id))
    markdown = [
        f"# {model_id} 路由决策结果", "", f"- 模式：{'流式' if params.stream else '非流式'}",
        f"- 截止时间：{cutoff.isoformat()}", f"- 最终选择：**{selected.endpoint_id}**", "",
        "| Endpoint | 成本 | 性能 | 成功率 | 有效样本 | 窗口分钟 | 历史权重 | Pareto | 得分 | 选择概率 |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for item in rows:
        markdown.append(
            f"| {item.endpoint_id} | {_fmt(item.cost_raw)} | {_fmt(item.performance)} | {_fmt(item.success_rate)} | "
            f"{_fmt(item.effective_sample_count,1)} | {item.window_minutes} | {_fmt(item.prior_weight,3)} | "
            f"{'是' if item.pareto else '否'} | {_fmt(item.score)} | {_fmt(item.selection_probability,3)} |"
        )
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")


def future_performance(
    observations: Sequence[Observation], endpoint_id: str, start: datetime, end: datetime,
    params: StrategyParameters, min_samples: int,
) -> float | None:
    future = [item for item in observations if item.endpoint_id == endpoint_id and item.is_stream == params.stream and start < item.occurred_at <= end and item.success]
    if params.stream:
        ttft = [item.ttft_ms for item in future if item.ttft_ms is not None]
        tpot = [item.tpot_ms for item in future if item.tpot_ms is not None]
        if max(len(ttft), len(tpot)) < min_samples:
            return None
        return combine_stream_performance(
            weighted_quantile([(value, 1) for value in ttft], .95),
            weighted_quantile([(value, 1) for value in tpot], .95),
            params.eta_ttft, params.slo_ttft_ms, params.slo_tpot_ms,
        )
    e2e = [item.e2e_ms for item in future if item.e2e_ms is not None]
    if len(e2e) < min_samples:
        return None
    value = weighted_quantile([(item, 1) for item in e2e], .95)
    return value / params.slo_e2e_ms if value is not None else None


def run_window_sweep(
    prices: dict[str, Price], observations: Sequence[Observation], base_params: StrategyParameters,
    output_dir: Path, windows: Sequence[int], half_lives: Sequence[float],
    horizon_minutes: int, step_minutes: int, min_future_samples: int,
) -> list[dict]:
    mode_obs = [item for item in observations if item.is_stream == base_params.stream]
    start = mode_obs[0].occurred_at + timedelta(minutes=max(windows))
    end = mode_obs[-1].occurred_at - timedelta(minutes=horizon_minutes)
    times, cursor = [], start
    while cursor <= end:
        times.append(cursor); cursor += timedelta(minutes=step_minutes)
    results = []
    for window in windows:
        for half_life in half_lives:
            params = StrategyParameters(**{**asdict(base_params), "window_candidates_minutes": (window,), "half_life_minutes": half_life, "prior_strength": 0, "min_success_rate": 0})
            errors, regrets, rankings, choices = [], [], [], []
            for cutoff in times:
                actual = {endpoint: future_performance(observations, endpoint, cutoff, cutoff + timedelta(minutes=horizon_minutes), params, min_future_samples) for endpoint in prices}
                actual = {key: value for key, value in actual.items() if value is not None}
                if len(actual) < 2:
                    continue
                candidates = []
                for endpoint in actual:
                    endpoint_obs = [item for item in observations if item.endpoint_id == endpoint and item.occurred_at <= cutoff]
                    candidate = compute_candidate(prices[endpoint], endpoint_obs, cutoff, None, params)
                    if candidate.performance is not None:
                        candidates.append(candidate)
                if len(candidates) < 2:
                    continue
                try:
                    selected = finalize_selection(candidates, params)
                except ValueError:
                    continue
                choices.append(selected.endpoint_id)
                errors += [abs(item.performance - actual[item.endpoint_id]) for item in candidates if item.endpoint_id in actual and item.performance is not None]
                regrets.append(actual[selected.endpoint_id] - min(actual.values()))
                pairs = [(a, b) for i, a in enumerate(candidates) for b in candidates[i+1:] if a.endpoint_id in actual and b.endpoint_id in actual]
                if pairs:
                    rankings.append(sum(((a.performance or 0) <= (b.performance or 0)) == (actual[a.endpoint_id] <= actual[b.endpoint_id]) for a, b in pairs) / len(pairs))
            switches = sum(a != b for a, b in zip(choices, choices[1:]))
            results.append({
                "window_minutes": window, "half_life_minutes": half_life,
                "evaluated_times": len(regrets),
                "prediction_mae": sum(errors)/len(errors) if errors else None,
                "mean_regret": sum(regrets)/len(regrets) if regrets else None,
                "ranking_accuracy": sum(rankings)/len(rankings) if rankings else None,
                "switch_rate": switches/max(1, len(choices)-1),
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "window_sweep.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    ranked = sorted([item for item in results if item["mean_regret"] is not None], key=lambda item: (item["mean_regret"], item["prediction_mae"], item["switch_rate"]))
    payload = {"note": "仅用于窗口调参，不构成无偏反事实评估。", "best_by_regret": ranked[0] if ranked else None, "all_results": results}
    (output_dir / "window_sweep.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
