"""Small pressure-rule fixtures; not an endpoint simulator or latency benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from .. import BusyThresholds, EndpointOffering, InMemoryRoutingState


def run_checks(thresholds: BusyThresholds) -> dict:
    now = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
    state = InMemoryRoutingState(busy_thresholds=thresholds)
    for model, ep in (("model-a", "ep-1"), ("model-a", "ep-2"), ("model-b", "ep-1")):
        state.register_endpoint(EndpointOffering(
            ep, model, "explicit_capacity_fixture", {},
            {"rpm": 1000, "tpm": 10000, "concurrency": 10},
        ))
    steps = []

    def capture(case_id, expected, model="model-a", endpoint_ids=None):
        started = perf_counter_ns()
        result = state.assess_busy(model, at_time=now, endpoint_ids=endpoint_ids)
        elapsed_us = (perf_counter_ns() - started) / 1000
        version, runtimes, health = state.catalog.capacity_snapshot(model)
        steps.append(dict(
            case_id=case_id, expected_status=expected, passed=result.status == expected,
            duration_us=elapsed_us, assessment=result.to_dict(),
            input_snapshot=dict(catalog_version=version, endpoints=[dict(
                endpoint_id=item.endpoint_id, enabled=item.enabled,
                cooldown_until=item.cooldown_until.isoformat() if item.cooldown_until else None,
                health=health[item.endpoint_id],
                limits={name: getattr(item, f"capacity_{name}") for name in ("rpm", "tpm", "concurrency")},
                usage={name: getattr(item, f"current_{name}") for name in ("rpm", "tpm", "concurrency")},
            ) for item in runtimes]),
        ))

    capture("idle", "idle")
    state.catalog.update_model_endpoint_load("model-a", "ep-1", current_rpm=thresholds.enter * 1000)
    capture("one_busy_one_idle", "idle")
    capture("selected_busy_subset", "busy", endpoint_ids=["ep-1"])
    state.catalog.update_model_endpoint_load("model-a", "ep-2", current_tpm=thresholds.enter * 10000)
    capture("all_busy", "busy")
    middle = (thresholds.enter + thresholds.exit) / 2
    state.catalog.update_model_endpoint_load("model-a", "ep-1", current_rpm=middle * 1000)
    state.catalog.update_model_endpoint_load("model-a", "ep-2", current_tpm=middle * 10000)
    capture("hysteresis_hold", "busy")
    state.catalog.update_model_endpoint_load("model-a", "ep-1", current_rpm=0)
    state.catalog.update_model_endpoint_load("model-a", "ep-2", current_tpm=0)
    capture("recovered", "idle")
    state.catalog.update_model_endpoint_load("model-a", "ep-1", current_concurrency=10)
    capture("full_subset", "busy", endpoint_ids=["ep-1"])
    state.catalog.update_endpoint_runtime("ep-1", current_concurrency=1000, current_rpm=100000)
    capture("other_model_same_endpoint_idle", "idle", model="model-b")
    for ep in ("ep-1", "ep-2"):
        state.catalog.set_enabled("model-a", ep, False)
    capture("all_disabled", "unavailable")
    state.register_endpoint(EndpointOffering("unknown-ep", "model-c", "explicit_capacity_fixture", {}, {}))
    capture("unknown_capacity", "unknown", model="model-c")
    root = Path(__file__).resolve().parents[1]
    return dict(
        schema_version="1.0", purpose="capacity_pressure_rule_validation_only",
        thresholds=dict(enter=thresholds.enter, exit=thresholds.exit),
        cases=len(steps), passed=sum(item["passed"] for item in steps), steps=steps,
        sources={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                 for name in ("busy.py", "memory_state.py", "models.py", "experiments/busy_check.py")},
        limitations=["explicit synthetic load snapshots", "no SLO/price filtering", "no request admission",
                     "no queue or automatic speed-first switch", "no provider call or latency claim",
                     "single-pass timing is diagnostic, not a statistical performance benchmark"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enter-threshold", type=float, default=0.8)
    parser.add_argument("--exit-threshold", type=float, default=0.6)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    report = run_checks(BusyThresholds(args.enter_threshold, args.exit_threshold))
    output = args.output_dir or Path(__file__).resolve().parent / "output" / (
        "busy_check_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                      encoding="utf-8")
    print(f"Busy checks: {report['passed']}/{report['cases']} passed")
    for step in report["steps"]:
        print(f"  {step['case_id']}: {step['assessment']['status']} ({step['duration_us']:.2f} us)")
    print(f"Report: {output.resolve() / 'report.json'}")
    return 0 if report["passed"] == report["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
