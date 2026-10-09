"""Standalone commands: extract/profile once, then independently validate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

from . import __version__
from .dataset import extract, file_hash, write_json, write_jsonl
from .statistics import analyze, assign_time_splits
from .validation import validate_run


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if config.get("model_id") != "模型BP":
        raise ValueError("This report is scoped to 模型BP; other models need a separately reviewed analysis")
    for key in ("input_bin_upper_bounds", "output_bin_upper_bounds"):
        values = config[key]
        if not values or any(type(v) is not int or v <= 0 for v in values) or values != sorted(set(values)):
            raise ValueError(f"{key} must be a strictly increasing list of positive integers")
    if type(config['minimum_group_samples']) is not int or config['minimum_group_samples'] < 2:
        raise ValueError("minimum_group_samples must be an integer >= 2")
    assign_time_splits([], config['train_fraction'], config['calibration_fraction'])
    if config.get('display_timezone') != 'UTC+08:00':
        raise ValueError("The current human-readable report uses UTC+08:00")
    contract = config['routing_contract']
    if (contract.get('candidate_pool') != 'pareto_frontier_only' or contract.get('slots') != 3
            or contract.get('empty_slot') is not None or contract.get('stability_ranking') is not False
            or contract.get('variance_in_score') is not False or contract.get('slo_status') != 'not_yet_calibrated'
            or contract.get('ranking') != ['linear_cost_performance_score', 'endpoint_id']):
        raise ValueError("Routing contract must remain Pareto-only top 3, no stability or variance ranking; SLO not calibrated")
    return config


def prepare(source_dir, run_dir, config_path):
    # Imports are deferred so that dataset validation does not require plotting.
    import matplotlib
    from .plots import make_plots
    from .report import write_reports

    config = load_config(config_path)
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never silently replace an experiment.
    rows, offerings, extraction = extract(source_dir, root / 'dataset', config['model_id'])
    analysis, splits = analyze(rows, offerings, config, extraction)
    write_json(root / 'config.json', config)
    write_json(root / 'analysis.json', analysis)
    write_jsonl(root / 'dataset' / 'diagnostic_splits.jsonl', splits)
    write_reports(root, analysis, rows, offerings, config)
    make_plots(rows, analysis, root / 'figures')
    artifacts = {p.relative_to(root).as_posix(): dict(sha256=file_hash(p), bytes=p.stat().st_size)
                 for p in sorted(root.rglob('*')) if p.is_file()}
    project = Path(__file__).resolve().parents[1]
    code_paths = [*sorted((project / 'bp_slo').glob('*.py')), *sorted((project / 'tests').glob('*.py')),
                  project / 'requirements.txt', project / 'pyproject.toml']
    manifest = dict(schema_version='1.0', project='bp-slo-lab', version=__version__,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    environment=dict(python=platform.python_version(), matplotlib=matplotlib.__version__),
                    config_source_sha256=file_hash(config_path),
                    sources=extraction['sources'], extraction=extraction, artifacts=artifacts,
                    code_sha256={p.relative_to(project).as_posix(): file_hash(p) for p in code_paths},
                    scope='BP completed-request facts; descriptive analysis only; no calibrated SLO')
    write_json(root / 'manifest.json', manifest)
    result = validate_run(root)
    write_json(root / 'validation.json', result)
    return root, analysis, result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independent BP dataset and SLO sample profile (no legacy imports)')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Extract BP facts and write a new validated analysis run')
    p.add_argument('--source-dir', type=Path, default=Path('../实验数据/历史性能数据包'))
    p.add_argument('--output-dir', type=Path, default=Path('runs') / ('bp_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'analysis.json')
    p = commands.add_parser('validate', help='Recompute facts/statistics and check hashes, without modifying artifacts')
    p.add_argument('run_dir', type=Path)
    p = commands.add_parser('slo-experiment', help='Fit train-only SLO quantiles and test coverage with synthetic token predictions')
    p.add_argument('--input', type=Path, default=Path('runs/bp_baseline_20261007/dataset/requests.jsonl'))
    p.add_argument('--output-dir', type=Path, default=Path('runs') / ('bp_slo_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'slo_experiment.json')
    p = commands.add_parser('slo-tolerance', help='Compare strict coverage with explicit tolerance on an unchanged SLO experiment')
    p.add_argument('--baseline', type=Path, default=Path('runs/bp_slo_70_30_20261007'))
    p.add_argument('--output-dir', type=Path, default=Path('runs') / ('bp_slo_tolerance_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'slo_tolerance.json')
    p = commands.add_parser('routing-replay', help='Replay requests with typical latency, Pareto-first fill and labelled evidence fallbacks')
    p.add_argument('--baseline', type=Path, default=Path('runs/bp_neighborhood_slo_relative20_20261008'))
    p.add_argument('--offerings', type=Path, default=Path('runs/bp_baseline_20261007/dataset/endpoint_offerings.jsonl'))
    p.add_argument('--output-dir', type=Path, default=Path('runs') / ('bp_routing_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'routing_available.json')
    p.add_argument('--tolerance-config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'slo_tolerance.json')
    p = commands.add_parser('neighborhood-slo', help='Fit token-indexed local SLOs with labelled sparse and extrapolation fallbacks')
    p.add_argument('--input', type=Path, default=Path('runs/bp_slo_70_30_20261007/dataset/all_requests.jsonl'))
    p.add_argument('--output-dir', type=Path, default=Path('runs') / ('bp_neighborhood_slo_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'slo_neighborhood.json')
    p.add_argument('--tolerance-config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs' / 'slo_tolerance.json')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            root, analysis, result = prepare(args.source_dir, args.output_dir, args.config)
            summary = dict(output=str(root), validation_passed=result['passed'], request_count=analysis['total'],
                           modes={m: {k: v[k] for k in ('total', 'success', 'length_e2e_eligible', 'ttft_input_eligible', 'tpot_output_eligible')}
                                  for m, v in analysis['modes'].items()},
                           failed_checks=[c for c in result['checks'] if not c['passed']], warnings=result['warnings'])
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == 'slo-experiment':
            from .slo_experiment import prepare_slo_experiment
            root, coverage, result = prepare_slo_experiment(args.input, args.output_dir, args.config)
            print(json.dumps(dict(output=str(root), validation_passed=result['passed'],
                                  split_metadata=coverage['split_metadata'],
                                  coverage=[{k: r[k] for k in ('mode', 'metric', 'grade', 'evaluated', 'coverage')} for r in coverage['summary']],
                                  failed_checks=[c for c in result['checks'] if not c['passed']], warnings=result['warnings']),
                             ensure_ascii=False, indent=2))
        elif args.command == 'slo-tolerance':
            from .tolerance import prepare_tolerance
            root, data, result = prepare_tolerance(args.baseline, args.output_dir, args.config)
            print(json.dumps(dict(output=str(root), validation_passed=result['passed'],
                                  coverage=data['summary'],
                                  failed_checks=[c for c in result['checks'] if not c['passed']]), ensure_ascii=False, indent=2))
        elif args.command == 'neighborhood-slo':
            from .neighborhood_experiment import prepare
            root, data, result = prepare(args.input, args.output_dir, args.config, args.tolerance_config)
            print(json.dumps(dict(output=str(root), validation_passed=result['passed'], methods=data['methods'],
                                  coverage=[{k:r[k] for k in ('metric','grade','evaluated','coverage')} for r in data['tolerant_coverage']['summary']],
                                  failed_checks=[c for c in result['checks'] if not c['passed']]), ensure_ascii=False, indent=2))
        elif args.command == 'routing-replay':
            from .routing_experiment import prepare_replay
            root, data, result = prepare_replay(args.baseline, args.offerings, args.output_dir, args.config, args.tolerance_config)
            print(json.dumps(dict(output=str(root), validation_passed=result['passed'], summary=data['summary'],
                                  failed_checks=[c for c in result['checks'] if not c['passed']]), ensure_ascii=False, indent=2))
        else:
            manifest = args.run_dir / 'manifest.json'
            stage = json.loads(manifest.read_text(encoding='utf-8-sig')).get('stage') if manifest.is_file() else None
            if stage == 'neighborhood_slo':
                from .neighborhood_experiment import validate
                result = validate(args.run_dir)
            elif stage == 'request_conditioned_routing':
                from .routing_experiment import validate_replay
                result = validate_replay(args.run_dir)
            elif stage == 'slo_optimization':
                raise ValueError('Optimization was archived. Use the code snapshot under runs/archive_optimization_20261007/code to validate this historical run.')
            elif stage == 'slo_tolerance_only':
                from .tolerance import validate_tolerance
                result = validate_tolerance(args.run_dir)
            elif stage == 'slo_coverage_only':
                from .slo_experiment import validate_slo_experiment
                result = validate_slo_experiment(args.run_dir)
            else:
                result = validate_run(args.run_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['passed'] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
