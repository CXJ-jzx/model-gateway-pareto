"""Fit train-only quantile tables, assign SLOs, and evaluate observed coverage."""
from collections import Counter, defaultdict
import hashlib
import math
import random

from .dataset import number, timestamp, token
from .statistics import bin_label, describe, quantile


METRICS = {
    "e2e": dict(mode="nonstream", field="request_e2e_ms", eligible="eligible_length_e2e", unit="ms",
                length="actual_output_tokens", prediction="predicted_output_tokens", bounds="output_bin_upper_bounds"),
    "ttft": dict(mode="stream", field="ttft_proxy_ms", eligible="eligible_ttft_input", unit="ms",
                 length="actual_input_tokens", prediction="predicted_input_tokens", bounds="input_bin_upper_bounds"),
    "tpot": dict(mode="stream", field="tpot_proxy_ms", eligible="eligible_tpot_output", unit="ms/token",
                 length=None, prediction=None, bounds=None),
}


def validate_config(config):
    if config.get("model_id") != "模型BP":
        raise ValueError("This experiment is scoped to 模型BP")
    fraction = config.get("train_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not 0 < fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if "calibration_fraction" in config or config.get("split_policy") != "per_mode_arrival_time":
        raise ValueError("Use a per-mode train/test split with no calibration interval")
    if config.get("quantiles") != {"p50": .5, "p75": .75, "p95": .95}:
        raise ValueError("The current SLO grades are p50, p75 and p95")
    for name in ("input_bin_upper_bounds", "output_bin_upper_bounds"):
        bounds = config[name]
        if not isinstance(bounds, list) or not bounds or any(type(v) is not int or v <= 0 for v in bounds) or bounds != sorted(set(bounds)):
            raise ValueError(f"Invalid bucket boundaries: {name}")
    if type(config.get("minimum_group_samples")) is not int or config["minimum_group_samples"] < 2:
        raise ValueError("minimum_group_samples must be an integer >= 2")
    mock = config["mock_prediction"]
    if type(mock.get("seed")) is not int or type(mock.get("max_absolute_noise")) is not int or mock["max_absolute_noise"] < 0 or mock.get("clip_min") != 0:
        raise ValueError("Mock predictions need an integer seed, nonnegative integer amplitude and zero lower bound")
    if config.get("sparse_rule_policy") != "return_insufficient_evidence" or config.get("stage") != "slo_coverage_only":
        raise ValueError("This stage evaluates SLO coverage and reports sparse rules explicitly")
    return config


def split_train_test(rows, train_fraction=.7):
    """Split arrivals per mode, moving every tied boundary arrival into test.

    Training outcomes finished at/after the cutoff are retained as facts but
    cannot contribute to fitted rules. Invalid modes/times remain unassigned.
    """
    if isinstance(train_fraction, bool) or not isinstance(train_fraction, (int, float)) or not 0 < train_fraction < 1:
        raise ValueError("Invalid train_fraction")
    ids = [r.get("request_id") for r in rows]
    if any(not isinstance(rid, str) or not rid for rid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Request IDs must be unique nonempty strings")
    states = {rid: dict(split="unassigned", training_outcome_visible=False) for rid in ids}
    metadata = {}
    for mode in ("nonstream", "stream"):
        ordered = sorted((r for r in rows if r['stream_type'] == mode and timestamp(r.get('arrived_at')) is not None),
                         key=lambda r: (timestamp(r['arrived_at']), r['request_id']))
        if not ordered:
            continue
        cutoff = timestamp(ordered[int(len(ordered)*train_fraction)]['arrived_at'])
        for row in ordered:
            is_train = timestamp(row['arrived_at']) < cutoff
            finished = timestamp(row.get('finished_at'))
            states[row['request_id']] = dict(split='train' if is_train else 'test',
                                             training_outcome_visible=is_train and finished is not None and finished < cutoff)
        metadata[mode] = dict(cutoff=cutoff.isoformat(), total=len(ordered),
                              counts=dict(Counter(states[r['request_id']]['split'] for r in ordered)),
                              train_outcomes_visible=sum(states[r['request_id']]['training_outcome_visible'] for r in ordered),
                              requested_train_fraction=train_fraction)
    splits = [dict(request_id=r['request_id'], stream_type=r['stream_type'], arrived_at=r.get('arrived_at'),
                   finished_at=r.get('finished_at'), **states[r['request_id']]) for r in rows]
    return splits, metadata


def fit_rules(rows, splits, config):
    """Each threshold consumes only visible, valid, successful training facts."""
    validate_config(config)
    state = {r['request_id']: r for r in splits}
    visible = [r for r in rows if state[r['request_id']]['split'] == 'train' and state[r['request_id']]['training_outcome_visible']]
    rules = []
    for metric, spec in METRICS.items():
        bounds = config[spec['bounds']] if spec['bounds'] else None
        labels = [bin_label(v, bounds) for v in [0]+[b+1 for b in bounds]] if bounds else ['all']
        grouped = defaultdict(list)
        for row in visible:
            if (row['stream_type'] != spec['mode'] or not row['success'] or row.get(spec['eligible']) is not True
                    or 'token_counts_locally_estimated' in row.get('quality_flags', []) or number(row.get(spec['field'])) is None):
                continue
            length = token(row.get(spec['length'])) if spec['length'] else None
            if bounds and length is None:
                continue
            grouped[bin_label(length, bounds) if bounds else 'all'].append(row)
        for label in labels:
            items = grouped[label]
            values = [r[spec['field']] for r in items]
            stats = describe(values)
            supported = len(values) >= config['minimum_group_samples']
            rules.append(dict(rule_id=f"{spec['mode']}:{metric}:{label}", mode=spec['mode'], metric=metric, range=label,
                              unit=spec['unit'], n=len(values), status='supported' if supported else 'insufficient_evidence',
                              statistics=stats,
                              exploratory_quantiles={g: quantile(values, q) for g, q in config['quantiles'].items()},
                              thresholds={g: quantile(values, q) if supported else None for g, q in config['quantiles'].items()},
                              training_request_ids=[r['request_id'] for r in items]))
    return dict(schema_version='1.0', model_id=config['model_id'], quantiles=config['quantiles'],
                minimum_group_samples=config['minimum_group_samples'], input_bin_upper_bounds=config['input_bin_upper_bounds'],
                output_bin_upper_bounds=config['output_bin_upper_bounds'], split_policy=config['split_policy'],
                rule_source='visible_successful_train_only', rules=rules)


def _mock_length(actual, request_id, field, seed, amplitude):
    actual = token(actual)
    if actual is None:
        return None, dict(draw=None, applied_error=None, clipped=False)
    # A request/field-specific seed keeps draws independent of processing order.
    digest = hashlib.sha256(f"{seed}|{request_id}|{field}".encode()).digest()
    draw = random.Random(int.from_bytes(digest, 'big')).randint(-amplitude, amplitude)
    predicted = max(0, actual+draw)
    return predicted, dict(draw=draw, applied_error=predicted-actual, clipped=actual+draw < 0)


def mock_test_request(row, config):
    mock = config['mock_prediction']
    inp, input_noise = _mock_length(row.get('actual_input_tokens'), row['request_id'], 'input', mock['seed'], mock['max_absolute_noise'])
    out, output_noise = _mock_length(row.get('actual_output_tokens'), row['request_id'], 'output', mock['seed'], mock['max_absolute_noise'])
    return dict(schema_version='1.0', record_kind='synthetic_request_input', request_id=row['request_id'],
                model_id=row['model_id'], stream_type=row['stream_type'], arrived_at=row.get('arrived_at'),
                predicted_input_tokens=inp, predicted_output_tokens=out,
                prediction_source='mock_actual_tokens_plus_uniform_integer_noise',
                prediction_noise=dict(input=input_noise, output=output_noise),
                prediction_seed=mock['seed'], max_absolute_noise=mock['max_absolute_noise'])


def assign_slo(request, rules):
    """Only consume model, mode and predicted lengths; no outcome lookup."""
    if getattr(rules, 'method', None) == 'token_neighborhood':
        return rules.assign(request)
    if isinstance(rules, dict) and rules.get('method') == 'token_neighborhood':
        from .neighborhood_slo import NeighborhoodSLO
        return NeighborhoodSLO(rules).assign(request)
    index = {(r['metric'], r['range']): r for r in rules['rules']}
    assignments = {}
    for grade, q in rules['quantiles'].items():
        assigned = {}
        for metric, spec in METRICS.items():
            if request['stream_type'] != spec['mode']:
                continue
            length = token(request.get(spec['prediction'])) if spec['prediction'] else None
            label = bin_label(length, rules[spec['bounds']]) if spec['bounds'] else 'all'
            rule = index.get((metric, label))
            status = ('model_mismatch' if request['model_id'] != rules['model_id'] else
                      'missing_predicted_length' if spec['bounds'] and length is None else
                      'insufficient_evidence' if rule is None or rule['status'] != 'supported' else 'assigned')
            assigned[metric] = dict(status=status, range=label, rule_id=rule['rule_id'] if rule else None,
                                    reference_n=rule['n'] if rule else 0, unit=spec['unit'], quantile=q,
                                    limit=rule['thresholds'][grade] if status == 'assigned' else None)
        assignments[grade] = assigned
    return dict(request_id=request['request_id'], model_id=request['model_id'], stream_type=request['stream_type'], grades=assignments)


def coverage_interval(passed, total):
    """Descriptive Wilson 95% interval; not a time-series independence claim."""
    if not total:
        return None
    p, z = passed/total, 1.959963984540054
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    margin = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.0, center-margin), min(1.0, center+margin)]


def _coverage_group(items, mode, grade, metric, label='all'):
    evaluated = [r for r in items if r['evaluation_status'] in ('covered', 'exceeded')]
    passed = sum(r['evaluation_status'] == 'covered' for r in evaluated)
    return dict(mode=mode, grade=grade, metric=metric, range=label, total=len(items),
                assigned=sum(r['assignment_status'] == 'assigned' for r in items),
                successful_metric_valid=sum(r['successful_metric_valid'] for r in items),
                evaluated=len(evaluated), covered=passed, exceeded=len(evaluated)-passed,
                coverage=passed/len(evaluated) if evaluated else None,
                coverage_wilson_95=coverage_interval(passed, len(evaluated)),
                assignment_status_counts=dict(Counter(r['assignment_status'] for r in items)),
                evaluation_status_counts=dict(Counter(r['evaluation_status'] for r in items)),
                exceedance=describe(r['exceedance'] for r in evaluated if r['evaluation_status'] == 'exceeded'))


def evaluate_test(test_rows, inputs, assignments, config):
    facts = {r['request_id']: r for r in test_rows}
    predictions = {r['request_id']: r for r in inputs}
    results, flat = [], []
    for assignment in assignments:
        row = facts[assignment['request_id']]
        for grade, metrics in assignment['grades'].items():
            observed = {}
            for metric, rule in metrics.items():
                spec = METRICS[metric]
                value = number(row.get(spec['field']))
                valid = row['success'] and value is not None
                status = ('request_failed' if not row['success'] else 'missing_metric' if value is None else
                          'rule_unavailable' if rule['status'] != 'assigned' else
                          'covered' if value <= rule['limit'] else 'exceeded')
                observed[metric] = dict(assignment_status=rule['status'], evaluation_status=status, observed=value,
                                        limit=rule['limit'], unit=rule['unit'], range=rule['range'], reference_n=rule['reference_n'],
                                        successful_metric_valid=valid,
                                        exceedance=max(0.0, value-rule['limit']) if status in ('covered', 'exceeded') else None)
            if row['stream_type'] == 'stream':
                ttft, tpot = observed['ttft'], observed['tpot']
                assigned = ttft['assignment_status'] == tpot['assignment_status'] == 'assigned'
                valid = ttft['successful_metric_valid'] and tpot['successful_metric_valid']
                joint_status = ('request_failed' if not row['success'] else 'missing_metric' if not valid else
                                'rule_unavailable' if not assigned else
                                'covered' if ttft['evaluation_status'] == tpot['evaluation_status'] == 'covered' else 'exceeded')
                observed['joint'] = dict(assignment_status='assigned' if assigned else 'partial_or_unavailable',
                                         evaluation_status=joint_status, successful_metric_valid=valid,
                                         range=ttft['range'], observed=None, limit=None, unit=None, exceedance=None)
            results.append(dict(request_id=row['request_id'], stream_type=row['stream_type'], grade=grade,
                                success=row['success'], metrics=observed))
            flat.extend(dict(mode=row['stream_type'], grade=grade, metric=metric, **result) for metric, result in observed.items())
    summary, groups = [], []
    for mode, metrics in (('nonstream', ('e2e',)), ('stream', ('ttft', 'tpot', 'joint'))):
        for grade in config['quantiles']:
            for metric in metrics:
                part = [r for r in flat if r['mode'] == mode and r['grade'] == grade and r['metric'] == metric]
                summary.append(_coverage_group(part, mode, grade, metric))
                for label in sorted({r['range'] for r in part}, key=lambda x: (x is None, x or '')):
                    groups.append(_coverage_group([r for r in part if r['range'] == label], mode, grade, metric, label))
    prediction_stats = []
    for mode in ('nonstream', 'stream'):
        for field, bounds_name in (('input', 'input_bin_upper_bounds'), ('output', 'output_bin_upper_bounds')):
            pairs = [(r, predictions[r['request_id']]) for r in test_rows if r['stream_type'] == mode and
                     token(r.get(f'actual_{field}_tokens')) is not None and token(predictions[r['request_id']].get(f'predicted_{field}_tokens')) is not None]
            errors = [p[f'predicted_{field}_tokens']-r[f'actual_{field}_tokens'] for r, p in pairs]
            flips = sum(bin_label(r[f'actual_{field}_tokens'], config[bounds_name]) != bin_label(p[f'predicted_{field}_tokens'], config[bounds_name]) for r, p in pairs)
            prediction_stats.append(dict(mode=mode, field=field, n=len(pairs), error=describe(errors),
                                         absolute_error=describe(abs(e) for e in errors), bucket_changes=flips,
                                         bucket_change_rate=flips/len(pairs) if pairs else None,
                                         clipped=sum(p['prediction_noise'][field]['clipped'] for _, p in pairs)))
    modes = {mode: dict(total=sum(r['stream_type'] == mode for r in test_rows),
                       success=sum(r['stream_type'] == mode and r['success'] for r in test_rows),
                       failed=sum(r['stream_type'] == mode and not r['success'] for r in test_rows)) for mode in ('nonstream', 'stream')}
    test_statistics = [dict(mode=spec['mode'], metric=metric, unit=spec['unit'],
                            statistics=describe(r['observed'] for r in flat if r['grade'] == 'p50' and r['metric'] == metric and r['successful_metric_valid']))
                       for metric, spec in METRICS.items()]
    return dict(schema_version='1.0', model_id=config['model_id'], stage='slo_coverage_only',
                test_modes=modes, summary=summary, by_predicted_bucket=groups, mock_prediction=prediction_stats,
                test_metric_statistics=test_statistics,
                coverage_denominator='successful metric-valid test requests with an assigned threshold',
                joint_definition='TTFT and TPOT both covered on the same evaluated stream request'), results


def run_experiment(rows, config):
    validate_config(config)
    if any(r.get('model_id') != config['model_id'] for r in rows):
        raise ValueError("Input facts contain another model")
    rows = sorted(rows, key=lambda r: (r.get('arrived_at') or '9999', r['request_id']))
    splits, metadata = split_train_test(rows, config['train_fraction'])
    rules = fit_rules(rows, splits, config)
    rules['split_metadata'] = metadata
    test_ids = {s['request_id'] for s in splits if s['split'] == 'test'}
    test_rows = [r for r in rows if r['request_id'] in test_ids]
    inputs = [mock_test_request(r, config) for r in test_rows]
    assignments = [assign_slo(r, rules) for r in inputs]
    coverage, results = evaluate_test(test_rows, inputs, assignments, config)
    coverage['split_metadata'] = metadata
    coverage['unassigned_requests'] = sum(s['split'] == 'unassigned' for s in splits)
    return dict(splits=splits, rules=rules, test_inputs=inputs, assignments=assignments, coverage=coverage, results=results)
