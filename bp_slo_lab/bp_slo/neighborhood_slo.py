"""Train-only, indexed local SLOs with explicitly labelled sparse fallbacks.

Grades name operational tiers, not promised coverage probabilities. Latency
quantiles are computed from the selected delays, never from token-sort ranks.
"""
from bisect import bisect_left, bisect_right
import math

from .dataset import number, timestamp, token
from .slo import METRICS, evaluate_test, mock_test_request, split_train_test
from .statistics import bin_label, correlations, describe, quantile


TIERS = ('strict', 'standard', 'relaxed')


def validate_config(config):
    if (config.get('schema_version') != '1.0' or config.get('stage') != 'neighborhood_slo'
            or config.get('model_id') != '模型BP' or config.get('split_policy') != 'per_mode_arrival_time'
            or 'calibration_fraction' in config or config.get('quantiles') != dict(zip(TIERS, (.5, .75, .9)))):
        raise ValueError('Use the explicit BP train/test neighborhood SLO contract')
    fraction = config.get('train_fraction')
    if number(fraction) is None or not 0 < fraction < 1:
        raise ValueError('Invalid train fraction')
    for field in ('input_bin_upper_bounds', 'output_bin_upper_bounds'):
        values = config[field]
        if not values or values != sorted(set(values)) or any(type(v) is not int or v <= 0 for v in values):
            raise ValueError('Invalid '+field)
    for field in ('forward_width_tokens', 'maximum_local_samples', 'minimum_local_samples', 'tail_fit_samples', 'ttft_prior_strength'):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError('Invalid '+field)
    if 'e2e_forward_neighborhood' in config:
        policy = config['e2e_forward_neighborhood']
        if (not isinstance(policy, dict) or set(policy) != {'relative_width', 'minimum_width_tokens'}
                or number(policy['relative_width']) is None or not 0 < policy['relative_width'] <= 1
                or type(policy['minimum_width_tokens']) is not int or policy['minimum_width_tokens'] <= 0):
            raise ValueError('E2E neighborhood needs relative width in (0,1] and a positive integer floor')
    if not 2 <= config['minimum_local_samples'] <= config['maximum_local_samples'] or config['tail_fit_samples'] < 2:
        raise ValueError('Invalid local evidence or fit sample counts')
    factors = config['fallback_factors']
    if (set(factors) != set(TIERS) or factors['strict'] != 1
            or any(number(v) is None for v in factors.values())
            or not 1 <= factors['standard'] <= factors['relaxed'] <= 2):
        raise ValueError('Fallback multipliers must be ordered, with first=1 and largest<=2')
    if config.get('stream_reference_quantile') != .9:
        raise ValueError('Streaming base is an explicit P90 reference')
    if set(config['cold_start_base']) != set(METRICS) or any(number(v) is None or v <= 0 for v in config['cold_start_base'].values()):
        raise ValueError('Cold-start bases must be positive, explicit engineering defaults')
    if config['cold_start_units'] != {m: s['unit'] for m, s in METRICS.items()}:
        raise ValueError('Wrong cold-start units')
    if number(config.get('maximum_documented_extrapolation_ratio')) is None or config['maximum_documented_extrapolation_ratio'] <= 1:
        raise ValueError('Extrapolation evidence boundary must exceed 1')
    mock = config['mock_prediction']
    if type(mock['seed']) is not int or type(mock['max_absolute_noise']) is not int or mock['max_absolute_noise'] < 0 or mock['clip_min'] != 0:
        raise ValueError('Invalid deterministic mock forecast')
    return config


def fit_linear_reference(samples):
    """Nonnegative slope + P90 residual offset; a heuristic, not a guarantee."""
    if not samples:
        return None
    xs, ys = [s['tokens'] for s in samples], [s['value'] for s in samples]
    mean_x, mean_y = sum(xs)/len(xs), sum(ys)/len(ys)
    variance = sum((x-mean_x)**2 for x in xs)
    slope = max(0.0, sum((x-mean_x)*(y-mean_y) for x, y in zip(xs, ys))/variance) if variance else 0.0
    source = 'nonnegative_ols_slope'
    if slope == 0:
        rates = [y/x for x, y in zip(xs, ys) if x > 0]
        slope = quantile(rates, .75) or 0.0
        source = 'p75_observed_per_token_rate_fallback'
    offset = max(0.0, quantile([y-slope*x for x, y in zip(xs, ys)], .9))
    return dict(slope_ms_per_token=slope, offset_ms=offset, samples=len(samples),
                minimum_tokens=min(xs), maximum_tokens=max(xs), latency_p90=quantile(ys, .9),
                slope_source=source, residual_reference_quantile=.9,
                source_request_ids=[s['request_id'] for s in samples], guarantee=False)


def fit_profiles(rows, splits, config):
    validate_config(config)
    states = {s['request_id']: s for s in splits}
    pools, association = {}, {}
    for metric, spec in METRICS.items():
        eligible = []
        for row in rows:
            state = states[row['request_id']]
            if (state['split'] != 'train' or not state['training_outcome_visible'] or row['stream_type'] != spec['mode'] or row.get('success') is not True
                    or row.get(spec['eligible']) is not True or number(row.get(spec['field'])) is None
                    or 'token_counts_locally_estimated' in row.get('quality_flags', [])):
                continue
            eligible.append(row)
        field = 'actual_output_tokens' if metric == 'e2e' else 'actual_input_tokens' if metric == 'ttft' else None
        samples = [dict(request_id=r['request_id'], tokens=r[field] if field else None,
                        value=r[spec['field']], finished_at=r['finished_at']) for r in eligible
                   if field is None or token(r.get(field)) is not None]
        if field:
            # Newer records win equal-token ties; response values never rank candidates.
            samples.sort(key=lambda s: (s['tokens'], -timestamp(s['finished_at']).timestamp(), s['request_id']))
        else:
            samples.sort(key=lambda s: s['request_id'])
        pools[metric] = dict(samples=samples, statistics=describe(s['value'] for s in samples),
                             global_p90=quantile([s['value'] for s in samples], .9), unit=spec['unit'])
        association[metric] = {f: correlations(eligible, 'actual_'+f+'_tokens', spec['field']) for f in ('input', 'output')}
    samples = pools['e2e']['samples']
    pools['e2e']['global_linear_reference'] = fit_linear_reference(samples)
    pools['e2e']['tail_linear_reference'] = fit_linear_reference(samples[-config['tail_fit_samples']:])
    return dict(schema_version='1.0', model_id=config['model_id'], method='token_neighborhood',
                quantiles=config['quantiles'], input_bin_upper_bounds=config['input_bin_upper_bounds'],
                output_bin_upper_bounds=config['output_bin_upper_bounds'], config=config, pools=pools,
                training_associations=association, rule_source='visible_successful_train_only',
                tier_semantics='local_e2e_quantiles_or_reference_multipliers_not_guaranteed_percentiles')


class NeighborhoodSLO:
    """Build token vectors once, then use bisect + at most K latency samples."""
    method = 'token_neighborhood'

    def __init__(self, rules):
        self.rules, self.config = rules, validate_config(rules['config'])
        self.lengths = {m: [s['tokens'] for s in rules['pools'][m]['samples']] for m in ('e2e', 'ttft')}
        if any(v != sorted(v) for v in self.lengths.values()):
            raise ValueError('Token index must be sorted')

    def _local(self, metric, predicted):
        lengths = self.lengths[metric]
        if predicted is None:
            return [], dict(left=None, right=None, available=0, selected=0)
        width = self.config['forward_width_tokens']
        policy = self.config.get('e2e_forward_neighborhood') if metric == 'e2e' else None
        if policy is not None:
            width = max(policy['minimum_width_tokens'], math.ceil(predicted*policy['relative_width']))
        left = bisect_left(lengths, predicted)
        right = bisect_right(lengths, predicted+width)
        stop = min(right, left+self.config['maximum_local_samples'])
        search = dict(left=left, right=right, selected=stop-left,
                      available=right-left, lower_tokens=predicted, upper_tokens=predicted+width)
        if policy is not None:
            search.update(width_tokens=width, relative_width=policy['relative_width'],
                          minimum_width_tokens=policy['minimum_width_tokens'], policy='relative_width_with_floor')
        return self.rules['pools'][metric]['samples'][left:stop], search

    def _estimate(self, metric, predicted):
        cfg, pool = self.config, self.rules['pools'][metric]
        local, search = self._local(metric, predicted) if metric != 'tpot' else ([], None)
        samples = pool['samples']
        factors, quantiles = cfg['fallback_factors'], cfg['quantiles']
        evidence = dict(metric=metric, search=search, local_samples=len(local), model_training_samples=len(samples),
                        source_request_ids=[s['request_id'] for s in local], extrapolated=False,
                        fallback=False, guarantee=False)
        if metric == 'e2e' and len(local) >= cfg['minimum_local_samples']:
            values = [s['value'] for s in local]
            return {g: quantile(values, q) for g, q in quantiles.items()}, dict(evidence, method='forward_local_quantiles',
                    evidence_level='local_empirical', reference_quantiles=quantiles, variance=describe(values)['variance'])
        if metric == 'e2e' and local:
            base = quantile([s['value'] for s in local], .9)
            evidence.update(method='sparse_local_p90_times_factors', fallback=True, evidence_level='sparse_fallback', reference_quantile=.9)
        elif metric == 'e2e' and samples and predicted is not None:
            maximum = self.lengths[metric][-1]
            extrapolated = predicted > maximum
            fit = pool['tail_linear_reference'] if extrapolated else pool['global_linear_reference']
            base = fit['offset_ms']+fit['slope_ms_per_token']*predicted
            if extrapolated:
                base = max(base, fit['latency_p90'])
            evidence.update(method='tail_linear_extrapolation' if extrapolated else 'linear_gap_fallback',
                            extrapolated=extrapolated, fallback=True, evidence_level='heuristic_fit',
                            linear_reference={k: v for k, v in fit.items() if k != 'source_request_ids'},
                            linear_reference_rule='tail_linear_reference' if extrapolated else 'global_linear_reference',
                            source_request_ids=fit['source_request_ids'] if extrapolated else [], global_reference_pool='e2e',
                            extrapolation_ratio=predicted/maximum if maximum > 0 else None,
                            beyond_documented_domain=(maximum == 0 or predicted > maximum*cfg['maximum_documented_extrapolation_ratio']))
        elif metric == 'ttft' and samples:
            global_base = pool['global_p90']
            weight = len(local)/(len(local)+cfg['ttft_prior_strength']) if len(local) >= cfg['minimum_local_samples'] else 0.0
            local_base = quantile([s['value'] for s in local], .9) if local else None
            base = max(global_base, weight*local_base+(1-weight)*global_base) if weight else global_base
            evidence.update(method='ttft_global_p90_floor_with_local_shrinkage', reference_quantile=.9,
                            global_p90=global_base, local_p90=local_base, local_weight=weight,
                            fallback=weight == 0, evidence_level='local_plus_global' if weight else 'global_fallback',
                            global_reference_pool='ttft',
                            extrapolated=predicted is not None and predicted > self.lengths['ttft'][-1])
        elif samples:
            base = pool['global_p90']
            evidence.update(method='tpot_global_p90_times_factors' if metric == 'tpot' else 'missing_prediction_global_p90',
                            reference_quantile=.9, fallback=metric != 'tpot', evidence_level='global_empirical',
                            source_request_ids=[], global_reference_pool=metric)
        else:
            base = cfg['cold_start_base'][metric]
            evidence.update(method='configured_cold_start_default', fallback=True,
                            evidence_level='no_historical_evidence', business_approval_required=True)
        return {grade: base*factor for grade, factor in factors.items()}, evidence

    def assign(self, request):
        grades = {g: {} for g in TIERS}
        for metric, spec in METRICS.items():
            if request.get('stream_type') != spec['mode']:
                continue
            field = 'predicted_output_tokens' if metric == 'e2e' else 'predicted_input_tokens' if metric == 'ttft' else None
            predicted = token(request.get(field)) if field else None
            limits, evidence = self._estimate(metric, predicted)
            bounds = self.rules['output_bin_upper_bounds' if metric == 'e2e' else 'input_bin_upper_bounds'] if field else None
            label = bin_label(predicted, bounds) if bounds else 'all'
            for grade in TIERS:
                matched_model = request.get('model_id') == self.rules['model_id']
                grades[grade][metric] = dict(status='assigned' if matched_model else 'model_mismatch',
                    limit=limits[grade] if matched_model else None, unit=spec['unit'], range=label,
                    rule_id=f'neighborhood:{metric}:{evidence["method"]}',
                    quantile=self.config['quantiles'][grade] if evidence['method'] == 'forward_local_quantiles' else None,
                    reference_n=evidence['local_samples'] if evidence['local_samples'] else evidence['model_training_samples'],
                    evidence=evidence, tier_factor=None if evidence['method'] == 'forward_local_quantiles' else self.config['fallback_factors'][grade])
        return dict(request_id=request['request_id'], model_id=request.get('model_id'), stream_type=request.get('stream_type'), grades=grades)


def run_experiment(rows, config):
    validate_config(config)
    if any(r.get('model_id') != config['model_id'] for r in rows):
        raise ValueError('Facts contain a different model')
    rows = sorted(rows, key=lambda r: (r.get('arrived_at') or '9999', r['request_id']))
    splits, metadata = split_train_test(rows, config['train_fraction'])
    rules = fit_profiles(rows, splits, config)
    rules['split_metadata'] = metadata
    ids = {s['request_id'] for s in splits if s['split'] == 'test'}
    tests = [r for r in rows if r['request_id'] in ids]
    inputs = [mock_test_request(r, config) for r in tests]
    index = NeighborhoodSLO(rules)
    assignments = [index.assign(r) for r in inputs]
    coverage, results = evaluate_test(tests, inputs, assignments, config)
    coverage.update(stage='neighborhood_slo', split_metadata=metadata,
                    unassigned_requests=sum(s['split'] == 'unassigned' for s in splits),
                    tier_semantics=rules['tier_semantics'])
    return dict(splits=splits, rules=rules, test_inputs=inputs, assignments=assignments, coverage=coverage, results=results)
