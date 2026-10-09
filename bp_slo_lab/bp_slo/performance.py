"""Causal, bounded in-memory estimates conditioned on the incoming lengths.

Index entries reference sample IDs; completed facts are stored once. Length
bands are configured engineering assumptions, not fitted to the test results.
"""
from collections import defaultdict, deque
import math

from .dataset import number, timestamp, token
from .slo import METRICS


def validate_performance_config(config):
    windows = config['window_candidates_minutes']
    if (not isinstance(windows, list) or not windows or windows != sorted(set(windows))
            or any(type(v) is not int or v <= 0 for v in windows)):
        raise ValueError('Windows must be increasing positive integer minutes')
    for field in ('target_samples', 'minimum_samples', 'input_index_width_tokens', 'output_index_width_tokens'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError('Invalid '+field)
    for field in ('half_life_minutes', 'minimum_effective_samples'):
        if number(config[field]) is None or config[field] <= 0:
            raise ValueError('Invalid '+field)
    if not 2 <= config['minimum_effective_samples'] <= config['minimum_samples'] <= config['target_samples']:
        raise ValueError('Need 2 <= minimum_effective_samples <= minimum_samples <= target_samples')
    statistic = config.get('statistic', 'p95')
    if statistic not in ('p95', 'mean', 'p50') or config.get('quantile') != {'p95': .95, 'mean': None, 'p50': .5}[statistic]:
        raise ValueError('Statistic must be p95/mean/p50 with quantile .95/null/.5 respectively')
    for name in ('input', 'output'):
        band = config[name+'_similarity']
        if number(band['relative']) is None or type(band['absolute_tokens']) is not int or band['absolute_tokens'] <= 0:
            raise ValueError('Length bands need nonnegative relative width and positive integer floor')
    features = config.get('metric_features')
    if (not isinstance(features, dict) or set(features) != {'e2e', 'ttft', 'tpot'}
            or features['e2e'] != ['input', 'output'] or features['ttft'] != ['input']
            or features['tpot'] not in (['input'], ['input', 'output'])):
        raise ValueError('Explicit length conditioning must not silently drop a feature')
    return config


def effective_samples(weights):
    total = sum(weights)
    squares = sum(w*w for w in weights)
    return total*total/squares if squares > 0 else 0.0


def weighted_quantile(pairs, q=.95):
    pairs = sorted((value, weight) for value, weight in pairs if weight > 0)
    if not pairs:
        return None
    target, cumulative = q*sum(w for _, w in pairs), 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= target:
            return value
    return pairs[-1][0]


def weighted_mean(pairs):
    pairs = [(value, weight) for value, weight in pairs if weight > 0]
    total = sum(weight for _, weight in pairs)
    return sum(value * (weight / total) for value, weight in pairs) if total > 0 else None


def eligible_metric(row, metric):
    spec = METRICS[metric]
    arrival, finish = timestamp(row.get('arrived_at')), timestamp(row.get('finished_at'))
    return (row.get('success') is True and row.get('stream_type') == spec['mode']
            and row.get('attempt_count') == 1 and row.get(spec['eligible']) is True
            and isinstance(row.get('final_endpoint_id'), str) and bool(row['final_endpoint_id'])
            and number(row.get(spec['field'])) is not None
            and arrival is not None and finish is not None and finish >= arrival
            and token(row.get('actual_input_tokens')) is not None
            and token(row.get('actual_output_tokens')) is not None
            and 'token_counts_locally_estimated' not in row.get('quality_flags', []))


class PerformanceIndex:
    """Advance by completion time; never make future/current labels visible."""
    def __init__(self, rows, config):
        self.config = validate_performance_config(config)
        ids = [row.get('request_id') for row in rows]
        if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
            raise ValueError('History needs unique nonempty request IDs')
        self.pending = sorted((row for row in rows if any(eligible_metric(row, m) for m in METRICS)),
                              key=lambda r: (timestamp(r['finished_at']), r['request_id']))
        self.position, self.as_of = 0, None
        self.samples, self.expiry = {}, deque()
        self.by_pair, self.by_length = defaultdict(set), defaultdict(set)

    @staticmethod
    def _key(row):
        return row['model_id'], row['final_endpoint_id'], row['stream_type']

    def _length_key(self, row, field):
        width = self.config[field+'_index_width_tokens']
        return (*self._key(row), field, row['actual_'+field+'_tokens']//width)

    def advance(self, cutoff):
        cutoff = timestamp(cutoff) if isinstance(cutoff, str) else cutoff
        if cutoff is None or cutoff.tzinfo is None:
            raise ValueError('As-of timestamp must be timezone-aware')
        if self.as_of is not None and cutoff < self.as_of:
            raise ValueError('Replay arrivals must be nondecreasing')
        self.as_of = cutoff
        oldest = cutoff.timestamp() - self.config['window_candidates_minutes'][-1]*60
        while self.position < len(self.pending) and timestamp(self.pending[self.position]['finished_at']) < cutoff:
            row = self.pending[self.position]
            self.position += 1
            if timestamp(row['finished_at']).timestamp() < oldest:
                continue
            rid = row['request_id']
            self.samples[rid] = row
            self.expiry.append(rid)
            self.by_pair[self._key(row)].add(rid)
            for field in ('input', 'output'):
                self.by_length[self._length_key(row, field)].add(rid)
        while self.expiry and timestamp(self.samples[self.expiry[0]]['finished_at']).timestamp() < oldest:
            rid = self.expiry.popleft()
            row = self.samples.pop(rid)
            for index, key in [(self.by_pair, self._key(row)),
                               *[(self.by_length, self._length_key(row, f)) for f in ('input', 'output')]]:
                index[key].remove(rid)
                if not index[key]:
                    del index[key]

    def estimate(self, request, endpoint_id, metric):
        spec = METRICS[metric]
        cutoff = timestamp(request.get('arrived_at'))
        if cutoff is None or cutoff != self.as_of:
            raise ValueError('Advance the index to this request arrival before estimating')
        if request.get('stream_type') != spec['mode']:
            raise ValueError('Metric and request mode differ')
        key = request['model_id'], endpoint_id, request['stream_type']
        bands, source_ids = {}, None
        for field in self.config['metric_features'][metric]:
            predicted = token(request.get('predicted_'+field+'_tokens'))
            if predicted is None:
                return dict(status='missing_predicted_length', estimate=None, metric=metric, source_request_ids=[])
            band = self.config[field+'_similarity']
            tolerance = max(band['absolute_tokens'], predicted*band['relative'])
            low, high = max(0, predicted-tolerance), predicted+tolerance
            bands[field] = dict(predicted=predicted, lower=low, upper=high, tolerance=tolerance)
            width = self.config[field+'_index_width_tokens']
            ids = set()
            for bucket in range(math.floor(low/width), math.floor(high/width)+1):
                ids.update(self.by_length.get((*key, field, bucket), ()))
            source_ids = ids if source_ids is None else source_ids & ids
        matched = []
        for rid in source_ids or ():
            row = self.samples[rid]
            if rid == request['request_id'] or not eligible_metric(row, metric):
                continue
            distance = sum(((row['actual_'+f+'_tokens']-b['predicted'])/b['tolerance'])**2 for f, b in bands.items())
            if any(not b['lower'] <= row['actual_'+f+'_tokens'] <= b['upper'] for f, b in bands.items()):
                continue
            age = (cutoff-timestamp(row['finished_at'])).total_seconds()/60
            weight = 2**(-age/self.config['half_life_minutes'])*math.exp(-.5*distance)
            matched.append((row, age, weight))
        matched.sort(key=lambda item: (item[0]['finished_at'], item[0]['request_id']))
        trials, selected = [], []
        for minutes in self.config['window_candidates_minutes']:
            selected = [item for item in matched if item[1] <= minutes]
            neff = effective_samples([w for _, _, w in selected])
            trials.append(dict(minutes=minutes, samples=len(selected), effective_samples=neff))
            if len(selected) >= self.config['target_samples'] and neff >= self.config['minimum_effective_samples']:
                break
        minutes = trials[-1]['minutes']
        supported = len(selected) >= self.config['minimum_samples'] and neff >= self.config['minimum_effective_samples']
        q = weighted_quantile([(row[spec['field']], w) for row, _, w in selected])
        mixed = []
        for rid in sorted(self.by_pair.get(key, ())):
            row = self.samples[rid]
            age = (cutoff-timestamp(row['finished_at'])).total_seconds()/60
            if rid != request['request_id'] and age <= minutes and eligible_metric(row, metric):
                mixed.append((row[spec['field']], 2**(-age/self.config['half_life_minutes'])))
        result = dict(metric=metric, unit=spec['unit'],
                    status='estimated' if supported else 'insufficient_evidence', estimate=q if supported else None,
                    exploratory_p95=q, mixed_window_p95=weighted_quantile(mixed), mixed_samples=len(mixed),
                    samples=len(selected), effective_samples=neff, window_minutes=minutes,
                    quantile=.95, bands=bands, window_trials=trials,
                    source_request_ids=[row['request_id'] for row, _, _ in selected],
                    source_finished_at=[row['finished_at'] for row, _, _ in selected],
                    sample_weights=[w for _, _, w in selected],
                    scope=spec['field'], history_policy='strictly_finished_before_arrival_single_attempt_success')
        statistic = self.config.get('statistic', 'p95')
        if statistic != 'p95':
            pairs = [(row[spec['field']], w) for row, _, w in selected]
            typical = weighted_mean(pairs) if statistic == 'mean' else weighted_quantile(pairs, .5)
            result.update(statistic=statistic, quantile=None if statistic == 'mean' else .5,
                          estimate=typical if supported else None, exploratory_estimate=typical,
                          weighted_mean=weighted_mean(pairs), weighted_p50=weighted_quantile(pairs, .5),
                          evidence_level='sufficient' if supported else ('sparse' if pairs else 'unknown'))
        return result
