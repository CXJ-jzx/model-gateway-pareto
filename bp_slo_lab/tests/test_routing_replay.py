"""Length mixing, causal evidence, SLO gates, pricing and replay contracts."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bp_slo.dataset import file_hash, write_json, write_jsonl
from bp_slo.performance import PerformanceIndex, effective_samples, weighted_quantile
from bp_slo.routing import ModelEndpointCatalog, RoutingEngine, validate_routing_config
from bp_slo.routing_experiment import prepare_replay, select_requests, validate_replay
from bp_slo.slo import run_experiment
from bp_slo.slo_experiment import prepare_slo_experiment


PROJECT = Path(__file__).resolve().parents[1]
START = datetime(2026, 9, 21, tzinfo=timezone.utc)


def config():
    value = json.loads((PROJECT/'configs/routing_experiment.json').read_text(encoding='utf-8'))
    value['performance'].update(window_candidates_minutes=[10, 60], target_samples=4, minimum_samples=3,
                                minimum_effective_samples=2)
    value['sample_requests_per_group'] = 1
    return value


def tolerance():
    return json.loads((PROJECT/'configs/slo_tolerance.json').read_text(encoding='utf-8'))


def fact(i, endpoint='A', mode='nonstream', inp=200, out=300, latency=100):
    start = START+timedelta(minutes=i)
    return dict(request_id=f'{endpoint}_{mode}_{i}', model_id='模型BP', stream_type=mode,
                arrived_at=start.isoformat(), finished_at=(start+timedelta(seconds=1)).isoformat(), success=True,
                final_endpoint_id=endpoint, attempt_count=1, actual_input_tokens=inp, actual_output_tokens=out,
                request_e2e_ms=latency, ttft_proxy_ms=latency if mode == 'stream' else None,
                tpot_proxy_ms=latency/100 if mode == 'stream' else None,
                eligible_length_e2e=True, eligible_ttft_input=mode == 'stream', eligible_tpot_output=mode == 'stream',
                quality_flags=[])


def request(mode='nonstream', at=20, inp=200, out=300):
    return dict(request_id='incoming', model_id='模型BP', stream_type=mode,
                arrived_at=(START+timedelta(minutes=at)).isoformat(),
                predicted_input_tokens=inp, predicted_output_tokens=out)


def offering(endpoint='A', pin=1, pout=2, currency='CNY'):
    tier = dict(currency=currency, billing_unit='per_million_tokens', input_tokens_min_exclusive=0,
                input_tokens_max_inclusive=10000, conditions={}, input_per_million=pin, output_per_million=pout)
    return dict(model_id='模型BP', endpoint_id=endpoint, price_config=dict(pricing_status='configured',
                currency=currency, billing_unit='per_million_tokens', price_tiers=[tier]))


def rules(mode='nonstream', limit=100):
    rows = [fact(i, mode=mode, latency=limit) for i in range(10)]
    base = json.loads((PROJECT/'configs/slo_experiment.json').read_text(encoding='utf-8'))
    base['minimum_group_samples'] = 2
    return run_experiment(rows, base)['rules']


class PerformanceTests(unittest.TestCase):
    def test_weighted_statistics(self):
        self.assertEqual(effective_samples([1, 1, 1]), 3)
        self.assertEqual(weighted_quantile([(1, 1), (2, 1), (100, .01)]), 2)

    def test_long_requests_do_not_pollute_short_estimate(self):
        rows = [fact(i) for i in range(4)] + [fact(i, out=3000, latency=10000) for i in range(4, 10)]
        index = PerformanceIndex(rows, config()['performance'])
        req = request()
        index.advance(req['arrived_at'])
        estimate = index.estimate(req, 'A', 'e2e')
        self.assertEqual(estimate['status'], 'estimated')
        self.assertEqual(estimate['estimate'], 100)
        self.assertEqual(estimate['mixed_window_p95'], 10000)
        self.assertEqual(estimate['samples'], 4)

    def test_large_input_also_excluded_from_e2e(self):
        rows = [fact(i) for i in range(4)] + [fact(6, inp=2000, latency=10000)]
        index = PerformanceIndex(rows, config()['performance'])
        req = request()
        index.advance(req['arrived_at'])
        self.assertNotIn(rows[-1]['request_id'], index.estimate(req, 'A', 'e2e')['source_request_ids'])

    def test_current_future_failed_and_retry_labels_are_unavailable(self):
        rows = [fact(i) for i in range(4)]
        rows += [dict(fact(5), success=False), dict(fact(6), attempt_count=2), fact(30)]
        rows += [dict(fact(20), request_id='incoming', finished_at=request()['arrived_at'])]
        index = PerformanceIndex(rows, config()['performance'])
        req = request()
        index.advance(req['arrived_at'])
        sources = index.estimate(req, 'A', 'e2e')['source_request_ids']
        self.assertEqual(set(sources), {r['request_id'] for r in rows[:4]})

    def test_missing_and_sparse_lengths_do_not_use_mixed_fallback(self):
        rows = [fact(i, out=3000) for i in range(8)]
        index = PerformanceIndex(rows, config()['performance'])
        req = request()
        index.advance(req['arrived_at'])
        estimate = index.estimate(req, 'A', 'e2e')
        self.assertEqual(estimate['status'], 'insufficient_evidence')
        self.assertIsNone(estimate['estimate'])
        self.assertIsNotNone(estimate['mixed_window_p95'])
        self.assertEqual(index.estimate({**req, 'predicted_output_tokens': None}, 'A', 'e2e')['status'], 'missing_predicted_length')

    def test_window_expands_based_on_matching_samples(self):
        cfg = config()['performance']
        rows = [fact(i) for i in range(4)] + [fact(i, out=3000) for i in range(16, 20)]
        index = PerformanceIndex(rows, cfg)
        req = request()
        index.advance(req['arrived_at'])
        estimate = index.estimate(req, 'A', 'e2e')
        self.assertEqual(estimate['window_minutes'], 60)
        self.assertEqual(estimate['window_trials'][0]['samples'], 0)

    def test_expiry_cleans_canonical_and_length_indices(self):
        index = PerformanceIndex([fact(i) for i in range(4)], config()['performance'])
        index.advance(request()['arrived_at'])
        self.assertEqual(len(index.samples), 4)
        index.advance(request(at=100)['arrived_at'])
        self.assertFalse(index.samples)
        self.assertFalse(index.by_length)
        self.assertFalse(index.by_pair)
        with self.assertRaises(ValueError):
            index.advance(request(at=20)['arrived_at'])

    def test_feature_and_effective_evidence_config_are_explicit(self):
        cfg = config()
        cfg['performance']['metric_features']['e2e'] = ['output']
        with self.assertRaises(ValueError):
            validate_routing_config(cfg)
        cfg = config()
        cfg['performance']['minimum_effective_samples'] = 0
        with self.assertRaises(ValueError):
            validate_routing_config(cfg)


class PriceAndRoutingTests(unittest.TestCase):
    def test_predicted_token_cost_currency_and_price_tiers(self):
        item = offering(currency='USD')
        catalog = ModelEndpointCatalog([item])
        quote = catalog.quote(request(), 'A', 'CNY', {'CNY': 1, 'USD': 7})
        self.assertAlmostEqual(quote['cost'], (200+300*2)/1e6*7)
        self.assertEqual(catalog.quote(request(), 'A', 'CNY', {'CNY': 1})['status'], 'missing_fx_rate')
        second = deepcopy(item['price_config']['price_tiers'][0])
        second.update(input_tokens_min_exclusive=10000, input_tokens_max_inclusive=20000, input_per_million=3)
        item['price_config']['price_tiers'].append(second)
        catalog.upsert(item)
        self.assertEqual(catalog.quote(request(inp=10000), 'A', 'CNY', {'USD': 7})['tier_index'], 0)
        self.assertEqual(catalog.quote(request(inp=10001), 'A', 'CNY', {'USD': 7})['tier_index'], 1)

    def test_ambiguous_tiers_and_missing_predictions_are_not_guessed(self):
        item = offering()
        item['price_config']['price_tiers'].append(deepcopy(item['price_config']['price_tiers'][0]))
        catalog = ModelEndpointCatalog([item])
        self.assertEqual(catalog.quote(request(), 'A', 'CNY', {'CNY': 1})['status'], 'ambiguous_price_tier')
        self.assertEqual(catalog.quote(request(out=None), 'A', 'CNY', {'CNY': 1})['status'], 'missing_predicted_length')

    def test_catalog_dynamic_add_remove_is_model_indexed(self):
        catalog = ModelEndpointCatalog([offering()])
        catalog.upsert(offering('B'))
        self.assertEqual(catalog.endpoints('模型BP'), ['A', 'B'])
        self.assertTrue(catalog.remove('模型BP', 'A'))
        self.assertFalse(catalog.remove('模型BP', 'A'))
        self.assertEqual(catalog.endpoints('模型BP'), ['B'])

    def _engine(self, rows, mode='nonstream', cfg=None):
        cfg = cfg or config()
        return RoutingEngine(rules(mode), tolerance(), cfg,
                             ModelEndpointCatalog([offering('A'), offering('B', 2, 4)]),
                             PerformanceIndex(rows, cfg['performance']))

    def test_only_pareto_candidates_fill_slots_and_lengths_affect_cost(self):
        rows = [fact(i, ep) for ep in ('A', 'B') for i in range(4)]
        decision, timing = self._engine(rows).route(request())
        self.assertEqual(decision['slots'], ['A', None, None])
        self.assertFalse(decision['candidates'][1]['pareto'])
        self.assertEqual(timing['decision_ns'], timing['lookup_ns']+timing['prediction_and_cost_ns']+timing['pareto_ns'])

    def test_tolerance_boundary_is_inclusive(self):
        rows = [fact(i, latency=110) for i in range(4)]
        decision, _ = self._engine(rows).route(request())
        self.assertEqual(decision['slots'][0], 'A')
        rows = [fact(i, latency=110.01) for i in range(4)]
        decision, _ = self._engine(rows).route(request())
        self.assertEqual(decision['slots'], [None]*3)

    def test_stream_metric_cannot_be_hidden_by_zero_weight(self):
        rows = [dict(fact(i, mode='stream', latency=10), tpot_proxy_ms=3) for i in range(4)]
        cfg = config()
        cfg['eta_ttft'] = 1
        decision, _ = self._engine(rows, 'stream', cfg).route(request('stream'))
        self.assertEqual(decision['slots'], [None]*3)
        self.assertIn('slo_exceeded:tpot', decision['candidates'][0]['exclusion_reasons'])

    def test_stream_route_with_sufficient_evidence_succeeds(self):
        rows = [fact(i, mode='stream', latency=100) for i in range(4)]
        decision, _ = self._engine(rows, 'stream').route(request('stream'))
        self.assertEqual(decision['slots'], ['A', None, None])

    def test_current_outcome_fields_and_future_values_cannot_change_decision(self):
        rows = [fact(i) for i in range(4)] + [fact(30)]
        first, _ = self._engine(rows).route(request())
        rows[-1]['request_e2e_ms'] = 999999
        changed_request = dict(request(), actual_output_tokens=999999, request_e2e_ms=999999, final_endpoint_id='B', success=False)
        second, _ = self._engine(rows).route(changed_request)
        self.assertEqual(first, second)

    def test_lambda_changes_order_only_inside_frontier(self):
        rows = [fact(i, 'A', latency=100) for i in range(4)] + [fact(i, 'B', latency=10) for i in range(4)]
        cfg = config()
        cfg['lambda_cost'] = 1
        cheap, _ = self._engine(rows, cfg=cfg).route(request())
        cfg['lambda_cost'] = 0
        fast, _ = self._engine(rows, cfg=cfg).route(request())
        self.assertEqual(cheap['slots'][:2], ['A', 'B'])
        self.assertEqual(fast['slots'][:2], ['B', 'A'])


class ReplayArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        facts = [fact(i, 'A' if i % 2 == 0 else 'B', mode=mode) for mode in ('nonstream', 'stream') for i in range(20)]
        write_jsonl(cls.root/'facts.jsonl', facts)
        base = json.loads((PROJECT/'configs/slo_experiment.json').read_text(encoding='utf-8'))
        base['minimum_group_samples'] = 2
        write_json(cls.root/'base_config.json', base)
        write_json(cls.root/'route_config.json', config())
        write_json(cls.root/'tolerance.json', tolerance())
        write_jsonl(cls.root/'offerings.jsonl', [offering('A'), offering('B', 2, 4)])
        cls.baseline, _, result = prepare_slo_experiment(cls.root/'facts.jsonl', cls.root/'baseline', cls.root/'base_config.json')
        if not result['passed']:
            raise AssertionError(result)
        cls.original, data, result = prepare_replay(cls.baseline, cls.root/'offerings.jsonl', cls.root/'replay',
                                                   cls.root/'route_config.json', cls.root/'tolerance.json')
        if not result['passed']:
            raise AssertionError(result)
        cls.data = data

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.case.cleanup)
        self.run_dir = Path(self.case.name)/'run'
        shutil.copytree(self.original, self.run_dir)

    def _rehash(self, relative):
        manifest = json.loads((self.run_dir/'manifest.json').read_text(encoding='utf-8'))
        path = self.run_dir/relative
        manifest['artifacts'][relative] = dict(sha256=file_hash(path), bytes=path.stat().st_size)
        write_json(self.run_dir/'manifest.json', manifest)

    def test_generated_package_figures_and_reproduction(self):
        self.assertTrue(validate_replay(self.run_dir)['passed'])
        self.assertEqual(len(list((self.run_dir/'figures').glob('*.png'))), len(self.data['requests']))

    def test_rehashed_decision_tampering_is_detected(self):
        path = self.run_dir/'decisions.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['slots'] = ['wrong', None, None]
        write_jsonl(path, rows)
        self._rehash('decisions.jsonl')
        self.assertFalse(validate_replay(self.run_dir)['passed'])

    def test_rehashed_bad_timing_is_detected(self):
        path = self.run_dir/'timings.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['decision_ns'] = -1
        write_jsonl(path, rows)
        self._rehash('timings.jsonl')
        self.assertFalse(validate_replay(self.run_dir)['passed'])

    def test_original_package_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            prepare_replay(self.baseline, self.root/'offerings.jsonl', self.original,
                           self.root/'route_config.json', self.root/'tolerance.json')

    def test_sampling_is_based_only_on_incoming_features_and_slo(self):
        inputs = [json.loads(line) for line in (self.baseline/'dataset/test_requests.jsonl').read_text(encoding='utf-8').splitlines()]
        assignments = [json.loads(line) for line in (self.baseline/'test_assignments.jsonl').read_text(encoding='utf-8').splitlines()]
        first = select_requests(inputs, assignments, config())
        for item in assignments:
            item['unread_outcome'] = {'success': False, 'latency': 999999}
        self.assertEqual(first, select_requests(inputs, assignments, config()))

    def test_missing_figure_fails_validation(self):
        next((self.run_dir/'figures').glob('*.png')).unlink()
        self.assertFalse(validate_replay(self.run_dir)['passed'])


if __name__ == '__main__':
    unittest.main()
