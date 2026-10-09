"""Typical latency, explicit tolerance, dominated fill, and evidence fallbacks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from bp_slo.dataset import file_hash, write_json, write_jsonl
from bp_slo.pareto import Candidate, rank_frontier
from bp_slo.performance import PerformanceIndex, weighted_mean
from bp_slo.routing import ModelEndpointCatalog, RoutingEngine, validate_routing_config
from bp_slo.routing_experiment import prepare_replay, validate_replay
from bp_slo.slo_experiment import prepare_slo_experiment
from test_routing_replay import PROJECT, config, fact, offering, request, rules, tolerance


def available_config(statistic='mean'):
    cfg = config()
    cfg['performance'].update(statistic=statistic, quantile=None if statistic == 'mean' else .5)
    cfg['selection_policy'] = dict(slo_limit_multiplier=1.2, fill_dominated=True,
        low_evidence_fallback='only_if_no_regular_candidates', allow_unknown_performance=True)
    return cfg


def engine(rows, cfg=None, items=None, mode='nonstream'):
    cfg = cfg or available_config()
    return RoutingEngine(rules(mode), tolerance(), cfg,
                         ModelEndpointCatalog(items or [offering('A'), offering('B', 2, 4)]),
                         PerformanceIndex(rows, cfg['performance']))


class AvailableRoutingTests(unittest.TestCase):
    def test_weighted_mean_and_p50_are_not_p95(self):
        self.assertAlmostEqual(weighted_mean([(10, 1), (100, 3)]), 77.5)
        self.assertIsNone(weighted_mean([]))
        rows = [fact(i, latency=value) for i, value in enumerate([10, 10, 10, 100])]
        for statistic in ('mean', 'p50'):
            index = PerformanceIndex(rows, available_config(statistic)['performance'])
            index.advance(request()['arrived_at'])
            result = index.estimate(request(), 'A', 'e2e')
            self.assertLess(result['estimate'], result['exploratory_p95'])
            self.assertEqual(result['statistic'], statistic)
            if statistic == 'p50':
                self.assertEqual(result['estimate'], 10)

    def test_multiplier_replaces_tolerance_and_boundary_is_inclusive(self):
        rows = [fact(i, latency=120) for i in range(4)]
        result, _ = engine(rows, items=[offering('A')]).route(request())
        self.assertEqual(result['slo']['e2e']['acceptance_limit'], 120)
        self.assertEqual(result['slots'], ['A', None, None])
        result, _ = engine([fact(i, latency=120.001) for i in range(4)], items=[offering('A')]).route(request())
        self.assertEqual(result['slots'], [None]*3)

    def test_multiplier_can_be_changed_to_1_1(self):
        cfg = available_config()
        cfg['selection_policy']['slo_limit_multiplier'] = 1.1
        result, _ = engine([fact(i, latency=115) for i in range(4)], cfg, [offering('A')]).route(request())
        self.assertEqual(result['slots'], [None]*3)
        self.assertAlmostEqual(result['slo']['e2e']['acceptance_limit'], 110)

    def test_dominated_endpoints_fill_without_being_relabelled_frontier(self):
        rows = [fact(i, endpoint) for endpoint in ('A', 'B') for i in range(4)]
        result, _ = engine(rows).route(request())
        self.assertEqual(result['slots'], ['A', 'B', None])
        self.assertEqual(result['frontier_ranked'], ['A'])
        self.assertFalse(result['candidates'][1]['pareto'])
        self.assertEqual(result['candidates'][1]['selection_reason'], 'dominated_fill')

    def test_frontier_always_precedes_dominated_and_three_slots(self):
        points = [Candidate('A', 1, .4), Candidate('B', 2, .5), Candidate('C', 3, .6),
                  Candidate('D', 4, .7)]
        ranked = rank_frontier(points, .5, True)
        self.assertEqual(ranked['slots'], ['A', 'B', 'C'])
        self.assertEqual(ranked['frontier_ranked'], ['A'])

    def test_sparse_fallback_when_no_regular_candidates(self):
        result, _ = engine([fact(0)]).route(request())
        self.assertEqual(result['slots'], ['A', 'B', None])
        self.assertEqual(result['status'], 'selected_low_evidence')
        row = result['candidates'][0]
        self.assertEqual(row['evidence_quality'], 'sparse')
        self.assertIsNone(row['estimates']['e2e']['estimate'])
        self.assertEqual(row['routing_estimates']['e2e'], 100)
        self.assertFalse(row['feasible'])
        self.assertIn('best_effort_no_slo_guarantee', row['risk_warnings'])
        self.assertIn('sparse_evidence_fallback', row['selection_reason'])

    def test_unknown_cold_start_is_not_fake_pareto_performance(self):
        result, _ = engine([]).route(request())
        self.assertEqual(result['slots'], ['A', 'B', None])
        self.assertEqual(result['frontier_ranked'], [])
        self.assertEqual(result['status'], 'selected_low_evidence')
        for row in result['candidates']:
            self.assertIsNone(row['performance_loss'])
            self.assertIsNone(row['score'])
            self.assertFalse(row['pareto'])
            self.assertEqual(row['selection_reason'], 'unknown_performance_best_effort_fallback')

    def test_sparse_is_not_used_to_fill_regular_pool(self):
        rows = [fact(i, 'A') for i in range(4)] + [fact(0, 'B')]
        result, _ = engine(rows).route(request())
        self.assertEqual(result['slots'], ['A', None, None])
        self.assertEqual(result['status'], 'selected')

    def test_regular_dominated_fill_can_be_disabled(self):
        cfg = available_config()
        cfg['selection_policy']['fill_dominated'] = False
        rows = [fact(i, endpoint) for endpoint in ('A', 'B') for i in range(4)]
        result, _ = engine(rows, cfg).route(request())
        self.assertEqual(result['slots'], ['A', None, None])

    def test_fallback_and_unknown_switches(self):
        cfg = available_config()
        cfg['selection_policy']['allow_unknown_performance'] = False
        self.assertEqual(engine([], cfg).route(request())[0]['slots'], [None]*3)
        cfg['selection_policy']['low_evidence_fallback'] = 'disabled'
        self.assertEqual(engine([fact(0)], cfg).route(request())[0]['slots'], [None]*3)

    def test_known_exceeded_sparse_disabled_and_bad_prices_cannot_be_rescued(self):
        items = [offering('A'), offering('B')]
        items[1]['enabled'] = False
        result, _ = engine([fact(0, latency=200)], items=items).route(request())
        self.assertEqual(result['slots'], [None]*3)
        self.assertIn('slo_exceeded:e2e', result['candidates'][0]['exclusion_reasons'])
        items[0]['price_config']['price_tiers'].append(deepcopy(items[0]['price_config']['price_tiers'][0]))
        result, _ = engine([], items=items).route(request())
        self.assertEqual(result['slots'], [None]*3)

    def test_stream_known_excess_not_hidden_by_weight_or_missing_metric(self):
        cfg = available_config()
        cfg['eta_ttft'] = 1
        rows = [dict(fact(0, mode='stream', latency=10), tpot_proxy_ms=3)]
        result, _ = engine(rows, cfg, [offering('A')], 'stream').route(request('stream'))
        self.assertEqual(result['slots'], [None]*3)
        self.assertIn('slo_exceeded:tpot', result['candidates'][0]['exclusion_reasons'])

    def test_unknown_partial_metrics_are_explicit(self):
        rows = [dict(fact(0, mode='stream', latency=100), eligible_tpot_output=False)]
        result, _ = engine(rows, items=[offering('A')], mode='stream').route(request('stream'))
        self.assertEqual(result['status'], 'selected_low_evidence')
        self.assertIsNone(result['candidates'][0]['routing_estimates']['tpot'])
        self.assertEqual(result['candidates'][0]['slo_gate_status'], 'unknown')

    def test_future_and_current_labels_cannot_change_fallback(self):
        rows = [fact(0), fact(30)]
        first, _ = engine(rows).route(request())
        rows[-1]['request_e2e_ms'] = 1e9
        second, _ = engine(rows).route(dict(request(), request_e2e_ms=1e9, success=False))
        self.assertEqual(first, second)

    def test_invalid_policy_and_statistics_are_rejected(self):
        for field, value in [('slo_limit_multiplier', 3), ('fill_dominated', 1),
                             ('low_evidence_fallback', 'always'), ('allow_unknown_performance', 'yes')]:
            cfg = available_config()
            cfg['selection_policy'][field] = value
            with self.assertRaises(ValueError):
                validate_routing_config(cfg)
        cfg = available_config()
        cfg['performance']['quantile'] = .95
        with self.assertRaises(ValueError):
            validate_routing_config(cfg)

    def test_available_replay_is_reproducible_and_rehashed_risk_tampering_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [fact(i, 'A' if i % 2 == 0 else 'B', mode=mode)
                    for mode in ('nonstream', 'stream') for i in range(20)]
            write_jsonl(root/'facts.jsonl', rows)
            base = json.loads((PROJECT/'configs/slo_experiment.json').read_text(encoding='utf-8'))
            base['minimum_group_samples'] = 2
            write_json(root/'base.json', base)
            baseline, _, check = prepare_slo_experiment(root/'facts.jsonl', root/'base', root/'base.json')
            self.assertTrue(check['passed'])
            write_json(root/'route.json', available_config())
            write_json(root/'tolerance.json', tolerance())
            write_jsonl(root/'offerings.jsonl', [offering('A'), offering('B')])
            replay, _, check = prepare_replay(baseline, root/'offerings.jsonl', root/'replay',
                                             root/'route.json', root/'tolerance.json')
            self.assertTrue(check['passed'], check)
            decisions = [json.loads(line) for line in (replay/'decisions.jsonl').read_text(encoding='utf-8').splitlines()]
            decisions[0]['candidates'][0]['evidence_quality'] = 'fabricated'
            write_jsonl(replay/'decisions.jsonl', decisions)
            manifest = json.loads((replay/'manifest.json').read_text(encoding='utf-8'))
            path = replay/'decisions.jsonl'
            manifest['artifacts']['decisions.jsonl'] = dict(sha256=file_hash(path), bytes=path.stat().st_size)
            write_json(replay/'manifest.json', manifest)
            self.assertFalse(validate_replay(replay)['passed'])


if __name__ == '__main__':
    unittest.main()
