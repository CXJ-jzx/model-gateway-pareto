"""Offline checks of the saved REAL experiment. No HTTP, SSH, or model calls.

Run: python -B -m unittest discover -s . -p "test_*.py" -v
The raw log is read-only. These checks verify evidence and decision replay,
not the live availability of a provider or a formally authenticated invoice.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import unittest
import zipfile

from routing import CacheAwareRouter, cached_tokens
from supplier_experiment import supplier_fee


EVIDENCE = Path(__file__).resolve().parent / 'real_experiment' / '20261020-supplier-cost'


class RealExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = [json.loads(line) for line in
                      (EVIDENCE / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        cls.calls = [r for r in cls.events if r.get('type') == 'call']
        cls.comparisons = [r for r in cls.calls if r['phase'] == 'supplier_comparison']
        cls.summary = json.loads((EVIDENCE / 'summary.json').read_text(encoding='utf-8'))
        cls.protocol = json.loads((EVIDENCE / 'protocol.json').read_text(encoding='utf-8'))

    def test_call_counts_and_balanced_plan(self):
        self.assertEqual(len(self.calls), 100)
        self.assertEqual(len(self.comparisons), 96)
        plan = next(r['pairs'] for r in self.events if r.get('type') == 'supplier_plan')
        self.assertEqual(Counter((r['warm_endpoint'], r['size']) for r in plan),
                         {('official', 'short'): 3, ('official', 'long'): 3,
                          ('ai360-cdtx', 'short'): 3, ('ai360-cdtx', 'long'): 3})
        for policy in ('pareto', 'prefix_pareto'):
            rows = [r for r in self.comparisons if r['policy'] == policy]
            self.assertEqual(Counter(r['role'] for r in rows), {'warmup': 12, 'followup': 36})

    def test_calls_are_unique_verified_and_not_response_cache(self):
        self.assertEqual(len({r['response_id'] for r in self.calls}), 100)
        self.assertEqual(len({r['trial_id'] for r in self.calls}), 100)
        for row in self.calls:
            with self.subTest(trial=row['trial_id']):
                self.assertTrue(row['valid'])
                self.assertTrue(row['endpoint_identity_verified'])
                self.assertEqual(row['http_status'], 200)
                self.assertEqual(row['finish_reason'], 'stop')
                self.assertFalse(row['response_cache_hit'])
                self.assertEqual(row['gateway_headers']['x-litellm-attempted-retries'], '0')

    def test_answers_are_independently_correct(self):
        for row in self.calls:
            if row['role'] == 'warmup':
                self.assertEqual(row['content'].strip().upper(), 'READY')
            else:
                item = re.search(r'item_(\d+)', row['messages'][-1]['content'])[1]
                stock = re.search(rf'item_{item}: region=R\d+; stock=(\d+);',
                                  row['messages'][1]['content'])[1]
                self.assertEqual(int(row['content'].strip()), int(stock))

    def test_paired_documents_share_body_but_isolate_prefix(self):
        groups = {}
        for row in self.comparisons:
            if row['role'] == 'warmup':
                pair = row['family'].removesuffix('-pareto').removesuffix('-prefix_pareto')
                groups.setdefault(pair, {})[row['policy']] = row
        self.assertEqual(len(groups), 12)
        for arms in groups.values():
            a, b = arms['pareto'], arms['prefix_pareto']
            self.assertEqual(a['endpoint'], b['endpoint'])
            self.assertEqual(a['messages'][1], b['messages'][1])
            self.assertNotEqual(a['messages'][0], b['messages'][0])

    def test_actual_token_ranges_and_slo(self):
        for row in self.comparisons:
            inp = row['usage']['prompt_tokens']
            self.assertTrue(2450 <= inp <= 2471 if row['size'] == 'short' else 9650 <= inp <= 9671)
            self.assertEqual(row['usage']['completion_tokens'], 2)
            if row['role'] == 'followup':
                self.assertEqual(row['slo_ms'], 5000)
                self.assertLessEqual(row['e2e_ms'], row['slo_ms'])
                self.assertEqual(row['decision']['alpha'], .8)

    def test_replay_decisions_before_feedback(self):
        """Use recorded pre-call candidates, not this call's outcome, to choose."""
        router = CacheAwareRouter()
        for row in self.calls:
            start = datetime.fromisoformat(row['started_at_utc']).timestamp() * 1000
            saved = row.get('decision')
            if saved:
                actual = router.choose(
                    scope=row['family'], model='deepseek-flash', messages=row['messages'],
                    candidates=saved['candidates'], slo_ms=saved['slo_ms'], now_ms=start,
                    alpha=saved['alpha'], prefix_first=row['policy'] == 'prefix_pareto')
                self.assertEqual(actual['selected_endpoint'], saved['selected_endpoint'])
                self.assertEqual(actual['ranked_endpoint_ids'], saved['ranked_endpoint_ids'])
                self.assertEqual(actual['reason'], saved['reason'])
            router.feedback(scope=row['family'], model='deepseek-flash', messages=row['messages'],
                            endpoint=row['endpoint'],
                            now_ms=datetime.fromisoformat(row['finished_at_utc']).timestamp() * 1000,
                            success=row['valid'], identity_verified=row['endpoint_identity_verified'])

    def test_recompute_costs_from_supplier_evidence(self):
        for policy in ('pareto', 'prefix_pareto'):
            rows = [r for r in self.comparisons if r['policy'] == policy]
            fees = [supplier_fee(r)['estimated_procurement_cost_cny'] for r in rows]
            self.assertTrue(all(value is not None for value in fees))
            self.assertAlmostEqual(sum(fees), self.summary['arms'][policy]['estimated_supplier_total_cny'])
            warm = sum(supplier_fee(r)['estimated_procurement_cost_cny'] for r in rows if r['role'] == 'warmup')
            self.assertAlmostEqual(warm, self.summary['arms'][policy]['warmup_cost_cny'])
        arms = self.summary['arms']
        reduction = 1 - arms['prefix_pareto']['estimated_supplier_total_cny'] / arms['pareto']['estimated_supplier_total_cny']
        self.assertAlmostEqual(reduction, .06919388062496445)
        self.assertIsNone(self.summary['formal_invoice_reduction'])

    def test_recompute_latency_and_endpoint_distribution(self):
        for policy, choices in [('pareto', {'official': 36}),
                                ('prefix_pareto', {'official': 18, 'ai360-cdtx': 18})]:
            rows = [r for r in self.comparisons if r['policy'] == policy and r['role'] == 'followup']
            self.assertEqual(dict(Counter(r['endpoint'] for r in rows)), choices)
            self.assertAlmostEqual(statistics.mean(r['e2e_ms'] for r in rows),
                                   self.summary['arms'][policy]['mean_success_ms'])

    def test_360_unknown_cache_counters_do_not_mean_zero_benefit(self):
        rows = [r for r in self.comparisons if r['endpoint'] == 'ai360-cdtx' and r['role'] == 'followup']
        self.assertEqual(len(rows), 18)
        for row in rows:
            self.assertIsNone(cached_tokens(row['usage']))
            fee = supplier_fee(row)
            self.assertEqual(fee['source'], 'supplier_reported_usage_amount')
            self.assertIsNotNone(fee['estimated_procurement_cost_cny'])

    def test_frozen_execution_sources_match_protocol(self):
        # Frozen executed source, not today's report-only corrections, is the
        # authority for historical code hashes.
        with zipfile.ZipFile(EVIDENCE / 'executed_source.zip') as archive:
            self.assertEqual(set(archive.namelist()), set(self.protocol['source_sha256']))
            for name, expected in self.protocol['source_sha256'].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)

    def test_gateway_was_not_reconfigured(self):
        finished = next(r for r in reversed(self.events) if r.get('type') == 'finished')
        self.assertTrue(finished['config_unchanged'])
        self.assertEqual(finished['calls'], 100)


if __name__ == '__main__':
    unittest.main()
