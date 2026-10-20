import random
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from routing import CacheAwareRouter, PrefixIndex, actual_cost, cached_tokens, pareto_rank, prompt_key
from experiment import document, followup, gateway_cost, summarize
from validation import billing_audit, paired_bootstrap, report, run_stages
from billing_probe import content_correct, parse_amount, report as probe_report
from supplier_experiment import quote, supplier_fee, run_supplier, report as supplier_report


class PrefixTests(unittest.TestCase):
    def test_exact_longest_prefix_and_scope(self):
        index = PrefixIndex()
        index.observe('tenant', 'm/v1', b'abcdef', 'a', 0)
        index.observe('tenant', 'm/v1', b'abcxyz', 'b', 0)
        self.assertEqual(index.lookup('tenant', 'm/v1', b'abcZZ', 1), [])
        index.observe('tenant', 'm/v1', b'abc', 'c', 1)
        self.assertEqual([r['endpoint_id'] for r in index.lookup('tenant', 'm/v1', b'abcdefg', 2)], ['a', 'c'])
        self.assertFalse(index.lookup('other', 'm/v1', b'abcdef', 2))
        self.assertFalse(index.lookup('tenant', 'm/v2', b'abcdef', 2))

    def test_ttl_does_not_refresh_on_lookup_and_lru_is_bounded(self):
        index = PrefixIndex(ttl_ms=10, max_entries=2)
        index.observe('s', 'm', b'a', 'a', 0)
        index.observe('s', 'm', b'b', 'b', 1)
        index.lookup('s', 'm', b'ax', 2)
        index.observe('s', 'm', b'c', 'c', 3)
        self.assertFalse(index.lookup('s', 'm', b'by', 4))
        self.assertTrue(index.lookup('s', 'm', b'ax', 9))
        self.assertFalse(index.lookup('s', 'm', b'ax', 10))
        index.expire(13)
        self.assertEqual(index.roots, {})

    def test_randomized_tree_matches_naive_oracle_after_eviction(self):
        rng = random.Random(19)
        index = PrefixIndex(ttl_ms=35, max_entries=12)
        for now in range(250):
            key = bytes(rng.choice(b'abcde') for _ in range(rng.randint(1, 12)))
            index.observe('s', 'm', key, rng.choice(['a', 'b', 'c']), now)
            query = bytes(rng.choice(b'abcde') for _ in range(15))
            expected = {(ep, len(k)) for (ns, k, ep) in index.lru if query.startswith(k)}
            actual = {(r['endpoint_id'], r['matched_bytes']) for r in index.lookup('s', 'm', query, now)}
            self.assertEqual(actual, expected)

    def test_canonical_conversation_extension_is_conservative(self):
        messages = [dict(role='user', content='hello ')]
        self.assertTrue(prompt_key(messages+[dict(role='assistant', content='OK')]).startswith(prompt_key(messages)))
        self.assertFalse(prompt_key([dict(role='user', content='hello')]).startswith(prompt_key(messages)))
        self.assertNotEqual(prompt_key(messages, {'tools': []}), prompt_key(messages, {'tools': ['x']}))
        self.assertNotEqual(prompt_key([dict(role='system', content='hello ')]), prompt_key(messages))

    def test_repeated_refresh_does_not_grow_expiry_heap_without_bound(self):
        index = PrefixIndex(max_entries=2)
        for now in range(100):
            index.observe('s', 'm', b'same prefix', 'a', now)
        self.assertLessEqual(len(index.expirations), 4)
        self.assertEqual(len(index.lru), 1)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.router = CacheAwareRouter()
        self.messages = [dict(role='user', content='fixed long document')]
        self.candidates = [dict(endpoint_id='fast', estimated_e2e_ms=1, estimated_cost=2),
                           dict(endpoint_id='cheap', estimated_e2e_ms=4, estimated_cost=1),
                           dict(endpoint_id='dominated', estimated_e2e_ms=5, estimated_cost=3),
                           dict(endpoint_id='slow', estimated_e2e_ms=50, estimated_cost=.1)]

    def choose(self, **kwargs):
        return self.router.choose(**dict(scope='tenant', model='m', messages=self.messages,
                    candidates=self.candidates, slo_ms=10, now_ms=1, **kwargs))

    def warm(self, endpoint, **kwargs):
        return self.router.feedback(scope='tenant', model='m', messages=self.messages,
                    endpoint=endpoint, now_ms=0, **dict(success=True, identity_verified=True, **kwargs))

    def test_no_match_is_original_pareto(self):
        decision = self.choose()
        self.assertEqual(decision['reason'], 'pareto_fallback')
        self.assertEqual(decision['ranked_endpoint_ids'], pareto_rank(self.candidates, 10)[0][:3])
        self.assertNotIn('slow', decision['ranked_endpoint_ids'])

    def test_match_can_select_slo_qualified_dominated_endpoint(self):
        self.warm('dominated')
        result = self.choose()
        self.assertEqual(result['selected_endpoint'], 'dominated')
        self.assertEqual(result['reason'], 'prefix_affinity')

    def test_prefix_does_not_bypass_slo(self):
        self.warm('slow')
        result = self.choose()
        self.assertEqual(result['reason'], 'pareto_fallback')
        self.assertNotEqual(result['selected_endpoint'], 'slow')

    def test_multiple_matching_endpoints_use_original_score(self):
        self.warm('fast'); self.warm('cheap')
        self.assertEqual(self.choose()['selected_endpoint'], 'cheap')

    def test_failed_or_unverified_call_does_not_warm(self):
        for success, verified in ((False, True), (True, False)):
            self.assertFalse(self.router.feedback(scope='tenant', model='m', messages=self.messages,
                endpoint='cheap', now_ms=0, success=success, identity_verified=verified))
        self.assertFalse(self.choose()['matches'])

    def test_no_slo_feasible_means_no_dispatch(self):
        self.candidates = [dict(endpoint_id='a', estimated_e2e_ms=100, estimated_cost=1)]
        self.assertIsNone(self.choose()['selected_endpoint'])

    def test_lookup_does_not_invent_feedback(self):
        self.choose()
        self.assertEqual(len(self.router.index.lru), 0)

    def test_invalid_inputs(self):
        for alpha in (-1, 2, float('nan')):
            with self.assertRaises(ValueError):
                pareto_rank(self.candidates, 10, alpha)

    def test_shorter_qualified_prefix_used_when_longest_fails_slo(self):
        self.warm('cheap')
        extension = self.messages+[dict(role='assistant', content='READY')]
        self.router.feedback(scope='tenant', model='m', messages=extension, endpoint='slow',
                             now_ms=0, success=True, identity_verified=True)
        result = self.router.choose(scope='tenant', model='m', messages=extension,
                    candidates=self.candidates, slo_ms=10, now_ms=1)
        self.assertEqual(result['selected_endpoint'], 'cheap')


class ExperimentTests(unittest.TestCase):
    def test_paired_documents_same_body_but_no_shared_start(self):
        a = document('run', 'pair-official-0-pareto', 'short')
        b = document('run', 'pair-official-0-prefix_pareto', 'short')
        self.assertEqual(a[1], b[1])
        self.assertNotEqual(a[0], b[0])
        self.assertNotEqual(followup(a, 'READY', 1)[-1], followup(a, 'READY', 2)[-1])

    def test_unknown_or_invalid_gateway_cost_is_not_free(self):
        for value in (None, 'bad', 'NaN', '-1'):
            self.assertIsNone(gateway_cost(dict(gateway_headers={'x-litellm-response-cost': value})))

    def test_summary_does_not_turn_missing_cache_price_into_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            row = dict(type='call', phase='comparison', role='followup', policy='pareto',
                       valid=True, cached_tokens=100, slo_ms=100, e2e_ms=10,
                       accounting=dict(cache_adjusted_cost=None, full_input_reference=.2),
                       gateway_headers={'x-litellm-response-cost': '.1'})
            summary = summarize([row], directory)
            self.assertIsNone(summary['routing_cost_reduction'])
            self.assertFalse(summary['comparison']['pareto']['complete_cache_pricing'])
            self.assertEqual(summary['comparison']['pareto']['gateway_reported_total_cost'], .1)
            self.assertTrue((Path(directory)/'summary.md').is_file())


class AccountingTests(unittest.TestCase):
    def test_zero_hit_and_zero_miss_with_nonzero_prompt_is_unknown(self):
        usage = dict(prompt_tokens=1000, completion_tokens=2, prompt_cache_hit_tokens=0,
                     prompt_cache_miss_tokens=0, prompt_tokens_details=dict(cached_tokens=0))
        self.assertIsNone(cached_tokens(usage))
        prices = dict(input_cost_per_token=2e-6,output_cost_per_token=8e-6)
        self.assertIsNone(actual_cost(usage,prices)['cache_adjusted_cost'])

    def test_hit_miss_relationship_is_checked(self):
        usage = dict(prompt_tokens=100, prompt_cache_hit_tokens=20,prompt_cache_miss_tokens=80)
        self.assertEqual(cached_tokens(usage),20)
        usage['prompt_cache_miss_tokens'] = 81
        self.assertIsNone(cached_tokens(usage))

    def test_real_usage_and_cache_price(self):
        usage = dict(prompt_tokens=1000, completion_tokens=100, prompt_cache_hit_tokens=800)
        prices = dict(input_cost_per_token=2e-6, output_cost_per_token=8e-6, cache_read_input_token_cost=4e-8)
        cost = actual_cost(usage, prices)
        self.assertAlmostEqual(cost['cache_adjusted_cost'], .001232)
        self.assertAlmostEqual(cost['cache_saving'], .001568)

    def test_unknown_is_not_zero(self):
        prices = dict(input_cost_per_token=2e-6, output_cost_per_token=8e-6)
        for usage in (dict(prompt_tokens=1000, completion_tokens=1),
                      dict(prompt_tokens=1000, completion_tokens=1, prompt_cache_hit_tokens=2000),
                      dict(prompt_tokens=1000, completion_tokens=1, prompt_cache_hit_tokens=50,
                           prompt_tokens_details=dict(cached_tokens=60))):
            self.assertIsNone(cached_tokens(usage))
            self.assertIsNone(actual_cost(usage, prices)['cache_adjusted_cost'])
        usage = dict(prompt_tokens=1000, completion_tokens=1, prompt_cache_hit_tokens=100)
        self.assertIsNone(actual_cost(usage, prices)['cache_saving'])


class StagedValidationTests(unittest.TestCase):
    def test_supplier_amount_does_not_require_cache_hit_or_use_customer_cost(self):
        row = dict(endpoint='ai360-cdtx',started_at_utc='2026-10-20T03:00:00+00:00',
            usage=dict(prompt_tokens=1000,completion_tokens=2,amount='.001',prompt_cache_hit_tokens=0,
                       prompt_cache_miss_tokens=0),gateway_headers={'x-litellm-response-cost':'999'})
        self.assertEqual(supplier_fee(row)['estimated_procurement_cost_cny'],.001)
        del row['usage']['amount']
        self.assertIsNone(supplier_fee(row)['estimated_procurement_cost_cny'])

    def test_official_supplier_tariff_time_and_cache(self):
        self.assertEqual(quote('official','2026-10-20T03:00:00Z')['input'],2)
        self.assertEqual(quote('official','2026-10-20T05:00:00Z')['input'],1)
        row = dict(endpoint='official',started_at_utc='2026-10-20T03:00:00Z',
                   usage=dict(prompt_tokens=1000,completion_tokens=2,prompt_cache_hit_tokens=800,prompt_cache_miss_tokens=200))
        self.assertAlmostEqual(supplier_fee(row)['estimated_procurement_cost_cny'],.000448)

    def test_supplier_protocol_call_counts_and_provenance_with_mock_transport(self):
        router,events,clock = CacheAwareRouter(),[],[0]
        args = SimpleNamespace(model='model',max_calls=100,max_cost=1.2,pairs_per_cell=3,
                               followups=3,slo_ms=5000,cache_wait=0)
        def call(endpoint,messages,phase,role,family,size,policy,decision=None):
            clock[0]+=1
            row = dict(type='call',trial_id=str(clock[0]),endpoint=endpoint,messages=messages,phase=phase,
                role=role,family=family,size=size,policy=policy,decision=decision,valid=True,transport_success=True,
                endpoint_identity_verified=True,finish_reason='stop',content='READY',e2e_ms=500 if endpoint=='official' else 1000,
                started_at_utc='2026-10-20T03:00:00Z',usage=dict(prompt_tokens=1000,completion_tokens=2,
                    prompt_cache_hit_tokens=800,prompt_cache_miss_tokens=200,amount='.001'),
                gateway_headers={'x-litellm-response-cost':'999'})
            router.feedback(scope=family,model=args.model,messages=messages,endpoint=endpoint,
                now_ms=clock[0],success=True,identity_verified=True)
            events.append(row)
            return row
        with patch('supplier_experiment.time.sleep'):
            run_supplier(args,[dict(endpoint='official'),dict(endpoint='ai360-cdtx')],router,
                         call,events.append,lambda:clock[0],'unit-test')
        with tempfile.TemporaryDirectory() as directory:
            summary = supplier_report(events,directory)
            self.assertEqual(summary['calls'],100)
            self.assertEqual(summary['completed_pairs'],12)
            self.assertEqual(summary['arms']['pareto']['followups'],36)
            self.assertLess(summary['arms']['pareto']['estimated_supplier_total_cny'],1)
            self.assertIsNone(summary['formal_invoice_reduction'])

    def test_direct_probe_parses_content_and_amount_without_currency_claim(self):
        messages = document('test','test-probe','short')
        query = followup(messages,'READY',1)
        import re
        expected = re.search(r'item_0001: region=R\d+; stock=(\d+);',messages[1]['content'])[1]
        self.assertTrue(content_correct(query,expected,'followup'))
        self.assertTrue(content_correct(messages,'READY','warmup'))
        self.assertIsNone(parse_amount(dict(amount='NaN')))
        self.assertEqual(parse_amount(dict(amount='.001')), .001)
        with tempfile.TemporaryDirectory() as directory:
            result = probe_report([dict(type='call',phase='billing_probe',probe_path='direct',role='warmup',
                size='short',trial_id='t',valid=True,usage=dict(prompt_tokens=100,completion_tokens=2,
                prompt_cache_hit_tokens=0,prompt_cache_miss_tokens=0,amount='.001'))],directory)
            self.assertEqual(result['direct_contradictions'],1)
            self.assertFalse(result['provider_invoice_verified'])

    def test_amount_scenario_never_becomes_verified_saving(self):
        events = []
        for policy,value in [('pareto','.002'),('prefix_pareto','.001')]:
            events.append(dict(type='call',phase='large',policy=policy,role='followup',
                family='large-00-'+policy,endpoint='ai360-cdtx',
                usage=dict(prompt_tokens=100,completion_tokens=2,prompt_cache_hit_tokens=0,
                           prompt_cache_miss_tokens=0,amount=value),
                gateway_headers={'x-litellm-response-cost':'.01'}))
        audit = billing_audit(events)
        self.assertIsNone(audit['formal_provider_cost_reduction'])
        self.assertEqual(audit['by_endpoint'][0]['cache_hit_miss_contradictions'],2)
        self.assertEqual(audit['sensitivity_scenario']['conditional_reduction'],.5)

    def test_bootstrap_uses_pairs(self):
        self.assertEqual(paired_bootstrap([(10,5)]*4, repetitions=100), [.5,.5])
        self.assertIsNone(paired_bootstrap([]))

    def test_all_stages_with_mock_transport(self):
        router, events, clock = CacheAwareRouter(), [], [0]
        deployments = [dict(endpoint=e, input_cost_per_token=p, output_cost_per_token=p,
                            cache_read_input_token_cost=None) for e,p in [('a',2e-6),('b',1e-6),('c',3e-6),('d',4e-6)]]
        args = SimpleNamespace(pairs_per_cell=1, followups=1, cache_wait=0, model='model', slo_ms=60000)
        def call(endpoint, messages, phase, role, family, size, policy, decision=None):
            clock[0] += 1
            row = dict(type='call',endpoint=endpoint,messages=messages,phase=phase,role=role,family=family,
                size=size,policy=policy,decision=decision,valid=True,transport_success=True,content='READY',
                endpoint_identity_verified=True,finish_reason='stop',
                e2e_ms=(ord(endpoint)-96)*100, slo_ms=decision['slo_ms'] if decision else args.slo_ms,
                cached_tokens=0,gateway_headers={'x-litellm-response-cost':'.01'})
            router.feedback(scope=family,model=args.model,messages=messages,endpoint=endpoint,
                            now_ms=clock[0],success=True,identity_verified=True)
            events.append(row)
            return row
        with patch('validation.time.sleep'):
            run_stages(args,deployments,router,call,events.append,lambda:clock[0],'unit-test')
        checks = [r for r in events if r['type']=='branch_check']
        self.assertEqual(len(checks),5)
        self.assertTrue(all(r['passed'] for r in checks))
        with tempfile.TemporaryDirectory() as directory:
            result = report(events,directory)
            self.assertEqual(result['completed_pairs'],8)
            self.assertEqual(result['large']['pareto']['planned_followups'],8)
            self.assertEqual(result['large']['pareto']['calls'],16)
            self.assertAlmostEqual(result['gateway_cost_reduction'],0)


if __name__ == '__main__':
    unittest.main()
