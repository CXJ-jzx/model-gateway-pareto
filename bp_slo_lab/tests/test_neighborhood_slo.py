"""Token lookup, latency quantiles, bounded tier factors and causal profiles."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bp_slo.dataset import file_hash, write_json, write_jsonl
from bp_slo.neighborhood_slo import NeighborhoodSLO, fit_profiles, run_experiment, validate_config
from bp_slo.neighborhood_experiment import prepare, validate
from bp_slo.performance import PerformanceIndex
from bp_slo.routing import ModelEndpointCatalog, RoutingEngine
from bp_slo.routing_experiment import prepare_replay, validate_replay
from bp_slo.tolerance import acceptance_limit
from test_routing_replay import fact, offering, request, tolerance


PROJECT=Path(__file__).resolve().parents[1]


def config():
    value=json.loads((PROJECT/'configs/slo_neighborhood.json').read_text(encoding='utf-8'))
    value.update(maximum_local_samples=4,minimum_local_samples=3,tail_fit_samples=4)
    return value


def lookup(rows, cfg=None):
    cfg=cfg or config()
    states=[dict(request_id=r['request_id'],split='train',training_outcome_visible=True) for r in rows]
    return NeighborhoodSLO(fit_profiles(rows,states,cfg))


class NeighborRulesTests(unittest.TestCase):
    def test_relative_e2e_window_has_floor_and_inclusive_upper_bound(self):
        rows=[fact(i,out=n) for i,n in enumerate((10000,10199,10201,11999,12000,12001))]
        result=lookup(rows).assign(request(out=10000))['grades']['strict']['e2e']
        search=result['evidence']['search']
        self.assertEqual(search['upper_tokens'],12000)
        self.assertEqual(search['available'],5)
        self.assertEqual(search['selected'],4)
        self.assertEqual(search['policy'],'relative_width_with_floor')
        short=lookup([fact(0,out=200)]).assign(request(out=100))['grades']['strict']['e2e']
        self.assertEqual(short['evidence']['search']['upper_tokens'],300)

    def test_relative_width_rounds_up_and_does_not_change_ttft(self):
        index=lookup([fact(0,out=1101),fact(1,mode='stream',inp=1101)])
        e2e=index.assign(request(out=1101))['grades']['strict']['e2e']['evidence']['search']
        ttft=index.assign(request('stream',inp=1101))['grades']['strict']['ttft']['evidence']['search']
        self.assertEqual(e2e['upper_tokens'],1322)
        self.assertEqual(ttft['upper_tokens'],1301)
        self.assertNotIn('relative_width',ttft)

    def test_old_config_preserves_fixed_width_and_its_evidence_shape(self):
        cfg=config()
        del cfg['e2e_forward_neighborhood']
        result=lookup([fact(0,out=10000)],cfg).assign(request(out=10000))['grades']['strict']['e2e']
        self.assertEqual(result['evidence']['search']['upper_tokens'],10200)
        self.assertNotIn('policy',result['evidence']['search'])

    def test_invalid_relative_neighborhood_policy_rejected(self):
        for ratio in (0,-.2,1.1,True,float('nan')):
            cfg=config()
            cfg['e2e_forward_neighborhood']['relative_width']=ratio
            with self.assertRaises(ValueError): validate_config(cfg)
        cfg=config()
        cfg['e2e_forward_neighborhood']['minimum_width_tokens']=0
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_latency_quantile_is_not_token_sort_position(self):
        rows=[fact(i,out=300+i,latency=value) for i,value in enumerate((10,2,6))]
        result=lookup(rows).assign(request(out=300))
        self.assertEqual(result['grades']['strict']['e2e']['limit'],6)
        self.assertEqual(result['grades']['standard']['e2e']['limit'],8)
        self.assertAlmostEqual(result['grades']['relaxed']['e2e']['limit'],9.2)

    def test_forward_200_boundary_is_inclusive_and_can_cross_old_bucket(self):
        rows=[fact(i,out=n) for i,n in enumerate((510,512,513,514,711,712,713))]
        result=lookup(rows).assign(request(out=512))['grades']['strict']['e2e']
        self.assertEqual(result['evidence']['search']['available'],5)
        self.assertEqual(result['evidence']['search']['selected'],4)
        selected=set(result['evidence']['source_request_ids'])
        self.assertIn(rows[2]['request_id'],selected)
        self.assertNotIn(rows[0]['request_id'],selected)
        self.assertNotIn(rows[-1]['request_id'],selected)

    def test_equal_token_ties_prefer_recent_not_fastest(self):
        rows=[fact(i,out=300,latency=100+i) for i in range(7)]
        result=lookup(rows).assign(request(out=300))['grades']['strict']['e2e']
        self.assertEqual(result['evidence']['source_request_ids'],[rows[i]['request_id'] for i in (6,5,4,3)])

    def test_one_sample_has_explicit_130_150_fallback(self):
        result=lookup([fact(0,out=300,latency=10000)]).assign(request(out=300))
        values=[result['grades'][g]['e2e']['limit'] for g in ('strict','standard','relaxed')]
        self.assertEqual(values,[10000,13000,15000])
        self.assertTrue(result['grades']['strict']['e2e']['evidence']['fallback'])
        self.assertIsNone(result['grades']['relaxed']['e2e']['quantile'])
        self.assertEqual(acceptance_limit(values[-1],'e2e',tolerance()),16500)

    def test_empty_forward_neighborhood_uses_labelled_fit(self):
        rows=[fact(0,out=100,latency=100),fact(1,out=1000,latency=1000)]
        result=lookup(rows).assign(request(out=400))['grades']['strict']['e2e']
        self.assertEqual(result['status'],'assigned')
        self.assertEqual(result['evidence']['method'],'linear_gap_fallback')
        self.assertFalse(result['evidence']['guarantee'])

    def test_extrapolation_is_monotone_and_marks_far_domain(self):
        index=lookup([fact(i,out=100+i*100,latency=100+i*100) for i in range(4)])
        near=index.assign(request(out=500))['grades']['strict']['e2e']
        far=index.assign(request(out=2000))['grades']['strict']['e2e']
        self.assertTrue(near['evidence']['extrapolated'])
        self.assertEqual(near['evidence']['method'],'tail_linear_extrapolation')
        self.assertGreaterEqual(far['limit'],near['limit'])
        self.assertTrue(far['evidence']['beyond_documented_domain'])

    def test_negative_ols_slope_has_positive_rate_fallback(self):
        index=lookup([fact(i,out=100+i*100,latency=1000-i*100) for i in range(4)])
        result=index.assign(request(out=1000))['grades']['strict']['e2e']
        self.assertGreater(result['evidence']['linear_reference']['slope_ms_per_token'],0)
        self.assertEqual(result['evidence']['linear_reference']['slope_source'],'p75_observed_per_token_rate_fallback')

    def test_cold_start_and_missing_forecast_do_not_crash(self):
        cold=lookup([]).assign(request())['grades']['strict']['e2e']
        self.assertEqual(cold['evidence']['method'],'configured_cold_start_default')
        self.assertTrue(cold['evidence']['business_approval_required'])
        missing=lookup([fact(i) for i in range(4)]).assign(request(out=None))['grades']['strict']['e2e']
        self.assertEqual(missing['evidence']['method'],'missing_prediction_global_p90')

    def test_wrong_model_cannot_get_an_assigned_rule(self):
        result=lookup([fact(0)]).assign(dict(request(),model_id='other'))
        self.assertEqual(result['grades']['strict']['e2e']['status'],'model_mismatch')

    def test_ttft_global_floor_prevents_tiny_local_deadline(self):
        rows=[fact(i,mode='stream',inp=200,latency=10) for i in range(4)]
        rows += [fact(i,mode='stream',inp=2000,latency=1000) for i in range(4,12)]
        index=lookup(rows)
        result=index.assign(request('stream',inp=200))['grades']['strict']['ttft']
        self.assertGreaterEqual(result['limit'],index.rules['pools']['ttft']['global_p90'])
        self.assertGreater(result['evidence']['local_weight'],0)

    def test_tpot_deadline_does_not_change_with_requested_tokens(self):
        index=lookup([fact(i,mode='stream') for i in range(8)])
        first=index.assign(request('stream',inp=200,out=300))
        other=index.assign(request('stream',inp=2000,out=30000))
        for grade in ('strict','standard','relaxed'):
            self.assertEqual(first['grades'][grade]['tpot']['limit'],other['grades'][grade]['tpot']['limit'])
        self.assertEqual(first['grades']['relaxed']['tpot']['tier_factor'],1.5)

    def test_test_labels_do_not_change_training_profiles(self):
        rows=[fact(i) for i in range(20)]
        original=run_experiment(rows,config())['rules']
        for row in rows[14:]:
            row['request_e2e_ms']=1000000
            row['actual_output_tokens']=100000
        self.assertEqual(run_experiment(rows,config())['rules'],original)

    def test_invalid_large_tier_factors_and_index_counts_rejected(self):
        cfg=config()
        cfg['fallback_factors']['relaxed']=3
        with self.assertRaises(ValueError): validate_config(cfg)
        cfg=config()
        cfg['minimum_local_samples']=0
        with self.assertRaises(ValueError): validate_config(cfg)

    def test_router_applies_tolerance_once_and_uses_persistent_lookup(self):
        rows=[fact(i) for i in range(8)]
        index=lookup(rows)
        cfg=json.loads((PROJECT/'configs/routing_neighborhood.json').read_text(encoding='utf-8'))
        cfg['performance'].update(target_samples=4,minimum_samples=3,minimum_effective_samples=2,window_candidates_minutes=[60])
        engine=RoutingEngine(index.rules,tolerance(),cfg,ModelEndpointCatalog([offering()]),PerformanceIndex(rows,cfg['performance']))
        decision,_=engine.route(request())
        self.assertIsInstance(engine.slo_lookup,NeighborhoodSLO)
        limit=decision['slo']['e2e']['limit']
        self.assertAlmostEqual(decision['slo']['e2e']['acceptance_limit'],limit*1.1)

    def test_tpot_endpoint_estimate_keeps_input_but_not_output_condition(self):
        rows=[fact(i,mode='stream',out=5000) for i in range(8)]
        cfg=json.loads((PROJECT/'configs/routing_neighborhood.json').read_text(encoding='utf-8'))['performance']
        cfg.update(target_samples=4,minimum_samples=3,minimum_effective_samples=2,window_candidates_minutes=[60])
        index=PerformanceIndex(rows,cfg)
        req=request('stream',out=300)
        index.advance(req['arrived_at'])
        estimate=index.estimate(req,'A','tpot')
        self.assertEqual(estimate['status'],'estimated')
        self.assertEqual(set(estimate['bands']),{'input'})


class NeighborArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root=Path(cls.tmp.name)
        facts=[fact(i,mode=mode) for mode in ('stream','nonstream') for i in range(20)]
        write_jsonl(cls.root/'facts.jsonl',facts)
        write_json(cls.root/'config.json',config())
        write_json(cls.root/'tolerance.json',tolerance())
        write_jsonl(cls.root/'offerings.jsonl',[offering()])
        cls.original,_,result=prepare(cls.root/'facts.jsonl',cls.root/'original',cls.root/'config.json',cls.root/'tolerance.json')
        if not result['passed']: raise AssertionError(result)

    def setUp(self):
        self.case=tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.case.cleanup)
        self.run=Path(self.case.name)/'run'
        shutil.copytree(self.original,self.run)

    def test_package_reproduces(self):
        self.assertTrue(validate(self.run)['passed'])

    def test_rehashed_slo_tampering_is_detected(self):
        path=self.run/'test_assignments.jsonl'
        rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        first=next(iter(rows[0]['grades']['strict'].values()))
        first['limit']=999999
        write_jsonl(path,rows)
        manifest=json.loads((self.run/'manifest.json').read_text(encoding='utf-8'))
        manifest['artifacts']['test_assignments.jsonl']=dict(sha256=file_hash(path),bytes=path.stat().st_size)
        write_json(self.run/'manifest.json',manifest)
        self.assertFalse(validate(self.run)['passed'])

    def test_existing_run_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            prepare(self.root/'facts.jsonl',self.original,self.root/'config.json',self.root/'tolerance.json')

    def test_new_profile_can_be_used_by_replay_and_validator(self):
        cfg=json.loads((PROJECT/'configs/routing_neighborhood.json').read_text(encoding='utf-8'))
        cfg['performance'].update(target_samples=4,minimum_samples=3,minimum_effective_samples=2,window_candidates_minutes=[60])
        cfg['sample_requests_per_group']=1
        write_json(self.root/'route_config.json',cfg)
        root,_,result=prepare_replay(self.original,self.root/'offerings.jsonl',self.run/'replay',self.root/'route_config.json',self.root/'tolerance.json')
        self.assertTrue(result['passed'])
        self.assertTrue(validate_replay(root)['passed'])


if __name__=='__main__': unittest.main()
