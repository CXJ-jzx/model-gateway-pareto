"""Check leakage barriers, mock forecasts, lookup and coverage denominators."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bp_slo.dataset import file_hash, write_json, write_jsonl
from bp_slo.slo import assign_slo, fit_rules, mock_test_request, run_experiment, split_train_test, validate_config
from bp_slo.slo_experiment import prepare_slo_experiment, validate_slo_experiment


def config():
    value = json.loads((Path(__file__).resolve().parents[1]/'configs/slo_experiment.json').read_text(encoding='utf-8'))
    value['minimum_group_samples'] = 2
    return value


def fact(index, mode='nonstream'):
    start = datetime(2026, 9, 21, tzinfo=timezone.utc)+timedelta(minutes=index)
    return dict(request_id=f'{mode}_{index}', model_id='模型BP', stream_type=mode,
                arrived_at=start.isoformat(), finished_at=(start+timedelta(seconds=1)).isoformat(),
                success=True, actual_input_tokens=200, actual_output_tokens=100, quality_flags=[],
                request_e2e_ms=100+index*10, ttft_proxy_ms=50+index if mode == 'stream' else None,
                tpot_proxy_ms=2+index if mode == 'stream' else None,
                eligible_length_e2e=True, eligible_ttft_input=mode == 'stream', eligible_tpot_output=mode == 'stream')


class SloRulesTests(unittest.TestCase):
    def test_modes_have_seventy_thirty_split_and_no_calibration(self):
        rows = [fact(i, mode) for mode in ('stream', 'nonstream') for i in range(10)]
        splits, meta = split_train_test(list(reversed(rows)))
        for mode in ('stream', 'nonstream'):
            self.assertEqual(meta[mode]['counts'], {'train': 7, 'test': 3})
        self.assertNotIn('calibration', {r['split'] for r in splits})

    def test_tied_arrivals_never_cross_train_boundary(self):
        rows = [fact(i) for i in range(10)]
        for i in (6, 7, 8):
            rows[i]['arrived_at'] = rows[6]['arrived_at']
        splits, _ = split_train_test(rows)
        self.assertEqual({s['split'] for s in splits if s['request_id'] in ('nonstream_6', 'nonstream_7', 'nonstream_8')}, {'test'})

    def test_invalid_modes_and_times_are_retained_unassigned(self):
        rows = [fact(i) for i in range(10)]
        rows += [fact(20, 'unknown'), dict(fact(21), arrived_at=None)]
        splits, _ = split_train_test(rows)
        self.assertEqual([s['split'] for s in splits[-2:]], ['unassigned', 'unassigned'])

    def test_duplicate_ids_and_invalid_fractions_are_rejected(self):
        with self.assertRaises(ValueError):
            split_train_test([fact(1), fact(1)])
        for value in (0, 1, True, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                split_train_test([fact(1)], value)

    def test_quantile_thresholds_use_train_only(self):
        rows = [fact(i) for i in range(10)]
        data = run_experiment(rows, config())
        rule = next(r for r in data['rules']['rules'] if r['metric'] == 'e2e' and r['range'] == '0-512')
        self.assertEqual(rule['n'], 7)
        self.assertEqual(rule['thresholds'], {'p50': 130, 'p75': 145, 'p95': 157})
        altered = deepcopy(rows)
        for row in altered[7:]:
            row['request_e2e_ms'] = 1e9
            row['actual_output_tokens'] = 1_000_000
        self.assertEqual(run_experiment(altered, config())['rules'], data['rules'])

    def test_train_outcomes_at_or_after_cutoff_are_not_fitted(self):
        rows = [fact(i) for i in range(10)]
        rows[0]['finished_at'] = rows[7]['arrived_at']
        rows[1]['finished_at'] = fact(100)['finished_at']
        rows[2]['finished_at'] = None
        splits, _ = split_train_test(rows)
        self.assertTrue(all(s['split'] == 'train' and not s['training_outcome_visible'] for s in splits[:3]))
        fitted = fit_rules(rows, splits, config())
        rule = next(r for r in fitted['rules'] if r['metric'] == 'e2e' and r['range'] == '0-512')
        self.assertEqual(rule['training_request_ids'], [f'nonstream_{i}' for i in range(3, 7)])

    def test_failed_invalid_and_estimated_train_facts_do_not_fit(self):
        rows = [fact(i) for i in range(10)]
        rows[0]['success'] = False
        rows[1]['eligible_length_e2e'] = False
        rows[2]['quality_flags'] = ['token_counts_locally_estimated']
        rule = next(r for r in run_experiment(rows, config())['rules']['rules'] if r['metric'] == 'e2e' and r['range'] == '0-512')
        self.assertEqual(rule['n'], 4)

    def test_sparse_rules_and_missing_lengths_do_not_fallback(self):
        cfg = config()
        cfg['minimum_group_samples'] = 100
        data = run_experiment([fact(i) for i in range(10)], cfg)
        self.assertEqual(data['assignments'][0]['grades']['p50']['e2e']['status'], 'insufficient_evidence')
        self.assertIsNone(data['coverage']['summary'][0]['coverage'])
        request = dict(data['test_inputs'][0], predicted_output_tokens=None)
        self.assertEqual(assign_slo(request, data['rules'])['grades']['p50']['e2e']['status'], 'missing_predicted_length')

    def test_bucket_boundaries_use_predicted_length(self):
        rows = [fact(i) for i in range(10)]
        for row in rows[:3]:
            row['actual_output_tokens'] = 600
            row['request_e2e_ms'] = 1000
        data = run_experiment(rows, config())
        request = dict(data['test_inputs'][0], predicted_output_tokens=512)
        self.assertEqual(assign_slo(request, data['rules'])['grades']['p50']['e2e']['range'], '0-512')
        request['predicted_output_tokens'] = 513
        self.assertEqual(assign_slo(request, data['rules'])['grades']['p50']['e2e']['limit'], 1000)

    def test_assignment_ignores_outcome_fields_and_rejects_other_model(self):
        data = run_experiment([fact(i) for i in range(10)], config())
        request = data['test_inputs'][0]
        changed = dict(request, success=False, request_e2e_ms=1e9, actual_output_tokens=1_000_000, final_endpoint_id='other')
        self.assertEqual(assign_slo(request, data['rules']), assign_slo(changed, data['rules']))
        self.assertEqual(assign_slo(dict(request, model_id='other'), data['rules'])['grades']['p50']['e2e']['status'], 'model_mismatch')

    def test_mock_is_deterministic_bounded_order_independent_and_input_clean(self):
        cfg = config()
        rows = [fact(i) for i in range(100)]
        first = {r['request_id']: mock_test_request(r, cfg) for r in rows}
        second = {r['request_id']: mock_test_request(r, cfg) for r in reversed(rows)}
        self.assertEqual(first, second)
        for row in rows:
            request = first[row['request_id']]
            self.assertNotIn('success', request)
            self.assertNotIn('actual_output_tokens', request)
            self.assertNotIn('request_e2e_ms', request)
            for field in ('input', 'output'):
                self.assertLessEqual(abs(request[f'predicted_{field}_tokens']-row[f'actual_{field}_tokens']), 50)

    def test_mock_clips_zero_and_preserves_missing_values(self):
        row = dict(fact(1), actual_input_tokens=0, actual_output_tokens=None)
        request = mock_test_request(row, config())
        self.assertGreaterEqual(request['predicted_input_tokens'], 0)
        self.assertLessEqual(request['predicted_input_tokens'], 50)
        self.assertIsNone(request['predicted_output_tokens'])
        self.assertIsNone(request['prediction_noise']['output']['draw'])

    def test_coverage_denominator_keeps_excluded_cases_visible(self):
        rows = [fact(i) for i in range(10)]
        rows[7]['success'] = False
        rows[8]['request_e2e_ms'] = None
        rows[9]['request_e2e_ms'] = 100
        data = run_experiment(rows, config())
        for summary in data['coverage']['summary'][:3]:
            self.assertEqual(summary['total'], 3)
            self.assertEqual(summary['evaluated'], 1)
            self.assertEqual(summary['covered'], 1)
            self.assertEqual(summary['coverage'], 1)
            self.assertEqual(summary['evaluation_status_counts'], {'request_failed': 1, 'missing_metric': 1, 'covered': 1})

    def test_stream_tpot_is_global_and_joint_is_both_metrics(self):
        rows = [fact(i, 'stream') for i in range(10)]
        rows[7]['ttft_proxy_ms'], rows[7]['tpot_proxy_ms'] = 40, 1
        rows[8]['ttft_proxy_ms'], rows[8]['tpot_proxy_ms'] = 40, 100
        rows[9]['ttft_proxy_ms'], rows[9]['tpot_proxy_ms'] = 1000, 1
        data = run_experiment(rows, config())
        tpot_rules = [r for r in data['rules']['rules'] if r['metric'] == 'tpot']
        self.assertEqual(len(tpot_rules), 1)
        self.assertEqual(tpot_rules[0]['range'], 'all')
        p50 = {r['metric']: r for r in data['coverage']['summary'] if r['grade'] == 'p50' and r['mode'] == 'stream'}
        self.assertAlmostEqual(p50['ttft']['coverage'], 2/3)
        self.assertAlmostEqual(p50['tpot']['coverage'], 2/3)
        self.assertAlmostEqual(p50['joint']['coverage'], 1/3)

    def test_no_test_samples_means_null_coverage(self):
        data = run_experiment([fact(i) for i in range(10)], config())
        stream = [r for r in data['coverage']['summary'] if r['mode'] == 'stream']
        self.assertTrue(all(r['coverage'] is None and r['evaluated'] == 0 for r in stream))

    def test_config_does_not_allow_calibration_or_different_stage(self):
        for change in ({'calibration_fraction': 0}, {'stage': 'pareto'}, {'quantiles': {'p50': .5, 'p95': .95}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config({**config(), **change})


class SloArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.base = Path(cls.tmp.name)
        source = cls.base/'source.jsonl'
        write_jsonl(source, [fact(i, mode) for mode in ('nonstream', 'stream') for i in range(10)])
        cfg = cls.base/'config.json'
        write_json(cfg, config())
        cls.original, _, validation = prepare_slo_experiment(source, cls.base/'original', cfg)
        if not validation['passed']:
            raise AssertionError(validation)
        cls.source, cls.config_path = source, cfg

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.base)
        self.addCleanup(self.case.cleanup)
        self.run = Path(self.case.name)/'run'
        shutil.copytree(self.original, self.run)

    def _rehash(self, relative):
        manifest = json.loads((self.run/'manifest.json').read_text(encoding='utf-8'))
        manifest['artifacts'][relative] = dict(sha256=file_hash(self.run/relative), bytes=(self.run/relative).stat().st_size)
        write_json(self.run/'manifest.json', manifest)

    def test_real_pipeline_package_reproduces(self):
        self.assertTrue(validate_slo_experiment(self.run)['passed'])
        self.assertTrue((self.run/'REPORT.md').is_file())
        self.assertTrue((self.run/'figures/coverage.png').is_file())

    def test_existing_run_is_not_overwritten(self):
        original_hash = file_hash(self.original/'slo_rules.json')
        with self.assertRaises(FileExistsError):
            prepare_slo_experiment(self.source, self.original, self.config_path)
        self.assertEqual(original_hash, file_hash(self.original/'slo_rules.json'))

    def test_prediction_tampering_even_rehashed_is_detected(self):
        path = self.run/'dataset/test_requests.jsonl'
        rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['predicted_output_tokens'] += 1
        write_jsonl(path, rows)
        self._rehash('dataset/test_requests.jsonl')
        result = validate_slo_experiment(self.run)
        self.assertFalse(result['passed'])
        self.assertTrue(any(c['name'] == 'slo_experiment_reproduction' and not c['passed'] for c in result['checks']))

    def test_coverage_tampering_even_rehashed_is_detected(self):
        path = self.run/'coverage.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['summary'][0]['coverage'] = .12345
        write_json(path, value)
        self._rehash('coverage.json')
        self.assertFalse(validate_slo_experiment(self.run)['passed'])

    def test_artifact_hash_and_missing_file_fail_without_raise(self):
        (self.run/'test_assignments.jsonl').write_text('{}\n', encoding='utf-8')
        self.assertFalse(validate_slo_experiment(self.run)['passed'])
        (self.run/'slo_rules.json').unlink()
        self.assertFalse(validate_slo_experiment(self.run)['passed'])


if __name__ == '__main__':
    unittest.main()
