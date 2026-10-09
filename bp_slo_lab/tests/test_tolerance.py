"""Verify rollback baseline semantics and auditable, bounded acceptance slack."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bp_slo.dataset import file_hash, write_json, write_jsonl
from bp_slo.slo_experiment import prepare_slo_experiment
from bp_slo.tolerance import acceptance_limit, evaluate_tolerance, prepare_tolerance, validate_config, validate_tolerance


PROJECT = Path(__file__).resolve().parents[1]


def configs():
    base = json.loads((PROJECT/'configs/slo_experiment.json').read_text(encoding='utf-8'))
    base['minimum_group_samples'] = 2
    tolerance = json.loads((PROJECT/'configs/slo_tolerance.json').read_text(encoding='utf-8'))
    return base, tolerance


def facts():
    rows = []
    for mode in ('nonstream', 'stream'):
        for i in range(10):
            start = datetime(2026, 9, 21, tzinfo=timezone.utc)+timedelta(minutes=i)
            rows.append(dict(request_id=f'{mode}_{i}', model_id='模型BP', stream_type=mode,
                arrived_at=start.isoformat(), finished_at=(start+timedelta(seconds=1)).isoformat(),
                success=True, actual_input_tokens=200, actual_output_tokens=100, quality_flags=[],
                request_e2e_ms=100, ttft_proxy_ms=100 if mode == 'stream' else None,
                tpot_proxy_ms=10 if mode == 'stream' else None,
                eligible_length_e2e=True, eligible_ttft_input=mode == 'stream', eligible_tpot_output=mode == 'stream'))
    index = {r['request_id']: r for r in rows}
    for i, value in enumerate((110, 110.001, 109), 7):
        index[f'nonstream_{i}']['request_e2e_ms'] = value
    index['stream_7'].update(ttft_proxy_ms=110, tpot_proxy_ms=11)
    index['stream_8'].update(ttft_proxy_ms=110.001)
    index['stream_9'].update(tpot_proxy_ms=11.001)
    return rows


class ToleranceTests(unittest.TestCase):
    def test_relative_margin_and_absolute_floor_do_not_add(self):
        _, config = configs()
        self.assertEqual(acceptance_limit(100, 'ttft', config), 110)
        config['absolute_tolerance']['ttft'] = 20
        self.assertEqual(acceptance_limit(100, 'ttft', config), 120)
        self.assertEqual(acceptance_limit(0, 'ttft', config), 20)

    def test_invalid_tolerances_and_units_are_rejected(self):
        _, config = configs()
        for bad in (True, -1, float('nan'), float('inf'), None):
            changed = deepcopy(config)
            changed['relative_tolerance']['e2e'] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                validate_config(changed)
        changed = deepcopy(config)
        changed['absolute_units']['tpot'] = 'ms'
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_equal_to_acceptance_limit_passes_but_larger_does_not(self):
        base, config = configs()
        data, baseline = evaluate_tolerance(facts(), base, config)
        row = next(r for r in data['comparison']['summary'] if r['metric'] == 'e2e' and r['grade'] == 'p50')
        self.assertEqual(row['strict_covered'], 0)
        self.assertEqual(row['tolerant_covered'], 2)
        self.assertEqual(row['rescued_within_tolerance'], 2)
        self.assertEqual(row['beyond_tolerance'], 1)
        self.assertEqual(row['evaluated'], 3)
        self.assertEqual(baseline['assignments'][0]['grades']['p50']['e2e']['limit'], 100)

    def test_stream_joint_still_requires_both_metrics(self):
        base, config = configs()
        data, _ = evaluate_tolerance(facts(), base, config)
        row = next(r for r in data['comparison']['summary'] if r['metric'] == 'joint' and r['grade'] == 'p50')
        self.assertEqual(row['tolerant_covered'], 1)
        self.assertEqual(row['evaluated'], 3)

    def test_failed_and_missing_requests_cannot_be_rescued(self):
        base, config = configs()
        rows = facts()
        index = {r['request_id']: r for r in rows}
        index['nonstream_7']['success'] = False
        index['nonstream_8']['request_e2e_ms'] = None
        data, _ = evaluate_tolerance(rows, base, config)
        row = next(r for r in data['comparison']['summary'] if r['metric'] == 'e2e' and r['grade'] == 'p50')
        self.assertEqual(row['evaluated'], 1)
        self.assertEqual(row['tolerant_status_counts']['request_failed'], 1)
        self.assertEqual(row['tolerant_status_counts']['missing_metric'], 1)
        self.assertEqual(row['rescued_within_tolerance'], 1)

    def test_zero_tolerance_is_identical_to_strict_coverage(self):
        base, config = configs()
        config['relative_tolerance'] = dict.fromkeys(config['relative_tolerance'], 0)
        data, _ = evaluate_tolerance(facts(), base, config)
        for row in data['comparison']['summary']:
            self.assertEqual(row['strict_coverage'], row['tolerant_coverage'])
            self.assertEqual(row['rescued_within_tolerance'], 0)

    def test_sparse_rules_remain_unassigned(self):
        base, config = configs()
        base['minimum_group_samples'] = 100
        data, _ = evaluate_tolerance(facts(), base, config)
        self.assertTrue(all(row['evaluated'] == 0 and row['tolerant_coverage'] is None for row in data['comparison']['summary']))


class ToleranceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        source = cls.root/'source.jsonl'
        base, tolerance = configs()
        write_jsonl(source, facts())
        write_json(cls.root/'baseline_config.json', base)
        write_json(cls.root/'tolerance_config.json', tolerance)
        cls.baseline, _, check = prepare_slo_experiment(source, cls.root/'baseline', cls.root/'baseline_config.json')
        if not check['passed']:
            raise AssertionError(check)
        cls.original, _, check = prepare_tolerance(cls.baseline, cls.root/'original', cls.root/'tolerance_config.json')
        if not check['passed']:
            raise AssertionError(check)

    def setUp(self):
        self.tmp_case = tempfile.TemporaryDirectory(dir=self.root)
        self.addCleanup(self.tmp_case.cleanup)
        self.case_dir = Path(self.tmp_case.name)/'run'
        shutil.copytree(self.original, self.case_dir)

    def test_package_reproduces_and_baseline_remains_unchanged(self):
        self.assertTrue(validate_tolerance(self.case_dir)['passed'])
        self.assertEqual(file_hash(self.baseline/'test_assignments.jsonl'),
                         file_hash(self.case_dir/'baseline/test_assignments.jsonl'))

    def test_rehashed_coverage_tampering_is_still_detected(self):
        path = self.case_dir/'comparison.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        data['summary'][0]['tolerant_coverage'] = .123
        write_json(path, data)
        manifest = json.loads((self.case_dir/'manifest.json').read_text(encoding='utf-8'))
        manifest['artifacts']['comparison.json'] = dict(sha256=file_hash(path), bytes=path.stat().st_size)
        write_json(self.case_dir/'manifest.json', manifest)
        result = validate_tolerance(self.case_dir)
        self.assertFalse(result['passed'])
        self.assertTrue(any(c['name'] == 'strict_and_tolerant_reproduction' and not c['passed'] for c in result['checks']))

    def test_existing_result_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            prepare_tolerance(self.baseline, self.original, self.root/'tolerance_config.json')

    def test_missing_artifact_fails_without_crashing(self):
        (self.case_dir/'tolerance_results.jsonl').unlink()
        self.assertFalse(validate_tolerance(self.case_dir)['passed'])


if __name__ == '__main__':
    unittest.main()
