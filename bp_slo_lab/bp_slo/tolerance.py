"""Explicit acceptance tolerance on a frozen, reproducible SLO experiment.

No threshold fitting changes, policy search, confidence-bound inflation or
test-to-train feedback. Strict and tolerant coverage use the same denominator.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .dataset import file_hash, number, write_json, write_jsonl
from .report import table
from .slo import METRICS, evaluate_test, run_experiment
from .slo_experiment import _read, _rate, validate_slo_experiment


SNAPSHOTS = ('config.json', 'dataset/all_requests.jsonl', 'dataset/test_requests.jsonl',
             'test_assignments.jsonl', 'test_results.jsonl', 'coverage.json')
OUTPUTS = {'comparison': 'comparison.json', 'results': 'tolerance_results.jsonl'}
FORMULA = 'limit + max(limit * relative_tolerance, absolute_tolerance)'


def validate_config(config):
    if (config.get('schema_version') != '1.0' or config.get('model_id') != '模型BP'
            or config.get('stage') != 'slo_tolerance_only'
            or config.get('purpose') != 'experiment_example_not_business_commitment'
            or config.get('formula') != FORMULA):
        raise ValueError('Use the explicit BP experimental tolerance contract')
    for name in ('relative_tolerance', 'absolute_tolerance'):
        values = config.get(name)
        if not isinstance(values, dict) or set(values) != set(METRICS):
            raise ValueError(f'{name} needs e2e, ttft and tpot values')
        if any(number(value) is None for value in values.values()):
            raise ValueError(f'{name} values must be finite nonnegative numbers, not booleans')
    if config.get('absolute_units') != {metric: spec['unit'] for metric, spec in METRICS.items()}:
        raise ValueError('Absolute tolerance uses ms for E2E/TTFT and ms/token for TPOT')
    return config


def acceptance_limit(limit, metric, config):
    """Use the larger relative/absolute margin, not their sum."""
    if number(limit) is None:
        raise ValueError('Assigned SLO limit must be a finite nonnegative number')
    result = limit + max(limit * config['relative_tolerance'][metric],
                         config['absolute_tolerance'][metric])
    if number(result) is None:
        raise ValueError('Acceptance limit overflow')
    return result


def _compare_groups(strict, tolerant):
    key = lambda r: (r['mode'], r['metric'], r['grade'], r['range'])
    index = {key(row): row for row in tolerant}
    if set(map(key, strict)) != set(index):
        raise ValueError('Strict and tolerant evaluation groups differ')
    output = []
    for original in strict:
        relaxed = index[key(original)]
        for field in ('total', 'assigned', 'successful_metric_valid', 'evaluated'):
            if original[field] != relaxed[field]:
                raise ValueError('Tolerance must not change coverage denominators')
        if relaxed['covered'] < original['covered']:
            raise ValueError('Nonnegative tolerance cannot reduce coverage')
        output.append(dict(mode=original['mode'], metric=original['metric'], grade=original['grade'],
                           range=original['range'], total=original['total'], assigned=original['assigned'],
                           evaluated=original['evaluated'], strict_covered=original['covered'],
                           tolerant_covered=relaxed['covered'],
                           rescued_within_tolerance=relaxed['covered'] - original['covered'],
                           beyond_tolerance=relaxed['exceeded'],
                           strict_coverage=original['coverage'], tolerant_coverage=relaxed['coverage'],
                           strict_status_counts=original['evaluation_status_counts'],
                           tolerant_status_counts=relaxed['evaluation_status_counts']))
    return output


def evaluate_tolerance(rows, baseline_config, config):
    """Reproduce frozen baseline, then evaluate an explicitly different SLI."""
    validate_config(config)
    baseline = run_experiment(rows, baseline_config)
    adjusted = deepcopy(baseline['assignments'])
    for assignment in adjusted:
        for metrics in assignment['grades'].values():
            for metric, rule in metrics.items():
                if rule['status'] == 'assigned':
                    rule['limit'] = acceptance_limit(rule['limit'], metric, config)
    test_ids = {r['request_id'] for r in baseline['test_inputs']}
    test_rows = sorted((r for r in rows if r['request_id'] in test_ids),
                       key=lambda r: (r.get('arrived_at') or '9999', r['request_id']))
    relaxed, relaxed_results = evaluate_test(test_rows, baseline['test_inputs'], adjusted, baseline_config)
    strict = baseline['coverage']
    comparison = dict(schema_version='1.0', stage='slo_tolerance_only',
                      purpose=config['purpose'], tolerance_contract=config,
                      test_modes=strict['test_modes'], coverage_denominator=strict['coverage_denominator'],
                      strict_definition='observed <= assigned SLO limit',
                      tolerant_definition='observed <= acceptance limit; assigned SLO limit unchanged',
                      joint_definition='TTFT and TPOT must both satisfy their respective acceptance limits',
                      summary=_compare_groups(strict['summary'], relaxed['summary']),
                      by_predicted_bucket=_compare_groups(strict['by_predicted_bucket'], relaxed['by_predicted_bucket']))
    results = []
    for original, accepted in zip(baseline['results'], relaxed_results, strict=True):
        if (original['request_id'], original['grade']) != (accepted['request_id'], accepted['grade']):
            raise ValueError('Result order mismatch')
        results.append(dict(request_id=original['request_id'], stream_type=original['stream_type'],
                            grade=original['grade'], success=original['success'],
                            strict_metrics=original['metrics'], tolerant_metrics=accepted['metrics']))
    return dict(comparison=comparison, results=results), baseline


def _compute(snapshot, config):
    rows, baseline_config = _read(snapshot/'dataset/all_requests.jsonl'), _read(snapshot/'config.json')
    data, baseline = evaluate_tolerance(rows, baseline_config, config)
    for key, relative in (('test_inputs', 'dataset/test_requests.jsonl'), ('assignments', 'test_assignments.jsonl'),
                          ('results', 'test_results.jsonl'), ('coverage', 'coverage.json')):
        if baseline[key] != _read(snapshot/relative):
            raise ValueError('Frozen baseline did not reproduce: '+relative)
    return data


def report_tolerance(data, config):
    comparison = data['comparison']
    fields = ('mode', 'metric', 'grade', 'range', 'evaluated', 'strict_coverage',
              'tolerant_coverage', 'rescued_within_tolerance', 'beyond_tolerance')
    def values(items):
        return [[_rate(row[f]) if f.endswith('_coverage') else row[f] for f in fields] for row in items]
    contract = [[metric, _rate(config['relative_tolerance'][metric]), config['absolute_tolerance'][metric], spec['unit']]
                for metric, spec in METRICS.items()]
    columns = ['模式', '指标', '档位', '区间', '可评估n', '严格覆盖率', '容差覆盖率', '容差新增达标', '超过容差']
    return '\n\n'.join([
        '# BP SLO 显式容差评价（实验示例）',
        '已回退到固定分桶 P50/P75/P95、70%训练/30%测试、±50 token 模拟预测的原版本。'
        '不再搜索候选，不采用保守分位数上界，不改训练表，不将测试反馈用于更新历史。'
        '本报告只是明确比较两种验收口径，不能声称原严格SLO已经达标或已经获得生产承诺。',
        '## 1. 容差定义\n\n'
        '`验收上限 = 原始阈值 + max(原始阈值 × 相对容差, 绝对容差)`\n\n'
        + table(['指标', '相对容差', '绝对容差', '绝对容差单位'], contract) +
        '\n\n容差采用上表配置。例如相对容差为10%时：10秒E2E允许至11秒，2秒TTFT允许至2.2秒，40毫秒/token TPOT允许至44毫秒/token。'
        '等于验收上限也算达标。原始SLO阈值与验收上限分别保存，不混用。'
        '如同时配置绝对容差，取两种余量中较大者，不叠加；绝对容差目前为0。',
        '## 2. 覆盖率对照\n\n'+table(columns, values(comparison['summary']))+
        '\n\n严格覆盖率按实际指标≤原阈值；容差覆盖率按实际指标≤验收上限。'
        '两者分母完全一致：成功、指标有效、已有阈值的测试请求。失败、缺失、规则不足不变为达标。'
        '流式联合仍要求TTFT与TPOT同时满足对应口径，不把两个单项P95自动当成联合95%。',
        '## 3. 按预测长度分组\n\n'+table(columns, values(comparison['by_predicted_bucket'])),
        '## 4. 完整性与边界\n\n'+table(['模式', '测试总数', '成功', '失败'],
            [[mode, *[counts[k] for k in ('total', 'success', 'failed')]] for mode, counts in comparison['test_modes'].items()])+
        '\n\n`baseline/`完整保留本次引用的冻结输入和严格结果，验证时重新训练并复核其一致性。'
        '`tolerance_results.jsonl`逐请求同时保存原阈值、验收上限、观测指标与两种状态。'
        '原来的BP数据和实验目录未改写。所有排除原因在`comparison.json`保留。'
        '容差由配置明确给定；默认采用用户选择的10%实验示例，不遍历不同容差以追逐测试覆盖率。'
        '大幅超时无法由小容差消除；P50/P75/P95仍是历史分位数档位，不等于未来的保证达标率。'
        '真实业务若采用容差，应提前公开验收上限、度量口径、统计窗口及达标比例，不能事后放宽。',
    ])+'\n'


def prepare_tolerance(baseline_dir, run_dir, config_path):
    baseline_dir, root = Path(baseline_dir).resolve(), Path(run_dir).resolve()
    if root.exists():
        raise FileExistsError('Refusing to overwrite: '+str(root))
    config = validate_config(_read(config_path))
    validation = validate_slo_experiment(baseline_dir)
    if not validation['passed']:
        raise ValueError('Baseline SLO experiment failed validation')
    data = _compute(baseline_dir, config)
    root.mkdir(parents=True, exist_ok=False)
    for relative in SNAPSHOTS:
        destination = root/'baseline'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline_dir/relative, destination)
    write_json(root/'config.json', config)
    for key, relative in OUTPUTS.items():
        (write_jsonl if relative.endswith('.jsonl') else write_json)(root/relative, data[key])
    (root/'REPORT.md').write_text(report_tolerance(data, config), encoding='utf-8')
    project = Path(__file__).resolve().parents[1]
    write_json(root/'manifest.json', dict(schema_version='1.0', stage='slo_tolerance_only',
        created_at=datetime.now(timezone.utc).isoformat(),
        sources={relative: dict(path=str(baseline_dir/relative), sha256=file_hash(baseline_dir/relative)) for relative in SNAPSHOTS},
        artifacts={p.relative_to(root).as_posix(): dict(sha256=file_hash(p), bytes=p.stat().st_size)
                   for p in sorted(root.rglob('*')) if p.is_file()},
        code_sha256={p.relative_to(project).as_posix(): file_hash(p) for p in sorted((project/'bp_slo').glob('*.py'))}))
    validation = validate_tolerance(root)
    write_json(root/'validation.json', validation)
    return root, data['comparison'], validation


def validate_tolerance(run_dir):
    """Read-only integrity and full strict/tolerant reproduction checks."""
    root, checks, warnings = Path(run_dir).resolve(), [], []
    def check(name, action):
        try:
            action()
            checks.append(dict(name=name, passed=True))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, detail=f'{type(exc).__name__}: {exc}'))
    def integrity():
        manifest = _read(root/'manifest.json')
        required = {'config.json', 'REPORT.md', *OUTPUTS.values(), *('baseline/'+r for r in SNAPSHOTS)}
        if manifest['stage'] != 'slo_tolerance_only' or not required <= manifest['artifacts'].keys():
            raise ValueError('Wrong stage or incomplete manifest')
        for relative, entry in manifest['artifacts'].items():
            path = (root/relative).resolve()
            if Path(relative).is_absolute() or root not in path.parents or relative in ('manifest.json', 'validation.json'):
                raise ValueError('Unsafe artifact path')
            if file_hash(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
                raise ValueError('Artifact digest mismatch: '+relative)
    def sources():
        manifest = _read(root/'manifest.json')
        if set(manifest['sources']) != set(SNAPSHOTS):
            raise ValueError('Incomplete baseline source references')
        for relative, source in manifest['sources'].items():
            if file_hash(root/'baseline'/relative) != source['sha256']:
                raise ValueError('Snapshot differs from registered source: '+relative)
            path = Path(source['path'])
            if not path.is_file():
                warnings.append('Original baseline unavailable; using copied snapshot: '+str(path))
            elif file_hash(path) != source['sha256']:
                raise ValueError('Original baseline changed: '+relative)
    def reproduce():
        data = _compute(root/'baseline', validate_config(_read(root/'config.json')))
        for key, relative in OUTPUTS.items():
            if data[key] != _read(root/relative):
                raise ValueError('Recomputed tolerance evaluation differs: '+relative)
    check('artifact_integrity', integrity)
    check('baseline_source_integrity', sources)
    check('strict_and_tolerant_reproduction', reproduce)
    return dict(schema_version='1.0', stage='slo_tolerance_only', passed=all(c['passed'] for c in checks), checks=checks, warnings=warnings)
