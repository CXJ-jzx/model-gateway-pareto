"""SLO-only experiment artifacts, reporting and independent reproduction."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from .dataset import file_hash, read_jsonl, write_json, write_jsonl
from .report import fmt, table
from .slo import run_experiment, validate_config


OUTPUTS = {
    'splits': 'dataset/train_test_splits.jsonl',
    'test_inputs': 'dataset/test_requests.jsonl',
    'assignments': 'test_assignments.jsonl',
    'results': 'test_results.jsonl',
    'rules': 'slo_rules.json',
    'coverage': 'coverage.json',
}
REQUIRED = ('config.json', 'dataset/all_requests.jsonl', *OUTPUTS.values())


def _read(path):
    return [r for _, r in read_jsonl(path)] if str(path).endswith('.jsonl') else json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _rate(value):
    return '—' if value is None else f'{value:.2%}'


def report_experiment(data, config):
    coverage, rules = data['coverage'], data['rules']
    split_rows = []
    for mode, m in coverage['split_metadata'].items():
        split_rows.append([mode, m['total'], m['counts'].get('train', 0), m['counts'].get('test', 0),
                           m['train_outcomes_visible'], m['cutoff']])
    rule_rows = []
    for rule in rules['rules']:
        scale = 1000 if rule['metric'] == 'e2e' else 1
        stats = rule['statistics']
        rule_rows.append([rule['metric'], rule['range'], rule['n'], rule['status'],
                          *[fmt(rule['thresholds'][g]/scale) if rule['thresholds'][g] is not None else '—' for g in config['quantiles']],
                          fmt(stats['variance']/scale**2) if stats['variance'] is not None else '—'])
    summary_rows = [[r['mode'], r['metric'], r['grade'], r['total'], r['assigned'], r['successful_metric_valid'], r['evaluated'], r['covered'],
                     _rate(r['coverage']), '—' if r['coverage_wilson_95'] is None else ' – '.join(_rate(x) for x in r['coverage_wilson_95'])]
                    for r in coverage['summary']]
    group_rows = [[r['mode'], r['metric'], r['grade'], r['range'] or '长度缺失', r['total'], r['assigned'], r['evaluated'], _rate(r['coverage'])]
                  for r in coverage['by_predicted_bucket']]
    failures = [[mode, counts['total'], counts['success'], counts['failed']] for mode, counts in coverage['test_modes'].items()]
    mock_rows = [[r['mode'], r['field'], r['n'], fmt(r['error']['min']), fmt(r['error']['max']), fmt(r['absolute_error']['mean']),
                 r['bucket_changes'], _rate(r['bucket_change_rate']), r['clipped']] for r in coverage['mock_prediction']]
    test_stat_rows = []
    for r in coverage['test_metric_statistics']:
        s, scale = r['statistics'], 1000 if r['metric'] == 'e2e' else 1
        test_stat_rows.append([r['metric'], s['n'], *[fmt(s[k]/(scale**2 if k == 'variance' else scale)) if s[k] is not None else '—'
                                                    for k in ('mean', 'variance', 'std', 'p50', 'p75', 'p95')]])
    index = {(r['mode'], r['metric'], r['grade']): r for r in coverage['summary']}
    e2e95, ttft95, tpot95, joint95 = [index[(mode, metric, 'p95')] for mode, metric in
                                     (('nonstream', 'e2e'), ('stream', 'ttft'), ('stream', 'tpot'), ('stream', 'joint'))]
    ratio = f"{config['train_fraction']:.0%}/{1-config['train_fraction']:.0%}"
    return '\n\n'.join([
        '# BP SLO 训练/测试覆盖率实验\n\n'
        f"采用 {ratio} 时间切分，只有训练与测试两个区间。档位为 P50/P75/P95，无校准区间。"
        '本次实现截止于 SLO 分配与观测覆盖率验证，未接入 Pareto、Endpoint 模拟或在线路由。',
        '## 1. 切分与训练标签可见性\n\n' + table(
            ['模式', '全部请求', '训练请求', '测试请求', '截止前已完成的训练请求', '测试开始时间（UTC）'], split_rows) +
        f"\n\n模式/时间无法识别的请求 {coverage['unassigned_requests']} 条，保留并标记 unassigned。"
        '两种模式分别按到达时间排序，以 floor(n×训练比例) 处的到达时刻为边界。相同边界时间全部进入测试，比例可能略有偏差。'
        '训练中在截止时刻或之后完成的请求保留，但其标签不参与阈值拟合。训练、测试都合并 C/D，不按 Endpoint 定 SLO。',
        '## 2. 仅由训练数据生成的阈值\n\n'
        '非流式按实际输出长度分桶计算 E2E，流式按实际输入长度分桶计算 TTFT，TPOT 使用全部有效流式训练样本。'
        '仅成功、指标有效、截止前已完成且没有本地估算 tokens 标记的训练请求参与。\n\n' +
        table(['指标', '长度区间', '参考 n', '证据状态', 'P50', 'P75', 'P95', '样本方差'], rule_rows) +
        '\n\n上表 E2E 以秒显示、方差为秒²；TTFT 为毫秒、方差毫秒²；TPOT 为毫秒/token、方差为其单位平方。JSON统一保存ms和ms/token。'
        f"每组至少 {config['minimum_group_samples']} 条才启用阈值，此门槛是可调初筛条件，不代表统计可靠性保证。"
        '不足组保存探索分位数，但分配阈值为 null，返回 insufficient_evidence；不自动回退到其他桶。'
        '参考请求 ID 保存在规则中，可以检查阈值来源。方差只描述分布，不参与阈值修改。',
        '## 3. 测试请求的模拟预测\n\n'
        f"输入、输出独立采样整数扰动 ε∈[-{config['mock_prediction']['max_absolute_noise']}, {config['mock_prediction']['max_absolute_noise']}]，"
        f"预测长度=max(0, 实际长度+ε)，随机种子={config['mock_prediction']['seed']}。缺失长度保持 null。"
        '使用请求ID和字段派生各自随机种子，调整处理顺序不会改变预测，同一请求的三个档位使用同一组预测。\n\n' +
        table(['模式', '长度字段', '有效 n', '最小误差', '最大误差', '平均绝对误差', '跨桶请求', '跨桶率', '截为0'], mock_rows) +
        '\n\n这是基于实际长度构造的模拟预测，结果只能说明这个±50误差假设下的行为，不是已训练预测器的真实误差。'
        '`dataset/test_requests.jsonl` 只提供身份、模式、到达信息、模拟预测及其来源；SLO分配函数仅消费这些字段。'
        '完成后的真实耗时、实际端点、成功与否留在事实和评价阶段。',
        '## 4. 测试覆盖率\n\n' + table(
            ['模式', '指标', '档位', '测试请求', '有阈值', '成功且指标有效', '可评估 n', '达标 n', '覆盖率', 'Wilson 95%区间'], summary_rows) +
        '\n\n覆盖率=达标请求数 / 可评估请求数。可评估要求请求成功、指标有效且阈值已分配。'
        '失败、指标缺失、规则不足均有独立计数，不能当成已达标，也不会隐去后只报告覆盖率。'
        '没有可评估样本时覆盖率为 null，不写成0或100%。Wilson区间是二项近似参考，未证明时序样本独立。'
        'P50/P75/P95是训练分布的分位点，不要求测试覆盖率机械等于50%/75%/95%；不根据测试结果重新调整阈值。'
        '流式joint要求同一请求的TTFT与TPOT同时达标；两个单项P95不意味着联合覆盖95%。\n\n' +
        table(['模式', '测试请求', '成功', '失败'], failures) +
        '\n\n测试段成功且指标有效请求的分布如下（不要求已分配阈值；单位与上面的阈值表一致）：\n\n' +
        table(['指标', '有效 n', '均值', '样本方差', '标准差', 'P50', 'P75', 'P95'], test_stat_rows) +
        '\n\n![测试覆盖率](figures/coverage.png)',
        '## 5. 按预测长度区间查看覆盖\n\n' + table(
            ['模式', '指标', '档位', '预测长度区间', '请求数', '有阈值', '可评估 n', '覆盖率'], group_rows) +
        '\n\n分桶根据模拟预测，阈值来自实际训练长度；跨桶误差会同时影响阈值与可评估样本数。'
        '`coverage.json` 还记录所有排除原因、超出阈值幅度、误差分布；`test_results.jsonl` 可以逐条核对。',
        '## 6. 结果解读\n\n'
        f"P95档非流式E2E总体覆盖率为 {_rate(e2e95['coverage'])}，流式TTFT为 {_rate(ttft95['coverage'])}，"
        f"TPOT为 {_rate(tpot95['coverage'])}，联合覆盖为 {_rate(joint95['coverage'])}。"
        '应同时读取各预测长度区间的结果，总体接近目标分位点可能掩盖不同桶方向相反的偏差。'
        'TPOT使用全模型统一阈值，输入/输出预测扰动不直接改变其阈值；其训练与测试覆盖差异需要结合时间分布、样本构成分析，'
        '不能归因为长度预测误差，也不能通过修改测试阈值消除。'
        '这些结果是固定规则的观测表现，不根据本轮测试结果再调整阈值。',
        '## 7. 实验验证与边界\n\n'
        '`manifest.json`记录输入事实、代码、产物的SHA-256；`validation.json`记录切分、训练来源、预测、阈值、逐请求结果和覆盖率的重算结果。'
        '验证通过表示实验实现与数据一致，不表示测试覆盖率已经达到业务目标。'
        '本轮是现有一天历史数据的回顾性实验，包含历史路由比例与时间分布变化；不评估更换Endpoint的反事实收益。'
        'TTFT仍为最终成功attempt的首响应代理，TPOT为平均间隔代理。原始事实不改写，旧画像产物保留。',
    ])+'\n'


def plot_coverage(coverage, output):
    from .plots import plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, (mode, metric, title) in zip(axes.flat, (
        ('nonstream', 'e2e', 'Non-stream E2E'), ('stream', 'ttft', 'Stream TTFT proxy'),
        ('stream', 'tpot', 'Stream average TPOT proxy'), ('stream', 'joint', 'Stream TTFT + TPOT jointly'),
    )):
        items = [r for r in coverage['summary'] if r['mode'] == mode and r['metric'] == metric]
        values = [r['coverage'] if r['coverage'] is not None else 0 for r in items]
        ax.bar(range(3), values, color=['#227c9d', '#59a14f', '#e4a331'], width=.6)
        for i, r in enumerate(items):
            annotation_y = values[i]+.09
            reference = {'p50': .5, 'p75': .75, 'p95': .95}[r['grade']]
            if metric != 'joint' and annotation_y <= reference <= annotation_y+.18:
                annotation_y = reference+.04
            ax.text(i, annotation_y, f"{values[i]:.1%}\nn={r['evaluated']}" if r['coverage'] is not None else 'N/A\nn=0', ha='center', va='bottom', fontsize=10)
            if metric != 'joint':
                ax.plot([i-.32, i+.32], [{'p50': .5, 'p75': .75, 'p95': .95}[r['grade']]]*2, color='#444444', linestyle='--', linewidth=1.2)
        ax.set(ylim=(0, 1.22), title=title, ylabel='Observed coverage', xticks=range(3), xticklabels=['P50', 'P75', 'P95'])
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('BP 70/30 time split | synthetic token forecasts +/-50\nCoverage on successful evaluable requests; dashed = reference quantile', fontsize=13)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def prepare_slo_experiment(source_path, run_dir, config_path):
    source_path, root = Path(source_path).resolve(), Path(run_dir).resolve()
    config = validate_config(_read(config_path))
    rows = _read(source_path)
    if not rows:
        raise ValueError('No input BP facts')
    # Verify the original extraction package if this is its canonical facts file.
    parent = source_path.parent.parent
    if source_path.name == 'requests.jsonl' and (parent/'manifest.json').is_file():
        from .validation import validate_run
        source_validation = validate_run(parent)
        if not source_validation['passed']:
            raise ValueError('Input extraction package failed validation: '+str([c for c in source_validation['checks'] if not c['passed']]))
    data = run_experiment(rows, config)
    root.mkdir(parents=True, exist_ok=False)
    (root/'dataset').mkdir()
    write_json(root/'config.json', config)
    write_jsonl(root/'dataset/all_requests.jsonl', sorted(rows, key=lambda r: (r.get('arrived_at') or '9999', r['request_id'])))
    for key, relative in OUTPUTS.items():
        (write_jsonl if relative.endswith('.jsonl') else write_json)(root/relative, data[key])
    (root/'REPORT.md').write_text(report_experiment(data, config), encoding='utf-8')
    plot_coverage(data['coverage'], root/'figures/coverage.png')
    project = Path(__file__).resolve().parents[1]
    code = [*sorted((project/'bp_slo').glob('*.py')), *sorted((project/'tests').glob('*.py')), project/'requirements.txt', project/'pyproject.toml']
    write_json(root/'manifest.json', dict(schema_version='1.0', stage='slo_coverage_only',
        created_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        source=dict(path=str(source_path), bytes=source_path.stat().st_size, sha256=file_hash(source_path)),
        artifacts={p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=file_hash(p)) for p in sorted(root.rglob('*')) if p.is_file()},
        code_sha256={p.relative_to(project).as_posix(): file_hash(p) for p in code}))
    validation = validate_slo_experiment(root)
    write_json(root/'validation.json', validation)
    return root, data['coverage'], validation


def validate_slo_experiment(run_dir):
    """Read-only checks; source absence warns, available changed sources fail."""
    root, checks, warnings, files = Path(run_dir).resolve(), [], [], {}
    def check(name, action):
        try:
            detail = action()
            checks.append(dict(name=name, passed=True, detail=detail or 'OK'))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, detail=f'{type(exc).__name__}: {exc}'))
    for relative in ('manifest.json', *REQUIRED):
        def load(relative=relative):
            files[relative] = _read(root/relative)
            json.dumps(files[relative], allow_nan=False)
        check('read:'+relative, load)
    def integrity():
        manifest = files['manifest.json']
        if manifest['stage'] != 'slo_coverage_only' or not set(REQUIRED) <= manifest['artifacts'].keys():
            raise ValueError('Wrong experiment stage or incomplete manifest coverage')
        for relative, entry in manifest['artifacts'].items():
            path = (root/relative).resolve()
            if Path(relative).is_absolute() or root not in path.parents or relative in ('manifest.json', 'validation.json'):
                raise ValueError('Artifact outside experiment directory or includes itself')
            if file_hash(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
                raise ValueError('Artifact digest mismatch: '+relative)
        return f"Verified {len(manifest['artifacts'])} artifact hashes"
    check('artifact_integrity', integrity)
    def source_check():
        source = files['manifest.json']['source']
        path = Path(source['path'])
        if not path.is_file():
            warnings.append('Source unavailable; verified local copied facts only: '+str(path))
            return warnings[-1]
        if file_hash(path) != source['sha256'] or path.stat().st_size != source['bytes']:
            raise ValueError('Source facts digest changed')
        original = sorted(_read(path), key=lambda r: (r.get('arrived_at') or '9999', r['request_id']))
        if original != files['dataset/all_requests.jsonl']:
            raise ValueError('Copied facts differ from source')
    check('source_facts', source_check)
    def recompute():
        config = validate_config(files['config.json'])
        rows = files['dataset/all_requests.jsonl']
        expected = run_experiment(rows, config)
        for key, relative in OUTPUTS.items():
            if expected[key] != files[relative]:
                raise ValueError('Recomputed experiment differs: '+relative)
        split_index = {s['request_id']: s for s in expected['splits']}
        for rule in expected['rules']['rules']:
            for rid in rule['training_request_ids']:
                if split_index[rid]['split'] != 'train' or not split_index[rid]['training_outcome_visible']:
                    raise ValueError('Rule includes a non-visible training outcome')
        for request in expected['test_inputs']:
            if split_index[request['request_id']]['split'] != 'test':
                raise ValueError('Mock prediction outside test interval')
            for field in ('input', 'output'):
                noise = request['prediction_noise'][field]
                if noise['applied_error'] is not None and abs(noise['applied_error']) > config['mock_prediction']['max_absolute_noise']:
                    raise ValueError('Prediction error exceeds configured amplitude')
        return 'Recomputed split, train-only rules, deterministic mock inputs, SLO assignments and coverage'
    check('slo_experiment_reproduction', recompute)
    return dict(schema_version='1.0', stage='slo_coverage_only', passed=all(c['passed'] for c in checks), checks=checks, warnings=warnings)
