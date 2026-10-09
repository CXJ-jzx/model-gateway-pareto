"""Independent neighbor-SLO experiment; legacy runs remain reproducible."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .dataset import file_hash, write_json, write_jsonl
from .neighborhood_slo import NeighborhoodSLO, run_experiment, validate_config
from .report import table
from .slo import evaluate_test
from .slo_experiment import OUTPUTS, _read, _rate
from .tolerance import acceptance_limit, validate_config as validate_tolerance_config


EXTRA = {'tolerant_coverage': 'tolerant_coverage.json', 'tolerant_results': 'tolerant_results.jsonl',
         'boundary_examples': 'boundary_examples.json', 'methods': 'assignment_methods.json'}


def compute(rows, config, tolerance):
    data = run_experiment(rows, config)
    validate_tolerance_config(tolerance)
    adjusted = deepcopy(data['assignments'])
    for assignment in adjusted:
        for metrics in assignment['grades'].values():
            for metric, rule in metrics.items():
                if rule['status'] == 'assigned':
                    rule['limit'] = acceptance_limit(rule['limit'], metric, tolerance)
    test_ids = {r['request_id'] for r in data['test_inputs']}
    tests = sorted((r for r in rows if r['request_id'] in test_ids), key=lambda r: (r.get('arrived_at') or '9999', r['request_id']))
    tolerant, results = evaluate_test(tests, data['test_inputs'], adjusted, config)
    tolerant['stage'] = 'neighborhood_slo'
    data.update(tolerant_coverage=tolerant, tolerant_results=results)
    counters = {}
    for metric in ('e2e', 'ttft', 'tpot'):
        counters[metric] = dict(Counter(a['grades']['strict'][metric]['evidence']['method'] for a in data['assignments']
                                      if metric in a['grades']['strict']))
    data['methods'] = counters
    lookup = NeighborhoodSLO(data['rules'])
    maximum = max(lookup.lengths['e2e'], default=0)
    base = dict(model_id=config['model_id'], arrived_at='2026-09-22T00:00:00+00:00', predicted_input_tokens=300, stream_type='nonstream')
    probes = [('old_bucket_boundary_512', 512), ('old_bucket_boundary_513', 513),
              ('above_training_max', int(maximum*1.25)+1), ('far_outside_training', int(maximum*5)+1),
              ('missing_forecast', None)]
    data['boundary_examples'] = [dict(probe=name, request={**base, 'request_id':'probe_'+name, 'predicted_output_tokens':length},
                                         assignment=lookup.assign({**base, 'request_id':'probe_'+name, 'predicted_output_tokens':length}))
                                  for name, length in probes]
    return data


def report(data, config, tolerance):
    policy = config.get('e2e_forward_neighborhood')
    width_description = (f"Δ=max({policy['minimum_width_tokens']},ceil(L×{policy['relative_width']}))，"
                         '查找[L,L+Δ]；短请求保留最小跨度，长请求使用相对跨度。'
                         if policy else f"Δ={config['forward_width_tokens']}，查找[L,L+Δ]。")
    raw = {(r['metric'], r['grade']): r for r in data['coverage']['summary']}
    rows = [[r['mode'], r['metric'], r['grade'], r['evaluated'],
             _rate(raw[(r['metric'], r['grade'])]['coverage']), _rate(r['coverage'])]
            for r in data['tolerant_coverage']['summary']]
    methods = [[metric, method, count] for metric, counts in data['methods'].items() for method, count in counts.items()]
    groups = [[r['mode'], r['metric'], r['grade'], r['range'], r['evaluated'], _rate(r['coverage'])]
              for r in data['tolerant_coverage']['by_predicted_bucket']]
    associations = [[metric, feature, stats['n'], stats['pearson'], stats['spearman']]
                    for metric, features in data['rules']['training_associations'].items() for feature, stats in features.items()]
    return '\n\n'.join([
        '# BP 邻域SLO与显式兜底实验',
        '原固定分桶、容差与路由版本已经完整备份，新结果另建目录。继续采用分模式70%训练/30%测试，无校准区间；'
        '测试输入/输出tokens仍是实际长度加固定种子±50扰动，缺失保持null。所有SLO参考仅使用测试开始前已完成的成功有效训练记录，合并Endpoint。'
        'strict/standard/relaxed是紧、中、宽三档，不是未来50%/75%/90%的保证达标率。',
        '## 1. 非流式：二分定位，而不是按延迟排序取位次\n\n'
        '预先按(actual_output_tokens,完成时间倒序,request_id)排序，构建长度数组。'
        '新请求以预测输出长度L执行bisect_left(L)、bisect_right(L+Δ)。'+width_description+
        '取区间开头至多20条，等长记录优先较新样本。'
        '全模型统一索引天然跨过原512/2048等分桶边界；桶仅用于结果分组，不限制取样。'
        '二分不需要遍历所有历史，但求延迟分位数仍须读取至多20个E2E值并按延迟计算分位数；tokens排序不代表延迟排序。'
        '满5条时原阈值为局部E2E的P50/P75/P90；1–4条时以局部P90为基准乘1/1.3/1.5，单个样本也能生成三个参考值。'
        '20条、5条及邻域宽度均为可调工程配置，并非统计可信度保证。范围扩大不等于必须取更多样本；'
        '仍最多取20条，优先预测长度右侧较近样本。单侧选择有一定保守性，20%也不是已证明最优的值。',
        '## 2. 无邻域、超长与冷启动\n\n'
        '训练范围内没有前向邻域时，用全部训练样本的非负斜率线性参考兜底。'
        'L超过训练最大输出长度时，用训练最长的20条拟合：b为非负OLS斜率，a=max(0,P90(E2E−b×tokens))，参考值a+b×L。'
        '若斜率为0则使用观察到的每token耗时P75作为启发式斜率；外推基准至少为尾部实测E2E的P90。'
        '三档仍为基准×1/1.3/1.5。拟合、来源和超域比例完整记录；超过历史最大长度2倍额外标注超出文档证据范围，但不通过截断时间使长任务变得苛刻。'
        '缺少预测长度时使用模型整体P90；整个指标都无历史时使用配置中的冷启动默认值。'
        '稀疏、缺预测、拟合与冷启动均标注fallback/evidence_level/guarantee=false，不能包装成统计承诺。',
        '## 3. 流式：较宽容的参考基准\n\n'
        f"TTFT保持输入前向固定{config['forward_width_tokens']}tokens邻域，使用模型整体P90作底线。"
        '输入前向邻域至少5条时，α=n/(n+20)，'
        'base=max(整体P90,α×邻域P90+(1−α)×整体P90)；不足或超出训练输入范围时保留整体P90，并标记证据范围。'
        '三档为base×1/1.3/1.5。不根据弱相关假定TTFT随tokens线性增长，也不让很小的邻域把阈值压得过低。'
        'TPOT按全部有效流式训练记录取P90，三档同样为P90×1/1.3/1.5。'
        'BP训练关联如下；相关不代表因果，不能由“输出关系弱”推断“输入输出都无关”：\n\n'+
        table(['指标', '长度字段', 'n', 'Pearson', 'Spearman'], associations)+
        '\n\n因此TPOT的模型级SLO可以统一，但新增路由配置仍按Endpoint和输入长度估计TPOT，移除缺乏证据的输出长度拆分；不同Endpoint不会共用性能值。',
        '## 4. 容差只应用一次\n\n'
        '原阈值T再应用此前明确的10%验收容差，验收上限=1.1T；不在SLO内部先乘1.1再让路由重复乘1.1。'
        '例如单个E2E样本10秒：原三档10/13/15秒，验收三档11/14.3/16.5秒。倍率是宽松等级，不是置信水平。'
        '新方案最宽档不能一概称为P90或P95；在样本充足的非流式邻域，它来自局部P90，其余来自参考基准的倍率。',
        '## 5. 测试覆盖率\n\n'+table(['模式', '指标', '档位', '可评估n', '原阈值覆盖率', '容差覆盖率'], rows)+
        '\n\n分母仍为成功、指标有效且分配阈值的测试请求；失败和缺失单列，联合覆盖要求TTFT与TPOT同时满足。'
        '兜底使原来不支持的长度也获得参考值，覆盖率分母可能改变，不能只比较总体百分比宣称改善。'
        '测试集已在之前研发中观察过，本次仍是回顾性验证；需要新的时间段验证稳定性。\n\n'+
        table(['模式', '测试总数', '成功', '失败'], [[m,c['total'],c['success'],c['failed']] for m,c in data['coverage']['test_modes'].items()]),
        '## 6. 兜底使用与分组\n\n'+table(['指标', '方法', '请求数（每请求只计一次）'], methods)+
        '\n\n单侧前向取样天然偏向更长的参考请求，可能使阈值偏保守，尤其短输出区间。'
        '若某组不同档位覆盖率相同，不代表期限完全相同，也不证明档位校准完美；应结合各档时限和业务区分度判断。'
        '\n\n'+table(['模式', '指标', '档位', '预测长度桶（仅展示）', '可评估n', '容差覆盖率'], groups)+
        '\n\n![覆盖率](figures/coverage.png)',
        '## 7. 可复核产物与边界\n\n'
        '`slo_rules.json`保存排序后的样本、训练关联和拟合；`test_assignments.jsonl`保存二分区间、样本ID、方法和证据等级。'
        '`boundary_examples.json`单列512/513边界、超长、远超域和缺预测演示，不混入覆盖率分母。'
        '`assignment_methods.json`统计真实测试请求的兜底次数；`validation.json`重算训练来源、模拟预测、查找、兜底及两种覆盖率。'
        '无报错兜底不等于有可信证据，宽阈值不等于性能改善，成功样本覆盖率不等于全请求可用性。'
        'TTFT仍为首响应代理、TPOT为请求平均间隔代理，没有升级为用户逐token可见指标。',
    ])+'\n'


def plot(data, path):
    from .plots import plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for ax, metric in zip(axes.flat, ('e2e','ttft','tpot','joint')):
        raw = [r for r in data['coverage']['summary'] if r['metric'] == metric]
        relaxed = [r for r in data['tolerant_coverage']['summary'] if r['metric'] == metric]
        for offset, items, label in ((-.18, raw, 'Original limit'), (.18, relaxed, '10% tolerance')):
            ax.bar([i+offset for i in range(3)], [r['coverage'] or 0 for r in items], width=.35, label=label)
        ax.set(title=metric.upper(), ylim=(0,1.05), xticks=range(3), xticklabels=['strict','standard','relaxed'])
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def prepare(input_path, run_dir, config_path, tolerance_path):
    input_path, root = Path(input_path).resolve(), Path(run_dir).resolve()
    if root.exists():
        raise FileExistsError('Refusing to overwrite '+str(root))
    config, tolerance = validate_config(_read(config_path)), validate_tolerance_config(_read(tolerance_path))
    parent = input_path.parent.parent
    if (parent/'manifest.json').is_file():
        record = _read(parent/'manifest.json')['artifacts'].get(input_path.relative_to(parent).as_posix())
        if record is None or record['sha256'] != file_hash(input_path):
            raise ValueError('Input facts differ from registered source package')
    rows = _read(input_path)
    if not rows:
        raise ValueError('No BP facts')
    data = compute(rows, config, tolerance)
    root.mkdir(parents=True, exist_ok=False)
    (root/'dataset').mkdir()
    write_json(root/'config.json', config)
    write_json(root/'tolerance_config.json', tolerance)
    write_jsonl(root/'dataset/all_requests.jsonl', sorted(rows, key=lambda r: (r.get('arrived_at') or '9999',r['request_id'])))
    for key, relative in {**OUTPUTS, **EXTRA}.items():
        (write_jsonl if relative.endswith('.jsonl') else write_json)(root/relative, data[key])
    (root/'REPORT.md').write_text(report(data, config, tolerance), encoding='utf-8')
    plot(data, root/'figures/coverage.png')
    project = Path(__file__).resolve().parents[1]
    write_json(root/'manifest.json', dict(schema_version='1.0', stage='neighborhood_slo',
        created_at=datetime.now(timezone.utc).isoformat(),
        source=dict(path=str(input_path), sha256=file_hash(input_path), bytes=input_path.stat().st_size),
        artifacts={p.relative_to(root).as_posix(): dict(sha256=file_hash(p),bytes=p.stat().st_size)
                   for p in sorted(root.rglob('*')) if p.is_file()},
        code_sha256={p.relative_to(project).as_posix():file_hash(p) for p in sorted((project/'bp_slo').glob('*.py'))}))
    result = validate(root)
    write_json(root/'validation.json', result)
    return root, data, result


def validate(run_dir):
    root, checks, warnings = Path(run_dir).resolve(), [], []
    def check(name, action):
        try:
            action()
            checks.append(dict(name=name,passed=True))
        except Exception as exc:
            checks.append(dict(name=name,passed=False,detail=f'{type(exc).__name__}: {exc}'))
    def integrity():
        manifest = _read(root/'manifest.json')
        required = {'config.json','tolerance_config.json','dataset/all_requests.jsonl','REPORT.md','figures/coverage.png',
                    *OUTPUTS.values(),*EXTRA.values()}
        if manifest['stage'] != 'neighborhood_slo' or not required <= manifest['artifacts'].keys():
            raise ValueError('Wrong stage or incomplete artifacts')
        for relative, entry in manifest['artifacts'].items():
            path = (root/relative).resolve()
            if Path(relative).is_absolute() or root not in path.parents or relative in ('manifest.json','validation.json'):
                raise ValueError('Unsafe artifact path')
            if file_hash(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
                raise ValueError('Artifact hash differs: '+relative)
    def source():
        record = _read(root/'manifest.json')['source']
        if file_hash(root/'dataset/all_requests.jsonl') != record['sha256']:
            # Canonical sorting/serialization may differ; compare records instead.
            original = Path(record['path'])
            if original.is_file() and sorted(_read(original),key=lambda r:(r.get('arrived_at') or '9999',r['request_id'])) != _read(root/'dataset/all_requests.jsonl'):
                raise ValueError('Copied facts differ')
        original = Path(record['path'])
        if not original.is_file():
            warnings.append('Source unavailable; local facts only')
        elif file_hash(original) != record['sha256'] or original.stat().st_size != record['bytes']:
            raise ValueError('Source facts changed')
    def reproduce():
        expected = compute(_read(root/'dataset/all_requests.jsonl'),_read(root/'config.json'),_read(root/'tolerance_config.json'))
        for key, relative in {**OUTPUTS,**EXTRA}.items():
            if expected[key] != _read(root/relative):
                raise ValueError('Recomputed neighbor experiment differs: '+relative)
    def evidence():
        rows = {r['request_id']:r for r in _read(root/'dataset/all_requests.jsonl')}
        splits = {r['request_id']:r for r in _read(root/'dataset/train_test_splits.jsonl')}
        rules = _read(root/'slo_rules.json')
        for metric,pool in rules['pools'].items():
            for sample in pool['samples']:
                state = splits[sample['request_id']]
                if state['split'] != 'train' or not state['training_outcome_visible']:
                    raise ValueError('Reference includes test or uncompleted outcome')
        for assignment in _read(root/'test_assignments.jsonl'):
            for metrics in assignment['grades'].values():
                for metric,rule in metrics.items():
                    e=rule['evidence']
                    if e['method']=='forward_local_quantiles' or e['method']=='sparse_local_p90_times_factors':
                        band=e['search']
                        for rid in e['source_request_ids']:
                            value=rows[rid]['actual_output_tokens']
                            if not band['lower_tokens'] <= value <= band['upper_tokens']:
                                raise ValueError('Sample outside forward neighborhood')
    check('artifact_integrity',integrity)
    check('source_facts',source)
    check('neighborhood_and_tolerance_reproduction',reproduce)
    check('train_only_and_forward_evidence',evidence)
    return dict(stage='neighborhood_slo',passed=all(c['passed'] for c in checks),checks=checks,warnings=warnings)
