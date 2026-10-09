"""Small causal replay, decision plots, timing and self-contained verification."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from .dataset import file_hash, number, timestamp, token, write_json, write_jsonl
from .performance import PerformanceIndex, eligible_metric
from .report import table
from .routing import ModelEndpointCatalog, RoutingEngine, validate_routing_config
from .slo import METRICS
from .slo_experiment import _read, validate_slo_experiment
from .statistics import describe
from .tolerance import validate_config as validate_tolerance_config


BASELINE_FILES = ('config.json', 'dataset/all_requests.jsonl', 'dataset/test_requests.jsonl',
                  'slo_rules.json', 'test_assignments.jsonl', 'coverage.json',
                  'dataset/train_test_splits.jsonl', 'test_results.jsonl')
OUTPUTS = {'requests': 'replay_requests.jsonl', 'decisions': 'decisions.jsonl',
           'observed': 'observed_checks.jsonl', 'summary': 'summary.json', 'sampling': 'sampling.json'}


def select_requests(inputs, assignments, config):
    """Select by incoming features and frozen SLO availability, never outcomes."""
    assigned = {r['request_id']: r for r in assignments}
    groups, excluded = defaultdict(list), Counter()
    for request in inputs:
        if any(token(request.get('predicted_'+f+'_tokens')) is None for f in ('input', 'output')):
            excluded['missing_predictions'] += 1
            continue
        metric = 'ttft' if request['stream_type'] == 'stream' else 'e2e'
        rule = assigned[request['request_id']]['grades'][config['grade']][metric]
        if rule['status'] != 'assigned':
            excluded['unsupported_slo'] += 1
            continue
        groups[(request['stream_type'], rule['range'])].append(request)
    chosen, description = [], []
    for key, items in sorted(groups.items()):
        items.sort(key=lambda r: (timestamp(r['arrived_at']), r['request_id']))
        count = min(len(items), config['sample_requests_per_group'])
        positions = [round(i*(len(items)-1)/(count-1)) for i in range(count)] if count > 1 else [len(items)//2]
        chosen.extend(items[p] for p in positions)
        description.append(dict(mode=key[0], range=key[1], available=len(items), selected=count,
                                request_ids=[items[p]['request_id'] for p in positions]))
    chosen.sort(key=lambda r: (timestamp(r['arrived_at']), r['request_id']))
    return chosen, dict(input_count=len(inputs), selected_count=len(chosen), groups=description,
                        excluded=dict(excluded), policy='evenly_spaced_arrivals_per_supported_predicted_length_group_no_outcome_selection')


def run_replay(facts, inputs, assignments, rules, offerings, tolerance, config):
    config, tolerance = validate_routing_config(config), validate_tolerance_config(tolerance)
    requests, sampling = select_requests(inputs, assignments, config)
    index = PerformanceIndex(facts, config['performance'])
    engine = RoutingEngine(rules, tolerance, config, ModelEndpointCatalog(offerings), index)
    decisions, timings = [], []
    for request in requests:
        decision, timing = engine.route(request)
        decisions.append(decision)
        timings.append(timing)
    fact_index = {r['request_id']: r for r in facts}
    typical = config['performance'].get('statistic', 'p95') != 'p95'
    observed, comparisons = [], defaultdict(list)
    for decision in decisions:
        fact = fact_index[decision['request_id']]
        candidate = next((c for c in decision['candidates'] if c['endpoint_id'] == fact['final_endpoint_id']), None)
        metrics = {}
        for metric in decision['slo']:
            prediction = candidate['estimates'][metric] if candidate else {}
            value = fact.get(METRICS[metric]['field'])
            valid = eligible_metric(fact, metric)
            estimate, mixed = prediction.get('estimate'), prediction.get('mixed_window_p95')
            if typical:
                estimate = candidate['routing_estimates'].get(metric) if candidate else None
            comparable = valid and estimate is not None and mixed is not None
            metrics[metric] = dict(observed=value, conditional_estimate=estimate, mixed_estimate=mixed,
                                   comparable=comparable, status='compared' if comparable else 'not_comparable')
            if comparable:
                pinball = lambda p: .95*max(value-p, 0)+.05*max(p-value, 0)
                comparisons[metric].append(dict(conditional_covered=value <= estimate, mixed_covered=value <= mixed,
                                               conditional_pinball=pinball(estimate), mixed_pinball=pinball(mixed),
                                               conditional_abs_error=abs(value-estimate)))
        observed.append(dict(request_id=fact['request_id'], historical_final_endpoint=fact['final_endpoint_id'],
                             historical_success=fact['success'], selected_endpoint=decision['slots'][0], metrics=metrics,
                             scope='observed_historical_endpoint_only_no_counterfactual_outcome'))
    diagnostic = []
    for metric in METRICS:
        items = comparisons[metric]
        diagnostic.append(dict(metric=metric, comparable=len(items),
            conditional_coverage=sum(i['conditional_covered'] for i in items)/len(items) if items else None,
            mixed_coverage=sum(i['mixed_covered'] for i in items)/len(items) if items else None,
            conditional_pinball_mean=sum(i['conditional_pinball'] for i in items)/len(items) if items else None,
            mixed_pinball_mean=sum(i['mixed_pinball'] for i in items)/len(items) if items else None))
        if typical:
            diagnostic[-1] = dict(metric=metric, comparable=len(items),
                statistic=config['performance']['statistic'],
                conditional_absolute_error_mean=sum(i['conditional_abs_error'] for i in items)/len(items) if items else None,
                interpretation='single_observed_endpoint_diagnostic_not_tail_coverage_or_counterfactual_benefit')
    reasons = Counter(reason for d in decisions for c in d['candidates'] for reason in c['exclusion_reasons'])
    summary = dict(stage='request_conditioned_routing', replay_count=len(requests),
                   selected_count=sum(d['status'] == 'selected' for d in decisions),
                   no_feasible_count=sum(d['status'] != 'selected' for d in decisions),
                   selected_endpoints=dict(Counter(d['slots'][0] for d in decisions if d['slots'][0] is not None)),
                   candidate_exclusions=dict(reasons), observed_prediction_diagnostics=diagnostic,
                   interpretation='logic_verification_only_no_production_promise_or_measured_routing_benefit')
    if 'selection_policy' in config:
        summary.update(selected_count=sum(d['slots'][0] is not None for d in decisions),
                       no_feasible_count=sum(d['slots'][0] is None for d in decisions),
                       regular_selected_count=sum(d['status'] == 'selected' for d in decisions),
                       low_evidence_selected_count=sum(d['status'] == 'selected_low_evidence' for d in decisions),
                       dominated_fill_count=sum(c['selection_reason'] == 'dominated_fill' for d in decisions for c in d['candidates']),
                       statistic=config['performance'].get('statistic', 'p95'),
                       slo_limit_multiplier=config['selection_policy']['slo_limit_multiplier'])
    return dict(requests=requests, decisions=decisions, observed=observed, summary=summary, sampling=sampling, timings=timings)


def _timing_summary(timings):
    return {field: describe(t[field]/1_000_000 for t in timings)
            for field in ('state_update_ns', 'lookup_ns', 'prediction_and_cost_ns', 'pareto_ns', 'decision_ns', 'total_ns')}


def plot_decision(decision, config, path):
    from .plots import plt
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    available = [c for c in decision['candidates'] if c['price']['cost'] is not None and c['performance_loss'] is not None]
    for i, c in enumerate(available):
        selected = c['endpoint_id'] == decision['slots'][0]
        color = '#217a3b' if selected else '#2878b5' if c['pareto'] else '#999999'
        ax.scatter(c['price']['cost'], c['performance_loss'], c=color, s=110 if selected else 75,
                   marker='*' if selected else 'o' if c['feasible'] else 'x', zorder=3)
        name = c['endpoint_id'].encode('ascii', errors='ignore').decode() or f'EP{i+1}'
        ax.annotate(name, (c['price']['cost'], c['performance_loss']), xytext=(7, 7), textcoords='offset points')
    ax.axhline(1, color='#b63e36', linestyle=':', label='Loss=1 (stream gates are per metric)')
    reference, lam = decision['cost_reference'], config['lambda_cost']
    best = next((c for c in available if c['endpoint_id'] == decision['slots'][0]), None)
    if best and reference and 0 < lam < 1:
        xs = [0, max(c['price']['cost'] for c in available)*1.2]
        ys = [(best['score']-lam*x/reference)/(1-lam) for x in xs]
        ax.plot(xs, ys, '--', color='#217a3b', linewidth=1, label='Selected score iso-line')
    if available:
        ax.set_xlim(0, max(c['price']['cost'] for c in available)*1.25 or 1)
        ax.set_ylim(0, max(1.05, max(c['performance_loss'] for c in available)*1.25))
    else:
        ax.text(.5, .5, 'No endpoint with supported price + performance estimate', ha='center', transform=ax.transAxes)
    unavailable = len(decision['candidates'])-len(available)
    ax.set(xlabel=f'Estimated request cost ({config["currency"]})', ylabel='SLO-normalized performance loss (lower is better)',
           title=f'{decision["request_id"]} | {decision["stream_type"]}\n'
                 f'in={decision["predicted_input_tokens"]}, out={decision["predicted_output_tokens"]} | '
                 f'{decision["status"]}, unavailable points={unavailable}')
    ax.grid(alpha=.2)
    if 'selection_policy' in config:
        by_endpoint = {c['endpoint_id']: c for c in decision['candidates']}
        selected_rows = [by_endpoint[name] for name in decision['slots'] if name is not None]
        notes = ' | '.join(f'{i+1}:{c["endpoint_id"].encode("ascii", errors="ignore").decode()}:{c["selection_reason"]}'
                           for i, c in enumerate(selected_rows))
        ax.text(.01, .99, decision['selection_mode']+'\n'+notes, transform=ax.transAxes,
                va='top', fontsize=7, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
    ax.legend(loc='best', fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def report_replay(data, config, tolerance):
    if 'selection_policy' in config:
        return report_available_replay(data, config)
    summary, rows = data['summary'], []
    for i, decision in enumerate(data['decisions'], 1):
        rows.append([decision['request_id'], decision['stream_type'], decision['predicted_input_tokens'],
                     decision['predicted_output_tokens'], decision['status'], ', '.join(str(x) for x in decision['slots']),
                     f'[图](figures/{i:03d}.png)'])
    evidence = [[d['request_id'], c['endpoint_id'], metric, e['status'], e.get('samples', 0),
                 round(e.get('effective_samples', 0), 2), e.get('window_minutes'), e['estimate'], e.get('mixed_window_p95')]
                for d in data['decisions'] for c in d['candidates'] for metric, e in c['estimates'].items()]
    timing = _timing_summary(data['timings'])
    timing_rows = [[name, s['n'], *[round(s[k], 3) if s[k] is not None else None for k in ('mean', 'p50', 'p95', 'max')]]
                   for name, s in timing.items()]
    return '\n\n'.join([
        '# BP 请求长度条件化性能估计与Pareto回放',
        f"本轮回放{summary['replay_count']}条：选出候选{summary['selected_count']}条，无可用候选{summary['no_feasible_count']}条。"
        '不执行真实Endpoint请求，不模拟繁忙、容量或健康状态，不接其他路由策略。原SLO训练规则及10%容差保持不变。',
        '## 1. 数据和时序边界\n\n'
        f'SLO继续采用分模式70/30训练测试切分、±50 token模拟预测；本轮使用{config["grade"]}档。'
        '按测试输入的模式、预测长度桶、已冻结SLO支持情况分组，每组按到达顺序均匀选择请求，不根据结果或路由成功率挑选。'
        '筛选统计见`sampling.json`，预测缺失及SLO不支持的请求单列。'
        '性能缓存按完整历史日志的完成时间推进，只允许finished_at严格早于当前到达时刻；'
        '可以使用较早测试请求已完成的反馈，但不重新训练SLO。这是预先定义的因果历史回放，不是完全冻结训练历史。'
        '历史环境仍来自原路由结果，不能假设重新选路后会产生同样的后续环境。',
        '## 2. 当前请求相关性能估计\n\n'
        '`PerformanceIndex`以(model, endpoint, stream_type)及输入/输出长度索引保存样本引用，过期时同步清理。'
        '只取成功、单次尝试、时间有效、token有效及指标有效的记录，避免将重试总耗时归因给最终Endpoint。'
        'E2E同时匹配输入/输出长度；TTFT匹配输入；'
        + ('TPOT匹配输入长度。' if config['performance']['metric_features']['tpot']==['input'] else 'TPOT同时匹配输入/输出。')
        + '这是显式条件规则，并非已经证明最优的特征选择。\n\n'
        '`匹配余量=max(绝对token余量, 预测tokens×相对余量)`\n\n'
        '`w=2^(-完成后年龄/半衰期) × exp(-0.5×Σ(长度差/匹配余量)^2)`\n\n'
        '在匹配后的成功有效样本中选择达到目标样本数和有效样本门槛的最短时间窗；全部不足时检查最大窗。'
        '样本数或有效样本数不足则estimate=null，不回退到长短混合P95。'
        '`mixed_window_p95`只作同一时间窗的混合统计诊断，不进入决策。'
        f"当前半衰期{config['performance']['half_life_minutes']}分钟、目标{config['performance']['target_samples']}条、"
        f"最低{config['performance']['minimum_samples']}条/有效{config['performance']['minimum_effective_samples']}条；"
        '这些是固定工程实验配置，不是生产可信度保证，也未根据本轮测试结果搜索或放宽。'
        '窗口大小、匹配范围、权重、来源请求ID和完成时间全部保存在`decisions.jsonl`中。',
        '## 3. SLO、成本与Pareto\n\n'
        '`验收上限=T+max(T×相对容差,绝对容差)`；流式TTFT、TPOT必须分别通过，不能用加权平均抵消超限。'
        '`成本=(预测输入tokens×输入价+预测输出tokens×输出价)/1,000,000`，按显式配置汇率统一币种；'
        '价格档按输入长度及明确条件匹配，重叠、缺失、未知币种均返回不可用；不假定缓存折扣。'
        '`非流式性能损失=估计E2E/验收E2E上限`；'
        '`流式性能损失=η×估计TTFT/验收TTFT上限+(1−η)×估计TPOT/验收TPOT上限`。'
        '`score=λ×成本/可用候选正成本中位数+(1−λ)×性能损失`。'
        '先做单项SLO检查，再取非支配前沿，按score及endpoint_id排序，仅取前沿前三，不足填null；不做稳定性优先补位。'
        'BP仅C/D两个配置Endpoint，第三位置留空；无可用Endpoint时三个位置均为空，不强行选择。',
        '## 4. 逐请求结果\n\n'+table(['请求', '模式', '预测输入', '预测输出', '状态', '前三位置', 'Pareto图'], rows),
        '## 5. 性能证据与混合窗口对照\n\n'+table(
            ['请求', 'Endpoint', '指标', '状态', '同类n', '有效n', '窗口分钟', '条件P95', '混合P95'], evidence)+
        '\n\nJSON单位：E2E/TTFT为ms，TPOT为ms/token。TTFT和TPOT仍为已有日志代理，不能解释为用户逐token可见延迟保证。'
        '实际观测对照只在历史真实Endpoint及可比较记录上计算，见`observed_checks.jsonl`和`summary.json`。'
        '分位数pinball损失越小越好，但这十余条样本不足以证明预测稳定改善，也没有未选择Endpoint的反事实结果。',
        '## 6. 路由开销\n\n'+table(['阶段(ns字段，显示ms)', 'n', '均值ms', 'P50 ms', 'P95 ms', '最大ms'], timing_rows)+
        '\n\n`decision_ns`包括查表、性能估计、成本及Pareto；`state_update_ns`单列完成事件导入/缓存清理。'
        '计时截止Pareto排序返回，包含本调试版混合统计诊断和证据计算，不包括最终解释对象封装。'
        '绘图、文件写出、数据加载和SLO训练不计入决策计时。当前是小规模本机单次测量，不是生产吞吐或并发基准。'
        'timings.jsonl保留逐请求原始ns；验证器校验计时结构和阶段和，但不要求墙钟计时逐次一致。',
        '## 7. 已验证与未实现\n\n'
        '本轮验证代码与冻结数据一致、样本来源无未来标签、长度匹配可解释、SLO逐项检查及Pareto前三契约成立。'
        '验证通过不代表Endpoint实际调用成功、真实延迟改善或正式业务承诺成立。'
        '尚未接入真实token预测、线上观测、真实请求执行/重试、容量健康准入、繁忙调度及其他策略收益对比。'
        '旧endpoint_routing_strategy项目和已归档优化方案均未调用或修改。',
    ])+'\n'


def report_available_replay(data, config):
    policy, summary = config['selection_policy'], data['summary']
    rows = [[d['request_id'], d['stream_type'], d['predicted_input_tokens'], d['predicted_output_tokens'],
             d['status'], ', '.join(str(value) for value in d['slots']), f'[图](figures/{i:03d}.png)']
            for i, d in enumerate(data['decisions'], 1)]
    evidence = [[d['request_id'], c['endpoint_id'], metric, e.get('samples', 0),
                 round(e.get('effective_samples', 0), 2), c['routing_estimates'][metric],
                 d['slo'][metric]['acceptance_limit'], c['evidence_quality'], c['selection_reason'],
                 ', '.join(c['exclusion_reasons'])]
                for d in data['decisions'] for c in d['candidates'] for metric, e in c['estimates'].items()]
    return '\n\n'.join([
        '# BP 可用性优先路由回放',
        f"回放{summary['replay_count']}条，常规选择{summary['regular_selected_count']}条，"
        f"低证据兜底{summary['low_evidence_selected_count']}条，无候选{summary['no_feasible_count']}条。"
        '这是候选生成结果，不是实际调用成功率或SLO覆盖率。',
        '## 1. 性能和SLO\n\n'
        f"本轮使用{config['performance']['statistic']}，配置可切换mean或p50。"
        '`mean=Σ(w×延迟)/Σw`；`w=2^(-样本年龄/半衰期)×exp(-0.5×Σ归一化长度差²)`。'
        '保留请求长度条件化；E2E匹配输入和输出，TTFT/TPOT匹配输入。'
        '常规证据门槛仍为30条原始/20条有效样本，目标40条，窗口60至1440分钟。'
        f"验收上限=原始SLO×{policy['slo_limit_multiplier']}，这是替代原评估容差的独立路由配置，"
        '不再叠加10%，不是1.1×1.2。流式两项分别检查，任何已知指标超限不能被其他指标抵消。'
        '均值/P50描述典型耗时，不能解释成P95保证；更高可选率不自动代表更高SLO达标率。',
        '## 2. 候选排序和降级\n\n'
        '首先用证据充分且通过逐项SLO检查的Endpoint组成常规集合；Pareto前沿按score排序在前，'
        '不足三个位置时，用同一常规集合内被支配的Endpoint按score补齐。'
        '`score=λ×成本/可用候选正成本中位数+(1−λ)×性能损失`，成本包含预测输入/输出长度，'
        '非流式损失为E2E/验收上限，流式损失为η×TTFT比值+(1−η)×TPOT比值。'
        '只有常规集合为空时，才启用低证据兜底：先按相似稀疏样本计算相同统计量，逐项不超限才能加入；'
        '完全没有相似样本或缺某项指标时可作为unknown best-effort候选，不伪造性能=0、不画伪Pareto点。'
        '未知候选在已知稀疏候选之后，按已知指标数量、预测成本和ID确定性排序。'
        '已知指标超限、明确禁用、价格不合法或SLO缺失仍不能因兜底被放行。'
        '不使用长短请求混合P95补缺。兜底记录selected_low_evidence和best_effort_no_slo_guarantee。',
        '## 3. 逐请求结果\n\n'+table(['请求', '模式', '预测输入', '预测输出', '状态', '前三位置', '图'], rows),
        '## 4. 指标及证据\n\nJSON单位：E2E/TTFT为ms，TPOT为ms/token。\n\n'
        +table(['请求', 'Endpoint', '指标', 'n', '有效n', '路由估计', '验收上限', '证据', '入选原因', '排除原因'], evidence),
        '## 5. 回放边界及验证\n\n'
        'SLO冻结于原70/30训练规则，测试预测仍为固定±50扰动。每个预测长度组取按到达时间排序的中间一条，'
        '抽样不看结果；使用哪个SLO版本以baseline目录中的冻结快照为准，不能假设不同基线阈值相同。'
        '仅使用当前到达前严格完成的历史结果。'
        '原历史流量不是新策略产生的反事实流量。没有真实调用、重试执行、容量/健康模拟或繁忙机制。'
        '完全无历史兜底仅做合约及单元测试；结果中的低证据兜底需人工看到风险标记。'
        '验证器重算原SLO、抽样、全部候选及证据、排名和计时结构，不能验证实际调用成功。',
        '## 6. 路由开销\n\n'+table(['阶段', '均值ms', 'P95 ms'],
            [[key, round(value['mean'], 3), round(value['p95'], 3)] for key, value in _timing_summary(data['timings']).items()]),
        '计时包含调试证据和混合窗口诊断，不含画图和写文件；7条本机测量不是生产吞吐基准。'
    ])


def _compute(root):
    baseline = root/'baseline'
    return run_replay(_read(baseline/'dataset/all_requests.jsonl'), _read(baseline/'dataset/test_requests.jsonl'),
                      _read(baseline/'test_assignments.jsonl'), _read(baseline/'slo_rules.json'),
                      _read(root/'endpoint_offerings.jsonl'), _read(root/'tolerance_config.json'), _read(root/'config.json'))


def prepare_replay(baseline_dir, offerings_path, run_dir, config_path, tolerance_path):
    baseline, root, offerings_path = Path(baseline_dir).resolve(), Path(run_dir).resolve(), Path(offerings_path).resolve()
    if root.exists():
        raise FileExistsError('Refusing to overwrite '+str(root))
    baseline_stage = _read(baseline/'manifest.json').get('stage')
    if baseline_stage == 'neighborhood_slo':
        from .neighborhood_experiment import validate as validate_baseline
    else:
        validate_baseline = validate_slo_experiment
    if not validate_baseline(baseline)['passed']:
        raise ValueError('Baseline SLO package failed validation')
    config, tolerance = validate_routing_config(_read(config_path)), validate_tolerance_config(_read(tolerance_path))
    package = offerings_path.parent.parent
    if (package/'manifest.json').is_file():
        registered = _read(package/'manifest.json')['artifacts'].get(offerings_path.relative_to(package).as_posix())
        if registered is None or file_hash(offerings_path) != registered['sha256']:
            raise ValueError('Endpoint configuration does not match its registered package')
    root.mkdir(parents=True, exist_ok=False)
    sources = {}
    for relative in BASELINE_FILES:
        destination = root/'baseline'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline/relative, destination)
        sources['baseline/'+relative] = dict(path=str(baseline/relative), sha256=file_hash(baseline/relative))
    shutil.copyfile(offerings_path, root/'endpoint_offerings.jsonl')
    sources['endpoint_offerings.jsonl'] = dict(path=str(offerings_path), sha256=file_hash(offerings_path))
    write_json(root/'config.json', config)
    write_json(root/'tolerance_config.json', tolerance)
    data = _compute(root)
    for key, relative in OUTPUTS.items():
        (write_jsonl if relative.endswith('.jsonl') else write_json)(root/relative, data[key])
    write_jsonl(root/'timings.jsonl', data['timings'])
    write_json(root/'timing_summary.json', _timing_summary(data['timings']))
    (root/'REPORT.md').write_text(report_replay(data, config, tolerance), encoding='utf-8')
    for i, decision in enumerate(data['decisions'], 1):
        plot_decision(decision, config, root/'figures'/f'{i:03d}.png')
    project = Path(__file__).resolve().parents[1]
    write_json(root/'manifest.json', dict(schema_version='1.0', stage='request_conditioned_routing',
        created_at=datetime.now(timezone.utc).isoformat(), sources=sources,
        artifacts={p.relative_to(root).as_posix(): dict(sha256=file_hash(p), bytes=p.stat().st_size)
                   for p in sorted(root.rglob('*')) if p.is_file()},
        code_sha256={p.relative_to(project).as_posix(): file_hash(p) for p in sorted((project/'bp_slo').glob('*.py'))}))
    result = validate_replay(root)
    write_json(root/'validation.json', result)
    return root, data, result


def validate_replay(run_dir):
    root, checks, warnings = Path(run_dir).resolve(), [], []
    def check(name, action):
        try:
            action()
            checks.append(dict(name=name, passed=True))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, detail=f'{type(exc).__name__}: {exc}'))
    def integrity():
        manifest = _read(root/'manifest.json')
        required = {'config.json', 'tolerance_config.json', 'endpoint_offerings.jsonl', 'REPORT.md',
                    'timings.jsonl', 'timing_summary.json', *OUTPUTS.values(), *('baseline/'+r for r in BASELINE_FILES)}
        if manifest['stage'] != 'request_conditioned_routing' or not required <= manifest['artifacts'].keys():
            raise ValueError('Wrong stage or incomplete manifest')
        for relative, entry in manifest['artifacts'].items():
            path = (root/relative).resolve()
            if Path(relative).is_absolute() or root not in path.parents or relative in ('manifest.json', 'validation.json'):
                raise ValueError('Unsafe artifact path')
            if file_hash(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
                raise ValueError('Artifact digest mismatch: '+relative)
        decisions = _read(root/'decisions.jsonl')
        for i in range(1, len(decisions)+1):
            if f'figures/{i:03d}.png' not in manifest['artifacts']:
                raise ValueError('Missing request Pareto figure')
    def sources():
        entries = _read(root/'manifest.json')['sources']
        if set(entries) != {'endpoint_offerings.jsonl', *('baseline/'+r for r in BASELINE_FILES)}:
            raise ValueError('Incomplete source coverage')
        for relative, entry in entries.items():
            if file_hash(root/relative) != entry['sha256']:
                raise ValueError('Snapshot differs from registered source')
            original = Path(entry['path'])
            if not original.is_file():
                warnings.append('Source unavailable; using registered copied data: '+str(original))
            elif file_hash(original) != entry['sha256']:
                raise ValueError('Source changed: '+relative)
    def reproduce():
        if _read(root/'baseline/config.json').get('stage') == 'neighborhood_slo':
            from .neighborhood_slo import run_experiment
        else:
            from .slo import run_experiment
        baseline = run_experiment(_read(root/'baseline/dataset/all_requests.jsonl'), _read(root/'baseline/config.json'))
        for key, relative in (('rules', 'slo_rules.json'), ('test_inputs', 'dataset/test_requests.jsonl'),
                              ('assignments', 'test_assignments.jsonl'), ('coverage', 'coverage.json'),
                              ('splits', 'dataset/train_test_splits.jsonl'), ('results', 'test_results.jsonl')):
            if baseline[key] != _read(root/'baseline'/relative):
                raise ValueError('Frozen SLO baseline failed reproduction: '+relative)
        data = _compute(root)
        for key, relative in OUTPUTS.items():
            if data[key] != _read(root/relative):
                raise ValueError('Recomputed routing differs: '+relative)
    def audit():
        facts = {r['request_id']: r for r in _read(root/'baseline/dataset/all_requests.jsonl')}
        for decision in _read(root/'decisions.jsonl'):
            for candidate in decision['candidates']:
                for metric, estimate in candidate['estimates'].items():
                    for rid in estimate['source_request_ids']:
                        row = facts[rid]
                        if (rid == decision['request_id'] or timestamp(row['finished_at']) >= timestamp(decision['arrived_at'])
                                or row['model_id'] != decision['model_id'] or row['stream_type'] != decision['stream_type']
                                or row['final_endpoint_id'] != candidate['endpoint_id'] or not eligible_metric(row, metric)):
                            raise ValueError('Invalid performance evidence source')
                        for feature, band in estimate['bands'].items():
                            if not band['lower'] <= row['actual_'+feature+'_tokens'] <= band['upper']:
                                raise ValueError('Evidence outside request length band')
    def timing():
        items, decisions = _read(root/'timings.jsonl'), _read(root/'decisions.jsonl')
        if [r['request_id'] for r in items] != [r['request_id'] for r in decisions]:
            raise ValueError('Timing and decision IDs differ')
        for item in items:
            fields = ('state_update_ns', 'lookup_ns', 'prediction_and_cost_ns', 'pareto_ns', 'decision_ns', 'total_ns')
            if any(type(item[f]) is not int or item[f] < 0 for f in fields):
                raise ValueError('Timing must be nonnegative integer nanoseconds')
            if item['decision_ns'] != sum(item[f] for f in ('lookup_ns', 'prediction_and_cost_ns', 'pareto_ns')):
                raise ValueError('Decision timing phase sum differs')
            if item['total_ns'] != item['state_update_ns']+item['decision_ns']:
                raise ValueError('Total timing phase sum differs')
        if _timing_summary(items) != _read(root/'timing_summary.json'):
            raise ValueError('Timing summary does not reproduce')
    check('artifact_integrity', integrity)
    check('source_integrity', sources)
    check('frozen_slo_and_routing_reproduction', reproduce)
    check('causal_length_evidence_audit', audit)
    check('timing_structure', timing)
    return dict(stage='request_conditioned_routing', passed=all(c['passed'] for c in checks), checks=checks, warnings=warnings)
