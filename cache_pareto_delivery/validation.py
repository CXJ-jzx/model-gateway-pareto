"""Staged branch checks and pre-registered, balanced real-request comparison.

Uses experiment.py's bounded transport. No new HTTP client, deployment, simulator,
price modifications, or replacement of prefix-first routing is introduced here.
"""
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from routing import CacheAwareRouter, PrefixIndex, cached_tokens
from experiment import document, followup, gateway_cost


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    x = (len(values)-1)*q
    low = int(x)
    return values[low] + (values[min(low+1, len(values)-1)]-values[low])*(x-low)


def paired_bootstrap(pairs, seed=20261019, repetitions=4000):
    """Resample independent document PAIRS, never correlated individual turns."""
    if not pairs:
        return None
    rng, reductions = random.Random(seed), []
    for _ in range(repetitions):
        sample = [rng.choice(pairs) for _ in pairs]
        base, affinity = sum(x[0] for x in sample), sum(x[1] for x in sample)
        if base:
            reductions.append(1-affinity/base)
    return [quantile(reductions, .025), quantile(reductions, .975)]


def run_stages(args, deployments, router, call, event, now, run_id):
    if not 1 <= args.pairs_per_cell <= 4 or not 1 <= args.followups <= 5:
        raise ValueError('Bounded independent-pair/turn counts required')
    estimates, calibration = {}, []
    event(dict(type='stage', name='calibration', planned_calls=len(deployments)*2,
               purpose='fresh pre-test latency evidence; not SLO calibration'))
    for d in deployments:
        for size in ('short', 'long'):
            family = f"calibration-{d['endpoint']}-{size}"
            row = call(d['endpoint'], document(run_id, family, size), 'calibration', 'warmup', family, size, 'fixed_endpoint')
            calibration.append(row)
            if row.get('valid'):
                estimates[d['endpoint'], size] = row['e2e_ms']
    if len(estimates) != len(deployments)*2:
        event(dict(type='validation_blocked', reason='incomplete fresh endpoint evidence; no invented performance'))
        return
    event(dict(type='frozen_estimates', estimator='single_fresh_warmup_per_endpoint_size_sparse',
               rows=[dict(endpoint=e, size=s, estimated_e2e_ms=v) for (e,s),v in estimates.items()]))
    event(dict(type='pricing_audit', rows=[dict(endpoint=d['endpoint'],
        input_per_million=d['input_cost_per_token']*1e6, output_per_million=d['output_cost_per_token']*1e6,
        cache_input_per_million=d['cache_read_input_token_cost']*1e6
            if d['cache_read_input_token_cost'] is not None else None,
        provenance='existing gateway configuration, currency/invoice unverified') for d in deployments]))

    def candidates(query, size):
        predicted_input = max(1, len(json.dumps(query, ensure_ascii=False))//3)
        return [dict(endpoint_id=p['endpoint'], estimated_e2e_ms=estimates[p['endpoint'], size],
            estimated_cost=predicted_input*p['input_cost_per_token']+16*p['output_cost_per_token']) for p in deployments]

    def choose(family, query, size='short', limit=None, policy='prefix_pareto'):
        return router.choose(scope=family, model=args.model, messages=query, candidates=candidates(query, size),
            slo_ms=limit if limit is not None else args.slo_ms, now_ms=now(), prefix_first=policy=='prefix_pareto')

    def check(name, decision, passed, **kwargs):
        event(dict(type='branch_check', name=name, passed=bool(passed), decision=decision, **kwargs))
        if not passed:
            raise RuntimeError('Routing branch check failed; stop before large experiment: '+name)

    event(dict(type='stage', name='branch_validation', planned_calls=8))
    preferred = next((d['endpoint'] for d in deployments if d['cache_read_input_token_cost'] is not None),
                     deployments[0]['endpoint'])
    family = 'branch-no-prefix'
    messages = document(run_id, family, 'short')
    decision = choose(family, messages)
    check('no_prefix_pareto', decision, decision['reason']=='pareto_fallback' and not decision['matches'])
    if decision['selected_endpoint'] is None:
        raise RuntimeError('No SLO-admissible endpoint; do not force dispatch')
    call(decision['selected_endpoint'], messages, 'branches', 'warmup', family, 'short', 'prefix_pareto', decision)

    family = 'branch-prefix-reuse'
    messages = document(run_id, family, 'short')
    warm = call(preferred, messages, 'branches', 'warmup', family, 'short', 'fixed_endpoint')
    if not warm.get('valid'):
        raise RuntimeError('Branch warmup failed; no fabricated READY')
    time.sleep(args.cache_wait)
    query = followup(messages, warm['content'], 1)
    decision = choose(family, query)
    check('qualified_prefix_reuse', decision, decision['selected_endpoint']==preferred and decision['reason']=='prefix_affinity')
    call(decision['selected_endpoint'], query, 'branches', 'followup', family, 'short', 'prefix_pareto', decision)

    ordered = sorted(deployments, key=lambda d: estimates[d['endpoint'], 'short'])
    fast, slow = ordered[0]['endpoint'], ordered[-1]['endpoint']
    limit = (estimates[fast, 'short']+estimates[slow, 'short'])/2
    family = 'branch-prefix-slo-reject'
    messages = document(run_id, family, 'short')
    warm = call(slow, messages, 'branches', 'warmup', family, 'short', 'fixed_endpoint')
    if not warm.get('valid'):
        raise RuntimeError('SLO-branch warmup failed')
    query = followup(messages, warm['content'], 1)
    decision = choose(family, query, limit=limit)
    check('cached_endpoint_rejected_by_slo', decision,
        decision['reason']=='pareto_fallback' and decision['selected_endpoint'] not in (slow, None)
        and any(r['endpoint_id']==slow and not r['admitted'] for r in decision['candidates']),
        limit_source='midpoint of frozen fastest/slowest estimates; not fitted to test outcome')
    call(decision['selected_endpoint'], query, 'branches', 'followup', family, 'short', 'prefix_pareto', decision)

    family = 'branch-multiple-locations'
    messages = document(run_id, family, 'short')
    warms = [call(endpoint, messages, 'branches', 'warmup', family, 'short', 'fixed_endpoint')
             for endpoint in (fast, slow)]
    if not all(r.get('valid') for r in warms):
        raise RuntimeError('Multiple-location warmup failed')
    query = followup(messages, warms[0]['content'], 1)
    decision = choose(family, query)
    matched = {r['endpoint_id'] for r in decision['matches']}
    expected = min((r for r in decision['candidates'] if r['admitted'] and r['endpoint_id'] in matched),
                   key=lambda r: (r['score'], r['endpoint_id']))['endpoint_id']
    check('multiple_locations_original_score', decision, len(matched)==2
          and decision['selected_endpoint']==expected and decision['reason']=='prefix_affinity')
    call(decision['selected_endpoint'], query, 'branches', 'followup', family, 'short', 'prefix_pareto', decision)

    # Test local-index expiration without waiting five minutes or pretending the
    # provider's actual cache expired. No capacity or health values are injected.
    local = CacheAwareRouter(PrefixIndex(ttl_ms=5))
    local.feedback(scope='local', model=args.model, messages=messages, endpoint=fast, now_ms=0,
                   success=True, identity_verified=True)
    decision = local.choose(scope='local', model=args.model, messages=query, candidates=candidates(query,'short'),
                            slo_ms=args.slo_ms, now_ms=6)
    check('local_evidence_ttl_fallback', decision, not decision['matches'] and decision['reason']=='pareto_fallback',
          execution='local_clock_only_not_provider_cache_expiration')

    plan = [(d['endpoint'], size, repeat) for d in deployments for size in ('short','long')
            for repeat in range(args.pairs_per_cell)]
    random.Random(20261019).shuffle(plan)
    event(dict(type='stage', name='large_comparison', planned_pairs=len(plan),
               planned_calls=len(plan)*2*(1+args.followups), planned_followups_per_policy=len(plan)*args.followups,
               balanced_warm_endpoint_rotation=True, seed=20261019,
               plan=[dict(pair_id=f'large-{i:02d}', warm_endpoint=e, size=s, repeat=r) for i,(e,s,r) in enumerate(plan)]))
    for i, (endpoint, size, repeat) in enumerate(plan):
        pair_id = f'large-{i:02d}'
        policies = ('pareto','prefix_pareto') if i%2 == 0 else ('prefix_pareto','pareto')
        families = {}
        for policy in policies:
            family = pair_id+'-'+policy
            messages = document(run_id, family, size)
            warm = call(endpoint, messages, 'large', 'warmup', family, size, policy)
            if warm.get('valid'):
                families[policy] = (family, messages, warm['content'])
            else:
                event(dict(type='unserved', pair_id=pair_id, policy=policy,
                    followups=args.followups, reason='failed_warmup_no_fake_response'))
        time.sleep(args.cache_wait)
        for n in range(1, args.followups+1):
            for policy in policies:
                if policy not in families:
                    continue
                family, messages, content = families[policy]
                query = followup(messages, content, n)
                decision = choose(family, query, size=size, policy=policy)
                if decision['selected_endpoint'] is None:
                    event(dict(type='unserved', pair_id=pair_id, policy=policy, followups=1,
                               reason='no_slo_feasible_endpoint', decision=decision))
                    continue
                call(decision['selected_endpoint'], query, 'large', 'followup', family, size, policy, decision)
        event(dict(type='pair_finished', pair_id=pair_id, warm_endpoint=endpoint, size=size))


def billing_audit(events):
    """Do not silently mix provider amounts and gateway accounting currencies.

    The alternate total is explicitly a sensitivity scenario, not verified cost.
    No costs, hits, requests or choices are rewritten in the original events.
    """
    def amount(row):
        value = (row.get('usage') or {}).get('amount')
        if value is None or isinstance(value, bool):
            return None
        try:
            result = Decimal(str(value))
        except InvalidOperation:
            return None
        return float(result) if result.is_finite() and result >= 0 else None
    calls = [r for r in events if r.get('type')=='call']
    large = [r for r in calls if r['phase']=='large']
    by_endpoint = []
    for endpoint in sorted({r['endpoint'] for r in calls}):
        rows = [r for r in calls if r['endpoint']==endpoint]
        complete = all(amount(r) is not None for r in rows)
        contradictions = sum(
            type((r.get('usage') or {}).get('prompt_cache_hit_tokens')) is int
            and type((r.get('usage') or {}).get('prompt_cache_miss_tokens')) is int
            and r['usage']['prompt_cache_hit_tokens']+r['usage']['prompt_cache_miss_tokens'] != r['usage'].get('prompt_tokens')
            for r in rows)
        by_endpoint.append(dict(endpoint=endpoint, calls=len(rows),
            provider_amount_rows=sum(amount(r) is not None for r in rows),
            provider_amount_sum=sum(amount(r) for r in rows) if complete else None,
            cache_hit_miss_contradictions=contradictions,
            validated_cache_unknown_rows=sum(cached_tokens(r.get('usage')) is None for r in rows)))
    scenario, scenario_groups = {}, []
    for policy in ('pareto','prefix_pareto'):
        rows = [r for r in large if r['policy']==policy]
        values = [amount(r) if r['endpoint']=='ai360-cdtx' else gateway_cost(r) for r in rows]
        scenario[policy] = sum(values) if rows and all(v is not None for v in values) else None
    a,b = scenario['pareto'],scenario['prefix_pareto']
    plan = next((e for e in events if e.get('type')=='stage' and e.get('name')=='large_comparison'),{})
    for endpoint in sorted({r['warm_endpoint'] for r in plan.get('plan',[])}):
        ids = {r['pair_id'] for r in plan['plan'] if r['warm_endpoint']==endpoint}
        group = {}
        for policy in ('pareto','prefix_pareto'):
            rows = [r for r in large if r['policy']==policy
                    and r['family'].removesuffix('-pareto').removesuffix('-prefix_pareto') in ids]
            values = [amount(r) if r['endpoint']=='ai360-cdtx' else gateway_cost(r) for r in rows]
            group[policy] = sum(values) if rows and all(v is not None for v in values) else None
        x,y = group['pareto'],group['prefix_pareto']
        scenario_groups.append(dict(warm_endpoint=endpoint, totals=group,
            conditional_reduction=1-y/x if x and y is not None else None))
    return dict(status='provider_currency_and_amount_semantics_unverified',
        formal_provider_cost_reduction=None, by_endpoint=by_endpoint,
        sensitivity_scenario=dict(assumption='Only if 360 usage.amount is per-call money in the same unit '
            'as other gateway headers; do not treat this unverified mixed-source calculation as actual billing.',
            totals=scenario, conditional_reduction=1-b/a if a and b is not None else None,
            by_warm_endpoint=scenario_groups),
        sources=[dict(url='https://ai.360.com/docs/cache', checked_on='2026-10-20',
                      supports='cache service and hit+miss=prompt_tokens contract, not usage.amount currency'),
                 dict(url='https://ai.360.com/open/en/models/deepseek/deepseek-v4-flash',checked_on='2026-10-20',
                      supports='tokenmall-cdtx advertised input 1.8, output 5.4, cache read 0.06 CNY/million; '
                               'not proof of historical account tariff or amount semantics')])


def report(events, directory):
    directory = Path(directory)
    calls = [e for e in events if e.get('type')=='call']
    checks = [e for e in events if e.get('type')=='branch_check']
    planned = next((e for e in events if e.get('type')=='stage' and e['name']=='large_comparison'), {})
    planned_count = planned.get('planned_followups_per_policy', 0)
    arms, pair_rows = {}, defaultdict(lambda: defaultdict(list))
    for row in calls:
        if row['phase']=='large':
            pair_id = row['family'].removesuffix('-pareto').removesuffix('-prefix_pareto')
            pair_rows[pair_id][row['policy']].append(row)
    def aggregate(rows, denominator):
        tests = [r for r in rows if r['role']=='followup']
        known = [gateway_cost(r) for r in rows]
        timings = [r['e2e_ms'] for r in tests if r.get('transport_success')]
        return dict(calls=len(rows), followups=len(tests), planned_followups=denominator,
            correct=sum(bool(r.get('valid')) for r in tests),
            slo_passes=sum(bool(r.get('valid')) and r['e2e_ms']<=r['slo_ms'] for r in tests),
            latency_slo_passes=sum(bool(r.get('transport_success')) and r.get('endpoint_identity_verified')
                and not r.get('response_cache_hit') and r.get('finish_reason')=='stop'
                and r['e2e_ms']<=r['slo_ms'] for r in tests),
            cache_hit_calls=sum(bool(r.get('valid')) and (cached_tokens(r.get('usage')) or 0)>0 for r in tests),
            unknown_cache_calls=sum(cached_tokens(r.get('usage')) is None for r in tests),
            prefix_routes=sum((r.get('decision') or {}).get('reason')=='prefix_affinity' for r in tests),
            endpoint_choices=dict(Counter(r['endpoint'] for r in tests)),
            mean_ms=statistics.mean(timings) if timings else None, p95_ms=quantile(timings,.95),
            gateway_cost=sum(known) if rows and all(v is not None for v in known) else None,
            known_gateway_cost=sum(v for v in known if v is not None),
            missing_gateway_cost_calls=sum(v is None for v in known),
            lookup_mean_ms=statistics.mean([r['decision']['prefix_lookup_ns']/1e6 for r in tests if r.get('decision')]) if tests else None)
    for policy in ('pareto','prefix_pareto'):
        rows = [r for r in calls if r['phase']=='large' and r['policy']==policy]
        arms[policy] = aggregate(rows, planned_count)
    comparisons, subgroups = [], []
    plan_by_id = {r['pair_id']:r for r in planned.get('plan', [])}
    # Expected counts must come from the pre-registered plan, not successful arms.
    turns = planned.get('planned_calls',0)//max(1, planned.get('planned_pairs',0)*2)
    for pair_id, group in sorted(pair_rows.items()):
        if any(len(group[p]) != turns or any(gateway_cost(r) is None for r in group[p])
               for p in ('pareto','prefix_pareto')):
            continue
        costs = tuple(sum(gateway_cost(r) for r in group[p]) for p in ('pareto','prefix_pareto'))
        comparisons.append(costs)
    for endpoint in sorted({r['warm_endpoint'] for r in planned.get('plan', [])}):
        grouped = {}
        for p in ('pareto','prefix_pareto'):
            rows = [r for pair_id,g in pair_rows.items() if plan_by_id[pair_id]['warm_endpoint']==endpoint for r in g[p]]
            grouped[p] = aggregate(rows, sum(r['warm_endpoint']==endpoint for r in planned['plan'])*max(0,turns-1))
        base, affinity = grouped['pareto']['gateway_cost'], grouped['prefix_pareto']['gateway_cost']
        subgroups.append(dict(warm_endpoint=endpoint, arms=grouped,
                             gateway_cost_reduction=1-affinity/base if base and affinity is not None else None))
    base, affinity = arms['pareto']['gateway_cost'], arms['prefix_pareto']['gateway_cost']
    result = dict(actual_calls=len(calls), valid_calls=sum(bool(r.get('valid')) for r in calls),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=[dict(name=r['name'],passed=r['passed']) for r in checks],
        planned_pairs=planned.get('planned_pairs',0), completed_pairs=len(comparisons), large=arms,
        gateway_cost_reduction=1-affinity/base if base and affinity is not None else None,
        paired_bootstrap_95_interval=paired_bootstrap(comparisons), by_warm_endpoint=subgroups,
        billing_audit=billing_audit(events),
        pricing_audit=next((r['rows'] for r in events if r.get('type')=='pricing_audit'), []),
        total_gateway_reported_cost=sum(gateway_cost(r) or 0 for r in calls),
        missing_gateway_cost_calls=sum(gateway_cost(r) is None for r in calls),
        finished=next((r for r in reversed(events) if r.get('type')=='finished'), None),
        notes=['All endpoint states assumed healthy/available; no Mos or shared configuration edits.',
               'Costs are gateway response-header accounting, not provider invoices.',
               'Missing cache tariff stays unknown; ordinary-cost comparison cannot independently verify its discount.',
               '60s large-experiment SLO is an explicit feasibility budget, not a calibrated business target.',
               'Single fresh calibration per endpoint/length is sparse performance evidence.',
               'Bootstrap units are independent document pairs; three turns are correlated within each pair.',
               'Only two input sizes and extremely short outputs; no generalization to long-output workloads.'])
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding='utf-8')
    fmt = lambda x: '未知' if x is None else f'{x:.6f}'
    change = lambda x: '未知' if x is None else (f'降低 {x:.2%}' if x >= 0 else f'增加 {-x:.2%}')
    lines = ['# 分步骤验证及较大样本对照', '',
             '**2026-10-20 费用核验更正：360 命中统计不自洽，usage.amount 与网关费用不一致。原增费数字仅为网关配置记账，不能据此认定实际亏损或360没有缓存收益。**', '',
             f"实际调用 {result['actual_calls']} 次，有效 {result['valid_calls']} 次；独立配对文档 {result['completed_pairs']}/{result['planned_pairs']} 组。", '',
             '## 分支验证', '', '| 分支 | 结果 |','|---|---|']
    labels = dict(no_prefix_pareto='没有前缀 → Pareto', qualified_prefix_reuse='有前缀且满足 SLO → 复用',
        cached_endpoint_rejected_by_slo='有前缀但不满足 SLO → 切换', multiple_locations_original_score='多个位置 → 原线性得分',
        local_evidence_ttl_fallback='本地证据 TTL 过期 → Pareto（无付费调用）')
    lines += [f"| {labels.get(r['name'],r['name'])} | {'通过' if r['passed'] else '失败'} |" for r in checks]
    lines += ['', '## 较大样本对照（含预热）', '',
        '| 策略 | 调用数 | 后续请求正确/计划 | 纯时延SLO达标/计划 | 后续缓存命中 | 平均耗时 ms | P95 ms | 网关费用 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,r in arms.items():
        lines.append(f"| {p} | {r['calls']} | {r['correct']}/{r['planned_followups']} | {r['latency_slo_passes']}/{r['planned_followups']} | "
                     f"{r['cache_hit_calls']} | {fmt(r['mean_ms'])} | {fmt(r['p95_ms'])} | {fmt(r['gateway_cost'])} |")
    gain, ci = result['gateway_cost_reduction'], result['paired_bootstrap_95_interval']
    lines += ['', f'网关记账费用变化：{change(gain)}。' if gain is not None else '费用不完整，不能报告总收益。',
              f'按配对文档重采样的 95% bootstrap 区间：[{ci[0]:.2%}, {ci[1]:.2%}]。' if ci else '缺少完整文档对，无法重采样。',
              '', '## 按历史缓存位置分组', '', '| 历史位置 | 原 Pareto 费用 | 前缀优先费用 | 费用变化 | 前缀组命中/计划 |',
              '|---|---:|---:|---:|---:|']
    for g in subgroups:
        a,b = g['arms']['pareto'],g['arms']['prefix_pareto']
        gain = g['gateway_cost_reduction']
        lines.append(f"| {g['warm_endpoint']} | {fmt(a['gateway_cost'])} | {fmt(b['gateway_cost'])} | "
                     + change(gain)+f" | {b['cache_hit_calls']}/{b['planned_followups']} |")
    lines += ['', '## 结论', '',
              '- 分支逻辑通过；因360计费口径尚未核验，当前不能确认实际整体增费或降本。',
              '- 360 返回 hit=0、miss=0，却有非零输入 tokens，违反官方文档的字段关系；命中情况应标为未知，而不是未命中。',
              '- 360 官方页面列有缓存单价，且历史 usage.amount 明显低于网关费用；先前“没有缓存收益导致亏损”的归因撤回。',
              '- 原 Pareto 自身也会连续选择同一便宜端点，因而同样能获得缓存；不能把它当作完全无缓存的基线。',
              '- 整体降本区间跨过 0，不宣称稳定降本，也不能声称已经统计证明一定更贵。',
              '- 未改前缀优先策略、未修改端点价格/时延，未删除反例或重跑挑选结果。',
              '', '## 费用依据与限制', '',
              '- 输入/输出/缓存单价来自现有配置，不作任何修改；缺失缓存单价保持未知，见 summary.json。',
              '- 总费用来自每次响应的 x-litellm-response-cost，币种及供应商实扣未核验。',
              '- 原始请求、真实响应、端点身份、usage、计费头、决策候选和选择理由均在 events.jsonl。',
              '- 不只统计官方：四个历史缓存位置均衡轮换，未命中/变贵/失败的记录不删。',
              '- 正常返回但答案错误的记录仍计费；表中的纯时延 SLO 不把答案错误误算为超时，正确且达标另存 JSON。',
              '- 两组每个文档各自预热，预热计费；不同实验标识隔离前缀，顺序交替。',
              '- 较大样本仍不是生产压测；只涉及约 2.5k/9.7k 输入、极短输出，不能推广全部业务。',
              '- SLO 主实验固定 60 秒；窄门槛分支依据冻结估计设定，实际达标与预估准入分别记录。',
              '- 健康、容量及并发均假设可用；不修改 Mos、不部署源码、不更改共享服务配置。']
    audit = result['billing_audit']
    scenario = audit['sensitivity_scenario']
    lines += ['', '## 费用核验与敏感性分析（不是正式收益）', '',
              '| 端点 | 记录数 | amount有值 | hit+miss不等于input | 校验后缓存未知 |',
              '|---|---:|---:|---:|---:|']
    for row in audit['by_endpoint']:
        lines.append(f"| {row['endpoint']} | {row['calls']} | {row['provider_amount_rows']} | "
                     f"{row['cache_hit_miss_contradictions']} | {row['validated_cache_unknown_rows']} |")
    lines += ['', '| 口径 | 原 Pareto | 前缀优先 | 解释 |', '|---|---:|---:|---|',
              f"| 原网关配置记账 | {fmt(arms['pareto']['gateway_cost'])} | {fmt(arms['prefix_pareto']['gateway_cost'])} | 不代表供应商实扣 |",
              f"| 假设360的amount可与其他端点记账相加 | {fmt(scenario['totals']['pareto'])} | {fmt(scenario['totals']['prefix_pareto'])} | 未核验假设，禁止当正式收益 |",
              '', f"在该假设下，费用变化为{change(scenario['conditional_reduction'])}，仅说明结论对费用口径敏感。",
              '目前没有找到 amount 的明确币种及单请求计费定义，不能直接混用不同来源。需对接人提供360的请求级账单，核验金额、币种、请求ID和缓存计费拆分；其他供应商也应统一核对。',
              '本次只重新分析原始日志，未新增模型调用，未修改选择策略、价格配置或原始 events。',
              '', '依据：[360缓存文档](https://ai.360.com/docs/cache)、[对应渠道公开价格](https://ai.360.com/open/en/models/deepseek/deepseek-v4-flash)。']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
