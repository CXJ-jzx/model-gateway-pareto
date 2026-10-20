"""Supplier procurement-cost estimate experiment, never customer-facing charges.

Primary pool is frozen BEFORE calls to the two providers with an interpretable
fee basis. 360 reported amounts remain a fee proxy, not an invoice assertion.
No cache-hit field is required to account for an explicit supplier amount.
"""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import time

from experiment import document, followup
from routing import cached_tokens


def quote(endpoint,stamp):
    if endpoint=='official':
        day = datetime.fromisoformat(stamp.replace('Z','+00:00')).astimezone(timezone(timedelta(hours=8)))
        # This dated experiment is not a general Chinese public-holiday calendar.
        if day.date().isoformat()!='2026-10-20':
            raise ValueError('Official peak calendar must be revalidated for a different experiment day')
        hour = day.hour+day.minute/60
        peak = day.weekday()<5 and (9<=hour<12 or 14<=hour<18)
        return dict(input=2. if peak else 1.,output=8. if peak else 4.,cache=.04 if peak else .02,
                    currency='CNY',source='official_public_procurement_tariff',period='peak' if peak else 'offpeak')
    if endpoint=='ai360-cdtx':
        return dict(input=1.8,output=5.4,cache=.06,currency='CNY',
                    source='360_tokenmall_cdtx_public_procurement_tariff')
    raise ValueError('Provider fee basis not established for this main experiment')


def supplier_fee(row):
    """No LiteLLM cost-header fallback. Unknown procurement fees stay unknown."""
    usage = row.get('usage') or {}
    inp,out = usage.get('prompt_tokens'),usage.get('completion_tokens')
    q = quote(row['endpoint'],row['started_at_utc'])
    reference = ((inp*q['input']+out*q['output'])/1e6
                 if type(inp) is int and inp>=0 and type(out) is int and out>=0 else None)
    if row['endpoint']=='ai360-cdtx':
        value = usage.get('amount')
        try:
            value = float(value) if not isinstance(value,bool) else None
        except (ValueError,TypeError):
            value = None
        known = value is not None and math.isfinite(value) and value>=0
        return dict(estimated_procurement_cost_cny=value if known else None,
                    source='supplier_reported_usage_amount',full_input_quote_reference=reference,
                    confidence='CNY inferred from repeated cold-price agreement; invoice/unit not authenticated',
                    cache_hit_required=False)
    hit = cached_tokens(usage)
    cost = ((inp-hit)*q['input']+hit*q['cache']+out*q['output'])/1e6 if reference is not None and hit is not None else None
    return dict(estimated_procurement_cost_cny=cost,source='actual_usage_times_public_supplier_tariff',
                full_input_quote_reference=reference,confidence='public list price, not account invoice',
                cache_hit_required=True,tariff=q)


def run_supplier(args,deployments,router,call,event,now,run_id):
    endpoints = ['official','ai360-cdtx']
    by_id = {d['endpoint']:d for d in deployments}
    if not set(endpoints)<=set(by_id):
        raise RuntimeError('Both pre-registered suppliers required')
    if args.max_calls<100 or args.max_cost>1.2 or args.pairs_per_cell!=3 or args.followups!=3 or args.slo_ms!=5000:
        raise ValueError('Frozen protocol: 100 calls, cap<=1.2, 3 pairs/cell, 3 turns, SLO=5000ms')
    stamp = datetime.now(timezone.utc).isoformat()
    quotes = {e:quote(e,stamp) for e in endpoints}
    event(dict(type='supplier_protocol',target='gateway procurement costs, NOT customer fees',
        main_pool=endpoints,excluded_pool=[e for e in by_id if e not in endpoints],
        exclusion_reason='relay/volc cache procurement fees not verified; no mixed customer cost headers',
        alpha=.8,slo_ms=5000,pairs=12,followups_per_policy=36,planned_calls=100,
        quotes=quotes,quote_time=stamp,
        cost_confidence='supplier-side reported/quoted cost estimate, not independently authenticated invoices',
        sources=['https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
                 'https://ai.360.com/open/en/models/deepseek/deepseek-v4-flash'],
        response_amount_currency='360 CNY is an explicitly stated working inference, not verified account metadata'))
    estimates = {}
    for endpoint in endpoints:
        for size in ('short','long'):
            family = f'supplier-calibration-{endpoint}-{size}'
            row = call(endpoint,document(run_id,family,size),'supplier_calibration','warmup',family,size,'fixed_endpoint')
            if not row.get('valid'):
                raise RuntimeError('Calibration failed; no invented endpoint estimate')
            estimates[endpoint,size] = row['e2e_ms']
    event(dict(type='supplier_estimates',estimator='one pre-test observation, sparse evidence',
        values=[dict(endpoint=e,size=s,e2e_ms=v) for (e,s),v in estimates.items()]))
    plan = [(e,size,r) for e in endpoints for size in ('short','long') for r in range(3)]
    random.Random(20261020).shuffle(plan)
    event(dict(type='supplier_plan',pairs=[dict(pair_id=f'supplier-{i:02d}',warm_endpoint=e,size=s,repeat=r)
                                         for i,(e,s,r) in enumerate(plan)]))
    for i,(warm_endpoint,size,repeat) in enumerate(plan):
        pair_id = f'supplier-{i:02d}'
        order = ('pareto','prefix_pareto') if i%2==0 else ('prefix_pareto','pareto')
        families = {}
        for policy in order:
            family = pair_id+'-'+policy
            messages = document(run_id,family,size)
            warm = call(warm_endpoint,messages,'supplier_comparison','warmup',family,size,policy)
            if warm.get('valid'):
                families[policy] = (family,messages,warm['content'])
            else:
                event(dict(type='supplier_unserved',pair_id=pair_id,policy=policy,turns=3,reason='invalid_warmup'))
        time.sleep(args.cache_wait)
        for turn in (1,2,3):
            for policy in order:
                if policy not in families:
                    continue
                family,messages,content = families[policy]
                query = followup(messages,content,turn)
                inp = max(1,len(json.dumps(query,ensure_ascii=False))//3)
                # No actual outcome/amount for THIS request is used in its choice.
                candidates = [dict(endpoint_id=e,estimated_e2e_ms=estimates[e,size],
                    estimated_cost=(inp*quotes[e]['input']+2*quotes[e]['output'])/1e6) for e in endpoints]
                decision = router.choose(scope=family,model=args.model,messages=query,candidates=candidates,
                    slo_ms=args.slo_ms,now_ms=now(),alpha=.8,prefix_first=policy=='prefix_pareto')
                if decision['selected_endpoint'] is None:
                    event(dict(type='supplier_unserved',pair_id=pair_id,policy=policy,turns=1,reason='no_slo_feasible'))
                    continue
                call(decision['selected_endpoint'],query,'supplier_comparison','followup',family,size,policy,decision)
        event(dict(type='supplier_pair_finished',pair_id=pair_id,warm_endpoint=warm_endpoint,size=size))


def report(events,directory):
    rows = [r for r in events if r.get('type')=='call']
    comparisons = [r for r in rows if r['phase']=='supplier_comparison']
    protocol = next((r for r in events if r.get('type')=='supplier_protocol'),{})
    plan = next((r['pairs'] for r in events if r.get('type')=='supplier_plan'),[])
    per_request = [dict(trial_id=r['trial_id'],family=r['family'],policy=r['policy'],endpoint=r['endpoint'],
                        role=r['role'],valid=r.get('valid'),**supplier_fee(r)) for r in comparisons]
    fee_by_id = {r['trial_id']:r['estimated_procurement_cost_cny'] for r in per_request}
    def aggregate(calls,denominator):
        tests = [r for r in calls if r['role']=='followup']
        costs = [fee_by_id[r['trial_id']] for r in calls]
        warmups = [r for r in calls if r['role']=='warmup']
        warm_costs = [fee_by_id[r['trial_id']] for r in warmups]
        return dict(calls=len(calls),planned_followups=denominator,followups=len(tests),
            correct=sum(bool(r.get('valid')) for r in tests),
            latency_slo_passes=sum(bool(r.get('transport_success')) and r.get('endpoint_identity_verified')
                and r.get('finish_reason')=='stop' and r['e2e_ms']<=5000 for r in tests),
            endpoint_choices=dict(Counter(r['endpoint'] for r in tests)),
            estimated_supplier_total_cny=sum(costs) if costs and all(x is not None for x in costs) else None,
            warmup_cost_cny=sum(warm_costs) if warm_costs and all(x is not None for x in warm_costs) else None,
            unknown_fee_calls=sum(x is None for x in costs),
            mean_success_ms=sum(r['e2e_ms'] for r in tests if r.get('transport_success'))/
                max(1,sum(bool(r.get('transport_success')) for r in tests)))
    arms = {p:aggregate([r for r in comparisons if r['policy']==p],36) for p in ('pareto','prefix_pareto')}
    groups = []
    for endpoint in protocol.get('main_pool',[]):
        ids = {p['pair_id'] for p in plan if p['warm_endpoint']==endpoint}
        group = {policy:aggregate([r for r in comparisons if r['policy']==policy and
            r['family'].removesuffix('-pareto').removesuffix('-prefix_pareto') in ids],18)
            for policy in ('pareto','prefix_pareto')}
        a,b = [group[p]['estimated_supplier_total_cny'] for p in ('pareto','prefix_pareto')]
        groups.append(dict(warm_endpoint=endpoint,arms=group,reduction=1-b/a if a and b is not None else None))
    a,b = [arms[p]['estimated_supplier_total_cny'] for p in ('pareto','prefix_pareto')]
    calibration_fees = [supplier_fee(r)['estimated_procurement_cost_cny'] for r in rows if r['phase']=='supplier_calibration']
    calibration_cost = sum(calibration_fees) if calibration_fees and all(x is not None for x in calibration_fees) else None
    allocated = {p:arms[p]['estimated_supplier_total_cny']+calibration_cost/2
                 if calibration_cost is not None and arms[p]['estimated_supplier_total_cny'] is not None else None
                 for p in ('pareto','prefix_pareto')}
    cold360 = [r for r in comparisons if r['endpoint']=='ai360-cdtx' and r['role']=='warmup']
    agreement = [abs(supplier_fee(r)['estimated_procurement_cost_cny']-supplier_fee(r)['full_input_quote_reference'])
                 for r in cold360 if supplier_fee(r)['estimated_procurement_cost_cny'] is not None]
    summary = dict(calls=len(rows),valid=sum(bool(r.get('valid')) for r in rows),
        completed_pairs=sum(r.get('type')=='supplier_pair_finished' for r in events),protocol=protocol,
        arms=arms,estimated_procurement_cost_reduction=1-b/a if a and b is not None else None,
        shared_calibration_cost_cny=calibration_cost,totals_with_shared_calibration_allocated=allocated,
        reduction_with_shared_calibration=1-allocated['prefix_pareto']/allocated['pareto']
            if allocated['pareto'] and allocated['prefix_pareto'] is not None else None,
        formal_invoice_reduction=None,by_warm_endpoint=groups,per_request_costs=per_request,
        cold360_amount_quote_agreement=dict(calls=len(cold360),comparable=len(agreement),
                                           max_absolute_difference=max(agreement) if agreement else None),
        finished=next((r for r in reversed(events) if r.get('type')=='finished'),None),
        label='supplier-reported/public-tariff CNY procurement ESTIMATE, not customer fee or authenticated invoice')
    fmt = lambda x:'未知' if x is None else f'{x:.6f}'
    change = lambda x:'未知' if x is None else (f'降低{x:.2%}' if x>=0 else f'增加{-x:.2%}')
    lines = ['# 供应商采购成本视角：真实缓存路由对照', '',
        '**本报告不使用LiteLLM的response-cost作为采购成本。360使用供应商返回amount；官方使用实际usage乘公开采购价。360的CNY单位由冷请求与公开价一致性支持，但未认证到账单。**', '',
        f"真实调用{summary['calls']}次，正确有效{summary['valid']}次，完成{summary['completed_pairs']}/12组配对文档。", '',
        '| 指标 | 原Pareto | 前缀优先＋Pareto |','|---|---:|---:|']
    for label,key in [('含预热采购成本估算','estimated_supplier_total_cny'),('其中预热费用','warmup_cost_cny'),('后续平均耗时ms','mean_success_ms')]:
        lines.append(f"| {label} | {fmt(arms['pareto'][key])} | {fmt(arms['prefix_pareto'][key])} |")
    for label,key in [('后续回答正确','correct'),('5秒纯时延SLO达标','latency_slo_passes')]:
        lines.append(f"| {label} | {arms['pareto'][key]}/36 | {arms['prefix_pareto'][key]}/36 |")
    gain = summary['estimated_procurement_cost_reduction']
    lines += ['', f'采购成本估算{change(gain)}。' if gain is not None else '费用不完整，不能计算总收益。', '',
        '| 历史缓存位置 | 原Pareto估算 | 前缀优先估算 | 变化 |', '|---|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['warm_endpoint']} | {fmt(g['arms']['pareto']['estimated_supplier_total_cny'])} | "
            f"{fmt(g['arms']['prefix_pareto']['estimated_supplier_total_cny'])} | "+
            change(g['reduction'])+' |')
    lines += ['', f"4次公共校准费用估算合计{fmt(calibration_cost)}，单独列出、不重复算进两组预热。若各分摊一半，总费用为 "
              f"{fmt(allocated['pareto'])} 对 {fmt(allocated['prefix_pareto'])}，"
              f"{change(summary['reduction_with_shared_calibration'])}。"]
    lines += ['', '## 设计与费用依据', '',
        '- 12组独立文档：官方/360历史位置均衡轮换，各自包含短/长输入、每格3组。每组两种策略各预热1次、后续3次，预热均计费。',
        '- 两组分别使用独立前缀标识，正文相同；顺序交替；温度0、关闭思考、输出上限64。真实输出、错误和费用均保留。',
        '- 沿用原前缀树与前缀优先规则。α=0.8、SLO=5秒在调用前固定，不按结果调整。SLO是实验约束，不代表业务校准承诺。',
        '- 用测试前4次真实观测冻结耗时估计，证据较少；本次实际amount不用于本次选择，避免后验作弊。',
        '- 主池只包含官方与360：另外两家的缓存采购计费依据不完整，事前排除，不能外推本报告到四家组合。',
        '- 360没有有效cache_hit字段也可记账，只要amount存在。缺失amount保留未知，不退回LiteLLM记账费用。',
        '- 正式账单尚未核验，结果称采购成本估算，不能宣称已经确认财务实扣；供应商版本不同，不比较模型质量。',
        '- 收益限于每份已有上下文继续3条请求的场景。复用避免迁移后的首次冷输入，但目标端点后续也会缓存；更长会话不保证仍然划算。',
        '- 两组正文相同，但独立标识存在少量tokens差异；按各自真实usage收费，不事后修平费用。',
        '- 不修改Mos、共享服务配置、原实验或端点价格，不提交仓库。', '',
        '依据：[官方采购价](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、[360渠道价格](https://ai.360.com/open/en/models/deepseek/deepseek-v4-flash)。']
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
