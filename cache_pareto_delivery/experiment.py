"""Bounded real LiteLLM cache experiment; no Mos or shared-config modifications.

Run directly beside a gateway, or use remote_run(client, args) with an existing
authenticated Paramiko client. Remote Python code runs in memory, not deployed.
Only this experiment's local output directory is written by remote_run.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shlex
import statistics
import sys
import time

from routing import CacheAwareRouter, actual_cost, cached_tokens


def load_gateway(config_path, env_path):
    import os
    import yaml
    env = dict(os.environ)
    for line in Path(env_path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.removeprefix('export ').split('=', 1)
            parts = shlex.split(value, comments=True)
            env[key.strip()] = parts[0] if parts else ''
    def resolve(value):
        return env.get(value.split('/', 1)[1], '') if isinstance(value, str) and value.startswith('os.environ/') else value
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    key = resolve(config['general_settings']['master_key'])
    if not isinstance(key, str) or not key:
        raise ValueError('Gateway credential missing; do not send requests')
    deployments = []
    for row in config['model_list']:
        p = row['litellm_params']
        deployments.append(dict(endpoint=(p.get('tags') or [p.get('api_base')])[0],
            logical_model=row['model_name'], api_base=p['api_base'], upstream_model=p['model'],
            **{k: p.get(k) for k in ('input_cost_per_token', 'output_cost_per_token',
                 'cache_read_input_token_cost', 'extra_body', 'rpm', 'tpm', 'max_parallel_requests')}))
    return key, deployments, config


def document(run_id, family, size):
    """Synthetic, non-sensitive, fixed documents; lengths are measured by usage."""
    rng = random.Random(family.removesuffix('-pareto').removesuffix('-prefix_pareto'))
    count = 96 if size == 'short' else 384
    lines = [f'item_{i:04d}: region=R{rng.randrange(100):02d}; stock={rng.randrange(10000):04d}; '
             f'status=normal; audit=check_{rng.randrange(100000):05d}.' for i in range(count)]
    return [dict(role='system', content=f'Experiment {run_id}/{family}. Read the document. '
                 'On this first turn reply only READY. On later turns answer the requested field only. '
                 'Do not reason aloud.'), dict(role='user', content='\n'.join(lines))]


def followup(messages, warm_content, number):
    return messages + [dict(role='assistant', content=warm_content),
                       dict(role='user', content=f'Return only the stock value of item_{number:04d}.')]


def gateway_cost(row):
    """A recorded gateway charge, separate from independently verified tariffs."""
    try:
        value = float(row.get('gateway_headers', {}).get('x-litellm-response-cost'))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def summarize(events, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    calls = [e for e in events if e.get('type') == 'call']
    pilot = [e for e in calls if e['phase'] == 'pilot']
    groups = defaultdict(list)
    for row in pilot:
        groups[row['endpoint']].append(row)
    table = []
    for endpoint, rows in sorted(groups.items()):
        tests = [r for r in rows if r['role'] == 'followup']
        known = [r for r in tests if cached_tokens(r.get('usage')) is not None and r.get('valid')]
        savings = [r['accounting']['cache_saving'] for r in tests if r.get('valid')
                   and cached_tokens(r.get('usage')) is not None
                   and r.get('accounting', {}).get('cache_saving') is not None]
        table.append(dict(endpoint=endpoint, calls=len(rows), followups=len(tests),
            valid=sum(bool(r.get('valid')) for r in rows), cache_evidence_known=len(known),
            cache_hit_calls=sum(cached_tokens(r['usage']) > 0 for r in known),
            followup_cached_tokens=sum(cached_tokens(r['usage']) for r in known),
            priced_cache_saving=sum(savings) if savings else None))
    arms = {}
    for policy in ('pareto', 'prefix_pareto'):
        rows = [r for r in calls if r['phase'] == 'comparison' and r['policy'] == policy]
        tests = [r for r in rows if r['role'] == 'followup']
        complete = bool(rows) and all(r.get('valid') and r.get('accounting', {}).get('cache_adjusted_cost') is not None
                                      for r in rows)
        reported = [gateway_cost(r) for r in rows]
        arms[policy] = dict(calls=len(rows), followups=len(tests), valid=sum(bool(r.get('valid')) for r in rows),
            prefix_routes=sum((r.get('decision') or {}).get('reason') == 'prefix_affinity' for r in tests),
            cache_hit_calls=sum((cached_tokens(r.get('usage')) or 0) > 0 for r in tests if r.get('valid')),
            successful_followup_mean_ms=statistics.mean([r['e2e_ms'] for r in tests if r.get('valid')])
                                        if any(r.get('valid') for r in tests) else None,
            raw_slo_passes=sum(bool(r.get('valid')) and r['e2e_ms'] <= r['slo_ms'] for r in tests),
            raw_slo_denominator=len(tests),
            all_calls_cache_adjusted_cost=sum(r['accounting']['cache_adjusted_cost'] for r in rows) if complete else None,
            all_calls_full_input_reference=sum(r.get('accounting', {}).get('full_input_reference') or 0 for r in rows),
            gateway_reported_total_cost=sum(reported) if rows and all(v is not None for v in reported) else None,
            complete_cache_pricing=complete)
    environment = next((r for r in events if r.get('type') == 'environment'), {})
    matched = [r for r in calls if (r.get('decision') or {}).get('reason') == 'prefix_affinity']
    routing_times = [r['decision']['prefix_lookup_ns']/1e6 for r in calls if r.get('decision')]
    summary = dict(run_id=environment.get('run_id'), actual_call_attempts=len(calls),
        total_valid=sum(bool(r.get('valid')) for r in calls), pilot=table, comparison=arms,
        prefix_routed_calls=len(matched), prefix_routes_with_confirmed_hit=sum(
            bool(r.get('valid')) and (cached_tokens(r.get('usage')) or 0) > 0 for r in matched),
        mean_prefix_lookup_ms=statistics.mean(routing_times) if routing_times else None,
        finished=next((r for r in reversed(events) if r.get('type') == 'finished'), None),
        limitations=['Small feasibility pilot, not a statistical production-benefit guarantee.',
            'All endpoints assumed healthy and available; no capacity/failure/cooldown experiment.',
            'SLO is a frozen experimental budget, not a newly calibrated business commitment.',
            'Configured prices/currency not vendor invoices; missing cache tariff is unknown, not zero.',
            'Provider model versions differ; not a controlled model-quality comparison.',
            'Cache benefit within a call vs full-input billing is not the same as incremental routing benefit.'])
    a, b = arms['pareto']['all_calls_cache_adjusted_cost'], arms['prefix_pareto']['all_calls_cache_adjusted_cost']
    summary['routing_cost_reduction'] = (a-b)/a if a and b is not None else None
    a, b = arms['pareto']['gateway_reported_total_cost'], arms['prefix_pareto']['gateway_reported_total_cost']
    summary['gateway_accounted_cost_reduction'] = (a-b)/a if a and b is not None else None
    (directory/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 前缀缓存路由：独立小实验', '', f"- 实际调用：{len(calls)}；有效响应：{summary['total_valid']}。",
             '- 2026-10-20更正：360的hit/miss统计不自洽，保留未知；网关费用不等于供应商实际扣费。详见较大实验的费用核验章节。',
             '- 所有 Endpoint 健康可用；不修改 Mos、服务配置或原实验。',
             '- 前缀树匹配不等于供应商缓存命中；仅按实际 usage 与已知缓存价格核算。', '',
             '| Endpoint | 调用 | 有效 | 后续请求 | 可见缓存证据 | 命中次数 | 命中 tokens | 可核算缓存节省 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        saving = f"{r['priced_cache_saving']:.6f}" if r['priced_cache_saving'] is not None else '未知'
        lines.append(f"| {r['endpoint']} | {r['calls']} | {r['valid']} | {r['followups']} | "
                     f"{r['cache_evidence_known']} | {r['cache_hit_calls']} | {r['followup_cached_tokens']} | {saving} |")
    lines += ['', '## 策略小规模对照', '',
              '| 策略 | 含预热调用数 | 有效 | 后续请求达标 | 前缀优先次数 | 网关返回总费用 | 独立缓存计价总费用 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label, r in arms.items():
        cost = f"{r['all_calls_cache_adjusted_cost']:.6f}" if r['all_calls_cache_adjusted_cost'] is not None else '未知（缓存计价不完整）'
        reported = f"{r['gateway_reported_total_cost']:.6f}" if r['gateway_reported_total_cost'] is not None else '未知'
        lines.append(f"| {label} | {r['calls']} | {r['valid']} | {r['raw_slo_passes']}/{r['raw_slo_denominator']} | {r['prefix_routes']} | {reported} | {cost} |")
    gain = summary['routing_cost_reduction']
    lines += ['', f"- 路由增量成本变化：{gain:.2%} 降低。" if gain is not None else
              '- 缓存单价不完整，无法独立复核供应商实际费用差异。',
              f"- 网关返回费用口径：含两组各自的预热，前缀路由比原 Pareto 低 {summary['gateway_accounted_cost_reduction']:.2%}。"
                  if summary['gateway_accounted_cost_reduction'] is not None else '- 网关费用返回不完整，不比较费用。',
              '- 网关返回费用是已观测到的记账数，不等于供应商账单；不得把缺失缓存单价当作无折扣。',
              '- 两组使用独立但同长度/同任务的文档；预热计入费用；不人为修改价格、时延或挑选胜出响应。',
              '- 各模型版本、输入/输出量、完整调用与费用明细见 events.jsonl。', '', '## 限制', '']
    lines += [
        '- 仅为小规模可行性验证，不能外推长期收益或声称统计显著。',
        '- 全部端点假设健康且可用，不验证并发、RPM/TPM、错误和 cooldown。',
        '- 本轮采用预先固定的 60 秒实验 SLO，不是重新标定的业务承诺；它只用于确认策略不会绕过门槛。',
        '- 当前模型版本不同；本轮不比较模型质量，也没有人为更改价格、延迟。',
        '- 缓存相对同端点全价输入的折扣，与路由相对其他策略的增量收益，是两个不同指标。',
        '- 首轮估计来自少量预热观测、输入长度是粗估，不代表生产性能预测模型。',
        '- tree 匹配但未真正命中缓存的请求保留在结果中，不能承诺前缀匹配必然命中。']
    (directory/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


def run(args, emit):
    import httpx
    if (not 1 <= args.max_calls <= 260 or not math.isfinite(args.max_cost) or not 0 < args.max_cost <= 4
            or not math.isfinite(args.slo_ms) or args.slo_ms <= 0
            or not 1 <= args.timeout <= 120 or not .1 <= args.spacing <= 10 or not 0 <= args.cache_wait <= 30):
        raise ValueError('Invalid experiment bounds')
    key, deployments, config = load_gateway(args.config, args.env_file)
    deployments = [d for d in deployments if d['logical_model'] == args.model]
    if not deployments or len({d['endpoint'] for d in deployments}) != len(deployments):
        raise ValueError('Unique configured deployments for the chosen model required')
    if any((d.get('extra_body') or {}).get('thinking', {}).get('type') != 'disabled' for d in deployments):
        raise ValueError('Thinking must already be disabled; shared configuration will not be changed')
    if any(not isinstance(d.get(k), (float, int)) or d[k] < 0 for d in deployments
           for k in ('input_cost_per_token', 'output_cost_per_token')):
        raise ValueError('Configured input/output prices required for bounded calls')
    run_id = datetime.now(timezone.utc).strftime('cache_%Y%m%dT%H%M%SZ')
    before = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    by_id = {d['endpoint']: d for d in deployments}
    events, spent, count = [], 0., 0
    started = time.monotonic()
    now = lambda: round((time.monotonic()-started)*1000)
    def event(row):
        events.append(row); emit(row)
    event(dict(type='environment', run_id=run_id, deployments=deployments, config_sha256=before,
               max_calls=args.max_calls, max_known_cost=args.max_cost, pilot_slo_ms=args.slo_ms,
               slo_source='explicit_frozen_experimental_budget_not_business_SLO', all_endpoints_assumed_available=True))
    with httpx.Client(base_url=args.gateway, headers={'Authorization': 'Bearer '+key},
                      timeout=httpx.Timeout(args.timeout, connect=10), trust_env=False) as client:
        info = client.get('/model/info'); info.raise_for_status()
        for d in deployments:
            matching = [r for r in info.json().get('data', [])
                        if r.get('litellm_params', {}).get('api_base') == d['api_base']]
            if len(matching) != 1:
                raise ValueError('Fixed deployment identity is ambiguous; no model calls')
            d['deployment_id'] = matching[0].get('model_info', {}).get('id')
        # Read-only startup check; no budget updates or restarts.
        spend = client.get('/global/spend'); spend.raise_for_status()
        event(dict(type='budget', values=spend.json()))
        budget = spend.json()
        if isinstance(budget, dict) and budget.get('max_budget') is not None:
            if float(budget['max_budget'])-float(budget.get('spend', 0)) < args.max_cost:
                raise ValueError('Remaining shared budget below experiment cap; no budget changes allowed')
        router = CacheAwareRouter()

        def call(endpoint, messages, phase, role, family, size, policy, decision=None):
            nonlocal spent, count
            if count >= args.max_calls or spent >= args.max_cost:
                raise RuntimeError('Experiment call/cost limit reached')
            d = by_id[endpoint]
            # UTF-8 bytes are an intentionally loose upper token bound. Cost cap
            # is in configured units, includes unknown-cache calls at full price.
            upper = sum(len(json.dumps(m, ensure_ascii=False).encode('utf-8'))+256 for m in messages)+1024
            reservation = upper*d['input_cost_per_token'] + 64*d['output_cost_per_token']
            if spent + reservation > args.max_cost:
                raise RuntimeError('Conservative pre-call cost bound would exceed cap')
            trial_id = f'{run_id}-{count:03d}'
            payload = dict(model=args.model, messages=messages, stream=False, max_tokens=64,
                           temperature=0, num_retries=0, **({'extra_body': d['extra_body']} if d.get('extra_body') else {}),
                           cache={'no-cache': True}, metadata=dict(endpoint_priority=[d['api_base']],
                              session_id=run_id, experiment_trial_id=trial_id, experiment_kind='prefix_cache_pilot'))
            row = dict(type='call', trial_id=trial_id, phase=phase, role=role, family=family, size=size,
                       policy=policy, endpoint=endpoint, messages=messages,
                       slo_ms=decision['slo_ms'] if decision else args.slo_ms,
                       decision=decision, started_at_utc=datetime.now(timezone.utc).isoformat())
            count += 1
            clock = time.perf_counter()
            try:
                response = client.post('/v1/chat/completions', json=payload)
                row['e2e_ms'] = (time.perf_counter()-clock)*1000
                row['http_status'] = response.status_code
                data = response.json()
                headers = {k: response.headers[k] for k in ('x-litellm-model-api-base', 'x-litellm-model-id',
                    'x-litellm-cache-hit', 'x-litellm-attempted-retries', 'x-litellm-response-cost',
                    'x-litellm-call-id') if k in response.headers}
                row['gateway_headers'] = headers
                actual_base, actual_id = headers.get('x-litellm-model-api-base'), headers.get('x-litellm-model-id')
                verified = ((actual_base == d['api_base'] or actual_id == d['deployment_id'])
                            and (actual_base is None or actual_base == d['api_base'])
                            and (actual_id is None or actual_id == d['deployment_id']))
                response_cached = str(headers.get('x-litellm-cache-hit', 'false')).lower() == 'true'
                choice = (data.get('choices') or [{}])[0]
                content = (choice.get('message') or {}).get('content')
                row.update(transport_success=response.is_success, endpoint_identity_verified=verified,
                    response_cache_hit=response_cached, response_id=data.get('id'), usage=data.get('usage'),
                    content=content, finish_reason=choice.get('finish_reason'))
                # Full-response cache must not masquerade as provider prefix cache.
                content_ok = isinstance(content, str) and bool(content.strip())
                if content_ok and role in ('warmup', 'negative_prefix'):
                    content_ok = content.strip().upper() == 'READY'
                elif content_ok and role == 'followup':
                    item = re.search(r'item_(\d+)', messages[-1]['content'])
                    expected = re.search(rf"item_{item[1]}: region=R\d+; stock=(\d+);", messages[1]['content']) if item else None
                    content_ok = bool(expected and content.strip().isdigit() and int(content.strip()) == int(expected[1]))
                row['content_correct'] = content_ok
                row['valid'] = bool(response.is_success and verified and not response_cached and content_ok
                                    and choice.get('finish_reason') == 'stop')
                row['cached_tokens'] = cached_tokens(row['usage'])
                row['accounting'] = actual_cost(row['usage'], d)
            except Exception as exc:
                row.update(valid=False, transport_success=False, error_type=type(exc).__name__,
                           e2e_ms=(time.perf_counter()-clock)*1000)
            row['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
            accounting = row.get('accounting', {})
            # Invalid/timeout/unknown responses reserve the entire preflight bound:
            # they may still have incurred a charge and must never count as free.
            spent += (accounting.get('cache_adjusted_cost') if accounting.get('cache_adjusted_cost') is not None
                      else accounting.get('full_input_reference') if accounting.get('full_input_reference') is not None
                      else reservation)
            row['conservative_running_cost'] = spent
            event(row)
            router.feedback(scope=family, model=args.model, messages=messages, endpoint=endpoint,
                now_ms=now(), success=row['valid'], identity_verified=row.get('endpoint_identity_verified', False))
            if row.get('response_cache_hit'):
                raise RuntimeError('Gateway response cache observed; stop before confusing it with provider cache')
            time.sleep(args.spacing)
            return row

        try:
            if getattr(args, 'suite', 'pilot') == 'supplier-cost':
                from supplier_experiment import run_supplier
                run_supplier(args, deployments, router, call, event, now, run_id)
                return events
            if getattr(args, 'suite', 'pilot') == 'billing-probe':
                from billing_probe import run_probe
                run_probe(args, deployments, config, call, event, now, run_id)
                return events
            if getattr(args, 'suite', 'pilot') == 'validation':
                from validation import run_stages
                run_stages(args, deployments, router, call, event, now, run_id)
                return events
            for d in deployments:
                for size in ('short', 'long'):
                    family = f"pilot-{d['endpoint']}-{size}"
                    messages = document(run_id, family, size)
                    warm = call(d['endpoint'], messages, 'pilot', 'warmup', family, size, 'fixed_endpoint')
                    if not warm['valid']:
                        event(dict(type='group_stopped', family=family, reason='invalid_warmup_no_fake_response'))
                        continue
                    time.sleep(args.cache_wait)
                    for n in (1, 2):
                        query = followup(messages, warm['content'], n)
                        # This phase isolates cache evidence, not competitive ranking.
                        candidates = [dict(endpoint_id=d['endpoint'], estimated_e2e_ms=warm['e2e_ms'], estimated_cost=1.)]
                        decision = router.choose(scope=family, model=args.model, messages=query,
                            candidates=candidates, slo_ms=args.slo_ms, now_ms=now())
                        if decision['selected_endpoint'] is None:
                            event(dict(type='unserved', family=family, decision=decision)); continue
                        call(d['endpoint'], query, 'pilot', 'followup', family, size, 'prefix_pareto', decision)
                family = f"negative-{d['endpoint']}"
                call(d['endpoint'], document(run_id, family, 'short'), 'pilot', 'negative_prefix', family, 'short', 'fixed_endpoint')

            # Only expand on endpoints with BOTH observed cache hits and an explicit
            # cache tariff. Others stay in Pareto, but cannot support monetary claims.
            validated = [d for d in deployments if d['cache_read_input_token_cost'] is not None and any(
                r.get('valid') and r.get('cached_tokens', 0) and r['endpoint'] == d['endpoint']
                and r.get('role') == 'followup' for r in events if r.get('type') == 'call')]
            event(dict(type='comparison_readiness', validated_endpoints=[d['endpoint'] for d in validated]))
            pilot = [r for r in events if r.get('type') == 'call' and r.get('valid') and r['phase'] == 'pilot']
            # Freeze small-sample performance observations BEFORE the comparison.
            estimates = {(d['endpoint'], size): statistics.mean([r['e2e_ms'] for r in pilot
                if r['endpoint'] == d['endpoint'] and r['size'] == size]) for d in deployments for size in ('short','long')
                if any(r['endpoint'] == d['endpoint'] and r['size'] == size for r in pilot)}
            event(dict(type='frozen_estimates', rows=[dict(endpoint=e, size=s, estimated_e2e_ms=v)
                                                     for (e,s),v in estimates.items()], estimator='pilot_arithmetic_mean_sparse'))
            # Two paired families per priced/cache-validated endpoint. Arm order
            # alternates. Independent prefixes prevent one arm warming the other.
            for d in validated:
                for pair, size in enumerate(('short', 'long')):
                    policies = ('pareto','prefix_pareto') if pair % 2 == 0 else ('prefix_pareto','pareto')
                    families = {}
                    for policy in policies:
                        family = f"pair-{d['endpoint']}-{pair}-{policy}"
                        messages = document(run_id, family, size)
                        warm = call(d['endpoint'], messages, 'comparison', 'warmup', family, size, policy)
                        if warm['valid']:
                            families[policy] = (family, messages, warm['content'])
                    time.sleep(args.cache_wait)
                    for n in (1, 2):
                        for policy in policies:
                            if policy not in families:
                                continue
                            family, messages, content = families[policy]
                            query = followup(messages, content, n)
                            predicted_input = max(1, len(json.dumps(query, ensure_ascii=False))//3)
                            candidates = [dict(endpoint_id=p['endpoint'],
                                estimated_e2e_ms=estimates.get((p['endpoint'], size)),
                                estimated_cost=predicted_input*p['input_cost_per_token']+16*p['output_cost_per_token'])
                                for p in deployments]
                            decision = router.choose(scope=family, model=args.model, messages=query,
                                candidates=candidates, slo_ms=args.slo_ms, now_ms=now(), prefix_first=policy=='prefix_pareto')
                            if decision['selected_endpoint'] is None:
                                event(dict(type='unserved', family=family, decision=decision)); continue
                            call(decision['selected_endpoint'], query, 'comparison', 'followup', family, size, policy, decision)
        except RuntimeError as exc:
            event(dict(type='bounded_stop', reason=str(exc)))
        finally:
            if getattr(args, 'suite', 'pilot') == 'billing-probe':
                count = sum(row.get('type')=='call' for row in events)
                direct = [row for row in events if row.get('type')=='call' and row.get('probe_path')=='direct']
                if direct:
                    spent = direct[-1]['combined_conservative_running_cost']
            after = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
            event(dict(type='finished', calls=count, conservative_cost=spent, config_unchanged=before==after,
                       duration_s=time.monotonic()-started))
    return events


def remote_run(client, args):
    """No upload or remote file writes; stream all evidence to ONE local run dir."""
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    options = {k: v for k, v in vars(args).items() if k not in ('output', 'ssh', 'ssh_password_env')}
    routing_source = Path(__file__).with_name('routing.py').read_text(encoding='utf-8')
    experiment_source = Path(__file__).read_text(encoding='utf-8')
    validation_source = (Path(__file__).with_name('validation.py').read_text(encoding='utf-8')
                         if getattr(args, 'suite', 'pilot') == 'validation' else None)
    probe_source = (Path(__file__).with_name('billing_probe.py').read_text(encoding='utf-8')
                    if getattr(args, 'suite', 'pilot') == 'billing-probe' else None)
    supplier_source = (Path(__file__).with_name('supplier_experiment.py').read_text(encoding='utf-8')
                       if getattr(args, 'suite', 'pilot') == 'supplier-cost' else None)
    extra_module = ("v=types.ModuleType('validation');sys.modules['validation']=v\n"
                    f"exec(compile({validation_source!r},'validation.py','exec'),v.__dict__)\n") if validation_source else ''
    if probe_source:
        extra_module += ("b=types.ModuleType('billing_probe');sys.modules['billing_probe']=b\n"
                         f"exec(compile({probe_source!r},'billing_probe.py','exec'),b.__dict__)\n")
    if supplier_source:
        extra_module += ("s=types.ModuleType('supplier_experiment');sys.modules['supplier_experiment']=s\n"
                         f"exec(compile({supplier_source!r},'supplier_experiment.py','exec'),s.__dict__)\n")
    bootstrap = (
        'import sys,types,json,argparse\n'
        "m=types.ModuleType('routing');sys.modules['routing']=m\n"
        f"exec(compile({routing_source!r},'routing.py','exec'),m.__dict__)\n"
        "e=types.ModuleType('experiment');sys.modules['experiment']=e\n"
        f"exec(compile({experiment_source!r},'experiment.py','exec'),e.__dict__)\n"
        + extra_module + f"a=argparse.Namespace(**json.loads({json.dumps(options)!r}))\n"
        "e.run(a,lambda row:print(json.dumps(row,ensure_ascii=False),flush=True))\n")
    sources = (('routing.py','experiment.py') + (('validation.py',) if validation_source else ())
               + (('billing_probe.py',) if probe_source else ()) + (('supplier_experiment.py',) if supplier_source else ()))
    (directory/'protocol.json').write_text(json.dumps(dict(options=options,
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in sources},
        remote_execution='in_memory_no_service_or_remote_files_modified'), indent=2), encoding='utf-8')
    import zipfile
    with zipfile.ZipFile(directory/'executed_source.zip', 'w', zipfile.ZIP_DEFLATED) as frozen:
        for name in sources:
            frozen.write(Path(__file__).with_name(name), name)
    stdin, stdout, stderr = client.exec_command('/home/ubuntu/litellm-real/.venv/bin/python -u -', timeout=900)
    stdin.write(bootstrap); stdin.channel.shutdown_write()
    events = []
    with (directory/'events.jsonl').open('w', encoding='utf-8') as output:
        for line in stdout:
            row = json.loads(line)
            output.write(json.dumps(row, ensure_ascii=False)+'\n'); output.flush(); events.append(row)
            if row.get('type') == 'call':
                print(json.dumps({k:row.get(k) for k in ('trial_id','endpoint','phase','role','valid','cached_tokens','e2e_ms')}) ,flush=True)
            else:
                print(json.dumps({k:v for k,v in row.items() if k not in ('deployments','rows')}) ,flush=True)
    code = stdout.channel.recv_exit_status()
    # Never expose raw remote errors: a dependency could include a credential.
    if code:
        print(f'Remote runner exited {code}; raw stderr withheld; saved partial evidence.')
    if supplier_source:
        from supplier_experiment import report
        summary = report(events, directory)
    elif probe_source:
        from billing_probe import report
        summary = report(events, directory)
    elif validation_source:
        from validation import report
        summary = report(events, directory)
    else:
        summary = summarize(events, directory)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='/home/ubuntu/litellm-real/config_custom.yaml')
    p.add_argument('--env-file', default='/home/ubuntu/litellm-real/real_keys.env')
    p.add_argument('--gateway', default='http://127.0.0.1:4000')
    p.add_argument('--model', default='deepseek-flash')
    p.add_argument('--suite', choices=('pilot', 'validation', 'billing-probe', 'supplier-cost'), default='pilot')
    p.add_argument('--pairs-per-cell', type=int, default=3)
    p.add_argument('--followups', type=int, default=3)
    p.add_argument('--max-calls', type=int, default=40)
    p.add_argument('--max-cost', type=float, default=.8)
    p.add_argument('--slo-ms', type=float, default=60_000)
    p.add_argument('--timeout', type=float, default=60)
    p.add_argument('--spacing', type=float, default=1)
    p.add_argument('--cache-wait', type=float, default=3)
    p.add_argument('--output', default=str(Path(__file__).parent/'output'/datetime.now().strftime('%Y%m%d-%H%M%S')))
    p.add_argument('--ssh', help='user@host; otherwise run beside the gateway')
    p.add_argument('--ssh-password-env', default='CACHE_LAB_SSH_PASSWORD')
    return p


def main():
    import os
    args = parser().parse_args()
    if not 1 <= args.max_calls <= 260 or not 0 < args.max_cost <= 4 or args.slo_ms <= 0:
        raise ValueError('Use bounded calls [1,260], configured cost (0,4] and a positive SLO')
    if args.ssh:
        import getpass
        import paramiko
        user, host = args.ssh.split('@', 1)
        client = paramiko.SSHClient(); client.load_system_host_keys()
        # No AutoAddPolicy: an unknown server must be verified outside this script.
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(host, username=user, password=os.environ.get(args.ssh_password_env) or getpass.getpass('SSH password: '),
                       timeout=15, allow_agent=True, look_for_keys=True)
        try:
            remote_run(client, args)
        finally:
            client.close()
    else:
        directory = Path(args.output); directory.mkdir(parents=True, exist_ok=False)
        with (directory/'events.jsonl').open('w', encoding='utf-8') as output:
            def emit(row):
                output.write(json.dumps(row, ensure_ascii=False)+'\n'); output.flush()
            events = run(args, emit)
        if args.suite == 'supplier-cost':
            from supplier_experiment import report
            report(events, directory)
        elif args.suite == 'billing-probe':
            from billing_probe import report
            report(events, directory)
        elif args.suite == 'validation':
            from validation import report
            report(events, directory)
        else:
            summarize(events, directory)


if __name__ == '__main__':
    main()
