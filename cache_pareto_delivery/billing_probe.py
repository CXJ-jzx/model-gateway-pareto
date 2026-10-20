"""At most 16 real 360 calls to distinguish provider fields from LiteLLM accounting.

No login automation, undocumented billing-API guessing, shared config changes, or
currency assumptions. Direct and gateway inputs use independent prefix markers.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import time

from experiment import document, followup, gateway_cost
from routing import cached_tokens


def provider_key(config, deployment, env_path):
    rows = [r for r in config['model_list'] if r['litellm_params'].get('api_base')==deployment['api_base']
            and r['model_name']==deployment['logical_model']]
    if len(rows)!=1:
        raise ValueError('Ambiguous provider credentials; no direct calls')
    value = rows[0]['litellm_params'].get('api_key')
    if isinstance(value,str) and value.startswith('os.environ/'):
        name = value.split('/',1)[1]
        values = dict(os.environ)
        for line in Path(env_path).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k,v = line.removeprefix('export ').split('=',1)
                pieces = shlex.split(v,comments=True)
                values[k.strip()] = pieces[0] if pieces else ''
        value = values.get(name)
    if not isinstance(value,str) or not value:
        raise ValueError('Provider credential unavailable')
    return value


def parse_amount(usage):
    if not isinstance(usage,dict) or isinstance(usage.get('amount'),bool):
        return None
    try:
        value = float(usage.get('amount'))
    except (ValueError,TypeError):
        return None
    return value if math.isfinite(value) and value>=0 else None


def content_correct(messages,content,role):
    if not isinstance(content,str):
        return False
    if role in ('warmup','negative_prefix'):
        return content.strip().upper()=='READY'
    item = re.search(r'item_(\d+)',messages[-1]['content'])
    expected = re.search(rf"item_{item[1]}: region=R\d+; stock=(\d+);",messages[1]['content']) if item else None
    return bool(expected and content.strip().isdigit() and int(content.strip())==int(expected[1]))


def run_probe(args,deployments,config,call,event,now,run_id):
    import httpx
    ds = [d for d in deployments if d['endpoint']=='ai360-cdtx']
    if len(ds)!=1 or ds[0]['api_base'].rstrip('/')!='https://ai.360.com/api/v1':
        raise ValueError('Probe restricted to the previously authorized 360 API base')
    if args.max_calls<16 or args.max_cost>.35:
        raise ValueError('Probe needs max_calls>=16 and a cost cap <=0.35')
    d = ds[0]
    event(dict(type='probe_plan',calls=16,paths=['gateway','direct'],
        sizes=['short','long'],per_cell='new prefix, followup1, followup2, negative new prefix',
        direct_calls_max=8,total_configured_cost_cap=args.max_cost,
        ledger_status='360 usage console redirects to account login; API-key billing endpoint not documented'))
    gateway_spent = 0.
    # Gateway first, so the subsequent direct-call guard can include all gateway
    # spending. max_tokens=64 and no automatic retries in either path.
    for size in ('short','long'):
        family = f'probe-gateway-{size}'
        messages = document(run_id,f'probe-common-{size}',size)
        messages[0]['content'] = messages[0]['content'].replace('probe-common', 'probe-gateway')
        warm = call(d['endpoint'],messages,'billing_probe','warmup',family,size,'fixed_endpoint')
        gateway_spent = warm['conservative_running_cost']
        if not warm.get('valid'):
            raise RuntimeError('Invalid gateway warmup; stop without fabricated continuation')
        time.sleep(args.cache_wait)
        for n in (1,2):
            row = call(d['endpoint'],followup(messages,warm['content'],n),'billing_probe',
                       'followup',family,size,'fixed_endpoint')
            gateway_spent = row['conservative_running_cost']
        row = call(d['endpoint'],document(run_id,family+'-negative',size),'billing_probe',
                   'negative_prefix',family+'-negative',size,'fixed_endpoint')
        gateway_spent = row['conservative_running_cost']
    credential = provider_key(config,d,args.env_file)
    direct_spent, direct_count = 0., 0
    with httpx.Client(base_url=d['api_base'],headers={'Authorization':'Bearer '+credential},
                      timeout=httpx.Timeout(args.timeout,connect=10),trust_env=False) as client:
        def direct(messages,role,family,size):
            nonlocal direct_spent,direct_count
            if direct_count>=8:
                raise RuntimeError('Direct call count cap reached')
            upper = sum(len(json.dumps(m,ensure_ascii=False).encode('utf-8'))+256 for m in messages)+1024
            reserve = upper*d['input_cost_per_token']+64*d['output_cost_per_token']
            if gateway_spent+direct_spent+reserve>args.max_cost:
                raise RuntimeError('Combined gateway/direct conservative cost cap reached')
            row = dict(type='call',phase='billing_probe',probe_path='direct',role=role,family=family,size=size,
                trial_id=f'{run_id}-direct-{direct_count:02d}',endpoint=d['endpoint'],messages=messages,
                started_at_utc=datetime.now(timezone.utc).isoformat(),slo_ms=args.slo_ms)
            direct_count+=1
            payload = dict(model=d['upstream_model'].removeprefix('openai/'),messages=messages,
                           stream=False,temperature=0,max_tokens=64,**(d.get('extra_body') or {}))
            start = time.perf_counter()
            try:
                response = client.post('/chat/completions',json=payload)
                data = response.json()
                choice = (data.get('choices') or [{}])[0]
                message = choice.get('message') or {}
                correct = content_correct(messages,message.get('content'),role)
                row.update(http_status=response.status_code,transport_success=response.is_success,
                    e2e_ms=(time.perf_counter()-start)*1000,usage=data.get('usage'),
                    response_id=data.get('id'),response_model=data.get('model'),content=message.get('content'),
                    reasoning_content=message.get('reasoning_content'),finish_reason=choice.get('finish_reason'),
                    content_correct=correct,valid=bool(response.is_success and correct and choice.get('finish_reason')=='stop'),
                    direct_api_base=d['api_base'],response_headers={k:response.headers[k] for k in
                        ('content-type','x-request-id','request-id') if k in response.headers},
                    cached_tokens=cached_tokens(data.get('usage')))
                if not response.is_success:
                    row['error_type'] = 'provider_http_error'  # secret-bearing errors are not copied
            except Exception as exc:
                row.update(valid=False,transport_success=False,error_type=type(exc).__name__,
                           e2e_ms=(time.perf_counter()-start)*1000)
            usage = row.get('usage') or {}
            inp,out = usage.get('prompt_tokens'),usage.get('completion_tokens')
            full = (inp*d['input_cost_per_token']+out*d['output_cost_per_token']
                    if type(inp) is int and inp>=0 and type(out) is int and out>=0 else None)
            # amount is not used to enforce the budget: its currency is unverified.
            direct_spent += full if full is not None else reserve
            row.update(full_input_configured_reference=full,finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       combined_conservative_running_cost=gateway_spent+direct_spent)
            event(row)
            time.sleep(args.spacing)
            return row
        for size in ('short','long'):
            family = f'probe-direct-{size}'
            messages = document(run_id,f'probe-common-{size}',size)
            messages[0]['content'] = messages[0]['content'].replace('probe-common', 'probe-direct')
            warm = direct(messages,'warmup',family,size)
            if not warm.get('valid'):
                raise RuntimeError('Invalid direct warmup; stop, no automatic retries')
            time.sleep(args.cache_wait)
            for n in (1,2):
                direct(followup(messages,warm['content'],n),'followup',family,size)
            direct(document(run_id,family+'-negative',size),'negative_prefix',family+'-negative',size)
    event(dict(type='probe_finished',total_calls=8+direct_count,direct_calls=direct_count,
               combined_conservative_cost=gateway_spent+direct_spent,provider_invoice_verified=False))


def report(events,directory):
    rows = [r for r in events if r.get('type')=='call' and r.get('phase')=='billing_probe']
    results = []
    for row in rows:
        usage = row.get('usage') or {}
        contradiction = (type(usage.get('prompt_cache_hit_tokens')) is int
            and type(usage.get('prompt_cache_miss_tokens')) is int
            and usage['prompt_cache_hit_tokens']+usage['prompt_cache_miss_tokens'] != usage.get('prompt_tokens'))
        results.append(dict(trial_id=row['trial_id'],path=row.get('probe_path','gateway'),size=row['size'],
            role=row['role'],valid=row.get('valid'),input_tokens=usage.get('prompt_tokens'),
            output_tokens=usage.get('completion_tokens'),raw_hit=usage.get('prompt_cache_hit_tokens'),
            raw_miss=usage.get('prompt_cache_miss_tokens'),contradiction=contradiction,
            validated_hit=cached_tokens(usage),provider_amount=parse_amount(usage),
            gateway_cost=gateway_cost(row),full_input_reference=row.get('full_input_configured_reference',
                (row.get('accounting') or {}).get('full_input_reference'))))
    summary = dict(calls=len(rows),valid=sum(bool(r.get('valid')) for r in rows),rows=results,
        provider_invoice_verified=False,amount_currency_verified=False,
        direct_contradictions=sum(r['path']=='direct' and r['contradiction'] for r in results),
        gateway_contradictions=sum(r['path']=='gateway' and r['contradiction'] for r in results),
        probe_finished=next((e for e in events if e.get('type')=='probe_finished'),None),
        configuration_check=next((e for e in reversed(events) if e.get('type')=='finished'),None))
    lines = ['# 360实时计费与缓存字段核验','','只读配置；最多16次真实请求。网关和直连接口使用独立前缀，不重放旧响应。',
        '', '| 路径 | 长度 | 请求 | 输入tokens | 返回hit/miss | amount | 网关费用 | 普通价参考 |',
        '|---|---|---|---:|---|---:|---:|---:|']
    fmt = lambda value: '未知' if value is None else f'{value:.8f}'
    for r in results:
        lines.append(f"| {r['path']} | {r['size']} | {r['role']} | {r['input_tokens']} | {r['raw_hit']}/{r['raw_miss']} | "
                     f"{fmt(r['provider_amount'])} | {fmt(r['gateway_cost'])} | {fmt(r['full_input_reference'])} |")
    lines += ['', f"实际调用{summary['calls']}次，正确有效{summary['valid']}次。直连字段矛盾{summary['direct_contradictions']}次，网关字段矛盾{summary['gateway_contradictions']}次。",
        '', '## 可以与不可以确认的事情',
        '- 若直连也出现0/0但输入非零，则问题不只是LiteLLM转换造成；原始供应商响应就不符合公开字段约束。',
        '- 新前缀和重复前缀的amount变化可以作为费用差异证据，但不等于已经验证币种和账户实际扣款。',
        '- 平台用量页要求账户登录；现有API Key没有已核验的账单查询能力。不猜测未公开接口，也不把LiteLLM自身spend当供应商余额。',
        '- 未自动修改旧实验的费用结论；实际收益仍须请求级账单核对。',
        '', '依据：[官方缓存文档](https://ai.360.com/docs/cache)、[对应模型渠道价格](https://ai.360.com/open/en/models/deepseek/deepseek-v4-flash)。']
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
