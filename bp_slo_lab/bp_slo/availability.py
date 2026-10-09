"""Availability-first selection, without pretending weak evidence is an SLO guarantee.

Legacy routing is intentionally unchanged. This policy is enabled explicitly
by selection_policy, and replaces (not multiplies) the evaluation tolerance.
"""
from time import perf_counter_ns

from .dataset import number, timestamp
from .pareto import Candidate, rank_frontier
from .slo import assign_slo


def validate_selection_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {
            'slo_limit_multiplier', 'fill_dominated', 'low_evidence_fallback', 'allow_unknown_performance'}:
        raise ValueError('Selection policy needs an explicit multiplier, fill and fallback contract')
    multiplier = number(policy['slo_limit_multiplier'])
    if multiplier is None or not 1 <= multiplier <= 2:
        raise ValueError('SLO multiplier must be finite and in [1,2]')
    if any(type(policy[field]) is not bool for field in ('fill_dominated', 'allow_unknown_performance')):
        raise ValueError('Fill and unknown-performance switches must be booleans')
    if policy['low_evidence_fallback'] not in ('disabled', 'only_if_no_regular_candidates'):
        raise ValueError('Weak evidence may only be enabled when regular candidates are absent')
    return policy


def _loss(request, ratios, metrics, eta):
    if set(ratios) != set(metrics) or any(value is None for value in ratios.values()):
        return None
    if request['stream_type'] == 'stream':
        return eta * ratios['ttft'] + (1-eta) * ratios['tpot']
    return ratios['e2e']


def route_available(engine, request):
    start = perf_counter_ns()
    if timestamp(request.get('arrived_at')) is None or request.get('stream_type') not in ('stream', 'nonstream'):
        raise ValueError('Routing needs a valid arrival and stream mode')
    engine.index.advance(request['arrived_at'])
    state_done = perf_counter_ns()
    policy = engine.config['selection_policy']
    slo = assign_slo(request, engine.slo_lookup)['grades'][engine.config['grade']]
    for rule in slo.values():
        limit = rule['limit'] * policy['slo_limit_multiplier'] if rule['status'] == 'assigned' else None
        if limit is not None and number(limit) is None:
            raise ValueError('SLO multiplier overflow')
        rule.update(acceptance_limit=limit, acceptance_formula='raw_slo * selection_policy.slo_limit_multiplier',
                    acceptance_multiplier=policy['slo_limit_multiplier'])
    endpoints = engine.catalog.endpoints(request['model_id'])
    lookup_done = perf_counter_ns()
    rows = []
    for endpoint in endpoints:
        quote = engine.catalog.quote(request, endpoint, engine.config['currency'], engine.config['fx_rates_to_currency'])
        estimates = {metric: engine.index.estimate(request, endpoint, metric) for metric in slo}
        ratios, exclusions, warnings, values = {}, [], [], {}
        if engine.catalog.records[(request['model_id'], endpoint)].get('enabled', True) is not True:
            exclusions.append('endpoint_disabled')
        if quote['status'] != 'quoted':
            exclusions.append('price:'+quote['status'])
        sufficient, unknown = True, False
        for metric, rule in slo.items():
            estimate = estimates[metric]
            status = estimate['status']
            value = estimate.get('estimate')
            if status == 'insufficient_evidence':
                sufficient = False
                # This remains exploratory evidence, never re-labelled sufficient.
                value = estimate.get('exploratory_estimate', estimate.get('exploratory_p95'))
                warnings.append('sparse_evidence:'+metric)
            elif status != 'estimated':
                sufficient = False
                exclusions.append('performance:'+metric+':'+status)
            if value is None:
                unknown = True
                warnings.append('unknown_performance:'+metric)
            values[metric] = value
            limit = rule['acceptance_limit']
            if limit is None:
                exclusions.append('slo:'+metric+':'+rule['status'])
            elif value is not None:
                ratios[metric] = value/limit if limit > 0 else (0.0 if value == 0 else None)
                if value > limit:
                    exclusions.append('slo_exceeded:'+metric)
        loss = _loss(request, ratios, slo, engine.config['eta_ttft'])
        regular = sufficient and not exclusions and loss is not None
        rows.append(dict(endpoint_id=endpoint, price=quote, estimates=estimates,
                         routing_estimates=values, slo_ratios=ratios, performance_loss=loss,
                         feasible=regular, exclusion_reasons=exclusions,
                         evidence_quality='sufficient' if sufficient else ('unknown' if unknown else 'sparse'),
                         slo_gate_status='exceeded_or_invalid' if exclusions else ('unknown' if unknown else 'passed'),
                         risk_warnings=warnings, eligible_for_selection=False, selection_reason=None,
                         pareto=False, score=None))
    prediction_done = perf_counter_ns()
    regular = [row for row in rows if row['feasible']]
    fallback = not regular and policy['low_evidence_fallback'] == 'only_if_no_regular_candidates'
    supported = regular or ([row for row in rows if not row['exclusion_reasons']
                            and row['evidence_quality'] == 'sparse' and row['performance_loss'] is not None]
                           if fallback else [])
    points = [Candidate(row['endpoint_id'], row['price']['cost'], row['performance_loss'], True,
                        engine.config['currency']) for row in supported]
    ranked = rank_frontier(points, engine.config['lambda_cost'], policy['fill_dominated'])
    names = [name for name in ranked['slots'] if name is not None]
    point_index = {point['endpoint_id']: point for point in ranked['candidates']}
    for row in supported:
        point = point_index[row['endpoint_id']]
        row.update(eligible_for_selection=True, pareto=point['pareto'], score=point['score'])
        if row['endpoint_id'] in names:
            reason = 'pareto_frontier' if point['pareto'] else 'dominated_fill'
            row['selection_reason'] = ('sparse_evidence_fallback:' if fallback else '') + reason
        if fallback:
            row['risk_warnings'].append('best_effort_no_slo_guarantee')
    # No fake performance=0 or made-up coordinates for endpoints with no samples.
    if fallback and policy['allow_unknown_performance'] and len(names) < 3:
        unknown_rows = sorted((row for row in rows if not row['exclusion_reasons']
                               and row['evidence_quality'] == 'unknown'),
                              key=lambda row: (-sum(value is not None for value in row['routing_estimates'].values()),
                                               row['price']['cost'], row['endpoint_id']))
        for row in unknown_rows[:3-len(names)]:
            names.append(row['endpoint_id'])
            row.update(eligible_for_selection=True, selection_reason='unknown_performance_best_effort_fallback')
            row['risk_warnings'].append('best_effort_no_slo_guarantee')
    slots = (names + [None]*3)[:3]
    ranking_done = perf_counter_ns()
    decision = dict(request_id=request['request_id'], model_id=request['model_id'], stream_type=request['stream_type'],
                    arrived_at=request['arrived_at'], predicted_input_tokens=request.get('predicted_input_tokens'),
                    predicted_output_tokens=request.get('predicted_output_tokens'), grade=engine.config['grade'],
                    slo=slo, candidates=rows, frontier_ranked=ranked['frontier_ranked'], slots=slots,
                    cost_reference=ranked['cost_reference'],
                    status=('selected_low_evidence' if fallback else 'selected') if names else 'no_feasible_endpoint',
                    selection_mode='best_effort_fallback' if fallback and names else 'regular',
                    decision_rule='typical_latency_independent_slo_gates_pareto_first_dominated_fill_then_empty_pool_fallback',
                    capacity_health_policy='not_simulated_not_certified_by_this_offline_experiment')
    timing = dict(request_id=request['request_id'], state_update_ns=state_done-start,
                  lookup_ns=lookup_done-state_done, prediction_and_cost_ns=prediction_done-lookup_done,
                  pareto_ns=ranking_done-prediction_done, decision_ns=ranking_done-state_done,
                  total_ns=ranking_done-start, candidate_count=len(endpoints))
    return decision, timing
