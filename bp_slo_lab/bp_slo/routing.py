"""Request -> fixed SLO -> conditional estimates -> feasibility -> Pareto.

Only the BP project is used. No endpoint calls, capacity simulation, retry
execution or stability-based fallback ordering are performed here.
"""
from copy import deepcopy
from time import perf_counter_ns

from .dataset import number, timestamp, token
from .pareto import Candidate, rank_frontier
from .performance import validate_performance_config
from .slo import assign_slo
from .tolerance import acceptance_limit, validate_config as validate_tolerance_config


def validate_routing_config(config):
    if config.get('schema_version') != '1.0' or config.get('stage') != 'request_conditioned_routing':
        raise ValueError('Wrong routing experiment contract')
    if config.get('grade') not in ('p50', 'p75', 'p95', 'strict', 'standard', 'relaxed'):
        raise ValueError('Unknown SLO grade')
    for field in ('lambda_cost', 'eta_ttft'):
        if number(config.get(field)) is None or config[field] > 1:
            raise ValueError(field+' must be in [0,1]')
    if type(config.get('sample_requests_per_group')) is not int or not 1 <= config['sample_requests_per_group'] <= 10:
        raise ValueError('Replay uses 1-10 requests per supported group')
    currency, rates = config.get('currency'), config.get('fx_rates_to_currency')
    if not isinstance(currency, str) or not currency or not isinstance(rates, dict) or rates.get(currency) != 1:
        raise ValueError('Explicit FX rates need the target currency at rate 1')
    if any(number(v) is None or v <= 0 for v in rates.values()):
        raise ValueError('FX rates must be finite and positive')
    validate_performance_config(config['performance'])
    if 'selection_policy' in config:
        from .availability import validate_selection_policy
        validate_selection_policy(config['selection_policy'])
    return config


class ModelEndpointCatalog:
    """Model-indexed in-memory configuration; add/remove without full scans."""
    def __init__(self, offerings):
        self.records, self.model_to_endpoints = {}, {}
        for offering in offerings:
            key = offering.get('model_id'), offering.get('endpoint_id')
            if key in self.records:
                raise ValueError('Duplicate model/endpoint offering')
            self.upsert(offering)

    def upsert(self, offering):
        model, endpoint = offering.get('model_id'), offering.get('endpoint_id')
        if not isinstance(model, str) or not model or not isinstance(endpoint, str) or not endpoint:
            raise ValueError('Offering IDs must be nonempty strings')
        self.records[(model, endpoint)] = deepcopy(offering)
        self.model_to_endpoints.setdefault(model, set()).add(endpoint)

    def remove(self, model, endpoint):
        if self.records.pop((model, endpoint), None) is None:
            return False
        self.model_to_endpoints[model].remove(endpoint)
        if not self.model_to_endpoints[model]:
            del self.model_to_endpoints[model]
        return True

    def endpoints(self, model):
        return sorted(self.model_to_endpoints.get(model, ()))

    def quote(self, request, endpoint, currency, rates):
        inp, out = token(request.get('predicted_input_tokens')), token(request.get('predicted_output_tokens'))
        if inp is None or out is None:
            return dict(status='missing_predicted_length', cost=None)
        price = self.records[(request['model_id'], endpoint)].get('price_config') or {}
        if price.get('pricing_status') != 'configured' or price.get('billing_unit') != 'per_million_tokens':
            return dict(status='unsupported_price_config', cost=None)
        matches = []
        for index, tier in enumerate(price.get('price_tiers') or []):
            low, high = tier.get('input_tokens_min_exclusive'), tier.get('input_tokens_max_inclusive')
            if any(v is not None and token(v) is None for v in (low, high)) or (low is not None and high is not None and high <= low):
                return dict(status='invalid_price_bounds', cost=None)
            conditions = tier.get('conditions') or {}
            if not isinstance(conditions, dict):
                return dict(status='invalid_price_conditions', cost=None)
            if ((low is None or inp > low) and (high is None or inp <= high)
                    and all(k in request and request[k] == v for k, v in conditions.items())):
                matches.append((index, tier))
        if len(matches) != 1:
            return dict(status='ambiguous_price_tier' if matches else 'no_matching_price_tier', cost=None)
        index, tier = matches[0]
        source_currency = tier.get('currency', price.get('currency'))
        rate = rates.get(source_currency)
        if rate is None:
            return dict(status='missing_fx_rate', cost=None, source_currency=source_currency)
        if tier.get('billing_unit', price['billing_unit']) != 'per_million_tokens':
            return dict(status='unsupported_price_unit', cost=None)
        pin, pout = tier.get('input_per_million'), tier.get('output_per_million')
        if number(pin) is None or number(pout) is None:
            return dict(status='missing_token_prices', cost=None)
        cost = (inp*pin + out*pout)/1_000_000*rate
        if number(cost) is None:
            return dict(status='invalid_cost', cost=None)
        return dict(status='quoted', cost=cost, currency=currency, source_currency=source_currency,
                    fx_rate=rate, tier_index=index, input_per_million=pin, output_per_million=pout,
                    predicted_input_tokens=inp, predicted_output_tokens=out,
                    cache_discount_assumed=False, billing_unit='per_request_estimated_cost')


class RoutingEngine:
    def __init__(self, rules, tolerance, config, catalog, performance_index):
        self.rules, self.tolerance = rules, validate_tolerance_config(tolerance)
        self.config = validate_routing_config(config)
        if self.config['grade'] not in rules['quantiles']:
            raise ValueError('Routing grade does not exist in the selected SLO profile')
        if rules.get('method') == 'token_neighborhood':
            from .neighborhood_slo import NeighborhoodSLO
            self.slo_lookup = NeighborhoodSLO(rules)
        else:
            self.slo_lookup = rules
        self.catalog, self.index = catalog, performance_index
        if self.index.config != self.config['performance']:
            raise ValueError('Estimator and router performance configurations differ')

    def route(self, request):
        if 'selection_policy' in self.config:
            from .availability import route_available
            return route_available(self, request)
        start = perf_counter_ns()
        if timestamp(request.get('arrived_at')) is None or request.get('stream_type') not in ('stream', 'nonstream'):
            raise ValueError('Routing needs a valid arrival and stream mode')
        # Advance simulates observation arrival, separately timed from decision work.
        self.index.advance(request['arrived_at'])
        state_done = perf_counter_ns()
        slo = assign_slo(request, self.slo_lookup)['grades'][self.config['grade']]
        for metric, rule in slo.items():
            rule['acceptance_limit'] = acceptance_limit(rule['limit'], metric, self.tolerance) if rule['status'] == 'assigned' else None
        endpoints = self.catalog.endpoints(request['model_id'])
        lookup_done = perf_counter_ns()
        rows, points = [], []
        for endpoint in endpoints:
            quote = self.catalog.quote(request, endpoint, self.config['currency'], self.config['fx_rates_to_currency'])
            estimates = {metric: self.index.estimate(request, endpoint, metric) for metric in slo}
            ratios, reasons = {}, []
            if quote['status'] != 'quoted':
                reasons.append('price:'+quote['status'])
            for metric, rule in slo.items():
                estimate, limit = estimates[metric]['estimate'], rule['acceptance_limit']
                if limit is None:
                    reasons.append('slo:'+metric+':'+rule['status'])
                if estimate is None:
                    reasons.append('performance:'+metric+':'+estimates[metric]['status'])
                if limit is not None and estimate is not None:
                    ratios[metric] = estimate/limit if limit > 0 else (0.0 if estimate == 0 else None)
                    if estimate > limit:
                        reasons.append('slo_exceeded:'+metric)
            if set(ratios) != set(slo) or any(v is None for v in ratios.values()):
                loss = None
            elif request['stream_type'] == 'stream':
                eta = self.config['eta_ttft']
                loss = eta*ratios['ttft']+(1-eta)*ratios['tpot']
            else:
                loss = ratios['e2e']
            if loss is None:
                reasons.append('missing_performance_loss')
            feasible = not reasons
            rows.append(dict(endpoint_id=endpoint, price=quote, estimates=estimates, slo_ratios=ratios,
                             performance_loss=loss, feasible=feasible, exclusion_reasons=reasons))
            if quote['cost'] is not None and loss is not None:
                points.append(Candidate(endpoint, quote['cost'], loss, feasible, self.config['currency']))
        prediction_done = perf_counter_ns()
        ranked = rank_frontier(points, self.config['lambda_cost'])
        ranking_done = perf_counter_ns()
        point_index = {p['endpoint_id']: p for p in ranked['candidates']}
        for row in rows:
            point = point_index.get(row['endpoint_id']) or {}
            row.update(pareto=point.get('pareto', False), score=point.get('score'))
        decision = dict(request_id=request['request_id'], model_id=request['model_id'], stream_type=request['stream_type'],
                        arrived_at=request['arrived_at'], predicted_input_tokens=request.get('predicted_input_tokens'),
                        predicted_output_tokens=request.get('predicted_output_tokens'),
                        grade=self.config['grade'], slo=slo, candidates=rows,
                        frontier_ranked=ranked['frontier_ranked'], slots=ranked['slots'], cost_reference=ranked['cost_reference'],
                        status='selected' if ranked['slots'][0] is not None else 'no_feasible_endpoint',
                        decision_rule='independent_metric_slo_gates_then_pareto_only_score_top3',
                        capacity_health_policy='not_simulated_not_certified_by_this_offline_experiment')
        timing = dict(request_id=request['request_id'], state_update_ns=state_done-start,
                      lookup_ns=lookup_done-state_done, prediction_and_cost_ns=prediction_done-lookup_done,
                      pareto_ns=ranking_done-prediction_done, decision_ns=ranking_done-state_done,
                      total_ns=ranking_done-start, candidate_count=len(endpoints))
        return decision, timing
