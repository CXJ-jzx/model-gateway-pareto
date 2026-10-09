"""Future router contract: score-ranked Pareto frontier only, exactly three slots."""
from dataclasses import dataclass
import math
from statistics import median


def _nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class Candidate:
    endpoint_id: str
    cost: float
    performance: float
    eligible: bool = True
    currency: str = "CNY"

    def __post_init__(self):
        if not isinstance(self.endpoint_id, str) or not self.endpoint_id or not isinstance(self.currency, str) or not self.currency:
            raise ValueError("Nonempty endpoint and currency required")
        if not _nonnegative(self.cost) or not _nonnegative(self.performance) or not isinstance(self.eligible, bool):
            raise ValueError("Cost/performance must be finite nonnegative numbers and eligible must be bool")


def rank_frontier(candidates, lambda_cost=0.5, fill_dominated=False):
    if not isinstance(fill_dominated, bool):
        raise ValueError('fill_dominated must be bool')
    if not _nonnegative(lambda_cost) or lambda_cost > 1:
        raise ValueError("lambda_cost must be in [0,1]")
    candidates = list(candidates)
    if any(not isinstance(c, Candidate) for c in candidates):
        raise ValueError("Expected Candidate objects")
    if len({c.endpoint_id for c in candidates}) != len(candidates):
        raise ValueError("Duplicate endpoint_id")
    if len({c.currency for c in candidates}) > 1:
        raise ValueError("Mixed currencies require upstream conversion")
    eligible = [c for c in candidates if c.eligible]
    positive = [c.cost for c in eligible if c.cost > 0]
    reference = (median(positive) if positive else 1.0) if eligible else None
    frontier = [c for c in eligible if not any(
        other.cost <= c.cost and other.performance <= c.performance
        and (other.cost < c.cost or other.performance < c.performance) for other in eligible)]
    scores = {c.endpoint_id: lambda_cost * (c.cost / reference) + (1-lambda_cost)*c.performance for c in eligible}
    ranked = sorted(frontier, key=lambda c: (scores[c.endpoint_id], c.endpoint_id))
    names = [c.endpoint_id for c in ranked]
    if fill_dominated:
        dominated = sorted((c for c in eligible if c.endpoint_id not in names),
                           key=lambda c: (scores[c.endpoint_id], c.endpoint_id))
        selected = (names + [c.endpoint_id for c in dominated])[:3]
        return dict(rule='pareto_first_dominated_fill_top3', frontier_ranked=names,
                    slots=(selected + [None]*3)[:3], cost_reference=reference,
                    candidates=[dict(endpoint_id=c.endpoint_id, eligible=c.eligible, cost=c.cost,
                                     performance=c.performance, pareto=c.endpoint_id in names,
                                     score=scores.get(c.endpoint_id),
                                     selection_reason=('pareto_frontier' if c.endpoint_id in names else 'dominated_fill')
                                     if c.endpoint_id in selected else None) for c in candidates])
    slots = (names[:3] + [None]*3)[:3]
    return dict(rule="pareto_only_top3", frontier_ranked=names, slots=slots, cost_reference=reference,
                candidates=[dict(endpoint_id=c.endpoint_id, eligible=c.eligible, cost=c.cost, performance=c.performance,
                                 pareto=c.endpoint_id in names, score=scores.get(c.endpoint_id)) for c in candidates])
