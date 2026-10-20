"""Exact-prefix affinity followed by the existing SLO/Pareto ranking rule.

Standalone experiment: no Mos imports, gateway configuration or health simulator.
The index stores evidence of *processed inputs*, not provider KV-cache contents.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from statistics import median
from time import perf_counter_ns


def prompt_key(messages, options=None):
    """Message-boundary serialization: a conversation extension preserves its prefix.

    Roles, whitespace, tool calls and options remain exact. This is conservative
    application-level matching, NOT a reconstruction of a provider's tokenizer.
    """
    if not messages or any(not isinstance(m, dict) for m in messages):
        raise ValueError('Nonempty message objects required')
    encode = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return encode(options or {}) + b'\n' + b''.join(encode(m) + b'\n' for m in messages)


@dataclass
class _Node:
    edge: bytes = b''
    children: dict = field(default_factory=dict)
    endpoints: dict = field(default_factory=dict)


class PrefixIndex:
    """Compressed byte Radix Tree, separated by trusted tenant and model/version.

    Only explicit completed-input terminals count as matches. Tree branch nodes
    are never mistaken for cached evidence. TTL is an evidence lifetime, not a
    provider cache-retention guarantee. All timestamps use monotonic milliseconds.
    """
    def __init__(self, ttl_ms=300_000, max_entries=512, max_prefix_bytes=256_000):
        if any(type(v) is not int or v <= 0 for v in (ttl_ms, max_entries, max_prefix_bytes)):
            raise ValueError('Positive integer cache bounds required')
        self.ttl_ms, self.max_entries, self.max_prefix_bytes = ttl_ms, max_entries, max_prefix_bytes
        self.roots, self.lru, self.expirations = {}, OrderedDict(), []

    @staticmethod
    def _scope(scope, model):
        if any(not isinstance(v, str) or not v for v in (scope, model)):
            raise ValueError('Explicit trusted cache scope and model/version required')
        return scope, model

    def _remove(self, identity):
        namespace, key, endpoint = identity
        self.lru.pop(identity, None)
        root = self.roots.get(namespace)
        if root is None:
            return
        node, offset, path = root, 0, []
        while offset < len(key):
            child = node.children[key[offset]]
            path.append((node, key[offset], child))
            offset += len(child.edge)
            node = child
        node.endpoints.pop(endpoint, None)
        for parent, first, child in reversed(path):
            if not child.endpoints and not child.children:
                del parent.children[first]
            elif not child.endpoints and len(child.children) == 1:
                descendant = next(iter(child.children.values()))
                child.edge += descendant.edge
                child.endpoints, child.children = descendant.endpoints, descendant.children
        if not root.children and not root.endpoints:
            self.roots.pop(namespace, None)

    def expire(self, now_ms):
        # Expiry min-heap avoids scanning every recorded prefix on each lookup.
        # LRU access order must not be confused with completion/expiry order.
        while self.expirations and self.expirations[0][0] <= now_ms:
            expiry, identity = heapq.heappop(self.expirations)
            seen = self.lru.get(identity)
            if seen is not None and seen + self.ttl_ms == expiry:
                self._remove(identity)

    def observe(self, scope, model, key, endpoint, now_ms):
        namespace = self._scope(scope, model)
        if not isinstance(key, bytes) or not key or len(key) > self.max_prefix_bytes:
            return False
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError('Endpoint ID required')
        self.expire(now_ms)
        node, remainder = self.roots.setdefault(namespace, _Node()), key
        while remainder:
            child = node.children.get(remainder[0])
            if child is None:
                child = _Node(remainder)
                node.children[remainder[0]] = child
                node = child
                break
            common = 0
            while common < min(len(remainder), len(child.edge)) and remainder[common] == child.edge[common]:
                common += 1
            if common < len(child.edge):
                split = _Node(child.edge[:common])
                node.children[remainder[0]] = split
                child.edge = child.edge[common:]
                split.children[child.edge[0]] = child
                child = split
            node, remainder = child, remainder[common:]
        node.endpoints[endpoint] = now_ms
        identity = namespace, key, endpoint
        self.lru[identity] = now_ms
        self.lru.move_to_end(identity)
        heapq.heappush(self.expirations, (now_ms+self.ttl_ms, identity))
        while len(self.lru) > self.max_entries:
            self._remove(next(iter(self.lru)))
        # Refresh/eviction leaves stale heap records. Compact them so repeatedly
        # observing one prefix cannot grow memory without bound.
        if len(self.expirations) > 2*self.max_entries:
            self.expirations = [(seen+self.ttl_ms, identity) for identity, seen in self.lru.items()]
            heapq.heapify(self.expirations)
        return True

    def lookup(self, scope, model, key, now_ms):
        namespace = self._scope(scope, model)
        self.expire(now_ms)
        node = self.roots.get(namespace)
        offset, matches = 0, []
        while node is not None:
            for endpoint, seen in node.endpoints.items():
                prefix = key[:offset]
                matches.append(dict(endpoint_id=endpoint, matched_bytes=offset, age_ms=now_ms-seen,
                                    prefix_sha256=hashlib.sha256(prefix).hexdigest()))
                self.lru.move_to_end((namespace, prefix, endpoint))
            if offset == len(key):
                break
            child = node.children.get(key[offset])
            if child is None or not key.startswith(child.edge, offset):
                break
            offset += len(child.edge)
            node = child
        return sorted(matches, key=lambda r: (-r['matched_bytes'], r['endpoint_id']))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def pareto_rank(candidates, slo_ms, alpha=.5):
    """Same linear rule: alpha*cost/median_cost + (1-alpha)*E2E/SLO.

    Strict 1.0 SLO admission; frontier first, then dominated fill, at most three.
    Missing estimates do not get invented. Caller supplies model-specific rows.
    """
    if not _number(slo_ms) or slo_ms <= 0 or not _number(alpha) or alpha > 1:
        raise ValueError('Positive SLO and alpha in [0,1] required')
    if len({r['endpoint_id'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate endpoint')
    rows = [dict(r) for r in candidates]
    for r in rows:
        r['admitted'] = (_number(r.get('estimated_e2e_ms')) and _number(r.get('estimated_cost'))
                         and r['estimated_e2e_ms'] <= slo_ms)
    eligible = [r for r in rows if r['admitted']]
    reference = median([r['estimated_cost'] for r in eligible if r['estimated_cost'] > 0] or [1.])
    for r in rows:
        r['pareto'] = r['admitted'] and not any(
            p['estimated_cost'] <= r['estimated_cost'] and p['estimated_e2e_ms'] <= r['estimated_e2e_ms']
            and (p['estimated_cost'] < r['estimated_cost'] or p['estimated_e2e_ms'] < r['estimated_e2e_ms'])
            for p in eligible)
        r['score'] = (alpha*r['estimated_cost']/reference + (1-alpha)*r['estimated_e2e_ms']/slo_ms
                      if r['admitted'] else None)
    ranked = sorted(eligible, key=lambda r: (not r['pareto'], r['score'], r['endpoint_id']))
    return [r['endpoint_id'] for r in ranked], rows


class CacheAwareRouter:
    """Preserve prefix-first policy, with a Pareto fallback, not a cache bonus.

    All endpoints are assumed healthy/available in this experiment. SLO admission
    still applies. Prefix-matched but dominated endpoints remain eligible for
    affinity; membership in the top-three Pareto output is not a prerequisite.
    """
    def __init__(self, index=None):
        self.index = index if index is not None else PrefixIndex()

    def choose(self, *, scope, model, messages, candidates, slo_ms, now_ms, alpha=.5,
               options=None, prefix_first=True):
        started = perf_counter_ns()
        key = prompt_key(messages, options)
        lookup_start = perf_counter_ns()
        matches = self.index.lookup(scope, model, key, now_ms) if prefix_first else []
        lookup_ns = perf_counter_ns()-lookup_start
        ranked, rows = pareto_rank(candidates, slo_ms, alpha)
        admitted = {r['endpoint_id']: r for r in rows if r['admitted']}
        valid = [r for r in matches if r['endpoint_id'] in admitted]
        chosen = None
        if valid:
            length = valid[0]['matched_bytes']
            best = min((r for r in valid if r['matched_bytes'] == length),
                       key=lambda r: (admitted[r['endpoint_id']]['score'], r['endpoint_id']))
            chosen = best['endpoint_id']
        reason = 'prefix_affinity' if chosen else ('pareto_fallback' if ranked else 'no_slo_feasible_endpoint')
        selected = ([chosen] + [e for e in ranked if e != chosen]) if chosen else ranked
        return dict(selected_endpoint=selected[0] if selected else None, ranked_endpoint_ids=selected[:3],
                    reason=reason, matches=matches, candidates=rows, slo_ms=slo_ms, alpha=alpha,
                    prefix_lookup_ns=lookup_ns, decision_ns=perf_counter_ns()-started,
                    matched_prefix_is_not_confirmed_cache_hit=True)

    def feedback(self, *, scope, model, messages, endpoint, now_ms, success, identity_verified,
                 options=None):
        # No pending/in-flight request is recorded as a completed cache location.
        if success and identity_verified:
            return self.index.observe(scope, model, prompt_key(messages, options), endpoint, now_ms)
        return False


def cached_tokens(usage):
    """Unknown and inconsistent provider evidence remain unknown, never zero."""
    if not isinstance(usage, dict) or type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens'] < 0:
        return None
    details = usage.get('prompt_tokens_details') or {}
    values = [usage[k] for k in ('prompt_cache_hit_tokens',) if k in usage]
    if isinstance(details, dict) and 'cached_tokens' in details:
        values.append(details['cached_tokens'])
    if not values or any(type(v) is not int or not 0 <= v <= usage['prompt_tokens'] for v in values):
        return None
    if len(set(values)) != 1:
        return None
    hit = values[0]
    if 'prompt_cache_miss_tokens' in usage:
        miss = usage['prompt_cache_miss_tokens']
        if type(miss) is not int or miss < 0 or hit+miss != usage['prompt_tokens']:
            return None
    return hit


def actual_cost(usage, prices):
    """Verified configured token arithmetic, not a vendor invoice.

    When hit evidence or its tariff is missing, return a full-input reference,
    but leave cache-adjusted cost/savings unknown instead of fabricating savings.
    """
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                         for k in ('prompt_tokens', 'completion_tokens')):
        return dict(cache_adjusted_cost=None, full_input_reference=None, cache_saving=None)
    pin, pout = prices.get('input_cost_per_token'), prices.get('output_cost_per_token')
    if not _number(pin) or not _number(pout):
        return dict(cache_adjusted_cost=None, full_input_reference=None, cache_saving=None)
    inp, out = usage['prompt_tokens'], usage['completion_tokens']
    full = inp*pin + out*pout
    hit, read_price = cached_tokens(usage), prices.get('cache_read_input_token_cost')
    known = hit is not None and (hit == 0 or _number(read_price))
    adjusted = full - hit*(pin-read_price) if known and hit else (full if known else None)
    return dict(cache_adjusted_cost=adjusted, full_input_reference=full,
                cache_saving=full-adjusted if adjusted is not None else None,
                cached_tokens=hit, cache_price_known=_number(read_price))
