"""In-window prefix-cache check for cached-context windows. Energy-blind.

A decode window with `context_cached` > 0, or a prefix_access window with `control: cached`, is
meant to read its context from the server's prefix cache. The harness confirms the cache is warm
BEFORE the metered window; this module is the check DURING it. When the target batch times the
tokens one request holds exceeds the server's KV pool, vLLM evicts prefix blocks and re-prefills the
context on every request: the window is then decode plus an uncached prefill, not the shape its run
id states (11 Nov 2026: 66 of 883 replication windows, 41 of them without any flag).

    hit_ratio = prefix_hits_delta / prefix_queries_delta     # vLLM's cumulative counters over the window
    expected  = cached / (cached + fresh_in)                 # the share of each prompt that is the prefix
    r         = hit_ratio / expected                         # ~1.0 healthy, ~0 context recomputed

The same definitions serve the harness (meter.cell, at run time) and the offline diagnostic
(analysis.cache_diagnostic, on stored windows), so a flag and a diagnostic row can never disagree.
"""

from __future__ import annotations

import math
import re
from typing import Any

R_MIN_DEFAULT = 0.8      # below this the window is flagged; observed r is bimodal (0.95-1.05 or below 0.2)
EVICTION_FLAG = re.compile(r"^cache evicted in window: ")


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def cached_context(run: dict[str, Any]) -> tuple[int, int] | None:
    """(cached tokens, fresh_in) for a window that is meant to read a cached context, else None."""
    p = {**(run.get("fixed") or {}), **(run.get("variant") or {})}
    fresh = _int(p.get("fresh_in"))
    ctx = _int(p.get("context_cached"))
    if ctx > 0:
        return ctx, fresh
    if run.get("phase") == "prefix_access" and p.get("control") == "cached" and _int(p.get("prefix_cached")) > 0:
        return _int(p.get("prefix_cached")), fresh
    return None


def relative_hit_ratio(hits_delta: Any, queries_delta: Any, cached: int, fresh: int) -> float | None:
    """r = (hits / queries) / (cached / (cached + fresh)); None when the counters are absent."""
    try:
        h, q = float(hits_delta), float(queries_delta)
    except (TypeError, ValueError):
        return None
    if math.isnan(h) or math.isnan(q) or q <= 0 or cached <= 0:
        return None
    return (h / q) / (cached / (cached + fresh))


def request_tokens(run: dict[str, Any]) -> int:
    """Tokens one resident request holds in the KV pool: cached prefix + fresh input + output."""
    p = {**(run.get("fixed") or {}), **(run.get("variant") or {})}
    return sum(_int(p.get(k)) for k in ("context_cached", "prefix_cached", "fresh_in", "out"))


def window_check(run: dict[str, Any], hits_delta: Any, queries_delta: Any, prefix_caching: bool,
                 pool_tokens: Any = None, r_min: float = R_MIN_DEFAULT) -> dict[str, Any] | None:
    """The record cell.json keeps for a cached-context window, or None when the check does not apply
    (no cached context, or the server runs without prefix caching). `flag` is the text to append to the
    cell's flags when the cache was evicted, else None. `pool_utilisation` is target batch x request
    tokens over the server's reported KV pool: above 1.0 the per-slot prefixes cannot all be resident."""
    cc = cached_context(run)
    if cc is None or not prefix_caching:
        return None
    cached, fresh = cc
    r = relative_hit_ratio(hits_delta, queries_delta, cached, fresh)
    out: dict[str, Any] = {"cached": cached, "fresh_in": fresh, "prefix_hits_delta": hits_delta,
                           "prefix_queries_delta": queries_delta, "r": r, "r_min": r_min, "flag": None}
    tb = (run.get("variant") or {}).get("target_batch")
    req = request_tokens(run)
    if isinstance(tb, int) and _int(pool_tokens) > 0 and req > 0:
        out.update({"request_tokens": req, "kv_pool_tokens": _int(pool_tokens),
                    "pool_utilisation": tb * req / _int(pool_tokens)})
    if r is not None and r < r_min:
        out["flag"] = (f"cache evicted in window: prefix hit ratio {r:.2f} of expected < {r_min:.2f}"
                       + (f" (target batch x request tokens = {out['pool_utilisation']:.2f} of the KV pool)"
                          if "pool_utilisation" in out else ""))
    return out


# ---------------------------------------------------------------------------
# c8 (simulated agent): scripted hits and misses
# ---------------------------------------------------------------------------

R_MAX_REPLAY = 1.10      # above this the scripted misses hit the cache (the v1.0.0 replay fault: ratio 1.18)
REPLAY_FAULT_FLAG = re.compile(r"^scripted misses hit the cache: ")


def _step_fields(st: Any) -> tuple[int, str]:
    if isinstance(st, dict):
        return int(st["context_tokens"]), str(st["cache_expected"])
    return int(st.context_tokens), str(st.cache_expected)


def replay_expected_hit_ratio(script: list[Any]) -> float | None:
    """Share of the replay's prompt tokens the server should find in its prefix cache if every scripted hit
    hits and every scripted miss misses. A hit step's prompt starts with the previous step's whole prompt
    (the previous answer is replaced by synthetic text, so the generated tokens do not count); a miss step
    rewrites the first line, so nothing of it is cached. Every agent follows the same script, so the ratio
    does not depend on the number of agents. v1.0.0's server counters matched the faulty harness exactly
    under this rule (0.791 for 11 steps, 0.928 for 30); the fixed harness should give 0.673 and 0.732."""
    steps = [_step_fields(st) for st in script]
    total = sum(ctx for ctx, _ in steps)
    if not steps or total <= 0:
        return None
    hits = sum(steps[i - 1][0] for i in range(1, len(steps)) if steps[i][1] == "hit")
    return hits / total


def replay_check(script: list[Any], hits_delta: Any, queries_delta: Any, prefix_caching: bool,
                 r_min: float = R_MIN_DEFAULT, r_max: float = R_MAX_REPLAY) -> dict[str, Any] | None:
    """The cache_check record of a c8 window: observed hit ratio over the scripted one. Below `r_min`, the
    cache was evicted (hits that should have hit missed); above `r_max`, scripted misses hit the cache (the
    replay did not perturb the prefix). Either makes the window not the workload its script states."""
    if not script or not prefix_caching:
        return None
    expected = replay_expected_hit_ratio(script)
    try:
        h, q = float(hits_delta), float(queries_delta)
        observed = h / q if q > 0 and not (math.isnan(h) or math.isnan(q)) else None
    except (TypeError, ValueError):
        observed = None
    r = observed / expected if observed is not None and expected else None
    steps = [_step_fields(st) for st in script]
    out: dict[str, Any] = {"kind": "replay", "steps": len(steps),
                           "misses_scripted": sum(1 for _, c in steps if c == "miss"),
                           "expected_hit_ratio": expected, "observed_hit_ratio": observed,
                           "prefix_hits_delta": hits_delta, "prefix_queries_delta": queries_delta,
                           "r": r, "r_min": r_min, "r_max": r_max, "flag": None}
    if r is not None and r < r_min:
        out["flag"] = (f"cache evicted in window: prefix hit ratio {r:.2f} of expected < {r_min:.2f} "
                       f"(replay: observed {observed:.3f}, scripted {expected:.3f})")
    elif r is not None and r > r_max:
        out["flag"] = (f"scripted misses hit the cache: prefix hit ratio {r:.2f} of expected > {r_max:.2f} "
                       f"(replay: observed {observed:.3f}, scripted {expected:.3f})")
    return out
