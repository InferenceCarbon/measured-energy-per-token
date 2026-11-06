"""Forward calculations from a template: bytes of weights, bytes of KV,
expected MoE expert union, and memory feasibility.

These are LOGICAL byte demands. Whether they track physical HBM traffic
is what cell c9 (Nsight profile) tests; the paper describes them as
logical predictors until that is shown.

All byte quantities are in bytes; parameters in count; context in tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .templates import Catalogue, ModelTemplate

GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def weight_bytes_total(m: ModelTemplate, cat: Catalogue, precision: str) -> float:
    """Resident weight bytes at the served precision."""
    return m.n_params_total_b * 1e9 * cat.bytes_per_param[precision]


def weight_bytes_active_one_token(m: ModelTemplate, cat: Catalogue, precision: str) -> float:
    """Bytes a single token's forward pass reads. For dense models this is
    the whole model; for MoE it is shared layers plus top_k experts."""
    return m.n_params_active_b * 1e9 * cat.bytes_per_param[precision]


def expected_expert_union(n_experts: int, top_k: int, batch: int) -> float:
    """Expected number of DISTINCT experts touched by `batch` tokens under
    independent uniform routing. Bounded by k <= U(B) <= min(kB, N).

    128 experts, k=8, B=64 -> ~126 (the review's illustration)."""
    if batch <= 0:
        return 0.0
    p_untouched = (1.0 - top_k / n_experts) ** batch
    u = n_experts * (1.0 - p_untouched)
    return min(max(u, float(top_k)), float(min(top_k * batch, n_experts)))


def weight_bytes_batch(m: ModelTemplate, cat: Catalogue, precision: str,
                       batch: int, skew: float = 1.0) -> float:
    """Shared weight bytes read in ONE decode iteration serving `batch` tokens.

    Dense: the whole model, once.
    MoE: non-expert (shared) parameters once, plus the expected union of
    experts across the batch. `skew` scales the union toward k (skew<1,
    correlated routing) or leaves it uniform (skew=1). The measured
    per-layer union from the router hook replaces this estimate for
    calibrated models; this function is the closed-model estimate.
    """
    bpp = cat.bytes_per_param[precision]
    if not m.is_moe:
        return m.n_params_total_b * 1e9 * bpp

    moe = m.moe
    per_expert_params = _expert_params(m)
    routed_total = per_expert_params * moe["n_experts"] * moe["routed_layers"]
    shared_params = m.n_params_total_b * 1e9 - routed_total
    union = expected_expert_union(moe["n_experts"], moe["top_k"], batch)
    union = max(float(moe["top_k"]), union * skew)
    expert_bytes = union * per_expert_params * moe["routed_layers"] * bpp
    return shared_params * bpp + expert_bytes


def _expert_params(m: ModelTemplate) -> float:
    """Parameters in one expert's MLP (gate, up, down projections)."""
    return 3.0 * m.hidden_size * m.moe["expert_intermediate"]


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

def kv_bytes_per_token(m: ModelTemplate, cat: Catalogue, kv_precision: str) -> float:
    """Bytes of KV cache stored per token, from the template's layout.

    GQA:  2 (K and V) * n_layers * n_kv_heads * head_dim * bytes
    MLA:  n_layers * (kv_lora_rank + qk_rope_head_dim) * bytes
          (compressed latent plus decoupled rotary key; no separate V)
    Hybrid local/global (Gemma 3): same per-layer formula as GQA, but the
          STORED amount per layer is bounded by the window for local
          layers; this function returns the unbounded per-token figure and
          kv_bytes_attended applies the window.
    """
    b = cat.bytes_per_kv_element[kv_precision]
    if m.is_mla:
        return m.n_layers * (m.mla["kv_lora_rank"] + m.mla["qk_rope_head_dim"]) * b
    return 2.0 * m.n_layers * m.n_kv_heads * m.head_dim * b


def kv_bytes_attended(m: ModelTemplate, cat: Catalogue, kv_precision: str,
                      context: int) -> float:
    """Bytes of KV read per request per decode step at attended context L."""
    if not m.is_hybrid_local_global:
        return kv_bytes_per_token(m, cat, kv_precision) * context
    b = cat.bytes_per_kv_element[kv_precision]
    per_layer_per_token = 2.0 * m.n_kv_heads * m.head_dim * b
    pattern = m.local_global_pattern
    n_global = sum(1 for x in pattern if x == "global")
    n_local = len(pattern) - n_global
    layers_global = m.n_layers * n_global / len(pattern)
    layers_local = m.n_layers * n_local / len(pattern)
    local_ctx = min(context, m.sliding_window)
    return per_layer_per_token * (layers_global * context + layers_local * local_ctx)


# ---------------------------------------------------------------------------
# Memory feasibility
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Feasibility:
    fits: bool
    n_gpus: int
    weight_gb: float
    free_for_kv_gb: float
    kv_per_request_gb: float
    b_max: int
    reason: str = ""


def feasibility(m: ModelTemplate, cat: Catalogue, accelerator: str,
                n_gpus: int, weight_precision: str, kv_precision: str,
                context: int, gpu_memory_utilization: float | None = None) -> Feasibility:
    """Does the model fit on n_gpus of `accelerator`, and how many requests
    at `context` can be resident at once? `gpu_memory_utilization` overrides
    the serving default (a cell that varies vLLM's memory target moves the
    KV pool, hence B_max). This is the TEMPLATE pool estimate; the pool vLLM
    actually allocates is read from its startup line (server.kv_pool_from_log)
    and was 12-56% larger on the observed pods."""
    acc = cat.accelerators[accelerator]
    if weight_precision == "nvfp4" and not acc["nvfp4"]:
        return Feasibility(False, n_gpus, 0, 0, 0, 0, "nvfp4 unsupported on this accelerator")
    fz = cat.serving["feasibility"]
    gmu = gpu_memory_utilization if gpu_memory_utilization is not None else cat.serving["vllm"]["gpu_memory_utilization"]
    hbm_total = acc["hbm_gb"] * GB * n_gpus * gmu
    w = weight_bytes_total(m, cat, weight_precision) * fz["weight_overhead_factor"]
    if w > hbm_total:
        return Feasibility(False, n_gpus, w / GB, 0, 0, 0,
                           f"weights {w/GB:.0f} GB exceed usable HBM {hbm_total/GB:.0f} GB")
    free = (hbm_total - w) * fz["kv_fraction_of_free"]
    kv_req = kv_bytes_attended(m, cat, kv_precision, context)
    # Storage, not read-per-step: for hybrid models stored KV equals attended.
    b_max = int(free // kv_req) if kv_req > 0 else 0
    fits = b_max >= fz["min_batch_for_cell"]
    reason = "" if fits else f"no room for one request at context {context}"
    return Feasibility(fits, n_gpus, w / GB, free / GB, kv_req / GB, b_max, reason)


def min_gpus(m: ModelTemplate, cat: Catalogue, accelerator: str,
             weight_precision: str, max_gpus: int = 8) -> int | None:
    """Smallest power-of-two GPU count whose HBM holds the weights."""
    n = 1
    kvp = weight_precision if weight_precision in cat.bytes_per_kv_element else "fp8"   # KV cache has no fp4 format
    while n <= max_gpus:
        f = feasibility(m, cat, accelerator, n, weight_precision, kvp, 1024)
        if f.weight_gb and f.fits:
            return n
        n *= 2
    return None


def crossover_batch(m: ModelTemplate, cat: Catalogue, weight_precision: str,
                    kv_precision: str, context: int) -> float:
    """Batch depth at which per-token weight share equals per-token KV read:
    B* = W / K(L). Model- and context-specific; it need not be 64."""
    w = weight_bytes_batch(m, cat, weight_precision, batch=1)
    k = kv_bytes_attended(m, cat, kv_precision, context)
    return math.inf if k == 0 else w / k
