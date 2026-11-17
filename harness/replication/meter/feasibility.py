"""Turn cells.yaml into a run manifest: one row per (model, accelerator,
cell, variant, repeat), with infeasible rows dropped or resized and the
reason recorded, and an estimate of node-hours and GPU-hours per
accelerator that follows from the protocol minimums.

Usage:
    python -m meter.feasibility --out manifest.json
"""

from __future__ import annotations

import argparse
import itertools
import math
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from . import work
from .templates import CONFIG_DIR, Catalogue, load_catalogue


@dataclass
class Run:
    run_id: str
    model: str
    accelerator: str
    n_gpus: int
    tp: int
    weight_precision: str
    kv_precision: str
    cell: str
    phase: str
    variant: dict[str, Any]
    repeat: int
    est_seconds: float
    note: str = ""


def _resolve_grid(value: Any, m, cat: Catalogue, native: int | None = None) -> list[Any]:
    native = m.native_context if native is None else native
    if value == "context_grid":
        return [native if v == "native_max" else v
                for v in cat.serving.get("context_grid", [1024, 4096, 16384, "native_max"])]
    if isinstance(value, list):
        return [native if v == "native_max" else v for v in value]
    return [value]


def backend_of(cat: Catalogue, accelerator: str) -> str:
    return cat.accelerators[accelerator].get("backend", "vllm_cuda")


def model_on_accelerator(m, accelerator: str, cat: Catalogue) -> bool:
    """A template may name the accelerators it is planned on (`accelerators:`);
    on a Neuron backend it must also carry a `neuron:` block (tested tp), on
    the TPU backend a `tpu:` block (config/models_tpu.yaml)."""
    allowed = m.raw.get("accelerators")
    if allowed and accelerator not in allowed:
        return False
    be = backend_of(cat, accelerator)
    if be == "neuron" and "neuron" not in m.raw:
        return False
    if be == "tpu" and "tpu" not in m.raw:
        return False
    return True


def neuron_chips(tp: int, cat: Catalogue, accelerator: str) -> int:
    per_chip = int(cat.accelerators[accelerator].get("logical_cores_per_chip", 4))
    return max(1, -(-int(tp) // per_chip))


def compiled_tp(m, be: str, accelerator: str) -> int:
    """Base tensor-parallel degree from the template's `neuron:` / `tpu:` block.
    `tp` may be one integer or a map keyed by accelerator (the 70B needs 8
    v6e chips but fits 4 Ironwood chips)."""
    v = m.raw[be]["tp"]
    return int(v[accelerator] if isinstance(v, dict) else v)


def compiled_tp_options(m, be: str, accelerator: str) -> list[int]:
    nb = m.raw[be]
    v = nb.get("tp_options", [compiled_tp(m, be, accelerator)])
    if isinstance(v, dict):
        v = v.get(accelerator, [compiled_tp(m, be, accelerator)])
    return [int(x) for x in v]


def _tp_options(model_key: str, cat: Catalogue, accelerator: str, precision: str) -> list[int]:
    m = cat.model(model_key)
    be = backend_of(cat, accelerator)
    if be in ("neuron", "tpu"):
        return compiled_tp_options(m, be, accelerator)
    base = work.min_gpus(m, cat, accelerator, precision) or 1
    if model_key == "qwen3-8b":
        return [1, 2]
    if model_key == "llama-3.3-70b":
        return [max(base, 2), max(base, 2) * 2]
    return [base]


def _est_seconds(cell: dict[str, Any], variant: dict[str, Any], defaults: dict[str, Any]) -> float:
    """Protocol minimum: idle before + window + idle after, plus warm-up.
    Long-context and large-batch cells get a modest uplift because reaching
    min_tokens takes longer at low throughput."""
    idle = 2 * defaults["idle_window_s"]
    window = defaults["min_window_s"]
    warm = 20.0
    ctx = variant.get("context_cached", variant.get("fresh_in", variant.get("prefix_cached", 1024)))
    if isinstance(ctx, int) and ctx >= 16384:
        window *= 1.5
    if variant.get("target_batch", 1) == 1:
        window *= 1.5   # batch-1 decode is slow to accumulate tokens
    if cell["id"] == "c8_agent_replay":
        window = cell["fixed"]["steps"] * 25.0
        warm += min(int(defaults.get("replay_warmup_steps", 0)), cell["fixed"]["steps"]) * 25.0
    if cell["phase"] == "idle_loaded":
        window = 60.0
        warm = 0.0
    if cell["phase"] == "logprob_probe":
        # a fixed prompt set, not a protocol window: n_prompts x out tokens at ~20 tok/s per slot
        fx = cell.get("fixed", {})
        window = float(fx.get("n_prompts", 200)) * float(fx.get("out", 64)) / (20.0 * float(fx.get("concurrency", 4)))
        warm = 30.0
    return idle + window + warm


POOL_MARGIN_DEFAULT = 0.95      # share of the KV pool a plan may fill (pods of one type differ by up to 10%)
_REQUEST_KEYS = ("context_cached", "prefix_cached", "fresh_in", "out")


def load_kv_pool_observed(path: Path = CONFIG_DIR / "kv_pool_observed.yaml") -> dict[tuple, int]:
    """(model, accelerator, tp, weight precision, KV precision, gpu_memory_utilization) -> the smallest
    KV pool vLLM reported for that deployment, in tokens. Empty if the file is absent."""
    if not path.exists():
        return {}
    out: dict[tuple, int] = {}
    for e in (yaml.safe_load(path.read_text()) or {}).get("entries") or []:
        k = (e["model"], e["accelerator"], int(e["tp"]), e["weight_precision"], e["kv_precision"],
             round(float(e["gpu_memory_utilization"]), 3))
        out[k] = min(int(e["tokens_min"]), out.get(k, int(e["tokens_min"])))
    return out


def request_tokens(fixed: dict[str, Any], variant: dict[str, Any], ctx: Any) -> int:
    """Tokens one resident request holds: cached prefix + fresh input + output (never less than the
    cell's context). The harness gives each concurrent slot its own prefix, so this is per slot."""
    p = {**fixed, **variant}
    total = sum(int(p[k]) for k in _REQUEST_KEYS if isinstance(p.get(k), int) and not isinstance(p.get(k), bool))
    return max(total, int(ctx) if isinstance(ctx, int) else 1024)


def build_manifest(cat: Catalogue, cells_path: Path = CONFIG_DIR / "cells.yaml",
                   accelerators: tuple[str, ...] = ("h100", "b200")) -> dict[str, Any]:
    spec = yaml.safe_load(cells_path.read_text())
    defaults = spec["defaults"]
    cat.serving["context_grid"] = spec["context_grid"]
    # How B_max is sized (work.feasibility). A cells file without a `planner` block is planned with the
    # rule it was written under ("attended"), so every existing plan and run id rebuilds unchanged; a new
    # campaign declares `planner: {kv_sizing: pool_tokens}` and gets targets the server's KV pool can hold.
    # Under the old rule the plan still reports, in `kv_pool_warnings`, every target the pool cannot hold.
    planner = spec.get("planner") or {}
    kv_sizing = planner.get("kv_sizing", "attended")
    if kv_sizing not in ("attended", "pool_tokens"):
        raise ValueError(f"{cells_path.name}: unknown planner.kv_sizing {kv_sizing!r}")
    pool_margin = float(planner.get("pool_margin", POOL_MARGIN_DEFAULT))
    observed_pools = load_kv_pool_observed() if planner.get("use_observed_pools", True) else {}
    kv_pool_warnings: list[dict[str, Any]] = []

    runs: list[Run] = []
    dropped: list[dict[str, Any]] = []
    load_events: set[tuple[str, str, int, str]] = set()
    seen_bases: set[str] = set()

    for acc in accelerators:
        acc_cfg = cat.accelerators[acc]
        be = backend_of(cat, acc)
        neuron = be == "neuron"
        tpu = be == "tpu"
        d_acc = dict(defaults)
        if neuron or tpu:
            d_acc.update(cat.serving.get(be, {}).get("protocol", {}))
        seqs_cap = (max(int(x) for x in cat.serving["neuron"]["num_seqs_buckets"]) if neuron
                    else int(cat.serving["tpu"]["max_num_seqs"]) if tpu else None)
        for m in cat.calibration_models():
            if not model_on_accelerator(m, acc, cat):
                continue
            if tpu:
                from . import tpu as tpu_mod
                tpu_allowed = tpu_mod.allowed_weight_precisions(m, acc_cfg)
                if not tpu_allowed:
                    dropped.append({"model": m.key, "acc": acc, "cell": "*", "variant": {},
                                    "reason": f"no weight precision of {m.raw['tpu'].get('weight_precisions', m.weight_precisions)} "
                                              f"is served on {acc} ({acc_cfg.get('weight_precisions')})"})
                    continue
            # a template tagged `matrix: <name>` belongs to that campaign matrix only (the replication
            # models stay out of the original matrix)
            if m.raw.get("matrix") and m.raw["matrix"] != spec.get("matrix"):
                continue
            # compiled decode graphs are sized to max_model_len; the context the server can take may be below native
            native_ctx = min(m.native_context, int(acc_cfg.get("max_model_len_cap", m.native_context)))
            for cell in spec["cells"]:
                if cell.get("disabled"):
                    continue
                if "only_models" in cell and m.key not in cell["only_models"]:
                    continue
                if "only_accelerators" in cell and acc not in cell["only_accelerators"]:
                    continue
                repeats = cell.get("repeats", defaults["repeats"])
                precisions = cell["vary"].get("weight_precision", ["fp8"])
                if tpu:
                    precisions = [p for p in precisions if p in tpu_allowed] or [tpu_allowed[0]]
                else:
                    precisions = [p for p in precisions if p in m.weight_precisions
                                  and (p != "nvfp4" or cat.accelerators[acc]["nvfp4"])]
                    if not precisions:
                        precisions = [m.weight_precisions[0]]

                for precision in precisions:
                    kv_precision = (tpu_mod.allowed_kv_precision(m, acc_cfg) if tpu
                                    else "fp8" if "fp8" in m.kv_precisions else m.kv_precisions[0])
                    if neuron or tpu:
                        tps = (_tp_options(m.key, cat, acc, precision) if cell["vary"].get("tp") == "per_model"
                               else [compiled_tp(m, be, acc)])
                    else:
                        tps = (_tp_options(m.key, cat, acc, precision)
                               if cell["vary"].get("tp") == "per_model"
                               else [work.min_gpus(m, cat, acc, precision) or 1])

                    vary = {k: v for k, v in cell["vary"].items()
                            if k not in ("weight_precision", "tp")}
                    keys = list(vary.keys())
                    grids = [_resolve_grid(vary[k], m, cat, native_ctx) for k in keys]

                    for tp in tps:
                        for combo in itertools.product(*grids):
                            variant = dict(zip(keys, combo))
                            variant["weight_precision"] = precision
                            ctx = variant.get("context_cached",
                                              variant.get("fresh_in",
                                                          variant.get("prefix_cached", 1024)))
                            # "native_max" means: the whole REQUEST (cached prefix + fresh
                            # suffix + output) must fit the model's context window, with a
                            # margin for the instruction text and tokenizer slack. The
                            # router pass showed 32,972-token prompts rejected at 32,512, and
                            # a 256-token margin still lost by 2.2% (32,969 vs 32,512 on
                            # Qwen3-8B): the composed prefix+suffix text tokenises longer than
                            # its sized parts. Margin = 256 + 4% of the window.
                            fixed_c = cell.get("fixed", {})
                            margin = 256 + math.ceil(0.04 * native_ctx)
                            if variant.get("context_cached") == native_ctx:
                                fresh = int(variant.get("fresh_in", fixed_c.get("fresh_in", 200)))
                                out = int(variant.get("out", fixed_c.get("out", 256)))
                                variant["context_cached"] = native_ctx - fresh - out - margin
                                ctx = variant["context_cached"]
                            if variant.get("fresh_in") == native_ctx:
                                out = int(variant.get("out", fixed_c.get("out", 1)))
                                variant["fresh_in"] = native_ctx - out - margin
                                ctx = variant["fresh_in"]
                            if cell["id"] == "c8_agent_replay":
                                ctx = min(cell["fixed"]["context_start"]
                                          + cell["fixed"]["steps"] * cell["fixed"]["context_step"],
                                          native_ctx - 1024)
                                variant["context_cap"] = ctx
                            prof: dict[str, Any] = {}
                            served_ctx = native_ctx
                            if neuron and variant.get("neuron_profile") not in (None, "default"):
                                prof = cat.serving["neuron"].get("profiles", {}).get(variant["neuron_profile"]) or {}
                                if not prof:
                                    raise ValueError(f"unknown neuron_profile {variant['neuron_profile']!r}")
                                served_ctx = min(native_ctx, int(prof.get("max_model_len", native_ctx)))
                            if isinstance(ctx, int) and ctx > served_ctx:
                                dropped.append({"model": m.key, "acc": acc, "cell": cell["id"],
                                                "variant": variant,
                                                "reason": f"context {ctx} > served maximum {served_ctx}"
                                                          + (" (max_model_len_cap)" if native_ctx < m.native_context else "")})
                                continue
                            n_dev = neuron_chips(tp, cat, acc) if neuron else tp
                            f = work.feasibility(m, cat, acc, n_dev, precision, kv_precision,
                                                 int(ctx) if isinstance(ctx, int) else 1024,
                                                 gpu_memory_utilization=variant.get("gpu_memory_utilization"),
                                                 kv_budget_cap_fraction=prof.get("kv_budget_cap_fraction"))
                            # the same deployment sized from the KV pool: whole request, pool as vLLM reports it
                            gmu_v = variant.get("gpu_memory_utilization", fixed_c.get("gpu_memory_utilization"))
                            gmu_eff = (float(gmu_v) if gmu_v is not None
                                       else cat.serving["vllm"].get("gpu_memory_utilization_nvfp4", 0.7) if precision == "nvfp4"
                                       else cat.serving["vllm"]["gpu_memory_utilization"])
                            req_tok = request_tokens(fixed_c, variant, ctx)
                            f_pool = work.feasibility(
                                m, cat, acc, n_dev, precision, kv_precision, int(ctx) if isinstance(ctx, int) else 1024,
                                gpu_memory_utilization=float(gmu_v) if gmu_v is not None else None,
                                kv_budget_cap_fraction=prof.get("kv_budget_cap_fraction"),
                                kv_sizing="pool_tokens", request_tokens=req_tok, pool_margin=pool_margin,
                                pool_tokens=None if (neuron or tpu) else observed_pools.get(
                                    (m.key, acc, tp, precision, kv_precision, round(float(gmu_eff), 3))))
                            if kv_sizing == "pool_tokens":
                                f = f_pool
                            if not f.fits:
                                dropped.append({"model": m.key, "acc": acc, "cell": cell["id"],
                                                "variant": variant, "reason": f.reason})
                                continue
                            note = ""
                            b_max = f.b_max
                            kv_tokens_obs = None
                            obs = ((m.raw.get("neuron") or {}).get("kv_tokens_observed") or {}) if neuron else {}
                            if obs.get("tp") == tp and obs.get("tokens"):
                                base_frac = float(acc_cfg.get("kv_budget_cap_fraction") or 0.30)
                                frac = float(prof.get("kv_budget_cap_fraction") or base_frac)
                                kv_tokens_obs = int(obs["tokens"] * frac / base_frac)
                                req_tokens = work.kv_bytes_attended(m, cat, kv_precision, int(ctx) if isinstance(ctx, int) else 1024)
                                per_tok = work.kv_bytes_attended(m, cat, kv_precision, 1024) / 1024
                                b_max = max(1, int(kv_tokens_obs * per_tok // req_tokens)) if req_tokens else b_max
                            cap_v = (max(int(x) for x in prof["num_seqs_buckets"]) if prof.get("num_seqs_buckets")
                                     else seqs_cap)
                            if cap_v is not None and b_max > cap_v:
                                b_max = cap_v             # the server's largest compiled decode bucket
                            tb = variant.get("target_batch")
                            if neuron and isinstance(tb, int):
                                # the plugin caps concurrency at worst-case KV / max_model_len (4 Nov 2026), whatever
                                # the prompts: report it (template estimate), never resize on it (run ids stay stable)
                                if kv_tokens_obs is not None:
                                    policy, basis = kv_tokens_obs // served_ctx, "observed KV"
                                else:
                                    per_req = work.kv_bytes_attended(m, cat, kv_precision, served_ctx)
                                    policy = int(f.free_for_kv_gb * work.GB // per_req) if per_req > 0 else 0
                                    basis = "template estimate"
                                if tb > policy:
                                    note = f"worst-case KV cap ~{policy} at max_model_len {served_ctx} ({basis})"
                            if isinstance(tb, int) and tb > b_max:
                                if cell.get("no_resize"):
                                    # a diagnostic cell that deliberately targets beyond the template B_max
                                    # (c9b: the real pool is larger; where it ends is the measurement)
                                    note = f"target_batch {tb} above template B_max {b_max}; kept (no_resize)"
                                else:
                                    note = ((note + "; ") if note else "") + (f"target_batch {tb} resized to B_max {b_max}"
                                            + (" (largest compiled decode bucket)" if cap_v is not None and b_max == cap_v else "")
                                            + (f" (KV pool {f.kv_pool_tokens} tokens, {f.pool_basis}; request {req_tok} tokens)"
                                               if kv_sizing == "pool_tokens" and not (cap_v is not None and b_max == cap_v) else ""))
                                    variant["target_batch"] = b_max
                            tb_final = variant.get("target_batch")
                            if (kv_sizing != "pool_tokens" and isinstance(tb_final, int) and not (neuron or tpu)
                                    and tb_final > f_pool.b_max):
                                # planned under the old rule: say so, change nothing (run ids stay stable)
                                kv_pool_warnings.append({
                                    "model": m.key, "acc": acc, "tp": tp, "cell": cell["id"], "variant": dict(variant),
                                    "target_batch": tb_final, "b_max_pool": f_pool.b_max, "request_tokens": req_tok,
                                    "kv_pool_tokens": f_pool.kv_pool_tokens, "pool_basis": f_pool.pool_basis,
                                    "no_resize": bool(cell.get("no_resize"))})
                            vid = "_".join(f"{k}-{v}" for k, v in sorted(variant.items()))
                            base_id = f"{acc}/{m.key}/tp{tp}/{cell['id']}/{vid}"
                            if base_id in seen_bases:
                                # two grid points resized onto the same B_max: one run, noted
                                dropped.append({"model": m.key, "acc": acc, "cell": cell["id"],
                                                "variant": variant,
                                                "reason": f"duplicate after resize ({note or 'same variant'})"})
                                continue
                            seen_bases.add(base_id)
                            for r in range(1, repeats + 1):
                                run_id = f"{base_id}/r{r}"
                                runs.append(Run(run_id, m.key, acc, n_dev, tp, precision,
                                                kv_precision, cell["id"], cell["phase"],
                                                variant, r,
                                                _est_seconds(cell, variant, d_acc), note))
                            load_events.add((acc, m.key, tp, precision, variant.get("neuron_profile")))

    # Hours
    per_acc: dict[str, dict[str, float]] = {}
    for acc in accelerators:
        acc_runs = [r for r in runs if r.accelerator == acc]
        node_s = sum(r.est_seconds for r in acc_runs)
        loads = [e for e in load_events if e[0] == acc]
        load_s = float(cat.accelerators[acc].get("server_start_seconds", 300.0))
        node_s += len(loads) * load_s
        dev_of = {(e[1], e[2]): (neuron_chips(e[2], cat, acc) if backend_of(cat, acc) == "neuron" else e[2]) for e in loads}
        gpu_s = sum(r.est_seconds * r.n_gpus for r in acc_runs) + sum(load_s * dev_of[(e[1], e[2])] for e in loads)
        price = cat.accelerators[acc]["runpod_price_per_hr_usd"]
        per_acc[acc] = {
            "runs": len(acc_runs),
            "node_hours": round(node_s / 3600, 1),
            "gpu_hours": round(gpu_s / 3600, 1),
            "usd_at_list": round(gpu_s / 3600 * price, 0),
            "server_starts": len(loads),
        }
        if backend_of(cat, acc) == "neuron":
            host_rate = cat.accelerators[acc].get("capacity_block_usd_per_instance_hour")
            per_acc[acc]["usd_whole_host_at_block_rate"] = round(node_s / 3600 * host_rate, 0) if host_rate else None
        if backend_of(cat, acc) == "tpu":
            chips = int(cat.accelerators[acc].get("chips_per_host", 1))
            per_acc[acc]["slice_chips"] = chips
            per_acc[acc]["usd_whole_slice_on_demand"] = round(node_s / 3600 * chips * price, 0)

    return {
        "matrix_version": spec["matrix_version"],
        "templates_version": 1,
        "runs": [asdict(r) for r in runs],
        "dropped": dropped,
        "summary": per_acc,
        "planner": {"kv_sizing": kv_sizing, "pool_margin": pool_margin,
                    "observed_pools": bool(observed_pools)},
        "kv_pool_warnings": kv_pool_warnings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("manifest.json"))
    args = ap.parse_args()
    cat = load_catalogue()
    man = build_manifest(cat)
    args.out.write_text(json.dumps(man, indent=1))
    print(json.dumps(man["summary"], indent=2))
    print(f"runs: {len(man['runs'])}, dropped: {len(man['dropped'])}")
    if man["kv_pool_warnings"]:
        print(f"WARNING: {len(man['kv_pool_warnings'])} planned variants target more than the KV pool holds "
              "(cache eviction inside the window); see kv_pool_warnings, or plan with planner.kv_sizing: pool_tokens")


if __name__ == "__main__":
    main()
