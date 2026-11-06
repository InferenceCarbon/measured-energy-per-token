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


def _resolve_grid(value: Any, m, cat: Catalogue) -> list[Any]:
    if value == "context_grid":
        return [m.native_context if v == "native_max" else v
                for v in cat.serving.get("context_grid", [1024, 4096, 16384, "native_max"])]
    if isinstance(value, list):
        return [m.native_context if v == "native_max" else v for v in value]
    return [value]


def _tp_options(model_key: str, cat: Catalogue, accelerator: str, precision: str) -> list[int]:
    m = cat.model(model_key)
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
    if cell["id"] == "c0_static":
        window = 60.0
        warm = 0.0
    return idle + window + warm


def build_manifest(cat: Catalogue, cells_path: Path = CONFIG_DIR / "cells.yaml",
                   accelerators: tuple[str, ...] = ("h100", "b200")) -> dict[str, Any]:
    spec = yaml.safe_load(cells_path.read_text())
    defaults = spec["defaults"]
    cat.serving["context_grid"] = spec["context_grid"]

    runs: list[Run] = []
    dropped: list[dict[str, Any]] = []
    load_events: set[tuple[str, str, int, str]] = set()
    seen_bases: set[str] = set()

    for acc in accelerators:
        for m in cat.calibration_models():
            for cell in spec["cells"]:
                if cell.get("disabled"):
                    continue
                if "only_models" in cell and m.key not in cell["only_models"]:
                    continue
                if "only_accelerators" in cell and acc not in cell["only_accelerators"]:
                    continue
                repeats = cell.get("repeats", defaults["repeats"])
                precisions = cell["vary"].get("weight_precision", ["fp8"])
                precisions = [p for p in precisions if p in m.weight_precisions
                              and (p != "nvfp4" or cat.accelerators[acc]["nvfp4"])]
                if not precisions:
                    precisions = [m.weight_precisions[0]]

                for precision in precisions:
                    kv_precision = "fp8" if "fp8" in m.kv_precisions else m.kv_precisions[0]
                    tps = (_tp_options(m.key, cat, acc, precision)
                           if cell["vary"].get("tp") == "per_model"
                           else [work.min_gpus(m, cat, acc, precision) or 1])

                    vary = {k: v for k, v in cell["vary"].items()
                            if k not in ("weight_precision", "tp")}
                    keys = list(vary.keys())
                    grids = [_resolve_grid(vary[k], m, cat) for k in keys]

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
                            margin = 256 + math.ceil(0.04 * m.native_context)
                            if variant.get("context_cached") == m.native_context:
                                fresh = int(variant.get("fresh_in", fixed_c.get("fresh_in", 200)))
                                out = int(variant.get("out", fixed_c.get("out", 256)))
                                variant["context_cached"] = m.native_context - fresh - out - margin
                                ctx = variant["context_cached"]
                            if variant.get("fresh_in") == m.native_context:
                                out = int(variant.get("out", fixed_c.get("out", 1)))
                                variant["fresh_in"] = m.native_context - out - margin
                                ctx = variant["fresh_in"]
                            if cell["id"] == "c8_agent_replay":
                                ctx = min(cell["fixed"]["context_start"]
                                          + cell["fixed"]["steps"] * cell["fixed"]["context_step"],
                                          m.native_context - 1024)
                                variant["context_cap"] = ctx
                            if isinstance(ctx, int) and ctx > m.native_context:
                                dropped.append({"model": m.key, "acc": acc, "cell": cell["id"],
                                                "variant": variant,
                                                "reason": f"context {ctx} > native {m.native_context}"})
                                continue
                            f = work.feasibility(m, cat, acc, tp, precision, kv_precision,
                                                 int(ctx) if isinstance(ctx, int) else 1024,
                                                 gpu_memory_utilization=variant.get("gpu_memory_utilization"))
                            if not f.fits:
                                dropped.append({"model": m.key, "acc": acc, "cell": cell["id"],
                                                "variant": variant, "reason": f.reason})
                                continue
                            note = ""
                            tb = variant.get("target_batch")
                            if isinstance(tb, int) and tb > f.b_max:
                                if cell.get("no_resize"):
                                    # a diagnostic cell that deliberately targets beyond the template B_max
                                    # (c9b: the real pool is larger; where it ends is the measurement)
                                    note = f"target_batch {tb} above template B_max {f.b_max}; kept (no_resize)"
                                else:
                                    note = f"target_batch {tb} resized to B_max {f.b_max}"
                                    variant["target_batch"] = f.b_max
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
                                runs.append(Run(run_id, m.key, acc, tp, tp, precision,
                                                kv_precision, cell["id"], cell["phase"],
                                                variant, r,
                                                _est_seconds(cell, variant, defaults), note))
                            load_events.add((acc, m.key, tp, precision))

    # Hours
    per_acc: dict[str, dict[str, float]] = {}
    for acc in accelerators:
        acc_runs = [r for r in runs if r.accelerator == acc]
        node_s = sum(r.est_seconds for r in acc_runs)
        loads = [e for e in load_events if e[0] == acc]
        node_s += len(loads) * 300.0   # ~5 min per model load
        gpu_s = sum(r.est_seconds * r.n_gpus for r in acc_runs) + sum(300.0 * e[2] for e in loads)
        price = cat.accelerators[acc]["runpod_price_per_hr_usd"]
        per_acc[acc] = {
            "runs": len(acc_runs),
            "node_hours": round(node_s / 3600, 1),
            "gpu_hours": round(gpu_s / 3600, 1),
            "usd_at_list": round(gpu_s / 3600 * price, 0),
        }

    return {
        "matrix_version": spec["matrix_version"],
        "templates_version": 1,
        "runs": [asdict(r) for r in runs],
        "dropped": dropped,
        "summary": per_acc,
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


if __name__ == "__main__":
    main()
