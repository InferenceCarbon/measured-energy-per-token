"""Independent check of the v1.1 tables against the corpus, without analysis/build.py.

    python tools/audit_v1_1.py

Reads data/features.parquet directly, keeps only the readings in use (disposition `used` or `kept-marked`), recomputes
the central value of every row of the main per-row tables and compares it with the printed one (to the printed
precision). A reading left out of the tables that reached one would show here as a mismatch.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "paper" / "tables"
f = pd.read_parquet(ROOT / "data" / "features.parquet")
assert f.uid.is_unique
u = f[f.disposition.isin(["used", "kept-marked"])].copy()
assert u.dropped.isna().all() and f[~f.uid.isin(u.uid)].dropped.notna().all()
u["tb"] = pd.to_numeric(u.run_id.str.extract(r"target_batch-(\d+)")[0])
u.loc[(u.source == "v1.0.0") & (u.tb > 100), "tb"] = 100          # the version 1.0 client's connection limit
u["fresh"] = pd.to_numeric(u.run_id.str.extract(r"fresh_in-(\d+)")[0])
u["ctx"] = pd.to_numeric(u.run_id.str.extract(r"context_cached-(\d+)")[0])
LAB = {"qwen3-8b": "Qwen3-8B", "qwen3-30b-a3b": "Qwen3-30B-A3B", "deepseek-v2-lite": "DeepSeek-V2-Lite", "llama-3.3-70b": "Llama-3.3-70B",
       "gemma-3-27b": "Gemma-3-27B", "gpt-oss-120b": "gpt-oss-120b", "qwen3-235b-a22b": "Qwen3-235B-A22B", "llama-3.1-405b": "Llama-3.1-405B"}
u["Model"], u["Acc."] = u.model.map(LAB), u.accelerator.str.upper()


def central(cell: str) -> float:
    return float(str(cell).split(" [")[0].split(" (")[0].replace(",", ""))


def agree(printed: float, value: float) -> bool:
    return abs(value / printed - 1) < 6e-3 if printed else abs(value) < 1e-9


bad = total = 0
def check(table, series, exp, keys, metric, col, where=None):
    global bad, total
    t = pd.read_csv(T / f"{table}.csv", dtype=str)
    d = u[(u.series == series) & (u.experiment == exp) & u[metric].notna()]
    if where is not None:
        d = d[where(d)]
    n = 0
    for _, r in t.iterrows():
        g = d
        for tc, fc in keys.items():
            g = g[g[fc].astype(float).round(6) == float(str(r[tc]).replace(",", ""))] if fc not in ("Model", "Acc.") else g[g[fc] == r[tc]]
        if "runs differ" in str(r[col]):
            continue
        n += 1
        twins = t[(t[list(keys)] == r[list(keys)]).all(axis=1)]
        if len(twins) > 1:    # the same shape reached from two design targets: two rows share these keys, so check the pooled mean lies between them
            vals = [float(v) for x in twins[col] for v in (str(x).split(" (")[0].replace(" to ", " / ").split(" / ") if "runs differ" in str(x) else [central(x)])]
            ok = len(g) > 0 and min(vals) * (1 - 6e-3) <= g[metric].mean() <= max(vals) * (1 + 6e-3)
        else:
            ok = len(g) > 0 and agree(central(r[col]), g[metric].mean())
        if not ok:
            bad += 1
            print("  MISMATCH", table, dict(r[list(keys)]), "printed", r[col], "recomputed", round(g[metric].mean(), 5) if len(g) else "no readings")
    total += n
    print(f"{table:24s} {n:4d} rows checked")


dec = lambda d: d.phase == "decode"
for series, tag in (("main", ""), ("size", "size_")):
    check(tag + "decode_batch", series, "c1_decode_batch", {"Model": "Model", "Acc.": "Acc.", "TP": "tp", "Target B": "tb"}, "e_d_j_per_tok", "J per output token", dec)
    check(tag + "decode_context", series, "c2_decode_context", {"Model": "Model", "Acc.": "Acc.", "TP": "tp", "Target as run": "tb", "Cached context (tokens)": "ctx"}, "e_d_j_per_tok", "J per output token", dec)
    check(tag + "prefill", series, "c3_prefill_miss", {"Model": "Model", "Acc.": "Acc.", "TP": "tp", "Target B": "tb", "Prompt tokens": "fresh"}, "e_p_j_per_tok", "J per prompt token")
    check(tag + "idle_power", series, "c0_static", {"Model": "Model", "Accelerator": "Acc.", "GPUs": "tp"}, "mean_power_w", "Board W, all GPUs")
# the two sides of the ratio tables, the agent table and the per-run KV table
u["ctl"] = u.run_id.str.extract(r"control-([a-z]+)_")[0]
u["pc"] = pd.to_numeric(u.run_id.str.extract(r"prefix_cached-(\d+)")[0])
u["spec"] = u.run_id.str.extract(r"speculation-([a-z0-9]+)")[0]
for side, col in (("cached", "Cached, J per request"), ("uncached", "Uncached, J per request")):
    check("cached_prefix", "main", "c4_cached_prefix", {"Model": "Model", "Acc.": "Acc.", "Cached prefix": "pc", "Fresh in": "fresh", "Target B": "tb"}, "e_per_request_j", col, lambda d, side=side: d.ctl == side)
for side, col in (("none", "Plain, J per output token"), ("ngram3", "Speculative, J per output token")):
    check("speculation", "main", "c10_speculation", {"Model": "Model", "Acc.": "Acc.", "TP": "tp", "Target B": "tb"}, "e_d_j_per_tok", col, lambda d, side=side: d.spec == side)
u["kj_per_agent"] = u.energy_j / u.tb / 1e3
check("agent_replay", "main", "c8_agent_replay", {"Model": "Model", "Acc.": "Acc.", "TP": "tp", "Agents": "tb"}, "kj_per_agent", "kJ per agent")
kv_t = pd.read_csv(T / "kv_pressure.csv", dtype=str)
kv_c = u[u.experiment.isin(["c9_kv_pressure", "c9b_kv_pressure_diag"]) & u.e_d_j_per_tok.notna()]
ok = len(kv_t) == len(kv_c) and all(agree(a, b) for a, b in zip(sorted(central(x) for x in kv_t["J per output token"]), sorted(kv_c.e_d_j_per_tok)))
total += len(kv_t); bad += 0 if ok else 1
print(f"{'kv_pressure':24s} {len(kv_t):4d} rows checked" + ("" if ok else "  MISMATCH"))
# corpus counts
h = __import__("json").loads((T / "headline.json").read_text())
assert h["runs"] == len(f) and h["usable_runs"] == len(u[u.energy_j.notna()]), (h["runs"], h["usable_runs"], len(f), len(u))
lo = pd.read_csv(ROOT / "data" / "readings_left_out.csv", dtype=str)
assert (lo.in_features_file == "yes").sum() == len(f) - len(u)
print(f"corpus: {len(f)} readings, {len(u)} in use, {len(f) - len(u)} left out in the file, {len(lo)} left out in all")
print(f"{total} rows checked, {bad} mismatches")
sys.exit(1 if bad else 0)
