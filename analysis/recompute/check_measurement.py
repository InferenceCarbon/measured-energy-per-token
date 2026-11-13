"""Independent recompute of eight published tables of the measurement paper from features.parquet
(and, for the KV-pressure scheduler columns and the agent script, store_tables/cells.parquet).

Rules applied (from the table captions, build.py lines 1-25 and the paper text):
  central  mean of per-pod means; per-pod mean = mean of that pod's runs
  band     two pods: range of per-pod means; one pod: range of the repeats; one run: "(n=1)"
  ratio    pairing 'pod': ratio per pod, central = mean of per-pod ratios, band = their range
           pairing 'repeat': both sides on one pod, ratio per repeat, central = mean, band = range
           pairing 'none': ratio of centrals, band = [lo_num/hi_den, hi_num/lo_den] from the two sides' bands
  rated    700 W per H100, 1,000 W per B200, times the GPUs serving the model
Match rule: my value rounds to the printed value at the printed precision, or differs from it by one unit
in the last printed digit. Counts (pods, TP, agents, steps, targets, preemptions) must match exactly.
Shares no code with analysis/build.py.  Usage: python analysis/recompute/check_measurement.py [REPO]
"""
import json, re, sys
import numpy as np, pandas as pd

import os
REPO = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RATED = {"h100": 700.0, "b200": 1000.0}
NAME = {"deepseek-v2-lite": "DeepSeek-V2-Lite", "llama-3.3-70b": "Llama-3.3-70B", "qwen3-30b-a3b": "Qwen3-30B-A3B", "qwen3-8b": "Qwen3-8B"}
ACC = {"h100": "H100", "b200": "B200"}

# ----------------------------------------------------------------------------- data
raw = pd.read_parquet(f"{REPO}/data/features.parquet")
f = raw[raw["dropped"].isna()].copy()
print(f"features: {len(raw)} rows, dropped excluded: {int(raw['dropped'].notna().sum())}, corpus: {len(f)}")
# Version 1.0.1 cache-reuse rule (Appendix C of the paper), applied here independently of analysis/build.py:
# a decode-type run (c1, c2, c6, c7) whose in-window prefix-cache reuse was below 0.95 of the expected is excluded.
import os as _os
_cr = _os.path.join(REPO, "data", "cache_reuse.csv")
if _os.path.exists(_cr):
    _r = pd.read_csv(_cr)[["run_id", "cache_reuse_r"]]
    _rr = dict(zip(_r.run_id, _r.cache_reuse_r))
    _ev = f.run_id.map(_rr).lt(0.95) & f.cell.isin(["c1_decode_batch", "c2_decode_context", "c6_tensor_parallel", "c7_precision"])
    print(f"cache-reuse rule: {int(_ev.sum())} runs excluded")
    f = f[~_ev].copy()

cells = pd.read_parquet(f"{REPO}/data/store_tables/cells.parquet")
act = cells[cells.kind == "active"].set_index("run_id")


def v(key, default=np.nan):
    """Design setting from the run id's variant segment."""
    return f["run_id"].str.extract(rf"(?:^|[_/]){key}-([\d.]+)")[0].astype(float).fillna(default)


f["tb"] = v("target_batch")
f["ctx"] = v("context_cached")
f["fresh"] = v("fresh_in")
f["pref"] = v("prefix_cached")
f["cap"] = v("context_cap")
f["offer"] = v("gpu_memory_utilization", 0.9)
f["e_tok"] = f.energy_j / f.completion_tokens          # J per output token
f["e_prompt"] = f.energy_j / f.prompt_tokens           # J per prompt token
f["e_req"] = f.energy_j / f.n_requests                 # J per request
f["tps"] = f.completion_tokens / f.duration_s
f["ptps"] = f.prompt_tokens / f.duration_s
f["W"] = f.mean_power_w                              # raw column; differs from energy_j/duration_s by up to 0.6 W (see report)
f["rated"] = f.accelerator.map(RATED) * f.n_gpus

# ----------------------------------------------------------------------------- helpers (parsing and matching, after indep_check.py)
def nums(s):
    """Printed numbers in a cell, with the decimals each is printed to. '(n=1)' and '%' are ignored."""
    s = re.sub(r"\(n=\d+\)", "", str(s))
    out = []
    for t in re.findall(r"\d[\d,]*\.\d+|\d[\d,]*", s):
        d = len(t.split(".")[1]) if "." in t else 0
        out.append((float(t.replace(",", "")), d))
    return out


def close(pv, d, m):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return False
    return abs(round(m, d) - pv) <= 10 ** (-d) + 1e-9


def fmt(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.6g}"


class Table:
    def __init__(self, name):
        self.name, self.rows, self.fields, self.bad = name, 0, 0, 0

    def check(self, rowkey, field, printed, mine, exact=False):
        """mine: list of my values in the printed order (central, lo, hi ...) or a string for text fields."""
        self.fields += 1
        if isinstance(mine, str):
            ok = str(printed).strip() == mine
            if not ok:
                self.bad += 1; print(f"MISMATCH {self.name} | {rowkey} | {field}: published '{printed}' mine '{mine}'")
            return ok
        pv = nums(printed)
        mine = [None if m is None else float(m) for m in mine]
        if len(pv) != len(mine):
            ok = False
        elif exact:
            ok = all(abs(p - m) < 1e-9 for (p, _), m in zip(pv, mine))
        else:
            ok = all(close(p, d, m) for (p, d), m in zip(pv, mine))
        if not ok:
            self.bad += 1
            print(f"MISMATCH {self.name} | {rowkey} | {field}: published '{printed}' mine [{', '.join(fmt(m) for m in mine)}]")
        return ok

    def row(self):
        self.rows += 1

    def done(self, unmatched=0):
        print(f"{self.name}: {self.rows} rows checked, {self.fields} fields, {self.bad} mismatches" + (f" ({unmatched} published rows with no computed match)" if unmatched else ""))


def agg(g, col):
    """central, lo, hi, n_pods, n_runs for one configuration's runs (per the band rule)."""
    pod = g.groupby("pod_id")[col].mean()
    c = pod.mean()
    if len(pod) >= 2:
        lo, hi = pod.min(), pod.max()
    else:
        lo, hi = g[col].min(), g[col].max()
    return c, lo, hi, len(pod), len(g)


def ratio(gn, gd, col):
    """Ratio of side gn over gd: (central, lo, hi, pairing) under the pairing rule decided by the pods."""
    pn, pd_ = set(gn.pod_id), set(gd.pod_id)
    if pn == pd_ and len(pn) >= 2:
        r = gn.groupby("pod_id")[col].mean() / gd.groupby("pod_id")[col].mean()
        return r.mean(), r.min(), r.max(), "pod"
    if pn == pd_ and len(pn) == 1 and len(gn) > 1 and len(gd) > 1:
        r = gn.set_index("repeat")[col] / gd.set_index("repeat")[col]
        return r.mean(), r.min(), r.max(), "repeat"
    cn, lon, hin, _, _ = agg(gn, col)
    cd, lod, hid, _, _ = agg(gd, col)
    return cn / cd, lon / hid, hin / lod, "none"


def pub(name):
    return pd.read_csv(f"{REPO}/paper/tables/{name}.csv", dtype=str)


def num(s):
    return float(str(s).replace(",", ""))


def band_fields(T, key, field, printed, c, lo, hi, n_runs):
    if n_runs == 1:
        T.check(key, field, printed, [c])
    else:
        T.check(key, field, printed, [c, lo, hi])


# ============================================================================= decode_context (c2)
T = Table("decode_context")
c2 = f[f.cell == "c2_decode_context"].copy()
c2["design"] = np.where(c2.tb <= 8, 8, 64)
P = pub("decode_context")
unmatched = 0
for _, r in P.iterrows():
    key = f"{r.Model} {r['Acc.']} TP{r.TP} design {r['Design target B']} run {r['Target as run']} ctx {r['Cached context (tokens)']}"
    g = c2[(c2.model.map(NAME) == r.Model) & (c2.accelerator.map(ACC) == r["Acc."]) & (c2.tp == int(r.TP))
           & (c2.tb == num(r["Target as run"])) & (c2.ctx == num(r["Cached context (tokens)"]))]
    if len(g) == 0:
        unmatched += 1; print(f"NO DATA {T.name} | {key}"); continue
    T.row()
    T.check(key, "Design target B", r["Design target B"], [g.design.iloc[0]], exact=True)
    T.check(key, "Attended context", r["Attended context (tokens)"], [num(r["Cached context (tokens)"]) + 200 + 128], exact=True)
    c, lo, hi, npods, nruns = agg(g, "e_tok")
    # collapse flags (Section 4.10 / caption)
    flag = "no"
    if nruns == 2 and g.e_tok.max() / g.e_tok.min() > 1.5:
        flag = "regimes differ"
    elif g.design.iloc[0] == 64:
        twin = c2[(c2.model == g.model.iloc[0]) & (c2.accelerator == g.accelerator.iloc[0]) & (c2.tp == g.tp.iloc[0]) & (c2.design == 8) & (c2.ctx == g.ctx.iloc[0])]
        if len(twin) and c > agg(twin, "e_tok")[0]:
            flag = "above batch-8 figure"
    # v1.0.1: the Collapse column is gone (no flagged row survives the cache-reuse rule); a flagged row would list
    # its two runs in the J cell, marked "(runs differ)"
    T.check(key, "flag marker in J cell", "runs differ" if "(runs differ)" in str(r["J per output token"]) else "no",
            "no" if flag == "no" else "runs differ")
    cw, low, hiw, _, _ = agg(g, "W")
    band_fields(T, key, "Board W", r["Board W"], cw, low, hiw, nruns)
    T.check(key, "% of rated", r["% of rated"], [100 * cw / g.rated.iloc[0]])
    if flag != "no":
        T.check(key, "J per output token (two runs, sorted)", r["J per output token"], sorted(g.e_tok))
    else:
        band_fields(T, key, "J per output token", r["J per output token"], c, lo, hi, nruns)
    T.check(key, "Achieved B", r["Achieved B"], [agg(g, "B")[0]])
    T.check(key, "Tokens per second", r["Tokens per second"], [g.tps.mean()])
    T.check(key, "Pods", r.Pods, [npods], exact=True)
# every corpus c2 configuration should be in the table
cfgs = c2.groupby(["model", "accelerator", "tp", "tb", "ctx"]).size()
print(f"  c2 configurations in corpus: {len(cfgs)}, published rows: {len(P)}")
T.done(unmatched)

# ============================================================================= prefill (c3)
T = Table("prefill")
c3 = f[(f.cell == "c3_prefill_miss") & (f.fresh != 32768) & (f.prompt_tokens > 0)]   # 32,768-token runs appear in no table (Appendix A);
# five Llama-3.3-70B H100 runs served no prompt tokens at all (every request refused), so J per prompt token is undefined for them
print(f"  c3 runs with no prompt tokens (excluded): {int(((f.cell == 'c3_prefill_miss') & ~(f.prompt_tokens > 0)).sum())}")
alt_mean = 0
P = pub("prefill")
unmatched = 0
for _, r in P.iterrows():
    key = f"{r.Model} {r['Acc.']} TP{r.TP} B{r['Target B']} prompt {r['Prompt tokens']}"
    g = c3[(c3.model.map(NAME) == r.Model) & (c3.accelerator.map(ACC) == r["Acc."]) & (c3.tp == int(r.TP))
           & (c3.tb == num(r["Target B"])) & (c3.fresh == num(r["Prompt tokens"]))]
    if len(g) == 0:
        unmatched += 1; print(f"NO DATA {T.name} | {key}"); continue
    T.row()
    c, lo, hi, npods, nruns = agg(g, "e_prompt")
    band_fields(T, key, "J per prompt token", r["J per prompt token"], c, lo, hi, nruns)
    # pooled rate (all prompt tokens over all window seconds); the caption's "mean over runs" read literally is tested as the alternative
    ok = T.check(key, "Prompt tokens per second", r["Prompt tokens per second"], [g.prompt_tokens.sum() / g.duration_s.sum()])
    pv = nums(r["Prompt tokens per second"])[0]
    if close(pv[0], pv[1], g.ptps.mean()):
        alt_mean += 1
    cw, low, hiw, _, _ = agg(g, "W")
    band_fields(T, key, "Board W", r["Board W"], cw, low, hiw, nruns)
    T.check(key, "% of rated", r["% of rated"], [100 * cw / g.rated.iloc[0]])
    T.check(key, "Pods", r.Pods, [npods], exact=True)
print(f"  c3 configurations in corpus (excl. 32,768): {len(c3.groupby(['model','accelerator','tp','tb','fresh']))}, published rows: {len(P)}")
print(f"  prompt tokens per second: rows where the plain mean over runs would also match the printed value: {alt_mean} of {len(P)}")
T.done(unmatched)

# ============================================================================= cached_prefix (c4)
T = Table("cached_prefix")
c4 = f[f.cell == "c4_cached_prefix"]
P = pub("cached_prefix")
unmatched = 0
alt_hits = 0
for _, r in P.iterrows():
    key = f"{r.Model} {r['Acc.']} prefix {r['Cached prefix']} fresh {r['Fresh in']} B{r['Target B']}"
    g = c4[(c4.model.map(NAME) == r.Model) & (c4.accelerator.map(ACC) == r["Acc."]) & (c4.pref == num(r["Cached prefix"]))
           & (c4.fresh == num(r["Fresh in"])) & (c4.tb == num(r["Target B"]))]
    gc, gu = g[g.control == "cached"], g[g.control == "uncached"]
    if len(gc) == 0 or len(gu) == 0:
        unmatched += 1; print(f"NO DATA {T.name} | {key}"); continue
    T.row()
    cc, loc, hic, npc, nrc = agg(gc, "e_req")
    cu, lou, hiu, npu, nru = agg(gu, "e_req")
    band_fields(T, key, "Cached, J per request", r["Cached, J per request"], cc, loc, hic, nrc)
    band_fields(T, key, "Uncached, J per request", r["Uncached, J per request"], cu, lou, hiu, nru)
    rc, rlo, rhi, pairing = ratio(gc, gu, "e_req")
    ok = T.check(key, "Cached / uncached", r["Cached / uncached"], [rc, rlo, rhi])
    if not ok and pairing == "pod":
        pv = nums(r["Cached / uncached"])
        if close(pv[0][0], pv[0][1], cc / cu):
            alt_hits += 1; print(f"    (central would match as ratio of centrals: {cc/cu:.5g})")
    T.check(key, "Pods", r.Pods, [npc, npu], exact=True)
    T.check(key, "Pairing", r.Pairing, pairing)
print(f"  c4 configurations in corpus: {len(c4.groupby(['model','accelerator','pref','fresh','tb']))}, published rows: {len(P)}")
T.done(unmatched)

# ============================================================================= b200_vs_h100 (c1, c6)
T = Table("b200_vs_h100")
P = pub("b200_vs_h100")
unmatched = 0
for _, r in P.iterrows():
    key = f"{r.Comparison} {r.Model} B{r['Target B']} GPUs {r['GPUs B200 / H100']} {r['Exp.']}"
    gb_, gh_ = [int(x) for x in r["GPUs B200 / H100"].split("/")]
    cell = {"c1": "c1_decode_batch", "c6": "c6_tensor_parallel"}[r["Exp."]]
    d = f[(f.cell == cell) & (f.model.map(NAME) == r.Model) & (f.tb == num(r["Target B"]))]
    gb = d[(d.accelerator == "b200") & (d.n_gpus == gb_)]
    gh = d[(d.accelerator == "h100") & (d.n_gpus == gh_)]
    if len(gb) == 0 or len(gh) == 0:
        unmatched += 1; print(f"NO DATA {T.name} | {key}"); continue
    T.row()
    T.check(key, "Comparison", r.Comparison, "same GPUs" if gb_ == gh_ else "fewer B200s")
    rc, rlo, rhi, pairing = ratio(gb, gh, "e_tok")
    if pairing != "none":
        print(f"  note {key}: pods shared across accelerators? pairing={pairing}")
    T.check(key, "B200 / H100, J per output token", r["B200 / H100, J per output token"], [rc, rlo, rhi])
    T.check(key, "Pods B200 / H100", r["Pods B200 / H100"], [gb.pod_id.nunique(), gh.pod_id.nunique()], exact=True)
T.done(unmatched)

# ============================================================================= tensor_parallel (c6, c1)
T = Table("tensor_parallel")
P = pub("tensor_parallel")
unmatched = 0
for _, r in P.iterrows():
    key = f"{r.Model} {r['Acc.']} B{r['Target B']} TP {r['TP pair']}"
    lo_tp, hi_tp = [int(x) for x in r["TP pair"].split(" to ")]
    src_hi, src_lo = [s.strip() for s in r["Sides from"].split("/")]   # printed as 'lower-side exp / higher-side exp'? caption: 'c1 / c6' = TP1 side from c1
    d = f[(f.model.map(NAME) == r.Model) & (f.accelerator.map(ACC) == r["Acc."]) & (f.tb == num(r["Target B"]))]
    g_lo = d[(d.tp == lo_tp) & (d.cell == {"c1": "c1_decode_batch", "c6": "c6_tensor_parallel"}[src_hi])]
    g_hi = d[(d.tp == hi_tp) & (d.cell == {"c1": "c1_decode_batch", "c6": "c6_tensor_parallel"}[src_lo])]
    if len(g_lo) == 0 or len(g_hi) == 0:
        unmatched += 1; print(f"NO DATA {T.name} | {key}"); continue
    T.row()
    cl, lol, hil, npl, nrl = agg(g_lo, "e_tok")
    ch, loh, hih, nph, nrh = agg(g_hi, "e_tok")
    band_fields(T, key, "Lower TP, J per output token", r["Lower TP, J per output token"], cl, lol, hil, nrl)
    band_fields(T, key, "Higher TP, J per output token", r["Higher TP, J per output token"], ch, loh, hih, nrh)
    rc, rlo, rhi, pairing = ratio(g_hi, g_lo, "e_tok")
    T.check(key, "Ratio", r.Ratio, [rc, rlo, rhi])
    T.check(key, "Throughput ratio", r["Throughput ratio"], [g_hi.tps.mean() / g_lo.tps.mean()])
    T.check(key, "Pods higher / lower", r["Pods higher / lower"], [nph, npl], exact=True)
    T.check(key, "Pairing", r.Pairing, pairing)
    # 'Sides from': the TP1 side of Llama B200 is c1, everything else c6
    expect = "c1 / c6" if (r.Model == "Llama-3.3-70B" and r["Acc."] == "B200" and lo_tp == 1) else "c6 / c6"
    T.check(key, "Sides from", r["Sides from"], expect)
T.done(unmatched)

# ============================================================================= precision (c7)
T = Table("precision")
c7 = f[f.cell == "c7_precision"].copy()
c7["design"] = np.where(c7.tb <= 8, 8, 64)
P = pub("precision")
unmatched = 0
for _, r in P.iterrows():
    key = f"{r.Model} design {r['Design target B']} ctx {r['Cached context (tokens)']}"
    d = c7[(c7.model.map(NAME) == r.Model) & (c7.design == num(r["Design target B"])) & (c7.ctx == num(r["Cached context (tokens)"]))]
    g8, g4 = d[d.weight_precision == "fp8"], d[d.weight_precision == "nvfp4"]
    if len(g8) == 0 or len(g4) == 0:
        unmatched += 1; print(f"NO DATA {T.name} | {key}"); continue
    T.row()
    T.check(key, "Target as run fp8 / nvfp4", r["Target as run fp8 / nvfp4"], [g8.tb.iloc[0], g4.tb.iloc[0]], exact=True)
    T.check(key, "Achieved B fp8 / nvfp4", r["Achieved B fp8 / nvfp4"], [agg(g8, "B")[0], agg(g4, "B")[0]])
    c8_, lo8, hi8, np8, nr8 = agg(g8, "e_tok")
    c4_, lo4, hi4, np4, nr4 = agg(g4, "e_tok")
    band_fields(T, key, "fp8, J per output token", r["fp8, J per output token"], c8_, lo8, hi8, nr8)
    band_fields(T, key, "nvfp4, J per output token", r["nvfp4, J per output token"], c4_, lo4, hi4, nr4)
    rc, rlo, rhi, pairing = ratio(g4, g8, "e_tok")
    T.check(key, "nvfp4 / fp8", r["nvfp4 / fp8"], [rc, rlo, rhi])
    T.check(key, "Throughput ratio", r["Throughput ratio"], [g4.tps.mean() / g8.tps.mean()])
    T.check(key, "Pods nvfp4 / fp8", r["Pods nvfp4 / fp8"], [np4, np8], exact=True)
    T.check(key, "Pairing", r.Pairing, pairing)
T.done(unmatched)

# ============================================================================= kv_pressure (c9, c9b): one row per run
T = Table("kv_pressure")
k = f[f.cell.isin(["c9_kv_pressure", "c9b_kv_pressure_diag"])].copy()
k["exp"] = k.cell.map({"c9_kv_pressure": "c9", "c9b_kv_pressure_diag": "c9b"})
k = k.join(act[["sched_kv_usage_mean", "sched_waiting_mean", "sched_preemptions_delta", "sched_running_mean"]], on="run_id")
idle = cells[cells.kind.isin(["idle_before", "idle_after"])].pivot_table(index="run_id", columns="kind", values="mean_power_w_total")
k["idle_drift_pct"] = 100 * (idle.idle_after - idle.idle_before).abs().div(idle[["idle_before", "idle_after"]].min(axis=1)).reindex(k.run_id).values
k["used"] = False
P = pub("kv_pressure")
unmatched = 0
for _, r in P.iterrows():
    key = f"{r.Experiment} offer {r['GPU memory offer']} target {r['Target B']} achieved {r['Achieved B']}"
    cand = k[(k.exp == r.Experiment) & (k.offer == num(r["GPU memory offer"])) & (k.tb == num(r["Target B"])) & ~k.used]
    if len(cand) == 0:
        unmatched += 1; print(f"NO DATA {T.name} | {key}"); continue
    i = (cand.B - num(r["Achieved B"])).abs().idxmin()
    k.loc[i, "used"] = True
    g = k.loc[i]
    T.row()
    T.check(key, "Achieved B", r["Achieved B"], [g.B])
    T.check(key, "Achieved B (= sched_running_mean)", r["Achieved B"], [g.sched_running_mean])
    T.check(key, "J per output token", r["J per output token"], [g.e_tok])
    T.check(key, "Tokens per second", r["Tokens per second"], [g.tps])
    T.check(key, "Board W", r["Board W"], [g.W])
    T.check(key, "% of rated", r["% of rated"], [100 * g.W / g.rated])
    T.check(key, "Mean KV use", r["Mean KV use"], [100 * g.sched_kv_usage_mean])
    T.check(key, "Mean waiting", r["Mean waiting"], [g.sched_waiting_mean])
    if np.isnan(g.sched_preemptions_delta):
        T.check(key, "Preemptions", r.Preemptions, "not recorded")
    else:
        T.check(key, "Preemptions", r.Preemptions, [g.sched_preemptions_delta], exact=True)
    T.check(key, "Collapsed", r.Collapsed, "yes" if g.e_tok > 3 else "no")
    drift = "yes" if g.idle_drift_pct > 5 else "no"
    ok = T.check(key, "Idle-drift flag", r["Idle-drift flag"], drift)
    if not ok:
        print(f"    idle before/after differ by {g.idle_drift_pct:.1f}%; 'idle drift' in flags: {'idle drift' in str(g['flags'])}; "
              f"harness flags: '{g['flags']}'")
print(f"  c9+c9b runs in corpus: {len(k)}, published rows: {len(P)}, runs used: {int(k.used.sum())}")
T.done(unmatched)

# ============================================================================= agent_replay (c8)
T = Table("agent_replay")
c8 = f[(f.cell == "c8_agent_replay") & (f.cap != 32768)].copy()   # see report: build.py line 630 dependence
scripts = act.loc[c8.run_id, "replay_script"].map(json.loads)
c8["steps"] = scripts.map(len).values
c8["last_ctx"] = scripts.map(lambda s: s[-1]["context_tokens"]).values
c8["agents"] = c8.tb
c8["e_step"] = c8.energy_j / (c8.agents * c8.steps)
c8["kj_agent"] = c8.energy_j / c8.agents / 1000
P = pub("agent_replay")
unmatched = 0
for _, r in P.iterrows():
    key = f"{r.Model} {r['Acc.']} TP{r.TP} agents {r.Agents}"
    g = c8[(c8.model.map(NAME) == r.Model) & (c8.accelerator.map(ACC) == r["Acc."]) & (c8.tp == int(r.TP)) & (c8.agents == num(r.Agents))]
    if len(g) == 0:
        unmatched += 1; print(f"NO DATA {T.name} | {key}"); continue
    T.row()
    T.check(key, "Steps", r.Steps, [g.steps.iloc[0]], exact=True)
    T.check(key, "Context at last step", r["Context at last step (tokens)"], [g.last_ctx.iloc[0]], exact=True)
    for col, name in [("e_step", "J per step"), ("e_tok", "J per output token"), ("kj_agent", "kJ per agent")]:
        c, lo, hi, npods, nruns = agg(g, col)
        band_fields(T, key, name, r[name], c, lo, hi, nruns)
    T.check(key, "Wall seconds", r["Wall seconds"], [g.duration_s.mean()])
    T.check(key, "Pods", r.Pods, [g.pod_id.nunique()], exact=True)
print(f"  c8 runs in corpus: {int((f.cell == 'c8_agent_replay').sum())}, with 32,768 cap excluded: {len(c8)}, published rows: {len(P)}")
T.done(unmatched)

# ----------------------------------------------------------------------------- cross-check against the precomputed per-token columns (not an input)
chk = f[["e_tok", "e_d_j_per_tok"]].dropna()
print(f"cross-check e_tok vs e_d_j_per_tok: max rel diff {((chk.e_tok - chk.e_d_j_per_tok).abs() / chk.e_d_j_per_tok).max():.2e} over {len(chk)} runs")
chk = f[["e_prompt", "e_p_j_per_tok"]].dropna()
print(f"cross-check e_prompt vs e_p_j_per_tok: max rel diff {((chk.e_prompt - chk.e_p_j_per_tok).abs() / chk.e_p_j_per_tok).max():.2e} over {len(chk)} runs")
chk = f[["e_req", "e_per_request_j"]].dropna()
print(f"cross-check e_req vs e_per_request_j: max rel diff {((chk.e_req - chk.e_per_request_j).abs() / chk.e_per_request_j).max():.2e} over {len(chk)} runs")
