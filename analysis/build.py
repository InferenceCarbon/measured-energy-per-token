"""Paper 1 (measurement only): tables and figures from the frozen corpus.

    python analysis/build.py [--features PATH]

Reads `features.parquet` (one row per metered run; the 29 Oct 2026 freeze-time
table) and writes

    paper/tables/*.tex, *.md, *.csv     every table in the paper
    paper/figs/*.pdf, *.png             every figure in the paper
    paper/tables/headline.json          the numbers quoted in the text

Every central value carries a band and the pod and run counts behind it:

  band        the range of the runs behind the figure. No configuration ran on
              more than two pods: with two pods the band is the range of the
              per-pod means, with one pod the range of the repeats. A ratio is
              paired by pod (range of the per-pod ratios) where it can be,
              otherwise lowest-over-highest to highest-over-lowest.
  counts      the Pods column and "(n=1)" in the band say what each figure rests on.
              (An internal `conf` field survives for ordering; it is not printed:
              the paper does not use confidence labels, which the wider
              InferenceCarbon programme reserves for its GHG-accounting figures.)

Nothing here fits a model. No estimator, no closed models, no carbon.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_FEATURES = ROOT / "data" / "features.parquet"
TABLES = ROOT / "paper" / "tables"
FIGS = ROOT / "paper" / "figs"

MODEL_LABEL = {"qwen3-8b": "Qwen3-8B", "qwen3-30b-a3b": "Qwen3-30B-A3B",
               "deepseek-v2-lite": "DeepSeek-V2-Lite", "llama-3.3-70b": "Llama-3.3-70B"}
MODEL_ORDER = ["qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite", "llama-3.3-70b"]
ACC_LABEL = {"h100": "H100", "b200": "B200"}
# Okabe-Ito, validated with the dataviz palette checker (CVD dE >= 11, normal >= 24).
COLOR = {"qwen3-8b": "#0072B2", "qwen3-30b-a3b": "#E69F00", "deepseek-v2-lite": "#009E73",
         "llama-3.3-70b": "#D55E00"}
MARKER = {"qwen3-8b": "o", "qwen3-30b-a3b": "s", "deepseek-v2-lite": "^", "llama-3.3-70b": "D"}
VARIANT_KEYS = ("target_batch", "context_cached", "context_cap", "fresh_in", "prefix_cached", "control",
                "weight_precision", "clock_mode", "gpu_memory_utilization")
_VAR_RE = re.compile("(" + "|".join(VARIANT_KEYS) + r")-([^_/]+)")
# Design grids of the varied batch per cell (config/cells.yaml). A target as run that is not on
# the grid was resized DOWN by the feasibility generator, so its design point is the smallest
# grid value at or above it.
# v1.0.1 cache-reuse rule (data/cache_reuse.csv): a decode-type run keeps its stated cached-context shape only if the
# in-window prefix-cache hit share was at least this fraction of the expected one. The cut sits in an empty gap of the
# observed distribution (no run between 0.85 and 0.97). c9 and c9b are exempt: eviction is what they measure.
CACHE_REUSE_MIN = 0.95
CACHE_RULE_CELLS = ("c1_decode_batch", "c2_decode_context", "c6_tensor_parallel", "c7_precision")
DESIGN_BATCH = {"c1_decode_batch": [1, 4, 16, 64, 256], "c2_decode_context": [8, 64], "c3_prefill_miss": [1, 8],
                "c4_cached_prefix": [1, 8], "c6_tensor_parallel": [1, 16, 64], "c7_precision": [8, 64],
                "c8_agent_replay": [1, 8], "c9_kv_pressure": [12, 16, 20, 24, 28, 32], "c9b_kv_pressure_diag": [12, 20, 28, 32]}


def design_target(cell: str, tb: float) -> int:
    grid = DESIGN_BATCH.get(cell, [])
    for g in grid:
        if tb <= g:
            return g
    return int(tb)
REPLAY_OUT_PER_STEP = 300
REPLAY_CONTEXT_START, REPLAY_CONTEXT_STEP = 10240, 2048
NATIVE_CONTEXT = {"qwen3-8b": 32768, "qwen3-30b-a3b": 32768, "deepseek-v2-lite": 32768, "llama-3.3-70b": 131072}
RATED_W = {"h100": 700.0, "b200": 1000.0}
# HBM bandwidth per GPU from NVIDIA's product pages (H100 SXM 3.35 TB/s, B200 8 TB/s); see refs.bib nvidia_specs
HBM_BYTES_PER_S = {"h100": 3.35e12, "b200": 8.0e12}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["config"] = df.run_id.str.replace(r"/r\d+$", "", regex=True)
    var = df.run_id.str.split("/").str[4]
    for k in VARIANT_KEYS:
        df["v_" + k] = var.map(lambda s, k=k: dict(_VAR_RE.findall(s)).get(k))
    for k in ("target_batch", "context_cached", "context_cap", "fresh_in", "prefix_cached"):
        df["v_" + k] = pd.to_numeric(df["v_" + k], errors="coerce")
    df["v_gpu_memory_utilization"] = pd.to_numeric(df["v_gpu_memory_utilization"], errors="coerce").fillna(0.9)
    df["design_batch"] = [design_target(c, t) if t == t else -1 for c, t in zip(df.cell, df.v_target_batch)]
    # v1.0.1: in-window prefix-cache reuse (data/cache_reuse.csv, Appendix C). A decode-type run whose cached context
    # was recomputed on most requests is not the shape its run id states and is excluded with a stated reason; the
    # KV-pressure runs (c9, c9b) keep theirs, since eviction is the regime those experiments measure.
    cr = path.parent / "cache_reuse.csv"
    if cr.exists():
        c = pd.read_csv(cr)[["run_id", "cache_reuse_r"]]
        df = df.merge(c, on="run_id", how="left")
        evicted = (df.cache_reuse_r < CACHE_REUSE_MIN) & df.cell.isin(CACHE_RULE_CELLS) & df["dropped"].isna()
        df.loc[evicted, "dropped"] = ["prefix cache evicted in-window (reuse %.2f of expected)" % r for r in df.loc[evicted, "cache_reuse_r"]]
    else:
        df["cache_reuse_r"] = np.nan
    df["usable"] = df["dropped"].isna() & df.energy_j.notna()
    df["decode_ok"] = df.usable & df.e_d_j_per_tok.notna() & (df.phase == "decode")
    df["rated_w"] = df.accelerator.map(RATED_W) * df.n_gpus
    st = path.parent / "store_tables" / "cells.parquet"
    if st.exists():
        sched = ["sched_kv_usage_mean", "sched_waiting_mean", "sched_running_max", "sched_preemptions_delta",
                 "sched_preemptions_counter_present"]
        a = pd.read_parquet(st)
        a = a[a.kind == "active"][["run_id"] + sched].drop_duplicates("run_id")
        df = df.merge(a, on="run_id", how="left")
    return df


def store_facts(features: Path) -> dict:
    """Facts that live in the per-window store beside the features table (power
    limit, throttle reasons at window edges, energy source), for the 917 runs."""
    p = features.parent / "store_tables" / "cells.parquet"
    if not p.exists():
        return {"available": False}
    st = pd.read_parquet(p)
    a = st[st.kind == "active"]
    out = {"available": True, "energy_sources": a.energy_sources.value_counts().to_dict(), "limits": {}, "throttle": {}}
    for acc, g in a.groupby("accelerator"):
        lim, thr, edges = {}, {}, 0
        for s in g.per_gpu_json:
            for gpu in json.loads(s):
                lim[gpu.get("power_limit_w")] = lim.get(gpu.get("power_limit_w"), 0) + 1
                for k in ("throttle_start", "throttle_end"):
                    edges += 1
                    for t in (gpu.get(k) or ["unknown"]):
                        thr[t] = thr.get(t, 0) + 1
        out["limits"][acc] = lim
        out["throttle"][acc] = {"edges": edges, **thr}
    return out


# ---------------------------------------------------------------------------
# bands
# ---------------------------------------------------------------------------

class Bands:
    """Central values and their bands. No configuration ran on more than two
    pods, so every band is a plain range: of the per-pod means with two pods,
    of the repeats with one. (Until 5 Nov 2026 this was written as a pod
    bootstrap, 5th to 95th percentile; with two units each extreme of a
    resample has probability 1/4, so that percentile band was this range.)"""

    @staticmethod
    def _range(values: np.ndarray, pods: np.ndarray) -> tuple[float, float, str]:
        """(lo, hi, kind) of one side: range of per-pod means, or of the runs on one pod."""
        keys = sorted(set(pods))
        if len(keys) >= 2:
            pm = np.array([values[pods == q].mean() for q in keys])
            return float(pm.min()), float(pm.max()), "between-pod range"
        return float(values.min()), float(values.max()), ("range of repeats, one pod" if len(values) >= 2 else "single run")

    def band(self, values: pd.Series, pods: pd.Series, low: bool = False) -> dict:
        """Central mean and its band (range over pods, or over repeats on one pod)."""
        v = np.asarray(values, float)
        p = np.asarray(pods)
        ok = ~np.isnan(v)
        v, p = v[ok], p[ok]
        n, k = len(v), len(set(p))
        out = {"n": int(n), "pods": int(k), "mean": float(np.mean(v)) if n else math.nan}
        if n == 0:
            out.update(lo=math.nan, hi=math.nan, conf="none", band_kind="no data")
            return out
        lo, hi, kind = self._range(v, p)
        out.update(lo=lo, hi=hi, band_kind=kind)
        conf = ("high" if n >= 2 else "medium") if k >= 2 else "medium"
        out["conf"] = "low" if low else conf
        return out

    def ratio_band(self, num: pd.DataFrame, den: pd.DataFrame, col: str, low: bool = False, pair: bool = True) -> dict:
        """Ratio of central values. Where both sides ran on the same pods and
        `pair` is set, the ratio is formed within each pod (pod factors cancel)
        and the band is the range of the per-pod ratios (one pod: ratios paired
        by repeat, band the range of the repeats). Otherwise the band runs from
        the lowest numerator over the highest denominator to the highest over
        the lowest, each side's range taken as in `band`."""
        a, b = self.band(num[col], num.pod_id, low), self.band(den[col], den.pod_id, low)
        pods_str = f"{a['pods']} / {b['pods']}"
        if a["n"] == 0 or b["n"] == 0:
            return {"n": 0, "pods": 0, "pods_str": pods_str, "mean": math.nan, "lo": math.nan, "hi": math.nan, "conf": "none",
                    "band_kind": "no data", "pairing": "none"}
        common = sorted(set(num.pod_id) & set(den.pod_id))
        if pair and common and set(num.pod_id) == set(den.pod_id):
            if len(common) >= 2:
                rp = np.array([num.loc[num.pod_id == q, col].mean() / den.loc[den.pod_id == q, col].mean() for q in common])
                return {"n": a["n"] + b["n"], "pods": len(common), "pods_str": pods_str, "mean": float(rp.mean()),
                        "lo": float(rp.min()), "hi": float(rp.max()),
                        "conf": "low" if low else ("high" if a["n"] >= 2 and b["n"] >= 2 else "medium"),
                        "band_kind": "paired by pod (range of the per-pod ratios)", "pairing": "pod"}
            reps = sorted(set(num.repeat) & set(den.repeat))
            if len(reps) >= 2:
                rr = np.array([num.loc[num.repeat == r, col].mean() / den.loc[den.repeat == r, col].mean() for r in reps])
                return {"n": a["n"] + b["n"], "pods": 1, "pods_str": pods_str, "mean": float(rr.mean()), "lo": float(rr.min()),
                        "hi": float(rr.max()), "conf": "low" if low else "medium",
                        "band_kind": "paired by repeat, one pod (range of repeats)", "pairing": "repeat"}
        conf_rank = {"high": 2, "medium": 1, "low": 0}
        conf = min((a["conf"], b["conf"]), key=lambda c: conf_rank[c])
        return {"n": a["n"] + b["n"], "pods": max(a["pods"], b["pods"]), "pods_str": pods_str, "mean": a["mean"] / b["mean"],
                "lo": a["lo"] / b["hi"], "hi": a["hi"] / b["lo"],
                "conf": "low" if low else conf, "band_kind": "unpaired (lowest to highest combination)", "pairing": "none"}


def quotient_gap(rows: list[tuple[float, float, float]]) -> float:
    """Largest relative gap between (mean board W / mean token rate) and the mean energy per token,
    over a table's rows. Per run the quotient is exact; over runs a mean of ratios is not a ratio of means."""
    return max(abs(w / r / e - 1) for w, r, e in rows if r and e) if rows else 0.0


def fmt(x: float, sig: int = 3, extra: int = 0) -> str:
    """Three significant figures (one decimal from 10, none from 100); `extra`
    adds digits where a band would otherwise print with zero width."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if abs(x) >= 100:
        return f"{x:,.{extra}f}"
    if abs(x) >= 10:
        return f"{x:.{1 + extra}f}"
    return f"{x:.{sig + extra}g}"


def band_str(b: dict, sig: int = 3) -> str:
    if b["n"] == 0:
        return "--"
    if b["n"] == 1:
        return f"{fmt(b['mean'], sig)} (n=1)"
    # where two different runs round to the same figure, add significant figures
    # until the band shows its width, rather than print a zero-width band
    ex = 0
    while b["lo"] != b["hi"] and fmt(b["lo"], sig, ex) == fmt(b["hi"], sig, ex) and ex < 3:
        ex += 1
    return f"{fmt(b['mean'], sig, ex)} [{fmt(b['lo'], sig, ex)}, {fmt(b['hi'], sig, ex)}]"


# ---------------------------------------------------------------------------
# table writers
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    return (str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace("[", "{[}").replace("]", "{]}"))


def _wrap_header(h: str, maxlen: int = 12) -> list[str]:
    """Split a column header into lines of at most ~maxlen characters at spaces."""
    words = h.split()
    if len(h) <= maxlen or len(words) == 1:
        return [h]
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > maxlen:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return lines


# (font size command, point size, \tabcolsep in pt, landscape) tried in order until the table fits
_TABLE_STEPS = [("footnotesize", 9.0, 4.0, False), ("scriptsize", 8.0, 3.0, False), ("scriptsize", 8.0, 3.0, True)]
_TEXT_WIDTH_PT = 468.0        # letter, 1 in margins
_LANDSCAPE_WIDTH_PT = 706.0   # letter landscape, 0.6 in margins (set by \newgeometry around the table)


def _table_layout(cols: list[str], body_cells: list[list[str]]) -> tuple[str, float, bool]:
    """Pick a font size, column separation and orientation from an estimate of the natural width
    (0.5 em per character, which is generous for Latin Modern digits)."""
    widths = []
    for j, c in enumerate(cols):
        n = max([len(l) for l in _wrap_header(c)] + [len(r[j]) for r in body_cells])
        widths.append(n)
    for size, pt, sep, land in _TABLE_STEPS:
        est = sum(w * 0.5 * pt for w in widths) + 2 * sep * len(cols)
        if est <= (_LANDSCAPE_WIDTH_PT if land else _TEXT_WIDTH_PT):
            return size, sep, land
    print(f"  warning: table does not fit even in landscape (estimate {est:.0f} pt > {_LANDSCAPE_WIDTH_PT:.0f} pt)")
    return _TABLE_STEPS[-1][0], _TABLE_STEPS[-1][2], True


def write_table(name: str, df: pd.DataFrame, caption: str, label: str, col_fmt: str | None = None,
                small: bool = True, longtable: bool = False, notes: str | None = None) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if "Conf." in df.columns:
        # evidence is stated as counts (Pods column, n=1 in the band), not as a confidence label
        i = list(df.columns).index("Conf.")
        df = df.drop(columns="Conf.")
        if col_fmt:
            pos = [j for j, c in enumerate(col_fmt) if c in "lrc"][i]
            col_fmt = col_fmt[:pos] + col_fmt[pos + 1:]
    df.to_csv(TABLES / f"{name}.csv", index=False)
    cols = list(df.columns)
    col_fmt = col_fmt or ("l" + "r" * (len(cols) - 1))
    aligns = [c for c in col_fmt if c in "lrc"]
    body_cells = [[str(x) for x in row] for row in df.astype(str).itertuples(index=False)]
    size, sep, land = _table_layout(cols, body_cells)

    def head_cell(c: str, a: str) -> str:
        lines = _wrap_header(c)
        if len(lines) == 1:
            return _tex_escape(c)
        return "\\makecell[" + a + "]{" + "\\\\".join(_tex_escape(l) for l in lines) + "}"

    head = " & ".join(head_cell(c, a) for c, a in zip(cols, aligns)) + r" \\"
    body = "\n".join(" & ".join(_tex_escape(x) for x in row) + r" \\" for row in body_cells)
    cap = caption + (f" {notes}" if notes else "")
    setup = f"\\{size}\n\\setlength{{\\tabcolsep}}{{{sep:g}pt}}\n"
    if longtable:
        inner = (f"\\begin{{longtable}}{{{col_fmt}}}\n\\caption{{{cap}}}\\label{{{label}}}\\\\\n\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
                 f"\\toprule\n{head}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n{body}\n\\end{{longtable}}\n")
        tex = "\\begingroup\n" + setup + inner + "\\endgroup\n"
    else:
        inner = (f"\\begin{{table}}[t]\n\\centering\n{setup}\\caption{{{cap}}}\\label{{{label}}}\n"
                 f"\\begin{{tabular}}{{{col_fmt}}}\n\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")
        tex = inner
    if land:
        # pdflscape page with 0.6 in margins; the float or longtable sits on its own landscape page(s)
        tex = ("\\newgeometry{margin=0.6in}\n\\begin{landscape}\n" + tex + "\\end{landscape}\n\\restoregeometry\n")
    (TABLES / f"{name}.tex").write_text(tex)
    md = [f"**{cap}**", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.astype(str).itertuples(index=False)]
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

def corpus_tables(df: pd.DataFrame, head: dict) -> None:
    cfg = df.groupby("config").agg(n=("run_id", "size"), nu=("usable", "sum"), pods=("pod_id", "nunique"), cell=("cell", "first"))
    head.update(runs=int(len(df)), usable_runs=int(df.usable.sum()), configs=int(len(cfg)),
                configs_two_repeats=int((cfg.n >= 2).sum()), configs_two_usable=int((cfg.nu >= 2).sum()),
                configs_two_pods=int((cfg.pods >= 2).sum()), configs_one_pod=int((cfg.pods < 2).sum()),
                configs_no_usable=int((cfg.nu == 0).sum()), configs_one_run=int((cfg.n == 1).sum()),
                pods=int(df.pod_id.nunique()), data_centers=int(df.data_center.nunique()),
                metered_hours=float(df.duration_s.sum() / 3600),
                models=sorted(df.model.unique()), window_rows_dropped=int((~df.usable).sum()),
                dropped_runs=df.loc[~df.usable, "run_id"].tolist(),
                one_pod_configs_by_cell=cfg[cfg.pods < 2].cell.value_counts().to_dict())
    # configurations that appear in no results table (the omission rule, Appendix C)
    omitted = {"c7 fp8-only runs of the MoE models (no nvfp4 checkpoint)": int(((df.cell == "c7_precision") & df.model.isin(["qwen3-30b-a3b", "deepseek-v2-lite"])).sum()),
               "c8 early Qwen3-8B replays with a 32,768 context cap, one run each": int(((df.cell == "c8_agent_replay") & (df.v_context_cap == 32768)).sum()),
               "c3 runs at a 32,768-token prompt whose every request was refused (no prompt tokens)": int(((df.cell == "c3_prefill_miss") & (df.v_fresh_in == 32768)).sum()),
               "runs with no usable window": int((~df.usable).sum())}
    head["omitted_runs"] = omitted
    # cell x (model, accelerator) run counts
    t = df.groupby(["cell", "accelerator", "model"]).size().unstack(["accelerator", "model"], fill_value=0)
    t = t.reindex(columns=pd.MultiIndex.from_product([["h100", "b200"], MODEL_ORDER]), fill_value=0)
    t.columns = [f"{ACC_LABEL[a]} {MODEL_LABEL[m]}" for a, m in t.columns]
    t["Total"] = t.sum(axis=1)
    t.loc["Total"] = t.sum()
    t = t.reset_index().rename(columns={"cell": "Experiment"})
    write_table("corpus_cells", t, "Metered runs per experiment, model and accelerator (917 runs).", "tab:corpus",
                col_fmt="l" + "r" * (len(t.columns) - 1), small=True)
    # pods
    p = df.groupby("pod_id").agg(**{"Data center": ("data_center", "first"), "Accelerator": ("accelerator", "first"),
                                    "GPUs": ("n_gpus", "max"), "Models (count)": ("model", lambda s: len(set(s))),
                                    "Runs": ("run_id", "size"), "Metered hours": ("duration_s", lambda s: s.sum() / 3600)})
    p["Accelerator"] = p["Accelerator"].map(ACC_LABEL)
    p["Metered hours"] = p["Metered hours"].map(lambda x: f"{x:.2f}")
    p = p.sort_values(["Accelerator", "Runs"], ascending=[True, False]).reset_index().rename(columns={"pod_id": "Pod"})
    p["Pod"] = [f"pod {i + 1}" for i in range(len(p))]
    write_table("pods", p, "The 13 rented pods. Pod identifiers are replaced by an index (the mapping is in the deposited corpus). Metered hours are the summed lengths of the metered windows on that pod, warm-up, idle windows and model loads excluded.",
                "tab:pods", small=True)


def idle_table(df: pd.DataFrame, boot: Bands, head: dict) -> None:
    rows = []
    for (m, a, tp), g in df[(df.cell == "c0_static") & df.usable].groupby(["model", "accelerator", "tp"]):
        b = boot.band(g.mean_power_w, g.pod_id)
        per_gpu = b["mean"] / tp
        rows.append({"Model": MODEL_LABEL[m], "Accelerator": ACC_LABEL[a], "GPUs": tp,
                     "Board W, all GPUs": band_str(b), "W per GPU": fmt(per_gpu),
                     "Share of rated": f"{per_gpu / RATED_W[a]:.0%}", "Pods": b["pods"], "Conf.": b["conf"]})
        head.setdefault("idle_w_per_gpu", {})[f"{m}/{a}"] = per_gpu
    write_table("idle_power", pd.DataFrame(rows),
                "Loaded-idle board power (experiment c0: model loaded, no requests, default power management, 60 s window), summed over the GPUs serving the model. Mean over runs, with the range over the two pods in brackets. Share of rated: W per GPU over 700 W (H100) or 1,000 W (B200).",
                "tab:idle", col_fmt="llrrrrrl")
    idle = df[(df.cell == "c0_static") & df.usable].groupby("accelerator").apply(lambda g: (g.mean_power_w / g.tp).agg(["min", "max"]))
    head["idle_w_per_gpu_range"] = {a: [float(idle.loc[a, "min"]), float(idle.loc[a, "max"])] for a in idle.index}


def decode_batch(df: pd.DataFrame, boot: Bands, head: dict) -> pd.DataFrame:
    d = df[(df.cell == "c1_decode_batch") & df.decode_ok]
    rows, out, gaps = [], [], []
    for (m, a, tp), g in d.groupby(["model", "accelerator", "tp"]):
        for tb, gg in g.groupby("v_target_batch"):
            b = boot.band(gg.e_d_j_per_tok, gg.pod_id)
            tps = boot.band(gg.tokens_per_s, gg.pod_id)
            pw = boot.band(gg.mean_power_w, gg.pod_id)
            B = gg.B.mean()
            rated = RATED_W[a] * tp
            rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "TP": tp, "Target B": int(tb),
                         "Achieved B": f"{B:.1f}", "Mean KV use": f"{gg.sched_kv_usage_mean.mean():.0%}" if "sched_kv_usage_mean" in gg and gg.sched_kv_usage_mean.notna().any() else "--",
                         "Board W": band_str(pw, 3), "% of rated": f"{pw['mean'] / rated:.0%}", "Tokens per second": band_str(tps, 3),
                         "J per output token": band_str(b), "Pods": b["pods"], "Conf.": b["conf"]})
            gaps.append((pw["mean"], tps["mean"], b["mean"]))
            out.append(dict(model=m, accelerator=a, tp=tp, target_batch=int(tb), B=B, **{f"e_{k}": v for k, v in b.items()},
                            tokens_per_s=tps["mean"], tps_lo=tps["lo"], tps_hi=tps["hi"], power_w=pw["mean"], power_lo=pw["lo"], power_hi=pw["hi"],
                            rated_share=pw["mean"] / rated))
    o = pd.DataFrame(out)
    top = o[o.target_batch == 256]
    write_table("decode_batch", pd.DataFrame(rows),
                f"Energy per output token of the decode workload against batch (experiment c1: 1k cached context, 200 fresh input tokens, 256 output tokens; gross board energy including the idle draw). Target B is the number of requests the client kept in flight. Achieved B is the server's mean count of running requests over the middle 80\\% of the window. At target 256 the client's HTTP connection pool (100 connections, the library default) let at most 100 requests stream at once, and the server ran {top.B.min():.0f} to {top.B.max():.0f} on average, so that row is the highest load we offered, not the most the GPU could hold. Mean KV use: the server's mean KV-cache occupancy over the same part of the window. Board W is the mean board power over the window summed over the GPUs in the TP column, and \\% of rated divides it by 700 W per H100 or 1,000 W per B200. J per output token equals Board W divided by tokens per second to within {math.ceil(100 * quotient_gap(gaps))}\\%; the small difference arises because each column is a mean over runs. Brackets: the range over the two pods (one pod: the range of the repeats). TP: tensor-parallel degree (GPUs sharing the model).",
                "tab:decode_batch", col_fmt="llrrrrlrllrl", longtable=True)
    # headline: batch 1 vs highest bin per model x accelerator
    hl = {}
    for (m, a), g in o.groupby(["model", "accelerator"]):
        lo_b = g[g.target_batch == 1].iloc[0]
        b64 = g[g.target_batch == 64].iloc[0]
        hi_b = g.sort_values("target_batch").iloc[-1]
        hl[f"{m}/{a}"] = {"b1": [lo_b.e_mean, lo_b.e_lo, lo_b.e_hi, lo_b.e_conf],
                          "b64": [b64.e_mean, b64.e_lo, b64.e_hi, b64.e_conf, float(b64.B)],
                          "bmax": [hi_b.e_mean, hi_b.e_lo, hi_b.e_hi, hi_b.e_conf, float(hi_b.B)],
                          "ratio": float(lo_b.e_mean / hi_b.e_mean), "ratio_b1_b64": float(lo_b.e_mean / b64.e_mean),
                          "turn_up_bmax_over_b64": float(hi_b.e_mean / b64.e_mean), "tp": int(hi_b.tp),
                          "tokens_per_s_bmax": [float(hi_b.tokens_per_s), float(hi_b.tps_lo), float(hi_b.tps_hi)],
                          "power_range_w": [float(g.power_w.min()), float(g.power_w.max())],
                          "rated_share_range": [float(g.rated_share.min()), float(g.rated_share.max())]}
    head["decode_batch"] = hl
    head["decode_batch_summary"] = {"bmax_achieved_range": [float(top.B.min()), float(top.B.max())],
                                    "bmax_e_range_small_models": [float(o[(o.target_batch == 256) & (o.model != "llama-3.3-70b")].e_mean.min()),
                                                                  float(o[(o.target_batch == 256) & (o.model != "llama-3.3-70b")].e_mean.max())],
                                    "bmax_tps_small_models": [float(o[(o.target_batch == 256) & (o.model != "llama-3.3-70b")].tokens_per_s.min()),
                                                              float(o[(o.target_batch == 256) & (o.model != "llama-3.3-70b")].tokens_per_s.max())],
                                    "ratio_range": [min(v["ratio"] for v in hl.values()), max(v["ratio"] for v in hl.values())],
                                    "rated_share_all": [float(o.rated_share.min()), float(o.rated_share.max())]}
    # B200 / H100 at matched target batch (same model). Two comparisons are kept apart:
    # the same number of GPUs on each side (c1 at TP1, and c6 where both ran the same TP),
    # and Llama-3.3-70B on one B200 against two H100s (c1), where the B200 also uses fewer GPUs.
    rows, ratios, same, fewer = [], {}, {}, {}
    def add(m, tb, n, dn, exp, kind):
        r = boot.ratio_band(n, dn, "e_d_j_per_tok", pair=False)
        rows.append({"Comparison": kind, "Model": MODEL_LABEL[m], "Target B": int(tb), "GPUs B200 / H100": f"{n.tp.iloc[0]} / {dn.tp.iloc[0]}",
                     "B200 / H100, J per output token": band_str(r), "Pods B200 / H100": r["pods_str"], "Exp.": exp, "Conf.": r["conf"]})
        return [r["mean"], r["lo"], r["hi"], r["conf"]]
    for m in MODEL_ORDER:
        for tb in sorted(o.target_batch.unique()):
            n = d[(d.model == m) & (d.accelerator == "b200") & (d.v_target_batch == tb)]
            dn = d[(d.model == m) & (d.accelerator == "h100") & (d.v_target_batch == tb)]
            if len(n) and len(dn):
                v = add(m, tb, n, dn, "c1", "same GPUs" if n.tp.iloc[0] == dn.tp.iloc[0] else "fewer B200s")
                ratios.setdefault(m, {})[int(tb)] = v
                (same if n.tp.iloc[0] == dn.tp.iloc[0] else fewer).setdefault(m, {}).setdefault(int(n.tp.iloc[0]), {})[int(tb)] = v
    c6 = df[(df.cell == "c6_tensor_parallel") & df.decode_ok & (df.weight_precision == "fp8") & (df.tp > 1)]
    for (m, tp), g in c6.groupby(["model", "tp"]):
        for tb in sorted(g.v_target_batch.unique()):
            n, dn = (g[(g.accelerator == a) & (g.v_target_batch == tb)] for a in ("b200", "h100"))
            if len(n) and len(dn):
                same.setdefault(m, {}).setdefault(int(tp), {})[int(tb)] = add(m, tb, n, dn, "c6", "same GPUs")
    order = {"same GPUs": 0, "fewer B200s": 1}
    rows = sorted(rows, key=lambda r: (order[r["Comparison"]], [MODEL_LABEL[x] for x in MODEL_ORDER].index(r["Model"]), r["GPUs B200 / H100"], r["Target B"]))
    write_table("b200_vs_h100", pd.DataFrame(rows),
                "Ratio of energy per output token (decode workload), B200 to H100, at matched target batch (1k cached context, 256 output tokens). Below 1 means the B200 used less energy per token. 'Same GPUs': both sides ran the model on the same number of GPUs, from experiment c1 (one GPU each) or c6 (two or four each, tensor parallel). 'Fewer B200s': Llama-3.3-70B on one B200 against two H100s, the smallest H100 configuration that holds it, so the ratio includes the saving from serving on fewer GPUs. The two sides ran on different pods, so the ratio is of central values, and the band runs from the lowest to the highest combination of the two sides' pod values.",
                "tab:b200_h100", col_fmt="llrrllll", longtable=True)
    head["b200_same_gpus"] = same
    head["b200_fewer_gpus"] = fewer
    head["b200_over_h100"] = ratios
    return o


def decode_context(df: pd.DataFrame, boot: Bands, head: dict) -> pd.DataFrame:
    d = df[(df.cell == "c2_decode_context") & df.decode_ok]
    rows, out, gaps = [], [], []
    base8 = {}
    for (m, a, ctx), gg in d[d.design_batch == 8].groupby(["model", "accelerator", "v_context_cached"]):
        base8[(m, a, int(ctx))] = gg.e_d_j_per_tok.mean()
    for (m, a, tp), g in d.groupby(["model", "accelerator", "tp"]):
        for (dsg, tb, ctx), gg in g.groupby(["design_batch", "v_target_batch", "v_context_cached"]):
            b = boot.band(gg.e_d_j_per_tok, gg.pod_id)
            e = gg.e_d_j_per_tok.to_numpy()
            flag = ""
            if len(e) >= 2 and e.max() / e.min() > 1.5:
                flag = "regimes differ"
            elif dsg == 64 and (m, a, int(ctx)) in base8 and b["mean"] > base8[(m, a, int(ctx))]:
                flag = "above batch-8 figure"
            cell_e = band_str(b) if not flag else " / ".join(fmt(x) for x in sorted(e)) + " (runs differ)"
            conf = b["conf"] if not flag else "medium"
            pw = boot.band(gg.mean_power_w, gg.pod_id)
            rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "TP": tp, "Design target B": int(dsg), "Target as run": int(tb),
                         "Achieved B": f"{gg.B.mean():.1f}", "Cached context (tokens)": f"{int(ctx):,}", "Attended context (tokens)": f"{gg.L.mean():,.0f}",
                         "Board W": band_str(pw, 3), "% of rated": f"{pw['mean'] / (RATED_W[a] * tp):.0%}",
                         "Tokens per second": fmt(gg.tokens_per_s.mean(), 3), "J per output token": cell_e, "Pods": b["pods"], "Conf.": conf})
            if not flag:
                gaps.append((pw["mean"], gg.tokens_per_s.mean(), b["mean"]))
            out.append(dict(model=m, accelerator=a, tp=tp, target_batch=int(dsg), target_as_run=int(tb), context=int(ctx), L=gg.L.mean(), B=gg.B.mean(),
                            flag=flag, **{f"e_{k}": v for k, v in b.items()}, e_runs=sorted(e.tolist()), tokens_per_s=gg.tokens_per_s.mean()))
    flagged_note = (" A configuration whose two runs differ by more than 1.5 times, or whose design batch of 64 cost more per token than its batch-8 twin,"
                    " lists its two runs instead of a mean, marked 'runs differ'."
                    if any(str(r["J per output token"]).endswith("(runs differ)") for r in rows) else "")
    write_table("decode_context", pd.DataFrame(rows),
                "Energy per output token of the decode workload against attended context (experiment c2: 200 fresh input tokens, 256 output tokens; gross board energy including the idle draw). "
                "The cached context is the warmed prefix; the attended context adds the fresh input and half the output. "
                "Design target B is the grid point (8 or 64); target as run is what the feasibility rule resized it to where the KV cache could not hold 64 requests at that context; "
                "achieved B is the server's mean count of running requests. Board W and \\% of rated as in Table \\ref{tab:decode_batch}. "
                f"J per output token equals Board W divided by tokens per second to within {math.ceil(100 * quotient_gap(gaps))}\\%; "
                "the small difference arises because each column is a mean over runs. Brackets: the range over the two pods (one pod: the range of the repeats). "
                "Tokens per second: mean over runs, no band." + flagged_note,
                "tab:decode_context", col_fmt="llrrrrrrlrrlrl", longtable=True)
    o = pd.DataFrame(out)
    hl = {}
    for (m, a, tb), g in o.groupby(["model", "accelerator", "target_batch"]):
        g = g[g.flag == ""].sort_values("context")
        if len(g) >= 2:
            hl[f"{m}/{a}/B{tb}"] = {"ctx_min": int(g.context.iloc[0]), "ctx_max": int(g.context.iloc[-1]),
                                    "e_min_ctx": [float(g.e_mean.iloc[0]), float(g.e_lo.iloc[0]), float(g.e_hi.iloc[0])],
                                    "e_max_ctx": [float(g.e_mean.iloc[-1]), float(g.e_lo.iloc[-1]), float(g.e_hi.iloc[-1])],
                                    "B_max_ctx": float(g.B.iloc[-1]), "target_as_run_max_ctx": int(g.target_as_run.iloc[-1]),
                                    "ratio": float(g.e_mean.iloc[-1] / g.e_mean.iloc[0]),
                                    "conf": min(g.e_conf, key=lambda c: {"high": 2, "medium": 1, "low": 0}[c])}
    head["decode_context"] = hl
    head["decode_context_flagged"] = o[o.flag != ""][["model", "accelerator", "target_batch", "target_as_run", "context", "flag", "e_runs"]].to_dict("records")
    return o


def prefill(df: pd.DataFrame, boot: Bands, head: dict) -> pd.DataFrame:
    d = df[(df.cell == "c3_prefill_miss") & df.usable & df.e_p_j_per_tok.notna()]
    rows, out, gaps = [], [], []
    for (m, a, tp), g in d.groupby(["model", "accelerator", "tp"]):
        for (tb, fi), gg in g.groupby(["v_target_batch", "v_fresh_in"]):
            b = boot.band(gg.e_p_j_per_tok, gg.pod_id)
            tps = gg.prompt_tokens.sum() / gg.duration_s.sum()
            pw = boot.band(gg.mean_power_w, gg.pod_id)
            rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "TP": tp, "Target B": int(tb),
                         "Prompt tokens": f"{int(fi):,}", "Board W": band_str(pw, 3), "% of rated": f"{pw['mean'] / (RATED_W[a] * tp):.0%}",
                         "Prompt tokens per second": f"{tps:,.0f}", "J per prompt token": band_str(b),
                         "Pods": b["pods"], "Conf.": b["conf"]})
            gaps.append((pw["mean"], tps, b["mean"]))
            out.append(dict(model=m, accelerator=a, tp=tp, target_batch=int(tb), fresh_in=int(fi), **{f"e_{k}": v for k, v in b.items()},
                            prompt_tps=tps, power_w=pw["mean"], rated_share=pw["mean"] / (RATED_W[a] * tp)))
    write_table("prefill", pd.DataFrame(rows),
                f"Prefill energy per prompt token with prefix caching off (experiment c3: prefix caching off, one output token; gross board energy including the idle draw). Prompt length is the fresh input; the native-maximum point is the model's context window less a margin, run at 31,200 and 32,511 tokens for the 32k models as the margin rule was tightened. Prompt tokens per second: the runs' prompt tokens over their window lengths, pooled, no band. Board W and \\% of rated as in Table \\ref{{tab:decode_batch}}. For each run, J per prompt token equals board power divided by prompt tokens per second. The columns are averages over runs, so where the two pods differ widely (short prompts on B200) the quotient of the printed averages differs from the printed J per prompt token by up to {round(100 * quotient_gap(gaps)):.0f}\\%. Brackets: the range over the two pods.",
                "tab:prefill", col_fmt="llrrrlrrlrl", longtable=True)
    o = pd.DataFrame(out)
    hl = {}
    for (m, a), g in o.groupby(["model", "accelerator"]):
        for tb in sorted(g.target_batch.unique()):
            gt = g[g.target_batch == tb].sort_values("fresh_in")
            hl[f"{m}/{a}/B{tb}"] = {"points": [[int(r.fresh_in), float(r.e_mean), float(r.e_lo), float(r.e_hi), r.e_conf, int(r.e_n)] for r in gt.itertuples()],
                                    "power_range": [float(gt.power_w.min()), float(gt.power_w.max())],
                                    "rated_share_range": [float(gt.rated_share.min()), float(gt.rated_share.max())]}
    small = o[o.model != "llama-3.3-70b"]
    head["prefill"] = hl
    head["prefill_summary"] = {"small_models_e_range": [float(small.e_mean.min()), float(small.e_mean.max())],
                               "small_models_e_range_excl_qwen8b_b200": [float(small[~((small.model == "qwen3-8b") & (small.accelerator == "b200"))].e_mean.min()),
                                                                        float(small[~((small.model == "qwen3-8b") & (small.accelerator == "b200"))].e_mean.max())],
                               "llama_e_range_to_16k": [float(o[(o.model == "llama-3.3-70b") & (o.fresh_in <= 16384)].e_mean.min()),
                                                        float(o[(o.model == "llama-3.3-70b") & (o.fresh_in <= 16384)].e_mean.max())],
                               "llama_e_range_125k": [float(o[(o.model == "llama-3.3-70b") & (o.fresh_in > 100000)].e_mean.min()),
                                                      float(o[(o.model == "llama-3.3-70b") & (o.fresh_in > 100000)].e_mean.max())],
                               "concurrency_cut_at_1k": {f"{m}/{a}": 1 - g[(g.target_batch == 8) & (g.fresh_in == 1024)].e_mean.iloc[0] / g[(g.target_batch == 1) & (g.fresh_in == 1024)].e_mean.iloc[0]
                                                         for (m, a), g in o.groupby(["model", "accelerator"])},
                               "power_range_all": [float(o.power_w.min()), float(o.power_w.max())],
                               "rated_share_all": [float(o.rated_share.min()), float(o.rated_share.max())]}
    return o


def cached_prefix(df: pd.DataFrame, boot: Bands, head: dict) -> pd.DataFrame:
    d = df[(df.cell == "c4_cached_prefix") & df.usable & df.e_per_request_j.notna()]
    rows, out = [], []
    for (m, a, tp), g in d.groupby(["model", "accelerator", "tp"]):
        for (pc, fi, tb), gg in g.groupby(["v_prefix_cached", "v_fresh_in", "v_target_batch"]):
            c, u = gg[gg.v_control == "cached"], gg[gg.v_control == "uncached"]
            if not len(c) or not len(u):
                continue
            r = boot.ratio_band(c, u, "e_per_request_j")
            bc, bu = boot.band(c.e_per_request_j, c.pod_id), boot.band(u.e_per_request_j, u.pod_id)
            rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "Cached prefix": f"{int(pc):,}", "Fresh in": int(fi),
                         "Target B": int(tb), "Cached, J per request": band_str(bc), "Uncached, J per request": band_str(bu),
                         "Cached / uncached": band_str(r), "Pods": r["pods_str"], "Pairing": r["pairing"], "Conf.": r["conf"]})
            out.append(dict(model=m, accelerator=a, prefix_cached=int(pc), fresh_in=int(fi), target_batch=int(tb),
                            cached=bc["mean"], uncached=bu["mean"], **{f"r_{k}": v for k, v in r.items()}))
    write_table("cached_prefix", pd.DataFrame(rows),
                "Cached-prefix access (experiment c4): gross board energy per completed request with a warm cached prefix and a fresh suffix, against a matched uncached control of the same total length, both with 200 output tokens. The ratio is below 1 when caching saved energy. Pods: pods on the cached and uncached sides. Pairing 'pod': the ratio is formed within each pod and the band is the range over pods; 'repeat': both sides on one pod, ratio formed per repeat, band the range of the repeats. Brackets on the per-request columns: the range over pods.",
                "tab:cached", col_fmt="llrrrllllll", longtable=True)
    o = pd.DataFrame(out)
    hl = {}
    for (m, a), g in o.groupby(["model", "accelerator"]):
        best = g.sort_values("r_mean").iloc[0]
        worst = g.sort_values("r_mean").iloc[-1]
        hl[f"{m}/{a}"] = {"min_ratio": [float(best.r_mean), float(best.r_lo), float(best.r_hi), best.r_conf, int(best.prefix_cached), int(best.fresh_in), int(best.target_batch)],
                          "max_ratio": [float(worst.r_mean), float(worst.r_lo), float(worst.r_hi), worst.r_conf, int(worst.prefix_cached), int(worst.fresh_in), int(worst.target_batch)]}
    # pooled by (prefix, fresh) across models and accelerators
    pooled = o.groupby(["prefix_cached", "fresh_in"]).r_mean.agg(["median", "min", "max"]).reset_index()
    by_prefix = o.groupby("prefix_cached").r_mean.agg(["median", "min", "max"]).reset_index()
    b1_1k = o[(o.prefix_cached == 1024) & (o.target_batch == 1)]
    head["cached_prefix"] = {"per_model": hl, "pooled_by_shape": pooled.to_dict("records"), "by_prefix": by_prefix.to_dict("records"),
                             "overall_median": float(o.r_mean.median()), "overall_min": float(o.r_mean.min()), "overall_max": float(o.r_mean.max()),
                             "b1_1k": {"n": int(len(b1_1k)), "within_0.92_0.98": int(((b1_1k.r_mean >= 0.92) & (b1_1k.r_mean <= 0.985)).sum()),
                                       "bands_reach_1": int((b1_1k.r_hi >= 1.0).sum()), "min": float(b1_1k.r_mean.min()), "max": float(b1_1k.r_mean.max())},
                             "pairing_counts": o.r_pairing.value_counts().to_dict()}
    return o


def tensor_parallel(df: pd.DataFrame, boot: Bands, head: dict) -> pd.DataFrame:
    """Consecutive tensor-parallel pairs at matched target batch. The lowest
    TP of a model on an accelerator is the campaign's own serving TP and comes
    from cell c1 (same shape as c6); the higher TPs come from c6."""
    d = df[df.decode_ok & df.cell.isin(["c6_tensor_parallel", "c1_decode_batch"]) & (df.weight_precision == "fp8")]
    rows, out = [], []
    for (m, a), g in d[d.cell == "c6_tensor_parallel"].groupby(["model", "accelerator"]):
        pool = d[(d.model == m) & (d.accelerator == a)]
        tps = sorted(pool.tp.unique())
        for tb in sorted(g.v_target_batch.unique()):
            for lo_tp, hi_tp in zip(tps, tps[1:]):
                def pick(tp):
                    x = pool[(pool.tp == tp) & (pool.v_target_batch == tb) & (pool.cell == "c6_tensor_parallel")]
                    return x if len(x) else pool[(pool.tp == tp) & (pool.v_target_batch == tb)]
                glo, ghi = pick(lo_tp), pick(hi_tp)
                if not len(glo) or not len(ghi):
                    continue
                r = boot.ratio_band(ghi, glo, "e_d_j_per_tok")
                b0, bb = boot.band(glo.e_d_j_per_tok, glo.pod_id), boot.band(ghi.e_d_j_per_tok, ghi.pod_id)
                src = "c6 / c6" if glo.cell.iloc[0] == "c6_tensor_parallel" else "c1 / c6"
                rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "Target B": int(tb), "TP pair": f"{lo_tp} to {hi_tp}",
                             "Lower TP, J per output token": band_str(b0), "Higher TP, J per output token": band_str(bb),
                             "Ratio": band_str(r), "Throughput ratio": fmt(ghi.tokens_per_s.mean() / glo.tokens_per_s.mean()),
                             "Pods higher / lower": r["pods_str"], "Sides from": src, "Pairing": r["pairing"], "Conf.": r["conf"]})
                out.append(dict(model=m, accelerator=a, target_batch=int(tb), tp_lo=lo_tp, tp_hi=hi_tp, **{f"r_{k}": v for k, v in r.items()},
                                tps_ratio=ghi.tokens_per_s.mean() / glo.tokens_per_s.mean(), e_lo_tp=b0["mean"], e_hi_tp=bb["mean"]))
    write_table("tensor_parallel", pd.DataFrame(rows),
                "Tensor-parallel cost: gross board energy per output token, summed over the GPUs sharing the model, at the higher tensor-parallel degree over the next lower, at matched target batch (1k cached context, 256 output tokens). Both sides are experiment c6's own runs where c6 ran that degree ('c6 / c6'); the TP1 side of Llama-3.3-70B on B200 is experiment c1's run of the same shape ('c1 / c6'). A ratio above 1 is the energy cost of splitting the model; the throughput ratio (tokens per second, higher over lower, mean over runs, no band) is what the split bought. Pairing 'pod': ratio formed within each pod, band the range over pods; 'repeat': both sides on one pod, ratio per repeat, band the range of repeats; 'none': the two sides on different pods, band from the lowest to the highest combination of the two sides' pod values.",
                "tab:tp", col_fmt="llrllllrllll", longtable=True)
    o = pd.DataFrame(out)
    head["tensor_parallel"] = o.to_dict("records")
    return o


def precision(df: pd.DataFrame, boot: Bands, head: dict) -> pd.DataFrame:
    d = df[(df.cell == "c7_precision") & df.decode_ok]
    rows, out = [], []
    for (m, a), g in d.groupby(["model", "accelerator"]):
        for (dsg, ctx), gg in g.groupby(["design_batch", "v_context_cached"]):
            f8, f4 = gg[gg.weight_precision == "fp8"], gg[gg.weight_precision == "nvfp4"]
            if not len(f8) or not len(f4):
                continue
            r = boot.ratio_band(f4, f8, "e_d_j_per_tok", low=True)
            b8, b4 = boot.band(f8.e_d_j_per_tok, f8.pod_id), boot.band(f4.e_d_j_per_tok, f4.pod_id, low=True)
            rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "Design target B": int(dsg), "Cached context (tokens)": f"{int(ctx):,}",
                         "Target as run fp8 / nvfp4": f"{int(f8.v_target_batch.iloc[0])} / {int(f4.v_target_batch.iloc[0])}",
                         "Achieved B fp8 / nvfp4": f"{f8.B.mean():.1f} / {f4.B.mean():.1f}",
                         "fp8, J per output token": band_str(b8), "nvfp4, J per output token": band_str(b4), "nvfp4 / fp8": band_str(r),
                         "Throughput ratio": fmt(f4.tokens_per_s.mean() / f8.tokens_per_s.mean()), "Pods nvfp4 / fp8": r["pods_str"],
                         "Pairing": r["pairing"], "Conf.": r["conf"]})
            out.append(dict(model=m, accelerator=a, target_batch=int(dsg), context=int(ctx), **{f"r_{k}": v for k, v in r.items()},
                            e_fp8=b8["mean"], e_nvfp4=b4["mean"], tps_ratio=f4.tokens_per_s.mean() / f8.tokens_per_s.mean(),
                            B_fp8=f8.B.mean(), B_nvfp4=f4.B.mean()))
    write_table("precision", pd.DataFrame(rows),
                "Weight precision on B200 (experiment c7): gross board energy per output token with nvfp4 (4-bit) weights over fp8 (8-bit), at matched design target batch and cached context (200 fresh input, 256 output tokens). The fp8 side is c7's own fp8 runs. At 16k context and design target 64 the feasibility rule resized the target differently for the two servers (31 for fp8 at a 90\\% memory offer, 44 for nvfp4 at 70\\%), so that row compares different batches. The nvfp4 points are eight windows per model from one pod each, using 4-bit versions of the models made by third parties and a smaller memory offer, so every row is indicative. Pairing and brackets as in Table \\ref{tab:tp}; throughput ratio is nvfp4 over fp8 tokens per second, mean over runs, no band.",
                "tab:precision", col_fmt="llrrrrlllrlll", longtable=True)
    o = pd.DataFrame(out)
    head["precision"] = o.to_dict("records")
    return o


def agent_replay(df: pd.DataFrame, boot: Bands, head: dict) -> pd.DataFrame:
    d = df[(df.cell == "c8_agent_replay") & df.usable & (df.v_context_cap != 32768)]
    rows, out = [], []
    for (m, a, tp), g in d.groupby(["model", "accelerator", "tp"]):
        for tb, gg in g.groupby("v_target_batch"):
            agents = int(tb)
            steps_total = gg.completion_tokens / REPLAY_OUT_PER_STEP
            steps_per_agent = int(round(steps_total.iloc[0] / agents))
            cap = min(REPLAY_CONTEXT_START + 30 * REPLAY_CONTEXT_STEP, NATIVE_CONTEXT[m] - 1024)
            ctxs = [REPLAY_CONTEXT_START + i * REPLAY_CONTEXT_STEP for i in range(steps_per_agent)]
            ctxs = [c for c in ctxs if c <= cap]
            prompt_tokens_per_agent = sum(ctxs)
            gg = gg.assign(j_per_step=gg.energy_j / steps_total, j_per_out_tok=gg.energy_j / gg.completion_tokens,
                           j_per_agent=gg.energy_j / agents)
            bs = boot.band(gg.j_per_step, gg.pod_id)
            bt = boot.band(gg.j_per_out_tok, gg.pod_id)
            ba = boot.band(gg.j_per_agent, gg.pod_id)
            rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "TP": tp, "Agents": agents, "Steps": len(ctxs),
                         "Context at last step (tokens)": f"{ctxs[-1]:,}", "J per step": band_str(bs), "J per output token": band_str(bt),
                         "kJ per agent": band_str({**ba, "mean": ba["mean"] / 1e3, "lo": ba["lo"] / 1e3, "hi": ba["hi"] / 1e3}),
                         "Wall seconds": fmt(gg.duration_s.mean(), 3), "Pods": bs["pods"], "Conf.": bs["conf"]})
            out.append(dict(model=m, accelerator=a, tp=tp, agents=agents, steps=len(ctxs), ctx_last=ctxs[-1],
                            prompt_tokens_per_agent=prompt_tokens_per_agent, **{f"s_{k}": v for k, v in bs.items()},
                            j_per_out_tok=bt["mean"], kj_per_agent=ba["mean"] / 1e3, kj_lo=ba["lo"] / 1e3, kj_hi=ba["hi"] / 1e3))
    write_table("agent_replay", pd.DataFrame(rows),
                "Simulated agentic workload (experiment c8): simulated agents of 30 steps (Llama-3.3-70B) or 11 steps (the 32k-context models) run concurrently, each step re-reading the growing context (10,240 tokens, plus 2,048 per step, capped at the context window) and writing 300 tokens, with cache misses at steps 1 and 5 only (Appendix E). Energy is the gross board energy of the whole run; J per step divides it by the steps of all agents, kJ per agent by the agents. Agents is the number in flight as run (8 was resized to 7 for Llama-3.3-70B on B200 and to 5 on two H100s). Wall seconds: mean over runs, no band. Brackets: the range over the two pods.",
                "tab:agent", col_fmt="llrrrrlllrrl", longtable=True)
    o = pd.DataFrame(out)
    head["agent_replay"] = o.to_dict("records")
    alone, conc = o[o.agents == 1], o[o.agents > 1]
    head["agent_summary"] = {"alone_kj_range": [float(alone.kj_per_agent.min()), float(alone.kj_per_agent.max())],
                             "alone_small_kj_range": [float(alone[alone.model != "llama-3.3-70b"].kj_per_agent.min()), float(alone[alone.model != "llama-3.3-70b"].kj_per_agent.max())],
                             "alone_small_j_per_step_range": [float(alone[alone.model != "llama-3.3-70b"].s_mean.min()), float(alone[alone.model != "llama-3.3-70b"].s_mean.max())],
                             "conc_small_kj_range": [float(conc[conc.model != "llama-3.3-70b"].kj_per_agent.min()), float(conc[conc.model != "llama-3.3-70b"].kj_per_agent.max())],
                             "fold_range": [float(min(alone.set_index(["model", "accelerator"]).kj_per_agent / conc.set_index(["model", "accelerator"]).kj_per_agent)),
                                            float(max(alone.set_index(["model", "accelerator"]).kj_per_agent / conc.set_index(["model", "accelerator"]).kj_per_agent))],
                             "j_per_tok_alone": [float(alone.j_per_out_tok.min()), float(alone.j_per_out_tok.max())],
                             "j_per_tok_conc": [float(conc.j_per_out_tok.min()), float(conc.j_per_out_tok.max())]}
    return o


def kv_pressure(df: pd.DataFrame, head: dict) -> None:
    d = df[df.cell.isin(["c9_kv_pressure", "c9b_kv_pressure_diag"]) & df.decode_ok].copy()
    d = d.sort_values(["v_gpu_memory_utilization", "v_target_batch", "repeat"], ascending=[False, True, True])
    rows = []
    for _, r in d.iterrows():
        rows.append({"Experiment": r.cell.replace("_kv_pressure", "").replace("_diag", ""), "GPU memory offer": f"{r.v_gpu_memory_utilization:.1f}",
                     "Target B": int(r.v_target_batch), "Achieved B": f"{r.B:.1f}", "J per output token": fmt(r.e_d_j_per_tok),
                     "Tokens per second": f"{r.tokens_per_s:,.0f}", "Board W": f"{r.mean_power_w:,.0f}", "% of rated": f"{r.mean_power_w / 1000:.0%}",
                     "Mean KV use": f"{r.sched_kv_usage_mean:.0%}" if r.sched_kv_usage_mean == r.sched_kv_usage_mean else "--",
                     "Mean waiting": f"{r.sched_waiting_mean:.1f}" if r.sched_waiting_mean == r.sched_waiting_mean else "--",
                     "Preemptions": (f"{int(r.sched_preemptions_delta)}" if r.sched_preemptions_counter_present else "not recorded"),
                     "Collapsed": "yes" if r.e_d_j_per_tok >= 3 else "no",
                     "Cache reuse": f"{r.cache_reuse_r:.2f}" if r.cache_reuse_r == r.cache_reuse_r else "--",
                     "Idle-drift flag": "yes" if "idle drift" in str(r["flags"]) else "no"})
    write_table("kv_pressure", pd.DataFrame(rows),
                "KV-cache pressure on Llama-3.3-70B, one B200, 16,384 cached context, 200 fresh input and 256 output tokens (experiments c9 and c9b). Each row is one run, so no band; both experiments ran on one pod each. J per output token is gross board energy including the idle draw. GPU memory offer: the fraction of the board's memory the server was allowed, which sets the KV-cache pool. Mean KV use: the server's mean KV-cache occupancy over the middle 80\\% of the window. Mean waiting: requests admitted but not running. Preemptions: the server's preemption count over the window, recorded only in c9b ('not recorded' is missing telemetry, not zero). Collapsed: the run's energy per token exceeded 3 J, the gap between the two regimes. Cache reuse: the in-window prefix-cache hit share as a fraction of the expected one (Appendix C); near 1 the cached context was served from cache, near 0 it was recomputed on every request. Idle-drift flag: the idle power before and after the window differed by more than 5\\% (the run is kept and marked). \\% of rated: board W over 1,000 W.",
                "tab:kv", col_fmt="llrrrrrrrrllll", longtable=True)
    clean = d[d.e_d_j_per_tok < 3]
    collapsed = d[d.e_d_j_per_tok >= 3]
    head["kv_pressure"] = {"clean_e_range": [float(clean.e_d_j_per_tok.min()), float(clean.e_d_j_per_tok.max())],
                           "clean_B_range": [float(clean.B.min()), float(clean.B.max())],
                           "collapsed_e_range": [float(collapsed.e_d_j_per_tok.min()), float(collapsed.e_d_j_per_tok.max())] if len(collapsed) else None,
                           "collapsed_B_range": [float(collapsed.B.min()), float(collapsed.B.max())] if len(collapsed) else None,
                           "collapsed_power_range": [float(collapsed.mean_power_w.min()), float(collapsed.mean_power_w.max())] if len(collapsed) else None,
                           "clean_power_range": [float(clean.mean_power_w.min()), float(clean.mean_power_w.max())],
                           "n_runs": int(len(d)), "n_collapsed": int(len(collapsed))}


def variability(df: pd.DataFrame, head: dict) -> None:
    """Per configuration (at most two runs, at most two pods): the difference
    between the two runs as a share of their mean, and the same for the two
    per-pod means. With two values this is the honest statistic; a standard
    deviation or CV of two values is the same number divided by sqrt(2)."""
    d = df[df.usable].copy()
    d["metric"] = np.where(d.phase == "decode", d.e_d_j_per_tok,
                           np.where(d.phase == "prefill", d.e_p_j_per_tok,
                                    np.where(d.phase == "prefix_access", d.e_per_request_j,
                                             np.where(d.phase == "idle_loaded", d.mean_power_w,
                                                      d.energy_j / d.completion_tokens))))
    d["batch_bin"] = pd.cut(d.v_target_batch.fillna(0), [-1, 0, 1, 8, 16, 64, 1000],
                            labels=["n/a", "1", "4 to 8", "16", "64", "above 64"])
    rows = []
    excluded = []
    for cfg, g in d.groupby("config"):
        if g.metric.notna().sum() < 2:
            if len(df[df.config == cfg]) >= 2:
                excluded.append(cfg)
            continue
        m = g.metric.dropna()
        cv_rep = float((m.max() - m.min()) / m.mean())          # relative difference of the runs
        pods = g.groupby("pod_id").metric.mean()
        cv_pod = float((pods.max() - pods.min()) / pods.mean()) if len(pods) >= 2 else math.nan
        ratio_rep = float(m.max() / m.min())
        rows.append(dict(config=cfg, cell=g.cell.iloc[0], accelerator=g.accelerator.iloc[0], model=g.model.iloc[0],
                         batch_bin=str(g.batch_bin.iloc[0]), n=len(m), pods=len(pods), cv_repeat=cv_rep, cv_pod=cv_pod, ratio_repeat=ratio_rep,
                         same_pod=len(pods) == 1))
    v = pd.DataFrame(rows)
    v.to_csv(TABLES / "variability_per_config.csv", index=False)
    def q(s):
        s = s.dropna()
        return f"{s.median():.1%} / {s.quantile(0.9):.1%} (n={len(s)})" if len(s) else "--"
    out = []
    for cell, g in v.groupby("cell"):
        out.append({"Experiment": cell, "Repeat difference median / p90": q(g.cv_repeat), "Between-pod difference median / p90": q(g.cv_pod),
                    "Configs with 2 pods": int((g.pods >= 2).sum()), "Configs on one pod": int(g.same_pod.sum())})
    out.append({"Experiment": "all decode experiments", "Repeat difference median / p90": q(v[v.cell.str.contains("c1|c2|c6|c7|c9")].cv_repeat),
                "Between-pod difference median / p90": q(v[v.cell.str.contains("c1|c2|c6|c7|c9")].cv_pod),
                "Configs with 2 pods": int((v[v.cell.str.contains("c1|c2|c6|c7|c9")].pods >= 2).sum()),
                "Configs on one pod": int(v[v.cell.str.contains("c1|c2|c6|c7|c9")].same_pod.sum())})
    dd = v[v.cell.str.contains("c1|c2|c6|c7|c9")]
    for lab, g in (("decode, both repeats on one pod", dd[dd.same_pod]), ("decode, repeats on two pods", dd[~dd.same_pod])):
        out.append({"Experiment": lab, "Repeat difference median / p90": q(g.cv_repeat), "Between-pod difference median / p90": q(g.cv_pod),
                    "Configs with 2 pods": int((g.pods >= 2).sum()), "Configs on one pod": int(g.same_pod.sum())})
    write_table("variability_cells", pd.DataFrame(out),
                "Run-to-run variability of the metric each experiment reports (energy per output token, per prompt token or per request; board power for c0). Repeat difference: the difference between the two runs of one configuration as a percentage of their mean; we placed the two runs on different pods where we had two. Between-pod difference: the same for the two per-pod means, configurations run on two pods. Median and 90th percentile (p90) over configurations; n is the number of configurations. Configurations with fewer than two usable runs are excluded (Appendix C preamble). The last two rows split the decode configurations by whether both repeats ran on one pod; the two sets hold different configurations.",
                "tab:cv", col_fmt="lllrr")
    dec = v[v.cell.isin(["c1_decode_batch", "c2_decode_context"])]
    out = []
    bin_order = ["1", "4 to 8", "16", "64", "above 64"]
    for acc in ["h100", "b200"]:
        for bb in bin_order:
            g = dec[(dec.accelerator == acc) & (dec.batch_bin == bb)]
            if not len(g):
                continue
            out.append({"Accelerator": ACC_LABEL[acc], "Target batch": bb, "Repeat difference median / p90": q(g.cv_repeat),
                        "Between-pod difference median / p90": q(g.cv_pod), "Configs": len(g),
                        "Repeat difference max": f"{g.cv_repeat.max():.1%}"})
    write_table("variability_batch", pd.DataFrame(out),
                "Decode variability (energy per output token) by accelerator and target batch, experiments c1 and c2, as in Table \\ref{tab:cv}. The 'above 64' level is target 256 of experiment c1. The maximum is given beside the p90 because some batch levels hold as few as four configurations.",
                "tab:cv_batch", col_fmt="llllrr")
    dcv = v[v.cell.str.contains("c1|c2|c6|c7|c9")]
    head["variability"] = {"decode_repeat_cv_median": float(dcv.cv_repeat.median()), "decode_repeat_cv_p90": float(dcv.cv_repeat.quantile(0.9)),
                           "decode_pod_cv_median": float(dcv.cv_pod.median()),
                           "decode_same_pod_cv": [float(dcv[dcv.same_pod].cv_repeat.median()), float(dcv[dcv.same_pod].cv_repeat.quantile(0.9)), int(dcv.same_pod.sum())],
                           "decode_cross_pod_cv": [float(dcv[~dcv.same_pod].cv_repeat.median()), float(dcv[~dcv.same_pod].cv_repeat.quantile(0.9)), int((~dcv.same_pod).sum())], "decode_pod_cv_p90": float(dcv.cv_pod.quantile(0.9)),
                           "n_decode_repeat": int(dcv.cv_repeat.notna().sum()), "n_decode_pod": int(dcv.cv_pod.notna().sum()),
                           "n_configs_in_tables": int(len(v)), "n_two_pod_in_tables": int((v.pods >= 2).sum()),
                           "excluded_configs": excluded, "by_cell": out,
                           "cell_median_range": [float(pd.DataFrame([{"m": g.cv_repeat.median()} for _, g in v.groupby("cell")]).m.min()),
                                                 float(pd.DataFrame([{"m": g.cv_repeat.median()} for _, g in v.groupby("cell")]).m.max())],
                           "cell_p90_range": [float(pd.DataFrame([{"m": g.cv_repeat.quantile(0.9)} for _, g in v.groupby("cell")]).m.min()),
                                              float(pd.DataFrame([{"m": g.cv_repeat.quantile(0.9)} for _, g in v.groupby("cell")]).m.max())],
                           "c0_pod_cv_median": float(v[v.cell == "c0_static"].cv_pod.median())}
    top = dec[dec.batch_bin == "above 64"]
    head["variability"]["above64"] = {a: {"repeat_cv_median": float(g.cv_repeat.median()), "repeat_cv_max": float(g.cv_repeat.max()),
                                          "pod_cv_median": float(g.cv_pod.median()) if g.cv_pod.notna().any() else None, "n": int(len(g)),
                                          "same_pod": int(g.same_pod.sum())}
                                      for a, g in top.groupby("accelerator")}


def idle_share(df: pd.DataFrame, head: dict) -> None:
    d = df[(df.cell == "c1_decode_batch") & df.decode_ok].copy()
    d["idle_share"] = d.idle_power_w / d.mean_power_w
    t = d.groupby(["accelerator", "v_target_batch"]).idle_share.agg(["median", "min", "max"]).reset_index()
    rows = [{"Accelerator": ACC_LABEL[r.accelerator], "Target B": int(r.v_target_batch), "Idle share of board power, median": f"{r['median']:.0%}",
             "min to max": f"{r['min']:.0%} to {r['max']:.0%}"} for _, r in t.iterrows()]
    write_table("idle_share", pd.DataFrame(rows),
                "Idle share of decode power in the batch experiment (c1: 1k cached context, 200 fresh input and 256 output tokens). For each run, the board power the pod drew while idle with the model loaded, the mean of the 60 s idle windows before and after the run, divided by the mean board power of its decode window. Gross energy per token includes this share. Median, minimum and maximum over the runs of all four models at each target batch. No band, because each value is a ratio of two means from the same run.",
                "tab:idle_share", col_fmt="lrll")
    head["idle_share"] = {f"{r.accelerator}/B{int(r.v_target_batch)}": [float(r['median']), float(r['min']), float(r['max'])] for _, r in t.iterrows()}
    head["idle_share_summary"] = {"median_range": [float(t["median"].min()), float(t["median"].max())],
                                  "span": [float(t["min"].min()), float(t["max"].max())]}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e3e3e0", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8a86",
                         "xtick.color": "#4a4a47", "ytick.color": "#4a4a47", "figure.dpi": 150,
                         "pdf.fonttype": 42, "font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans"]})
    return plt


def _save(fig, name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", dpi=200)


def fig_decode_batch(df: pd.DataFrame, o: pd.DataFrame):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    raw = df[(df.cell == "c1_decode_batch") & df.decode_ok]
    for ax, acc in zip(axes, ["h100", "b200"]):
        for m in MODEL_ORDER:
            g = o[(o.model == m) & (o.accelerator == acc)].sort_values("B")
            r = raw[(raw.model == m) & (raw.accelerator == acc)]
            if not len(g):
                continue
            ax.scatter(r.B, r.e_d_j_per_tok, s=10, color=COLOR[m], alpha=0.45, linewidths=0, zorder=2)
            lo, hi = g.e_lo.where(g.e_n > 1, g.e_mean), g.e_hi.where(g.e_n > 1, g.e_mean)
            ax.fill_between(g.B, lo, hi, color=COLOR[m], alpha=0.18, linewidth=0)
            tp = int(g.tp.iloc[0])
            ax.plot(g.B, g.e_mean, color=COLOR[m], marker=MARKER[m], ms=4, lw=1.4,
                    label=MODEL_LABEL[m], zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{ACC_LABEL[acc]}, fp8 weights and KV")
        ax.set_xlabel("Achieved batch (mean running requests)")
        ax.set_xticks([1, 4, 16, 64]); ax.set_xticklabels(["1", "4", "16", "64"])
    axes[0].set_ylabel("Gross board energy, J per output token")
    axes[0].legend(frameon=False, loc="lower left")
    fig.text(0.5, -0.10, "Points: runs. Lines: mean. Bands: range over the two pods. 1k cached context, 256 output tokens. Llama-3.3-70B: two H100s (TP2), one B200.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig1_decode_batch")
    plt.close(fig)


def fig_decode_context(df: pd.DataFrame, o: pd.DataFrame):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=True)
    raw = df[(df.cell == "c2_decode_context") & df.decode_ok]
    for ax, acc in zip(axes, ["h100", "b200"]):
        for m in MODEL_ORDER:
            for tb, ls in ((8, "--"), (64, "-")):
                g = o[(o.model == m) & (o.accelerator == acc) & (o.target_batch == tb) & (o.flag == "")].sort_values("L")
                flagged = {(x.target_as_run, x.context) for x in o[(o.model == m) & (o.accelerator == acc) & (o.flag != "")].itertuples()}
                r = raw[(raw.model == m) & (raw.accelerator == acc) & (raw.design_batch == tb)]
                r = r[[(int(t), int(c)) not in flagged for t, c in zip(r.v_target_batch, r.v_context_cached)]]
                if not len(g):
                    continue
                ax.scatter(r.L, r.e_d_j_per_tok, s=9, color=COLOR[m], alpha=0.4, linewidths=0)
                lo, hi = g.e_lo.where(g.e_n > 1, g.e_mean), g.e_hi.where(g.e_n > 1, g.e_mean)
                ax.fill_between(g.L, lo, hi, color=COLOR[m], alpha=0.15, linewidth=0)
                ax.plot(g.L, g.e_mean, ls=ls, color=COLOR[m], marker=MARKER[m], ms=3.5, lw=1.3,
                        label=f"{MODEL_LABEL[m]}, B{tb}" if acc == "h100" else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{ACC_LABEL[acc]}")
        ax.set_xlabel("Attended context L (tokens)")
        ax.set_xticks([1024, 4096, 16384, 65536, 131072]); ax.set_xticklabels(["1k", "4k", "16k", "64k", "128k"])
    axes[0].set_ylabel("J per output token")
    fig.legend(frameon=False, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.15))
    fig.text(0.5, -0.21, "Dashed: design target batch 8. Solid: design target 64 (resized to the cache maximum at long context). Bands: range over pods. Collapsed configurations omitted.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig2_decode_context")
    plt.close(fig)


def fig_prefill(df: pd.DataFrame, o: pd.DataFrame):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    raw = df[(df.cell == "c3_prefill_miss") & df.usable & df.e_p_j_per_tok.notna()]
    for ax, acc in zip(axes, ["h100", "b200"]):
        for m in MODEL_ORDER:
            for tb, ls in ((1, "--"), (8, "-")):
                g = o[(o.model == m) & (o.accelerator == acc) & (o.target_batch == tb)].sort_values("fresh_in")
                r = raw[(raw.model == m) & (raw.accelerator == acc) & (raw.v_target_batch == tb)]
                if not len(g):
                    continue
                ax.scatter(r.v_fresh_in, r.e_p_j_per_tok, s=9, color=COLOR[m], alpha=0.4, linewidths=0)
                lo, hi = g.e_lo.where(g.e_n > 1, g.e_mean), g.e_hi.where(g.e_n > 1, g.e_mean)
                ax.fill_between(g.fresh_in, lo, hi, color=COLOR[m], alpha=0.15, linewidth=0)
                ax.plot(g.fresh_in, g.e_mean, ls=ls, color=COLOR[m], marker=MARKER[m], ms=3.5, lw=1.3,
                        label=f"{MODEL_LABEL[m]}, B{tb}" if acc == "h100" else None)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{ACC_LABEL[acc]}")
        ax.set_xlabel("Prompt length (tokens, cache miss)")
        ax.set_xticks([1024, 4096, 16384, 65536, 131072]); ax.set_xticklabels(["1k", "4k", "16k", "64k", "128k"])
    axes[0].set_ylabel("J per prompt token")
    fig.legend(frameon=False, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.17))
    fig.text(0.5, -0.24, "Dashed: one request at a time. Solid: eight concurrent. Prefix caching off; one output token. Bands: range over the two pods.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig3_prefill")
    plt.close(fig)


def fig_cached(o: pd.DataFrame):
    """Cached over uncached energy per request against the cached prefix length (log scale): columns are the
    GPUs, rows the fresh suffix; one colour per model, dashed for one request at a time, solid for eight."""
    plt = _style()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
    prefixes = [1024, 4096, 16384]
    for r_i, fresh in enumerate([200, 2048]):
        for c_i, acc in enumerate(["h100", "b200"]):
            ax = axes[r_i][c_i]
            for i, m in enumerate(MODEL_ORDER):
                for tb, ls in ((1, "--"), (8, "-")):
                    g = o[(o.model == m) & (o.accelerator == acc) & (o.target_batch == tb) & (o.fresh_in == fresh)].sort_values("prefix_cached")
                    if not len(g):
                        continue
                    xs = g.prefix_cached * (1 + (i - 1.5) * 0.025)   # small offsets so the bars do not overlap
                    ax.errorbar(xs, g.r_mean, yerr=[g.r_mean - g.r_lo, g.r_hi - g.r_mean], ls=ls, lw=1.3, marker=MARKER[m], ms=3.5,
                                color=COLOR[m], ecolor=COLOR[m], elinewidth=0.8, capsize=1.5,
                                label=f"{MODEL_LABEL[m]}, {'one at a time' if tb == 1 else 'eight concurrent'}" if (r_i, c_i) == (0, 0) else None)
            ax.axhline(1.0, color="#8a8a86", lw=0.8, ls=":")
            ax.set_xscale("log", base=2)
            ax.set_xticks(prefixes); ax.set_xticklabels(["1k", "4k", "16k"])
            ax.minorticks_off()
            ax.set_title(f"{ACC_LABEL[acc]}, {fresh:,} fresh tokens", fontsize=9)
            if r_i == 1:
                ax.set_xlabel("Cached prefix (tokens)")
            if c_i == 0:
                ax.set_ylabel("Energy per request,\ncached / uncached")
    fig.legend(frameon=False, ncol=4, fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, -0.09))
    fig.text(0.5, -0.13, "Dashed: one request at a time. Solid: eight concurrent. Bars: range over pods. 200 output tokens on both sides.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig4_cached_prefix")
    plt.close(fig)


def agent_vs_chat(agent: pd.DataFrame, o2: pd.DataFrame, head: dict) -> None:
    """Energy per output token of the concurrent agent replay over short-context decode
    (1k cached context, design target batch 8) of the same model on the same accelerator."""
    rows = []
    conc = agent[agent.agents > 1]
    for _, r in conc.iterrows():
        b = o2[(o2.model == r.model) & (o2.accelerator == r.accelerator) & (o2.target_batch == 8) & (o2.context == 1024) & (o2.flag == "")]
        if len(b):
            rows.append(dict(model=r.model, accelerator=r.accelerator, agents=int(r.agents), agent_j=float(r.j_per_out_tok),
                             chat_j=float(b.e_mean.iloc[0]), ratio=float(r.j_per_out_tok / b.e_mean.iloc[0])))
    t = pd.DataFrame(rows)
    head["agent_vs_chat"] = {"rows": t.to_dict("records"), "ratio_range": [float(t.ratio.min()), float(t.ratio.max())],
                             "ratio_median": float(t.ratio.median())}


def roofline(df: pd.DataFrame, o1: pd.DataFrame, head: dict) -> None:
    """At one request at a time, decode is bounded by reading the weights once per token.
    Compare the measured token rate with n_gpus * HBM bandwidth / weight bytes per step."""
    d = df[(df.cell == "c1_decode_batch") & df.decode_ok & (df.v_target_batch == 1)]
    rows, hl = [], {}
    for (m, a), g in d.groupby(["model", "accelerator"]):
        w = float(g.W_batch_bytes.mean())
        n = int(g.n_gpus.iloc[0])
        bound = n * HBM_BYTES_PER_S[a] / w
        tps = float(g.tokens_per_s.mean())
        idle = float(g.idle_power_w.mean() / g.mean_power_w.mean())
        rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "GPUs": n, "Weight bytes per step (GB)": f"{w / 1e9:.1f}",
                     "Bound (tokens/s)": f"{bound:,.0f}", "Measured (tokens/s)": f"{tps:,.0f}", "Share of bound": f"{tps / bound:.0%}",
                     "Idle share of board power": f"{idle:.0%}"})
        hl[f"{m}/{a}"] = {"weight_bytes": w, "bound_tps": bound, "tps": tps, "share": tps / bound, "idle_share": idle}
    write_table("roofline", pd.DataFrame(rows),
                "How close the GPUs came to their speed limit at one request at a time (experiment c1, target batch 1). To write each output token, a GPU must read from memory every weight that the step uses, so its token rate cannot exceed its memory bandwidth divided by the bytes read. Weight bytes per step: all the weights for the dense models, and for the MoE models the shared weights plus the experts one token uses. Bound: the GPUs' combined memory bandwidth (NVIDIA's figures: 3.35 TB/s per H100, 8 TB/s per B200) divided by those bytes. Measured: tokens per second, mean over the two runs. Share of bound: measured over bound. The bound leaves out reading the KV cache and processing the fresh input, so the true limit is lower and the share understates how fully the bandwidth was used. Idle share: the board power drawn while idle with the model loaded, from the idle windows bracketing each run, over the mean board power of the decode window.",
                "tab:roofline", col_fmt="llrrrrrr")
    head["roofline"] = hl
    head["roofline_summary"] = {"share_range": [min(v["share"] for v in hl.values()), max(v["share"] for v in hl.values())],
                                "b200_below_h100_every_model": all(hl[f"{m}/b200"]["share"] < hl[f"{m}/h100"]["share"] for m in MODEL_ORDER)}


def factors(o1, o2, o4, o6, o7, df, head) -> pd.DataFrame:
    """One row per measured factor: the effect on energy per token (or per request) as a
    multiplier, aggregated over the model x accelerator comparisons. Every value is a ratio
    of central values already reported in the section that measures it."""
    def agg(name, comparison, vals, unit, tell, section, doublings=None):
        v = pd.Series(vals, dtype=float)
        r = dict(factor=name, comparison=comparison, median=float(v.median()), lo=float(v.min()), hi=float(v.max()),
                 n=int(len(v)), unit=unit, tell=tell, section=section)
        if doublings is not None:
            pdbl = pd.Series([x ** (1 / d) for x, d in zip(vals, doublings)], dtype=float)
            r.update(per_doubling=float(pdbl.median()), per_doubling_lo=float(pdbl.min()), per_doubling_hi=float(pdbl.max()),
                     doublings_lo=float(min(doublings)), doublings_hi=float(max(doublings)))
        return r
    out = []
    # batch: batch 1 over the deepest batch, per model x accelerator
    b, bd = [], []
    for _, g in o1.groupby(["model", "accelerator"]):
        lo, hi = g[g.target_batch == 1].iloc[0], g.sort_values("target_batch").iloc[-1]
        b.append(lo.e_mean / hi.e_mean)
        bd.append(math.log2(hi.B / lo.B))
    out.append(agg("Batch", "one request at a time over the deepest batch", b, "J per output token",
                   "the dominant lever on every model and both GPUs", "sec:batch", doublings=bd))
    # cached prefix: uncached over cached at 16k cached tokens (both fresh lengths, both concurrencies)
    c = (1 / o4[o4.prefix_cached == 16384].r_mean).tolist()
    out.append(agg("Cached prefix", "fresh request over one with 16k tokens already cached", c, "J per request",
                   "large for long reused prompts, small (median 9%) at 1k", "sec:cached"))
    # cache boundary: collapsed over the last clear run at the same memory offer (Llama-3.3-70B, B200)
    kv = df[df.cell.isin(["c9_kv_pressure", "c9b_kv_pressure_diag"]) & df.decode_ok]
    k = []
    for _, g in kv.groupby("v_gpu_memory_utilization"):
        clear, coll = g[g.e_d_j_per_tok < 3], g[g.e_d_j_per_tok >= 3]
        if len(clear) and len(coll):
            ref = clear.sort_values("B").e_d_j_per_tok.iloc[-1]
            k += (coll.e_d_j_per_tok / ref).tolist()
    out.append(agg("Memory limit for stored context", "past the limit over just below it (Llama-3.3-70B, one B200)", k, "J per output token",
                   "a cliff, not a slope: admission limits and memory headroom reduce the risk, but the safe point depends on the workload and the scheduler", "sec:kv"))
    # context: longest shared context (up to 32k) over 1k, design batch 8 and 64, unflagged
    def ctx(tb):
        r, dbl = [], []
        for _, g in o2[(o2.target_batch == tb) & (o2.flag == "") & (o2.context <= 32768)].groupby(["model", "accelerator"]):
            g = g.sort_values("context")
            if len(g) >= 2 and g.context.iloc[0] == 1024:
                r.append(g.e_mean.iloc[-1] / g.e_mean.iloc[0])
                dbl.append(math.log2(g.context.iloc[-1] / g.context.iloc[0]))
        return r, dbl
    r64, d64 = ctx(64)
    out.append(agg("Context length, batch 64", "longest context whose runs kept their cache (4k for Llama-3.3-70B on H100, 16k on B200, 31k to 32k for the others) over 1k, design batch 64; long context forced a smaller batch on some servers", r64, "J per output token",
                   "rises with context and more steeply at deep batch, where the stored context fills memory", "sec:context", doublings=d64))
    r8, d8 = ctx(8)
    out.append(agg("Context length, batch 8", "longest context (16k to 32k) over 1k, design batch 8", r8, "J per output token",
                   "a moderate, steady rise", "sec:context", doublings=d8))
    # GPU generation: B200 over H100 on the same number of GPUs (two-sided); and, separately,
    # Llama-3.3-70B on one B200 against the two H100s it needs
    g = [v[0] for m in head["b200_same_gpus"].values() for t in m.values() for v in t.values()]
    out.append(agg("GPU generation, same GPU count", "B200 over H100 on the same number of GPUs, at the same batch", g, "J per output token",
                   "no consistent saving: lower for some models, higher for DeepSeek-V2-Lite", "sec:b200"))
    g = [v[0] for m in head["b200_fewer_gpus"].values() for t in m.values() for v in t.values()]
    out.append(agg("One B200 against two H100s", "one B200 over the two H100s Llama-3.3-70B needs, at the same batch; changes GPU generation and GPU count together, and at the highest load also the cache reuse", g, "J per output token",
                   "a saving because the larger memory holds the model on fewer GPUs", "sec:b200"))
    # precision: fp8 over nvfp4, excluding the row where both servers were at the cache boundary
    p = (1 / o7[~((o7.model == "llama-3.3-70b") & (o7.context == 16384) & (o7.target_batch == 64))].r_mean).tolist()
    out.append(agg("Weight precision", "8-bit over 4-bit weights, B200 (one pod per model)", p, "J per output token",
                   "a modest saving in the tested 4-bit deployments; one pod, so indicative", "sec:precision"))
    # tensor parallelism: higher TP over lower
    out.append(agg("Tensor parallelism", "a model split over more GPUs over fewer", o6.r_mean.tolist(), "J per output token",
                   "costs energy; buys throughput", "sec:tp"))
    # run-to-run noise: larger over smaller of the two runs, decode configurations
    v = pd.read_csv(TABLES / "variability_per_config.csv")
    rr = v[v.cell.str.contains("c1|c2|c6|c7|c9")].ratio_repeat.dropna()
    dv = rr
    out.append(dict(factor="Repeat-to-repeat variation", comparison="the larger over the smaller of two repeats", median=float(rr.median()),
                    lo=float(rr.min()), hi=float(rr.quantile(0.9)), n=int(len(dv)), unit="J per output token",
                    tell="small against most of the levers above, though not against the smallest", section="sec:variability", p90_only=True))
    f = pd.DataFrame(out)
    f["spread"] = [max(h, 1 / l) if l < 1 else h for l, h in zip(f.lo, f.hi)]
    rows = [{"Factor": r.factor, "Comparison": r.comparison,
             "Effect, median (range)": (f"{fmt(r.median)} (90th percentile {fmt(r.hi)})" if getattr(r, "p90_only", False) is True
                                        else f"{fmt(r.median)} ({fmt(r.lo)} to {fmt(r.hi)})"),
             "Comparisons": r.n,
             "Per doubling": (f"{r.per_doubling:.2f} ({r.per_doubling_lo:.2f} to {r.per_doubling_hi:.2f})"
                              if isinstance(getattr(r, "per_doubling", None), float) and not math.isnan(r.per_doubling) else "--"),
             "What the data tell us": r.tell} for r in f.itertuples()]
    write_table("factors", pd.DataFrame(rows),
                "The measured factors compared on one scale. Effect is the multiplier on energy per output token (per request for the cached prefix) between the two settings named in the Comparison column, as a ratio of central values: the median over the model and accelerator combinations, with the smallest and largest in parentheses. Comparisons counts those combinations (and, for the cached prefix, prompt shapes and concurrencies). Each row restates figures from the section that measures the factor; the GPU-generation row is two-sided because the direction depends on the model, and the GPU-memory row is one model (Llama-3.3-70B). Repeat-to-repeat variation is shown for scale, with its 90th percentile in place of a range.",
                "tab:factors", col_fmt="llllrl")
    # the generic writer sizes columns from the longest cell; this table needs wrapped text columns
    body = "\n".join(" & ".join(_tex_escape(str(x)) for x in r.values()) + r" \\" for r in rows)
    (TABLES / "factors.tex").write_text(
        "\\begin{landscape}\n\\begin{table}[p]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{" + (TABLES / "factors.md").read_text().split("**")[1] + "}\\label{tab:factors}\n"
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.5cm}>{\\raggedright\\arraybackslash}p{6.0cm}lr>{\\raggedright\\arraybackslash}p{2.7cm}>{\\raggedright\\arraybackslash}p{4.9cm}}\n"
        "\\toprule\nFactor & Comparison & \\makecell[l]{Effect, median\\\\(range)} & \\makecell[r]{Compar-\\\\isons} & \\makecell[l]{Per\\\\doubling} & What the data tell us \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\\end{landscape}\n")
    head["factors"] = f.to_dict("records")
    return f


def fig_factors(f: pd.DataFrame):
    plt = _style()
    f = f.sort_values("median", key=lambda s: [max(x, 1 / x) for x in s], ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for i, r in enumerate(f.itertuples()):
        col = "#8a8a86" if r.factor.startswith("Repeat") else "#0072B2"
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=5, alpha=0.35, solid_capstyle="butt")
        ax.plot(r.median, i, "o", color=col, ms=6)
        ax.text(r.hi * 1.12, i, f"{fmt(r.median)} ({fmt(r.lo)} to {fmt(r.hi)})", va="center", fontsize=7, color="#4a4a47")
    ax.axvline(1.0, color="#8a8a86", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.4, 200)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_yticks(range(len(f))); ax.set_yticklabels(f.factor, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Multiplier on energy per token (log scale)")
    ax.set_title("How much each measured factor moved energy per token")
    fig.text(0.5, -0.06, "Dot: median over models and GPUs. Bar: smallest to largest. Each comparison is defined in the table beside this figure.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_factors")
    plt.close(fig)


def fig_b200(head: dict):
    plt = _style()
    from matplotlib.ticker import NullFormatter
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})
    tbs = [1, 4, 16, 64, 256]
    LS = {1: "-", 2: "--", 4: ":"}
    def line(ax, m, r, ls, lab=None):
        xs = [tb for tb in tbs if tb in r]
        y = [r[tb][0] for tb in xs]; lo = [r[tb][1] for tb in xs]; hi = [r[tb][2] for tb in xs]
        ax.fill_between(xs, lo, hi, color=COLOR[m], alpha=0.13, linewidth=0)
        ax.plot(xs, y, color=COLOR[m], marker=MARKER[m], ms=4, lw=1.4, ls=ls, label=lab)
    ax = axes[0]
    for m in MODEL_ORDER:
        for tp, r in sorted(head["b200_same_gpus"].get(m, {}).items()):
            line(ax, m, r, LS[tp])
    ax.set_title("Same number of GPUs on each side")
    ax = axes[1]
    m = "llama-3.3-70b"
    line(ax, m, head["b200_fewer_gpus"][m][1], "-")
    line(ax, m, head["b200_same_gpus"][m][2], "--")
    ax.set_title("Llama-3.3-70B: one B200 vs two H100s")
    ax.annotate("one B200 vs two H100s", xy=(64, head["b200_fewer_gpus"][m][1][64][0]), xytext=(2.2, 0.55), fontsize=7, color="#4a4a47",
                arrowprops=dict(arrowstyle="-", color="#8a8a86", lw=0.6))
    ax.annotate("two B200s vs two H100s", xy=(16, head["b200_same_gpus"][m][2][16][0]), xytext=(1.2, 1.35), fontsize=7, color="#4a4a47",
                arrowprops=dict(arrowstyle="-", color="#8a8a86", lw=0.6))
    for ax in axes:
        ax.axhline(1.0, color="#8a8a86", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(tbs); ax.set_xticklabels(["1", "4", "16", "64", "deepest"])
        ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3]); ax.set_yticklabels(["0.5", "0.7", "1", "1.5", "2", "3"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Target batch (requests in flight)")
    axes[1].text(256, 1.03, "B200 uses more", fontsize=7, color="#4a4a47", va="bottom", ha="right")
    axes[1].text(256, 0.97, "B200 uses less", fontsize=7, color="#4a4a47", va="top", ha="right")
    axes[0].set_ylabel("J per output token, B200 / H100")
    handles = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ms=4, lw=1.4, label=MODEL_LABEL[m]) for m in MODEL_ORDER]
    handles += [Line2D([], [], color="#4a4a47", lw=1.2, ls=LS[t], label=f"{t} GPU{'s' if t > 1 else ''} each") for t in (1, 2, 4)]
    fig.legend(handles=handles, frameon=False, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.19, "Bands: lowest to highest combination of the two sides' pod values. 1k cached context, 256 output tokens.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_b200_h100")
    plt.close(fig)


def fig_context_multiplier(o2: pd.DataFrame):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, acc in zip(axes, ["h100", "b200"]):
        for m in MODEL_ORDER:
            for tb, ls in ((8, "--"), (64, "-")):
                g = o2[(o2.model == m) & (o2.accelerator == acc) & (o2.target_batch == tb) & (o2.context <= 32768)].sort_values("context")
                base = g[(g.context == 1024) & (g.flag == "")]
                if not len(base):
                    continue
                e0 = base.e_mean.iloc[0]
                ok = g[g.flag == ""]
                ax.plot(ok.context, ok.e_mean / e0, ls=ls, color=COLOR[m], lw=1.3, marker=MARKER[m], ms=3.5,
                        label=f"{MODEL_LABEL[m]}, B{tb}" if acc == "h100" else None)
                rs = ok[ok.target_as_run < tb]
                ax.scatter(rs.context, rs.e_mean / e0, s=46, facecolors="none", edgecolors=COLOR[m], linewidths=1.0, zorder=4)
                fl = g[g.flag != ""]
                for x in fl.itertuples():
                    ax.scatter([x.context] * len(x.e_runs), [e / e0 for e in x.e_runs], marker="x", s=22, color=COLOR[m], zorder=4)
        ax.axhline(1.0, color="#8a8a86", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(ACC_LABEL[acc])
        ax.set_xlabel("Cached context (tokens)")
        ax.set_xticks([1024, 4096, 16384, 32768]); ax.set_xticklabels(["1k", "4k", "16k", "32k"])
        ax.set_yticks([1, 2, 5, 10]); ax.set_yticklabels(["1", "2", "5", "10"])
    axes[0].set_ylabel("J per output token, relative to 1k")
    fig.legend(frameon=False, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.17))
    fig.text(0.5, -0.25, "Dashed: design batch 8. Solid: design batch 64. Circled: the server could not hold the design batch at that context and ran a smaller one.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_context_multiplier")
    plt.close(fig)


def fig_tp(tp: pd.DataFrame):
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    size = {1: 3.5, 16: 5.5, 64: 7.5}
    # label positions in data coordinates, chosen to clear the markers and bars
    label_at = {("qwen3-8b", "h100", 1): (1.325, 1.33, "left"), ("qwen3-8b", "b200", 1): (1.33, 1.045, "left"),
                ("llama-3.3-70b", "b200", 1): (1.58, 1.225, "center"), ("llama-3.3-70b", "b200", 2): (1.30, 1.25, "left"),
                ("llama-3.3-70b", "h100", 2): (1.53, 1.015, "center")}
    for (m, acc, lo_tp), g in tp.groupby(["model", "accelerator", "tp_lo"], sort=False):
        g = g.sort_values("target_batch")
        hi_tp = int(g.tp_hi.iloc[0])
        ax.plot(g.tps_ratio, g.r_mean, color=COLOR[m], lw=0.9, alpha=0.6)
        for _, r in g.iterrows():
            ax.errorbar(r.tps_ratio, r.r_mean, yerr=[[r.r_mean - r.r_lo], [r.r_hi - r.r_mean]], fmt=MARKER[m],
                        ms=size.get(int(r.target_batch), 5), color=COLOR[m], mfc=COLOR[m] if acc == "h100" else "white",
                        ecolor=COLOR[m], elinewidth=0.8, capsize=1.5)
        x, y, ha = label_at[(m, acc, int(lo_tp))]
        ax.text(x, y, f"{MODEL_LABEL[m]}, {ACC_LABEL[acc]}, {int(lo_tp)} to {hi_tp} GPUs", fontsize=6.6, color=COLOR[m], ha=ha, va="center")
    ax.axhline(1.0, color="#8a8a86", lw=0.8, ls=":")
    ax.set_xlim(1.0, 1.75); ax.set_ylim(0.97, 1.5)
    ax.set_xlabel("Throughput gained (tokens per second, more GPUs / fewer)")
    ax.set_ylabel("Energy cost (J per token, more GPUs / fewer)")
    ax.set_title("What splitting a model over more GPUs bought and cost")
    fig.text(0.5, -0.06, "Filled: H100. Open: B200. Marker size: target batch 1, 16, 64 (small to large). Bars: range over pods.",
             ha="center", fontsize=7.0, color="#4a4a47")
    _save(fig, "fig_tp")
    plt.close(fig)


def fig_precision(pr: pd.DataFrame):
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    k = 0; ticks, tl = [], []
    for m, g in pr.groupby("model", sort=False):
        g = g.sort_values(["context", "target_batch"])
        for _, r in g.iterrows():
            ax.errorbar(k, r.r_mean, yerr=[[r.r_mean - r.r_lo], [r.r_hi - r.r_mean]], fmt=MARKER[m], ms=4, color=COLOR[m],
                        ecolor=COLOR[m], elinewidth=0.9, capsize=1.5)
            if r.model == "llama-3.3-70b" and r.context == 16384 and r.target_batch == 64:
                ax.annotate("both servers at the memory limit;\none 8-bit run collapsed", (k, r.r_hi), xytext=(8, -6),
                            textcoords="offset points", fontsize=6.2, color="#4a4a47", va="top")
            ticks.append(k); tl.append(f"{r.context // 1024}k\nB{r.target_batch}")
            k += 1
        k += 0.6
    ax.axhline(1.0, color="#8a8a86", lw=0.8, ls=":")
    ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=6.6)
    ax.set_ylabel("J per token, 4-bit (nvfp4) / 8-bit (fp8)")
    ax.set_title("Weight precision on B200, one pod per model")
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="", ms=4, label=MODEL_LABEL[m]) for m in pr.model.unique()]
    ax.legend(handles=handles, frameon=False, fontsize=6.8, loc="upper left")
    fig.text(0.5, -0.08, "Labels: cached context and design batch. Bars: lowest to highest combination of the two sides' runs, each side on its own pod.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_precision")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    args = ap.parse_args()
    df = load(args.features)
    assert len(df) == 917, f"expected 917 runs, found {len(df)}"
    boot = Bands()
    feat = args.features.resolve()
    head: dict = {"features": str(feat.relative_to(ROOT)) if feat.is_relative_to(ROOT) else feat.name,
                  "store": store_facts(args.features)}
    corpus_tables(df, head)
    idle_table(df, boot, head)
    o1 = decode_batch(df, boot, head)
    o2 = decode_context(df, boot, head)
    o3 = prefill(df, boot, head)
    o4 = cached_prefix(df, boot, head)
    o6 = tensor_parallel(df, boot, head)
    o7 = precision(df, boot, head)
    o8 = agent_replay(df, boot, head)
    kv_pressure(df, head)
    variability(df, head)
    idle_share(df, head)
    agent_vs_chat(o8, o2, head)
    roofline(df, o1, head)
    fac = factors(o1, o2, o4, o6, o7, df, head)
    fig_decode_batch(df, o1)
    fig_decode_context(df, o2)
    fig_prefill(df, o3)
    fig_cached(o4)
    fig_tp(o6)
    fig_precision(o7)
    fig_b200(head)
    fig_context_multiplier(o2)
    fig_factors(fac)
    (TABLES / "headline.json").write_text(json.dumps(head, indent=1, default=str))
    print(json.dumps({k: head[k] for k in ("runs", "usable_runs", "configs", "configs_two_repeats", "configs_two_pods", "pods", "data_centers", "metered_hours")}, indent=1))


if __name__ == "__main__":
    main()
