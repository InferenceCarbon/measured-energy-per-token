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
               "deepseek-v2-lite": "DeepSeek-V2-Lite", "llama-3.3-70b": "Llama-3.3-70B",
               "gemma-3-27b": "Gemma-3-27B", "gpt-oss-120b": "gpt-oss-120b",
               "qwen3-235b-a22b": "Qwen3-235B-A22B", "llama-3.1-405b": "Llama-3.1-405B"}
# v1.1: the six models of the main tables; the two size-run models have their own section (series `size`)
MODEL_ORDER = ["qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite", "llama-3.3-70b", "gemma-3-27b", "gpt-oss-120b"]
SIZE_MODELS = ["qwen3-235b-a22b", "llama-3.1-405b"]
ACC_ORDER = ["h100", "h200", "b200"]
ACC_LABEL = {"h100": "H100", "h200": "H200", "b200": "B200"}
# Okabe-Ito, validated with the dataviz palette checker (CVD dE >= 11, normal >= 24).
COLOR = {"qwen3-8b": "#0072B2", "qwen3-30b-a3b": "#E69F00", "deepseek-v2-lite": "#009E73",
         "llama-3.3-70b": "#D55E00", "gemma-3-27b": "#CC79A7", "gpt-oss-120b": "#56B4E9",
         "qwen3-235b-a22b": "#000000", "llama-3.1-405b": "#8a8a86"}
MARKER = {"qwen3-8b": "o", "qwen3-30b-a3b": "s", "deepseek-v2-lite": "^", "llama-3.3-70b": "D",
          "gemma-3-27b": "v", "gpt-oss-120b": "P", "qwen3-235b-a22b": "X", "llama-3.1-405b": "*"}
VARIANT_KEYS = ("target_batch", "context_cached", "context_cap", "fresh_in", "prefix_cached", "control",
                "weight_precision", "clock_mode", "gpu_memory_utilization", "speculation")
_VAR_RE = re.compile("(" + "|".join(VARIANT_KEYS) + r")-([^_/]+)")
# Design grids of the varied batch per cell (config/cells.yaml). A target as run that is not on
# the grid was resized DOWN by the feasibility generator, so its design point is the smallest
# grid value at or above it.
# v1.0.1 cache-reuse rule (data/cache_reuse.csv): a decode-type run keeps its stated cached-context shape only if the
# in-window prefix-cache hit share was at least this fraction of the expected one. The cut sits in an empty gap of the
# observed distribution (no run between 0.85 and 0.97). c9 and c9b are exempt: eviction is what they measure.
CACHE_REUSE_MIN = 0.95
CACHE_RULE_CELLS = ("c1_decode_batch", "c2_decode_context", "c6_tensor_parallel", "c7_precision", "c10_speculation")
# keyed on the cell as run (column `cell_run`): several run cells feed one experiment of the paper
DESIGN_BATCH = {"c1_decode_batch": [1, 4, 16, 64, 256], "c2_decode_context": [8, 64], "c3_prefill_miss": [1, 8],
                "c4_cached_prefix": [1, 8], "c6_tensor_parallel": [1, 16, 64], "c7_precision": [8, 64],
                "c8_agent_replay": [1, 8], "c9_kv_pressure": [12, 16, 20, 24, 28, 32], "c9b_kv_pressure_diag": [12, 20, 28, 32],
                "b_top_batch_c1": [256], "b_top_batch_c2": [64], "c_batch_points": [8, 32, 128],
                "d_context_ladder_decode": [1, 8, 32], "d_context_ladder_prefill": [1, 8],
                "e_nvfp4_ladder": [1, 8, 32, 64, 128], "e_nvfp4_prefill_miss": [1, 8], "e_nvfp4_cached_prefix": [1, 8],
                "f_speculation": [1, 8, 32], "j_high_batch": [512, 1024, 2048]}
for _p in ("g_h200", "h_added", "i_size"):
    DESIGN_BATCH.update({_p + "_decode_batch": [1, 8, 32, 64, 128], _p + "_decode_context": [8, 64],
                         _p + "_prefill_miss": [1, 8], _p + "_cached_prefix": [1, 8]})
IN_USE = ("used", "kept-marked")


def design_target(cell: str, tb: float) -> int:
    grid = DESIGN_BATCH.get(cell, [])
    for g in grid:
        if tb <= g:
            return g
    return int(tb)
REPLAY_OUT_PER_STEP = 300
REPLAY_CONTEXT_START, REPLAY_CONTEXT_STEP = 10240, 2048
NATIVE_CONTEXT = {"qwen3-8b": 32768, "qwen3-30b-a3b": 32768, "deepseek-v2-lite": 32768, "llama-3.3-70b": 131072,
                  "gemma-3-27b": 131072, "gpt-oss-120b": 131072, "qwen3-235b-a22b": 262144, "llama-3.1-405b": 131072}
RATED_W = {"h100": 700.0, "h200": 700.0, "b200": 1000.0}   # the power limit the driver reported on every board
# HBM bandwidth per GPU from NVIDIA's product pages (H100 SXM 3.35 TB/s, H200 4.8 TB/s, B200 8 TB/s); see refs.bib nvidia_specs
HBM_BYTES_PER_S = {"h100": 3.35e12, "h200": 4.8e12, "b200": 8.0e12}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load(path: Path) -> pd.DataFrame:
    """The corpus, one row per reading. v1.1: readings from version 1.0.0 and from the replication, keyed on `uid`
    (run ids repeat between the two). `cell` is the paper's experiment; `cell_run` is the cell the harness ran.
    A reading the register leaves out arrives with its reason in `dropped` and never reaches a table."""
    df = pd.read_parquet(path)
    assert not df.uid.duplicated().any()
    left_out = ~df.disposition.isin(IN_USE)
    assert df.loc[left_out, "dropped"].notna().all(), "a reading left out of the tables has no recorded reason"
    assert df.loc[~left_out, "dropped"].isna().all()
    df["cell_run"] = df.cell
    df["cell"] = df.experiment
    var = df.run_id.str.split("/").str[4]
    for k in VARIANT_KEYS:
        df["v_" + k] = var.map(lambda s, k=k: dict(_VAR_RE.findall(s)).get(k))
    for k in ("target_batch", "context_cached", "context_cap", "fresh_in", "prefix_cached"):
        df["v_" + k] = pd.to_numeric(df["v_" + k], errors="coerce")
    df["v_gpu_memory_utilization"] = pd.to_numeric(df["v_gpu_memory_utilization"], errors="coerce")
    # The memory offer each server actually ran with, read from its own launch command (data/server_settings.csv,
    # written by tools/server_settings.py from the raw records). Run ids name the offer only where it was a varied setting.
    ss = path.parent / "server_settings.csv"
    if ss.exists():
        offer = pd.read_csv(ss).set_index("uid").gpu_memory_utilization
        df["v_gpu_memory_utilization"] = df.uid.map(offer).fillna(df["v_gpu_memory_utilization"])
    df["v_gpu_memory_utilization"] = df["v_gpu_memory_utilization"].fillna(0.9)
    df["v_speculation"] = df["v_speculation"].fillna("none")
    df["design_batch"] = [design_target(c, t) if t == t else -1 for c, t in zip(df.cell_run, df.v_target_batch)]
    # The load client of version 1.0.0 held at most 100 connections, so a target above 100 offered 100 requests.
    df["target_as_named"] = df.v_target_batch
    capped = df.client_cap.notna() & (df.v_target_batch > df.client_cap)
    df.loc[capped, "v_target_batch"] = df.loc[capped, "client_cap"]
    # a configuration: one shape on one model and hardware, whichever run cell or source measured it
    shape = [re.sub(r"target_batch-\d+", "target_batch-%d" % t, v) if t == t else v for v, t in zip(var, df.v_target_batch)]
    df["config"] = [f"{a}/{m}/tp{tp}/{c}/{v}" for a, m, tp, c, v in zip(df.accelerator, df.model, df.tp, df.cell, shape)]
    cr = path.parent / "cache_reuse.csv"
    c = pd.read_csv(cr)[["uid", "cache_reuse_r"]]
    df = df.merge(c, on="uid", how="left")
    # the cache rule is applied when the corpus is frozen; here it is only checked
    missed = (df.cache_reuse_r < CACHE_REUSE_MIN) & df.cell.isin(CACHE_RULE_CELLS) & (df.phase == "decode") & df["dropped"].isna()
    assert not missed.any(), f"{int(missed.sum())} decode runs below the cache-reuse rule are not marked as left out"
    df["usable"] = df["dropped"].isna() & df.energy_j.notna()
    df["decode_ok"] = df.usable & df.e_d_j_per_tok.notna() & (df.phase == "decode")
    df["rated_w"] = df.accelerator.map(RATED_W) * df.n_gpus
    st = path.parent / "store_tables" / "cells.parquet"
    sched = ["sched_kv_usage_mean", "sched_waiting_mean", "sched_running_max", "sched_preemptions_delta",
             "sched_preemptions_counter_present"]
    a = pd.read_parquet(st)
    a = a[a.kind == "active"][["uid"] + sched].drop_duplicates("uid")
    df = df.merge(a, on="uid", how="left")
    assert not df.loc[left_out.to_numpy(), "usable"].any()
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
    if name.startswith("size_") and name != "size_batch_summary":
        caption = "Two larger models: " + caption[0].lower() + caption[1:]    # these tables repeat the captions of the six-model tables
    caption = caption.replace("\\%", "%").replace("%", "\\%").replace("`", "'")    # a percent sign formatted into a caption must be escaped
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
    # the PDF font has no curly quotation marks: write TeX quotes in the .tex file (the .md and .csv keep the cells as they are)
    tex = tex.replace("\u2018", "`").replace("\u2019", "'").replace("\u201c", "``").replace("\u201d", "''")
    (TABLES / f"{name}.tex").write_text(tex)
    md = [f"**{cap}**", "", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(x) for x in row) + " |" for row in df.astype(str).itertuples(index=False)]
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------

LEFT_OUT_LABEL = {"excluded-cache": "Cached context evicted during the window (cache reuse below 0.95)",
                  "excluded-no-window": "No usable measurement window",
                  "excluded-server-stopped": "Server stopped during the window",
                  "superseded": "Superseded by a later measurement of the same configuration",
                  "duplicate": "Copy of a reading already in the corpus",
                  "excluded": "Pilot run made before the experiment was fixed",
                  "retried": "Earlier attempt of a run that was run again"}


def corpus_tables(df: pd.DataFrame, head: dict, data_dir: Path) -> None:
    cfg = df.groupby("config").agg(n=("run_id", "size"), nu=("usable", "sum"), pods=("pod_id", "nunique"), cell=("cell", "first"))
    use = df[df.usable]
    cfg_u = use.groupby("config").agg(n=("run_id", "size"), pods=("pod_id", "nunique"), cell=("cell", "first"))
    head.update(runs=int(len(df)), usable_runs=int(df.usable.sum()), configs=int(len(cfg)),
                configs_two_repeats=int((cfg.n >= 2).sum()), configs_two_usable=int((cfg.nu >= 2).sum()),
                configs_two_pods=int((cfg_u.pods >= 2).sum()), configs_one_pod=int((cfg_u.pods < 2).sum()),
                configs_three_pods=int((cfg_u.pods >= 3).sum()), configs_used=int(len(cfg_u)),
                configs_no_usable=int((cfg.nu == 0).sum()), configs_one_run=int((cfg_u.n == 1).sum()),
                pods=int(df.pod_id.nunique()), data_centers=int(df.data_center.nunique()),
                pods_by_accelerator=df.groupby("accelerator").pod_id.nunique().to_dict(),
                metered_hours=float(df.duration_s.sum() / 3600),
                models=sorted(df.model.unique()), window_rows_dropped=int((~df.usable).sum()),
                runs_by_source=df.groupby("source").size().to_dict(),
                usable_by_source=use.groupby("source").size().to_dict(),
                usable_by_series=use.groupby("series").size().to_dict(),
                usable_by_accelerator=use.groupby("accelerator").size().to_dict(),
                kept_marked=int((df.disposition == "kept-marked").sum()),
                pods_in_use=int(use.pod_id.nunique()),
                six_models={"runs_in_use": int((use.series == "main").sum()), "configs_used": int(use[use.series == "main"].config.nunique()),
                            "pods_in_use": int(use[use.series == "main"].pod_id.nunique()), "data_centers": int(use[use.series == "main"].data_center.nunique())},
                size_models={"runs_in_use": int((use.series == "size").sum()), "configs_used": int(use[use.series == "size"].config.nunique()),
                             "pods_in_use": int(use[use.series == "size"].pod_id.nunique())},
                top12_pod_share_all_runs=float(df.groupby("pod_id").size().sort_values(ascending=False).head(12).sum() / len(df)),
                one_pod_configs_by_cell=cfg_u[cfg_u.pods < 2].cell.value_counts().to_dict())
    # configurations that appear in no results table (the omission rule, Appendix C)
    # every prefill and decode run in use supplies a figure (runs that returned no tokens are left out with a reason)
    assert use[use.phase == "prefill"].e_p_j_per_tok.notna().all() and use[use.phase == "decode"].e_d_j_per_tok.notna().all()
    omitted = {"c7 fp8-only runs of the MoE models (no nvfp4 checkpoint)": int(((df.cell == "c7_precision") & df.model.isin(["qwen3-30b-a3b", "deepseek-v2-lite"])).sum()),
               "readings left out, with a recorded reason": int((~df.usable).sum())}
    head["omitted_runs"] = omitted
    # readings in use: experiment x accelerator rows, one column per model
    models = MODEL_ORDER + SIZE_MODELS
    t = use.groupby(["cell", "accelerator", "model"]).size().unstack("model", fill_value=0).reindex(columns=models, fill_value=0)
    t = t.reindex([(c, a) for c in sorted(use.cell.unique(), key=_exp_key) for a in ACC_ORDER if (c, a) in t.index])
    t.columns = [MODEL_LABEL[m] for m in t.columns]
    t["Total"] = t.sum(axis=1)
    tot = t.sum()
    t = t.reset_index().rename(columns={"cell": "Experiment", "accelerator": "Acc."})
    t["Acc."] = t["Acc."].map(ACC_LABEL)
    t.loc[len(t)] = ["Total", ""] + [int(x) for x in tot]
    write_table("corpus_cells", t, f"Runs used in the tables, by experiment, accelerator and model ({len(use):,} of the {len(df):,} runs in the corpus; Table \\ref{{tab:left_out}} counts the rest). "
                "The last two columns are the two larger models of Section \\ref{sec:size}.", "tab:corpus",
                col_fmt="ll" + "r" * (len(t.columns) - 2), small=True, longtable=True)
    # pods
    p = df.groupby("pod_id").agg(**{"Data center": ("data_center", "first"), "Accelerator": ("accelerator", "first"),
                                    "GPUs": ("n_gpus", "max"), "Models (count)": ("model", lambda s: len(set(s))),
                                    "Runs": ("run_id", "size"), "Metered hours": ("duration_s", lambda s: s.sum() / 3600),
                                    "Source": ("source", lambda s: "version 1.0" if set(s) == {"v1.0.0"} else "replication")})
    p["acc_rank"] = p["Accelerator"].map({a: i for i, a in enumerate(ACC_ORDER)})
    p["Accelerator"] = p["Accelerator"].map(ACC_LABEL)
    p["Metered hours"] = p["Metered hours"].map(lambda x: f"{x:.2f}")
    p = p.sort_values(["acc_rank", "Source", "Runs"], ascending=[True, False, False]).drop(columns="acc_rank").reset_index().rename(columns={"pod_id": "Pod"})
    p["Pod"] = [f"pod {i + 1}" for i in range(len(p))]
    write_table("pods", p, f"The {len(p)} rented pods. Pod identifiers are replaced by an index (the mapping is in the deposited corpus). Runs counts every run in the corpus, including those left out of the tables. Metered hours are the summed lengths of the metered windows on that pod, warm-up, idle windows and model loads excluded.",
                "tab:pods", small=True, longtable=True)
    # readings left out of every table, with the reason (data/readings_left_out.csv lists each one)
    lo = pd.read_csv(data_dir / "readings_left_out.csv", dtype=str)
    in_file = set(zip(df.loc[~df.disposition.isin(IN_USE), "source"], df.loc[~df.disposition.isin(IN_USE), "run_id"]))
    assert len(lo[lo.in_features_file == "yes"]) == int((~df.disposition.isin(IN_USE)).sum()) == len(in_file)
    rows = []
    for d, lab in LEFT_OUT_LABEL.items():
        g = lo[lo.disposition == d]
        if len(g):
            rows.append({"Reason": lab, "Version 1.0 runs": int((g.source == "v1.0.0").sum()), "Replication runs": int((g.source == "replication").sum()),
                         "In the corpus file": "yes, marked" if (g.in_features_file == "yes").all() else ("no" if (g.in_features_file == "no").all() else "version 1.0 only, marked")})
    assert sum(r["Version 1.0 runs"] + r["Replication runs"] for r in rows) == len(lo)
    rows.append({"Reason": "Total", "Version 1.0 runs": int((lo.source == "v1.0.0").sum()), "Replication runs": int((lo.source == "replication").sum()), "In the corpus file": ""})
    write_table("left_out", pd.DataFrame(rows),
                "Runs left out of every table, by reason and by round of measurement. The file data/readings\\_left\\_out.csv lists each one with its reason. Last column: yes means the run is a row of the corpus file, marked with its reason, and the analysis script skips it; no means it is not, and the deposit holds its raw records where they survive (Section \\ref{sec:stats}).",
                "tab:left_out", col_fmt="lrrl")
    head["left_out"] = {"total": int(len(lo)), "in_features_file": int((lo.in_features_file == "yes").sum()),
                        "by_reason": lo.groupby(["disposition", "source"]).size().unstack(fill_value=0).to_dict("index")}


def _exp_key(c: str):
    m = re.match(r"c(\d+)(.*)", c)
    return (int(m.group(1)), m.group(2)) if m else (99, c)


def idle_table(df: pd.DataFrame, boot: Bands, head: dict, tag: str = "") -> None:
    rows = []
    for (m, a, tp), g in df[(df.cell == "c0_static") & df.usable].groupby(["model", "accelerator", "tp"]):
        b = boot.band(g.mean_power_w, g.pod_id)
        per_gpu = b["mean"] / tp
        rows.append({"Model": MODEL_LABEL[m], "Accelerator": ACC_LABEL[a], "GPUs": tp,
                     "Board W, all GPUs": band_str(b), "W per GPU": fmt(per_gpu),
                     "Share of rated": f"{per_gpu / RATED_W[a]:.0%}", "Pods": b["pods"], "Conf.": b["conf"]})
        head.setdefault(tag + "idle_w_per_gpu", {})[f"{m}/{a}"] = per_gpu
    write_table(tag + "idle_power", _order(pd.DataFrame(rows)),
                "Loaded-idle board power (experiment c0: model loaded, no requests, default power management, 60 s window), summed over the GPUs serving the model. Mean over runs, with the range over the pods in brackets (one pod: the range of the repeats). Share of rated: W per GPU over 700 W (H100, H200) or 1,000 W (B200).",
                f"tab:{tag}idle", col_fmt="llrrrrrl")
    idle = df[(df.cell == "c0_static") & df.usable].groupby("accelerator").apply(lambda g: (g.mean_power_w / g.tp).agg(["min", "max"]))
    head[tag + "idle_w_per_gpu_range"] = {a: [float(idle.loc[a, "min"]), float(idle.loc[a, "max"])] for a in idle.index}


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    """Row order of the tables: as grouped (model and accelerator in alphabetical order of their keys), the order of version 1.0."""
    return frame


def decode_batch(df: pd.DataFrame, boot: Bands, head: dict, tag: str = "") -> pd.DataFrame:
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
                         "Achieved B": f"{B:.1f}" if B < 100 else f"{B:,.0f}",
                         "Mean KV use": f"{gg.sched_kv_usage_mean.mean():.0%}" if "sched_kv_usage_mean" in gg and gg.sched_kv_usage_mean.notna().any() else "--",
                         "Board W": band_str(pw, 3), "% of rated": f"{pw['mean'] / rated:.0%}", "Tokens per second": band_str(tps, 3),
                         "J per output token": band_str(b), "Pods": b["pods"], "Conf.": b["conf"]})
            gaps.append((pw["mean"], tps["mean"], b["mean"]))
            out.append(dict(model=m, accelerator=a, tp=tp, target_batch=int(tb), B=B, **{f"e_{k}": v for k, v in b.items()},
                            tokens_per_s=tps["mean"], tps_lo=tps["lo"], tps_hi=tps["hi"], power_w=pw["mean"], power_lo=pw["lo"], power_hi=pw["hi"],
                            rated_share=pw["mean"] / rated, capped=bool(gg.client_cap.notna().all() and (gg.target_as_named > gg.client_cap).all())))
    o = pd.DataFrame(out)
    capped = o[o.capped]
    cap_note = (f" The version 1.0 runs at target 100 were requested at 256, but in version 1.0 the load generator held at most 100 connections, and the server ran {capped.B.min():.0f} to {capped.B.max():.0f} on average. They kept the 256 cached prefixes of that target." if len(capped) else "")
    write_table(tag + "decode_batch", _order(pd.DataFrame(rows)),
                f"Energy per output token of the decode workload against batch (experiment c1: 1k cached context, 200 fresh input tokens, 256 output tokens; gross board energy including the idle draw). Target B is the number of requests the client kept in flight. Achieved B is the server's mean count of running requests over the middle 80\\% of the window; at the highest targets the server ran fewer requests than were offered, and the rest waited.{cap_note} Mean KV use: the server's mean KV-cache occupancy over the same part of the window. Board W is the mean board power over the window summed over the GPUs in the TP column, and \\% of rated divides it by 700 W per H100 or H200, or 1,000 W per B200. J per output token equals Board W divided by tokens per second to within {math.ceil(100 * quotient_gap(gaps))}\\%; the small difference arises because each column is a mean over runs. Brackets: the range over the pods (one pod: the range of the repeats). TP: tensor-parallel degree (GPUs sharing the model).",
                f"tab:{tag}decode_batch", col_fmt="llrrrrlrllrl", longtable=True)
    # headline per model x accelerator: one request at a time against target 32, the highest load reached and the lowest-energy point
    hl = {}
    for (m, a), g in o.groupby(["model", "accelerator"]):
        lo_b = g[g.target_batch == 1].iloc[0]
        hi_b = g.sort_values("B").iloc[-1]
        e_min = g.sort_values("e_mean").iloc[0]
        def pt(r):
            return [float(r.e_mean), float(r.e_lo), float(r.e_hi), r.e_conf, float(r.B), int(r.target_batch)]
        hl[f"{m}/{a}"] = {"b1": [lo_b.e_mean, lo_b.e_lo, lo_b.e_hi, lo_b.e_conf], "bmax": pt(hi_b), "emin": pt(e_min),
                          "ratio": float(lo_b.e_mean / hi_b.e_mean), "ratio_emin": float(lo_b.e_mean / e_min.e_mean),
                          "turn_up_bmax_over_emin": float(hi_b.e_mean / e_min.e_mean), "tp": int(hi_b.tp),
                          "tokens_per_s_bmax": [float(hi_b.tokens_per_s), float(hi_b.tps_lo), float(hi_b.tps_hi)],
                          "power_range_w": [float(g.power_w.min()), float(g.power_w.max())],
                          "rated_share_range": [float(g.rated_share.min()), float(g.rated_share.max())]}
        for t in (8, 32, 64, 128):
            x = g[g.target_batch == t]
            if len(x):
                hl[f"{m}/{a}"][f"b{t}"] = pt(x.iloc[0])
                hl[f"{m}/{a}"][f"ratio_b1_b{t}"] = float(lo_b.e_mean / x.iloc[0].e_mean)
    head[tag + "decode_batch"] = hl
    # one row per model and accelerator for the text: one request at a time, the highest load reached, the lowest-energy point
    srows = []
    for (m, a), g in o.groupby(["model", "accelerator"]):
        lo_b, hi_b, e_min = g[g.target_batch == 1].iloc[0], g.sort_values("B").iloc[-1], g.sort_values("e_mean").iloc[0]
        def cell(r):
            return band_str({"mean": r.e_mean, "lo": r.e_lo, "hi": r.e_hi, "n": r.e_n, "pods": r.e_pods, "conf": r.e_conf, "band_kind": r.e_band_kind})
        srows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "GPUs": int(hi_b.tp), "One request at a time, J per token": cell(lo_b),
                      "Highest load: target B": int(hi_b.target_batch), "achieved B": f"{hi_b.B:.0f}", "J per token": cell(hi_b),
                      "Fold": f"{lo_b.e_mean / hi_b.e_mean:.0f}" if lo_b.e_mean / hi_b.e_mean >= 9.95 else f"{lo_b.e_mean / hi_b.e_mean:.1f}",
                      "Lowest J per token": fmt(e_min.e_mean), "at achieved B": f"{e_min.B:.0f}", "Pods at highest load": int(hi_b.e_pods)})
    kept_note = ""
    if not tag:
        allc = df[df.cell == "c1_decode_batch"]
        short, above = [], {}
        for (m, a), g in o.groupby(["model", "accelerator"]):
            hi = g.sort_values("B").iloc[-1]
            made = allc[(allc.model == m) & (allc.accelerator == a) & (allc.v_target_batch == hi.target_batch)]
            if int(made.decode_ok.sum()) < len(made):
                short.append((m, a, int(hi.tp), int(made.decode_ok.sum()), len(made)))
            n_above = int(((allc.model == m) & (allc.accelerator == a) & (allc.v_target_batch > hi.target_batch)).sum())
            if n_above:
                above.setdefault(a, {})[m] = n_above
        # the sentence below is written for this pattern; a corpus that departs from it needs the sentence rewritten
        assert len(short) == 1 and short[0][:4] == ("llama-3.3-70b", "h100", 2, 1) and list(above) == ["h100"] and "llama-3.3-70b" not in above["h100"], (short, above)
        head["batch_summary_kept"] = {"highest_load_kept_of_made": short, "runs_above_highest_load": above}
        kept_note = (f" Runs that lost part of their cache are left out (Section \\ref{{sec:stats}}), so a highest load can rest on fewer runs than were made: the {MODEL_LABEL[short[0][0]]} figure on two H100s is one run of the {num_word(short[0][4])} made at that target."
                     f" On H100, the {sum(above['h100'].values())} runs made at higher targets for the other {num_word(len(above['h100']))} models are all left out, because the server had lost cached prefixes or was not serving (Table \\ref{{tab:left_out}}).")
    write_table(tag + "batch_summary", _order(pd.DataFrame(srows)),
                "Energy per output token at one request at a time and at the highest load each server reached (experiment c1, decode workload, 1k cached context, 256 output tokens; from Table \\ref{tab:" + tag + "decode_batch}). Fold: the first over the second, as a ratio of central values. The last two columns give the lowest energy per token in the batch series and the achieved batch where it occurred; where that is not the highest load, energy per token had begun to rise again. Brackets: the range over the pods (one pod: the range of the repeats)." + kept_note,
                f"tab:{tag}batch_summary", col_fmt="llrlrrlrrrr")
    def rng(vals):
        vals = [v for v in vals if v == v]
        return [float(min(vals)), float(max(vals)), float(np.median(vals)), len(vals)]
    head[tag + "decode_batch_summary"] = {
        "ratio_range": rng([v["ratio"] for v in hl.values()]),
        "ratio_emin_range": rng([v["ratio_emin"] for v in hl.values()]),
        "ratio_b1_b32_range": rng([v["ratio_b1_b32"] for v in hl.values() if "ratio_b1_b32" in v]),
        "ratio_b1_b64_range": rng([v["ratio_b1_b64"] for v in hl.values() if "ratio_b1_b64" in v]),
        "bmax_achieved_range": rng([v["bmax"][4] for v in hl.values()]),
        "emin_achieved_range": rng([v["emin"][4] for v in hl.values()]),
        "combinations_where_highest_load_is_not_lowest_energy": sorted(k for k, v in hl.items() if v["turn_up_bmax_over_emin"] > 1.0),
        "capped_achieved_range": [float(capped.B.min()), float(capped.B.max())] if len(capped) else None,
        "rated_share_all": [float(o.rated_share.min()), float(o.rated_share.max())],
        "b1_e_range": rng([v["b1"][0] for v in hl.values()]), "emin_e_range": rng([v["emin"][0] for v in hl.values()])}
    if tag:
        return o
    # Ratio between accelerators at matched target batch (same model). Two comparisons are kept apart:
    # the same number of GPUs on each side (c1 at TP1, and c6 where both ran the same TP),
    # and Llama-3.3-70B on one B200 or one H200 against two H100s (c1), where the newer board also uses fewer GPUs.
    rows = []
    pairs = {("b200", "h100"): ({}, {}, {}), ("h200", "h100"): ({}, {}, {}), ("b200", "h200"): ({}, {}, {})}
    def add(m, tb, n, dn, exp, kind, pair):
        r = boot.ratio_band(n, dn, "e_d_j_per_tok", pair=False)
        rows.append({"Pair": f"{ACC_LABEL[pair[0]]} / {ACC_LABEL[pair[1]]}", "Comparison": kind, "Model": MODEL_LABEL[m], "Target B": int(tb),
                     "GPUs": f"{n.tp.iloc[0]} / {dn.tp.iloc[0]}", "Ratio of J per output token": band_str(r), "Pods": r["pods_str"], "Exp.": exp, "Conf.": r["conf"]})
        return [r["mean"], r["lo"], r["hi"], r["conf"]]
    c6 = df[(df.cell == "c6_tensor_parallel") & df.decode_ok & (df.weight_precision == "fp8") & (df.tp > 1)]
    for pair, (ratios, same, fewer) in pairs.items():
        for m in MODEL_ORDER:
            for tb in sorted(o.target_batch.unique()):
                n = d[(d.model == m) & (d.accelerator == pair[0]) & (d.v_target_batch == tb)]
                dn = d[(d.model == m) & (d.accelerator == pair[1]) & (d.v_target_batch == tb)]
                if len(n) and len(dn):
                    eq = n.tp.iloc[0] == dn.tp.iloc[0]
                    v = add(m, tb, n, dn, "c1", "same GPUs" if eq else "fewer GPUs", pair)
                    ratios.setdefault(m, {})[int(tb)] = v
                    (same if eq else fewer).setdefault(m, {}).setdefault(int(n.tp.iloc[0]), {})[int(tb)] = v
        for (m, tp), g in c6.groupby(["model", "tp"]):
            for tb in sorted(g.v_target_batch.unique()):
                n, dn = (g[(g.accelerator == x) & (g.v_target_batch == tb)] for x in pair)
                if len(n) and len(dn):
                    same.setdefault(m, {}).setdefault(int(tp), {})[int(tb)] = add(m, tb, n, dn, "c6", "same GPUs", pair)
    po = {"B200 / H100": 0, "H200 / H100": 1, "B200 / H200": 2}
    order = {"same GPUs": 0, "fewer GPUs": 1}
    rows = sorted(rows, key=lambda r: (po[r["Pair"]], order[r["Comparison"]], [MODEL_LABEL[x] for x in MODEL_ORDER].index(r["Model"]), r["GPUs"], r["Target B"]))
    write_table("b200_vs_h100", pd.DataFrame(rows),
                "Ratio of energy per output token (decode workload) between accelerators at matched target batch (1k cached context, 256 output tokens). The pair names the two accelerators, newer over older; below 1 means the newer one used less energy per token. `Same GPUs': both sides ran the model on the same number of GPUs, from experiment c1 (one GPU each) or c6 (two or four each, tensor parallel). `Fewer GPUs': Llama-3.3-70B on one B200 or one H200 against two H100s, the smallest H100 configuration that holds it, so the ratio includes the saving from serving on fewer GPUs. The two sides ran on different pods, so the ratio is of central values, and the band runs from the lowest to the highest combination of the two sides' bands.",
                "tab:b200_h100", col_fmt="lllrrllll", longtable=True)
    head["b200_same_gpus"], head["b200_fewer_gpus"], head["b200_over_h100"] = pairs[("b200", "h100")][1], pairs[("b200", "h100")][2], pairs[("b200", "h100")][0]
    head["h200_same_gpus"], head["h200_fewer_gpus"], head["h200_over_h100"] = pairs[("h200", "h100")][1], pairs[("h200", "h100")][2], pairs[("h200", "h100")][0]
    head["b200_over_h200_same_gpus"], head["b200_over_h200"] = pairs[("b200", "h200")][1], pairs[("b200", "h200")][0]
    return o


def decode_context(df: pd.DataFrame, boot: Bands, head: dict, tag: str = "") -> pd.DataFrame:
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
            elif dsg in (32, 64) and (m, a, int(ctx)) in base8 and b["mean"] > base8[(m, a, int(ctx))]:
                flag = "above batch-8 figure"
            cell_e = band_str(b) if not flag else (" / ".join(fmt(x) for x in sorted(e)) if len(e) <= 2 else f"{fmt(e.min())} to {fmt(e.max())}") + " (runs differ)"
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
    flagged_note = (" A configuration whose runs differ by more than 1.5 times, or whose design batch of 32 or 64 cost more per token than its batch-8 twin,"
                    " lists its runs instead of a mean, marked 'runs differ'."
                    if any(str(r["J per output token"]).endswith("(runs differ)") for r in rows) else "")
    write_table(tag + "decode_context", _order(pd.DataFrame(rows)),
                "Energy per output token of the decode workload against attended context (experiment c2: 200 fresh input tokens, 256 output tokens; gross board energy including the idle draw). "
                "The cached context is the warmed prefix; the attended context adds the fresh input and half the output. "
                "Design target B is the grid point (1, 8, 32 or 64); target as run is what the feasibility rule resized it to where the KV cache could not hold that many requests at that context; "
                "achieved B is the server's mean count of running requests. Board W and \\% of rated as in Table \\ref{tab:decode_batch}. "
                f"J per output token equals Board W divided by tokens per second to within {math.ceil(100 * quotient_gap(gaps))}\\%; "
                "the small difference arises because each column is a mean over runs. Brackets: the range over the pods (one pod: the range of the repeats). "
                "Tokens per second: mean over runs, no band." + flagged_note,
                f"tab:{tag}decode_context", col_fmt="llrrrrrrlrrlrl", longtable=True)
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
    head[tag + "decode_context"] = hl
    head[tag + "decode_context_flagged"] = o[o.flag != ""][["model", "accelerator", "target_batch", "target_as_run", "context", "flag", "e_runs"]].to_dict("records")
    return o


def prefill(df: pd.DataFrame, boot: Bands, head: dict, tag: str = "") -> pd.DataFrame:
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
    write_table(tag + "prefill", _order(pd.DataFrame(rows)),
                f"Prefill energy per prompt token with prefix caching off (experiment c3: prefix caching off, one output token; gross board energy including the idle draw). Prompt length is the fresh input; the native-maximum point is the model's context window less a margin{'' if tag else ', run at 31,200 and 32,511 tokens for the 32k models as the margin rule was tightened'}. A prompt length a model's window cannot hold has no row. Prompt tokens per second: the runs' prompt tokens over their window lengths, pooled, no band. Board W and \\% of rated as in Table \\ref{{tab:decode_batch}}. For each run, J per prompt token equals board power divided by prompt tokens per second. The columns are averages over runs, so {'' if tag else 'where the pods differ widely (short prompts on B200) '}the quotient of the printed averages differs from the printed J per prompt token by up to {round(100 * quotient_gap(gaps)):.0f}\\%. Brackets: the range over the pods (one pod: the range of the repeats).",
                f"tab:{tag}prefill", col_fmt="llrrrlrrlrl", longtable=True)
    o = pd.DataFrame(out)
    hl = {}
    for (m, a), g in o.groupby(["model", "accelerator"]):
        for tb in sorted(g.target_batch.unique()):
            gt = g[g.target_batch == tb].sort_values("fresh_in")
            hl[f"{m}/{a}/B{tb}"] = {"points": [[int(r.fresh_in), float(r.e_mean), float(r.e_lo), float(r.e_hi), r.e_conf, int(r.e_n)] for r in gt.itertuples()],
                                    "power_range": [float(gt.power_w.min()), float(gt.power_w.max())],
                                    "rated_share_range": [float(gt.rated_share.min()), float(gt.rated_share.max())]}
    head[tag + "prefill"] = hl
    per_model, cut = {}, {}
    for (m, a_), g in o.groupby(["model", "accelerator"]):
        per_model[f"{m}/{a_}"] = {"e_range": [float(g.e_mean.min()), float(g.e_mean.max())], "lengths": [int(g.fresh_in.min()), int(g.fresh_in.max())],
                                  "e_to_16k": [float(g[g.fresh_in <= 16384].e_mean.min()), float(g[g.fresh_in <= 16384].e_mean.max())],
                                  "e_longest_b1": float(g[g.target_batch == 1].sort_values("fresh_in").e_mean.iloc[-1]) if (g.target_batch == 1).any() else None,
                                  "e_1k_b1": float(g[(g.target_batch == 1) & (g.fresh_in == 1024)].e_mean.iloc[0]) if ((g.target_batch == 1) & (g.fresh_in == 1024)).any() else None}
        one, eight = g[(g.target_batch == 1) & (g.fresh_in == 1024)], g[(g.target_batch == 8) & (g.fresh_in == 1024)]
        if len(one) and len(eight):
            cut[f"{m}/{a_}"] = float(1 - eight.e_mean.iloc[0] / one.e_mean.iloc[0])
    head[tag + "prefill_summary"] = {"per_model": per_model, "concurrency_cut_at_1k": cut,
                                     "concurrency_cut_range": [min(cut.values()), max(cut.values())] if cut else None,
                                     "e_range_all": [float(o.e_mean.min()), float(o.e_mean.max())],
                                     "e_range_to_16k": [float(o[o.fresh_in <= 16384].e_mean.min()), float(o[o.fresh_in <= 16384].e_mean.max())],
                                     "power_range_all": [float(o.power_w.min()), float(o.power_w.max())],
                                     "rated_share_all": [float(o.rated_share.min()), float(o.rated_share.max())]}
    return o


def cached_prefix(df: pd.DataFrame, boot: Bands, head: dict, tag: str = "") -> pd.DataFrame:
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
                         "Cached / uncached": band_str(r), "Cache reuse": f"{c.cache_reuse_r.min():.2f}" if c.cache_reuse_r.notna().any() else "--",
                         "Pods": r["pods_str"], "Pairing": r["pairing"], "Conf.": r["conf"]})
            out.append(dict(model=m, accelerator=a, prefix_cached=int(pc), fresh_in=int(fi), target_batch=int(tb),
                            cache_reuse_min=float(c.cache_reuse_r.min()) if c.cache_reuse_r.notna().any() else math.nan,
                            cached=bc["mean"], uncached=bu["mean"], **{f"r_{k}": v for k, v in r.items()}))
    write_table(tag + "cached_prefix", _order(pd.DataFrame(rows)),
                "Cached-prefix access (experiment c4): gross board energy per completed request with a warm cached prefix and a fresh suffix, against a matched uncached control of the same total length, both with 200 output tokens. The ratio is below 1 when caching saved energy. Cache reuse: the share of cached tokens the server found in its cache during the window, over the share expected, lowest of the cached runs. Pods: pods on the cached and uncached sides. Pairing 'pod': the ratio is formed within each pod and the band is the range over pods; 'repeat': both sides on one pod, ratio formed per repeat, band the range of the repeats. Brackets on the per-request columns: the range over pods.",
                f"tab:{tag}cached", col_fmt="llrrrlllrlll", longtable=True)
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
    head[tag + "cached_prefix"] = {"per_model": hl, "pooled_by_shape": pooled.to_dict("records"), "by_prefix": by_prefix.to_dict("records"),
                             "overall_median": float(o.r_mean.median()), "overall_min": float(o.r_mean.min()), "overall_max": float(o.r_mean.max()),
                             "b1_1k": {"n": int(len(b1_1k)), "within_0.92_0.98": int(((b1_1k.r_mean >= 0.92) & (b1_1k.r_mean <= 0.985)).sum()),
                                       "bands_reach_1": int((b1_1k.r_hi >= 1.0).sum()), "min": float(b1_1k.r_mean.min()), "max": float(b1_1k.r_mean.max())},
                             "pairing_counts": o.r_pairing.value_counts().to_dict(),
                             "rows_cache_evicted": o[o.cache_reuse_min < CACHE_REUSE_MIN][["model", "accelerator", "prefix_cached", "fresh_in", "target_batch", "r_mean", "cache_reuse_min"]].to_dict("records")}
    return o


def cached_vs_fresh(o4: pd.DataFrame, o3: pd.DataFrame, head: dict) -> pd.DataFrame:
    """Energy saved per reused token in experiment c4 (uncached request less cached request, over the cached
    length), and that saving set against the measured cost of reading one prompt token fresh in experiment c3
    (same model, GPU type and concurrency, at the prompt length nearest the request's total length). Ratios of
    central values; no model is fitted."""
    rows = []
    for r in o4.itertuples():
        q = o3[(o3.model == r.model) & (o3.accelerator == r.accelerator) & (o3.target_batch == r.target_batch)]
        if not len(q):
            continue
        total = int(r.prefix_cached) + int(r.fresh_in)
        q = q.loc[[(q.fresh_in - total).abs().idxmin()]]
        saved = (float(r.uncached) - float(r.cached)) / float(r.prefix_cached)
        rows.append(dict(model=r.model, accelerator=r.accelerator, prefix_cached=int(r.prefix_cached), fresh_in=int(r.fresh_in),
                         target_batch=int(r.target_batch), saved_j_per_token=saved, fresh_j_per_token=float(q.e_mean.iloc[0]),
                         matched_prompt_tokens=int(q.fresh_in.iloc[0]), share_of_fresh=saved / float(q.e_mean.iloc[0])))
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "cached_saving_per_token.csv", index=False)
    agg = lambda g, c: {"median": float(g[c].median()), "min": float(g[c].min()), "max": float(g[c].max()), "n": int(len(g))}
    worst = d[d.prefix_cached == d.prefix_cached.max()].sort_values("share_of_fresh").iloc[0]
    head["cached_saving_per_token"] = {
        "rows": int(len(d)), "rows_without_prefill_match": int(len(o4) - len(d)),
        "saved_j_per_token_by_prefix": {int(k): agg(g, "saved_j_per_token") for k, g in d.groupby("prefix_cached")},
        "share_of_fresh_by_prefix_and_concurrency": {f"{int(k[0])}/{int(k[1])}": agg(g, "share_of_fresh") for k, g in d.groupby(["prefix_cached", "target_batch"])},
        "lowest_share_at_longest_prefix": {"model": worst.model, "accelerator": worst.accelerator, "fresh_in": int(worst.fresh_in),
                                           "target_batch": int(worst.target_batch), "share_of_fresh": float(worst.share_of_fresh)},
        # Qwen3-8B on B200 with eight concurrent requests is the configuration whose prefill cost changes with prompt
        # length (Section prefill); the paper states its two rows separately from the range at the longest prefix
        "share_at_longest_prefix_by_concurrency_excl_qwen3_8b_b200_at_8": {
            int(k): agg(g, "share_of_fresh") for k, g in d[(d.prefix_cached == d.prefix_cached.max())
                                                         & ~((d.model == "qwen3-8b") & (d.accelerator == "b200") & (d.target_batch == 8))].groupby("target_batch")},
        "qwen3_8b_b200_at_8_longest_prefix": sorted(float(v) for v in d[(d.prefix_cached == d.prefix_cached.max()) & (d.model == "qwen3-8b")
                                                                    & (d.accelerator == "b200") & (d.target_batch == 8)].share_of_fresh),
        "saved_j_per_token_longest_prefix_by_model": {k: agg(g, "saved_j_per_token") for k, g in d[d.prefix_cached == d.prefix_cached.max()].groupby("model")}}
    return d


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
                "Tensor-parallel cost: gross board energy per output token, summed over the GPUs sharing the model, at the higher tensor-parallel degree over the next lower, at matched target batch (1k cached context, 256 output tokens). Both sides are experiment c6's own runs where c6 ran that degree ('c6 / c6'); the TP1 side of Llama-3.3-70B on B200 is experiment c1's run of the same shape ('c1 / c6'). A ratio above 1 is the energy cost of splitting the model; the throughput ratio (tokens per second, higher over lower, mean over runs, no band) is what the split bought. Pairing 'pod': ratio formed within each pod, band the range over pods; 'repeat': both sides on one pod, ratio per repeat, band the range of repeats; 'none': the two sides on different pods, band from the lowest to the highest combination of the two sides' bands.",
                "tab:tp", col_fmt="llrllllrllll", longtable=True)
    o = pd.DataFrame(out)
    head["tensor_parallel"] = o.to_dict("records")
    return o


def precision(df: pd.DataFrame, boot: Bands, head: dict) -> pd.DataFrame:
    """nvfp4 over fp8 at matched shape. Two series are kept apart, because their memory offers differ: version 1.0's
    experiment c7 (with its later repeats) and the replication's wider ladder (block E: five batches, two contexts)."""
    d = df[(df.cell == "c7_precision") & df.decode_ok]
    series = {"c7_precision": "c7", "e_nvfp4_ladder": "ladder"}
    rows, out = [], []
    for (m, a), g in d.groupby(["model", "accelerator"]):
        for (cr, dsg, ctx), gg in g.groupby(["cell_run", "design_batch", "v_context_cached"]):
            f8, f4 = gg[gg.weight_precision == "fp8"], gg[gg.weight_precision == "nvfp4"]
            if not len(f8) or not len(f4):
                continue
            r = boot.ratio_band(f4, f8, "e_d_j_per_tok", low=True)
            b8, b4 = boot.band(f8.e_d_j_per_tok, f8.pod_id), boot.band(f4.e_d_j_per_tok, f4.pod_id, low=True)
            def as_run(x):
                return "/".join(str(int(v)) for v in sorted(set(x.v_target_batch)))
            def offer(x):
                return "/".join(f"{v:.1f}" for v in sorted(set(x.v_gpu_memory_utilization)))
            rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "Series": series[cr], "Design B": int(dsg), "Cached context (tokens)": f"{int(ctx):,}",
                         "Memory offer fp8 ; nvfp4": f"{offer(f8)} ; {offer(f4)}",
                         "Achieved B fp8 ; nvfp4": f"{f8.B.mean():.1f} ; {f4.B.mean():.1f}",
                         "fp8, J per output token": band_str(b8), "nvfp4, J per output token": band_str(b4), "nvfp4 / fp8": band_str(r),
                         "Throughput ratio": fmt(f4.tokens_per_s.mean() / f8.tokens_per_s.mean()), "Pods nvfp4 / fp8": r["pods_str"],
                         "Conf.": r["conf"]})
            out.append(dict(model=m, accelerator=a, series=series[cr], target_batch=int(dsg), context=int(ctx), **{f"r_{k}": v for k, v in r.items()},
                            e_fp8=b8["mean"], e_nvfp4=b4["mean"], tps_ratio=f4.tokens_per_s.mean() / f8.tokens_per_s.mean(),
                            B_fp8=f8.B.mean(), B_nvfp4=f4.B.mean()))
    write_table("precision", _order(pd.DataFrame(rows)),
                "Weight precision on B200 (experiment c7): gross board energy per output token with nvfp4 (4-bit) weights over fp8 (8-bit), at the same design batch and cached context (200 fresh input, 256 output tokens). Series `c7' is the experiment of version 1.0 with its later repeats; series `ladder' is the replication's wider grid of five batches and two contexts. Each series compares its own fp8 and nvfp4 runs. Memory offer: the fraction of the board's memory each server was allowed; where the two sides differ, or where the harness cut the batch differently for the two servers, the row compares different batches (see the achieved B). The nvfp4 runs use 4-bit versions of the models made by third parties, so every row describes these two versions, not 4-bit weights in general. Brackets: the range of the per-pod ratios where both sides ran on the same pods, otherwise the lowest to the highest combination of the two sides' bands; throughput ratio is nvfp4 over fp8 tokens per second, mean over runs, no band.",
                "tab:precision", col_fmt="lllrrlllllrll", longtable=True)
    o = pd.DataFrame(out)
    head["precision"] = o.to_dict("records")
    head["precision_summary"] = {k: {"ratio_range": [float(g.r_mean.min()), float(g.r_mean.max())], "ratio_median": float(g.r_mean.median()), "rows": int(len(g))}
                                 for k, g in o.groupby(["model", "series"])} if len(o) else {}
    head["precision_summary"] = {f"{k[0]}/{k[1]}": v for k, v in head["precision_summary"].items()}
    # the same comparison for prefill (energy per prompt token) and cached-prefix access (energy per request), replication only
    rows, po = [], []
    dp = df[(df.cell == "c7_precision_prefill") & df.usable & df.e_p_j_per_tok.notna()]
    for (m, fi, tb), gg in dp.groupby(["model", "v_fresh_in", "v_target_batch"]):
        f8, f4 = gg[gg.weight_precision == "fp8"], gg[gg.weight_precision == "nvfp4"]
        if len(f8) and len(f4):
            r = boot.ratio_band(f4, f8, "e_p_j_per_tok", low=True)
            rows.append({"Model": MODEL_LABEL[m], "Acc.": "B200", "Workload": "prefill", "Tokens": f"{int(fi):,}", "Target B": int(tb), "Unit": "J per prompt token",
                         "fp8": band_str(boot.band(f8.e_p_j_per_tok, f8.pod_id)), "nvfp4": band_str(boot.band(f4.e_p_j_per_tok, f4.pod_id, low=True)),
                         "nvfp4 / fp8": band_str(r), "Pods nvfp4 / fp8": r["pods_str"], "Conf.": r["conf"]})
            po.append(dict(model=m, workload="prefill", tokens=int(fi), target_batch=int(tb), ratio=r["mean"]))
    dc = df[(df.cell == "c7_precision_cached") & df.usable & df.e_per_request_j.notna()]
    for (m, pc, tb, ctl), gg in dc.groupby(["model", "v_prefix_cached", "v_target_batch", "v_control"]):
        f8, f4 = gg[gg.weight_precision == "fp8"], gg[gg.weight_precision == "nvfp4"]
        if len(f8) and len(f4):
            r = boot.ratio_band(f4, f8, "e_per_request_j", low=True)
            rows.append({"Model": MODEL_LABEL[m], "Acc.": "B200", "Workload": f"prefix, {ctl}", "Tokens": f"{int(pc):,}", "Target B": int(tb), "Unit": "J per request",
                         "fp8": band_str(boot.band(f8.e_per_request_j, f8.pod_id)), "nvfp4": band_str(boot.band(f4.e_per_request_j, f4.pod_id, low=True)),
                         "nvfp4 / fp8": band_str(r), "Pods nvfp4 / fp8": r["pods_str"], "Conf.": r["conf"]})
            po.append(dict(model=m, workload="prefix " + ctl, tokens=int(pc), target_batch=int(tb), ratio=r["mean"]))
    if rows:
        write_table("precision_other", _order(pd.DataFrame(rows)),
                    "Weight precision on B200 for the prefill and cached-prefix workloads (replication): gross board energy with nvfp4 weights over fp8, at matched shape. Prefill: a cache-miss prompt of the stated length, one output token. Prefix: a request with a prefix of the stated length, cached or sent uncached, 200 fresh input and 200 output tokens. The rows describe the two tested 4-bit versions, as in Table \\ref{tab:precision}.",
                    "tab:precision_other", col_fmt="lllrrllllll", longtable=True)
        t = pd.DataFrame(po)
        head["precision_other"] = {"rows": t.to_dict("records"),
                                   "ratio_range_by_model_workload": {f"{m}/{w}": [float(g.ratio.min()), float(g.ratio.max())] for (m, w), g in t.groupby(["model", "workload"])}}
    return o


def speculation(df: pd.DataFrame, boot: Bands, head: dict, data_dir: Path) -> pd.DataFrame:
    """Speculative decoding (experiment c10): the same decode workload with vLLM's n-gram proposer (three speculative
    tokens) against the plain server, and what a client could see of it in the token stream."""
    d = df[(df.cell == "c10_speculation") & df.decode_ok]
    rows, out = [], []
    for (m, a, tp), g in d.groupby(["model", "accelerator", "tp"]):
        for tb, gg in g.groupby("v_target_batch"):
            off, on = gg[gg.v_speculation == "none"], gg[gg.v_speculation != "none"]
            if not len(off) or not len(on):
                continue
            r = boot.ratio_band(on, off, "e_d_j_per_tok")
            b0, b1 = boot.band(off.e_d_j_per_tok, off.pod_id), boot.band(on.e_d_j_per_tok, on.pod_id)
            tr = on.tokens_per_s.mean() / off.tokens_per_s.mean()
            rows.append({"Model": MODEL_LABEL[m], "Acc.": ACC_LABEL[a], "TP": tp, "Target B": int(tb),
                         "Plain, J per output token": band_str(b0), "Speculative, J per output token": band_str(b1),
                         "Speculative / plain": band_str(r), "Throughput ratio": fmt(tr), "Pods": r["pods_str"], "Pairing": r["pairing"], "Conf.": r["conf"]})
            out.append(dict(model=m, accelerator=a, tp=tp, target_batch=int(tb), e_plain=b0["mean"], e_spec=b1["mean"], tps_ratio=tr,
                            **{f"r_{k}": v for k, v in r.items()}))
    write_table("speculation", _order(pd.DataFrame(rows)),
                "Speculative decoding (experiment c10): gross board energy per output token of the decode workload (1k cached context, 200 fresh input, 256 output tokens) with speculative decoding by vLLM's n-gram method, guessing up to three tokens, over the same server without it. A ratio above 1 means speculation cost energy. Throughput ratio: tokens per second, speculative over plain, mean over runs, no band. Brackets: the range of the per-pod ratios. Pairing 'pod': the ratio is formed within each pod; 'none': the two sides ran on different pods, and the band runs from the lowest to the highest combination.",
                "tab:speculation", col_fmt="llrrlllrlll", longtable=True)
    o = pd.DataFrame(out)
    head["speculation"] = o.to_dict("records")
    head["speculation_summary"] = {"ratio_range": [float(o.r_mean.min()), float(o.r_mean.max())], "ratio_median": float(o.r_mean.median()), "rows": int(len(o)),
                                   "rows_above_1": int((o.r_mean > 1).sum()), "rows_band_above_1": int((o.r_lo > 1).sum()), "rows_band_below_1": int((o.r_hi < 1).sum()),
                                   "by_model": {m: [float(g.r_mean.min()), float(g.r_mean.max())] for m, g in o.groupby("model")},
                                   "by_batch": {int(t): [float(g.r_mean.min()), float(g.r_mean.max()), float(g.r_mean.median())] for t, g in o.groupby("target_batch")},
                                   "tps_ratio_range": [float(o.tps_ratio.min()), float(o.tps_ratio.max())]}
    j = data_dir / "detection" / "speculation_detection.json"
    if j.exists():
        det = json.loads(j.read_text())
        t = pd.DataFrame(det["summary"])
        rows = []
        for (m, sp), g in t.groupby(["model", "speculation"]):
            rows.append({"Model": MODEL_LABEL[m], "Server": "plain" if sp == "none" else "speculative", "Requests": int(g.n_requests.sum()),
                         "Tokens per stream event, median over batches": fmt(float(g.tokens_per_event.median())),
                         "range": f"{fmt(float(g.tokens_per_event.min()))} to {fmt(float(g.tokens_per_event.max()))}", "_m": m, "_plain": sp == "none"})
        dd = det["detectors"]["tokens_per_event"]
        # the confusion table of each model when it was the one held out
        for r in rows:
            v = dd["leave_one_model_out"][r.pop("_m")]
            as_spec, n = (v["plain_called_speculative_n"], v["n_plain"]) if r.pop("_plain") else (v["speculative_found_n"], v["n_speculative"])
            assert n == r["Requests"]
            r["Held out: labeled speculative"], r["Held out: labeled plain"] = as_spec, n - as_spec
        held = ", ".join(f"{MODEL_LABEL[m]} {pct_half_up(v['accuracy'])}% ({pct_half_up(v['always_plain_accuracy'])}%)" for m, v in sorted(dd["leave_one_model_out"].items(), key=lambda kv: MODEL_LABEL[kv[0]]))
        write_table("speculation_detect", pd.DataFrame(rows),
                    f"What a client could see of speculative decoding on the tested server: output tokens per stream event (the output tokens the server reported for a request over the events in its stream), from the per-request records of experiment c10 ({dd['n']:,} requests). Without speculation the server sent about one token per event, and with it fewer events than tokens. A single threshold on this number ({dd['threshold_all']:.3f}) labeled {dd['accuracy_in_sample']:.0%} of requests correctly. With the threshold set on three models and tested on the fourth, it labeled {dd['accuracy_loo_mean']:.0%} on average ({min(v['accuracy'] for v in dd['leave_one_model_out'].values()):.0%} to {max(v['accuracy'] for v in dd['leave_one_model_out'].values()):.0%}). The last two columns show, for each model when it was the one held out, how many of its requests the rule labeled speculative and how many plain. Held-out accuracy, with the share of the model's requests that were plain in parentheses: {held}.",
                    "tab:speculation_detect", col_fmt="llrllrr")
        head["speculation_detect"] = {"n": dd["n"], "threshold": dd["threshold_all"], "accuracy_in_sample": dd["accuracy_in_sample"], "accuracy_loo_mean": dd["accuracy_loo_mean"],
                                      "loo": {k: v["accuracy"] for k, v in dd["leave_one_model_out"].items()},
                                      "other_features_loo": {k: v["accuracy_loo_mean"] for k, v in det["detectors"].items() if k != "tokens_per_event"}}
    return o


def num_word(n: int) -> str:
    return {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}.get(n, str(n))


def pct_half_up(x: float) -> int:
    """A share as a whole percentage, rounded half up on its decimal value (float rounding would print 0.575 as 57)."""
    from decimal import Decimal, ROUND_HALF_UP
    return int((Decimal(repr(float(x))) * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def precision_probe(df: pd.DataFrame, head: dict, data_dir: Path) -> None:
    """Precision probe (experiment c11): do a model's outputs reveal its weight format? Results of the logprob analysis
    (data/detection/precision_probe.json, from the per-request records) and the energy of the probe windows."""
    j = data_dir / "detection" / "precision_probe.json"
    if not j.exists():
        return
    det = json.loads(j.read_text())
    lab = {"fp8_vs_nvfp4_b200": "fp8 against nvfp4, both on B200", "fp8_h100_vs_fp8_b200": "fp8 on H100 against fp8 on B200 (control)"}
    rows = []
    for k, v in det["paired"].items():
        for m, x in v.items():
            rows.append({"Model": MODEL_LABEL[m], "Pair": lab[k], "Same top token, share of positions": f"{x['top1_agreement']:.0%}",
                         "Identical replies": f"{x['text_identical']:.0%}", "First differing token, mean position": f"{x['first_divergence']:.1f}",
                         "Mean absolute change in log-probability": f"{x['abs_delta_logprob_mean']:.2f}"})
    acc = {k: v["accuracy_loo_mean"] for k, v in det["detectors"].items()}
    n = next(iter(det["detectors"].values()))["n"]
    write_table("precision_probe", pd.DataFrame(rows),
                f"Precision probe (experiment c11): the same 200 prompts sent to servers holding fp8 and nvfp4 weights of the same model on B200, and fp8 weights on H100 as a control (same format, other hardware). Replies were 64 tokens, always taking the most probable token, with the 20 most probable candidates and their log-probabilities recorded at each position. A threshold on any one of five features of a single reply (the gap between the two most probable tokens, the spread of probability over the candidates, the share of distinct values, the share of exact zeros and the mean log-probability), set on one model and tested on the other, labeled {pct_half_up(min(acc.values()))}\\% to {pct_half_up(max(acc.values()))}\\% of the {n} B200 replies correctly, where guessing gives 50\\%.",
                "tab:precision_probe", col_fmt="llrrrr")
    w = df[(df.cell == "c11_precision_probe") & df.usable]
    head["precision_probe"] = {"paired": det["paired"], "detector_loo": acc, "n_replies": n, "runs": int(len(w)),
                               "energy_kj": {r.run_id.split("/")[0] + "/" + r.model + "/" + r.weight_precision: float(r.energy_j / 1e3) for r in w.itertuples()}}


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
                "Simulated agentic workload (experiment c8): simulated agents of 30 steps (Llama-3.3-70B) or 11 steps (the 32k-context models) run concurrently, each step re-reading the growing context (10,240 tokens, plus 2,048 per step, capped at the context window) and writing 300 tokens, with a cache miss at step 1 and at every fifth step (Appendix E). Energy is the gross board energy of the whole run; J per step divides it by the steps of all agents, kJ per agent by the agents. Agents is the number in flight as run (8 was resized to 6 for Llama-3.3-70B on B200 and to 4 on two H100s, so that the last step fits the KV cache). The DeepSeek-V2-Lite runs were preceded by an unmetered replay of the script; the others were not (Appendix E). Wall seconds: mean over runs, no band. Brackets: the range over the two pods (one pod: the range of the repeats).",
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
                "KV-cache pressure on Llama-3.3-70B, one B200, 16,384 cached context, 200 fresh input and 256 output tokens (experiments c9 and c9b). Each row is one run, so no band; the runs come from three pods. J per output token is gross board energy including the idle draw. GPU memory offer: the fraction of the board's memory the server was allowed, which sets the KV-cache pool. Mean KV use: the server's mean KV-cache occupancy over the middle 80\\% of the window. Mean waiting: requests admitted but not running. Preemptions: the server's preemption count over the window, recorded only in c9b ('not recorded' is missing telemetry, not zero). Collapsed: the run's energy per token exceeded 3 J, the gap between the two regimes. Cache reuse: the in-window prefix-cache hit share as a fraction of the expected one (Appendix C); near 1 the cached context was served from cache, near 0 it was recomputed on every request. Idle-drift flag: the idle power before and after the window differed by more than 5\\% (the run is kept and marked). \\% of rated: board W over 1,000 W.",
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


DECODE_EXPS = ("c1_decode_batch", "c2_decode_context", "c6_tensor_parallel", "c7_precision", "c9_kv_pressure", "c9b_kv_pressure_diag", "c10_speculation")


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
    d["batch_bin"] = pd.cut(d.v_target_batch.fillna(0), [-1, 0, 1, 8, 32, 64, 256, 5000],
                            labels=["n/a", "1", "2 to 8", "9 to 32", "33 to 64", "65 to 256", "above 256"])
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
    out.append({"Experiment": "all decode experiments", "Repeat difference median / p90": q(v[v.cell.isin(DECODE_EXPS)].cv_repeat),
                "Between-pod difference median / p90": q(v[v.cell.isin(DECODE_EXPS)].cv_pod),
                "Configs with 2 pods": int((v[v.cell.isin(DECODE_EXPS)].pods >= 2).sum()),
                "Configs on one pod": int(v[v.cell.isin(DECODE_EXPS)].same_pod.sum())})
    dd = v[v.cell.isin(DECODE_EXPS)]
    for lab, g in (("decode, both repeats on one pod", dd[dd.same_pod]), ("decode, repeats on two pods", dd[~dd.same_pod])):
        out.append({"Experiment": lab, "Repeat difference median / p90": q(g.cv_repeat), "Between-pod difference median / p90": q(g.cv_pod),
                    "Configs with 2 pods": int((g.pods >= 2).sum()), "Configs on one pod": int(g.same_pod.sum())})
    write_table("variability_cells", pd.DataFrame(out),
                "Run-to-run variability of the metric each experiment reports (energy per output token, per prompt token or per request; board power for c0). Repeat difference: the difference between the highest and lowest run of one configuration as a percentage of their mean (most configurations have two runs, placed on different pods where we had two). Between-pod difference: the same for the per-pod means, configurations run on two or more pods. Median and 90th percentile (p90) over configurations; n is the number of configurations. Configurations with fewer than two usable runs are excluded (Appendix C preamble). The last two rows split the decode configurations by whether both repeats ran on one pod; the two sets hold different configurations.",
                "tab:cv", col_fmt="lllrr")
    dec = v[v.cell.isin(["c1_decode_batch", "c2_decode_context"])]
    out = []
    bin_order = ["1", "2 to 8", "9 to 32", "33 to 64", "65 to 256", "above 256"]
    for acc in ACC_ORDER:
        for bb in bin_order:
            g = dec[(dec.accelerator == acc) & (dec.batch_bin == bb)]
            if not len(g):
                continue
            out.append({"Accelerator": ACC_LABEL[acc], "Target batch": bb, "Repeat difference median / p90": q(g.cv_repeat),
                        "Between-pod difference median / p90": q(g.cv_pod), "Configs": len(g),
                        "Repeat difference max": f"{g.cv_repeat.max():.1%}"})
    write_table("variability_batch", pd.DataFrame(out),
                f"Decode variability (energy per output token) by accelerator and target batch, experiments c1 and c2, as in Table \\ref{{tab:cv}}. The maximum is given beside the p90 because some batch levels hold as few as {num_word(min(r['Configs'] for r in out))} configurations.",
                "tab:cv_batch", col_fmt="llllrr")
    dcv = v[v.cell.isin(DECODE_EXPS)]
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
    top = dec[dec.batch_bin.isin(["65 to 256", "above 256"])]
    head["variability"]["above64"] = {a: {"repeat_cv_median": float(g.cv_repeat.median()), "repeat_cv_max": float(g.cv_repeat.max()),
                                          "pod_cv_median": float(g.cv_pod.median()) if g.cv_pod.notna().any() else None, "n": int(len(g)),
                                          "same_pod": int(g.same_pod.sum())}
                                      for a, g in top.groupby("accelerator")}


def idle_share(df: pd.DataFrame, head: dict) -> None:
    d = df[(df.cell == "c1_decode_batch") & df.decode_ok].copy()
    d["idle_share"] = d.idle_power_w / d.mean_power_w
    t = d.groupby(["accelerator", "v_target_batch"]).idle_share.agg(["median", "min", "max"]).reset_index()
    t["k"] = t.accelerator.map({a_: i for i, a_ in enumerate(ACC_ORDER)})
    t = t.sort_values(["k", "v_target_batch"])
    rows = [{"Accelerator": ACC_LABEL[r.accelerator], "Target B": int(r.v_target_batch), "Idle share of board power, median": f"{r['median']:.0%}",
             "min to max": f"{r['min']:.0%} to {r['max']:.0%}"} for _, r in t.iterrows()]
    write_table("idle_share", pd.DataFrame(rows),
                "Idle share of decode power in the batch experiment (c1: 1k cached context, 200 fresh input and 256 output tokens). For each run, the board power the pod drew while idle with the model loaded, the mean of the 60 s idle windows before and after the run, divided by the mean board power of its decode window. Gross energy per token includes this share. Median, minimum and maximum over the runs of all six models at each target batch. No band, because each value is a ratio of two means from the same run.",
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
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 8,
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
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1), sharey=True)
    raw = df[(df.cell == "c1_decode_batch") & df.decode_ok]
    for ax, acc in zip(axes, ACC_ORDER):
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
        ax.set_title(ACC_LABEL[acc])
        ax.set_xlabel("Achieved batch")
        ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1k"])
        ax.minorticks_off()
    axes[0].set_ylabel("Gross board energy, J per output token")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=6, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.09))
    fig.text(0.5, -0.15, "Achieved batch: mean running requests. Points: runs. Lines: mean. Bands: range over the pods. 1k cached context, 256 output tokens.\nLlama-3.3-70B: two H100s (TP2), one H200, one B200.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig1_decode_batch")
    plt.close(fig)


def fig_size(df: pd.DataFrame, o: pd.DataFrame):
    """The two size runs: energy per output token against achieved batch."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    raw = df[(df.cell == "c1_decode_batch") & df.decode_ok]
    for m in SIZE_MODELS:
        g = o[o.model == m].sort_values("B")
        r = raw[raw.model == m]
        ax.scatter(r.B, r.e_d_j_per_tok, s=10, color=COLOR[m], alpha=0.45, linewidths=0, zorder=2)
        ax.plot(g.B, g.e_mean, color=COLOR[m], marker=MARKER[m], ms=4.5, lw=1.4, zorder=3,
                label=f"{MODEL_LABEL[m]}, {int(g.tp.iloc[0])} x {ACC_LABEL[g.accelerator.iloc[0]]}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 8, 32, 128]); ax.set_xticklabels(["1", "8", "32", "128"]); ax.minorticks_off()
    ax.set_xlabel("Achieved batch (mean running requests)"); ax.set_ylabel("Gross board energy, J per output token")
    ax.legend(frameon=False, fontsize=7)
    fig.text(0.5, -0.08, "Points: runs. Lines: mean. 1k cached context, 256 output tokens, fp8 weights.", ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_size")
    plt.close(fig)


def fig_decode_context(df: pd.DataFrame, o: pd.DataFrame):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.9), sharey=True)
    raw = df[(df.cell == "c2_decode_context") & df.decode_ok]
    for ax, acc in zip(axes, ACC_ORDER):
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
        ax.set_xticks([1024, 4096, 16384, 65536]); ax.set_xticklabels(["1k", "4k", "16k", "64k"])
        ax.minorticks_off()
    axes[0].set_ylabel("J per output token")
    fig.legend(frameon=False, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.19))
    fig.text(0.5, -0.25, "Dashed: design target batch 8. Solid: design target 64 (resized to the cache maximum at long context). Bands: range over pods. Collapsed configurations omitted.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig2_decode_context")
    plt.close(fig)


def fig_prefill(df: pd.DataFrame, o: pd.DataFrame):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1), sharey=True)
    raw = df[(df.cell == "c3_prefill_miss") & df.usable & df.e_p_j_per_tok.notna()]
    for ax, acc in zip(axes, ACC_ORDER):
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
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_xticks([1024, 4096, 16384, 65536]); ax.set_xticklabels(["1k", "4k", "16k", "64k"])
        ax.minorticks_off()
    axes[0].set_ylabel("J per prompt token")
    fig.legend(frameon=False, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    fig.text(0.5, -0.30, "Dashed: one request at a time. Solid: eight concurrent. Prefix caching off; one output token. Bands: range over the pods.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig3_prefill")
    plt.close(fig)


def fig_cached(o: pd.DataFrame):
    """Cached over uncached energy per request against the cached prefix length (log scale): columns are the
    GPUs, rows the fresh suffix; one colour per model, dashed for one request at a time, solid for eight."""
    plt = _style()
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0), sharex=True, sharey=True)
    prefixes = [1024, 4096, 16384]
    for r_i, fresh in enumerate([200, 2048]):
        for c_i, acc in enumerate(ACC_ORDER):
            ax = axes[r_i][c_i]
            for i, m in enumerate(MODEL_ORDER):
                for tb, ls in ((1, "--"), (8, "-")):
                    g = o[(o.model == m) & (o.accelerator == acc) & (o.target_batch == tb) & (o.fresh_in == fresh)].sort_values("prefix_cached")
                    if not len(g):
                        continue
                    xs = g.prefix_cached * (1 + (i - 2.5) * 0.02)   # small offsets so the bars do not overlap
                    ax.errorbar(xs, g.r_mean, yerr=[g.r_mean - g.r_lo, g.r_hi - g.r_mean], ls=ls, lw=1.3, marker=MARKER[m], ms=3.5,
                                color=COLOR[m], ecolor=COLOR[m], elinewidth=0.8, capsize=1.5,
                                label=f"{MODEL_LABEL[m]}, {'one at a time' if tb == 1 else 'eight concurrent'}" if (r_i, c_i) == (0, 0) else None)
            if not len(o[(o.accelerator == acc) & (o.fresh_in == fresh)]):
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#8a8a86")
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
                     "Reference rate (tokens/s)": f"{bound:,.0f}", "Measured (tokens/s)": f"{tps:,.0f}", "Share of reference": f"{tps / bound:.0%}",
                     "Idle share of board power": f"{idle:.0%}"})
        hl[f"{m}/{a}"] = {"weight_bytes": w, "bound_tps": bound, "tps": tps, "share": tps / bound, "idle_share": idle}
    write_table("roofline", _order(pd.DataFrame(rows)),
                "Measured token rate against a weight-only reference rate at one request at a time (experiment c1, target batch 1). The reference rate assumes that writing each output token takes one read from memory of the weights the step uses. Weight bytes per step: all the weights for the dense models, and for the MoE models the shared weights plus the experts one token uses. These are estimates from each model's structure, not measured counts, and the Gemma-3-27B figure includes its vision encoder, which a text request does not read. Reference rate: the GPUs' combined memory bandwidth (NVIDIA's figures: 3.35 TB/s per H100, 4.8 TB/s per H200, 8 TB/s per B200) divided by those bytes. Measured: tokens per second, mean over the runs. Share of reference: measured over the reference rate. The reference leaves out reading the KV cache, processing the fresh input and the other work of a step, so the share does not measure how fully the memory bandwidth was used, and a smaller share on one GPU than on another does not show that it was used less. Idle share: the board power drawn while idle with the model loaded, from the idle windows bracketing each run, over the mean board power of the decode window.",
                "tab:roofline", col_fmt="llrrrrrr")
    head["roofline"] = hl
    head["roofline_summary"] = {"share_range": [min(v["share"] for v in hl.values()), max(v["share"] for v in hl.values())],
                                "b200_below_h100_every_model": all(hl[f"{m}/b200"]["share"] < hl[f"{m}/h100"]["share"] for m in MODEL_ORDER),
                                "share_by_accelerator": {a_: [min(v["share"] for k, v in hl.items() if k.endswith("/" + a_)), max(v["share"] for k, v in hl.items() if k.endswith("/" + a_))] for a_ in ACC_ORDER}}


def factors(o1, o2, o4, o6, o7, o10, df, head) -> pd.DataFrame:
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
        lo, hi = g[g.target_batch == 1].iloc[0], g.sort_values("B").iloc[-1]
        b.append(lo.e_mean / hi.e_mean)
        bd.append(math.log2(hi.B / lo.B))
    out.append(agg("Batch", "one request at a time over the highest load reached", b, "J per output token",
                   "the dominant lever on every model and all three GPUs", "sec:batch", doublings=bd))
    # cached prefix: uncached over cached at 16k cached tokens (both fresh lengths, both concurrencies)
    c = (1 / o4[o4.prefix_cached == 16384].r_mean).tolist()
    at_1k = float(1 - o4[o4.prefix_cached == 1024].r_mean.median())
    head["cached_saving_at_1k_median"] = at_1k
    out.append(agg("Cached prefix", "fresh request over one with 16k tokens already cached", c, "J per request",
                   f"large for long reused prompts, small (median {at_1k:.0%}) at 1k", "sec:cached"))
    # cache boundary: collapsed over the last clear run at the same memory offer (Llama-3.3-70B, B200)
    kv = df[df.cell.isin(["c9_kv_pressure", "c9b_kv_pressure_diag"]) & df.decode_ok]
    k, kv_tps, kv_pw = [], [], []
    for _, g in kv.groupby(["v_gpu_memory_utilization", "pod_id"]):    # within a pod: a pod's own power level affects both sides alike
        clear, coll = g[g.e_d_j_per_tok < 3], g[g.e_d_j_per_tok >= 3]
        if len(clear) and len(coll):
            ref = clear.sort_values("B").e_d_j_per_tok.iloc[-1]
            k += (coll.e_d_j_per_tok / ref).tolist()
            top = clear.sort_values("B").iloc[-1]
            kv_tps += (top.tokens_per_s / coll.tokens_per_s).tolist()
            kv_pw += (coll.mean_power_w / top.mean_power_w).tolist()
    head["kv_collapse_within_pod"] = {"energy_fold": [float(min(k)), float(max(k))], "throughput_fold": [float(min(kv_tps)), float(max(kv_tps))],
                                      "power_ratio": [float(min(kv_pw)), float(max(kv_pw))], "n": len(k)}
    out.append(agg("Memory limit for stored context", "past the limit over just below it on the same pod (Llama-3.3-70B, one B200)", k, "J per output token",
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
    out.append(agg("Context length, batch 64", "longest context up to 32k whose runs kept their cache over 1k, design batch 64; long context forced a smaller batch on some servers; configurations marked ‘runs differ’ in Table 14 left out", r64, "J per output token",
                   "rises with context and more steeply at deep batch, where the stored context fills memory", "sec:context", doublings=d64))
    r8, d8 = ctx(8)
    out.append(agg("Context length, batch 8", "longest context up to 32k over 1k, design batch 8; configurations marked ‘runs differ’ in Table 14 left out", r8, "J per output token",
                   "a moderate, steady rise", "sec:context", doublings=d8))
    # GPU generation: B200 over H100 on the same number of GPUs (two-sided); and, separately,
    # Llama-3.3-70B on one B200 against the two H100s it needs
    g = [v[0] for m in head["b200_same_gpus"].values() for t in m.values() for v in t.values()]
    out.append(agg("GPU generation, same GPU count", "B200 over H100 on the same number of GPUs, at the same batch", g, "J per output token",
                   "no consistent saving: lower for some models, higher for others", "sec:b200"))
    g = [v[0] for m in head["h200_same_gpus"].values() for t in m.values() for v in t.values()]
    out.append(agg("H200 against H100, same GPU count", "H200 over H100 on the same number of GPUs, at the same batch", g, "J per output token",
                   "close to 1: the two share a design, and the H200 adds memory", "sec:b200"))
    g = [v[0] for m in head["b200_fewer_gpus"].values() for t in m.values() for v in t.values()]
    g += [v[0] for m in head["h200_fewer_gpus"].values() for t in m.values() for v in t.values()]
    out.append(agg("One larger GPU against two H100s", "one B200 or one H200 over the two H100s Llama-3.3-70B needs, at the same batch; changes GPU type and GPU count together", g, "J per output token",
                   "a saving, part of it from holding the model on fewer GPUs", "sec:b200"))
    # precision: fp8 over nvfp4, excluding the row where both servers were at the cache boundary
    p = (1 / o7[~((o7.model == "llama-3.3-70b") & (o7.context == 16384) & (o7.target_batch == 64))].r_mean).tolist()
    out.append(agg("Weight precision", "8-bit over 4-bit weights, B200", p, "J per output token",
                   "a saving in the tested 4-bit deployments; third-party 4-bit models, so indicative", "sec:precision"))
    one, deep = o10[o10.target_batch == 1].r_mean, o10[o10.target_batch == 32].r_mean
    out.append(agg("Speculative decoding", "n-gram speculation of three tokens over the plain server", o10.r_mean.tolist(), "J per output token",
                   f"at one request at a time {fmt(float(one.min()))} to {fmt(float(one.max()))}; at batch 32 {fmt(float(deep.min()))} to {fmt(float(deep.max()))}: it helped a lone request on most models and cost energy under load", "sec:speculation"))
    # tensor parallelism: higher TP over lower
    out.append(agg("Tensor parallelism", "a model split over more GPUs over fewer", o6.r_mean.tolist(), "J per output token",
                   "costs energy; buys throughput", "sec:tp"))
    # run-to-run noise: larger over smaller of the two runs, decode configurations
    v = pd.read_csv(TABLES / "variability_per_config.csv")
    rr = v[v.cell.isin(DECODE_EXPS)].ratio_repeat.dropna()
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
                "The measured factors compared on one scale. Effect is the multiplier on energy per output token (per request for the cached prefix) between the two settings named in the Comparison column, restating the ratios of the section that measures the factor (paired by pod or by repeat where both sides ran on the same pods, otherwise ratios of central values): the median over the comparisons, with the smallest and largest in parentheses. Comparisons counts them: one for each model, GPU type and tested batch or prompt shape. Each row restates figures from the section that measures the factor; the GPU-generation row is two-sided because the direction depends on the model, and the GPU-memory row is one model (Llama-3.3-70B). Repeat-to-repeat variation is shown for scale, with its 90th percentile in place of a range.",
                "tab:factors", col_fmt="llllrl")
    # the generic writer sizes columns from the longest cell; this table needs wrapped text columns
    body = "\n".join(" & ".join(_tex_escape(str(x)) for x in r.values()) + r" \\" for r in rows)
    factors_tex = (
        "\\begin{landscape}\n\\begin{table}[p]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\caption{" + (TABLES / "factors.md").read_text().split("**")[1] + "}\\label{tab:factors}\n"
        "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.5cm}>{\\raggedright\\arraybackslash}p{6.0cm}lr>{\\raggedright\\arraybackslash}p{2.7cm}>{\\raggedright\\arraybackslash}p{4.9cm}}\n"
        "\\toprule\nFactor & Comparison & \\makecell[l]{Effect, median\\\\(range)} & \\makecell[r]{Compar-\\\\isons} & \\makecell[l]{Per\\\\doubling} & What the data tell us \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\\end{landscape}\n")
    (TABLES / "factors.tex").write_text(factors_tex.replace("\u2018", "`").replace("\u2019", "'"))
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
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
    tbs = [1, 4, 8, 16, 32, 64, 100, 128, 256]
    LS = {1: "-", 2: "--", 4: ":"}
    def line(ax, m, r, ls, lab=None, color=None):
        xs = [tb for tb in tbs if tb in r]
        y = [r[tb][0] for tb in xs]; lo = [r[tb][1] for tb in xs]; hi = [r[tb][2] for tb in xs]
        ax.fill_between(xs, lo, hi, color=color or COLOR[m], alpha=0.13, linewidth=0)
        ax.plot(xs, y, color=color or COLOR[m], marker=MARKER[m], ms=3.5, lw=1.3, ls=ls, label=lab)
    for ax, key, title in ((axes[0], "b200_same_gpus", "B200 / H100, same GPUs"), (axes[1], "h200_same_gpus", "H200 / H100, same GPUs")):
        for m in MODEL_ORDER:
            for tp, r in sorted(head[key].get(m, {}).items()):
                line(ax, m, r, LS[tp])
        ax.set_title(title)
    ax = axes[2]
    m = "llama-3.3-70b"
    line(ax, m, head["b200_fewer_gpus"][m][1], "-")
    line(ax, m, head["h200_fewer_gpus"][m][1], "-", color="#4a4a47")
    ax.set_title("Llama-3.3-70B, 1 GPU / 2 H100s")
    b_last = max(head["b200_fewer_gpus"][m][1]); h_last = max(head["h200_fewer_gpus"][m][1])
    ax.annotate("one B200", xy=(b_last, head["b200_fewer_gpus"][m][1][b_last][0]), xytext=(6, 0.55), fontsize=7, color=COLOR[m],
                arrowprops=dict(arrowstyle="-", color="#8a8a86", lw=0.6))
    ax.annotate("one H200", xy=(h_last, head["h200_fewer_gpus"][m][1][h_last][0]), xytext=(6, 1.25), fontsize=7, color="#4a4a47",
                arrowprops=dict(arrowstyle="-", color="#8a8a86", lw=0.6))
    for ax in axes:
        ax.axhline(1.0, color="#8a8a86", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
        ax.minorticks_off()
        ax.set_yticks([0.5, 0.7, 1, 1.5, 2, 3]); ax.set_yticklabels(["0.5", "0.7", "1", "1.5", "2", "3"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Target batch")
    axes[0].set_ylabel("J per output token, ratio")
    handles = [Line2D([], [], color=COLOR[m], marker=MARKER[m], ms=4, lw=1.4, label=MODEL_LABEL[m]) for m in MODEL_ORDER]
    handles += [Line2D([], [], color="#4a4a47", lw=1.2, ls=LS[t], label=f"{t} GPU{'s' if t > 1 else ''} each") for t in (1, 2, 4)]
    fig.legend(handles=handles, frameon=False, ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.text(0.5, -0.19, "Below 1: the newer GPU used less energy per token. Bands: lowest to highest combination of the two sides' bands. 1k cached context, 256 output tokens.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_b200_h100")
    plt.close(fig)


def fig_context_multiplier(o2: pd.DataFrame):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1), sharey=True)
    for ax, acc in zip(axes, ACC_ORDER):
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
    fig.legend(frameon=False, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.24))
    fig.text(0.5, -0.31, "Dashed: design batch 8. Solid: design batch 64. Circled: the server could not hold the design batch at that context and ran a smaller one.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_context_multiplier")
    plt.close(fig)


def fig_tp(tp: pd.DataFrame):
    plt = _style()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    size = {1: 3.5, 16: 5.5, 64: 7.5}
    # label positions in data coordinates, chosen to clear the markers and bars
    label_at = {("qwen3-8b", "h100", 1): (1.325, 1.33, "left"), ("qwen3-8b", "b200", 1): (1.33, 1.045, "left"),
                ("llama-3.3-70b", "b200", 1): (1.60, 1.29, "center"), ("llama-3.3-70b", "b200", 2): (1.30, 1.25, "left"),
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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, ctx in zip(axes, [1024, 16384]):
        for (m, ser), g in pr[pr.context == ctx].groupby(["model", "series"], sort=False):
            g = g.sort_values("target_batch")
            x = g.target_batch * (1.0 if ser == "ladder" else 1.12)
            ax.errorbar(x, g.r_mean, yerr=[g.r_mean - g.r_lo, g.r_hi - g.r_mean], fmt=MARKER[m], ms=4.5, color=COLOR[m],
                        mfc=COLOR[m] if ser == "ladder" else "white", ecolor=COLOR[m], elinewidth=0.9, capsize=1.5,
                        ls="-" if ser == "ladder" else "", lw=1.0)
        ax.axhline(1.0, color="#8a8a86", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xticks([1, 8, 32, 64, 128]); ax.set_xticklabels(["1", "8", "32", "64", "128"]); ax.minorticks_off()
        ax.set_title(f"{ctx // 1024}k cached context"); ax.set_xlabel("Design target batch")
    axes[0].set_ylabel("J per token, 4-bit (nvfp4) / 8-bit (fp8)")
    handles = [plt.Line2D([], [], color=COLOR[m], marker=MARKER[m], ls="", ms=4.5, label=MODEL_LABEL[m]) for m in pr.model.unique()]
    axes[0].legend(handles=handles, frameon=False, fontsize=6.8, loc="lower left")
    fig.text(0.5, -0.08, "Weight precision on B200. Filled, joined: the wider grid of the replication. Open: experiment c7 of version 1.0. Bars: range over pods, or lowest to highest combination.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_precision")
    plt.close(fig)


def fig_speculation(o: pd.DataFrame):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, acc in zip(axes, ["h100", "b200"]):
        for m in MODEL_ORDER:
            g = o[(o.model == m) & (o.accelerator == acc)].sort_values("target_batch")
            if not len(g):
                continue
            ax.errorbar(g.target_batch, g.r_mean, yerr=[g.r_mean - g.r_lo, g.r_hi - g.r_mean], marker=MARKER[m], ms=4, lw=1.3, color=COLOR[m],
                        ecolor=COLOR[m], elinewidth=0.8, capsize=1.5, label=MODEL_LABEL[m] if acc == "h100" else None)
        ax.axhline(1.0, color="#8a8a86", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([1, 8, 32]); ax.set_xticklabels(["1", "8", "32"]); ax.minorticks_off()
        ax.set_yticks([0.7, 1, 2, 4, 8]); ax.set_yticklabels(["0.7", "1", "2", "4", "8"])
        ax.set_title(ACC_LABEL[acc]); ax.set_xlabel("Target batch (requests in flight)")
    axes[0].set_ylabel("J per output token,\nspeculative / plain")
    fig.legend(frameon=False, ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.10))
    fig.text(0.5, -0.16, "Above 1: speculative decoding cost energy. Bars: range over the two pods. 1k cached context, 256 output tokens.",
             ha="center", fontsize=7.2, color="#4a4a47")
    _save(fig, "fig_speculation")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--only", default="", help="development: stop after this step")
    args = ap.parse_args()
    df = load(args.features)
    boot = Bands()
    feat = args.features.resolve()
    head: dict = {"features": str(feat.relative_to(ROOT)) if feat.is_relative_to(ROOT) else feat.name,
                  "store": store_facts(args.features)}
    corpus_tables(df, head, feat.parent)
    dm = df[df.series == "main"]
    def done(step):
        if args.only == step:
            (TABLES / "headline.json").write_text(json.dumps(head, indent=1, default=str))
            raise SystemExit(0)
    done("corpus")
    idle_table(dm, boot, head)
    done("idle")
    o1 = decode_batch(dm, boot, head)
    done("batch")
    o2 = decode_context(dm, boot, head)
    done("context")
    o3 = prefill(dm, boot, head)
    done("prefill")
    o4 = cached_prefix(dm, boot, head)
    cached_vs_fresh(o4, o3, head)
    done("cached")
    o6 = tensor_parallel(dm, boot, head)
    o7 = precision(dm, boot, head)
    o10 = speculation(dm, boot, head, feat.parent)
    precision_probe(dm, head, feat.parent)
    done("precision")
    o8 = agent_replay(dm, boot, head)
    kv_pressure(dm, head)
    done("kv")
    variability(dm, head)
    idle_share(dm, head)
    agent_vs_chat(o8, o2, head)
    roofline(dm, o1, head)
    done("roofline")
    fac = factors(o1, o2, o4, o6, o7, o10, dm, head)
    done("factors")
    # the size runs: the same five experiments on two larger models, in their own tables
    ds = df[df.series == "size"]
    idle_table(ds, boot, head, tag="size_")
    s1 = decode_batch(ds, boot, head, tag="size_")
    decode_context(ds, boot, head, tag="size_")
    prefill(ds, boot, head, tag="size_")
    cached_prefix(ds, boot, head, tag="size_")
    done("size")
    fig_decode_batch(dm, o1)
    fig_size(ds, s1)
    fig_decode_context(dm, o2)
    fig_prefill(dm, o3)
    fig_cached(o4)
    fig_tp(o6)
    fig_precision(o7)
    fig_speculation(o10)
    fig_b200(head)
    fig_context_multiplier(o2)
    fig_factors(fac)
    (TABLES / "headline.json").write_text(json.dumps(head, indent=1, default=str))
    print(json.dumps({k: head[k] for k in ("runs", "usable_runs", "configs", "configs_two_repeats", "configs_two_pods", "pods", "data_centers", "metered_hours")}, indent=1))


if __name__ == "__main__":
    main()
