"""Independent recompute of the summary tables of Paper 1 (corpus_cells, pods, idle_share,
roofline, variability_cells, variability_batch, latency, factors) from features.parquet and
the raw per-request records. Shares no code with analysis/build.py. Rules taken from the
table captions, build.py lines 1-25 and paper.tex (see report_summary.md).

    python analysis/recompute/check_summary.py [REPO] [--store STORE]

Match rule: my value rounds to the printed value at the printed precision, or differs by
one unit in the last printed digit."""
import sys, re, json, glob, math
import numpy as np, pandas as pd

import os
REPO = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = sys.argv[sys.argv.index('--store') + 1] if '--store' in sys.argv else None   # raw per-request records (Zenodo deposit, raw/); latency is skipped without it
T = REPO + '/paper/tables/'
NAME = {'qwen3-8b': 'Qwen3-8B', 'qwen3-30b-a3b': 'Qwen3-30B-A3B', 'deepseek-v2-lite': 'DeepSeek-V2-Lite', 'llama-3.3-70b': 'Llama-3.3-70B'}
BW = {'h100': 3.35e12, 'b200': 8e12}   # HBM bandwidth per GPU, B/s (roofline caption)

# ---------------------------------------------------------------- helpers
def num(s):
    """First number in a printed string (commas removed)."""
    m = re.search(r'-?[\d,]*\.?\d+', str(s)); return float(m.group().replace(',', ''))

def nums(s):
    return [float(x.replace(',', '')) for x in re.findall(r'-?[\d,]*\.?\d+', str(s))]

def band(s):
    x = nums(s); return x[1:3] if len(x) >= 3 else [None, None]

def decimals(tok):
    tok = tok.replace(',', '').replace('%', '')
    return len(tok.split('.')[1]) if '.' in tok else 0

def match(printed_tok, value):
    """printed_tok: the printed number as a string. True if value rounds to it or is one unit off."""
    if value is None or (isinstance(value, float) and math.isnan(value)): return False
    d = decimals(printed_tok); p = num(printed_tok)
    return abs(round(value, d) - p) <= 10 ** (-d) + 1e-9

class Tally:
    def __init__(self, name): self.name, self.rows, self.fields, self.bad = name, 0, 0, 0
    def check(self, label, printed_tok, value, note=''):
        self.fields += 1
        if not match(printed_tok, value):
            self.bad += 1; print(f'MISMATCH {self.name} | {label} | published {printed_tok} | mine {value:.6g} {note}')
    def row(self): self.rows += 1
    def done(self): print(f'{self.name}: {self.rows} rows checked, {self.fields} fields, {self.bad} mismatches')

def q90(s): return float(np.percentile(np.asarray(s, float), 90))   # linear interpolation
def pct_diff(a, b): return abs(a - b) / ((a + b) / 2) * 100

# ---------------------------------------------------------------- data
F = pd.read_parquet(REPO + '/data/features.parquet')
F['tb'] = F.run_id.str.extract(r'target_batch-(\d+)')[0].astype(float)
F['config'] = F.run_id.str.replace(r'/r\d+$', '', regex=True)
f = F[F.dropped.isna()].copy()
print(f'runs: {len(F)} total, {len(f)} in corpus, {len(F) - len(f)} dropped')
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


# ================================================================ corpus_cells
t = Tally('corpus_cells')
pub = pd.read_csv(T + 'corpus_cells.csv')
cnt = F.groupby(['cell', 'accelerator', 'model']).size()
for _, r in pub.iterrows():
    t.row(); cell = r.Experiment
    for col in pub.columns[1:-1]:
        acc, model = col.split(' ', 1); acc = acc.lower(); mdl = {v: k for k, v in NAME.items()}[model]
        if cell == 'Total': v = int(((F.accelerator == acc) & (F.model == mdl)).sum())
        else: v = int(cnt.get((cell, acc, mdl), 0))
        t.check(f'{cell} / {col}', str(r[col]), v)
    v = len(F) if cell == 'Total' else int((F.cell == cell).sum())
    t.check(f'{cell} / Total', str(r.Total), v)
t.done()

# ================================================================ pods
# Metered window = the active window (duration_s). Idle windows are separate 60 s windows not in
# duration_s; warm-up and model loads precede the meter start (paper Sec. experiments). So metered
# hours = sum of duration_s over the pod's runs. Pod = pod_id; GPUs = the most GPUs any run used.
t = Tally('pods')
pub = pd.read_csv(T + 'pods.csv')
pods = F.groupby('pod_id').agg(dc=('data_center', 'first'), acc=('accelerator', 'first'), gpus=('n_gpus', 'max'),
                               models=('model', 'nunique'), runs=('run_id', 'count'), hours=('duration_s', lambda s: s.sum() / 3600)).reset_index()
hours_nodrop = f.groupby('pod_id').duration_s.sum() / 3600
for _, r in pub.iterrows():
    t.row()
    m = pods[(pods.dc == r['Data center']) & (pods.acc == r.Accelerator.lower()) & (pods.gpus == r.GPUs) & (pods.runs == r.Runs)]
    if len(m) == 0: print(f'MISMATCH pods | {r.Pod}: no pod matches (dc, acc, GPUs, runs)'); t.bad += 1; continue
    if len(m) > 1: print(f'   pods: {r.Pod}: {len(m)} pods share (dc, acc, GPUs, runs); matched by nearest hours'); m = m.iloc[[(m.hours - num(r['Metered hours'])).abs().argmin()]]
    m = m.iloc[0]
    t.check(f'{r.Pod} models', str(r['Models (count)']), m.models)
    t.check(f'{r.Pod} hours (all runs)', str(r['Metered hours']), m.hours, f'(excluding dropped runs: {hours_nodrop[m.pod_id]:.3f})')
print('   pods: data centers', pods.dc.nunique(), '| pods', len(pods), '| total hours', round(pods.hours.sum(), 2))
t.done()

# ================================================================ idle_share
t = Tally('idle_share')
pub = pd.read_csv(T + 'idle_share.csv')
c1 = f[f.cell == 'c1_decode_batch'].copy(); c1['share'] = c1.idle_power_w / c1.mean_power_w
for _, r in pub.iterrows():
    t.row(); g = c1[(c1.accelerator == r.Accelerator.lower()) & (c1.tb == r['Target B'])].share * 100
    t.check(f'{r.Accelerator} B{r["Target B"]} median', str(r.iloc[2]), g.median(), f'(n={len(g)})')
    lo, hi = nums(r.iloc[3])
    t.check(f'{r.Accelerator} B{r["Target B"]} min', str(int(lo)), g.min()); t.check(f'{r.Accelerator} B{r["Target B"]} max', str(int(hi)), g.max())
t.done()

# ================================================================ roofline
t = Tally('roofline')
pub = pd.read_csv(T + 'roofline.csv')
c0 = f[f.cell == 'c0_static']
for _, r in pub.iterrows():
    t.row(); mdl = {v: k for k, v in NAME.items()}[r.Model]; acc = r['Acc.'].lower()
    g = c1[(c1.model == mdl) & (c1.accelerator == acc) & (c1.tb == 1)]
    assert len(g) == 2 and g.W_batch_bytes.nunique() == 1 and g.n_gpus.nunique() == 1
    wb = g.W_batch_bytes.iloc[0]; ng = int(g.n_gpus.iloc[0])
    bound = BW[acc] * ng / wb; meas = g.tokens_per_s.mean()
    t.check(f'{r.Model} {acc} GPUs', str(r.GPUs), ng)
    t.check(f'{r.Model} {acc} GB', str(r.iloc[3]), wb / 1e9)
    t.check(f'{r.Model} {acc} bound', str(r.iloc[4]), bound)
    t.check(f'{r.Model} {acc} measured', str(r.iloc[5]), meas)
    t.check(f'{r.Model} {acc} share', str(r.iloc[6]), meas / bound * 100)
    idle_a = g.share.mean() * 100                                      # mean over 2 runs of bracketing idle / window mean power
    idle_c = c0[(c0.model == mdl) & (c0.accelerator == acc)].loaded_idle_power_w.mean() / g.mean_power_w.mean() * 100  # c0 loaded idle / mean board W
    t.check(f'{r.Model} {acc} idle share', str(r.iloc[7]), idle_a, f'(alt: c0 loaded idle over mean board W = {idle_c:.1f})')
t.done()

# ================================================================ variability (shared prep)
def metric(df):
    m = pd.Series(np.nan, index=df.index)
    m[df.cell == 'c0_static'] = df.loaded_idle_power_w
    m[df.cell == 'c3_prefill_miss'] = df.energy_j / df.prompt_tokens
    m[df.cell == 'c4_cached_prefix'] = df.energy_j / df.n_requests
    oth = ~df.cell.isin(['c0_static', 'c3_prefill_miss', 'c4_cached_prefix'])
    m[oth] = df.energy_j / df.completion_tokens
    return m
f['metric'] = metric(f)
u = f[np.isfinite(f.metric)]                                        # usable runs
print('   variability: runs without a finite metric (excluded):', int((~np.isfinite(f.metric)).sum()),
      '| configs total', F.config.nunique(), '| with >=2 usable runs', int((u.groupby("config").size() >= 2).sum()))
rows = []
for cfg, g in u.groupby('config'):
    if len(g) < 2: continue
    assert len(g) == 2, cfg
    pods_ = g.pod_id.nunique(); rep = pct_diff(*g.metric.values)
    pm = g.groupby('pod_id').metric.mean().values
    between = pct_diff(*pm) if pods_ == 2 else np.nan
    rows.append(dict(config=cfg, cell=g.cell.iloc[0], acc=g.accelerator.iloc[0], phase=g.phase.iloc[0], pods=pods_, rep=rep, between=between,
                     ratio=g.metric.max() / g.metric.min(), tb=g.tb.iloc[0]))
V = pd.DataFrame(rows)
DECODE = V.phase == 'decode'

def check_var_row(t, label, printed_rep, printed_pod, g, cfg2_tok=None, cfg1_tok=None, max_tok=None):
    t.row()
    toks = re.findall(r'[\d.]+', printed_rep)            # "a% / b% (n=N)"
    t.check(f'{label} rep median', toks[0], g.rep.median()); t.check(f'{label} rep p90', toks[1], q90(g.rep)); t.check(f'{label} rep n', toks[2], len(g))
    g2 = g[g.pods == 2]
    if printed_pod.strip() == '--': t.fields += 1; (None if len(g2) == 0 else (print(f'MISMATCH {t.name} | {label} pod: published -- but {len(g2)} two-pod configs'), setattr(t, 'bad', t.bad + 1)))
    else:
        toks = re.findall(r'[\d.]+', printed_pod)
        t.check(f'{label} pod median', toks[0], g2.between.median()); t.check(f'{label} pod p90', toks[1], q90(g2.between)); t.check(f'{label} pod n', toks[2], len(g2))
    if cfg2_tok is not None: t.check(f'{label} configs 2 pods', cfg2_tok, len(g2))
    if cfg1_tok is not None: t.check(f'{label} configs 1 pod', cfg1_tok, int((g.pods == 1).sum()))
    if max_tok is not None: t.check(f'{label} rep max', max_tok, g.rep.max())

# ================================================================ variability_cells
t = Tally('variability_cells')
pub = pd.read_csv(T + 'variability_cells.csv')
for _, r in pub.iterrows():
    e = r.Experiment
    if e == 'all decode experiments': g = V[DECODE]
    elif e == 'decode, both repeats on one pod': g = V[DECODE & (V.pods == 1)]
    elif e == 'decode, repeats on two pods': g = V[DECODE & (V.pods == 2)]
    else: g = V[V.cell == e]
    check_var_row(t, e, r.iloc[1], r.iloc[2], g, str(r.iloc[3]), str(r.iloc[4]))
cells_pub = set(pub.Experiment); cells_mine = set(V.cell)
if cells_mine - cells_pub: print('MISMATCH variability_cells | cells with >=1 two-run config but no row:', cells_mine - cells_pub); t.bad += 1
print('   decode cells:', sorted(V[DECODE].cell.unique()))
t.done()

# ================================================================ variability_batch
t = Tally('variability_batch')
pub = pd.read_csv(T + 'variability_batch.csv')
def bin_of(row):
    if row.cell == 'c1_decode_batch': return {1: '1', 4: '4 to 8', 16: '16', 64: '64', 256: 'above 64'}[int(row.tb)]
    return '4 to 8' if row.tb <= 8 else '64'            # c2 design targets are 8 and 64; a smaller target as run keeps its design bin
VB = V[V.cell.isin(['c1_decode_batch', 'c2_decode_context'])].copy(); VB['bin'] = VB.apply(bin_of, axis=1)
try:   # cross-check my binning against the published per-config file (published artefact, not derivation)
    pc = pd.read_csv(T + 'variability_per_config.csv').set_index('config').batch_bin
    dis = VB[VB.config.map(pc) != VB.bin]
    print(f'   variability_batch: bins disagreeing with published variability_per_config.csv: {len(dis)}'); [print('     ', c, b, '->', pc.get(c)) for c, b in zip(dis.config, dis.bin)]
except FileNotFoundError: pass
for _, r in pub.iterrows():
    g = VB[(VB.acc == r.Accelerator.lower()) & (VB.bin == str(r['Target batch']))]
    check_var_row(t, f'{r.Accelerator} {r["Target batch"]}', r.iloc[2], r.iloc[3], g, cfg2_tok=None, cfg1_tok=None, max_tok=str(r.iloc[5]))
    t.check(f'{r.Accelerator} {r["Target batch"]} configs', str(r.Configs), len(g))
t.done()

# ================================================================ latency
t = Tally('latency')
pub = pd.read_csv(T + 'latency.csv'); summ = json.load(open(T + 'latency_summary.json'))
missing, ambiguous, recs, recs_all = [], [], {}, {}
for _, r in c1.iterrows():
    cands = []
    for p in glob.glob(f'{STORE}/*/cells/{r.run_id}/requests.parquet'):
        pod = json.load(open(p.replace('requests.parquet', 'cell.json')))['pod']['pod_id']
        if pod != r.pod_id: continue
        rq = pd.read_parquet(p, columns=['in_window', 'error', 'ttft_s', 'completion_tokens'])
        rq = rq[(rq.in_window == True) & rq.error.isna()]
        cands.append((p, rq))
    if not cands: missing.append(r.run_id); continue
    recs_all[r.run_id] = pd.concat([c[1].ttft_s.dropna() for c in cands])
    if len(cands) > 1:
        same = [c for c in cands if abs(c[1].completion_tokens.sum() - r.completion_tokens) < 0.5]
        ambiguous.append((r.run_id, [(c[0].split('/store/')[1].split('/')[0], len(c[1]), int(c[1].completion_tokens.sum())) for c in cands], int(r.completion_tokens)))
        cands = same if len(same) == 1 else cands[:1]
    recs[r.run_id] = cands[0][1].ttft_s.dropna()
print('   latency: runs with records', len(recs), '| missing', missing)
for a in ambiguous: print('   latency: several same-pod copies for', a[0], '| (branch, in-window requests, completion tokens):', a[1], '| corpus completion tokens', a[2])
itl_rule = {'ratio_of_means': 0, 'mean_of_ratios': 0}; dup_ok = dup_n = 0
for _, r in pub.iterrows():
    t.row(); mdl = {v: k for k, v in NAME.items()}[r.Model]; acc = r['Acc.'].lower()
    g = c1[(c1.model == mdl) & (c1.accelerator == acc) & (c1.tb == r['Target B'])]
    tt = pd.concat([recs[x] for x in g.run_id if x in recs])
    B = g.B.mean(); itl_a = B / g.tokens_per_s.mean() * 1000; itl_b = (g.B / g.tokens_per_s).mean() * 1000
    t.check(f'{r.Model} {acc} B{r["Target B"]} achieved B', str(r['Achieved B']), B)
    t.check(f'{r.Model} {acc} B{r["Target B"]} TTFT median', str(r.iloc[4]), tt.median())
    t.check(f'{r.Model} {acc} B{r["Target B"]} TTFT p90', str(r.iloc[5]), tt.quantile(0.9))
    t.check(f'{r.Model} {acc} B{r["Target B"]} ITL', str(r.iloc[6]), itl_a, f'(mean of per-run ratios: {itl_b:.2f})')
    t.check(f'{r.Model} {acc} B{r["Target B"]} requests', str(r.Requests), len(tt))
    ta = pd.concat([recs_all[x] for x in g.run_id]); dup_n += 1   # alternative: every same-pod copy pooled (duplicates counted twice)
    dup_ok += int(len(ta) == r.Requests and match(str(r.iloc[4]), ta.median()) and match(str(r.iloc[5]), ta.quantile(0.9)))
    j = summ['by_config'][f'{mdl}/{acc}/{int(r["Target B"])}']   # full-precision json: relative 1e-6
    for k, v in [('ttft_median_s', tt.median()), ('ttft_p90_s', tt.quantile(0.9)), ('B', B)]:
        t.fields += 1
        if abs(v - j[k]) > 1e-6 * abs(j[k]): t.bad += 1; print(f'MISMATCH latency json | {mdl}/{acc}/{int(r["Target B"])} {k} | published {j[k]} | mine {v}')
    t.fields += 1
    if abs(itl_a / 1000 - j['itl_s']) <= 1e-6 * j['itl_s']: itl_rule['ratio_of_means'] += 1
    elif abs(itl_b / 1000 - j['itl_s']) <= 1e-6 * j['itl_s']: itl_rule['mean_of_ratios'] += 1
    else: t.bad += 1; print(f'MISMATCH latency json | {mdl}/{acc}/{int(r["Target B"])} itl_s | published {j["itl_s"]} | mine {itl_a/1000:.6g} / {itl_b/1000:.6g}')
print('   latency json itl_s matched by rule:', itl_rule)
print(f'   latency ALTERNATIVE (pool every same-pod copy, duplicates counted twice): {dup_ok} of {dup_n} rows reproduce Requests, median and p90')
t.done()

# ================================================================ factors (roll-up from PUBLISHED source CSVs)
t = Tally('factors')
pub = pd.read_csv(T + 'factors.csv').set_index('Factor')
def check_factor(label, vals, n_tok=None, dbl=None):
    t.row(); vals = np.asarray(vals, float); toks = re.findall(r'[\d.]+', pub.loc[label].iloc[1])
    t.check(f'{label} median', toks[0], np.median(vals)); t.check(f'{label} min', toks[1], vals.min()); t.check(f'{label} max', toks[2], vals.max())
    t.check(f'{label} comparisons', str(pub.loc[label].Comparisons), len(vals))
    if dbl is not None:
        d = np.asarray(dbl, float); toks = re.findall(r'[\d.]+', pub.loc[label]['Per doubling'])
        t.check(f'{label} per doubling median', toks[0], np.median(d)); t.check(f'{label} per doubling min', toks[1], d.min()); t.check(f'{label} per doubling max', toks[2], d.max())
    return vals
# 1 Batch: decode_batch.csv, e(target 1) / e(highest target present) per model x acc; per doubling over log2(Btop/B1), achieved B as printed
db = pd.read_csv(T + 'decode_batch.csv'); db['e'] = db['J per output token'].map(num)
rat, dbl, dbl_full = [], [], []
for (m, a), g in db.groupby(['Model', 'Acc.']):
    g = g.set_index('Target B'); top = g.index.max(); rr = g.e[1] / g.e[top]; rat.append(rr)   # highest load present (v1.0.1: 64 for Llama on H100)
    dbl.append(rr ** (1 / math.log2(g['Achieved B'][top] / g['Achieved B'][1])))
    gf = c1[(c1.model == {v: k for k, v in NAME.items()}[m]) & (c1.accelerator == a.lower())].groupby('tb').B.mean()
    dbl_full.append(rr ** (1 / math.log2(gf[gf.index.max()] / gf[1])))
check_factor('Batch', rat, dbl=dbl)
print('   factors Batch per doubling with full-precision achieved B from corpus: median %.3f (%.3f to %.3f)' % (np.median(dbl_full), min(dbl_full), max(dbl_full)))
# 2 Cached prefix: cached_prefix.csv, uncached / cached central values at 16,384 cached
cp = pd.read_csv(T + 'cached_prefix.csv'); cp['c'] = cp.iloc[:, 5].map(num); cp['u'] = cp.iloc[:, 6].map(num); cp['r'] = cp.iloc[:, 7].map(num); cp['pre'] = cp['Cached prefix'].map(num)
g = cp[cp.pre == 16384]; check_factor('Cached prefix', g.u / g.c)
print('   factors Cached prefix via 1/(printed Cached/uncached): median %.3f (%.3f to %.3f)' % (np.median(1 / g.r), (1 / g.r).min(), (1 / g.r).max()),
      '| at 1k: median saving %.1f%% (U/C) / %.1f%% (1-C/U)' % ((np.median(cp[cp.pre == 1024].u / cp[cp.pre == 1024].c) - 1) * 100, (1 - np.median(cp[cp.pre == 1024].r)) * 100))
# 3 Memory limit: kv_pressure.csv, each collapsed row over the nearest lower non-collapsed target of its own (experiment, offer) series
kv = pd.read_csv(T + 'kv_pressure.csv'); kv['e'] = kv['J per output token'].astype(float)
vals = []
for _, r in kv[kv.Collapsed == 'yes'].iterrows():
    s = kv[(kv.Experiment == r.Experiment) & (kv['GPU memory offer'] == r['GPU memory offer']) & (kv.Collapsed == 'no') & (kv['Target B'] < r['Target B'])]
    below = s[s['Target B'] == s['Target B'].max()].e.mean(); vals.append(r.e / below)
check_factor('Memory limit for stored context', vals)
# 4/5 Context: decode_context.csv, longest non-collapsed context over 1k per model x acc and design batch; per doubling over log2(context ratio)
dc = pd.read_csv(T + 'decode_context.csv'); dc['e'] = dc['J per output token'].map(num)
dc['cc'] = dc['Cached context (tokens)'].map(num); dc['ac'] = dc['Attended context (tokens)'].map(num)
for label, design, cap in [('Context length, batch 64', 64, np.inf), ('Context length, batch 8', 8, 32768)]:   # batch 8: "longest context (16k to 32k)"
    rat, d_c, d_a = [], [], []
    for (m, a), g in dc[(dc['Design target B'] == design) & (dc.Collapse == 'no') & (dc.cc <= cap)].groupby(['Model', 'Acc.']):
        base = g[g.cc == 1024]; top = g[g.cc == g.cc.max()]; assert len(base) == 1 and len(top) == 1, (m, a, design)
        rr = top.e.iloc[0] / base.e.iloc[0]; rat.append(rr)
        d_c.append(rr ** (1 / math.log2(top.cc.iloc[0] / base.cc.iloc[0]))); d_a.append(rr ** (1 / math.log2(top.ac.iloc[0] / base.ac.iloc[0])))
    check_factor(label, rat, dbl=d_c)
    print(f'   factors {label}: per doubling on attended-context basis: median %.3f (%.3f to %.3f)' % (np.median(d_a), min(d_a), max(d_a)))
# 6/7 GPU generation: b200_vs_h100.csv central ratios
bh = pd.read_csv(T + 'b200_vs_h100.csv'); bh['r'] = bh.iloc[:, 4].map(num)
check_factor('GPU generation, same GPU count', bh[bh.Comparison == 'same GPUs'].r)
check_factor('One B200 against two H100s', bh[bh.Comparison == 'fewer B200s'].r)
# 8 Weight precision: precision.csv, fp8 over nvfp4 = 1 / (nvfp4/fp8), excluding the Llama 16k / design-64 row (paper Sec. precision)
pr = pd.read_csv(T + 'precision.csv'); pr['r'] = pr['nvfp4 / fp8'].map(num); pr['cc'] = pr['Cached context (tokens)'].map(num)
pr['fp8'] = pr.iloc[:, 6].map(num); pr['nv'] = pr.iloc[:, 7].map(num)
keep = pr[~((pr.Model == 'Llama-3.3-70B') & (pr['Design target B'] == 64) & (pr.cc == 16384))]
check_factor('Weight precision', 1 / keep.r)
print('   factors Weight precision via fp8 J / nvfp4 J central values: median %.3f (%.3f to %.3f)' % (np.median(keep.fp8 / keep.nv), (keep.fp8 / keep.nv).min(), (keep.fp8 / keep.nv).max()))
# 9 Tensor parallelism: tensor_parallel.csv Ratio column
tp = pd.read_csv(T + 'tensor_parallel.csv'); check_factor('Tensor parallelism', tp.Ratio.map(num))
# 10 Repeat-to-repeat: larger over smaller of the two repeats, decode configurations (from the corpus, as in variability_cells)
t.row(); rr = V[DECODE].ratio; cellv = pub.loc['Repeat-to-repeat variation'].iloc[1]; toks = [str(num(cellv)), str(num(cellv.split('percentile')[1]))]
t.check('Repeat variation median', toks[0], rr.median()); t.check('Repeat variation p90', toks[1], q90(rr)); t.check('Repeat variation n', str(pub.loc['Repeat-to-repeat variation'].Comparisons), len(rr))
t.done()
print('NOTE: the factors check rolls up the PUBLISHED source tables; it is only as good as those tables (checked separately).')
