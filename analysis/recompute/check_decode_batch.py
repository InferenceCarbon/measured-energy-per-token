"""Independent recompute of the decode_batch and idle_power tables from features.parquet (version 1.0.0 audit),
sharing no code with analysis/build.py.  Usage: python analysis/recompute/check_decode_batch.py [REPO]
"""
import os, sys
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Independent recompute of Table decode_batch and idle_power from features.parquet.
Shares no code with analysis/build.py: energy_j / completion_tokens per run, mean of per-pod means, band = range of per-pod means."""
import re, sys, pandas as pd, numpy as np
f = pd.read_parquet(REPO+'/data/features.parquet')
f = f[f['dropped'].isna()];
_cr = pd.read_csv(REPO+'/data/cache_reuse.csv'); _rr = dict(zip(_cr.run_id, _cr.cache_reuse_r)); _ev = f.run_id.map(_rr).lt(0.95) & f.cell.isin(['c1_decode_batch','c2_decode_context','c6_tensor_parallel','c7_precision']); print('cache-reuse rule:', int(_ev.sum()), 'runs excluded'); f = f[~_ev].copy(); print('dropped excluded:', int((~pd.read_parquet(REPO+'/data/features.parquet')['dropped'].isna()).sum()))
f['tb'] = f['run_id'].str.extract(r'target_batch-(\d+)').astype(float)
c1 = f[f.cell=='c1_decode_batch'].copy()
c1['e'] = c1.energy_j / c1.completion_tokens
pod = c1.groupby(['model','accelerator','tp','tb','pod_id']).agg(e=('e','mean'), B=('B','mean'), W=('mean_power_w','mean')).reset_index()
g = pod.groupby(['model','accelerator','tp','tb']).agg(e=('e','mean'), elo=('e','min'), ehi=('e','max'), B=('B','mean'), W=('W','mean'), pods=('pod_id','nunique')).reset_index()
rep = c1.groupby(['model','accelerator','tp','tb']).e.agg(['min','max']).reset_index()
g = g.merge(rep, on=['model','accelerator','tp','tb']); one = g.pods==1; g.loc[one,'elo']=g.loc[one,'min']; g.loc[one,'ehi']=g.loc[one,'max']
pub = pd.read_csv(REPO+'/paper/tables/decode_batch.csv')
pub['model']=pub['Model'].str.lower(); pub['accelerator']=pub['Acc.'].str.lower()
m = pub.merge(g, left_on=['model','accelerator','TP','Target B'], right_on=['model','accelerator','tp','tb'], how='left')
def num(s): return float(re.match(r'[\d.,]+', str(s)).group().replace(',',''))
def band(s):
    x = re.findall(r'[\d.]+', str(s)); return [float(v) for v in x[1:3]] if len(x)>=3 else [None,None]
bad=0
for _,r in m.iterrows():
    pe=num(r['J per output token']); lo,hi=band(r['J per output token'])
    ok = abs(pe-r.e)/pe<0.006 and (lo is None or abs(lo-r.elo)/lo<0.006) and (hi is None or abs(hi-r.ehi)/hi<0.006) and abs(num(r['Achieved B'])-r.B)<0.06 and abs(num(r['Board W'])-r.W)/num(r['Board W'])<0.006 and int(r['Pods'])==r.pods
    if not ok: bad+=1; print('MISMATCH', r['Model'], r['Acc.'], r['TP'], r['Target B'], '| pub', r['J per output token'], r['Achieved B'], r['Board W'], r['Pods'], '| mine', round(r.e,4), round(r.elo,4), round(r.ehi,4), round(r.B,1), round(r.W,1), r.pods)
print(f'decode_batch: {len(m)} rows checked, {bad} mismatches')
# idle power: loaded_idle_power_w is per run; per-pod mean then mean/range
ip = f[f.cell=='c0_static'].groupby(['model','accelerator','n_gpus','pod_id']).loaded_idle_power_w.mean().reset_index()
gi = ip.groupby(['model','accelerator','n_gpus']).loaded_idle_power_w.agg(['mean','min','max','count']).reset_index()
pubi = pd.read_csv(REPO+'/paper/tables/idle_power.csv'); pubi['model']=pubi.Model.str.lower(); pubi['accelerator']=pubi.Accelerator.str.lower()
mi = pubi.merge(gi, left_on=['model','accelerator','GPUs'], right_on=['model','accelerator','n_gpus'])
badi=0
for _,r in mi.iterrows():
    pe=num(r['Board W, all GPUs']); lo,hi=band(r['Board W, all GPUs'])
    if not (abs(pe-r['mean'])/pe<0.006 and abs(lo-r['min'])/lo<0.006 and abs(hi-r['max'])/hi<0.006 and r.Pods==r['count']):
        badi+=1; print('MISMATCH idle', r.Model, r.Accelerator, r['Board W, all GPUs'], '| mine', round(r['mean'],1), round(r['min'],1), round(r['max'],1), r['count'])
print(f'idle_power: {len(mi)} rows checked, {badi} mismatches')
