"""Read every corpus run's raw cell.json (server command and run settings) from the meter repository's
result branches and write one row per run: data the tables depend on, as the server actually ran it.

    python tools/server_settings.py OUT.csv   (needs the meter repository with its result branches; writes the raw extract,
    from which data/server_settings.csv keeps the server columns)
"""
import json, re, subprocess, sys
from pathlib import Path
import pandas as pd

import os
METER = Path(os.environ.get('METER_REPO', Path.home() / 'inferencecarbon-meter'))   # the meter repository with its result branches
CORPUS = Path(__file__).resolve().parents[1] / 'data' / 'features.parquet'
out = sys.argv[1]
git = lambda *a: subprocess.run(['git', '-C', str(METER), *a], capture_output=True, text=True).stdout
branches = [b.strip() for b in git('branch', '-r').split() if b.startswith('live/')]
index = {}
for b in branches:
    for f in git('ls-tree', '-r', '--name-only', b).split():
        if f.startswith('cells/') and f.endswith('/cell.json'):
            index.setdefault(f[6:-10], []).append(b)
d = pd.read_parquet(CORPUS)
specs = []
for _, r in d.iterrows():
    cands = index.get(r.run_id, [])
    if r.source == 'replication': cands = [b for b in cands if b == 'live/' + str(r.branch)]
    else: cands = [b for b in cands if not b.startswith('live/repl')]
    if len(cands) > 1:
        # the same run id can sit on several branches (a shared campaign branch, a pod's own branch, an earlier attempt):
        # keep the copies recorded on this run's pod
        def on_pod(b):
            try:
                return json.loads(git('show', f'{b}:cells/{r.run_id}/cell.json'))['pod']['pod_id'] == r.pod_id
            except (KeyError, ValueError):
                return False
        cands = [b for b in cands if on_pod(b)] or cands
    specs.append((r.uid, cands[0] if cands else None, r.run_id))
req = ''.join(f'{b}:cells/{rid}/cell.json\n' for _, b, rid in specs if b)
p = subprocess.run(['git', '-C', str(METER), 'cat-file', '--batch'], input=req, capture_output=True, text=True)
blobs, s = [], p.stdout
while s:
    head, _, rest = s.partition('\n')
    if head.endswith('missing'): blobs.append(None); s = rest; continue
    n = int(head.split()[2]); blobs.append(rest[:n]); s = rest[n + 1:]
it = iter(blobs); rows = []
for uid, b, rid in specs:
    j = json.loads(next(it)) if b else None
    row = {'uid': uid, 'branch_read': b}
    if j:
        cmd = j.get('server', {}).get('command', '') or ''
        if isinstance(cmd, list): cmd = ' '.join(map(str, cmd))
        arg = lambda k: (re.search(rf'--{k}[ =](\S+)', cmd) or [None, None])[1]
        run = j.get('run', {}); var = run.get('variant', {}) or {}; fx = run.get('fixed', {}) or {}
        row.update(checkpoint=(re.search(r'vllm serve (\S+)', cmd) or [None, None])[1], tp_cmd=arg('tensor-parallel-size'),
                   mem_offer=arg('gpu-memory-utilization'), max_model_len=arg('max-model-len'), max_num_seqs=arg('max-num-seqs'),
                   kv_dtype=arg('kv-cache-dtype'), prefix_caching=('--enable-prefix-caching' in cmd), no_prefix=('--no-enable-prefix-caching' in cmd),
                   speculative=(re.search(r'--speculative-config (\{.*?\})', cmd) or [None, None])[1], quantization=arg('quantization'),
                   vllm_version=j.get('server', {}).get('vllm_version'), target_batch_run=var.get('target_batch'),
                   context_cached_run=var.get('context_cached', fx.get('context_cached')), fresh_in_run=var.get('fresh_in', fx.get('fresh_in')),
                   out_run=fx.get('out', var.get('out')), prefix_cached_run=var.get('prefix_cached', fx.get('prefix_cached')),
                   control_run=var.get('control', fx.get('control')), cmd=cmd)
    rows.append(row)
pd.DataFrame(rows).to_csv(out, index=False)
print('runs', len(rows), 'with a raw record', sum(1 for r in rows if r.get('checkpoint')))
