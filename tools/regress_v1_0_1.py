"""Regression check of the v1.1 build against the published v1.0.1 tables.

    python tools/regress_v1_0_1.py

For each table of v1.0.1 (read from the git tag), how many of its rows appear unchanged in the v1.1 table, compared on
the columns the two share. Rows are expected to change only where v1.1 pooled later readings with a configuration,
re-measured it (the simulated agent), or relabeled it (the capped target-256 rows are now target 100).
"""
import io
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
names = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "--name-only", "v1.0.1", "paper/tables/"], capture_output=True, text=True).stdout.split()
print(f"{'table':28s} {'v1.0.1 rows':>11s} {'v1.1 rows':>9s} {'unchanged':>9s} {'not found':>9s}")
for n in sorted(x for x in names if x.endswith(".csv")):
    old = pd.read_csv(io.StringIO(subprocess.run(["git", "-C", str(ROOT), "show", "v1.0.1:" + n], capture_output=True, text=True).stdout), dtype=str).fillna("")
    f = ROOT / n
    if not f.exists():
        print(f"{Path(n).stem:28s} {len(old):11d}   missing")
        continue
    new = pd.read_csv(f, dtype=str).fillna("")
    common = [c for c in old.columns if c in new.columns]
    so, sn = list(map(tuple, old[common].values)), set(map(tuple, new[common].values))
    same = sum(r in sn for r in so)
    print(f"{Path(n).stem:28s} {len(old):11d} {len(new):9d} {same:9d} {len(so) - same:9d}   (compared on {len(common)} of {len(old.columns)} columns)")
