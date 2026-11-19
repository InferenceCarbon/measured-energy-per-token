"""Compare two sets of build tables (paper/tables): which rows and which headline numbers differ.

    python tools/compare_tables.py --old <dir> --new <dir> [--out report.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}{k}."))
    elif isinstance(d, list):
        if all(not isinstance(x, (dict, list)) for x in d):
            out[pre[:-1]] = d
        else:
            for i, v in enumerate(d):
                out.update(flat(v, f"{pre}{i}."))
    else:
        out[pre[:-1]] = d
    return out


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b) or abs(a - b) <= 1e-12 * max(abs(a), abs(b))
    return a == b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rows", type=int, default=6)
    a = ap.parse_args(argv)
    L = ["# Table comparison", "", f"Old: `{a.old}`. New: `{a.new}`.", "", "| Table | Rows old | Rows new | Unchanged | Changed or new | Removed |", "|---|---|---|---|---|---|"]
    detail = []
    for f in sorted(a.old.glob("*.csv")):
        g = a.new / f.name
        if not g.exists():
            L.append(f"| {f.stem} | | missing | | | |")
            continue
        o, n = pd.read_csv(f, dtype=str).fillna(""), pd.read_csv(g, dtype=str).fillna("")
        if list(o.columns) != list(n.columns):
            L.append(f"| {f.stem} | {len(o)} | {len(n)} | columns differ | | |")
            continue
        so, sn = set(map(tuple, o.values)), set(map(tuple, n.values))
        L.append(f"| {f.stem} | {len(o)} | {len(n)} | {len(so & sn)} | {len(sn - so)} | {len(so - sn)} |")
        if sn != so:
            detail.append(f"\n## {f.stem}\n")
            detail.append("Columns: " + " | ".join(o.columns))
            for r in sorted(so - sn)[:a.rows]:
                detail.append("- old: " + " | ".join(r))
            for r in sorted(sn - so)[:a.rows]:
                detail.append("- new: " + " | ".join(r))
            if len(so - sn) > a.rows or len(sn - so) > a.rows:
                detail.append(f"- ({len(so - sn)} old and {len(sn - so)} new rows in all)")
    ho, hn = flat(json.loads((a.old / "headline.json").read_text())), flat(json.loads((a.new / "headline.json").read_text()))
    changed = [(k, ho.get(k), hn.get(k)) for k in sorted(set(ho) | set(hn)) if not same(ho.get(k), hn.get(k))]
    L += ["", f"Headline numbers (headline.json): {len(ho)} old keys, {len(hn)} new keys, {len(changed)} differ.", ""]
    L += detail
    L += ["", "## Headline numbers that differ", "", "| Key | Old | New |", "|---|---|---|"]
    for k, x, y in changed:
        fx = f"{x:.4g}" if isinstance(x, float) else str(x)
        fy = f"{y:.4g}" if isinstance(y, float) else str(y)
        L.append(f"| {k} | {fx[:60]} | {fy[:60]} |")
    text = "\n".join(L) + "\n"
    if a.out:
        a.out.write_text(text)
    print(text[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
