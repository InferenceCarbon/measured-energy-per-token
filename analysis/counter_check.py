"""Counter check (Appendix E): the board's cumulative energy counter against the
trapezoid integral of the raw power samples, for every GPU window of the 917 runs.

    python analysis/counter_check.py --store PATH_TO_RAW_STORE

The raw power samples are in the Zenodo deposit (raw/<branch>/cells/<run_id>/raw/),
not in this repository. Differences are counter minus integral, as a share of the
counter. The integral holds the first and last samples constant out to the window's
edges ("edge-extended"); the plain trapezoid is reported beside it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "features.parquet")
    a = ap.parse_args()

    feats = pd.read_parquet(a.features)
    cells = pd.read_parquet(a.features.parent / "store_tables" / "cells.parquet")
    act = cells[(cells.kind == "active") & cells.run_id.isin(set(feats.run_id))]
    rows, missing = [], 0
    for _, w in act.iterrows():
        run_dir = None
        for c in a.store.glob(f"*/cells/{w.run_id}"):
            try:
                if json.loads((c / "cell.json").read_text())["pod"]["pod_id"] == w.pod_id:
                    run_dir = c
            except (KeyError, ValueError, FileNotFoundError):
                continue
        if run_dir is None:
            missing += 1
            continue
        for g in json.loads(w.per_gpu_json):
            f = run_dir / "raw" / f"active.gpu{g['gpu_index']}.jsonl"
            if not f.exists():
                missing += 1
                continue
            s = pd.read_json(f, lines=True)
            t, p = s.t.to_numpy(), s.p_w.to_numpy()
            trap = float(np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(t)))
            ext = trap + p[0] * (t[0] - g["start_s"]) + p[-1] * (g["end_s"] - t[-1])
            rows.append(dict(run_id=w.run_id, dur=g["duration_s"], counter=g["energy_j"], trap=trap, ext=ext))
    r = pd.DataFrame(rows)
    print(f"GPU windows: {len(r)}, runs: {r.run_id.nunique()}, missing: {missing}")
    for col, label in (("ext", "edge-extended"), ("trap", "plain trapezoid")):
        d = (r.counter - r[col]) / r.counter
        print(f"{label}: signed median {d.median():+.3%}, p5 {d.quantile(.05):+.3%}, p95 {d.quantile(.95):+.3%}, "
              f"max |d| {d.abs().max():.3%}, median |d| {d.abs().median():.3%}")
    short = r[r.dur < 60]
    d = (short.counter - short.ext) / short.counter
    print(f"windows under 60 s: {len(short)}, signed median {d.median():+.3%}")


if __name__ == "__main__":
    main()
