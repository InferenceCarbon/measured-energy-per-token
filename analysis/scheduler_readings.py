"""How evenly the server's scheduler was read during the batch experiment (c1, replication runs), and how
far the count of running requests swung within a window. Supports the statements of the Limitations section.

    python analysis/scheduler_readings.py --store PATH_TO_RAW_STORE

Reads raw/<branch>/cells/<run_id>/sched_samples.jsonl (time `t`, `running`) for every replication run of the
batch experiment that is in use, and writes paper/tables/scheduler_readings.json:

- the largest gap between two readings, by target;
- the share of the window in which the running count was below four-fifths of the run's achieved batch, with the
  readings weighted four ways (each reading counted once; each held until the next; trapezoid; each given the
  interval before it), because the readings are unevenly spaced at the highest targets;
- the achieved batch recomputed with the middle readings weighted by time, over the corpus value.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

WEIGHTINGS = ("each_reading_once", "held_until_next", "trapezoid", "interval_before")


def one_run(path: Path, B: float) -> dict[str, float]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    t = np.array([r["t"] for r in rows], float)
    n = np.array([r["running"] for r in rows], float)
    low = (n < 0.8 * B).astype(float)
    dt = np.diff(t)
    k0, k1 = int(0.1 * len(n)), int(np.ceil(0.9 * len(n)))
    tm, nm = t[k0:k1], n[k0:k1]
    return {"each_reading_once": float(low.mean()), "held_until_next": float((low[:-1] * dt).sum() / dt.sum()),
            "trapezoid": float(((low[:-1] + low[1:]) / 2 * dt).sum() / dt.sum()), "interval_before": float((low[1:] * dt).sum() / dt.sum()),
            "gap_max_s": float(dt.max()), "readings": int(len(n)),
            "time_weighted_batch_over_corpus": float(((nm[:-1] + nm[1:]) / 2 * np.diff(tm)).sum() / np.diff(tm).sum() / B) if len(tm) > 2 else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True, nargs="+")
    ap.add_argument("--features", type=Path, default=build.DEFAULT_FEATURES)
    a = ap.parse_args()
    df = build.load(a.features)
    d = df[(df.cell == "c1_decode_batch") & df.decode_ok & (df.series == "main") & (df.source == "replication")]
    out = []
    for _, r in d.iterrows():
        hits = [p for st in a.store for p in [st / str(r.branch) / "cells" / r.run_id / "sched_samples.jsonl"] if p.exists()]
        if not hits:
            sys.exit(f"no scheduler readings found for {r.run_id} on {r.branch}")
        out.append({"target": int(r.v_target_batch), **one_run(hits[0], float(r.B))})
    o = pd.DataFrame(out)
    hi, lo = o[o.target > 256], o[o.target <= 64]
    summ = {"runs": int(len(o)), "runs_above_256": int(len(hi)),
            "gap_max_s": {"targets_above_256": float(hi.gap_max_s.max()), "targets_up_to_64": float(lo.gap_max_s.max())},
            "share_below_four_fifths_above_256": {w: {"median": float(hi[w].median()), "min": float(hi[w].min()), "max": float(hi[w].max())} for w in WEIGHTINGS},
            "share_below_four_fifths_up_to_64_median_by_target": {w: [float(lo.groupby("target")[w].median().min()), float(lo.groupby("target")[w].median().max())] for w in WEIGHTINGS},
            "time_weighted_batch_over_corpus": {"targets_above_256": [float(hi.time_weighted_batch_over_corpus.min()), float(hi.time_weighted_batch_over_corpus.max())],
                                                "targets_up_to_256": [float(o[o.target <= 256].time_weighted_batch_over_corpus.min()), float(o[o.target <= 256].time_weighted_batch_over_corpus.max())]}}
    (build.TABLES / "scheduler_readings.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
