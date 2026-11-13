"""Latency against batch (experiment c1): time to first token from the per-request
records in the Zenodo deposit, and the interval between a request's output tokens
from the corpus.

    python analysis/latency.py --store PATH_TO_RAW_STORE

Writes paper/tables/latency.{tex,md,csv}. The per-request records
(raw/<branch>/cells/<run_id>/requests.parquet) are not in this repository, so
analysis/build.py does not produce this table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

TARGETS = [1, 4, 16, 64, 256]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=build.DEFAULT_FEATURES)
    a = ap.parse_args()

    df = build.load(a.features)
    d = df[(df.cell == "c1_decode_batch") & df.decode_ok].copy()
    rows, hl = [], {}
    for (m, acc), g in d.groupby(["model", "accelerator"]):
        for tb in TARGETS:
            gg = g[g.v_target_batch == tb]
            if gg.empty:
                continue
            ttft = []
            for _, r in gg.iterrows():
                # v1.0.1: one copy per run. A run can sit on a shared campaign branch and on its pod's own results
                # branch; v1.0.0 pooled both copies, doubling the request counts of 21 of the 40 rows. Where the
                # copies differ (one run does), the copy whose completion tokens match the corpus row is the one used.
                copies = []
                for c in sorted(a.store.glob(f"*/cells/{r.run_id}")):
                    try:
                        if json.loads((c / "cell.json").read_text())["pod"]["pod_id"] != r.pod_id:
                            continue
                    except (KeyError, ValueError, FileNotFoundError):
                        continue
                    rq = pd.read_parquet(c / "requests.parquet")
                    rq = rq[(rq.in_window == True) & (rq.error.isna())]  # noqa: E712
                    copies.append((abs(float(rq.completion_tokens.fillna(0).sum()) - float(r.completion_tokens)), str(c), rq.ttft_s.dropna()))
                if copies:
                    ttft.append(min(copies)[2])
            if not ttft:
                sys.exit(f"no request records found for {m}/{acc}/target {tb}")
            t = pd.concat(ttft)
            B = float(gg.B.mean())
            itl = B / float(gg.tokens_per_s.mean())
            rows.append({"Model": build.MODEL_LABEL[m], "Acc.": build.ACC_LABEL[acc], "Target B": tb, "Achieved B": f"{B:.1f}",
                         "Time to first token, median (s)": f"{t.median():.2f}", "90th percentile (s)": f"{t.quantile(0.9):.2f}",
                         "Interval between output tokens (ms)": f"{1000 * itl:.1f}", "Requests": int(len(t))})
            hl[f"{m}/{acc}/{tb}"] = {"ttft_median_s": float(t.median()), "ttft_p90_s": float(t.quantile(0.9)), "itl_s": itl, "B": B}
    build.write_table("latency", pd.DataFrame(rows),
                      "Latency against batch (experiment c1, the runs of Table~\\ref{tab:decode_batch}). Time to first token is measured by the client from sending a request to its first streamed token, pooled over the two runs' in-window requests (one copy of each run); at a target of 256 it includes waiting for one of the client's 100 connections and for a server slot, so it is a queueing time, not the server's prefill time. The interval between a request's output tokens is the achieved batch over the window's token rate, from the corpus, so it is a mean over the window rather than a per-request measurement.",
                      "tab:latency", col_fmt="llrrrrrr", longtable=True)
    summ = {}
    for tb in TARGETS:
        v = [x for k, x in hl.items() if k.endswith(f"/{tb}")]
        summ[tb] = {"ttft_median_range_s": [min(x["ttft_median_s"] for x in v), max(x["ttft_median_s"] for x in v)],
                    "itl_range_ms": [1000 * min(x["itl_s"] for x in v), 1000 * max(x["itl_s"] for x in v)]}
    (build.TABLES / "latency_summary.json").write_text(json.dumps({"by_config": hl, "by_target": summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
