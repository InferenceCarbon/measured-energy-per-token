"""Latency against batch (experiment c1): time to first token and the interval between
a request's streamed tokens, both from the per-request records in the Zenodo deposit.

    python analysis/latency.py --store PATH_TO_RAW_STORE [PATH_TO_RAW_STORE ...]

Give every raw store the corpus was built from (version 1.0.0 and the replication).

Writes paper/tables/latency.{tex,md,csv}. The per-request records
(raw/<branch>/cells/<run_id>/requests.parquet) are not in this repository, so
analysis/build.py does not produce this table.
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


def stream_intervals(rq: pd.DataFrame) -> pd.DataFrame:
    """Per request: the mean interval between its streamed tokens (s) and the share of its output tokens that were
    streamed. Each stream event is (time, characters, tokens). The interval is the time from the first event to the
    last over the tokens streamed after the first. Requests with fewer than two events have no interval."""
    out = []
    for ev, n in zip(rq.events_json, rq.completion_tokens):
        ev = json.loads(ev) if isinstance(ev, str) else ev
        if ev is None or len(ev) < 2:
            continue
        tok = sum(e[2] for e in ev)
        after_first = tok - ev[0][2]
        if after_first < 1:
            continue
        out.append(((ev[-1][0] - ev[0][0]) / after_first, tok / n if n else np.nan))
    return pd.DataFrame(out, columns=["itl_s", "streamed_share"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True, nargs="+")
    ap.add_argument("--features", type=Path, default=build.DEFAULT_FEATURES)
    a = ap.parse_args()

    df = build.load(a.features)
    d = df[(df.cell == "c1_decode_batch") & df.decode_ok & (df.series == "main")].copy()
    rows, hl = [], {}
    for (m, acc), g in d.groupby(["model", "accelerator"]):
        for tb in sorted(int(x) for x in g.v_target_batch.unique()):
            gg = g[g.v_target_batch == tb]
            if gg.empty:
                continue
            ttft, itls = [], []
            for _, r in gg.iterrows():
                # v1.0.1: one copy per run. A run can sit on a shared campaign branch and on its pod's own results
                # branch; v1.0.0 pooled both copies, doubling the request counts of 21 of the 40 rows. Where the
                # copies differ (one run does), the copy whose completion tokens match the corpus row is the one used.
                copies = []
                for c in sorted(x for st in a.store for x in st.glob(f"*/cells/{r.run_id}")):
                    try:
                        if json.loads((c / "cell.json").read_text())["pod"]["pod_id"] != r.pod_id:
                            continue
                        if r.source != "v1.0.0" and c.parents[len(Path(r.run_id).parts)].name != r.branch:
                            continue
                    except (KeyError, ValueError, FileNotFoundError):
                        continue
                    rq = pd.read_parquet(c / "requests.parquet")
                    rq = rq[(rq.in_window == True) & (rq.error.isna())]  # noqa: E712
                    copies.append((abs(float(rq.completion_tokens.fillna(0).sum()) - float(r.completion_tokens)), str(c), rq))
                if copies:
                    rq = min(copies, key=lambda x: x[:2])[2]
                    ttft.append(rq.ttft_s.dropna())
                    itls.append(stream_intervals(rq))
            if not ttft:
                sys.exit(f"no request records found for {m}/{acc}/target {tb}")
            t = pd.concat(ttft)
            si = pd.concat(itls)
            B = float(gg.B.mean())
            derived = B / float(gg.tokens_per_s.mean())  # running requests over the window's token rate
            itl = float(si.itl_s.median())
            rows.append({"Model": build.MODEL_LABEL[m], "Acc.": build.ACC_LABEL[acc], "Target B": tb, "Achieved B": f"{B:.1f}" if B < 100 else f"{B:,.0f}",
                         "Time to first token, median (s)": f"{t.median():.2f}", "90th percentile (s)": f"{t.quantile(0.9):.2f}",
                         "Interval between streamed tokens, median (ms)": f"{1000 * itl:.1f}", "Requests": int(len(t))})
            hl[f"{m}/{acc}/{tb}"] = {"ttft_median_s": float(t.median()), "ttft_p90_s": float(t.quantile(0.9)), "itl_s": itl, "B": B,
                                     "itl_requests": int(len(si)), "streamed_share_median": float(si.streamed_share.median()),
                                     "batch_over_token_rate_s": derived, "itl_over_batch_over_token_rate": itl / derived}
    ratios = sorted(x["itl_over_batch_over_token_rate"] for x in hl.values())
    part = lambda k: k.startswith("gemma-3-27b/") and not k.startswith("gemma-3-27b/h100/")  # noqa: E731
    gemma = [x["streamed_share_median"] for k, x in hl.items() if part(k)]
    others = [x["streamed_share_median"] for k, x in hl.items() if not part(k)]
    agree = {"rows": len(ratios), "median": float(np.median(ratios)), "min": ratios[0], "max": ratios[-1],
             "gemma_streamed_share": [min(gemma), float(np.median(gemma)), max(gemma)], "other_models_streamed_share_min": min(others)}
    CAPTION = ("Latency against batch (experiment c1, the runs of Table~\\ref{tab:decode_batch}). Both columns are measured by the client, pooled over the runs' in-window requests (one copy of each run). Time to first token runs from sending a request to its first streamed token. Where the server ran fewer requests than were offered, it includes the wait for a server slot (and, for the version 1.0 readings at target 100, for one of the client's 100 connections), so it is a queueing time, not the server's prefill time. The interval between streamed tokens is, for each request, the time from its first streamed token to its last over the tokens streamed after the first, and the table gives the median over requests. Over the rows it is a median STAT_MED of the achieved batch divided by the window's token rate (STAT_LO to STAT_HI), a figure that also spreads a request's time before its first token over its tokens. On B200 and H200, Gemma-3-27B streamed text for a median GEMMA_SHARE\\% of its output tokens and the rest carried no text, so its interval there is per token streamed. At targets above 256 the client received the tokens in bursts."
                      ).replace("STAT_MED", f"{agree['median']:.2f}").replace("STAT_LO", f"{agree['min']:.2f}").replace("STAT_HI", f"{agree['max']:.2f}").replace("GEMMA_SHARE", f"{100 * agree['gemma_streamed_share'][1]:.0f}")
    build.write_table("latency", pd.DataFrame(rows),
                      CAPTION,
                      "tab:latency", col_fmt="llrrrrrr", longtable=True)
    summ = {}
    for tb in sorted({int(k.rsplit("/", 1)[1]) for k in hl}):
        v = [x for k, x in hl.items() if k.endswith(f"/{tb}")]
        summ[tb] = {"n": len(v), "ttft_median_range_s": [min(x["ttft_median_s"] for x in v), max(x["ttft_median_s"] for x in v)],
                    "itl_range_ms": [1000 * min(x["itl_s"] for x in v), 1000 * max(x["itl_s"] for x in v)]}
    (build.TABLES / "latency_summary.json").write_text(json.dumps({"by_config": hl, "by_target": summ, "interval_agreement": agree}, indent=1))
    print(json.dumps(summ))
    print(json.dumps(agree))


if __name__ == "__main__":
    main()
