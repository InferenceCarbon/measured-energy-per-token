"""Detection of speculative decoding from the token stream (experiment c10; paper Section "Speculative decoding").

Question: do token-arrival statistics on a streaming completion reveal speculative decoding? With a proposer (here
vLLM's n-gram proposer, three speculative tokens) the server emits the accepted draft tokens together, so a stream
event carries more than one token; without speculation one token arrives per scheduler step.

Data: requests.parquet of every speculation run in the raw records of the deposit (events_json = [[t, chars,
tokens_est], ...] per request). Each request becomes one feature row; the label is read from the run id
(`speculation-ngram3` / `speculation-none`).

    python analysis/detect_speculation.py --store RAW_STORE --branches repl-bcf-h100 repl-bcf-b200 \\
        --out data/detection/speculation_detection.json

The detector is a single threshold on tokens_per_event (completion tokens / stream events), chosen on the labelled
rows and scored leave-one-model-out, so the number reported is what a new model would get. Every feature is computed
from one request alone.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

_SPEC = re.compile(r"speculation-([a-z0-9]+)")
_BATCH = re.compile(r"target_batch-(\d+)")


def request_features(events: list[list[float]], completion_tokens: int | None) -> dict[str, float] | None:
    """Per-request features from the stream events of ONE request (no server knowledge)."""
    if not events or not completion_tokens:
        return None
    ts = [e[0] for e in events]
    chars = [e[1] for e in events]
    n = len(events)
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    med_gap = statistics.median(gaps) if gaps else 0.0
    mean_gap = statistics.fmean(gaps) if gaps else 0.0
    sd_gap = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
    return {
        "n_events": float(n),
        "tokens_per_event": completion_tokens / n,
        "chars_per_event_mean": statistics.fmean(chars),
        "chars_per_event_max": float(max(chars)),
        "gap_median_s": med_gap,
        "gap_cv": (sd_gap / mean_gap) if mean_gap > 0 else 0.0,
        # bursts: events that arrive far faster than the typical gap (several tokens released at once)
        "burst_fraction": (sum(1 for g in gaps if g < 0.25 * med_gap) / len(gaps)) if gaps and med_gap > 0 else 0.0,
    }


def rows_from_requests(path: Path, run_id: str) -> list[dict[str, Any]]:
    df = pq.read_table(path).to_pandas()
    spec = _SPEC.search(run_id)
    batch = _BATCH.search(run_id)
    parts = run_id.split("/")
    out = []
    for _, r in df.iterrows():
        if r.get("error") is not None and not pd.isna(r.get("error")):
            continue
        if "in_window" in df.columns and not bool(r.get("in_window", True)):
            continue
        try:
            events = json.loads(r["events_json"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        ct = r.get("completion_tokens")
        f = request_features(events, int(ct) if ct is not None and not pd.isna(ct) else None)
        if f is None:
            continue
        f.update({"run_id": run_id, "accelerator": parts[0], "model": parts[1],
                  "speculation": spec.group(1) if spec else "unknown",
                  "target_batch": int(batch.group(1)) if batch else None,
                  "request_index": int(r["request_index"])})
        out.append(f)
    return out


def load_store(store: Path, branches: list[str], cell: str = "f_speculation") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for br in branches:
        for p in sorted((store / br).glob(f"cells/*/*/*/{cell}/*/r*/requests.parquet")):
            rel = p.parent.relative_to(store / br / "cells")
            rows.extend(rows_from_requests(p, str(rel)))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["accelerator", "model", "target_batch", "speculation"]
    g = df.groupby(keys)
    return g.agg(n_requests=("tokens_per_event", "size"),
                 tokens_per_event=("tokens_per_event", "median"),
                 chars_per_event=("chars_per_event_mean", "median"),
                 gap_cv=("gap_cv", "median"),
                 burst_fraction=("burst_fraction", "median")).reset_index()


def best_threshold(x: list[float], y: list[bool]) -> tuple[float, float]:
    """Threshold t on x maximising accuracy of (x > t) == y; returns (t, accuracy)."""
    pairs = sorted(zip(x, y))
    xs = [p[0] for p in pairs]
    cands = [xs[0] - 1e-9] + [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [xs[-1] + 1e-9]
    best = (cands[0], 0.0)
    for t in cands:
        acc = sum(1 for xv, yv in pairs if (xv > t) == yv) / len(pairs)
        if acc > best[1]:
            best = (t, acc)
    return best


def detector(df: pd.DataFrame, feature: str = "tokens_per_event") -> dict[str, Any]:
    """Leave-one-model-out: threshold chosen on the other models, scored on the held-out one."""
    d = df[df["speculation"].isin(["none", "ngram3"])]
    if d.empty:
        return {"feature": feature, "n": 0}
    y_all = (d["speculation"] != "none").tolist()
    t_all, acc_all = best_threshold(d[feature].tolist(), y_all)
    per_model: dict[str, Any] = {}
    for m in sorted(d["model"].unique()):
        train, test = d[d["model"] != m], d[d["model"] == m]
        if train.empty or test.empty:
            continue
        t, _ = best_threshold(train[feature].tolist(), (train["speculation"] != "none").tolist())
        acc = float(((test[feature] > t) == (test["speculation"] != "none")).mean())
        spec, called = (test["speculation"] != "none"), (test[feature] > t)
        per_model[m] = {"threshold": t, "accuracy": acc, "n": int(len(test)),
                        # the held-out model's own baseline and the two kinds of error
                        "n_plain": int((~spec).sum()), "n_speculative": int(spec.sum()), "always_plain_accuracy": float((~spec).mean()),
                        "plain_called_speculative_n": int((called & ~spec).sum()), "speculative_found_n": int((called & spec).sum()),
                        "plain_called_speculative": float((called & ~spec).sum() / max((~spec).sum(), 1)),
                        "speculative_found": float((called & spec).sum() / max(spec.sum(), 1))}
    loo = [v["accuracy"] for v in per_model.values()]
    return {"feature": feature, "n": int(len(d)), "threshold_all": t_all, "accuracy_in_sample": acc_all,
            "leave_one_model_out": per_model,
            "accuracy_loo_mean": (sum(loo) / len(loo)) if loo else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=Path("data/store"))
    ap.add_argument("--branches", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    df = load_store(a.store, a.branches)
    if df.empty:
        print("no f_speculation requests found")
        return 1
    summ = summarise(df)
    det = {f: detector(df, f) for f in ("tokens_per_event", "burst_fraction", "gap_cv")}
    print(summ.to_string(index=False))
    for f, r in det.items():
        print(f"{f}: in-sample {r.get('accuracy_in_sample')}, leave-one-model-out mean {r.get('accuracy_loo_mean')}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summary": summ.to_dict(orient="records"), "detectors": det}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
