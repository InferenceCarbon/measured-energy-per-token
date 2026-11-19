"""Precision probe (experiment c11; paper Section "Precision probe").

Question: does the output distribution of a served model reveal its weight format? The probe serves one fixed prompt
set to fp8 and nvfp4 weights of the same model (B200) and to fp8 on H100, keeping the top-K log-probabilities of every
output token (logprobs.parquet in the raw records of the deposit).

Two readings:

  paired      fp8 against nvfp4 on the same prompt and model: top-1 agreement per position, text agreement per
              request, mean |delta logprob| of the fp8-chosen token. The H100-fp8 against B200-fp8 pair is the
              control: same format, different accelerator and kernels.

  single      features of one reply alone: mean top-1 margin, mean top-K entropy, the distinct-value ratio of the
              chosen-token logprobs and the fraction of exactly-zero logprobs. The detector is a per-feature
              threshold scored leave-one-model-out, as in detect_speculation.py.

    python analysis/detect_precision.py --store RAW_STORE --branches repl-w5-b200 repl-w5-h100 \\
        --out data/detection/precision_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_speculation import best_threshold  # noqa: E402


def single_features(token_logprobs: list[float | None], top_logprobs: list[dict[str, float]]) -> dict[str, float] | None:
    lps = [x for x in token_logprobs if x is not None]
    if not lps:
        return None
    margins, entropies = [], []
    for top in top_logprobs:
        vals = sorted((v for v in top.values() if v is not None), reverse=True)
        if len(vals) >= 2:
            margins.append(vals[0] - vals[1])
        if vals:
            z = sum(math.exp(v) for v in vals)
            ps = [math.exp(v) / z for v in vals]
            entropies.append(-sum(p * math.log(p) for p in ps if p > 0))
    q = {round(x, 4) for x in lps}
    return {
        "n_tokens": float(len(lps)),
        "mean_logprob": statistics.fmean(lps),
        "top1_margin_mean": statistics.fmean(margins) if margins else 0.0,
        "topk_entropy_mean": statistics.fmean(entropies) if entropies else 0.0,
        "distinct_value_ratio": len(q) / len(lps),
        "zero_fraction": sum(1 for x in lps if x == 0.0) / len(lps),
    }


def paired_features(a_tokens: list[str], a_lps: list[float | None], b_tokens: list[str],
                    b_tops: list[dict[str, float]]) -> dict[str, float]:
    """A against B on one prompt: A's chosen tokens looked up in B's top-K."""
    n = min(len(a_tokens), len(b_tokens))
    agree = sum(1 for i in range(n) if a_tokens[i] == b_tokens[i])
    deltas = []
    for i in range(n):
        if a_lps[i] is None:
            continue
        bv = b_tops[i].get(a_tokens[i]) if i < len(b_tops) else None
        if bv is not None:
            deltas.append(abs(a_lps[i] - bv))
    first_div = next((i for i in range(n) if a_tokens[i] != b_tokens[i]), n)
    return {"positions": float(n), "top1_agreement": agree / n if n else 0.0,
            "text_identical": 1.0 if (n and agree == n and len(a_tokens) == len(b_tokens)) else 0.0,
            "first_divergence": float(first_div),
            "abs_delta_logprob_mean": statistics.fmean(deltas) if deltas else float("nan")}


_PREC = re.compile(r"weight_precision-([a-z0-9]+)")


def precision_from_run_id(run_id: str) -> str:
    """`.../w5_precision_probe/weight_precision-nvfp4/r1` -> `nvfp4` (the key itself has an underscore,
    so the variant segment cannot be split on `_`)."""
    m = _PREC.search(run_id)
    return m.group(1) if m else "?"


def load_store(store: Path, branches: list[str], cell: str = "w5_precision_probe") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for br in branches:
        for p in sorted((store / br).glob(f"cells/*/*/*/{cell}/*/r*/logprobs.parquet")):
            rel = p.parent.relative_to(store / br / "cells")
            parts = str(rel).split("/")
            prec = precision_from_run_id(str(rel))
            df = pq.read_table(p).to_pandas()
            for _, r in df.iterrows():
                if r.get("error") is not None and not pd.isna(r.get("error")):
                    continue
                rows.append({"run_id": str(rel), "accelerator": parts[0], "model": parts[1], "precision": prec,
                             "request_index": int(r["request_index"]), "text": r["text"],
                             "tokens": json.loads(r["tokens_json"]),
                             "token_logprobs": json.loads(r["token_logprobs_json"]),
                             "top_logprobs": json.loads(r["top_logprobs_json"])})
    return pd.DataFrame(rows)


def single_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in df.iterrows():
        f = single_features(r["token_logprobs"], r["top_logprobs"])
        if f:
            f.update({k: r[k] for k in ("accelerator", "model", "precision", "request_index")})
            out.append(f)
    return pd.DataFrame(out)


def paired_table(df: pd.DataFrame, a: tuple[str, str], b: tuple[str, str]) -> pd.DataFrame:
    """a, b = (accelerator, precision); pairs rows by model and request_index."""
    da = df[(df["accelerator"] == a[0]) & (df["precision"] == a[1])].set_index(["model", "request_index"])
    db = df[(df["accelerator"] == b[0]) & (df["precision"] == b[1])].set_index(["model", "request_index"])
    out = []
    for key in da.index.intersection(db.index):
        ra, rb = da.loc[key], db.loc[key]
        f = paired_features(ra["tokens"], ra["token_logprobs"], rb["tokens"], rb["top_logprobs"])
        f.update({"model": key[0], "request_index": key[1], "pair": f"{a[0]}-{a[1]} vs {b[0]}-{b[1]}"})
        out.append(f)
    return pd.DataFrame(out)


def detector(single: pd.DataFrame, feature: str) -> dict[str, Any]:
    d = single[single["accelerator"] == "b200"]
    d = d[d["precision"].isin(["fp8", "nvfp4"])]
    if d.empty:
        return {"feature": feature, "n": 0}
    y = (d["precision"] == "nvfp4").tolist()
    t_all, acc_all = best_threshold(d[feature].tolist(), y)
    per_model: dict[str, Any] = {}
    for m in sorted(d["model"].unique()):
        train, test = d[d["model"] != m], d[d["model"] == m]
        if train.empty or test.empty:
            continue
        t, _ = best_threshold(train[feature].tolist(), (train["precision"] == "nvfp4").tolist())
        per_model[m] = {"threshold": t, "accuracy": float(((test[feature] > t) == (test["precision"] == "nvfp4")).mean()),
                        "n": int(len(test))}
    loo = [v["accuracy"] for v in per_model.values()]
    return {"feature": feature, "n": int(len(d)), "threshold_all": t_all, "accuracy_in_sample": acc_all,
            "leave_one_model_out": per_model, "accuracy_loo_mean": (sum(loo) / len(loo)) if loo else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=Path("data/store"))
    ap.add_argument("--branches", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    df = load_store(a.store, a.branches)
    if df.empty:
        print("no w5_precision_probe logprobs found")
        return 1
    single = single_table(df)
    feats = ["top1_margin_mean", "topk_entropy_mean", "distinct_value_ratio", "zero_fraction", "mean_logprob"]
    by = single.groupby(["accelerator", "model", "precision"])[feats].median().reset_index()
    pairs = {"fp8_vs_nvfp4_b200": paired_table(df, ("b200", "fp8"), ("b200", "nvfp4")),
             "fp8_h100_vs_fp8_b200": paired_table(df, ("h100", "fp8"), ("b200", "fp8"))}
    pair_summary = {k: (v.groupby("model")[["top1_agreement", "text_identical", "first_divergence",
                                            "abs_delta_logprob_mean"]].mean().to_dict(orient="index") if not v.empty else {})
                    for k, v in pairs.items()}
    det = {f: detector(single, f) for f in feats}
    print(by.to_string(index=False))
    print(json.dumps(pair_summary, indent=1, default=str))
    for f, r in det.items():
        print(f"{f}: in-sample {r.get('accuracy_in_sample')}, leave-one-model-out mean {r.get('accuracy_loo_mean')}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"single_by_group": by.to_dict(orient="records"), "paired": pair_summary,
                                     "detectors": det}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
