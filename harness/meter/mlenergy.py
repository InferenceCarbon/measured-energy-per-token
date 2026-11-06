"""Fetch the ML.ENERGY Benchmark v3 reference rows used by the pilot's
counter gate (Plan v3 §7: 8B energy per token within ±25% of ML.ENERGY at
matched batch and output).

The v3 results (H100 and B200, vLLM 0.11.1, BF16) live in the GATED Hugging
Face dataset `ml-energy/benchmark-v3`; the public ml.energy/leaderboard
repository only carries the v1 A100/A40/V100 CSVs. Access needs an HF
account that has accepted the dataset's conditions and a read token. The
fetch therefore runs ON THE POD (HF_TOKEN as a Runpod secret), never here.

Rows are queried through the datasets-server filter API, so nothing of the
98 GB dataset is downloaded beyond the matching rows. Every row returned is
stored with the results so the matching choice can be audited.

    python -m meter.mlenergy --model Qwen/Qwen3-8B --gpu H100 --out ref.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Any

import httpx

DATASET = "ml-energy/benchmark-v3"
API = "https://datasets-server.huggingface.co"


def _get(path: str, token: str, **params: Any) -> dict[str, Any]:
    """datasets-server can take a minute to wake a gated dataset's cache;
    allow it, and retry transient failures twice."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = httpx.get(API + path, params=params, headers={"Authorization": f"Bearer {token}"},
                          timeout=httpx.Timeout(240.0, connect=30.0))
            if r.status_code in (500, 502, 503, 504) and attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last = e
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"datasets-server {path} failed after retries: {last!r}")


def list_splits(token: str) -> list[dict[str, str]]:
    return _get("/splits", token, dataset=DATASET)["splits"]


def fetch_rows(token: str, model_id: str, gpu_substring: str = "H100",
               config: str | None = None, limit: int = 100,
               diagnostics: list[str] | None = None) -> list[dict[str, Any]]:
    """All rows for `model_id` on a GPU whose name contains `gpu_substring`.

    Tries the server-side filter first (where "model_id"='...'); if that
    yields nothing or errors, pages through /rows and filters client-side
    (the llm subset is ~1.1k rows, so a dozen pages). Every HTTP problem is
    appended to `diagnostics` so an empty result explains itself."""
    diag = diagnostics if diagnostics is not None else []
    splits = list_splits(token)
    cands = [s for s in splits if config is None or s["config"] == config]
    cands.sort(key=lambda s: (0 if "llm" in s["config"].lower() else 1))
    diag.append(f"splits: {[(s['config'], s['split']) for s in cands]}")
    rows: list[dict[str, Any]] = []

    def keep(row: dict[str, Any], s: dict[str, str]) -> None:
        if gpu_substring.lower() in str(row.get("gpu_model", row.get("gpu", ""))).lower():
            row["_config"], row["_split"] = s["config"], s["split"]
            rows.append(row)

    for s in cands:
        try:
            res = _get("/filter", token, dataset=DATASET, config=s["config"], split=s["split"],
                       where=f"\"model_id\"='{model_id}'", limit=limit)
            n = len(res.get("rows", []))
            diag.append(f"filter {s['config']}/{s['split']}: {n} rows")
            for item in res.get("rows", []):
                keep(item["row"], s)
        except httpx.HTTPStatusError as e:
            diag.append(f"filter {s['config']}/{s['split']}: HTTP {e.response.status_code} {e.response.text[:200]}")
        if rows:
            return rows
    # fallback: page /rows and match client-side
    for s in cands:
        offset, matched, total = 0, 0, None
        while True:
            try:
                res = _get("/rows", token, dataset=DATASET, config=s["config"], split=s["split"],
                           offset=offset, length=100)
            except httpx.HTTPStatusError as e:
                diag.append(f"rows {s['config']}/{s['split']} offset {offset}: HTTP {e.response.status_code} {e.response.text[:200]}")
                break
            items = res.get("rows", [])
            total = res.get("num_rows_total", total)
            for item in items:
                row = item["row"]
                if str(row.get("model_id", "")).lower() == model_id.lower():
                    matched += 1
                    keep(row, s)
            offset += len(items)
            if not items or (total is not None and offset >= total) or offset > 20000:
                break
        diag.append(f"rows {s['config']}/{s['split']}: scanned {offset} of {total}, model matched {matched}, kept {len(rows)}")
        if rows:
            break
    return rows


def pick_reference(rows: list[dict[str, Any]], target_batch: float,
                   task_preference: tuple[str, ...] = ("lm-arena-chat", "chat", "conversation")) -> dict[str, Any] | None:
    """Row with max_num_seqs nearest the achieved batch, preferring stable
    rows, single GPU, and the conversational task (short outputs, closest to
    our 256–600-token decode cells)."""
    def task_rank(r):
        t = str(r.get("task", "")).lower()
        for i, p in enumerate(task_preference):
            if p in t:
                return i
        return len(task_preference)

    def key(r):
        b = r.get("max_num_seqs") or 0
        return (task_rank(r), int(r.get("num_gpus", 1) or 1) != 1,
                not bool(r.get("is_stable", True)), abs(float(b) - target_batch))

    usable = [r for r in rows if r.get("energy_per_token_joules") is not None]
    return min(usable, key=key) if usable else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--target-batch", type=float, default=16)
    ap.add_argument("--out", type=Path, default=Path("mlenergy_reference.json"))
    a = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; cannot read the gated ML.ENERGY dataset", file=sys.stderr)
        return 2
    diag: list[str] = []
    rows = fetch_rows(token, a.model, a.gpu, diagnostics=diag)
    ref = pick_reference(rows, a.target_batch)
    out = {"dataset": DATASET, "model_id": a.model, "gpu": a.gpu, "n_rows": len(rows),
           "diagnostics": diag, "rows": rows, "reference": ref}
    a.out.write_text(json.dumps(out, indent=1, default=str))
    if ref is None:
        print(f"no usable rows for {a.model} on {a.gpu} ({len(rows)} rows)", file=sys.stderr)
        return 1
    print(json.dumps({k: ref.get(k) for k in ("model_id", "gpu_model", "task", "max_num_seqs", "num_gpus",
                                              "energy_per_token_joules", "output_throughput_tokens_per_sec",
                                              "avg_power_watts", "is_stable")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
