"""PREVIEW ONLY, not the v1.1 build: what pooling does to the tables of version 1.0.1.

Builds a features table in the v1.0.0 layout from the frozen v1.1 corpus, restricted to the ten experiments of
version 1.0.1 (the four models on H100 and B200), so that the unchanged analysis/build.py can be run on it and its
tables compared with the published ones:

  - the v1.0.0 readings, with the 36 simulated-agent (c8) readings marked as superseded;
  - block A (47 readings, pooled with v1.0.0): added as further repeats of the same configuration;
  - block K and its warm-up reruns (32 readings): the simulated agent, re-measured.

Two shortcuts make this a preview. A replication repeat rN is renamed r(N+2) so that run ids stay unique in a
layout keyed on run id alone; the v1.1 build keys on (source, run_id) instead. The four c1 target-256 runs of
block A on H100 are left out here, because they ran at about twice the batch of the v1.0.0 runs with the same run
id (the load client's connection limit was removed in between): they are a separate operating point.

    python tools/pool_core_preview.py --freeze <meter>/data/freeze_v1_1 --out /tmp/preview_data
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE_CELLS = ("c0_static", "c1_decode_batch", "c2_decode_context", "c3_prefill_miss", "c4_cached_prefix", "c6_tensor_parallel",
              "c7_precision", "c8_agent_replay", "c9_kv_pressure", "c9b_kv_pressure_diag")
CORE_MODELS = ("qwen3-8b", "qwen3-30b-a3b", "deepseek-v2-lite", "llama-3.3-70b")
CORE_BRANCHES = ("repl-a-b200", "repl-a-h100", "repl-k-b200", "repl-k-h100", "repl-k-warm-b200", "repl-k-warm-h100")


def bump(run_id: str) -> str:
    m = re.search(r"/r(\d+)$", run_id)
    return run_id[:m.start()] + "/r%d" % (int(m.group(1)) + 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    both = pd.read_parquet(a.freeze / "features_v1_1.parquet")
    base = pd.read_parquet(ROOT / "data" / "features.parquet")
    cols = list(base.columns)

    v = both[both.source == "v1.0.0"].copy()
    sup = v.disposition_v1_1 == "superseded"
    v.loc[sup, "dropped"] = "superseded in v1.1: simulated-agent run with the scripted-miss fault, re-measured in the replication"

    r = both[(both.source == "replication") & both.branch.isin(CORE_BRANCHES) & both.cell.isin(CORE_CELLS) & both.model.isin(CORE_MODELS)].copy()
    capped = (r.cell == "c1_decode_batch") & (r.accelerator == "h100") & r.run_id.str.contains("target_batch-256")
    print("replication readings in the ten experiments:", len(r), "| by branch:", r.branch.value_counts().to_dict())
    print("left out of the preview as a separate operating point (c1 target 256, H100, block A):", int(capped.sum()))
    r = r[~capped]
    r["old_run_id"] = r.run_id
    r["run_id"] = r.run_id.map(bump)
    r["repeat"] = r.run_id.str.extract(r"/r(\d+)$")[0].astype(int).astype(base["repeat"].dtype)
    nw = r.disposition_v1_1 == "excluded-no-window"
    r.loc[nw, "dropped"] = r.loc[nw, "reason_v1_1"].fillna("no usable active window")
    feat = pd.concat([v[cols], r[cols]], ignore_index=True)
    assert not feat.run_id.duplicated().any()

    a.out.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(a.out / "features.parquet", index=False)

    # in-window cache reuse: the published file, plus the replication runs from the freeze's cache diagnostic
    sys.path.insert(0, str(a.freeze.resolve().parents[1]))
    from analysis import cache_diagnostic   # the meter repository's module, read-only
    diag = pd.DataFrame([x for br in CORE_BRANCHES for x in cache_diagnostic.rows_for_branch(a.freeze / "store", br)])
    diag["r"] = pd.to_numeric(diag["r"], errors="coerce")
    d = r[["branch", "old_run_id", "run_id"]].merge(diag.rename(columns={"run_id": "old_run_id"})[["branch", "old_run_id", "r", "context_cached_or_prefix", "fresh_in"]],
                                                    on=["branch", "old_run_id"], how="inner")
    cr = pd.read_csv(ROOT / "data" / "cache_reuse.csv")
    extra = pd.DataFrame({"run_id": d.run_id, "cache_reuse_r": d.r.round(4), "cached_tokens": d.context_cached_or_prefix, "fresh_in": d.fresh_in})
    extra = extra[extra.cache_reuse_r.notna()]
    pd.concat([cr, extra], ignore_index=True).to_csv(a.out / "cache_reuse.csv", index=False)

    # per-window scheduler facts
    st = pd.read_parquet(ROOT / "data" / "store_tables" / "cells.parquet")
    rc = pd.read_parquet(a.freeze / "tables" / "cells.parquet")
    rc = rc.merge(r[["branch", "old_run_id", "run_id"]].rename(columns={"run_id": "new_run_id", "old_run_id": "run_id"}), on=["branch", "run_id"], how="inner")
    rc["run_id"] = rc.pop("new_run_id")
    (a.out / "store_tables").mkdir(exist_ok=True)
    pd.concat([st, rc[[c for c in st.columns if c in rc.columns]]], ignore_index=True).to_parquet(a.out / "store_tables" / "cells.parquet", index=False)
    missing = [c for c in st.columns if c not in rc.columns]
    print("rows:", len(feat), "| v1.0.0", len(v), "(superseded c8:", int(sup.sum()), ") | replication added", len(r),
          "| cache-reuse rows added", len(extra), "| store columns absent from the replication cells:", missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
