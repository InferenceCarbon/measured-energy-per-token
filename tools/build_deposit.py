"""Assemble the Zenodo deposit for this paper from the raw results store.

    python tools/build_deposit.py --store PATH_TO_RESULTS_STORE --out DEPOSIT_DIR [--dry-run]

The store holds one directory per results branch (campaign-h100, campaign-b200,
results-*, ...), each with cells/<run_id>/ (cell.json, per-GPU raw power samples,
window records) and branch-level files (counter qualification, nvidia-smi -q,
plan, manifest, server and campaign logs). The store also holds runs that are
not part of this paper (blind-test and calibration branches); only the 917 runs
in data/features.parquet, and the branch-level files of branches that hold at
least one of them, are copied.

Every copied text file is scanned for credentials (Hugging Face, GitHub and
bearer tokens). The script stops if it finds one, so nothing is deposited until
the file has been cleaned by hand.

The deposit layout:
    corpus/features.parquet, corpus/store_tables/cells.parquet
    raw/<branch>/cells/<run_id>/...      per-run records and raw power samples
    raw/<branch>/<branch-level files>
    code/  (git archive of this repository at the release tag; added by hand)
    SHA256SUMS
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BRANCH_FILES = ("counter_qualification.json", "manifest.json", "nvidia-smi-q.txt", "plan.json", "campaign.log", "logs")
SECRET = re.compile(r"(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|x-access-token:[^@\s]+@|Bearer\s+[A-Za-z0-9._-]{20,})")
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".yaml", ".yml", ".csv", ".md", ""}


def run_dirs(store: Path, features: pd.DataFrame) -> tuple[dict[str, Path], list[str]]:
    """Map each run_id to its directory, choosing the copy whose pod matches."""
    found, missing = {}, []
    for rid, pod in zip(features.run_id, features.pod_id):
        cands = [p for p in store.glob(f"*/cells/{rid}") if (p / "cell.json").exists()]
        pick = None
        for c in cands:
            try:
                if json.loads((c / "cell.json").read_text())["pod"]["pod_id"] == pod:
                    pick = c
            except (KeyError, ValueError):
                continue
        if pick is None:
            missing.append(rid)
        else:
            found[rid] = pick
    return found, missing


def scan(path: Path) -> list[str]:
    hits = []
    for f in [path] if path.is_file() else path.rglob("*"):
        if f.is_file() and f.suffix in TEXT_SUFFIXES and f.stat().st_size < 200_000_000:
            for n, line in enumerate(f.open(errors="replace"), 1):
                if SECRET.search(line):
                    hits.append(f"{f}:{n}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=ROOT / "data" / "features.parquet")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    feats = pd.read_parquet(a.features)
    found, missing = run_dirs(a.store, feats)
    branches = sorted({p.relative_to(a.store).parts[0] for p in found.values()})
    size = sum(f.stat().st_size for p in found.values() for f in p.rglob("*") if f.is_file())
    print(f"runs: {len(found)} found, {len(missing)} missing; branches: {', '.join(branches)}; run data {size / 1e6:.0f} MB")
    if missing:
        print("missing (first 10):", *missing[:10], sep="\n  ")
    if a.dry_run:
        return
    if missing:
        sys.exit("refusing to build a deposit with missing runs")

    out = a.out
    (out / "corpus" / "store_tables").mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.features, out / "corpus" / "features.parquet")
    shutil.copy2(a.features.parent / "store_tables" / "cells.parquet", out / "corpus" / "store_tables" / "cells.parquet")
    for rid, src in found.items():
        dst = out / "raw" / src.relative_to(a.store)
        shutil.copytree(src, dst)
    for b in branches:
        for name in BRANCH_FILES:
            src = a.store / b / name
            if src.exists():
                dst = out / "raw" / b / name
                (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)

    hits = scan(out)
    if hits:
        print("possible credentials found; clean these files before depositing:", *hits[:50], sep="\n  ")
        sys.exit(1)

    with open(out / "SHA256SUMS", "w") as fh:
        for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
            fh.write(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}\n")
    print(f"deposit written to {out}")


if __name__ == "__main__":
    main()
