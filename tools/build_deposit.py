"""Assemble the Zenodo deposit for this paper from the raw results.

    python tools/build_deposit.py --meter-repo PATH_TO_METER_REPO --out DEPOSIT_DIR [--dry-run]
    python tools/build_deposit.py --store PATH_TO_RESULTS_STORE [--store ...] --out DEPOSIT_DIR [--dry-run]

Raw results live on one branch per campaign (campaign-h100, campaign-b200, results-*, repl-*, ...),
each with cells/<run_id>/ (cell.json, per-GPU raw power samples, window records, per-request records)
and branch-level files (counter qualification, nvidia-smi -q, plan, manifest, server and campaign logs).
They can be read from directories holding one sub-directory per branch (--store, repeatable) or
straight from the branches of the meter repository (--meter-repo, remote given by --remote). A
directory copy is used where it holds the branch; otherwise the git branch.

What is deposited:
  - every run in data/features.parquet (version 1.0 and the replication), matched to its raw copy by
    branch (replication) or by pod (version 1.0, whose runs were stored on several branches); run ids
    repeat between the two sources, so a run is keyed by its source and run id, not its run id alone;
  - every run listed in data/readings_left_out.csv that is not in the corpus file (copies, retried and
    superseded attempts, pilots), so that the raw record of every reading the paper mentions is present.
    Each branch's done.json holds one summary entry per run (energy, tokens, flags, finish time); a retry
    replaced the entry, and usually the run directory, of the attempt before it. raw/<branch>/done_history.jsonl
    therefore keeps every entry done.json ever held, read from the branch's git history, so every recorded
    attempt has its summary. Where an earlier attempt's own run directory survives in the history it is
    copied to raw/<branch>/earlier_attempts/<run_id>/<commit>/. A listed reading with no run directory of
    its own (overwritten by its retry, or its server failed before any window ran) is listed in
    raw/READINGS_WITHOUT_RUN_DIRECTORY.csv with where its record is. The script stops only if a corpus
    run is missing;
  - the branch-level files of every branch that supplied one of those runs;
  - the corpus and its companion files: features.parquet, store_tables/cells.parquet, cache_reuse.csv,
    server_settings.csv, readings_left_out.csv, detection/*.json, README.md and LICENSE.

Every copied text file is scanned for credentials (Hugging Face, GitHub and bearer tokens). The script
stops if it finds one, so nothing is deposited until the file has been cleaned by hand.

The deposit layout:
    corpus/...                           the corpus and companion files listed above
    raw/<branch>/cells/<run_id>/...      per-run records and raw power samples
    raw/<branch>/<branch-level files>
    code/  (git archive of this repository at the release tag; added by hand)
    SHA256SUMS
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BRANCH_FILES = ("counter_qualification.json", "manifest.json", "nvidia-smi-q.txt", "plan.json", "campaign.log", "done.json", "logs")
CORPUS_FILES = ("features.parquet", "store_tables/cells.parquet", "cache_reuse.csv", "server_settings.csv",
                "readings_left_out.csv", "detection/precision_probe.json", "detection/speculation_detection.json",
                "README.md", "LICENSE")
SECRET = re.compile(r"(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|x-access-token:[^@\s]+@|Bearer\s+[A-Za-z0-9._-]{20,})")
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".yaml", ".yml", ".csv", ".md", ""}


class DirBranch:
    def __init__(self, path: Path):
        self.path, self.name = path, path.name

    def has_run(self, rid: str) -> bool:
        return (self.path / "cells" / rid / "cell.json").exists()

    def cell_json(self, rid: str) -> dict:
        return json.loads((self.path / "cells" / rid / "cell.json").read_text())

    def copy(self, rel: str, dst: Path) -> bool:
        src = self.path / rel
        if not src.exists():
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        return True


class GitBranch:
    def __init__(self, repo: Path, ref: str):
        self.repo, self.ref, self.name = repo, ref, ref.split("/", 1)[1]
        out = self._git("ls-tree", "-r", "--name-only", ref).stdout.decode()
        self.files = set(out.split("\n")) - {""}
        self.runs = {f[6:-10] for f in self.files if f.startswith("cells/") and f.endswith("/cell.json")}

    def _git(self, *a: str, inp: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.repo), *a], input=inp, capture_output=True, check=True)

    def has_run(self, rid: str) -> bool:
        return rid in self.runs

    def cell_json(self, rid: str) -> dict:
        return json.loads(self._git("show", f"{self.ref}:cells/{rid}/cell.json").stdout)

    def done_history(self) -> list[dict]:
        """Every entry done.json ever held on this branch, oldest first: one row per distinct attempt."""
        rows, seen = [], set()
        log = self._git("log", "--reverse", "--format=%H %cI", self.ref, "--", "done.json").stdout.decode().splitlines()
        for line in log:
            sha, date = line.split()
            try:
                done = json.loads(self._git("show", f"{sha}:done.json").stdout)
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                continue
            for rid, e in done.items():
                k = (rid, e.get("finished_at"))
                if k not in seen:
                    seen.add(k)
                    rows.append(dict(commit=sha, commit_date=date, run_id=rid, entry=e))
        return rows

    def earlier_run_dirs(self, rid: str) -> list[str]:
        """Commits holding an earlier version of the run's directory (its cell.json differs from the tip's)."""
        path = f"cells/{rid}/cell.json"
        tip = self._git("rev-parse", f"{self.ref}:{path}").stdout.decode().strip()
        out, seen = [], {tip}
        for sha in self._git("log", "--format=%H", self.ref, "--", path).stdout.decode().split():
            try:
                blob = self._git("rev-parse", f"{sha}:{path}").stdout.decode().strip()
            except subprocess.CalledProcessError:
                continue
            if blob not in seen:
                seen.add(blob)
                out.append(sha)
        return out

    def copy_at(self, sha: str, rel: str, dst: Path) -> None:
        tar = self._git("archive", "--format=tar", sha, rel).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            for m in t.getmembers():
                if m.isfile():
                    target = dst / Path(m.name).relative_to(rel)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(t.extractfile(m).read())

    def copy(self, rel: str, dst: Path) -> bool:
        if rel not in self.files and not any(f.startswith(rel + "/") for f in self.files):
            return False
        tar = self._git("archive", "--format=tar", self.ref, rel).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as t:
            for m in t.getmembers():
                if m.isfile():
                    target = dst / Path(m.name).relative_to(rel) if Path(m.name) != Path(rel) else dst
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(t.extractfile(m).read())
        return True


def branches(stores: list[Path], meter_repo: Path | None, remote: str) -> dict[str, object]:
    out: dict[str, object] = {}
    if meter_repo:
        refs = subprocess.run(["git", "-C", str(meter_repo), "branch", "-r"], capture_output=True, text=True, check=True).stdout.split()
        for r in refs:
            if r.startswith(remote + "/") and "->" not in r:
                out[r.split("/", 1)[1]] = GitBranch(meter_repo, r)
    for s in stores:                       # a directory copy takes precedence over the git branch
        for p in s.iterdir():
            if (p / "cells").is_dir():
                out[p.name] = DirBranch(p)
    return out


def targets(feats: pd.DataFrame, left: pd.DataFrame) -> list[dict]:
    t = [dict(key=r.uid, run_id=r.run_id, branch=(r.branch if r.source == "replication" else None), pod=r.pod_id, why="corpus")
         for r in feats.itertuples()]
    extra = left[left.in_features_file.astype(str).str.lower() == "no"]
    t += [dict(key=f"{r.source}:{r.branch}:{r.run_id}:{i}", run_id=r.run_id, branch=(r.branch if isinstance(r.branch, str) else None), pod=None,
               why="left out", disposition=r.disposition, reason=r.reason, source=r.source)
          for i, r in enumerate(extra.itertuples())]
    return t


def locate(t: dict, br: dict[str, object]) -> object | None:
    if t["branch"]:
        b = br.get(t["branch"])
        return b if b is not None and b.has_run(t["run_id"]) else None
    cands = [b for name, b in sorted(br.items()) if not name.startswith("repl-") and b.has_run(t["run_id"])]
    if t["pod"] is None:
        return cands[0] if cands else None
    for b in cands:
        try:
            if b.cell_json(t["run_id"])["pod"]["pod_id"] == t["pod"]:
                return b
        except (KeyError, ValueError, subprocess.CalledProcessError):
            continue
    return None


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
    ap.add_argument("--store", type=Path, action="append", default=[])
    ap.add_argument("--meter-repo", type=Path)
    ap.add_argument("--remote", default="live")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=DATA / "features.parquet")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.store and not a.meter_repo:
        sys.exit("give --store and/or --meter-repo")

    feats = pd.read_parquet(a.features)
    left = pd.read_csv(DATA / "readings_left_out.csv")
    br = branches(a.store, a.meter_repo, a.remote)
    found, missing, no_dir, earlier = {}, [], [], []
    for t in targets(feats, left):
        b = locate(t, br)
        retried = t.get("disposition") == "retried"      # the directory at the tip belongs to the attempt that replaced it
        if b is not None and not retried:
            found[t["key"]] = (b, t)
        elif t["why"] == "left out" and t["branch"] in br:
            shas = br[t["branch"]].earlier_run_dirs(t["run_id"]) if retried and b is not None and isinstance(b, GitBranch) else []
            (earlier.append((t, shas)) if shas else no_dir.append(t))
        else:
            missing.append(t)
    used = sorted({b.name for b, _ in found.values()})
    n_corpus = sum(1 for _, t in found.values() if t["why"] == "corpus")
    used = sorted(set(used) | {t["branch"] for t in no_dir} | {t["branch"] for t, _ in earlier})
    print(f"run directories found: {len(found)} ({n_corpus} corpus, {len(found) - n_corpus} left out and not in the corpus); "
          f"earlier attempts with a directory in the branch history: {len(earlier)}; "
          f"left-out readings with no directory of their own: {len(no_dir)}; missing: {len(missing)}")
    print(f"branches deposited ({len(used)}): {', '.join(used)}")
    for t in missing[:20]:
        print(f"  missing [{t['why']}] {t['key']}")
    if a.dry_run:
        return
    if missing:
        sys.exit("refusing to build a deposit with missing runs")

    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    for rel in CORPUS_FILES:
        dst = out / "corpus" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((a.features if rel == "features.parquet" else DATA / rel), dst)
    for key, (b, t) in found.items():
        dst = out / "raw" / b.name / "cells" / t["run_id"]
        if not dst.exists():                 # a byte-identical copy listed twice is written once
            b.copy(f"cells/{t['run_id']}", dst)
    for name in used:
        for f in BRANCH_FILES:
            br[name].copy(f, out / "raw" / name / f)
    hist = {}
    for name in used:
        if isinstance(br[name], GitBranch):
            hist[name] = br[name].done_history()
            with open(out / "raw" / name / "done_history.jsonl", "w") as fh:
                for row in hist[name]:
                    fh.write(json.dumps(row) + "\n")
    for t, shas in earlier:
        for sha in shas:
            br[t["branch"]].copy_at(sha, f"cells/{t['run_id']}", out / "raw" / t["branch"] / "earlier_attempts" / t["run_id"] / sha[:10])
    if no_dir:
        rows = []
        for t in no_dir:
            n = sum(1 for h in hist.get(t["branch"], []) if h["run_id"] == t["run_id"])
            rec = (f"raw/{t['branch']}/done_history.jsonl ({n} entries for this run id)" if n else "") + \
                  ("; " if n else "") + f"raw/{t['branch']}/campaign.log and raw/{t['branch']}/logs/"
            rows.append(dict(source=t["source"], branch=t["branch"], run_id=t["run_id"], disposition=t["disposition"], reason=t["reason"], record=rec))
        pd.DataFrame(rows).to_csv(out / "raw" / "READINGS_WITHOUT_RUN_DIRECTORY.csv", index=False)

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
