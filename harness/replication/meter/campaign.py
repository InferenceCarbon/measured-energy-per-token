"""Iterate the run manifest on this pod; checkpoint; resume; push.

    python -m meter.campaign --mode pilot    --results /workspace/results --push
    python -m meter.campaign --mode campaign --results /workspace/results --push
    python -m meter.campaign --mode campaign --dry-run --accelerator h100 --n-gpus 1

pilot    counter qualification on every device, then ONE cell of each type on
         the smallest model with short windows, then the full manifest with GPU-hours.
         Any gate miss aborts with exit code 2. (Plan v3 §8.)
campaign the manifest for this accelerator and GPU count; 8B first, 70B
         last; repeat 1 of every cell before repeat 2; cells shuffled within
         each server group; `done.json` makes a restart skip finished cells.

Results are pushed to the private results repository after every cell when
--push is given and RESULTS_REMOTE (an https URL carrying its token) is set.
Pod launch and termination are not this module's job (Inferencecarbon-live).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from . import cachecheck, feasibility, power
from .cell import Protocol, git_commit, run_cell
from .power import make_source
from .server import ServerConfig, VLLMServer, dump_nvidia_smi_q
from .templates import CONFIG_DIR, Catalogue, load_catalogue

log = logging.getLogger("campaign")


# ---------------------------------------------------------------------------
# Pod facts
# ---------------------------------------------------------------------------

def detect_gpus() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             check=True, capture_output=True, text=True).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def accelerator_from_name(name: str, cat: Catalogue) -> str | None:
    n = name.upper()
    for key, acc in cat.accelerators.items():
        if (acc.get("runpod_gpu_type_id") or "").upper() == n:
            return key
    for key in cat.accelerators:
        if key.upper() in n:
            return key
    return None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _cells_spec(cells_path: Path = CONFIG_DIR / "cells.yaml") -> dict[str, Any]:
    return yaml.safe_load(cells_path.read_text())


# Replication block F: a cell's `speculation` vary key selects one of these vLLM --speculative-config
# settings ("none" = the plain server). The server key includes it, so on/off never share a server.
SPECULATION: dict[str, dict[str, Any]] = {
    "ngram3": {"method": "ngram", "num_speculative_tokens": 3, "prompt_lookup_max": 4},
}


def server_config_for(run: dict[str, Any], cat: Catalogue, backend: str | None = None) -> ServerConfig:
    """`backend` None -> the accelerator's own (serving.yaml); "fake" swaps
    the launcher for meter.fakeserve while keeping the accelerator's shapes."""
    fixed = run.get("fixed", {})
    prefix_caching = bool(fixed.get("prefix_caching", True))
    gmu = run.get("variant", {}).get("gpu_memory_utilization", fixed.get("gpu_memory_utilization"))
    acc_backend = feasibility.backend_of(cat, run["accelerator"])
    backend = backend or acc_backend
    # vLLM's cap on concurrent sequences. 256 for every campaign so far; a cell that studies deeper batches
    # sets `fixed: {max_num_seqs: N}` (config/replication_cells_high_batch.yaml), which is a new server config
    max_num_seqs = int(fixed.get("max_num_seqs", 256))
    max_model_len = cat.model(run["model"]).native_context
    neuron_profile = None
    if acc_backend == "neuron":
        nz = cat.serving["neuron"]
        # compiled shapes: the server's batch cap is the largest decode bucket, its context the served cap
        max_num_seqs = max(int(x) for x in nz["num_seqs_buckets"])
        max_model_len = min(max_model_len, int(nz["max_model_len_cap"]))
        if run.get("neuron_max_model_len"):
            max_model_len = min(max_model_len, int(run["neuron_max_model_len"]))
        neuron_profile = neuron_profile_of(run, cat)
        if neuron_profile:
            prof = nz["profiles"][neuron_profile]
            if prof.get("num_seqs_buckets"):
                max_num_seqs = max(int(x) for x in prof["num_seqs_buckets"])
            if prof.get("max_model_len"):
                max_model_len = min(max_model_len, int(prof["max_model_len"]))
        if not prefix_caching and nz.get("fold_prefix_caching_off", False):
            prefix_caching = True
            run.setdefault("notes_neuron", []).append(
                "prefix_caching=false folded to true (a toggle recompiles on Neuron); prompts are all unique, hit counters recorded")
        if run["kv_precision"] == "fp8" and run["weight_precision"] != "fp8":
            run.setdefault("notes_neuron", []).append("fp8 KV on bf16 weights needs checkpoint scales; recorded as served")
    kv_precision = run["kv_precision"]
    accelerator = None
    if acc_backend == "tpu":
        tz = cat.serving["tpu"]
        acc_cfg = cat.accelerators[run["accelerator"]]
        accelerator = run["accelerator"]
        max_num_seqs = int(tz["max_num_seqs"])
        max_model_len = min(max_model_len, int(acc_cfg.get("max_model_len_cap", max_model_len)))
        kv_ok = acc_cfg.get("kv_precisions")
        if kv_precision == "fp8" and kv_ok is not None and "fp8" not in kv_ok:
            kv_precision = "bf16"
            run.setdefault("notes_tpu", []).append(
                f"kv_precision fp8 folded to bf16: the quantized-KV kernel is untested on {accelerator}")
        if not prefix_caching and tz.get("fold_prefix_caching_off", False):
            prefix_caching = True
            run.setdefault("notes_tpu", []).append("prefix_caching=false folded to true (serving.yaml tpu.fold_prefix_caching_off)")
    spec_key = run.get("variant", {}).get("speculation")
    if spec_key not in (None, "none") and spec_key not in SPECULATION:
        raise ValueError(f"unknown speculation setting {spec_key!r}; known: {sorted(SPECULATION)}")
    return ServerConfig(model_key=run["model"], tp=int(run["tp"]),
                        weight_precision=run["weight_precision"], kv_precision=kv_precision,
                        prefix_caching=prefix_caching, max_num_seqs=max_num_seqs,
                        max_model_len=max_model_len,
                        router_hook=bool(run.get("router_pass", False)) and backend == "vllm_cuda",
                        gpu_memory_utilization=float(gmu) if gmu is not None else None,
                        speculative_config=SPECULATION.get(spec_key) if spec_key not in (None, "none") else None,
                        backend=backend, accelerator=accelerator, neuron_profile=neuron_profile)


def neuron_profile_of(run: dict[str, Any], cat: Catalogue) -> str | None:
    """The run's server profile (variant key `neuron_profile`); "default" or absent = the standard server."""
    name = run.get("variant", {}).get("neuron_profile")
    if name in (None, "default"):
        return None
    profiles = cat.serving.get("neuron", {}).get("profiles", {})
    if name not in profiles:
        raise ValueError(f"unknown neuron_profile {name!r}; serving.yaml neuron.profiles has {sorted(profiles)}")
    return name


def router_pass_plan(runs: list[dict[str, Any]], cat: Catalogue) -> list[dict[str, Any]]:
    """Separate pass (Plan v3 §3): decode-by-batch and decode-by-context cells
    on the MoE calibration models, repeat 1 only, served with the router hook
    on. Energy is recorded too, so overhead = hooked minus the clean run of
    the same cell. Run ids get a `router/` prefix so they never collide
    with fitting data."""
    out = []
    for r in runs:
        m = cat.model(r["model"])
        if m.is_moe and r["cell"] in ("c1_decode_batch", "c2_decode_context") and r["repeat"] == 1:
            x = json.loads(json.dumps(r))
            x["router_pass"] = True
            x["run_id"] = "router/" + x["run_id"]
            x["server_key"] = server_config_for(x, cat).key()
            out.append(x)
    out.sort(key=lambda r: (r["server_key"], r["cell"], str(r["variant"])))
    return out


def shard_plan(ordered: list[dict[str, Any]], shard: int, n_shards: int) -> list[dict[str, Any]]:
    """Split a campaign plan across `n_shards` pods so that the two repeats of
    any cell run on DIFFERENT pods (the second repeat is meant to sample a
    different time in a pod's life; a different pod is a stronger version
    of that). Server groups are kept whole within a shard to limit model
    loads. Assignment: for each (model, repeat) block, server groups are
    dealt round-robin starting at an offset that advances with the repeat,
    so a group's repeat-1 and repeat-2 shards differ when n_shards >= 2."""
    if n_shards <= 1:
        return ordered
    import zlib
    out: list[dict[str, Any]] = []
    # deterministic per server group (independent of shuffle order), offset by repeat
    for r in ordered:
        base = zlib.crc32(r["server_key"].encode()) % n_shards
        if (base + (r["repeat"] - 1)) % n_shards == shard:
            out.append(r)
    return out


def repeats_separated(plans: list[list[dict[str, Any]]]) -> bool:
    """True if no shard holds more than one repeat of the same cell."""
    for p in plans:
        bases = [r["run_id"].rsplit("/r", 1)[0] for r in p]
        if len(bases) != len(set(bases)):
            return False
    return True


_VARIANT_KEYS = ("context_cached", "target_batch", "weight_precision", "fresh_in", "prefix_cached", "control",
                 "gpu_memory_utilization", "context_cap", "clock_mode", "speculation", "neuron_profile")
_VARIANT_RE = re.compile("(" + "|".join(_VARIANT_KEYS) + r")-([^_/]+)")


def _run_from_id(base_id: str, cat: Catalogue, spec: dict[str, Any], accelerator: str, n_gpus: int,
                 min_gpus: int) -> dict[str, Any] | None:
    """Rebuild a manifest row from a run id the current feasibility rules no longer produce
    (replication block A). Returns None when the id is for another accelerator or needs more GPUs."""
    acc, model, tp_s, cell_id, vid = base_id.split("/")
    if acc != accelerator:
        return None
    tp = int(tp_s[2:])
    if not (min_gpus <= tp <= n_gpus):
        return None
    variant: dict[str, Any] = {}
    for k, v in _VARIANT_RE.findall(vid):
        variant[k] = float(v) if k == "gpu_memory_utilization" else (int(v) if v.isdigit() else v)
    cell = next(c for c in spec["cells"] if c["id"] == cell_id)
    m = cat.model(model)
    precision = variant.get("weight_precision", "fp8")
    kv = "fp8" if "fp8" in m.kv_precisions else m.kv_precisions[0]
    run = feasibility.Run(f"{base_id}/r1", model, acc, tp, tp, precision, kv, cell_id, cell["phase"], variant, 1,
                          feasibility._est_seconds(cell, variant, spec["defaults"]),
                          "reconstructed from the corpus run id (configuration no longer generated by the current rules)")
    return dataclasses.asdict(run)


def neuron_short_context(r: dict[str, Any], cat: Catalogue) -> None:
    """Move a deep, short-context decode run to the short max_model_len server
    (serving.yaml neuron.short_context) and mark its run id. The Neuron plugin
    caps concurrency at KV capacity / max_model_len, whatever the prompts."""
    sc = cat.serving.get("neuron", {}).get("short_context")
    if not sc or r["cell"] not in sc.get("cells", []) or "neuron_profile" in r.get("variant", {}):
        return                              # an explicit profile chooses its own server
    v, f = r.get("variant", {}), r.get("fixed", {})
    tb = v.get("target_batch", f.get("target_batch"))
    if tb is None or int(tb) < int(sc["min_target_batch"]):
        return
    need = int(v.get("context_cached", f.get("context_cached", 0))) + int(f.get("fresh_in", 0)) + int(f.get("out", 0))
    L = int(sc["max_model_len"])
    if need * 1.04 + 256 > L:              # same safety margin as the native-max context points
        return
    r["neuron_max_model_len"] = L
    parts = r["run_id"].split("/")
    parts[-2] = f"{parts[-2]}_max_model_len-{L}"
    r["run_id"] = "/".join(parts)


def plan(cat: Catalogue, accelerator: str, n_gpus: int, mode: str,
         models: list[str] | None = None, include_separate_pass: bool = False,
         seed: int = 20261022, min_gpus: int = 1, cells: list[str] | None = None,
         backend: str | None = None, order: str = "model-major",
         cells_path: Path | None = None, run_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Ordered list of manifest rows (with the cell's `fixed` block and a
    `server_key` merged in) for this pod.

    order  model-major  (default) every repeat of the smallest model, then the next model
           repeat-major repeat 1 of every model first, then repeat 2: for a host with a hard
                        end one full pass of the grid lands before any repeat"""
    spec = _cells_spec(cells_path) if cells_path else _cells_spec()
    by_id = {c["id"]: c for c in spec["cells"]}
    man = (feasibility.build_manifest(cat, cells_path=cells_path, accelerators=(accelerator,)) if cells_path
           else feasibility.build_manifest(cat, accelerators=(accelerator,)))
    runs = [r for r in man["runs"] if min_gpus <= r["n_gpus"] <= n_gpus]
    if models:
        runs = [r for r in runs if r["model"] in models]
    if cells:
        runs = [r for r in runs if r["cell"] in cells]
    if run_ids:
        # replication block A: named configurations from the corpus, re-run once each on a new pod. The
        # new pod's done.json is empty, so nothing is skipped. A configuration the current rules no longer
        # generate (the native-max margin was tightened during the campaign) is reconstructed from its id.
        wanted = [rid.rsplit("/r", 1)[0] for rid in run_ids]
        have = {r["run_id"].rsplit("/r", 1)[0]: r for r in runs if r["repeat"] == 1}
        runs = [have[w] for w in wanted if w in have]
        runs += [_run_from_id(w, cat, spec, accelerator, n_gpus, min_gpus) for w in wanted if w not in have]
        runs = [r for r in runs if r is not None]
    if not include_separate_pass:
        runs = [r for r in runs if not by_id[r["cell"]].get("separate_pass")]
    neuron_acc = feasibility.backend_of(cat, accelerator) == "neuron"
    for r in runs:
        r["fixed"] = dict(by_id[r["cell"]].get("fixed", {}))
        if neuron_acc:
            neuron_short_context(r, cat)
        r["server_key"] = server_config_for(r, cat, backend).key()

    if mode == "pilot":
        spec_picks = {c["id"]: c for c in spec["cells"] if c.get("pilot")}
        if spec_picks:
            return spec_pilot_plan(runs, spec_picks)
        return pilot_plan(runs, n_gpus, cat, backend)
    if mode == "router":
        return router_pass_plan(runs, cat)

    # campaign ordering: model size ascending; repeat 1 before repeat 2;
    # within a repeat, server groups kept together, cells shuffled per repeat.
    size = {m.key: m.n_params_total_b for m in cat.models.values()}
    ordered: list[dict[str, Any]] = []
    model_keys = sorted({r["model"] for r in runs}, key=lambda k: size[k])
    reps = sorted({r["repeat"] for r in runs})
    outer, inner = ((reps, model_keys) if order == "repeat-major" else (model_keys, reps))
    for a in outer:
        for b in inner:
            model_key, rep = (b, a) if order == "repeat-major" else (a, b)
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in runs:
                if r["model"] == model_key and r["repeat"] == rep:
                    groups[r["server_key"]].append(r)
            rng = random.Random(f"{seed}:{model_key}:{rep}")
            keys = sorted(groups)
            rng.shuffle(keys)
            for k in keys:
                g = groups[k]
                rng.shuffle(g)
                ordered.extend(g)
    return ordered


PILOT_PICKS: dict[str, dict[str, Any]] = {
    # one variant per cell type, chosen to exercise each code path cheaply
    "c0_static":        {"clock_mode": "default"},
    "c1_decode_batch":  {"target_batch": 16},
    "c2_decode_context": {"context_cached": 4096, "target_batch": 8},
    "c3_prefill_miss":  {"fresh_in": 4096, "target_batch": 8},
    "c4_cached_prefix": {"prefix_cached": 1024, "fresh_in": 200, "target_batch": 8},   # both controls
    "c6_tensor_parallel": {"target_batch": 16},                                        # needs 2 GPUs
    "c8_agent_replay":  {"target_batch": 1},
}


def spec_pilot_plan(runs: list[dict[str, Any]], cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Pilot for a cells file whose cells name their own pilot points: a cell's
    `pilot: {variant subset}` (and optional `pilot_models: [...]`) selects repeat-1 runs; the built-in
    PILOT_PICKS key on the original c0-c8 ids and would select nothing from such a file."""
    picks = []
    for r in runs:
        c = cells.get(r["cell"])
        if not c or r["repeat"] != 1:
            continue
        if c.get("pilot_models") and r["model"] not in c["pilot_models"]:
            continue
        if all(r["variant"].get(k) == v for k, v in c["pilot"].items()):
            picks.append(r)
    picks.sort(key=lambda r: (r["server_key"], r["cell"]))
    return picks


def pilot_plan(runs: list[dict[str, Any]], n_gpus: int, cat: Catalogue | None = None,
               backend: str | None = None) -> list[dict[str, Any]]:
    """One cell of each type on the smallest calibration model in `runs`
    (qwen3-8b on the GPU campaign; llama-3.1-8b on trn2), at that model's
    base tensor-parallel degree; the TP cell at the next degree up."""
    if not runs:
        return []
    size = {m.key: m.n_params_total_b for m in cat.models.values()} if cat else {}
    small = min({r["model"] for r in runs}, key=lambda k: (size.get(k, 0), k))
    precision = "fp8" if any(r["model"] == small and r["weight_precision"] == "fp8" for r in runs) \
        else next(r["weight_precision"] for r in runs if r["model"] == small)
    base_tp = min(r["tp"] for r in runs if r["model"] == small)
    picks: list[dict[str, Any]] = []
    for cell_id, want in PILOT_PICKS.items():
        cands = [r for r in runs if r["model"] == small and r["cell"] == cell_id
                 and r["repeat"] == 1 and r["weight_precision"] == precision
                 and all(r["variant"].get(k) == v for k, v in want.items())]
        if cell_id == "c6_tensor_parallel":
            higher = sorted({r["tp"] for r in cands if r["tp"] > base_tp and r["n_gpus"] <= n_gpus})
            cands = [r for r in cands if higher and r["tp"] == higher[0]]
        else:
            cands = [r for r in cands if r["tp"] == base_tp]
        picks.extend(cands)
    for r in picks:
        if r["cell"] == "c8_agent_replay":
            r["fixed"] = {**r["fixed"], "steps": 6}   # short script in the pilot
    neuron_like = bool(runs) and cat is not None and feasibility.backend_of(cat, runs[0]["accelerator"]) in ("neuron", "tpu")
    base = next((r for r in picks if r["cell"] == "c1_decode_batch"), None)
    if base is not None and not neuron_like and small == "qwen3-8b":
        x = json.loads(json.dumps(base))
        x["weight_precision"], x["kv_precision"] = "bf16", "bf16"
        x["cell"] = "c1_decode_batch"
        x["variant"] = {**x["variant"], "weight_precision": "bf16", "out": XCHECK_OUT_TOKENS,
                        "target_batch": XCHECK_BATCH}
        x["run_id"] = (f"{x['accelerator']}/{x['model']}/tp{x['tp']}/xcheck_mlenergy/"
                       f"bf16_target_batch-{XCHECK_BATCH}_out-{XCHECK_OUT_TOKENS}/r1")
        x["note"] = "ML.ENERGY cross-check; not fitting data"
        x["server_key"] = ServerConfig(model_key=x["model"], tp=int(x["tp"]), weight_precision="bf16",
                                       kv_precision="bf16", prefix_caching=True, max_num_seqs=256,
                                       max_model_len=x["variant"].get("context_cap") or 32768).key()
        picks.append(x)
    # One router-hook cell on the MoE model: does the plugin attach to this
    # vLLM, and what does logging cost? (c1, B16, qwen3-30b-a3b, FP8.)
    moe = [r for r in runs if r["model"] == "qwen3-30b-a3b" and r["cell"] == "c1_decode_batch"
           and r["repeat"] == 1 and r["weight_precision"] == "fp8" and r["tp"] == 1
           and r["variant"].get("target_batch") == 16 and not neuron_like]
    if moe:
        x = json.loads(json.dumps(moe[0]))
        x["router_pass"] = True
        x["run_id"] = "router/" + x["run_id"]
        x["server_key"] = ServerConfig(model_key=x["model"], tp=1, weight_precision="fp8", kv_precision=x["kv_precision"],
                                       prefix_caching=True, max_num_seqs=256, max_model_len=32768,
                                       router_hook=True).key()
        picks.append(x)
    # same server grouping as the campaign so the pilot measures load count too;
    # the small model's servers first, the MoE hook cell last (its load is the slowest)
    picks.sort(key=lambda r: (r["model"] != small, r["server_key"], r["cell"]))
    return picks


XCHECK_BATCH = 16          # on ML.ENERGY's max_num_seqs grid
XCHECK_OUT_TOKENS = 600    # near their Text Conversation mean output (~630 tokens)


_BMAX_FLAG = re.compile(r"^achieved batch [\d.]+ vs target \d+$")


def _is_bmax_observation(info: dict[str, Any]) -> bool:
    """A finished, not-ok cell whose only flags say the target batch was not reachable, or that the
    prefix cache was evicted inside the window (meter/cachecheck.py). Both are observations of what the
    server can hold at this target, not faults: a rerun on the same deployment gives the same answer."""
    flags = info.get("flags") or []
    return (bool(flags) and all(_BMAX_FLAG.match(str(f)) or cachecheck.EVICTION_FLAG.match(str(f)) for f in flags)
            and info.get("energy_j_active") is not None)


# ---------------------------------------------------------------------------
# Results, checkpoint, push
# ---------------------------------------------------------------------------

class Results:
    def __init__(self, root: Path, push: bool, remote: str | None, branch: str):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.done_path = root / "done.json"
        self.done: dict[str, Any] = json.loads(self.done_path.read_text()) if self.done_path.exists() else {}
        self.push_enabled = push and bool(remote)
        self.remote = remote
        self.branch = branch
        if push and not remote:
            log.warning("--push given but RESULTS_REMOTE is not set; results stay on the pod")
        if self.push_enabled:
            self._git_init()
            self._merge_remote_done()

    def _merge_remote_done(self) -> None:
        """Shards of one campaign share a results branch; a pod that restarts
        (or a replacement pod) must skip cells any pod has finished. Fetch
        the branch's done.json and merge it before planning."""
        try:
            self._git("fetch", "-q", "origin", self.branch, check=False)
            show = self._git("show", f"origin/{self.branch}:done.json", check=False)
            if show.returncode == 0 and show.stdout.strip():
                remote_done = json.loads(show.stdout)
                for k, v in remote_done.items():
                    if k not in self.done or (v.get("ok") and not self.done[k].get("ok")):
                        self.done[k] = v
                self.done_path.write_text(json.dumps(self.done, indent=1, default=str))
                log.info("merged %d finished cells from origin/%s", len(remote_done), self.branch)
        except Exception as e:  # pragma: no cover
            log.warning("could not merge remote done.json: %s", e)

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(self.root), *args], check=check,
                              capture_output=True, text=True)

    def _git_init(self):
        if not (self.root / ".git").exists():
            self._git("init", "-q", "-b", self.branch)
            self._git("config", "user.email", "meter@inferencecarbon.ai")
            self._git("config", "user.name", "inferencecarbon-meter")
        if "origin" not in self._git("remote", check=False).stdout.split():
            self._git("remote", "add", "origin", self.remote)
        else:
            self._git("remote", "set-url", "origin", self.remote)

    def mark(self, run_id: str, info: dict[str, Any]):
        self.done[run_id] = {**info, "finished_at": time.time()}
        self.done_path.write_text(json.dumps(self.done, indent=1, default=str))

    MAX_FILE_BYTES = 90_000_000      # GitHub rejects blobs over 100 MB; one such blob blocks every later push

    def _exclude_oversize(self) -> list[str]:
        """Keep oversize files out of the index (git/info/exclude) so one big
        log can never poison the history. Returns the paths excluded."""
        excluded = []
        info = self.root / ".git" / "info"
        info.mkdir(parents=True, exist_ok=True)
        exclude_file = info / "exclude"
        existing = exclude_file.read_text().splitlines() if exclude_file.exists() else []
        for p in self.root.rglob("*"):
            if p.is_file() and ".git" not in p.parts and p.stat().st_size > self.MAX_FILE_BYTES:
                rel = "/" + p.relative_to(self.root).as_posix()
                if rel not in existing:
                    existing.append(rel)
                    excluded.append(rel)
                    self._git("rm", "-q", "--cached", "--ignore-unmatch", rel.lstrip("/"), check=False)
        if excluded:
            exclude_file.write_text("\n".join(existing) + "\n")
            log.warning("excluded oversize files from results push: %s", excluded)
        return excluded

    def _snapshot_push(self, message: str) -> bool:
        """Fallback that cannot be blocked by history: a fresh parentless
        commit of the current tree, force-pushed to this pod's own branch
        results-<pod id>. analysis.load reads every results branch, so data
        on a fallback branch is used exactly like data on the shared one."""
        pod = os.environ.get("RUNPOD_POD_ID", "local")
        branch = f"results-{pod}"
        tree = self._git("write-tree", check=False)
        if tree.returncode != 0:
            log.warning("snapshot push: write-tree failed: %s", tree.stderr.strip()[:300])
            return False
        commit = self._git("commit-tree", tree.stdout.strip(), "-m", f"snapshot {message}", check=False)
        if commit.returncode != 0:
            log.warning("snapshot push: commit-tree failed: %s", commit.stderr.strip()[:300])
            return False
        r = self._git("push", "-q", "-f", "origin", f"{commit.stdout.strip()}:refs/heads/{branch}", check=False)
        if r.returncode == 0:
            log.warning("results pushed to fallback branch %s (shared branch %s rejected)", branch, self.branch)
            return True
        log.warning("snapshot push to %s failed: %s", branch, r.stderr.strip()[:300])
        return False

    def push(self, message: str):
        """Commit everything and push to the shared branch. Several pods push
        to the same branch, so on rejection we fetch, merge done.json by
        hand (union; ok wins), take ours for everything else (cell
        directories never collide: one run_id per pod), and retry. Pods start
        from separate `git init`s, so the merge must allow unrelated
        histories. If the shared branch still refuses, the current tree goes
        to a per-pod fallback branch so no measured cell is ever stranded on
        a pod that will terminate itself."""
        if not self.push_enabled:
            return
        try:
            self._git("add", "-A")
            self._exclude_oversize()
            if self._git("diff", "--cached", "--quiet", check=False).returncode != 0:
                self._git("commit", "-q", "-m", message)
            last_err = ""
            for attempt in range(3):
                r = self._git("push", "-q", "-u", "origin", self.branch, check=False)
                if r.returncode == 0:
                    return
                last_err = r.stderr.strip()[-300:]
                self._git("fetch", "-q", "origin", self.branch, check=False)
                show = self._git("show", f"origin/{self.branch}:done.json", check=False)
                theirs = json.loads(show.stdout) if show.returncode == 0 and show.stdout.strip() else {}
                for k, v in theirs.items():
                    if k not in self.done or (v.get("ok") and not self.done[k].get("ok")):
                        self.done[k] = v
                # merge their tree, keeping ours on conflict, then overwrite done.json with the union
                m = self._git("merge", "-q", "-X", "ours", "--no-edit", "--allow-unrelated-histories",
                              f"origin/{self.branch}", check=False)
                if m.returncode != 0:
                    log.warning("merge of origin/%s failed: %s", self.branch, m.stderr.strip()[-300:])
                    self._git("merge", "--abort", check=False)
                self.done_path.write_text(json.dumps(self.done, indent=1, default=str))
                self._git("add", "-A")
                self._git("commit", "-q", "-m", f"merge shards ({message})", check=False)
                time.sleep(5 + 7 * attempt)
            log.warning("push to %s failed after retries: %s", self.branch, last_err)
            self._snapshot_push(message)
        except subprocess.CalledProcessError as e:
            log.warning("push failed (will retry after next cell): %s", e.stderr.strip()[:300])


# ---------------------------------------------------------------------------
# Pilot gates
# ---------------------------------------------------------------------------

def load_gate_refs(path: Path = CONFIG_DIR / "gates.yaml") -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def pilot_gates(quals: list[dict[str, Any]], cell_results: list[dict[str, Any]],
                refs: dict[str, Any], accelerator: str) -> dict[str, Any]:
    gates: dict[str, Any] = {}

    bad = [q for q in quals if not q.get("energy_counter_available") and q.get("power_reading_update_s", 9) > 2.0
           and q.get("power_unit", "W") == "W" and q.get("power_source") != "tpu"]
    stale = [q for q in quals if (q.get("power_unit") == "percent_of_max" or (q.get("power_source") == "tpu" and q.get("power_unit") == "W"))
             and not (q.get("power_reading_update_s", float("inf")) < float("inf"))]
    gates["counter"] = {
        "pass": not bad and not stale and bool(quals),
        "detail": [{k: q.get(k) for k in ("gpu_index", "name", "driver", "energy_counter_available",
                                          "energy_counter_update_s", "power_reading_update_s",
                                          "power_source", "power_unit", "power_scale_w")} for q in quals],
    }
    if any(q.get("power_unit") == "percent_of_max" for q in quals):
        gates["counter"]["note"] = ("power is a percent of an unpublished maximum, converted with the declared "
                                    "power_scale_w; one record per minute (see counter_qualification.json notes)")
    if any(q.get("power_source") == "tpu" and q.get("power_unit") == "W" for q in quals):
        gates["counter"]["note"] = ("power is the AI Telemetry Collector's per-chip power_draw (BETA), held between readings; "
                                    "its update interval is recorded (see counter_qualification.json notes)")
    if any(q.get("power_unit") == "rated_declared" for q in quals):
        # no power_draw series on this device: the gate records that fact; it cannot fail on it
        gates["counter"]["note"] = ("NO power_draw telemetry on this backend: energy is the declared rated power_scale_w x time, "
                                    "a bound, not a measurement; the measured quantity is throughput (see power_source.json)")

    # Every pilot cell ran cleanly
    failed = [(r["run_id"], r["flags"]) for r in cell_results if r["flags"]]
    gates["cells_clean"] = {"pass": not failed, "failed": failed, "n_cells": len(cell_results)}

    # 8B energy per output token vs ML.ENERGY at matched batch/output (±25%).
    # Reference: the row fetched on the pod from the gated v3 dataset
    # (results/mlenergy_reference.json), else config/gates.yaml.
    fetched = refs.get("_fetched") or {}
    ref_row = fetched.get("reference") if isinstance(fetched, dict) else None
    ref = (ref_row or {}).get("energy_per_token_joules") or \
        (refs.get("mlenergy", {}).get(accelerator, {}) or {}).get("qwen3-8b_j_per_output_token_b16")
    xc = next((r for r in cell_results if r["run_id"].split("/")[3] == "xcheck_mlenergy"), None)
    if xc and xc.get("energy_j_active") and xc.get("phase_tokens"):
        measured = xc["energy_j_active"] / xc["phase_tokens"]
        g = {"measured_j_per_token": measured, "measured_batch": xc.get("achieved_batch"),
             "measured_cell": xc["run_id"], "reference_j_per_token": ref,
             "reference_row": ref_row, "reference_source": "dataset" if ref_row else ("gates.yaml" if ref else None)}
        if ref:
            rel = measured / ref - 1
            g.update({"pass": abs(rel) <= 0.25, "relative_error": rel})
        else:
            g.update({"pass": None, "note": "no reference: HF_TOKEN missing or dataset conditions not accepted, "
                                            "and config/gates.yaml has no figure"})
        gates["mlenergy_cross_check"] = g
    elif xc is None:
        gates["mlenergy_cross_check"] = {"pass": None, "note": "cross-check cell did not run"}
    else:
        gates["mlenergy_cross_check"] = {"pass": False, "note": "cross-check cell produced no energy/tokens"}

    hard_fail = [k for k, g in gates.items() if g.get("pass") is False]
    gates["overall"] = {"pass": not hard_fail, "hard_failures": hard_fail,
                        "not_evaluated": [k for k, g in gates.items() if g.get("pass") is None]}
    return gates


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

TPU_ACCELERATOR_ALIAS = "v6e"


def catalogue_for(accelerator: str, backend: str) -> Catalogue:
    """Templates for other backends live in their own files; merge them when this run uses one."""
    from .templates import TPU_MODEL_FILES, TRN2_MODEL_FILES
    base = load_catalogue()
    acc_backend = base.accelerators.get(accelerator, {}).get("backend") if accelerator not in ("auto", "tpu") else (
        "tpu" if accelerator == "tpu" else None)
    if backend == "tpu" or acc_backend == "tpu":
        return load_catalogue(extra_model_files=TPU_MODEL_FILES)
    if backend in ("neuron", "fake") or acc_backend == "neuron":
        return load_catalogue(extra_model_files=TRN2_MODEL_FILES)
    return base


def run(args: argparse.Namespace) -> int:
    cat = catalogue_for(args.accelerator, args.backend)
    results = Results(args.results, args.push, os.environ.get("RESULTS_REMOTE"),
                      branch=os.environ.get("RESULTS_BRANCH", f"results-{os.environ.get('RUNPOD_POD_ID', 'local')}"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.results / "campaign.log")])
    logging.getLogger("httpx").setLevel(logging.WARNING)   # one line per request otherwise

    # -- which hardware, which backend --------------------------------------
    acc = None if args.accelerator == "auto" else args.accelerator
    if acc == "tpu":
        log.info("--accelerator tpu read as %s (the Trillium slice; Ironwood is v7x)", TPU_ACCELERATOR_ALIAS)
        acc = TPU_ACCELERATOR_ALIAS
    acc_backend = feasibility.backend_of(cat, acc) if acc else None
    backend = args.backend if args.backend != "auto" else (acc_backend or "nvml")
    if backend == "vllm_cuda":
        backend = "nvml"
    if backend == "neuron":
        from . import neuron
        dev = neuron.detect_devices()
        names = [f"neuron{i}" for i in dev]
        acc = acc or "trn2"
    elif backend == "tpu":
        from . import tpu
        dev = tpu.detect_devices()
        names = [f"tpu{i}" for i in dev]
        acc = acc or TPU_ACCELERATOR_ALIAS
    elif backend == "fake":
        acc = acc or "trn2"
        n_fake = int(cat.accelerators[acc].get("chips_per_host", 1)) if args.n_gpus == "auto" else int(args.n_gpus)
        names = [f"FAKE-{acc}-{i}" for i in range(n_fake)]
    else:
        names = detect_gpus()
        if acc is None:
            if not names:
                log.error("no GPU detected and --accelerator not given")
                return 1
            acc = accelerator_from_name(names[0], cat)
            if acc is None:
                log.error("unrecognised GPU %r", names[0])
                return 1
    acc_backend = feasibility.backend_of(cat, acc)
    n_gpus = len(names) if args.n_gpus == "auto" else int(args.n_gpus)
    if n_gpus < 1 and not args.dry_run:
        log.error("no devices")
        return 1
    gpu_indices = list(range(max(n_gpus, 1)))
    commit = git_commit()
    log.info("accelerator=%s backend=%s n_devices=%d mode=%s commit=%s devices=%s", acc, backend, n_gpus, args.mode, commit, names)
    log.info("open-file limit (soft, hard): %s", getattr(args, "fd_limit", None))

    run_ids = [ln.strip() for ln in Path(args.run_ids_file).read_text().splitlines() if ln.strip() and not ln.startswith("#")] if args.run_ids_file else None
    ordered = plan(cat, acc, n_gpus, args.mode, args.models, args.include_separate_pass,
                   min_gpus=args.min_gpus, cells=args.cells,
                   backend=backend if backend == "fake" else None, order=args.order,
                   cells_path=args.cells_file, run_ids=run_ids)
    if args.only_router:
        ordered = [r for r in ordered if r.get("router_pass")]
    shard_i, shard_n = _parse_shard(args.shard)
    if shard_n > 1:
        if args.mode != "campaign":
            log.warning("--shard ignored for mode %s", args.mode)
        else:
            all_shards = [shard_plan(ordered, i, shard_n) for i in range(shard_n)]
            assert repeats_separated(all_shards), "shard assignment put two repeats on one pod"
            ordered = all_shards[shard_i]
            os.environ["ICM_SHARD"] = f"{shard_i}/{shard_n}"
            log.info("shard %d of %d: %d runs", shard_i, shard_n, len(ordered))
    spec = _cells_spec(args.cells_file) if args.cells_file else _cells_spec()
    short = args.mode in ("pilot", "router")
    overrides = (cat.serving.get(acc_backend, {}).get("protocol")
                 if (acc_backend == "neuron" or (acc_backend == "tpu" and not short)) else None)
    protocol = Protocol.from_defaults(spec["defaults"], short=short, overrides=overrides, scale=args.window_scale)
    if args.window_scale != 1.0:
        log.warning("window scale %.3f: windows are NOT protocol length (dry run only)", args.window_scale)
    manifest_full = (feasibility.build_manifest(cat, cells_path=Path(args.cells_file), accelerators=(acc,))
                     if args.cells_file else feasibility.build_manifest(cat, accelerators=(acc,)))
    if manifest_full.get("kv_pool_warnings"):
        log.warning("%d planned variants target more than the server's KV pool holds: their per-slot prefixes will be "
                    "evicted inside the window and the cells flagged (manifest.json kv_pool_warnings). A new campaign "
                    "should plan with `planner: {kv_sizing: pool_tokens}` in its cells file.",
                    len(manifest_full["kv_pool_warnings"]))
    (args.results / "manifest.json").write_text(json.dumps(manifest_full, indent=1))
    (args.results / "plan.json").write_text(json.dumps(
        {"accelerator": acc, "backend": backend, "n_gpus": n_gpus, "mode": args.mode, "order": args.order,
         "protocol": dataclasses.asdict(protocol), "window_scale": args.window_scale,
         "n_runs": len(ordered), "server_groups": len({r["server_key"] for r in ordered}),
         "runs": [r["run_id"] for r in ordered]}, indent=1))
    log.info("plan: %d runs in %d server groups; manifest summary %s",
             len(ordered), len({r["server_key"] for r in ordered}), manifest_full["summary"])

    if args.dry_run:
        for r in ordered:
            print(("DONE  " if r["run_id"] in results.done else "      ") + r["run_id"])
        return 0

    # -- power source + counter qualification on every device (pilot: a gate; campaign: recorded) --
    acc_cfg = cat.accelerators[acc]
    if backend == "neuron":
        from . import neuron
        neuron.dump_system_info(args.results / "neuron-ls.txt")
        source = make_source("neuron", acc_cfg, devices=gpu_indices,
                             record_log=args.results / "neuron_power_records.jsonl")
        rs = cat.serving["neuron"].get("raw_sampler", {})
        if not args.no_raw_sampler:
            from . import neuron_sampler
            st = neuron_sampler.ensure_running(Path(rs.get("path", "/opt/icm/raw/neuron_raw_samples.csv")),
                                               args.results / "neuron_sampler.log", float(rs.get("hz", 4.0)))
            (args.results / "neuron_sampler.json").write_text(json.dumps(st, indent=1))
            log.info("4 Hz raw sampler: %s", st)
            if not st["growing"]:
                log.error("4 Hz raw sampler is not writing; stop (or pass --no-raw-sampler to run on per-minute records only)")
                return 2
        # the file updates once a minute: measure that on device 0 only (two updates), just read the rest
        quals = [dataclasses.asdict(source.qualify(i, seconds=args.qualify_seconds if i == gpu_indices[0] else 1.0))
                 for i in gpu_indices]
    elif backend == "tpu":
        from . import tpu
        tz = cat.serving["tpu"]
        info = tpu.dump_system_info(args.results / "tpu-info.txt", tz.get("telemetry_url", tpu.TELEMETRY_URL))
        source = make_source("tpu", acc_cfg, devices=gpu_indices, serving=tz, slice_info=info,
                             record_log=args.results / "tpu_telemetry_records.jsonl")
        # the collector's update interval is measured on device 0 (the background scrape covers every device)
        quals = [dataclasses.asdict(source.qualify(i, seconds=float(tz.get("qualify_seconds", 60)) if i == gpu_indices[0] else 2.0))
                 for i in gpu_indices]
    elif backend == "fake" and acc_backend == "tpu":
        source = make_source("tpu", acc_cfg, devices=gpu_indices, fetch=lambda: "", background=False,
                             slice_info={"dry_run": True})
        quals = [dataclasses.asdict(source.qualify(i, seconds=0.0)) for i in gpu_indices]
    elif backend == "fake":
        source = make_source("fake", acc_cfg, pct_mode=(acc_backend == "neuron"),
                             scale_w=acc_cfg.get("power_scale_w", 500.0))
        quals = [dataclasses.asdict(source.qualify(i)) for i in gpu_indices]
    else:
        dump_nvidia_smi_q(args.results / "nvidia-smi-q.txt")
        source = make_source("nvml")
        quals = [dataclasses.asdict(source.qualify(i)) for i in gpu_indices]
    if acc_backend == "tpu" and not short:
        # the collector's series may update every few seconds or once a minute: measured above; coarse -> longer windows
        tz = cat.serving["tpu"]
        upd = quals[0].get("power_reading_update_s", float("inf"))
        if quals[0].get("power_unit") == "W" and upd > float(tz.get("coarse_update_threshold_s", 20)) and tz.get("protocol_coarse"):
            protocol = Protocol.from_defaults(spec["defaults"], short=short, overrides=tz["protocol_coarse"], scale=args.window_scale)
            log.warning("power_draw updates every ~%.0f s: coarse protocol in force (%s)", upd, tz["protocol_coarse"])
            pj = json.loads((args.results / "plan.json").read_text())
            pj["protocol"] = dataclasses.asdict(protocol)
            pj["protocol_note"] = f"coarse telemetry ({upd:.0f} s updates): serving.yaml tpu.protocol_coarse applied"
            (args.results / "plan.json").write_text(json.dumps(pj, indent=1))
    (args.results / "counter_qualification.json").write_text(json.dumps(quals, indent=1))
    (args.results / "power_source.json").write_text(json.dumps(source.describe(), indent=1, default=str))
    for q in quals:
        log.info("device %d %s driver %s energy_counter=%s power_update=%.2fs unit=%s %s", q["gpu_index"], q["name"],
                 q["driver"], q["energy_counter_available"], q["power_reading_update_s"], q.get("power_unit"), q["notes"])
    results.push(f"qualification {acc} {commit[:8]}")

    deadline = time.time() + args.hard_timeout_h * 3600
    cell_results: list[dict[str, Any]] = []
    current: VLLMServer | None = None
    tok = None
    tok_for: str | None = None
    failed_server_keys: dict[str, list[str]] = {}      # server config key -> flags; do not retry the same start per cell
    try:
        for r in ordered:
            prev = results.done.get(r["run_id"])
            if prev and prev.get("ok", False):
                log.info("skip (done) %s", r["run_id"])
                continue
            if (prev and args.skip_measured_since is not None and prev.get("energy_j_active") is not None
                    and float(prev.get("finished_at") or 0) >= args.skip_measured_since):
                # restart inside a time-boxed block: a cell measured by this campaign is kept even if flagged
                # (e.g. idle drift); earlier entries (a short-window pilot) are rerun
                log.info("skip (measured, flagged) %s", r["run_id"])
                continue
            if prev and _is_bmax_observation(prev) and not args.rerun_bmax:
                # the server could not reach the target batch: that is a measurement of B_max, not a
                # fault, and a rerun gives the same answer (restarts were repeating these at ~4 min each)
                log.info("skip (B_max or cache-eviction observation) %s", r["run_id"])
                continue
            if time.time() > deadline:
                log.error("hard timeout reached; stopping before %s", r["run_id"])
                break
            cfg = server_config_for(r, cat, backend if backend == "fake" else None)
            if cfg.key() in failed_server_keys:
                info = {"run_id": r["run_id"], "ok": False, "flags": failed_server_keys[cfg.key()], "wall_s": 0.0}
                cell_results.append(info)
                results.mark(r["run_id"], info)
                results.push(f"{args.mode} server-failed {r['run_id']}")
                log.error("skipping %s: server config already failed to start on this pod", r["run_id"])
                continue
            if current is None or current.cfg.key() != cfg.key():
                if current:
                    current.stop()
                    current = None
                current, start_flags = _start_server(cfg, cat, args.results)
                if current is None and r.get("router_pass"):
                    # Was it the hook or the model? Try once more without the hook.
                    clean = dataclasses.replace(cfg, router_hook=False)
                    log.warning("hooked server failed; retrying %s without the hook", clean.key())
                    current, more = _start_server(clean, cat, args.results)
                    start_flags += ["router hook server failed; cell ran WITHOUT the hook"] + more
                if current is None:
                    failed_server_keys[cfg.key()] = start_flags
                    info = {"run_id": r["run_id"], "ok": False, "flags": start_flags, "wall_s": 0.0}
                    cell_results.append(info)
                    results.mark(r["run_id"], info)
                    results.push(f"{args.mode} server-failed {r['run_id']}")
                    log.error("skipping %s: %s", r["run_id"], start_flags[:1])
                    continue
                if start_flags:
                    log.warning("server started with notes: %s", start_flags)
            if tok_for != r["model"]:
                if args.tokenizer == "word" or backend == "fake":
                    from .loadgen import WordTokenizer
                    tok = WordTokenizer()          # dry run: prompts sized in words, like the fake server counts them
                elif backend == "neuron":
                    from .loadgen import HFTokenizer
                    from .neuron import resolve_neuron_hf_id
                    tok = HFTokenizer(resolve_neuron_hf_id(cat.model(r["model"]), r["weight_precision"]))
                elif backend == "tpu":
                    from .loadgen import HFTokenizer
                    from .tpu import resolve_tpu_hf_id
                    tok = HFTokenizer(resolve_tpu_hf_id(cat.model(r["model"]), r["weight_precision"], cat.accelerators[acc]))
                else:
                    from .loadgen import HFTokenizer
                    from .server import resolve_hf_id
                    tok = HFTokenizer(resolve_hf_id(cat.model(r["model"]), r["weight_precision"])[0])
                tok_for = r["model"]
            log.info("cell %s", r["run_id"])
            t0 = time.time()
            try:
                part = gpu_indices[: r["n_gpus"]]
                res = run_cell(r, cat, current, part, args.results / "cells",
                               protocol, tok, spec, commit, source=source,
                               extra_indices=[i for i in gpu_indices if i not in part]
                               if backend != "nvml" and acc_backend != "tpu" else None)
                info = dataclasses.asdict(res)
            except Exception as e:  # a dead cell must not kill the campaign
                log.exception("cell failed: %s", r["run_id"])
                info = {"run_id": r["run_id"], "ok": False, "flags": [f"exception: {e}"]}
                if current and not current.alive():
                    log.error("server died; will restart for the next cell")
                    current = None
            info["wall_s"] = time.time() - t0
            if current is not None and not current.alive():
                # a cell whose requests all failed returns normally; the next cell must not reuse a dead
                # server (24-25 Oct: four 70B TP2 cells failed with ConnectError after one crash)
                log.error("server process is gone after %s; will restart for the next cell", r["run_id"])
                info.setdefault("flags", []).append("server died during or after this cell")
                info["ok"] = False
                current = None
            if r.get("router_pass") and current is not None:
                _harvest_router_records(current, cat, args.results, r, info)
            cell_results.append(info)
            results.mark(r["run_id"], info)
            log.info("cell %s ok=%s flags=%s energy=%s J tokens=%s batch=%s wall=%.0fs",
                     r["run_id"], info.get("ok"), info.get("flags"), info.get("energy_j_active"),
                     info.get("phase_tokens"), info.get("achieved_batch"), info["wall_s"])
            results.push(f"{args.mode} {r['run_id']}")
            if args.mode == "pilot" and not info.get("ok") and args.abort_on_flag:
                log.error("pilot cell flagged; aborting pilot")
                break
    finally:
        if current:
            current.stop()
        if hasattr(source, "close"):
            source.close()

    if args.mode == "pilot":
        refs = load_gate_refs()
        if os.environ.get("HF_TOKEN"):
            from . import mlenergy
            try:
                diag: list[str] = []
                rows = mlenergy.fetch_rows(os.environ["HF_TOKEN"], "Qwen/Qwen3-8B",
                                           "H100" if acc == "h100" else "B200", diagnostics=diag)
                ref_row = mlenergy.pick_reference(rows, XCHECK_BATCH)
                refs["_fetched"] = {"n_rows": len(rows), "diagnostics": diag, "rows": rows, "reference": ref_row}
                log.info("ML.ENERGY fetch diagnostics: %s", diag)
                (args.results / "mlenergy_reference.json").write_text(json.dumps(refs["_fetched"], indent=1, default=str))
                log.info("ML.ENERGY reference: %s", {k: (ref_row or {}).get(k) for k in
                         ("task", "max_num_seqs", "energy_per_token_joules", "gpu_model")})
            except Exception as e:
                log.warning("ML.ENERGY fetch failed: %s", e)
                refs["_fetched"] = {"error": str(e)}
        else:
            log.warning("HF_TOKEN not set: ML.ENERGY reference not fetched")
        planned = {r["run_id"] for r in ordered}
        all_cells = [dict(v, run_id=k) for k, v in results.done.items() if k in planned]
        gates = pilot_gates(quals, all_cells, refs, acc)
        report = {"accelerator": acc, "n_gpus": n_gpus, "commit": commit, "gates": gates,
                  "cells": all_cells, "cells_this_process": cell_results, "manifest_summary": manifest_full["summary"],
                  "manifest_dropped": len(manifest_full["dropped"])}
        (args.results / "pilot_report.json").write_text(json.dumps(report, indent=1, default=str))
        results.push("pilot report")
        log.info("PILOT %s: %s", "PASS" if gates["overall"]["pass"] else "FAIL", json.dumps(gates["overall"]))
        return 0 if gates["overall"]["pass"] else 2

    n_ok = sum(1 for v in results.done.values() if v.get("ok"))
    log.info("campaign: %d/%d runs ok", n_ok, len(ordered))
    return 0 if n_ok == len(ordered) else 3


def _slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]


_HOOK_OFFSETS: dict[str, int] = {}


def _start_server(cfg: ServerConfig, cat: Catalogue, results: Path) -> tuple[VLLMServer | None, list[str]]:
    """Start a server; on failure return None and the tail of its log as a
    flag, so a model or plugin that will not serve is a recorded result and
    the campaign moves on to the next server group."""
    log.info("starting server %s", cfg.key())
    hook_dir = str(results / "moe_hook" / _slug(cfg.key()))
    srv = VLLMServer(cfg, cat, results / "logs", env=cfg.env(cat, hook_dir, results_dir=str(results)))
    start_timeout = None
    if cfg.backend == "vllm_cuda" and os.environ.get("ICM_PREDOWNLOAD", "1") != "0":
        # the checkpoint first, timed and logged, so a slow or full host stops early with a reason (meter/weights.py);
        # with the weights local, the server start needs no download allowance
        from . import weights
        from .server import resolve_hf_id
        m = cat.model(cfg.model_key)
        hf_id = resolve_hf_id(m, cfg.weight_precision)[0]
        try:
            dl = weights.predownload(hf_id, os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
                                     float(os.environ.get("ICM_DOWNLOAD_MAX_S", "5400")), log=log.info)
        except weights.DownloadError as e:
            log.error("weights not ready for %s: %s", cfg.key(), e)
            return None, [f"server failed: weights not downloaded: {e}"]
        gb = m.n_params_total_b * cat.bytes_per_param.get(cfg.weight_precision, 2)
        start_timeout = 1800.0 + 5.0 * gb       # load and compile only: 405B fp8 about 64 min, 8B about 31 min
        log.info("weights ready (%.0f s); server start limit %.0f s", dl["seconds"], start_timeout)
    try:
        srv.start(timeout_s=start_timeout)
    except (RuntimeError, TimeoutError) as e:
        tail = ""
        try:
            lines = srv.log_path.read_text(errors="replace").splitlines()
            tail = " | ".join(l.strip() for l in lines[-25:] if l.strip())[-1500:]
        except OSError:
            pass
        srv.stop()
        log.error("server failed: %s\n%s", e, tail)
        return None, [f"server failed: {e}", f"vllm log tail: {tail}"]
    d = srv.describe()
    log.info("server ready in %.0fs (vLLM %s)", d["load_seconds"] or -1, d["vllm_version"])
    if d.get("neuron_startup_metrics"):
        log.info("neuron start-up metrics: %s", json.dumps(d["neuron_startup_metrics"])[:800])
    (results / "logs" / f"server_ready.{_slug(cfg.key())}.json").write_text(json.dumps(d, indent=1, default=str))
    return srv, []


def _harvest_router_records(server: VLLMServer, cat: Catalogue, results: Path,
                            run: dict[str, Any], info: dict[str, Any]) -> None:
    """After a router-pass cell: summarise the hook's JSONL (records written
    since the cell started) into <cell dir>/moe_union.json and rotate the
    files so the next cell starts clean."""
    from . import moe_hook
    hook_dir = Path(server.env.get("ICM_MOE_HOOK_DIR", "")) if server.env else None
    if not hook_dir or not hook_dir.exists():
        info.setdefault("flags", []).append("router pass: no hook directory (plugin inactive?)")
        return
    recs = moe_hook.read_records(hook_dir, _HOOK_OFFSETS)
    summary = moe_hook.summarise(recs)
    summary["n_records"] = len(recs)
    logs = [p.read_text() for p in hook_dir.glob("moe_hook.rank*.log")]
    summary["hook_log"] = logs
    cell_dir = results / "cells" / run["run_id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / "moe_union.json").write_text(json.dumps(summary, indent=1))
    with (cell_dir / "moe_hook_records.jsonl").open("w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")
    if not recs:
        info.setdefault("flags", []).append("router pass: hook produced no records")
    info["moe_union_steps"] = summary["n_steps"]


def _parse_shard(spec: str) -> tuple[int, int]:
    i, n = spec.split("/")
    i, n = int(i), int(n)
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"bad --shard {spec!r}")
    return i, n


def raise_fd_limit(want: int = 65536) -> tuple[int, int]:
    """Raise the open-file soft limit towards the hard limit. Each concurrent request is a socket in the
    load generator and another in the server it spawns (which inherits the limit); the usual soft limit
    of 1024 is below what a target batch of 512 or 1024 needs. Returns the (soft, hard) now in force."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = want if hard == resource.RLIM_INFINITY else min(want, hard)
    if soft == resource.RLIM_INFINITY or soft >= target:
        return soft, hard
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        return soft, hard
    return target, hard


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pilot", "campaign", "router"], required=True,
                    help="router = MoE expert-union logging pass (separate from energy cells)")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--n-gpus", default="auto")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--cells", nargs="*", default=None, help="run only these cell ids (mop-up / diagnostic pods)")
    ap.add_argument("--cells-file", type=Path, default=None,
                    help="cells matrix to plan from instead of config/cells.yaml (the replication campaign uses config/replication_cells.yaml)")
    ap.add_argument("--run-ids-file", type=Path, default=None,
                    help="re-run exactly these run ids (one per line) once each: replication block A, a third pod for single-pod configurations")
    ap.add_argument("--rerun-bmax", action="store_true",
                    help="do not skip cells whose earlier result was a 'target batch unreachable' or 'cache evicted in window' observation (replication block B, after the connection-cap fix)")
    ap.add_argument("--include-separate-pass", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="plan only; no GPU needed")
    ap.add_argument("--hard-timeout-h", type=float, default=20.0)
    ap.add_argument("--no-raw-sampler", action="store_true",
                    help="Neuron: do not start/require the 4 Hz utilization_raw sampler (per-minute records only)")
    ap.add_argument("--skip-measured-since", type=float, default=None, metavar="UNIX_TIME",
                    help="on resume, also skip cells measured (energy recorded) at or after this time, even if flagged; "
                         "default reruns every not-ok cell")
    ap.add_argument("--only-router", action="store_true", help="run only the router-hook cells of the plan")
    ap.add_argument("--min-gpus", type=int, default=1,
                    help="run only cells needing at least this many GPUs (the multi-GPU pod takes the TP cells)")
    ap.add_argument("--shard", default="0/1", help="i/n: this pod runs shard i of n (campaign mode); repeats of a cell never share a shard")
    ap.add_argument("--abort-on-flag", action="store_true", default=False,
                    help="pilot: stop at the first flagged cell (default: run every pilot cell and report all flags)")
    ap.add_argument("--backend", choices=["auto", "nvml", "neuron", "tpu", "fake"], default="auto",
                    help="power source + launcher: auto = the accelerator's own; tpu = vLLM TPU plugin and the rated-power "
                         "source (no telemetry); fake = meter.fakeserve and a synthetic power source on CPU (the dry run; "
                         "exercises the whole pipeline, measures nothing)")
    ap.add_argument("--tokenizer", choices=["hf", "word"], default="hf",
                    help="word = whitespace tokenizer (dry run); hf = the model's tokenizer")
    ap.add_argument("--window-scale", type=float, default=1.0,
                    help="multiply window lengths and token minimums (dry run only; recorded in plan.json)")
    ap.add_argument("--order", choices=["model-major", "repeat-major"], default="model-major",
                    help="repeat-major: one pass of every model's grid before any repeat")
    ap.add_argument("--qualify-seconds", type=float, default=130.0,
                    help="Neuron: how long to watch device 0's power file for updates (two per-minute updates)")
    args = ap.parse_args(argv)
    args.results.mkdir(parents=True, exist_ok=True)
    args.fd_limit = raise_fd_limit()      # before any server is spawned: it inherits the limit
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
