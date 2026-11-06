"""Run ONE cell end to end and write its records.

Sequence (Plan v3 §3):
    clocks set (c0, c5) -> idle 60 s -> cache warm + warm-up (EXCLUDED)
    -> single-phase window >= min_window_s AND >= min_tokens -> idle 60 s
    -> drift check -> clocks reset

Outputs in <out_dir>/<run_id>/:
    cell.json          everything about the run: manifest row, server,
                       protocol, clocks, warm results, gate flags, commit
    windows.parquet    one row per metered window (idle_before, active,
                       idle_after) with per-GPU energy as JSON and the
                       achieved batch from scheduler samples
    requests.parquet   one row per request with raw stream events
    raw/               power samples per window per GPU (JSON lines)

Measured energy is stored as measured. No per-request allocation is made
here; analysis may allocate later and must label it as allocation.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from . import loadgen
from .power import Meter, WindowEnergy, window_summary
from .server import SchedulerSnapshot, VLLMServer, lock_sm_clocks, reset_sm_clocks
from .templates import Catalogue


@dataclass
class Protocol:
    min_window_s: float = 60.0
    min_tokens: int = 20000
    idle_window_s: float = 60.0
    idle_drift_tolerance: float = 0.05
    warmup_requests: int = 4
    temperature: float = 0.0
    seed: int = 20261022
    metrics_poll_s: float = 0.5
    max_window_s: float = 1800.0
    prefix_groups: str = "per_slot"     # each concurrent slot has its own prefix (private KV)

    @classmethod
    def from_defaults(cls, d: dict[str, Any], short: bool = False) -> "Protocol":
        p = cls(min_window_s=d["min_window_s"], min_tokens=d["min_tokens"],
                idle_window_s=d["idle_window_s"], idle_drift_tolerance=d["idle_drift_tolerance"],
                warmup_requests=d["warmup_requests"], temperature=d["temperature"], seed=d["seed"])
        if short:   # pilot: exercise every path, do not pay for full windows
            p.min_window_s, p.min_tokens, p.idle_window_s = 20.0, 2000, 20.0
            p.max_window_s = 300.0
        return p


@dataclass
class CellResult:
    run_id: str
    ok: bool
    flags: list[str]
    out_dir: str
    energy_j_active: float | None
    duration_s_active: float | None
    phase_tokens: int | None
    achieved_batch: float | None
    target_batch: int | None
    idle_before_w: float | None
    idle_after_w: float | None
    warm_confirmed: bool | None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def pod_identity() -> dict[str, Any]:
    """Where this measurement was made. Results from several pods and data
    centres are pooled, so every row carries it."""
    import os
    return {
        "pod_id": os.environ.get("RUNPOD_POD_ID", "local"),
        "data_center": os.environ.get("RUNPOD_DC_ID", os.environ.get("ICM_DC", "unknown")),
        "gpu_type": os.environ.get("RUNPOD_GPU_NAME", "unknown"),
        "shard": os.environ.get("ICM_SHARD", ""),
    }


def git_commit(repo: Path | None = None) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True, cwd=repo or Path(__file__).resolve().parent.parent
                              ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _idle_window(meter: Meter, seconds: float, window_id: str) -> list[WindowEnergy]:
    meter.start()
    time.sleep(seconds)
    return meter.stop(window_id)


class MetricsSampler:
    """Polls the server's scheduler metrics during the active window."""

    def __init__(self, server: VLLMServer, period_s: float):
        self.server = server
        self.period_s = period_s
        self.samples: list[SchedulerSnapshot] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(await asyncio.to_thread(self.server.metrics))
            except Exception as e:  # metrics endpoint hiccup: record, continue
                self.samples.append(SchedulerSnapshot(time.perf_counter(), float("nan"),
                                                      float("nan"), None, None, None, None, None, None))
            try:
                await asyncio.wait_for(self._stop.wait(), self.period_s)
            except asyncio.TimeoutError:
                pass

    def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    def dump(self, path: Path) -> None:
        """Persist the raw scheduler samples (JSON lines) so batch statistics
        can be recomputed under a different definition later (25 Oct: the
        obscal d1-d8 batch figures could not be, because only the summary
        was kept)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for s in self.samples:
                fh.write(json.dumps(asdict(s)) + "\n")

    def achieved_batch(self, t_start: float, t_end: float, trim: float = 0.10) -> dict[str, Any]:
        """Batch statistics over the steady part of the window (first and
        last `trim` of the time excluded). BATCH SEMANTICS (Astra, 25 Oct):
          running_mean   wall-time mean of num_requests_running, idle samples
                         included (what a window-average sees);
          active_batch   mean of num_requests_running over samples with
                         running > 0 -- the batch the forward model's q_W
                         sees while work is being done;
          idle_fraction  fraction of samples with running == 0; idle energy
                         enters through the duration model, never through the
                         batch.
        Also the deltas of the server's cumulative counters over the window,
        including scheduler preemptions where the server exposes them."""
        span = t_end - t_start
        lo, hi = t_start + trim * span, t_end - trim * span
        inside = [s for s in self.samples if lo <= s.t <= hi and s.running == s.running]
        running = [s.running for s in inside]
        active = [r for r in running if r > 0]
        waiting = [s.waiting for s in inside if s.waiting == s.waiting]
        allsamp = [s for s in self.samples if s.t >= t_start and s.t <= t_end]
        first = next((s for s in allsamp if s.prompt_tokens is not None), None)
        last = next((s for s in reversed(allsamp) if s.prompt_tokens is not None), None)
        pre = [s.preemptions for s in allsamp if s.preemptions is not None]
        return {
            "n_samples": len(inside),
            "running_mean": statistics.fmean(running) if running else None,
            "running_median": statistics.median(running) if running else None,
            "running_min": min(running) if running else None,
            "running_max": max(running) if running else None,
            "active_batch": statistics.fmean(active) if active else None,
            "idle_fraction": (1.0 - len(active) / len(running)) if running else None,
            "waiting_mean": statistics.fmean(waiting) if waiting else None,
            "preemptions_delta": (pre[-1] - pre[0]) if len(pre) >= 2 else None,
            "preemptions_counter_present": bool(pre),
            "kv_usage_mean": statistics.fmean([s.kv_usage for s in inside if s.kv_usage is not None])
            if any(s.kv_usage is not None for s in inside) else None,
            "server_prompt_tokens_delta": (last.prompt_tokens - first.prompt_tokens)
            if first and last and first.prompt_tokens is not None and last.prompt_tokens is not None else None,
            "server_generation_tokens_delta": (last.generation_tokens - first.generation_tokens)
            if first and last and first.generation_tokens is not None and last.generation_tokens is not None else None,
            "prefix_hits_delta": (last.prefix_hits - first.prefix_hits)
            if first and last and first.prefix_hits is not None and last.prefix_hits is not None else None,
            "prefix_queries_delta": (last.prefix_queries - first.prefix_queries)
            if first and last and first.prefix_queries is not None and last.prefix_queries is not None else None,
        }


# ---------------------------------------------------------------------------
# prompt factories per cell
# ---------------------------------------------------------------------------

def prompt_factory(run: dict[str, Any], corpus: loadgen.Corpus, concurrency: int,
                   per_slot: bool) -> tuple[Callable[[int], loadgen.Prompt], list[tuple[int, str]]]:
    """Return (next_prompt(i), prefixes to warm as [(tokens, group), ...])."""
    v, cell = run["variant"], run["cell"]
    fixed = run.get("fixed", {})

    def group(i: int) -> str:
        return f"g{i % concurrency}" if per_slot else "g0"

    if cell == "c3_prefill_miss":
        fresh = int(v["fresh_in"])
        return (lambda i: corpus.prefill_prompt(i, fresh)), []

    if cell == "c4_cached_prefix":
        pc, fi, out = int(v["prefix_cached"]), int(v["fresh_in"]), int(fixed.get("out", 200))
        control = v["control"]
        if control == "cached":
            warm = [(pc, group(i)) for i in range(concurrency if per_slot else 1)]

            def f(i: int) -> loadgen.Prompt:
                p = corpus.prefix_access_prompt(i, pc, fi, out, "cached")
                p.prefix = corpus.prefix(pc, group(i))
                p.prefix_group = group(i)
                return p
            return f, warm
        return (lambda i: corpus.prefix_access_prompt(i, pc, fi, out, "uncached")), []

    # decode-type cells: c1, c2, c5, c6, c7, c9
    ctx = int(v.get("context_cached", fixed.get("context_cached", 1024)))
    fresh = int(v.get("fresh_in", fixed.get("fresh_in", 200)))
    out = int(v.get("out", fixed.get("out", 256)))
    warm = [(ctx, group(i)) for i in range(concurrency if per_slot else 1)]
    return (lambda i: corpus.decode_prompt(i, ctx, fresh, out, group(i))), warm


# ---------------------------------------------------------------------------
# the cell
# ---------------------------------------------------------------------------

def run_cell(run: dict[str, Any], cat: Catalogue, server: VLLMServer | None,
             gpu_indices: list[int], out_root: Path, protocol: Protocol,
             tok: Any, cells_spec: dict[str, Any], commit: str | None = None) -> CellResult:
    """`run` is a manifest row (meter.feasibility.Run as dict) with the cell's
    `fixed` block merged in by the campaign runner. `server` may be None
    only for c0_static (model loaded, no requests still needs a server;
    the campaign passes the live one)."""
    run_id = run["run_id"]
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    flags: list[str] = []
    cell = run["cell"]
    variant = run["variant"]
    pod = pod_identity()
    meter = Meter(gpu_indices, sample_s=0.1, raw_dir=raw_dir)
    acc = cat.accelerators[run["accelerator"]]
    windows: list[dict[str, Any]] = []
    clock_state = None
    clocks_changed = False
    t_cell0 = time.time()

    def add_window(kind: str, wes: list[WindowEnergy], extra: dict[str, Any] | None = None):
        s = window_summary(wes)
        row = {"run_id": run_id, "window_id": f"{kind}", "kind": kind, **pod,
               "duration_s": s["duration_s"], "energy_j_total": s["energy_j_total"],
               "mean_power_w_total": s["mean_power_w_total"],
               "energy_sources": ",".join(s["sources"]), "n_gpus": len(wes),
               "per_gpu_json": json.dumps(s["per_gpu"])}
        if extra:
            row.update({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                        for k, v in extra.items()})
        windows.append(row)
        return s

    try:
        # -- clocks -----------------------------------------------------------
        wants_clock = variant.get("clock_mode") not in (None, "default") or "sm_clock_pct" in variant
        if wants_clock:
            req = variant.get("clock_mode", f"{variant.get('sm_clock_pct', 100)}%")
            clock_state = lock_sm_clocks(gpu_indices, req, acc.get("base_sm_clock_mhz"))
            clocks_changed = True
            if not clock_state.applied:
                flags.append(f"clock lock not applied: {clock_state.message}")

        # -- idle before ------------------------------------------------------
        idle_before = add_window("idle_before", _idle_window(meter, protocol.idle_window_s, "idle_before"))

        # -- active -----------------------------------------------------------
        requests_rows: list[dict[str, Any]] = []
        warm_results: list[dict[str, Any]] = []
        active_summary = None
        phase_tokens = None
        achieved = None
        target_batch = variant.get("target_batch")

        if cell == "c0_static":
            # model loaded, no requests: the window IS the measurement
            active_summary = add_window("active", _idle_window(meter, protocol.min_window_s, "active"),
                                        {"phase": "idle_loaded",
                                         "label": "loaded-idle board power under the default power-management policy"})
        else:
            assert server is not None, "a live server is required"
            concurrency = int(target_batch or 1)
            corpus = loadgen.Corpus(tok, protocol.seed + zlib.crc32(run_id.encode()) % 100000)
            url = server.base + "/v1/completions"
            model = run["model"]

            async def _active():
                nonlocal active_summary, phase_tokens, achieved
                async with httpx.AsyncClient(timeout=None) as client:
                    replay = cell == "c8_agent_replay"
                    if not replay:
                        nxt, warm = prompt_factory(run, corpus, concurrency,
                                                   protocol.prefix_groups == "per_slot")
                        # cache warming (excluded)
                        for tokens, group in warm:
                            wr = await loadgen.warm_prefix(client, url, model, corpus, tokens, group,
                                                           protocol.seed, server.metrics, tok)
                            warm_results.append(asdict(wr))
                        if warm and not all(w["confirmed"] for w in warm_results):
                            flags.append("cache warm not confirmed for every prefix group")
                        # warm-up requests (excluded): one round at the target concurrency
                        wu, _ = await loadgen.run_window(
                            client, url, model, lambda i: nxt(-1000 - i),
                            min(concurrency, max(1, protocol.warmup_requests)),
                            0.0, 0, protocol.temperature, protocol.seed, tok, max_seconds=120.0)
                        if any(r.error for r in wu):
                            flags.append(f"warm-up errors: {[r.error for r in wu if r.error][:2]}")

                    # metered window
                    sampler = MetricsSampler(server, protocol.metrics_poll_s)
                    meter.start()
                    sampler.start()
                    t0 = time.perf_counter()
                    if replay:
                        recs, script = await _replay(client, url, model, run, corpus,
                                                     concurrency, protocol, tok)
                        stats = loadgen.stats_from_records(recs, t0, time.perf_counter(), "decode")
                        extra = {"replay_script": [asdict(x) for x in script]}
                    else:
                        recs, stats = await loadgen.run_window(
                            client, url, model, nxt, concurrency, protocol.min_window_s,
                            protocol.min_tokens, protocol.temperature, protocol.seed, tok,
                            max_seconds=protocol.max_window_s)
                        extra = {}
                    await sampler.stop()
                    wes = meter.stop("active")
                    sampler.dump(out_dir / "sched_samples.jsonl")
                    achieved = sampler.achieved_batch(stats.t_start, stats.t_end)
                    phase_tokens = stats.phase_tokens
                    active_summary = add_window("active", wes, {
                        "phase": run["phase"], "n_requests": stats.n_requests,
                        "n_errors": stats.n_errors, "phase_tokens": stats.phase_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "prompt_tokens": stats.prompt_tokens, "cached_tokens": stats.cached_tokens,
                        "client_mean_concurrency": stats.client_mean_concurrency,
                        "target_batch": target_batch, **extra,
                        **{f"sched_{k}": v for k, v in achieved.items()},
                    })
                    rows = loadgen.records_to_rows(recs, run_id, "active")
                    for rw in rows:
                        rw["pod_id"] = pod["pod_id"]
                    requests_rows.extend(rows)

            asyncio.run(_active())

        # -- idle after -------------------------------------------------------
        idle_after = add_window("idle_after", _idle_window(meter, protocol.idle_window_s, "idle_after"))

        # -- checks -----------------------------------------------------------
        pb, pa_ = idle_before["mean_power_w_total"], idle_after["mean_power_w_total"]
        drift = abs(pa_ - pb) / pb if pb > 0 else None
        if drift is not None and drift > protocol.idle_drift_tolerance:
            flags.append(f"idle drift {drift:.1%} > {protocol.idle_drift_tolerance:.0%}: rerun")
        # Achieved-batch check applies to decode windows only: in prefill cells a
        # request lives for one or two iterations, so num_requests_running is not
        # the batch; tokens per iteration (server_prompt_tokens_delta) is.
        if run["phase"] == "decode" and achieved and target_batch and int(target_batch) >= 4 \
                and achieved.get("running_mean") is not None:
            if abs(achieved["running_mean"] - int(target_batch)) / int(target_batch) > 0.25:
                flags.append(f"achieved batch {achieved['running_mean']:.1f} vs target {target_batch}")
        # Window-length rules apply to single-phase windows; c0 is a fixed window
        # and c8's length is set by its script, not by the protocol minimum.
        if active_summary and cell not in ("c0_static", "c8_agent_replay"):
            if active_summary["duration_s"] < protocol.min_window_s * 0.95:
                flags.append("active window shorter than protocol minimum")
            if phase_tokens is not None and phase_tokens < protocol.min_tokens:
                flags.append(f"phase tokens {phase_tokens} < {protocol.min_tokens}")

    finally:
        if clocks_changed:
            try:
                reset_sm_clocks(gpu_indices)
            except Exception as e:  # pragma: no cover
                flags.append(f"clock reset failed: {e}")

    # -- write ----------------------------------------------------------------
    # rows carry different keys (idle vs active); pad to the union so pyarrow
    # keeps every column (it otherwise infers the schema from the first row)
    keys: list[str] = []
    for w in windows:
        for k in w:
            if k not in keys:
                keys.append(k)
    pq.write_table(pa.Table.from_pylist([{k: w.get(k) for k in keys} for w in windows]), out_dir / "windows.parquet")
    if requests_rows:
        pq.write_table(pa.Table.from_pylist(requests_rows), out_dir / "requests.parquet")
    meta = {
        "run": run,
        "protocol": asdict(protocol),
        "server": server.describe() if server else None,
        "gpu_indices": gpu_indices,
        "clock_state": asdict(clock_state) if clock_state else None,
        "warm_results": warm_results,
        "flags": flags,
        "harness_commit": commit or git_commit(),
        "pod": pod,
        "started_at": t_cell0,
        "finished_at": time.time(),
        "energy_note": "energy_j_total is GROSS board energy summed over participating GPUs; "
                       "no per-request allocation is made here",
    }
    (out_dir / "cell.json").write_text(json.dumps(meta, indent=1, default=str))

    active = next((w for w in windows if w["kind"] == "active"), None)
    return CellResult(
        run_id=run_id, ok=not flags, flags=flags, out_dir=str(out_dir),
        energy_j_active=active["energy_j_total"] if active else None,
        duration_s_active=active["duration_s"] if active else None,
        phase_tokens=phase_tokens, achieved_batch=(achieved or {}).get("running_mean"),
        target_batch=int(target_batch) if target_batch else None,
        idle_before_w=windows[0]["mean_power_w_total"] if windows else None,
        idle_after_w=windows[-1]["mean_power_w_total"] if len(windows) > 1 else None,
        warm_confirmed=all(w["confirmed"] for w in warm_results) if warm_results else None,
    )


async def _replay(client: httpx.AsyncClient, url: str, model: str, run: dict[str, Any],
                  corpus: loadgen.Corpus, n_agents: int, protocol: Protocol, tok: Any):
    """Cell c8: `n_agents` independent 30-step agents run concurrently; each
    step's prompt grows by context_step. Returns all records and the script."""
    fx = run.get("fixed", {})
    cap = int(run["variant"].get("context_cap", fx.get("context_cap", 10**9)))
    script = loadgen.replay_script(int(fx["steps"]), int(fx["context_start"]),
                                   int(fx["context_step"]), int(fx["out_per_step"]), cap)
    all_recs: list[loadgen.RequestRecord] = []

    async def agent(a: int):
        prev_prompt, prior = None, ""
        for step in script:
            p = loadgen.replay_prompt(corpus, a, step, prior, prev_prompt)
            rec = await loadgen.send_stream(client, url, completion_body_for(model, p, protocol), p, tok)
            rec.prefix_group = f"agent{a}:step{step.step}:{step.cache_expected}"
            all_recs.append(rec)
            prev_prompt = p.text
            prior = f"[answer {step.step}] " + loadgen.prose(
                __import__("random").Random(f"{a}:{step.step}"), step.out_tokens)

    await asyncio.gather(*(agent(a) for a in range(n_agents)))
    return all_recs, script


def completion_body_for(model: str, p: loadgen.Prompt, protocol: Protocol) -> dict[str, Any]:
    return loadgen.completion_body(model, p, protocol.temperature, protocol.seed)
