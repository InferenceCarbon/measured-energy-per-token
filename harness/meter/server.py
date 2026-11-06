"""Launch vLLM with the fixed campaign settings (Plan v3 §3) and expose
what the cell runner needs: health, Prometheus metrics, and SM clock
control.

Everything that varies between cells is an explicit field of
ServerConfig; everything that does not comes from serving.yaml. The
command line is built by a pure function so it can be tested off-pod.

Prefix caching is ON except in cache-miss cells (c3). Speculative
decoding is OFF in calibration; the field exists only for the
observation-calibration set and blind endpoints. A configured
max_num_seqs is a CAP, not the batch: achieved batch comes from the
scheduler metrics (num_requests_running), sampled during the window.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .templates import Catalogue, ModelTemplate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    model_key: str
    tp: int
    weight_precision: str = "fp8"
    kv_precision: str = "fp8"
    prefix_caching: bool = True
    max_num_seqs: int = 256
    max_model_len: int | None = None          # None -> model native context
    speculative_config: dict[str, Any] | None = None   # observation-calibration only
    port: int | None = None                   # None -> serving.yaml
    seed: int = 20261022
    router_hook: bool = False                 # MoE expert-union logging pass only (meter.moe_hook)
    # KV-pool size lever (c9 diagnostics, 25 Oct): None -> serving.yaml (0.9; 0.7 for NVFP4). The allocated
    # pool is what the scheduler preempts against, so a cell that varies it moves the unsupported-region boundary.
    gpu_memory_utilization: float | None = None

    def key(self) -> str:
        """Stable identity: cells sharing a key share one server process."""
        spec = json.dumps(self.speculative_config, sort_keys=True) if self.speculative_config else "none"
        return (f"{self.model_key}|tp{self.tp}|{self.weight_precision}|kv-{self.kv_precision}"
                f"|pc-{int(self.prefix_caching)}|seqs{self.max_num_seqs}"
                f"|len{self.max_model_len or 'native'}|spec-{spec}" + ("|hook" if self.router_hook else "")
                + (f"|gmu{self.gpu_memory_utilization}" if self.gpu_memory_utilization is not None else ""))

    def env(self, cat: Catalogue, hook_dir: str | None = None) -> dict[str, str]:
        """Extra process environment for this server."""
        if not self.router_hook:
            return {}
        m = cat.model(self.model_key)
        return {"ICM_MOE_HOOK": "1", "ICM_MOE_HOOK_DIR": hook_dir or "/workspace/moe_hook",
                "ICM_MOE_N_EXPERTS": str(m.moe["n_experts"]) if m.is_moe else "0"}


def resolve_hf_id(m: ModelTemplate, weight_precision: str) -> tuple[str, str | None]:
    """Return (hf_id, vllm --quantization value or None) for the served precision.

    Templates may carry precision-specific checkpoints (hf_id_fp8, hf_id_bf16,
    hf_id_nvfp4). Where an FP8 checkpoint is not listed and the default
    checkpoint is bf16, vLLM's online FP8 weight quantization is used and
    recorded (it is a different artefact from a pre-quantized checkpoint).
    """
    raw = m.raw
    if weight_precision == "nvfp4":
        if "hf_id_nvfp4" not in raw:
            raise ValueError(f"{m.key}: NVFP4 requested but no hf_id_nvfp4 checkpoint in template")
        return raw["hf_id_nvfp4"], None
    if weight_precision == "fp8":
        if "hf_id_fp8" in raw:
            return raw["hf_id_fp8"], None
        if "fp8" in raw["hf_id"].lower():
            return raw["hf_id"], None
        return raw["hf_id"], "fp8"
    if weight_precision == "bf16":
        if "hf_id_bf16" in raw:
            return raw["hf_id_bf16"], None
        if "fp8" in raw["hf_id"].lower():
            raise ValueError(f"{m.key}: bf16 requested but only an FP8 checkpoint is listed")
        return raw["hf_id"], None
    raise ValueError(f"unknown weight precision {weight_precision!r}")


def build_command(cfg: ServerConfig, cat: Catalogue) -> list[str]:
    """The exact `vllm serve` command for a config. Pure; tested off-pod."""
    m = cat.model(cfg.model_key)
    v = cat.serving["vllm"]
    hf_id, quant = resolve_hf_id(m, cfg.weight_precision)
    max_len = cfg.max_model_len or m.native_context
    port = cfg.port or v["port"]

    cmd = [
        "vllm", "serve", hf_id,
        "--tensor-parallel-size", str(cfg.tp),
        "--pipeline-parallel-size", str(v.get("pipeline_parallel_size", 1)),
        "--distributed-executor-backend", v.get("distributed_executor_backend", "mp"),
        # NVFP4 kernels allocate a workspace AFTER vLLM has sized the KV cache to the utilisation
        # target (both NVFP4 checkpoints OOMed at init with 0.9 on B200, 25 Oct; 70B still OOMed at 0.8 on a 14 GB workspace); leave headroom
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization if cfg.gpu_memory_utilization is not None
                                        else v.get("gpu_memory_utilization_nvfp4", 0.7) if cfg.weight_precision == "nvfp4"
                                        else v["gpu_memory_utilization"]),
        "--max-model-len", str(max_len),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--max-num-batched-tokens", str(v["max_num_batched_tokens"]),
        "--kv-cache-dtype", "fp8" if cfg.kv_precision == "fp8" else "auto",
        "--seed", str(cfg.seed),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--served-model-name", cfg.model_key,
        "--trust-remote-code",
        "--disable-uvicorn-access-log",      # one line per HTTP request made the server log >100 MB/h; GitHub refused the blob
    ]
    cmd.append("--enable-chunked-prefill" if v.get("enable_chunked_prefill", True)
               else "--no-enable-chunked-prefill")
    cmd.append("--enable-prefix-caching" if cfg.prefix_caching else "--no-enable-prefix-caching")
    if quant:
        cmd += ["--quantization", quant]
    if cfg.speculative_config:
        cmd += ["--speculative-config", json.dumps(cfg.speculative_config, sort_keys=True)]
    if cfg.router_hook:
        # makes vLLM call RoutedExpertsCapturer.capture per MoE layer, which the
        # meter.moe_hook plugin wraps (separate pass; overhead checked). Eager
        # execution: under CUDA graphs the capture buffer is written only by the
        # eager dummy runs (identical dummy tokens -> identical routing), so the
        # decode rows showed exactly k experts per layer on DeepSeek-V2-Lite
        # (24 Oct). Routing is a function of the tokens, not of graph mode, so
        # the union curve is unchanged by eager mode; this pass measures no
        # energy that is fitted.
        cmd += ["--enable-return-routed-experts", "--enforce-eager"]
    return cmd


# ---------------------------------------------------------------------------
# Prometheus text parsing
# ---------------------------------------------------------------------------

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?[0-9.eE+naNIif]+)')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Minimal exposition-format parser: name -> [(labels, value), ...]."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        mobj = _LINE.match(line)
        if not mobj:
            continue
        name, labels_s, val = mobj.groups()
        labels = dict(_LABEL.findall(labels_s)) if labels_s else {}
        try:
            value = float(val)
        except ValueError:
            continue
        out.setdefault(name, []).append((labels, value))
    return out


def metric_value(metrics: dict, candidates: list[str], default: float | None = None) -> float | None:
    """First candidate present, summed over label sets (one model served)."""
    for name in candidates:
        if name in metrics:
            return sum(v for _, v in metrics[name])
    return default


# Metric names differ between vLLM engine versions; each is tried in order.
METRIC_RUNNING = ["vllm:num_requests_running"]
METRIC_WAITING = ["vllm:num_requests_waiting"]
METRIC_KV_USAGE = ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"]
METRIC_PREFIX_QUERIES = ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries",
                         "vllm:gpu_prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries"]
METRIC_PREFIX_HITS = ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits",
                      "vllm:gpu_prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits"]
METRIC_PROMPT_TOKENS = ["vllm:prompt_tokens_total", "vllm:prompt_tokens"]
METRIC_GEN_TOKENS = ["vllm:generation_tokens_total", "vllm:generation_tokens"]
# scheduler preemptions (KV pressure: a running request evicted and requeued). Counter in vLLM >= 0.6;
# both spellings tried. Absent -> None, recorded as such (the 25 Oct c9 sweep had no counter).
METRIC_PREEMPTIONS = ["vllm:num_preemptions_total", "vllm:num_preemptions"]


@dataclass
class SchedulerSnapshot:
    t: float
    running: float
    waiting: float
    kv_usage: float | None
    prefix_queries: float | None
    prefix_hits: float | None
    prompt_tokens: float | None
    generation_tokens: float | None
    preemptions: float | None = None


def snapshot_from_metrics(metrics: dict, t: float | None = None) -> SchedulerSnapshot:
    return SchedulerSnapshot(
        t=time.perf_counter() if t is None else t,
        running=metric_value(metrics, METRIC_RUNNING, 0.0),
        waiting=metric_value(metrics, METRIC_WAITING, 0.0),
        kv_usage=metric_value(metrics, METRIC_KV_USAGE),
        prefix_queries=metric_value(metrics, METRIC_PREFIX_QUERIES),
        prefix_hits=metric_value(metrics, METRIC_PREFIX_HITS),
        prompt_tokens=metric_value(metrics, METRIC_PROMPT_TOKENS),
        generation_tokens=metric_value(metrics, METRIC_GEN_TOKENS),
        preemptions=metric_value(metrics, METRIC_PREEMPTIONS),
    )


_KV_POOL = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_KV_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x")


def kv_pool_from_log(text: str) -> dict[str, Any]:
    """The KV pool vLLM ALLOCATED, from its startup lines ("GPU KV cache size:
    N tokens"; "Maximum concurrency for L tokens per request: Cx"). This is
    the denominator of KV occupancy (c9 diagnostics): the pool the scheduler
    preempts against, not HBM minus weights. Last occurrence wins (a log that
    has seen several starts reports the current server)."""
    out: dict[str, Any] = {}
    m = None
    for m in _KV_POOL.finditer(text):
        pass
    if m:
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = None
    for m in _KV_CONC.finditer(text):
        pass
    if m:
        out["max_concurrency_at_tokens"] = int(m.group(1).replace(",", ""))
        out["max_concurrency"] = float(m.group(2))
    return out


# ---------------------------------------------------------------------------
# Process control
# ---------------------------------------------------------------------------

class VLLMServer:
    """One vLLM process. `start()` returns once /health answers; `metrics()`
    returns a SchedulerSnapshot; `stop()` terminates and waits."""

    def __init__(self, cfg: ServerConfig, cat: Catalogue, log_dir: Path,
                 env: dict[str, str] | None = None):
        self.cfg = cfg
        self.cat = cat
        self.cmd = build_command(cfg, cat)
        self.port = cfg.port or cat.serving["vllm"]["port"]
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_dir = log_dir
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.log_path = log_dir / f"vllm.{_slug(cfg.key())}.log"
        self._client = httpx.Client(timeout=10.0)
        self.started_at: float | None = None
        self.ready_at: float | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self, timeout_s: float | None = None) -> "VLLMServer":
        """Wait for /health. The default allowance scales with checkpoint
        size: a cold download runs at tens of MB/s on these pods, so 30 GB of
        weights needs more than the 30 minutes an 8B model needs."""
        if timeout_s is None:
            m = self.cat.model(self.cfg.model_key)
            gb = m.n_params_total_b * self.cat.bytes_per_param.get(self.cfg.weight_precision, 2)
            timeout_s = 900.0 + 90.0 * gb       # 8B fp8: ~27 min; 30B fp8: ~60 min; 70B fp8: ~2 h
        import os
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        self.started_at = time.time()
        with self.log_path.open("ab") as log:
            log.write((" ".join(self.cmd) + "\n").encode())
        logf = self.log_path.open("ab")
        self.proc = subprocess.Popen(self.cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        self.wait_healthy(timeout_s)
        self.ready_at = time.time()
        return self

    def wait_healthy(self, timeout_s: float) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.proc.returncode}; see {self.log_path}")
            try:
                r = self._client.get(self.base + "/health", timeout=2.0)
                if r.status_code == 200:
                    # /health answers before the model list does on some versions
                    r2 = self._client.get(self.base + "/v1/models", timeout=5.0)
                    if r2.status_code == 200:
                        return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"vLLM not healthy after {timeout_s}s; see {self.log_path}")

    def stop(self, grace_s: float = 30.0) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(grace_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(10.0)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -- observation --------------------------------------------------------
    def metrics_text(self) -> str:
        return self._client.get(self.base + "/metrics").text

    def metrics(self) -> SchedulerSnapshot:
        return snapshot_from_metrics(parse_prometheus(self.metrics_text()))

    def version(self) -> str | None:
        try:
            r = self._client.get(self.base + "/version", timeout=5.0)
            if r.status_code == 200:
                return r.json().get("version")
        except (httpx.HTTPError, ValueError):
            pass
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "config_key": self.cfg.key(),
            "command": self.cmd,
            "port": self.port,
            "vllm_version": self.version(),
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "load_seconds": (self.ready_at - self.started_at)
            if self.started_at and self.ready_at else None,
            "gpu_memory_utilization": next((self.cmd[j + 1] for j, a in enumerate(self.cmd) if a == "--gpu-memory-utilization"), None) if hasattr(self, "cmd") else None,
            "log": str(self.log_path),
            **self.kv_pool(),
        }

    def kv_pool(self) -> dict[str, Any]:
        """Allocated KV pool from this server's log (see kv_pool_from_log)."""
        try:
            return kv_pool_from_log(self.log_path.read_text(errors="replace"))
        except OSError:
            return {}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:120]


# ---------------------------------------------------------------------------
# SM clock control (cells c0 and c5)
# ---------------------------------------------------------------------------

@dataclass
class ClockState:
    gpu_indices: list[int]
    requested: str                    # "boost" | "base" | "60%" ...
    target_mhz: int | None
    max_sm_mhz: int
    applied: bool
    message: str = ""
    actual_sm_mhz: dict[int, int] = field(default_factory=dict)


def _nvidia_smi(args: list[str]) -> str:
    return subprocess.run(["nvidia-smi", *args], check=True, capture_output=True,
                          text=True).stdout


def query_max_sm_clock(gpu_index: int) -> int:
    out = _nvidia_smi(["-i", str(gpu_index), "--query-gpu=clocks.max.sm",
                       "--format=csv,noheader,nounits"])
    return int(float(out.strip().splitlines()[0]))


def query_current_sm_clock(gpu_index: int) -> int:
    out = _nvidia_smi(["-i", str(gpu_index), "--query-gpu=clocks.sm",
                       "--format=csv,noheader,nounits"])
    return int(float(out.strip().splitlines()[0]))


def resolve_clock_target(requested: str | int, max_sm_mhz: int,
                         base_sm_mhz: int | None) -> int | None:
    """Map a cell's clock setting to a target in MHz. `boost`/100% -> None
    (reset to default behaviour); `base` -> the accelerator's documented
    base clock if serving.yaml has it, else 60% of max (recorded as such);
    an int or 'NN%' -> that fraction of max."""
    if requested in ("boost", 100, "100%", "100"):
        return None
    if requested == "base":
        return base_sm_mhz if base_sm_mhz else int(round(max_sm_mhz * 0.60))
    pct = int(str(requested).rstrip("%"))
    return int(round(max_sm_mhz * pct / 100.0))


def lock_sm_clocks(gpu_indices: list[int], requested: str | int,
                   base_sm_mhz: int | None = None) -> ClockState:
    """Lock (or reset) the SM clock on every participating GPU. Requires
    root on the pod, which Runpod containers have. The actual clock is
    read back and recorded; a lock that the board does not honour is a
    recorded fact, not an assumption."""
    max_mhz = query_max_sm_clock(gpu_indices[0])
    target = resolve_clock_target(requested, max_mhz, base_sm_mhz)
    state = ClockState(gpu_indices=gpu_indices, requested=str(requested),
                       target_mhz=target, max_sm_mhz=max_mhz, applied=False)
    try:
        for i in gpu_indices:
            if target is None:
                _nvidia_smi(["-i", str(i), "-rgc"])
            else:
                _nvidia_smi(["-i", str(i), "-lgc", f"{target},{target}"])
        state.applied = True
        state.message = "reset" if target is None else f"locked at {target} MHz"
    except subprocess.CalledProcessError as e:
        # nvidia-smi reports permission problems on STDOUT ("Insufficient
        # Permissions"), so keep both streams.
        state.message = f"nvidia-smi failed: {(e.stdout or '').strip()} {(e.stderr or '').strip()}".strip()
    time.sleep(1.0)
    for i in gpu_indices:
        try:
            state.actual_sm_mhz[i] = query_current_sm_clock(i)
        except subprocess.CalledProcessError:
            pass
    return state


def reset_sm_clocks(gpu_indices: list[int]) -> ClockState:
    return lock_sm_clocks(gpu_indices, "boost")


def dump_nvidia_smi_q(path: Path) -> None:
    try:
        path.write_text(_nvidia_smi(["-q"]))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:  # pragma: no cover
        path.write_text(f"nvidia-smi -q failed: {e}\n")
