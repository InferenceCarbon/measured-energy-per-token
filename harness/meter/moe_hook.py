"""MoE router hook: log the set of experts touched per layer per forward
step, so the routing-aware weight coordinate W_batch (Plan v3 §2) is
MEASURED on the calibration models instead of assumed from independent
uniform routing (meter.work.expected_expert_union is the closed-model
estimate).

Runs INSIDE each vLLM worker process as a general plugin
(entry point group `vllm.general_plugins`, declared in pyproject.toml), so
it works with the multiprocessing executor. Enabled only when
ICM_MOE_HOOK=1 is in the environment; the campaign sets it for the
separate router-logging pass only (§3 "Records": overhead checked
against clean runs, never mixed with energy cells).

What it records, per worker rank, as JSON lines in ICM_MOE_HOOK_DIR:
    {"t": perf_counter, "layer": <index in call order within the step>,
     "tokens": n_tokens_in_this_forward, "unique": n_distinct_experts,
     "topk": k, "n_experts": N}
A "step" is one forward pass of the whole model; layers are numbered by
their call order within the step (the hook sees no layer id), which is
stable for a given model.

Union per step = sum over layers of `unique` × per-expert bytes + shared
bytes; `tokens` is the achieved batch of that iteration (decode) or the
chunk size (prefill), so the two can be separated afterwards.

Patch point (vLLM >= 0.15): RoutedExpertsCapturer.capture, which vLLM
itself calls once per MoE layer per forward with the exact layer id when
the server runs with --enable-return-routed-experts (meter.server adds
that flag to hooked servers). Older versions: FusedMoE.select_experts.
If neither is found the hook logs loudly, with a survey of the installed
API, and does nothing, so a pass without data is a visible failure.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

_state: dict[str, Any] = {"enabled": False, "layer": 0, "last_t": None, "buf": [], "fh": None,
                          "n_experts": None, "topk": None}


# ---------------------------------------------------------------------------
# Pure helpers (tested off-pod)
# ---------------------------------------------------------------------------

def unique_count(topk_ids: Any) -> int:
    """Number of distinct expert ids in a [tokens, k] tensor or nested list."""
    try:
        import torch  # pod-only
        if isinstance(topk_ids, torch.Tensor):
            return int(torch.unique(topk_ids).numel())
    except ImportError:
        pass
    flat = [x for row in topk_ids for x in (row if isinstance(row, (list, tuple)) else [row])]
    return len(set(flat))


def step_boundary(now: float, last_t: float | None, gap_s: float = 0.0005) -> bool:
    """A new forward step is inferred from a time gap larger than `gap_s`
    between consecutive router calls. Within one step the MoE layers run
    back to back (tens of microseconds apart on H100); between steps the
    scheduler, sampling and host work take at least ~1 ms."""
    return last_t is None or (now - last_t) > gap_s


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate hook records: per-step totals and the union-vs-tokens curve
    the fit needs (mean unique experts per layer at each batch size)."""
    # layers whose distinct-expert count never exceeds 1 are dense layers that the capture buffer carries
    # as a constant row (DeepSeek-V2-Lite's first layer): they are not MoE layers and must not enter the
    # per-layer mean (6 x 26/27 = 5.8 was the symptom; Astra, 25 Oct)
    max_unique: dict[int, int] = {}
    for r in records:
        if not r.get("diag") and "layer" in r:
            max_unique[r["layer"]] = max(max_unique.get(r["layer"], 0), int(r.get("unique", 0)))
    dense_layers = {l for l, u in max_unique.items() if u <= 1}
    moe_layers = sorted(l for l in max_unique if l not in dense_layers)
    first_moe = moe_layers[0] if moe_layers else 0
    steps: list[dict[str, Any]] = []
    cur: list[dict[str, Any]] = []
    for r in records:
        if r.get("diag") or r.get("layer") in dense_layers:
            continue
        if r.get("layer", 0) == first_moe and cur:
            steps.append(_fold(cur))
            cur = []
        cur.append(r)
    if cur:
        steps.append(_fold(cur))
    by_tokens: dict[int, list[float]] = {}
    for s in steps:
        by_tokens.setdefault(s["tokens"], []).append(s["mean_unique_per_layer"])
    curve = {t: sum(v) / len(v) for t, v in sorted(by_tokens.items())}
    return {"n_steps": len(steps), "n_layers": max((s["n_layers"] for s in steps), default=0),
            "dense_layers_excluded": sorted(dense_layers),
            "union_curve_mean_unique_per_layer_by_tokens": curve, "steps": steps}


def _fold(cur: list[dict[str, Any]]) -> dict[str, Any]:
    uniq = [r["unique"] for r in cur]
    return {"t": cur[0]["t"], "tokens": cur[0]["tokens"], "n_layers": len(cur),
            "sum_unique": sum(uniq), "mean_unique_per_layer": sum(uniq) / len(uniq),
            "topk": cur[0].get("topk"), "n_experts": cur[0].get("n_experts")}


def read_records(hook_dir: Path, offsets: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Records from every rank file. With `offsets` (path -> bytes already
    consumed) only new records are returned and the offsets are advanced,
    so one long-lived worker file can be split per cell without renaming
    a file the worker still holds open."""
    out: list[dict[str, Any]] = []
    for f in sorted(hook_dir.glob("moe_hook.rank*.jsonl")):
        start = (offsets or {}).get(str(f), 0)
        with f.open("rb") as fh:
            fh.seek(start)
            data = fh.read()
        # keep a trailing partial line for next time
        cut = data.rfind(b"\n") + 1
        if offsets is not None:
            offsets[str(f)] = start + cut
        for line in data[:cut].splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(rec)                   # diag records included; summarise() skips them
    return out


# ---------------------------------------------------------------------------
# vLLM plugin
# ---------------------------------------------------------------------------

def _find_select_experts() -> tuple[Any, str, str] | None:
    """Return (owner, attribute, kind) for the hook strategy.

    kind "capturer" (vLLM >= 0.15): three patches around vLLM's own
        RoutedExpertsCapturer, which the server creates when run with
        --enable-return-routed-experts:
          - FusedMoE.__init__: remember every MoE layer instance;
          - RoutedExpertsCapturer.init_buffer: AFTER the buffer exists, bind
            each layer's `capture` to capturer.capture(layer_id, topk_ids).
            vLLM 0.15.1 binds this at layer construction, which happens
            before the capturer exists in the server flow, so without this
            step no layer ever writes the buffer (pilot 3: capturer created,
            zero capture calls);
          - RoutedExpertsCapturer.save_captured_experts(indices): called by
            the model runner once per step on the host side, after the
            forward (and after CUDA-graph replay has refreshed the device
            buffer); we read buffer[:len(indices)] and log one record per
            layer: tokens in the step and distinct experts touched.
    kind "router" (older vLLM): FusedMoE.select_experts(...) returning
        (topk_weights, topk_ids); layer index inferred from call order.
    """
    import importlib
    try:
        cap_mod = importlib.import_module("vllm.model_executor.layers.fused_moe.routed_experts_capturer")
        layer_mod = importlib.import_module("vllm.model_executor.layers.fused_moe.layer")
        cap = getattr(cap_mod, "RoutedExpertsCapturer", None)
        fm = getattr(layer_mod, "FusedMoE", None)
        if cap is not None and fm is not None and all(
                callable(getattr(cap, a, None)) for a in ("init_buffer", "save_captured_experts", "capture")):
            return (cap, fm), "capturer", "capturer"
    except Exception:
        pass
    try:
        layer_mod = importlib.import_module("vllm.model_executor.layers.fused_moe.layer")
        fm = getattr(layer_mod, "FusedMoE", None)
        if fm is not None and callable(getattr(fm, "select_experts", None)):
            return fm, "select_experts", "router"
    except Exception:
        pass
    return None


_layers: list[Any] = []     # FusedMoE instances in this process (weakrefs)


def _install_capturer_hooks(cap: Any, fm: Any) -> None:
    import weakref
    orig_init = fm.__dict__["__init__"]
    orig_init_buffer = cap.__dict__["init_buffer"]
    orig_save = cap.__dict__["save_captured_experts"]

    def init_wrapped(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            _layers.append(weakref.ref(self))
        except Exception as e:
            _state.setdefault("errors", []).append(repr(e))

    def init_buffer_wrapped(self, *args, **kwargs):
        out = orig_init_buffer(self, *args, **kwargs)
        bound = 0
        for ref in list(_layers):
            layer = ref()
            if layer is None:
                continue
            if getattr(layer, "capture", None) is None:
                lid = getattr(layer, "layer_id", None)
                layer.capture = (lambda topk_ids, _lid=lid, _cap=self: _cap.capture(_lid, topk_ids))
                bound += 1
        _state["bound_layers"] = bound
        try:
            with open(_state["log_path"], "a") as f:
                f.write(f"ICM_MOE_HOOK: init_buffer seen; bound capture on {bound} of {len(_layers)} FusedMoE layers\n")
        except Exception:
            pass
        return out

    def save_wrapped(self, indices, *args, **kwargs):
        try:
            _log_step_from_buffer(self, indices)
        except Exception as e:
            _state.setdefault("errors", []).append(repr(e))
            if len(_state["errors"]) <= 3:
                with open(_state["log_path"], "a") as f:
                    f.write(f"ICM_MOE_HOOK: save hook error {e!r}\n")
        return orig_save(self, indices, *args, **kwargs)

    fm.__init__ = init_wrapped
    cap.init_buffer = init_buffer_wrapped
    cap.save_captured_experts = save_wrapped


def _log_step_from_buffer(capturer: Any, indices: Any) -> None:
    """One record per MoE layer for this step, from vLLM's device buffer
    [max_tokens, n_layers, k]; `indices` has one entry per token in the step."""
    import torch
    buf = getattr(capturer, "_device_buffer", None)
    if buf is None:
        return
    n = int(len(indices))
    if n <= 0:
        return
    step = buf[:n]                                   # [n, L, k] on device
    L = int(step.shape[1])
    k = int(step.shape[2]) if step.dim() > 2 else None
    now = time.perf_counter()
    uniq = [int(torch.unique(step[:, l, :]).numel()) for l in range(L)]
    fh = _state["fh"]
    # diagnostic for the first steps: where vLLM says the tokens sit and what the first rows hold,
    # so a capture that writes elsewhere than rows 0..n-1 (seen as union == k at every B on
    # DeepSeek-V2-Lite, 24 Oct) can be told apart from a routing collapse
    _state["diag_steps"] = _state.get("diag_steps", 0) + 1
    if _state["diag_steps"] <= 30:
        try:
            idx = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            rows = step[: min(n, 4), min(L - 1, 1), :].tolist()
            fh.write(json.dumps({"diag": True, "tokens": n, "indices_head": idx[:8], "indices_max": max(idx) if idx else None,
                                 "buffer_rows": int(buf.shape[0]), "rows_head_layer1": rows}) + "\n")
        except Exception as e:  # pragma: no cover
            fh.write(json.dumps({"diag": True, "error": repr(e)}) + "\n")
    for l in range(L):
        fh.write(json.dumps({"t": now, "layer": l, "tokens": n, "unique": uniq[l],
                             "topk": k, "n_experts": _state["n_experts"]}) + "\n")
    fh.flush()


def _api_survey() -> str:
    """What the installed vLLM exposes around the router, for the hook log:
    the next revision of the candidate list is written from this."""
    import importlib
    out = []
    for mod_name in ("vllm.model_executor.layers.fused_moe.router",
                     "vllm.model_executor.layers.fused_moe.layer",
                     "vllm.model_executor.layers.fused_moe"):
        try:
            mod = importlib.import_module(mod_name)
            names = [n for n in dir(mod) if not n.startswith("_")]
            out.append(f"{mod_name}: {names}")
            fm = getattr(mod, "FusedMoE", None)
            if fm is not None:
                out.append(f"  FusedMoE attrs: {[n for n in dir(fm) if 'expert' in n.lower() or 'rout' in n.lower() or 'topk' in n.lower()]}")
        except Exception as e:
            out.append(f"{mod_name}: import failed: {e!r}")
    try:
        import vllm
        out.insert(0, f"vllm {vllm.__version__}")
    except Exception:
        pass
    return "\n".join(out)


def _record(layer_hint: int | None, topk_ids: Any) -> None:
    now = time.perf_counter()
    if layer_hint is not None:
        layer = int(layer_hint)
        if layer <= _state.get("prev_layer", -1):      # layer ids restarted: new step
            _state["fh"].flush()
        _state["prev_layer"] = layer
    else:
        if step_boundary(now, _state["last_t"]):
            _state["layer"] = 0
            _state["fh"].flush()
        layer = _state["layer"]
        _state["layer"] += 1
    tokens = int(topk_ids.shape[0]) if hasattr(topk_ids, "shape") else len(topk_ids)
    k = int(topk_ids.shape[1]) if hasattr(topk_ids, "shape") and len(topk_ids.shape) > 1 else None
    rec = {"t": now, "layer": layer, "tokens": tokens, "unique": unique_count(topk_ids),
           "topk": k, "n_experts": _state["n_experts"]}
    _state["fh"].write(json.dumps(rec) + "\n")
    _state["last_t"] = time.perf_counter()


def _wrap_capture(original: Callable) -> Callable:
    """RoutedExpertsCapturer.capture(self, layer_id, topk_ids)."""
    def wrapped(self, layer_id, topk_ids, *args, **kwargs):
        try:
            _record(layer_id, topk_ids)
        except Exception as e:  # never break serving
            _state.setdefault("errors", []).append(repr(e))
        return original(self, layer_id, topk_ids, *args, **kwargs)
    return wrapped


def _wrap_router(original: Callable) -> Callable:
    """Older API: a router call returning (topk_weights, topk_ids)."""
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            topk_ids = result[1] if isinstance(result, (tuple, list)) and len(result) >= 2 else None
            if topk_ids is not None:
                _record(None, topk_ids)
        except Exception as e:
            _state.setdefault("errors", []).append(repr(e))
        return result
    return wrapped


def register() -> None:
    """Entry point called by vLLM in every process at start-up. Must never
    raise: an exception here aborts `vllm serve` itself."""
    if os.environ.get("ICM_MOE_HOOK") != "1":
        return
    try:
        _register()
    except Exception as e:  # pragma: no cover - defensive
        try:
            out_dir = Path(os.environ.get("ICM_MOE_HOOK_DIR", "/workspace/moe_hook"))
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"moe_hook.pid{os.getpid()}.log").write_text(f"ICM_MOE_HOOK: register failed: {e!r}\n")
        except Exception:
            pass


def _register() -> None:
    found = _find_select_experts()
    out_dir = Path(os.environ.get("ICM_MOE_HOOK_DIR", "/workspace/moe_hook"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", str(os.getpid())))
    log = out_dir / f"moe_hook.rank{rank}.log"
    if found is None:
        log.write_text("ICM_MOE_HOOK: router function not found in this vLLM; hook inactive\n"
                       + _api_survey() + "\n")
        return
    owner, attr, kind = found
    _state["log_path"] = str(log)
    if kind == "capturer":
        cap, fm = owner
        _install_capturer_hooks(cap, fm)
        owner = cap
        attr = "init_buffer+save_captured_experts"
    else:
        original = owner.__dict__.get(attr, getattr(owner, attr))
        raw = original.__func__ if isinstance(original, staticmethod) else original
        setattr(owner, attr, staticmethod(_wrap_router(raw)) if isinstance(original, staticmethod) else _wrap_router(raw))
    _state["fh"] = (out_dir / f"moe_hook.rank{rank}.jsonl").open("a")
    _state["enabled"] = True
    try:
        _state["n_experts"] = int(os.environ.get("ICM_MOE_N_EXPERTS", "0")) or None
    except ValueError:
        _state["n_experts"] = None
    log.write_text(f"ICM_MOE_HOOK: patched {getattr(owner, '__module__', owner)}."
                   f"{getattr(owner, '__name__', '')}.{attr} (kind={kind})\n" + _api_survey() + "\n")
