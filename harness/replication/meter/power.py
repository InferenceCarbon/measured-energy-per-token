"""Board power and energy, with counter qualification, behind a PowerSource.

The first thing the pilot does on a pod is run `qualify()`. It identifies
which interface is in use, measures its actual update interval, and
records clocks, power limit and driver. On Hopper and later,
nvmlDeviceGetPowerUsage is a ONE-SECOND average: sampling it every 10 ms
yields repeated values, not 10 ms resolution. Where the total-energy
counter (nvmlDeviceGetTotalEnergyConsumption) is available it is the
primary source, because it integrates on the device; power samples are
kept as the raw stream for diagnostics and re-integration.

Energy boundary: GROSS board energy during the window (Plan v3 §1).
Idle-subtracted energy is computed alongside as a diagnostic only.

Power sources:
    nvml    NVIDIA boards: energy counter + 0.1 s power (the Paper 1 meter)
    fake    a synthetic source for the CPU dry run (no hardware)
The Meter is the same for every source; only the source differs, so the
window records keep one schema.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

try:
    import pynvml
except ImportError:  # pragma: no cover - not present off-pod
    pynvml = None


@dataclass
class CounterQualification:
    gpu_index: int
    name: str
    driver: str
    power_limit_w: float
    energy_counter_available: bool
    energy_counter_update_s: float | None
    power_reading_update_s: float
    sm_clock_mhz: int
    mem_clock_mhz: int
    temp_c: int
    notes: list[str] = field(default_factory=list)
    power_source: str = "nvml"
    power_unit: str = "W"
    power_scale_w: float | None = None


@dataclass
class WindowEnergy:
    gpu_index: int
    start_s: float
    end_s: float
    duration_s: float
    energy_j: float               # gross, from the energy counter if available
    energy_source: str
    mean_power_w: float
    n_samples: int
    sm_clock_start: int
    sm_clock_end: int
    temp_start: int
    temp_end: int
    mem_clock_start: int | None = None
    mem_clock_end: int | None = None
    power_limit_w: float | None = None
    throttle_start: list[str] | None = None
    throttle_end: list[str] | None = None
    power_pct_mean: float | None = None
    power_scale_w: float | None = None
    n_power_updates: int | None = None
    power_source: str = "nvml"
    notes: list[str] = field(default_factory=list)
    start_wall: float | None = None
    end_wall: float | None = None


# ---------------------------------------------------------------------------
# The source abstraction
# ---------------------------------------------------------------------------

class PowerSource(Protocol):
    """What the Meter needs from a backend. Indices are the backend's own
    device numbering (NVML index, Neuron device index, fake slot)."""
    name: str

    def qualify(self, index: int) -> CounterQualification: ...

    def snapshot(self, index: int) -> dict[str, Any]:
        """Operating-policy snapshot at a window edge: must carry "t"
        (perf_counter), "wall" (time.time()), "sm", "mem", "temp",
        "power_limit_w", "throttle_reasons", "energy_j" (None if no
        integrating counter)."""
        ...

    def sample(self, index: int) -> tuple[float | None, dict[str, Any]]:
        """(instantaneous power in W or None, extra fields for the raw stream)."""
        ...

    def integrate(self, index: int, samples: list[tuple[float, float | None, dict[str, Any]]],
                  start: dict[str, Any], end: dict[str, Any]) -> tuple[float, str, dict[str, Any]] | None:
        """Backend-specific window energy, or None to use the Meter's default
        (counter delta, else trapezoid over the power samples). Returns
        (energy_j, source_label, extra)."""
        ...

    def describe(self) -> dict[str, Any]: ...


def _measure_update_interval(fn, seconds: float = 3.0, poll_s: float = 0.002) -> float:
    """How often does `fn()` return a NEW value? Returns the median gap in
    seconds between value changes, or `seconds` if it never changed."""
    t0 = time.perf_counter()
    last = fn()
    changes: list[float] = []
    last_t = t0
    while time.perf_counter() - t0 < seconds:
        v = fn()
        now = time.perf_counter()
        if v != last:
            changes.append(now - last_t)
            last, last_t = v, now
        time.sleep(poll_s)
    return statistics.median(changes) if changes else seconds


# ---------------------------------------------------------------------------
# NVML source (the Paper 1 meter, unchanged in behaviour)
# ---------------------------------------------------------------------------

def _init():
    if pynvml is None:
        raise RuntimeError("pynvml not available; are you on the pod?")
    pynvml.nvmlInit()


_THROTTLE_NAMES = {
    "nvmlClocksEventReasonGpuIdle": "gpu_idle",
    "nvmlClocksEventReasonApplicationsClocksSetting": "app_clocks",
    "nvmlClocksEventReasonSwPowerCap": "sw_power_cap",
    "nvmlClocksEventReasonHwSlowdown": "hw_slowdown",
    "nvmlClocksEventReasonSyncBoost": "sync_boost",
    "nvmlClocksEventReasonSwThermalSlowdown": "sw_thermal",
    "nvmlClocksEventReasonHwThermalSlowdown": "hw_thermal",
    "nvmlClocksEventReasonHwPowerBrakeSlowdown": "hw_power_brake",
    "nvmlClocksEventReasonDisplayClockSetting": "display_clock",
    # older NVML names
    "nvmlClocksThrottleReasonGpuIdle": "gpu_idle",
    "nvmlClocksThrottleReasonApplicationsClocksSetting": "app_clocks",
    "nvmlClocksThrottleReasonSwPowerCap": "sw_power_cap",
    "nvmlClocksThrottleReasonHwSlowdown": "hw_slowdown",
    "nvmlClocksThrottleReasonSyncBoost": "sync_boost",
    "nvmlClocksThrottleReasonSwThermalSlowdown": "sw_thermal",
    "nvmlClocksThrottleReasonHwThermalSlowdown": "hw_thermal",
    "nvmlClocksThrottleReasonHwPowerBrakeSlowdown": "hw_power_brake",
}


def _throttle_reasons(h) -> list[str] | None:
    """Decoded clock-event (throttle) reasons bitmask, or None if unsupported."""
    fn = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
        getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
    if fn is None:
        return None
    try:
        mask = fn(h)
    except pynvml.NVMLError:  # type: ignore[attr-defined]
        return None
    out = []
    for const, name in _THROTTLE_NAMES.items():
        bit = getattr(pynvml, const, None)
        if bit and mask & bit and name not in out:
            out.append(name)
    return out or (["none"] if mask == 0 else [f"0x{mask:x}"])


class NvmlSource:
    name = "nvml"

    def __init__(self):
        _init()
        self._handles: dict[int, Any] = {}

    def _h(self, i: int):
        if i not in self._handles:
            self._handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
        return self._handles[i]

    def qualify(self, gpu_index: int = 0) -> CounterQualification:
        h = self._h(gpu_index)
        name = pynvml.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        notes: list[str] = []

        energy_ok = True
        energy_update = None
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
            energy_update = _measure_update_interval(
                lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
        except pynvml.NVMLError as e:  # type: ignore[attr-defined]
            energy_ok = False
            notes.append(f"energy counter unavailable: {e}")

        power_update = _measure_update_interval(lambda: pynvml.nvmlDeviceGetPowerUsage(h))
        if power_update > 0.5:
            notes.append(f"power reading updates every ~{power_update:.2f}s: treat as a "
                         "1-second average; do not claim sub-second resolution")

        return CounterQualification(
            gpu_index=gpu_index, name=name, driver=driver,
            power_limit_w=pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0,
            energy_counter_available=energy_ok, energy_counter_update_s=energy_update,
            power_reading_update_s=power_update,
            sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            mem_clock_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            temp_c=pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            notes=notes, power_source=self.name, power_unit="W",
        )

    def snapshot(self, i: int) -> dict[str, Any]:
        """Operating-policy snapshot: with no clock locking available on rented
        pods, the calibrated object is the OBSERVED policy, so every window
        records the clocks, power limit and throttle reasons it ran under."""
        h = self._h(i)
        d: dict[str, Any] = {
            "t": time.perf_counter(), "wall": time.time(),
            "sm": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "temp": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        }
        try:
            d["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
        except pynvml.NVMLError:  # type: ignore[attr-defined]
            d["power_limit_w"] = None
        d["throttle_reasons"] = _throttle_reasons(h)
        try:
            d["energy_j"] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
        except pynvml.NVMLError:  # type: ignore[attr-defined]
            d["energy_j"] = None
        return d

    def sample(self, i: int) -> tuple[float | None, dict[str, Any]]:
        h = self._h(i)
        return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, \
            {"sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)}

    def integrate(self, index, samples, start, end):
        return None

    def describe(self) -> dict[str, Any]:
        return {"source": self.name, "unit": "W", "energy_counter": "nvmlDeviceGetTotalEnergyConsumption"}


def qualify(gpu_index: int = 0) -> CounterQualification:
    """Backward-compatible NVML qualification (Paper 1 call site)."""
    return NvmlSource().qualify(gpu_index)


# ---------------------------------------------------------------------------
# Fake source (CPU dry run)
# ---------------------------------------------------------------------------

class FakeSource:
    """Synthetic board: idle draw plus noise, with an integrating counter so
    the counter path is exercised, and optional `pct_mode` so the Neuron
    percent-of-max bookkeeping is exercised too. Deterministic per index."""
    name = "fake"

    def __init__(self, idle_w: float = 120.0, noise_w: float = 2.0, seed: int = 1,
                 pct_mode: bool = False, scale_w: float | None = 500.0):
        self.idle_w, self.noise_w, self.pct_mode, self.scale_w = idle_w, noise_w, pct_mode, scale_w
        self._rng = random.Random(seed)
        self._energy: dict[int, float] = {}
        self._last_t: dict[int, float] = {}

    def _advance(self, i: int) -> float:
        now = time.perf_counter()
        p = self.idle_w + self._rng.uniform(-self.noise_w, self.noise_w)
        if i in self._last_t:
            self._energy[i] = self._energy.get(i, 0.0) + p * (now - self._last_t[i])
        else:
            self._energy[i] = 0.0
        self._last_t[i] = now
        return p

    def qualify(self, index: int) -> CounterQualification:
        return CounterQualification(index, "FAKE-DEVICE", "fake-0.0", self.idle_w * 5, not self.pct_mode,
                                    0.0 if not self.pct_mode else None, 0.0, 1000, 1000, 40,
                                    ["synthetic source: dry run only; no measurement"],
                                    power_source=self.name,
                                    power_unit="percent_of_max" if self.pct_mode else "W",
                                    power_scale_w=self.scale_w if self.pct_mode else None)

    def snapshot(self, i: int) -> dict[str, Any]:
        self._advance(i)
        return {"t": time.perf_counter(), "wall": time.time(), "sm": 1000, "mem": 1000, "temp": 40,
                "power_limit_w": self.idle_w * 5, "throttle_reasons": ["none"],
                "energy_j": None if self.pct_mode else self._energy[i]}

    def sample(self, i: int) -> tuple[float | None, dict[str, Any]]:
        p = self._advance(i)
        if self.pct_mode:
            pct = 100.0 * p / (self.scale_w or 1.0)
            return (p if self.scale_w else None), {"pct_avg": pct, "pct_min": pct - 0.5, "pct_max": pct + 0.5,
                                                   "ts": int(time.time()) // 60 * 60}
        return p, {"sm_mhz": 1000}

    def integrate(self, index, samples, start, end):
        if not self.pct_mode:
            return None
        pcts = [x["pct_avg"] for _, _, x in samples if "pct_avg" in x]
        return Meter.trapezoid(samples), "fake_pct_x_scale", {
            "pct_mean": statistics.fmean(pcts) if pcts else None, "scale_w": self.scale_w,
            "n_updates": len({x.get("ts") for _, _, x in samples if "ts" in x}),
            "notes": ["synthetic percent-of-max source; dry run only"]}

    def describe(self) -> dict[str, Any]:
        return {"source": self.name, "unit": "percent_of_max" if self.pct_mode else "W",
                "scale_w": self.scale_w if self.pct_mode else None, "note": "synthetic; dry run only"}


def make_source(backend: str, acc: dict[str, Any] | None = None, **kw) -> PowerSource:
    """Registry. `acc` is the serving.yaml accelerator block (Neuron reads its
    declared power scale from it)."""
    if backend in ("nvml", "cuda", "vllm_cuda"):
        return NvmlSource()
    if backend == "neuron":
        from .neuron import NeuronSysfsSource
        return NeuronSysfsSource.from_accelerator(acc or {}, **kw)
    if backend == "tpu":
        from .tpu import TpuTelemetrySource
        return TpuTelemetrySource.from_accelerator(acc or {}, **kw)
    if backend == "fake":
        return FakeSource(**kw)
    raise ValueError(f"unknown power backend {backend!r}")


# ---------------------------------------------------------------------------
# The meter
# ---------------------------------------------------------------------------

class Meter:
    """Samples every device in the group for the life of a window and returns
    one WindowEnergy per device. Raw samples are saved to `raw_dir` as JSON
    lines so any window can be re-integrated later."""

    def __init__(self, gpu_indices: list[int], sample_s: float = 0.1,
                 raw_dir: Path | None = None, source: PowerSource | None = None):
        self.source: PowerSource = source if source is not None else NvmlSource()
        self.indices = list(gpu_indices)
        self.sample_s = sample_s
        self.raw_dir = raw_dir
        self._stop = threading.Event()
        self._samples: dict[int, list[tuple[float, float | None, dict[str, Any]]]] = {i: [] for i in self.indices}
        self._thread: threading.Thread | None = None
        self._start: dict[int, dict] = {}

    # kept for callers that poke at the NVML handles (none in the repo; cheap to keep)
    @property
    def handles(self) -> dict[int, Any]:
        return {i: i for i in self.indices}

    def _loop(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            for i in self.indices:
                try:
                    p, extra = self.source.sample(i)
                except Exception as e:  # a transient read error must not kill the window
                    p, extra = None, {"error": f"{type(e).__name__}: {e}"}
                self._samples[i].append((now, p, extra))
            time.sleep(self.sample_s)

    def start(self):
        self._start = {i: self.source.snapshot(i) for i in self.indices}
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @staticmethod
    def trapezoid(samples: list[tuple[float, float | None, dict[str, Any]]]) -> float:
        pts = [(t, p) for t, p, _ in samples if p is not None and not math.isnan(p)]
        trap = 0.0
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            trap += 0.5 * (p0 + p1) * (t1 - t0)
        return trap

    def stop(self, window_id: str) -> list[WindowEnergy]:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        out: list[WindowEnergy] = []
        for i in self.indices:
            end = self.source.snapshot(i)
            s = self._samples[i]
            start = self._start[i]
            dur = end["t"] - start["t"]
            trap = self.trapezoid(s)
            extra: dict[str, Any] = {}
            custom = self.source.integrate(i, s, start, end)
            if custom is not None:
                e, src, extra = custom
            elif start.get("energy_j") is not None and end.get("energy_j") is not None:
                e, src = end["energy_j"] - start["energy_j"], "counter"
            else:
                e, src = trap, "trapezoid"
            pcts = [x.get("pct_avg") for _, _, x in s if x.get("pct_avg") is not None]
            out.append(WindowEnergy(
                gpu_index=i, start_s=start["t"], end_s=end["t"], duration_s=dur,
                energy_j=e, energy_source=src,
                mean_power_w=(e / dur) if dur > 0 else 0.0, n_samples=len(s),
                sm_clock_start=start["sm"], sm_clock_end=end["sm"],
                temp_start=start["temp"], temp_end=end["temp"],
                mem_clock_start=start.get("mem"), mem_clock_end=end.get("mem"),
                power_limit_w=end.get("power_limit_w"),
                throttle_start=start.get("throttle_reasons"), throttle_end=end.get("throttle_reasons"),
                power_pct_mean=extra.get("pct_mean", statistics.fmean(pcts) if pcts else None),
                power_scale_w=extra.get("scale_w"),
                n_power_updates=extra.get("n_updates"),
                power_source=self.source.name,
                notes=list(extra.get("notes", [])),
                start_wall=start.get("wall"), end_wall=end.get("wall"),
            ))
            if self.raw_dir:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                with (self.raw_dir / f"{window_id}.gpu{i}.jsonl").open("w") as f:
                    for t, p, x in s:
                        f.write(json.dumps({"t": t, "p_w": p, **x}) + "\n")
            self._samples[i] = []
        return out


def refinalize(source: PowerSource, windows: list[WindowEnergy], wait_s: float = 75.0) -> list[str]:
    """For sources whose records arrive late (Neuron: one record per minute,
    landing after the minute ends), wait once for the record that closes the
    LAST window, then re-integrate every window from the full record set.
    Returns notes. No-op for sources without integrate_interval."""
    notes: list[str] = []
    waiter = getattr(source, "wait_for_record_after", None)
    if waiter is None or not hasattr(source, "integrate_interval") or not windows:
        return notes
    last_end = max(w.end_wall or 0.0 for w in windows)
    if not waiter(last_end, wait_s):
        notes.append(f"closing power record did not land within {wait_s:.0f}s; last window may be under-covered")
    for w in windows:
        if w.start_wall is None or w.end_wall is None:
            continue
        res = source.integrate_interval(w.gpu_index, w.start_wall, w.end_wall)
        w.energy_j = res["energy_j"]
        w.mean_power_w = (w.energy_j / w.duration_s) if w.duration_s > 0 else 0.0
        w.power_pct_mean = res["pct_mean"]
        w.n_power_updates = res["n_updates"]
        w.energy_source = "neuron_pct_x_scale" if res["coverage"] >= 0.95 else "neuron_pct_x_scale_partial"
        w.notes = [n for n in w.notes if not n.startswith("coverage ")] + [f"coverage {res['coverage']:.0%} after cell-end re-integration"]
    return notes


def window_summary(windows: list[WindowEnergy]) -> dict:
    """Sum across participating devices: gross energy and mean board power."""
    return {
        "energy_j_total": sum(w.energy_j for w in windows),
        "mean_power_w_total": sum(w.mean_power_w for w in windows),
        "duration_s": max(w.duration_s for w in windows),
        "sources": sorted({w.energy_source for w in windows}),
        "per_gpu": [asdict(w) for w in windows],
    }
