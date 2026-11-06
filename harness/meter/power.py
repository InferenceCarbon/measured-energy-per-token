"""Board power and energy, with counter qualification.

The first thing the pilot does on a pod is run `qualify()`. It identifies
which NVML interface is in use, measures its actual update interval, and
records clocks, power limit and driver. On Hopper and later,
nvmlDeviceGetPowerUsage is a ONE-SECOND average: sampling it every 10 ms
yields repeated values, not 10 ms resolution. Where the total-energy
counter (nvmlDeviceGetTotalEnergyConsumption) is available it is the
primary source, because it integrates on the device; power samples are
kept as the raw stream for diagnostics and re-integration.

Energy boundary: GROSS board energy during the window (Plan v3 §1).
Idle-subtracted energy is computed alongside as a diagnostic only.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


@dataclass
class WindowEnergy:
    gpu_index: int
    start_s: float
    end_s: float
    duration_s: float
    energy_j: float               # gross, from the energy counter if available
    energy_source: str            # "counter" | "trapezoid"
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


def _init():
    if pynvml is None:
        raise RuntimeError("pynvml not available; are you on the pod?")
    pynvml.nvmlInit()


def _measure_update_interval(fn, seconds: float = 3.0) -> float:
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
        time.sleep(0.002)
    return statistics.median(changes) if changes else seconds


def qualify(gpu_index: int = 0) -> CounterQualification:
    _init()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
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

    q = CounterQualification(
        gpu_index=gpu_index,
        name=name,
        driver=driver,
        power_limit_w=pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0,
        energy_counter_available=energy_ok,
        energy_counter_update_s=energy_update,
        power_reading_update_s=power_update,
        sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
        mem_clock_mhz=pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
        temp_c=pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        notes=notes,
    )
    return q


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


class Meter:
    """Samples every GPU in the pod for the life of a window and returns one
    WindowEnergy per GPU. Raw samples are saved to `raw_dir` as JSON lines
    so any window can be re-integrated later."""

    def __init__(self, gpu_indices: list[int], sample_s: float = 0.1,
                 raw_dir: Path | None = None):
        _init()
        self.handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_indices}
        self.sample_s = sample_s
        self.raw_dir = raw_dir
        self._stop = threading.Event()
        self._samples: dict[int, list[tuple[float, float]]] = {i: [] for i in gpu_indices}
        self._thread: threading.Thread | None = None
        self._start: dict[int, dict] = {}

    def _snap(self, h) -> dict:
        """Operating-policy snapshot: with no clock locking available on rented
        pods, the calibrated object is the OBSERVED policy, so every window
        records the clocks, power limit and throttle reasons it ran under."""
        d = {
            "t": time.perf_counter(),
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
            d["energy_mj"] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except pynvml.NVMLError:  # type: ignore[attr-defined]
            d["energy_mj"] = None
        return d

    def _loop(self):
        while not self._stop.is_set():
            now = time.perf_counter()
            for i, h in self.handles.items():
                self._samples[i].append((now, pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                                         pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)))
            time.sleep(self.sample_s)

    def start(self):
        self._start = {i: self._snap(h) for i, h in self.handles.items()}
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, window_id: str) -> list[WindowEnergy]:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        out: list[WindowEnergy] = []
        for i, h in self.handles.items():
            end = self._snap(h)
            s = self._samples[i]
            start = self._start[i]
            dur = end["t"] - start["t"]
            # Trapezoid over power samples (diagnostic / fallback)
            trap = 0.0
            for (t0, p0, _), (t1, p1, _) in zip(s, s[1:]):
                trap += 0.5 * (p0 + p1) * (t1 - t0)
            if start["energy_mj"] is not None and end["energy_mj"] is not None:
                e, src = (end["energy_mj"] - start["energy_mj"]) / 1000.0, "counter"
            else:
                e, src = trap, "trapezoid"
            out.append(WindowEnergy(
                gpu_index=i, start_s=start["t"], end_s=end["t"], duration_s=dur,
                energy_j=e, energy_source=src,
                mean_power_w=(e / dur) if dur > 0 else 0.0, n_samples=len(s),
                sm_clock_start=start["sm"], sm_clock_end=end["sm"],
                temp_start=start["temp"], temp_end=end["temp"],
                mem_clock_start=start.get("mem"), mem_clock_end=end.get("mem"),
                power_limit_w=end.get("power_limit_w"),
                throttle_start=start.get("throttle_reasons"), throttle_end=end.get("throttle_reasons"),
            ))
            if self.raw_dir:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                with (self.raw_dir / f"{window_id}.gpu{i}.jsonl").open("w") as f:
                    for t, p, sm in s:
                        f.write(json.dumps({"t": t, "p_w": p, "sm_mhz": sm}) + "\n")
            self._samples[i] = []
        return out


def window_summary(windows: list[WindowEnergy]) -> dict:
    """Sum across participating GPUs: gross energy and mean board power."""
    return {
        "energy_j_total": sum(w.energy_j for w in windows),
        "mean_power_w_total": sum(w.mean_power_w for w in windows),
        "duration_s": max(w.duration_s for w in windows),
        "sources": sorted({w.energy_source for w in windows}),
        "per_gpu": [asdict(w) for w in windows],
    }
