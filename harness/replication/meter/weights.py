"""Download a checkpoint into the Hugging Face cache BEFORE vLLM starts, timed and logged.

Why (13 Nov 2026): the Llama-3.1-405B pod on 4 x H200 spent 3.5 h of its 5.5 h cap inside vLLM's own download
(410 GB at about 33 MB/s) with nothing in its log, so a slow host could not be told from a hung one, and the
server-start timeout (sized to cover a cold download: about 10 h for 406 GB) could never fire before the cap.

Here the download runs in a child process, with parallel workers, before the server starts. Every `poll_s` it
logs the bytes in the cache, the rate and the free space on the volume. It stops early, with a clear reason,
when the projected time exceeds `max_seconds` (judged after `judge_after_s`), when the volume is nearly full,
or when `max_seconds` has passed. A complete cache returns at once (snapshot_download is idempotent).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

PATTERNS = ["*.safetensors", "*.json", "*.model", "*.tiktoken", "*.txt", "tokenizer*", "*.py"]
_SCRIPT = ("import sys; from huggingface_hub import snapshot_download; "
           "snapshot_download(sys.argv[1], allow_patterns=sys.argv[3].split(','), max_workers=int(sys.argv[2]))")


class DownloadError(RuntimeError):
    pass


def repo_cache_dir(hf_home: str | os.PathLike, hf_id: str) -> Path:
    return Path(hf_home) / "hub" / ("models--" + hf_id.replace("/", "--"))


def cached_bytes(path: Path) -> int:
    """Bytes of regular files under `path` (blobs and partial downloads; symlinks not followed)."""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                st = os.lstat(os.path.join(root, f))
            except OSError:
                continue
            if not os.path.islink(os.path.join(root, f)):
                total += st.st_size
    return total


def expected_bytes(hf_id: str) -> int | None:
    """Total size of the files the download will fetch, from the Hub's metadata; None if unavailable."""
    try:
        from fnmatch import fnmatch
        from huggingface_hub import HfApi
        info = HfApi().model_info(hf_id, files_metadata=True)
        return sum(int(s.size or 0) for s in info.siblings or []
                   if any(fnmatch(s.rfilename, p) for p in PATTERNS)) or None
    except Exception:   # noqa: BLE001  (progress is reported without a total)
        return None


def predownload(hf_id: str, hf_home: str | os.PathLike, max_seconds: float, *, log: Callable[[str], None] = print,
                poll_s: float = 60.0, judge_after_s: float = 600.0, min_free_gb: float = 5.0, workers: int = 16,
                total_bytes: int | None = None, cmd: list[str] | None = None,
                disk_usage: Callable[[str], Any] = shutil.disk_usage) -> dict[str, Any]:
    """Run the download; return {seconds, bytes, rate_mb_s, total_bytes}. Raise DownloadError on a stop."""
    hf_home = str(hf_home)
    os.makedirs(hf_home, exist_ok=True)
    cache = repo_cache_dir(hf_home, hf_id)
    total = total_bytes if total_bytes is not None else expected_bytes(hf_id)
    start_bytes = cached_bytes(cache) if cache.exists() else 0
    env = dict(os.environ, HF_HOME=hf_home, HF_XET_HIGH_PERFORMANCE="1")
    argv = cmd or [sys.executable, "-c", _SCRIPT, hf_id, str(workers), ",".join(PATTERNS)]
    t0 = time.time()
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    gb = 1e9

    def stop(reason: str) -> None:
        proc.kill()
        proc.wait()
        log(f"weights {hf_id}: STOPPED: {reason}")
        raise DownloadError(reason)

    while True:
        try:
            proc.wait(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            pass
        el = time.time() - t0
        have = cached_bytes(cache) if cache.exists() else 0
        rate = max(have - start_bytes, 0) / el if el > 0 else 0.0
        free = disk_usage(hf_home).free
        left = (total - have) / rate if total and rate > 0 else None
        log(f"weights {hf_id}: {have / gb:.1f}" + (f" of {total / gb:.1f}" if total else "") +
            f" GB, {rate / 1e6:.0f} MB/s, {el / 60:.0f} min, free {free / gb:.0f} GB" +
            (f", about {left / 60:.0f} min to go" if left is not None else ""))
        if free < min_free_gb * gb:
            stop(f"volume nearly full ({free / gb:.1f} GB free at {have / gb:.1f} GB downloaded)")
        if el > max_seconds:
            stop(f"not finished after {el / 60:.0f} min (limit {max_seconds / 60:.0f} min)")
        if el > judge_after_s and left is not None and el + left > max_seconds:
            stop(f"too slow: {rate / 1e6:.0f} MB/s projects {(el + left) / 60:.0f} min against a limit of "
                 f"{max_seconds / 60:.0f} min")
    out = (proc.stdout.read() if proc.stdout else "")[-1500:]
    el = time.time() - t0
    have = cached_bytes(cache) if cache.exists() else 0
    if proc.returncode != 0:
        log(f"weights {hf_id}: download failed (exit {proc.returncode}): {out}")
        raise DownloadError(f"download failed (exit {proc.returncode}): {out[-300:]}")
    rate = max(have - start_bytes, 0) / el if el > 0 else 0.0
    log(f"weights {hf_id}: complete, {have / gb:.1f} GB in {el / 60:.1f} min ({rate / 1e6:.0f} MB/s), "
        f"free {disk_usage(hf_home).free / gb:.0f} GB")
    return {"seconds": el, "bytes": have, "rate_mb_s": rate / 1e6, "total_bytes": total}
