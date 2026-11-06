"""Request generation for one cell: prompt corpus, cache warming, concurrent
dispatch to a target batch, and raw stream-event capture.

Prompts are AGENT-SHAPED (Plan v3 §3, §4): a constant preamble, an
accumulated context, a fresh tool-output block that is unique per request,
and a short instruction. The cached part (preamble + accumulated) is the
PREFIX; the unique part is the SUFFIX. Suffixes are varied so repeated
requests do not become cache hits themselves.

Timing is recorded per STREAM EVENT (arrival timestamp, payload size),
never per token: the client sees delivery events, and the estimator's
delivery model (Connecting Model v2) consumes exactly that. Exact token
counts come from the server's usage block at the end of the stream,
including provider-reported cached tokens where vLLM supplies them.

Nothing here touches the power meter; the cell runner brackets a window
of these requests with the meter.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

import httpx


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class WordTokenizer:
    """Whitespace tokenizer for off-pod tests: one token per word. Sized
    prompts built with it are correct in word count only."""
    def encode(self, text: str) -> list[int]:
        return [hash(w) & 0xFFFF for w in text.split()]

    def decode(self, ids: list[int]) -> str:  # not invertible; only used via `sized_text`
        raise NotImplementedError

    def truncate(self, text: str, n: int) -> str:
        return " ".join(text.split()[:n])


class HFTokenizer:
    """Model tokenizer on the pod (transformers)."""
    def __init__(self, hf_id: str):
        from transformers import AutoTokenizer  # pod-only import
        self.tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def truncate(self, text: str, n: int) -> str:
        ids = self.encode(text)[:n]
        out = self.decode(ids)
        # decode->encode is not always length-preserving; trim once more
        for _ in range(3):
            k = len(self.encode(out))
            if k <= n:
                break
            out = self.decode(self.encode(out)[: n - (k - n)])
        return out


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_WORDS = (
    "accelerator allocation anchor aperture archive attention bandwidth baseline batch "
    "boundary buffer cache calibration capacity carbon cell checkpoint cluster coefficient "
    "compute context counter cycle datacentre decode deployment discrepancy dispatch domain "
    "duration energy envelope estimator event execution expert facility factor fleet forward "
    "framework generation gradient grid hardware hidden host idle inference interval "
    "iteration joule kernel kilowatt latency layer logical manifest measurement memory metric "
    "model module network node observation operator output overhead packet parallel parameter "
    "phase pipeline placement platform power precision prefix prefill probe profile prompt "
    "provider quantile queue rack rate record regime register release replica request residual "
    "resident response routing runtime sample scenario schedule scheduler segment sequence "
    "server service shape signal speculation stage state static stream suffix support "
    "template tensor threshold throughput token tolerance trace traffic transfer transport "
    "utilisation validation variance vector verdict watt weight window workload"
).split()

_CONNECTIVES = ["and", "but", "so", "then", "while", "because", "whereas", "although"]


def prose(rng: random.Random, n_words: int, tag: str = "") -> str:
    """Deterministic pseudo-technical prose. `tag` salts the text so two
    blocks with the same rng state are still different (fresh suffixes)."""
    words: list[str] = []
    if tag:
        words.append(f"[{tag}]")
    sentence_len = rng.randint(9, 19)
    while len(words) < n_words:
        w = rng.choice(_WORDS)
        if rng.random() < 0.12:
            w = f"{w}-{rng.randint(1, 9999)}"
        words.append(w)
        sentence_len -= 1
        if sentence_len <= 0:
            words[-1] += "."
            if rng.random() < 0.3:
                words.append(rng.choice(_CONNECTIVES))
            sentence_len = rng.randint(9, 19)
    return " ".join(words[:n_words])


def sized_text(tok: Any, target_tokens: int, rng: random.Random, tag: str = "") -> str:
    """Text whose token count under `tok` is <= target and close to it.
    Over-generate in words (a token is ~0.7 word for this vocabulary with
    BPE tokenizers) then truncate to the exact id count."""
    if target_tokens <= 0:
        return ""
    text = prose(rng, int(target_tokens * 1.6) + 8, tag)
    return tok.truncate(text, target_tokens)


PREAMBLE_TOKENS_MAX = 512
INSTRUCTION = "\n\nUsing only the material above, state the three most important figures and what they imply, in one paragraph."


@dataclass
class Prompt:
    request_index: int
    prefix: str                 # cached part (preamble + accumulated); "" when all fresh
    suffix: str                 # fresh part (tool output + instruction)
    prefix_tokens_target: int
    suffix_tokens_target: int
    out_tokens: int
    prefix_group: str           # requests sharing a prefix share this id
    kind: str                   # decode | prefill | prefix_access | replay_step

    @property
    def text(self) -> str:
        return self.prefix + self.suffix


class Corpus:
    """Builds prompts for one cell from the cell's shape parameters.

    seed      : cell-level seed; prefixes are reproducible across repeats
    tok       : tokenizer for sizing
    The fresh suffix of request i is salted with i, so no two suffixes in a
    cell are equal and none repeats a suffix from the warm-up.
    """

    def __init__(self, tok: Any, seed: int):
        self.tok = tok
        self.seed = seed
        self._prefix_cache: dict[tuple[int, str], str] = {}

    def prefix(self, tokens: int, group: str = "g0") -> str:
        key = (tokens, group)
        if key not in self._prefix_cache:
            rng = random.Random(f"{self.seed}:prefix:{group}:{tokens}")
            pre_n = min(PREAMBLE_TOKENS_MAX, tokens // 2)
            preamble = ("You are an analysis agent. Follow the operating notes, then the record.\n"
                        + sized_text(self.tok, pre_n, rng, "notes"))
            rest = max(0, tokens - len(self.tok.encode(preamble)))
            accumulated = "\n\nAccumulated record:\n" + sized_text(self.tok, rest, rng, "record")
            text = preamble + accumulated
            self._prefix_cache[key] = self.tok.truncate(text, tokens)
        return self._prefix_cache[key]

    def suffix(self, tokens: int, request_index: int, salt: str = "") -> str:
        rng = random.Random(f"{self.seed}:suffix:{salt}:{request_index}")
        instr_n = len(self.tok.encode(INSTRUCTION))
        body = sized_text(self.tok, max(1, tokens - instr_n - 4), rng, f"tool-output-{request_index}")
        return "\n\nFresh tool output:\n" + body + INSTRUCTION

    def decode_prompt(self, i: int, context_cached: int, fresh_in: int, out: int,
                      group: str = "g0") -> Prompt:
        return Prompt(i, self.prefix(context_cached, group), self.suffix(fresh_in, i),
                      context_cached, fresh_in, out, group, "decode")

    def prefill_prompt(self, i: int, fresh_in: int) -> Prompt:
        """Cache-miss prefill: everything fresh and unique; max_tokens=1."""
        return Prompt(i, "", self.suffix(fresh_in, i, salt="prefill"), 0, fresh_in, 1,
                      f"unique-{i}", "prefill")

    def prefix_access_prompt(self, i: int, prefix_cached: int, fresh_in: int, out: int,
                             control: str) -> Prompt:
        if control == "cached":
            return Prompt(i, self.prefix(prefix_cached), self.suffix(fresh_in, i),
                          prefix_cached, fresh_in, out, "g0", "prefix_access")
        # uncached control: same total length, all fresh and unique
        total = prefix_cached + fresh_in
        return Prompt(i, "", self.suffix(total, i, salt="uncached"), 0, total, out,
                      f"unique-{i}", "prefix_access")


# ---------------------------------------------------------------------------
# Agent replay script (cell c8)
# ---------------------------------------------------------------------------

@dataclass
class ReplayStep:
    step: int
    context_tokens: int        # prompt length this step
    out_tokens: int
    cache_expected: str        # "hit" | "miss" (scripted)


def replay_script(steps: int, context_start: int, context_step: int, out_per_step: int,
                  context_cap: int, miss_every: int = 5) -> list[ReplayStep]:
    """Scripted cache policy: a miss every `miss_every` steps (step 1 is
    always a miss: nothing is cached yet). The energy of the whole replay
    is predicted from this script BEFORE the run (Plan v3 §4, cell 8)."""
    out: list[ReplayStep] = []
    ctx = context_start
    for s in range(1, steps + 1):
        if ctx > context_cap:
            break
        miss = (s == 1) or (s % miss_every == 0)
        out.append(ReplayStep(s, ctx, out_per_step, "miss" if miss else "hit"))
        ctx += context_step
    return out


def replay_prompt(corpus: Corpus, agent: int, step: ReplayStep, prior_answer: str,
                  previous_prompt: str | None) -> Prompt:
    """Step prompt = previous prompt + prior answer + fresh tool output.
    A scripted MISS perturbs the first line of the preamble for this agent
    from this step onward, so the cached prefix no longer matches (a
    stand-in for eviction; recorded as such)."""
    if previous_prompt is None:
        base = corpus.prefix(step.context_tokens - 256, f"agent{agent}")
        fresh_n = 256
    else:
        base = previous_prompt + "\n\nAssistant:\n" + prior_answer
        have = len(corpus.tok.encode(base))
        fresh_n = max(32, step.context_tokens - have)
    if step.cache_expected == "miss" and previous_prompt is not None:
        base = base.replace("You are an analysis agent.",
                            f"You are an analysis agent (session {step.step}).", 1)
    suffix = corpus.suffix(fresh_n, step.step, salt=f"agent{agent}")
    return Prompt(step.step, base, suffix, len(corpus.tok.encode(base)), fresh_n,
                  step.out_tokens, f"agent{agent}", "replay_step")


# ---------------------------------------------------------------------------
# Requests and stream capture
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    t: float            # perf_counter at arrival
    chars: int
    tokens_est: int | None


@dataclass
class RequestRecord:
    request_index: int
    kind: str
    prefix_group: str
    prefix_tokens_target: int
    suffix_tokens_target: int
    out_tokens_target: int
    t_send: float
    t_first_event: float | None
    t_last_event: float | None
    n_events: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None           # server-reported, where available
    finish_reason: str | None
    status: int
    error: str | None = None
    events: list[StreamEvent] = field(default_factory=list)
    in_window: bool = True

    @property
    def ttft_s(self) -> float | None:
        return None if self.t_first_event is None else self.t_first_event - self.t_send

    @property
    def duration_s(self) -> float | None:
        return None if self.t_last_event is None else self.t_last_event - self.t_send


def completion_body(model: str, prompt: Prompt, temperature: float, seed: int,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt.text,
        "max_tokens": prompt.out_tokens,
        "temperature": temperature,
        "seed": seed + prompt.request_index,
        "stream": True,
        "stream_options": {"include_usage": True},
        # vLLM extras: force exactly `out` tokens so output length is a control,
        # not a model choice.
        "ignore_eos": True,
        "min_tokens": prompt.out_tokens,
    }
    if extra:
        body.update(extra)
    return body


async def send_stream(client: httpx.AsyncClient, url: str, body: dict[str, Any],
                      prompt: Prompt, tok: Any | None = None,
                      timeout_s: float = 600.0) -> RequestRecord:
    rec = RequestRecord(prompt.request_index, prompt.kind, prompt.prefix_group,
                        prompt.prefix_tokens_target, prompt.suffix_tokens_target,
                        prompt.out_tokens, time.perf_counter(), None, None, 0,
                        None, None, None, None, 0)
    try:
        async with client.stream("POST", url, json=body, timeout=timeout_s) as r:
            rec.status = r.status_code
            if r.status_code != 200:
                rec.error = (await r.aread())[:500].decode(errors="replace")
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage")
                choices = obj.get("choices") or []
                text = "".join(c.get("text", "") for c in choices)
                if text:
                    if rec.t_first_event is None:
                        rec.t_first_event = now
                    rec.t_last_event = now
                    rec.n_events += 1
                    rec.events.append(StreamEvent(
                        now, len(text), len(tok.encode(text)) if tok is not None else None))
                for c in choices:
                    if c.get("finish_reason"):
                        rec.finish_reason = c["finish_reason"]
                if usage:
                    rec.prompt_tokens = usage.get("prompt_tokens")
                    rec.completion_tokens = usage.get("completion_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    rec.cached_tokens = details.get("cached_tokens")
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        rec.error = f"{type(e).__name__}: {e}"
    return rec


# ---------------------------------------------------------------------------
# Cache warming
# ---------------------------------------------------------------------------

@dataclass
class WarmResult:
    prefix_group: str
    prefix_tokens_target: int
    warm_prompt_tokens: int | None
    probe_cached_tokens: int | None
    probe_prompt_tokens: int | None
    hit_fraction: float | None            # cached / prefix target, from the probe
    metrics_hit_delta: float | None       # prefix-cache hit counter delta across the probe
    confirmed: bool


async def warm_prefix(client: httpx.AsyncClient, url: str, model: str, corpus: Corpus,
                      prefix_tokens: int, group: str, seed: int,
                      metrics_fn: Callable[[], Any] | None = None,
                      tok: Any | None = None) -> WarmResult:
    """Send the prefix once (max_tokens=1), then a probe with a throwaway
    suffix, and confirm the hit from the probe's usage and the server's
    prefix-cache counters. A warm that cannot be confirmed is reported, and
    the cell runner decides (pilot: gate miss)."""
    warm = Prompt(-1, corpus.prefix(prefix_tokens, group), "", prefix_tokens, 0, 1, group, "warm")
    w = await send_stream(client, url, completion_body(model, warm, 0.0, seed), warm, tok)
    before = metrics_fn() if metrics_fn else None
    probe = Prompt(-2, corpus.prefix(prefix_tokens, group),
                   corpus.suffix(64, -2, salt="probe"), prefix_tokens, 64, 1, group, "probe")
    p = await send_stream(client, url, completion_body(model, probe, 0.0, seed), probe, tok)
    after = metrics_fn() if metrics_fn else None
    delta = None
    if before is not None and after is not None and \
            getattr(before, "prefix_hits", None) is not None and \
            getattr(after, "prefix_hits", None) is not None:
        delta = after.prefix_hits - before.prefix_hits
    frac = (p.cached_tokens / prefix_tokens) if (p.cached_tokens and prefix_tokens) else None
    confirmed = bool((frac is not None and frac >= 0.9) or (delta is not None and delta > 0))
    return WarmResult(group, prefix_tokens, w.prompt_tokens, p.cached_tokens, p.prompt_tokens,
                      frac, delta, confirmed)


# ---------------------------------------------------------------------------
# Concurrent dispatch to a target batch
# ---------------------------------------------------------------------------

@dataclass
class WindowStats:
    t_start: float
    t_end: float
    n_requests: int
    n_errors: int
    phase_tokens: int
    completion_tokens: int
    prompt_tokens: int
    cached_tokens: int
    client_mean_concurrency: float   # from request overlap on the client side


def _phase_tokens(kind: str, rec: RequestRecord) -> int:
    """Tokens that count toward the window's single phase."""
    if kind == "prefill":
        return rec.prompt_tokens or rec.suffix_tokens_target
    if kind == "prefix_access":
        return (rec.suffix_tokens_target) + (rec.completion_tokens or 0)
    return rec.completion_tokens or 0


async def run_window(client: httpx.AsyncClient, url: str, model: str,
                     next_prompt: Callable[[int], Prompt], concurrency: int,
                     min_seconds: float, min_tokens: int, temperature: float, seed: int,
                     tok: Any | None = None, max_seconds: float = 1800.0,
                     extra_body: dict[str, Any] | None = None,
                     on_record: Callable[[RequestRecord], None] | None = None,
                     ) -> tuple[list[RequestRecord], WindowStats]:
    """Closed loop: `concurrency` workers each keep one request in flight.
    New requests are issued until BOTH min_seconds and min_tokens are met
    (or max_seconds), then in-flight requests drain. The caller starts the
    meter before calling and stops it after it returns, so the metered
    window brackets everything here including the drain."""
    records: list[RequestRecord] = []
    counter = {"i": 0, "tokens": 0}
    t0 = time.perf_counter()
    stop = asyncio.Event()

    def done() -> bool:
        el = time.perf_counter() - t0
        return el >= max_seconds or (el >= min_seconds and counter["tokens"] >= min_tokens)

    async def worker():
        while not stop.is_set():
            i = counter["i"]
            counter["i"] += 1
            p = next_prompt(i)
            rec = await send_stream(client, url, completion_body(model, p, temperature, seed, extra_body), p, tok)
            records.append(rec)
            if rec.error is None:
                counter["tokens"] += _phase_tokens(p.kind, rec)
            if on_record:
                on_record(rec)
            if done():
                stop.set()

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    t1 = time.perf_counter()
    return records, stats_from_records(records, t0, t1)


def stats_from_records(records: list[RequestRecord], t0: float, t1: float,
                       kind_override: str | None = None) -> WindowStats:
    good = [r for r in records if r.error is None]
    # client-side mean concurrency: integral of in-flight time over the window
    busy = sum(((r.t_last_event or t1) - r.t_send) for r in records)
    return WindowStats(
        t_start=t0, t_end=t1, n_requests=len(records),
        n_errors=len(records) - len(good),
        phase_tokens=sum(_phase_tokens(kind_override or r.kind, r) for r in good),
        completion_tokens=sum(r.completion_tokens or 0 for r in good),
        prompt_tokens=sum(r.prompt_tokens or 0 for r in good),
        cached_tokens=sum(r.cached_tokens or 0 for r in good),
        client_mean_concurrency=busy / (t1 - t0) if t1 > t0 else 0.0,
    )


def records_to_rows(records: list[RequestRecord], run_id: str, window_id: str) -> list[dict[str, Any]]:
    """Flatten for Parquet: one row per request; events kept as JSON."""
    rows = []
    for r in records:
        d = asdict(r)
        events = d.pop("events")
        d["events_json"] = json.dumps([[e["t"], e["chars"], e["tokens_est"]] for e in events])
        d["ttft_s"] = r.ttft_s
        d["duration_s"] = r.duration_s
        d["run_id"] = run_id
        d["window_id"] = window_id
        rows.append(d)
    return rows
