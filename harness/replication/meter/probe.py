"""W5 precision-detection probe: greedy completions with top-k logprobs on a FIXED prompt set.

Cell phase `logprob_probe`. Not an energy protocol window: the metered window still brackets
the requests (energy is recorded as a by-product and labelled with the phase), but no window
length or token minimum applies, and the achieved-batch check does not either.

The prompt set is seeded from the cell id and the model only (not the run id), so the fp8 and
nvfp4 servings of one model, and the same model on another accelerator, answer IDENTICAL
prompts: the analysis (analysis/w5_precision.py) pairs them position by position.

Requests are non-streaming `/v1/completions` calls with `logprobs: K`; vLLM returns, per output
token, the chosen token, its logprob and the top-K alternatives. Everything is kept verbatim
(as JSON strings) in logprobs.parquet, one row per request.
"""

from __future__ import annotations

import asyncio
import json
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Callable

import httpx

from . import loadgen
from .loadgen import Prompt, RequestRecord, WindowStats, stats_from_records

PHASE = "logprob_probe"


@dataclass
class ProbeRecord:
    request_index: int
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    status: int
    error: str | None
    text: str
    tokens_json: str            # chosen tokens, in order
    token_logprobs_json: str    # logprob of each chosen token
    top_logprobs_json: str      # per position: {token: logprob} for the top-K alternatives
    t_send: float
    t_done: float


def corpus_seed(cell: str, model: str, protocol_seed: int) -> int:
    """Same prompts for every precision and accelerator of one model (the run id carries both)."""
    return protocol_seed + zlib.crc32(f"{cell}/{model}".encode()) % 100000


def probe_body(model: str, prompt: Prompt, temperature: float, seed: int, top_logprobs: int) -> dict[str, Any]:
    body = loadgen.completion_body(model, prompt, temperature, seed)
    body.pop("stream_options", None)
    body["stream"] = False
    body["logprobs"] = int(top_logprobs)
    return body


def parse_logprobs(choice: dict[str, Any]) -> tuple[list[str], list[float | None], list[dict[str, float]]]:
    lp = choice.get("logprobs") or {}
    tokens = list(lp.get("tokens") or [])
    token_lps = list(lp.get("token_logprobs") or [])
    tops = [dict(t) if isinstance(t, dict) else {} for t in (lp.get("top_logprobs") or [])]
    return tokens, token_lps, tops


async def send_probe(client: httpx.AsyncClient, url: str, body: dict[str, Any], prompt: Prompt,
                     timeout_s: float = 600.0) -> tuple[RequestRecord, ProbeRecord]:
    """One non-streaming request. Returns the usual RequestRecord (so requests.parquet and the
    window statistics stay comparable) and the ProbeRecord with the logprobs."""
    t_send = time.perf_counter()
    rec = RequestRecord(prompt.request_index, prompt.kind, prompt.prefix_group,
                        prompt.prefix_tokens_target, prompt.suffix_tokens_target,
                        prompt.out_tokens, t_send, None, None, 0, None, None, None, None, 0)
    text, tokens, token_lps, tops = "", [], [], []
    try:
        r = await client.post(url, json=body, timeout=timeout_s)
        now = time.perf_counter()
        rec.status = r.status_code
        if r.status_code != 200:
            rec.error = r.text[:500]
        else:
            obj = r.json()
            choices = obj.get("choices") or []
            if choices:
                c = choices[0]
                text = c.get("text", "") or ""
                rec.finish_reason = c.get("finish_reason")
                tokens, token_lps, tops = parse_logprobs(c)
                if not tokens:
                    rec.error = "no logprobs in response"
            usage = obj.get("usage") or {}
            rec.prompt_tokens = usage.get("prompt_tokens")
            rec.completion_tokens = usage.get("completion_tokens")
            details = usage.get("prompt_tokens_details") or {}
            rec.cached_tokens = details.get("cached_tokens")
            rec.t_first_event = rec.t_last_event = now
            rec.n_events = 1
            rec.events.append(loadgen.StreamEvent(now, len(text), len(tokens) or None))
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        rec.error = f"{type(e).__name__}: {e}"
    t_done = time.perf_counter()
    probe = ProbeRecord(prompt.request_index, rec.prompt_tokens, rec.completion_tokens, rec.finish_reason,
                        rec.status, rec.error, text, json.dumps(tokens), json.dumps(token_lps),
                        json.dumps(tops), t_send, t_done)
    return rec, probe


async def run_probe(client: httpx.AsyncClient, url: str, model: str,
                    next_prompt: Callable[[int], Prompt], n_prompts: int, concurrency: int,
                    temperature: float, seed: int, top_logprobs: int,
                    ) -> tuple[list[RequestRecord], list[ProbeRecord], WindowStats]:
    """Exactly `n_prompts` requests (indices 0..n-1), at most `concurrency` in flight."""
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    recs: list[RequestRecord] = []
    probes: list[ProbeRecord] = []
    t0 = time.perf_counter()

    async def one(i: int) -> None:
        async with sem:
            p = next_prompt(i)
            rec, pr = await send_probe(client, url, probe_body(model, p, temperature, seed, top_logprobs), p)
            recs.append(rec)
            probes.append(pr)

    await asyncio.gather(*(one(i) for i in range(int(n_prompts))))
    recs.sort(key=lambda r: r.request_index)
    probes.sort(key=lambda r: r.request_index)
    return recs, probes, stats_from_records(recs, t0, time.perf_counter(), "decode")


def probe_rows(probes: list[ProbeRecord], run_id: str) -> list[dict[str, Any]]:
    rows = []
    for p in probes:
        d = asdict(p)
        d["run_id"] = run_id
        rows.append(d)
    return rows
