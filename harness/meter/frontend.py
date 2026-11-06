"""A provider-style frontend in front of vLLM: applies a DELIVERY POLICY to
the token stream (Connecting Model v2 A3 "Causal delivery model").

    direct    forward each upstream event as it arrives
    paced     release no faster than one token per `cadence_ms`
              (r_i = max(c_i, r_{i-1} + delta)); a buffer builds when
              production outruns the cadence
    chunked   group `chunk_tokens` tokens into one event
    buffered  hold the first `startup_tokens` tokens, then release at
              `cadence_ms` (can be FASTER than production until the buffer
              drains, then stall) — optionally chunked too

Runs as a small asyncio HTTP server on `--listen`, forwarding
/v1/completions (and /v1/chat/completions) to `--upstream`. Everything
other than the stream shaping is passed through untouched, including the
usage block, so the probe sees exactly what a provider frontend would
show. Per-request release log (upstream arrival vs release time per event)
is written to `--log` for the delivery-model fit.

Tokens are counted per upstream event from the server's own per-event
text; vLLM emits one token per event by default, so 'tokens' here are
upstream events. Chunked policies regroup them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from aiohttp import web


@dataclass
class Policy:
    policy: str = "direct"
    cadence_ms: float = 0.0
    chunk_tokens: int = 1
    startup_tokens: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Policy":
        d = d or {}
        return cls(policy=d.get("policy", "direct"), cadence_ms=float(d.get("cadence_ms", 0) or 0),
                   chunk_tokens=int(d.get("chunk_tokens", 1) or 1), startup_tokens=int(d.get("startup_tokens", 0) or 0))


def _merge_payloads(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge n completion/chat SSE payloads into one event carrying their
    concatenated text. finish_reason and usage from the last that has them."""
    base = json.loads(json.dumps(parts[0]))
    for c in base.get("choices") or []:
        c["text"] = ""
        if "delta" in c:
            c["delta"] = {k: ("" if isinstance(v, str) else v) for k, v in c["delta"].items()}
    for p in parts:
        for i, c in enumerate(p.get("choices") or []):
            if i >= len(base["choices"]):
                base["choices"].append(json.loads(json.dumps(c)))
                continue
            b = base["choices"][i]
            if "text" in c:
                b["text"] = (b.get("text") or "") + (c.get("text") or "")
            if "delta" in c:
                for k, v in (c["delta"] or {}).items():
                    if isinstance(v, str):
                        b.setdefault("delta", {})[k] = (b["delta"].get(k) or "") + v
                    else:
                        b.setdefault("delta", {})[k] = v
            if c.get("finish_reason"):
                b["finish_reason"] = c["finish_reason"]
        if p.get("usage"):
            base["usage"] = p["usage"]
    return base


async def shape(upstream_events: AsyncIterator[tuple[float, dict[str, Any] | None]], pol: Policy,
                log: list[dict[str, Any]]) -> AsyncIterator[bytes]:
    """Apply the policy. `upstream_events` yields (arrival_time, payload) with
    payload None for [DONE]. Yields SSE bytes to send downstream."""
    delta = pol.cadence_ms / 1000.0
    chunk = max(1, pol.chunk_tokens)
    held: list[tuple[float, dict[str, Any]]] = []
    pending: list[tuple[float, dict[str, Any]]] = []
    last_release: float | None = None
    released_tokens = 0
    startup_done = pol.startup_tokens <= 0

    async def release(parts: list[tuple[float, dict[str, Any]]]):
        nonlocal last_release, released_tokens
        if delta > 0:
            earliest = (last_release + delta) if last_release is not None else 0.0
            now = time.perf_counter()
            if earliest > now:
                await asyncio.sleep(earliest - now)
        now = time.perf_counter()
        merged = _merge_payloads([p for _, p in parts])
        last_release = now
        released_tokens += len(parts)
        log.append({"arrivals": [a for a, _ in parts], "release": now, "n_tokens": len(parts)})
        return f"data: {json.dumps(merged)}\n\n".encode()

    def has_text(p: dict[str, Any]) -> bool:
        for c in p.get("choices") or []:
            if c.get("text") or (c.get("delta") or {}).get("content") or (c.get("delta") or {}).get("reasoning_content"):
                return True
        return False

    async for arrival, payload in upstream_events:
        if payload is None:
            break
        if not has_text(payload):
            # usage / finish-only payloads: queue behind held tokens, flush at the end
            pending.append((arrival, payload))
            continue
        held.append((arrival, payload))
        if not startup_done:
            if len(held) >= pol.startup_tokens:
                startup_done = True
            else:
                continue
        while len(held) >= chunk:
            parts, held = held[:chunk], held[chunk:]
            yield await release(parts)
    # drain
    if held:
        yield await release(held)
    for _, p in pending:
        yield f"data: {json.dumps(p)}\n\n".encode()
    yield b"data: [DONE]\n\n"


async def _upstream_events(client: httpx.AsyncClient, url: str, body: dict[str, Any],
                           headers: dict[str, str]) -> AsyncIterator[tuple[float, dict[str, Any] | None]]:
    async with client.stream("POST", url, json=body, headers=headers, timeout=None) as r:
        if r.status_code != 200:
            text = (await r.aread()).decode(errors="replace")
            raise web.HTTPBadGateway(text=text)
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                yield time.perf_counter(), None
                return
            try:
                yield time.perf_counter(), json.loads(payload)
            except json.JSONDecodeError:
                continue


def make_app(upstream: str, pol: Policy, log_path: Path | None, api_key: str | None = None) -> web.Application:
    """`api_key`: when set, requests must carry `Authorization: Bearer <api_key>`
    (a public blind endpoint must not take traffic from anyone but the probe,
    or its realised batch is not the configured one). The health route stays
    open; the policy is NOT reported on it when a key is set."""
    app = web.Application()
    client = httpx.AsyncClient(timeout=None)
    log_fh = log_path.open("a") if log_path else None

    async def proxy(request: web.Request) -> web.StreamResponse:
        if api_key is not None and request.headers.get("Authorization", "") != f"Bearer {api_key}":
            return web.json_response({"error": {"message": "unauthorized", "code": 401}}, status=401)
        body = await request.json()
        headers = {k: v for k, v in request.headers.items() if k.lower() in ("authorization", "content-type")}
        url = upstream.rstrip("/") + request.path
        t_arrival = time.perf_counter()
        if not body.get("stream"):
            r = await client.post(url, json=body, headers=headers)
            return web.Response(status=r.status_code, body=r.content, content_type="application/json")
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                        "Cache-Control": "no-cache"})
        await resp.prepare(request)
        rel_log: list[dict[str, Any]] = []
        async for chunk in shape(_upstream_events(client, url, body, headers), pol, rel_log):
            await resp.write(chunk)
        await resp.write_eof()
        if log_fh:
            log_fh.write(json.dumps({"t_request": t_arrival, "policy": pol.__dict__, "events": rel_log}) + "\n")
            log_fh.flush()
        return resp

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"ok": True} if api_key is not None else {"ok": True, "policy": pol.__dict__})

    app.router.add_post("/v1/completions", proxy)
    app.router.add_post("/v1/chat/completions", proxy)
    app.router.add_get("/health", health)
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000")
    ap.add_argument("--listen", default="127.0.0.1:8001")
    ap.add_argument("--policy", default='{"policy":"direct"}', help="JSON policy")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--api-key", default=None, help="require this bearer token (blind endpoints)")
    a = ap.parse_args(argv)
    host, port = a.listen.rsplit(":", 1)
    web.run_app(make_app(a.upstream, Policy.from_dict(json.loads(a.policy)), a.log, a.api_key), host=host, port=int(port),
                print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
