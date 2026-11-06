# Harness versions used by the 917 runs

`harness/` is the metering harness as it stood at commit `e14783d` (25 Oct 2026), the last version used in the campaign. Each run in the corpus records the commit it ran under (`harness_commit` in `data/store_tables/cells.parquet`). To rebuild an earlier version, apply its patch to this snapshot from inside `harness/`, for example `git apply runtime_patches/416f489.patch`.

| Commit | Date (UTC+1) | Runs | Change |
|---|---|---|---|
| `416f489` | 2026-10-24 20:53 | 377 | feasibility: native_max margin 256 + 4% (composed prompts tokenise 2.2% longer than sized; Qwen3-8B 32,969 > 32,512); analysis: dedupe run_ids across branches, drop zero-token decode windows |
| `5e49d84` | 2026-10-24 18:04 | 130 | campaign: results push survives unrelated shard histories; logs push stderr; excludes oversize files; falls back to per-pod snapshot branch |
| `2ae978b` | 2026-10-24 20:54 | 121 | HANDOFF: check 4 |
| `1cd7299` | 2026-10-24 11:38 | 109 | cell/obscal: pad window rows to the union of keys so windows.parquet keeps the active window's tokens, phase and scheduler metrics (pyarrow inferred the schema from the idle row); analysis package: load, features, fits with identifiability assessments, analyze report |
| `2ff9809` | 2026-10-24 14:50 | 100 | feasibility: native_max cells sized so prefix + fresh + output fit the context window (router pass: 32,972-token prompts rejected at 32,512) |
| `2c1453f` | 2026-10-25 12:22 | 19 | campaign: restart the server when it is gone after any cell, not only after an exception |
| `5d6c596` | 2026-10-24 18:12 | 18 | server: disable uvicorn access log (server log grew >100 MB/h and GitHub refused the blob, stalling every results push from ~11:46 to 17:04 UTC) |
| `a07b037` | 2026-10-25 18:12 | 12 | design additions after Astra review: obscal d9-d18 (batch 0/16/48, both MoE models, varied delivery); cell c9_kv_pressure (70B/B200, 16k context, batch 12-32) to measure the unsupported-region boundary |
| `5bf1bed` | 2026-10-25 17:25 | 8 | nvfp4: gpu-memory-utilization 0.7 (70B NVFP4 still OOMed at 0.8) |
| `e14783d` | 2026-10-25 21:14 | 8 | harness: energy boundary, batch semantics and c9 diagnostics (Astra's second review, items 9-11, 14, 17) |
| `e7a8e34` | 2026-10-25 16:08 | 8 | nvfp4: gpu-memory-utilization 0.8 (FP4 workspace OOMed after KV sizing at 0.9 on both checkpoints); campaign: a server config that failed to start is not retried for every cell of its group |
| `1053950` | 2026-10-24 20:49 | 6 | router pass: --enforce-eager for hooked servers (CUDA-graph decode left the capture buffer holding the dummy run's identical routing); diag records survive harvest |
| `88424d7` | 2026-10-25 13:57 | 1 | campaign: do not rerun B_max observations (target batch unreachable is a measurement, not a fault) |

Not included here because no run in the corpus used them: the blind-test endpoint, the observation-calibration set, the pod bootstrap script and the harness's own test suite.
