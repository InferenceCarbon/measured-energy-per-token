# Harness versions used by the corpus runs

`harness/config`, `harness/meter` and `harness/pyproject.toml` are the metering harness as it stood at commit `e14783d` (25 Oct 2026), the last version used in version 1.0's campaign. `harness/replication/` holds the harness of the replication (see below). Each run in the corpus records the commit it ran under (`harness_commit` in `data/store_tables/cells.parquet`). To rebuild an earlier version, apply its patch to this snapshot from inside `harness/`, for example `git apply runtime_patches/416f489.patch`.

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


## The replication (9 to 14 November 2026)

The replication's 1,610 runs used 12 later versions, listed below with what each changed. Runs are counted from `harness_commit` in `data/store_tables/cells.parquet`; dates are the commit times (UTC). The code of the last of them is in `harness/replication/` (see the end of this file). Each replication run's own settings are in its raw record (`cell.json`: the run's settings, the protocol and the server's full launch command), and `data/server_settings.csv` gathers the server settings of every run.

| Commit | Date (UTC) | Runs | Change |
|---|---|---|---|
| `adb8baf` | 2026-11-09 09:44 | 19 | Replication preparation: harness fixes, H200, Gemma-3-27B and gpt-oss-120b, the replication matrix |
| `20f8dd6` | 2026-11-09 13:44 | 189 | Prefill and cached-prefix prompts for the replication cells (dispatch on phase) |
| `0dc03ee` | 2026-11-09 14:13 | 88 | Load generator: no minimum-tokens setting (vLLM rejects it under speculative decoding) |
| `57353cb` | 2026-11-09 14:42 | 302 | The log-probability probe cell and the speculation and precision detectors |
| `148897f` | 2026-11-09 16:55 | 521 | Run sheet change only (no change to measurement) |
| `b391aa8` | 2026-11-10 10:40 | 262 | Llama-3.3-70B nvfp4 cells at a memory offer of 0.7 (the 0.8 servers ran out of memory) |
| `f6f6d2b` | 2026-11-11 14:48 | 108 | Plans for the rerun of runs that lost their cache and for the Qwen3-235B-A22B size runs |
| `cf6a53a` | 2026-11-12 15:59 | 31 | Decode beyond a target batch of 256; server-side latency pages |
| `1b4d07b` | 2026-11-13 13:29 | 40 | Rerun of the simulated agent (c8) on the corrected replay script |
| `a5e69c3` | 2026-11-14 10:48 | 42 | Model weights downloaded before the server starts, timed and logged |
| `3456750` | 2026-11-14 14:26 | 4 | Simulated agent: optional warm-up replay; DeepSeek-V2-Lite B200 rerun plan |
| `431d47a` | 2026-11-14 17:19 | 4 | Simulated agent: DeepSeek-V2-Lite H100 warm-up rerun plan |

Between the periods the meter's code (`meter/power.py`) was restructured: the NVIDIA reading moved into an `NvmlSource` class. The measurement is the same in every version: each window's energy is the difference of the board's cumulative energy counter between the window's start and end, summed over the GPUs, with 0.1 s power samples kept alongside. Every window in the corpus took its energy from that counter (`energy_sources` in `cells.parquet`).

### The replication snapshot

`harness/replication/` holds the harness as it stood at the last of these versions (commit `431d47a`): the `meter` package, the settings files in `config/` and `pyproject.toml`. It is the code that ran, with these omissions and edits:

- **Left out, because no run in the corpus used them:** the modules and settings files for accelerators this paper does not report, for endpoint tests that are not part of this paper, and for the harness's test server; the pod bootstrap script; the harness's test suite.
- **Removed from `config/serving.yaml`:** the entries for those accelerators.
- **Comments and descriptions:** those that discussed that other work were removed, and a few were reworded.
- **Message strings:** three were reworded (an option's help text, a plan label and a note). No other statement of code was changed.

The code paths that would call the omitted modules remain in the shared files, and no run in the corpus takes them. The snapshot's dry-run planner (`python -m meter.campaign --mode campaign --dry-run ...`) gives the same run plans as the unedited code for the replication's settings files. The 11 earlier replication versions are described in the table above and are not rebuilt by patch.

## The ML.ENERGY check (`meter/mlenergy.py`)

This module belongs to a check made before the measurements. It is not a result of the paper, and no figure in the paper depends on it. The same file is in `harness/replication/meter/`.

In pilot mode the harness made one extra Qwen3-8B run set up like the conversation task of the ML.ENERGY Benchmark, version 3 (16-bit weights, target batch 16, 600 output tokens). It compared the run's energy per output token with that benchmark's published figure for the same model on one H100. The module fetches the figure from the benchmark's dataset on Hugging Face (`ml-energy/benchmark-v3`), which needs an account that has accepted the dataset's conditions. The check passes if the two figures agree within 25%.

Four pilots ran on 23 October 2026, the day before the first run in the corpus, each on one H100. The check was evaluated on one of them: 0.252 J per output token at an achieved batch of 15.7, against the benchmark's 0.259 J at a batch setting of 16, a difference of 2.7%. On the other three it was not evaluated: two had no reference figure, and on one the extra run did not run. It was not evaluated on H200 or B200, and no run of the replication made the check.

The four pilots are not in the corpus, and their records are not in the Zenodo deposit.
