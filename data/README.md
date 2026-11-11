# Data dictionary

Two tables hold the corpus. Both are frozen: they are the files the paper was built from, unchanged. Where a field's meaning is narrower than its name suggests, this file says so.

- `features.parquet`: one row per run (917 rows). The analysis script reads this file.
- `store_tables/cells.parquet`: one row per metered window (2,841 rows), with three windows per run: `idle_before`, `active` and `idle_after`. This table holds the per-window telemetry and the server's counters.
- `cache_reuse.csv` (added in version 1.0.1): one row per run with a cached context (529 rows): the in-window prefix-cache reuse the run achieved as a fraction of the expected (`cache_reuse_r`, blank for six early runs whose counters were not recorded), with the cached and fresh token counts used. The analysis script reads it to exclude decode runs below 0.95 (Appendix C of the paper); `features.parquet` itself is unchanged from version 1.0.0.

The raw power samples, per-run records, counter qualification and server logs are in the Zenodo deposit, under `raw/`. They are not in this repository.

Energy is gross GPU-board energy: the board's cumulative energy counter at the end of the window minus its value at the start, summed over the GPUs serving the model. Nothing is subtracted.

## features.parquet

| Field | Meaning |
|---|---|
| `run_id` | `accelerator/model/tpN/experiment/variant/rN`. The variant segment encodes the varied settings (target batch, cached context, context cap, fresh input, prefix length, control, memory offer). |
| `model`, `accelerator`, `n_gpus`, `tp` | Model, GPU type, GPUs serving the model, tensor-parallel degree. |
| `weight_precision`, `kv_precision` | Number formats of the weights and the KV cache (`fp8`, `nvfp4`). |
| `cell`, `repeat` | Experiment label (c0 to c9b) and repeat number. |
| `pod_id`, `data_center` | Marketplace pod identifier and data center code. These are not hardware serial numbers. |
| `flags`, `n_flags` | Quality flags raised by the harness, such as idle drift (the idle power before and after the window differed by more than 5%). |
| `energy_j`, `duration_s`, `mean_power_w` | Gross board energy of the active window (J), its length (s), and the mean board power the harness computed from the power samples (W). `mean_power_w` differs from `energy_j / duration_s` by up to 0.6 W; the paper's energy figures use `energy_j`. |
| `idle_power_w` | Mean board power over the run's two bracketing 60 s idle windows (W). |
| `loaded_idle_power_w` | Experiment c0 only: board power with the model loaded and no requests (W). |
| `arch`, `attention` | Dense or MoE, and the attention type. |
| `phase` | `decode`, `prefill`, `prefix_access`, `replay`, `idle_loaded`, or `decode_dropped` (a decode window with no output tokens). |
| `B` | Decode and cached-prefix runs: the achieved batch, the server's mean count of running requests over the window. Prefill runs: the target batch. |
| `L` | Decode: attended context per output token, cached prefix + fresh input + half the output (tokens). Prefill: the fresh prompt length. |
| `completion_tokens`, `prompt_tokens` | Output and prompt tokens of the requests in the window, from the server's usage report. |
| `e_d_j_per_tok` | Decode workload: `energy_j / completion_tokens` (J per output token). |
| `e_d_idle_sub_j_per_tok` | Diagnostic only: the same with `idle_power_w × duration_s` subtracted. The paper does not use it. |
| `e_p_j_per_tok` | Prefill workload: `energy_j / prompt_tokens` (J per prompt token). |
| `e_per_request_j` | Cached-prefix workload: `energy_j / n_requests` (J per request). |
| `tokens_per_s` | Decode: `completion_tokens / duration_s`. |
| `control`, `prefix_cached`, `fresh_in`, `out`, `n_requests` | Cached-prefix workload: cached or uncached side, prefix length, fresh input, output budget, and requests completed. |
| `cached_tokens` | **Not a measurement.** The server did not report per-request cached tokens. The harness stored a missing value as 0, so this field is 0 or empty everywhere. To see cache reuse, use the server counters in `cells.parquet`, with the caveats given there. |
| `q_W`, `q_KV`, `tau`, `W_batch_bytes`, `K_bytes`, `union_source` | Model-based predictors kept for later modelling: weight bytes per output token at the achieved batch, KV bytes attended per output token, seconds per output token, and their inputs. For MoE models the weight bytes use a routing estimate (`union_source`). These are computed from the model architecture, not measured, and the paper does not use them. |
| `dropped` | The reason a run was excluded, if it was. Empty for usable runs. In the file, four runs carry a reason (no usable window); the analysis script adds the eleven cache-eviction exclusions of version 1.0.1 at load time from `cache_reuse.csv`, so the frozen file is unchanged. |

## store_tables/cells.parquet

| Field | Meaning |
|---|---|
| `run_id`, `window_id`, `kind` | Run, window, and window type (`idle_before`, `active`, `idle_after`). |
| `duration_s`, `energy_j_total`, `mean_power_w_total`, `n_gpus` | Window length, gross board energy summed over the GPUs, mean power. |
| `energy_sources` | The meter used (the NVML cumulative energy counter for every window in the paper). |
| `per_gpu_json` | Per GPU: energy, start and end times, sample count, power limit, clocks, temperature and throttle reasons at the window edges. |
| `variant_json`, `fixed_json` | The varied and fixed settings of the experiment. |
| `harness_commit` | The harness version the run used. `harness/RUNTIME.md` lists all 13. |
| `vllm_version` | Serving software version (0.15.1 in every window that recorded it). |
| `warm_confirmed` | Whether the prefix was confirmed cached before the window, from the server's prefix-cache counters. |
| `n_requests`, `n_errors` | Requests completed and failed in the window. |
| `prompt_tokens`, `completion_tokens` | From the server's per-request usage reports, summed. |
| `cached_tokens` | **Not a measurement.** See `features.parquet` above. |
| `client_mean_concurrency`, `target_batch` | Client-side requests in flight (mean) and the offered concurrency. |
| `sched_running_*`, `sched_waiting_mean`, `sched_kv_usage_mean` | The server's scheduler counters, sampled every 0.5 s: running requests, waiting requests, and the fraction of the KV-cache pool in use. |
| `sched_server_prompt_tokens_delta`, `sched_server_generation_tokens_delta` | The server's own prompt and output token counters over the window. |
| `sched_prefix_queries_delta`, `sched_prefix_hits_delta` | The server's prefix-cache lookup counters over the window: tokens looked up and tokens found. **These count lookups, not tokens processed.** In most windows the queries equal the prompt tokens. Under memory pressure (experiments c9 and c9b) they can be several times larger, because the scheduler looks up the same prompt again after preempting it. `prompt_tokens − hits` is therefore not a measured count of tokens recomputed. |
| `sched_preemptions_delta`, `sched_preemptions_counter_present` | Preemptions over the window, recorded in experiment c9b only. |
| `sched_active_batch`, `sched_idle_fraction` | Requests advanced per step, and the share of samples with no running request. |
| `replay_script` | Simulated agent (c8): the scripted steps. Its `cache_expected` labels show the *intended* schedule, a miss every fifth step. A harness fault meant only steps 1 and 5 missed. The prefix-cache counters confirm the executed schedule (paper, Appendix E). |
| `flags`, `n_flags`, `pod_id`, `data_center`, `model`, `accelerator`, `tp`, `cell`, `repeat` | As in `features.parquet`. |

## Checksums

```
1a186b929cf09b60da5dec06020fa6a0c1498ae142ecdaffbf765d031981abaf  data/features.parquet
4b11fa45d691bc53a20204c8a3983eca4e758faaed49053ade908747043d675a  data/store_tables/cells.parquet
```
