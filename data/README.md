# Data dictionary

Two tables hold the corpus. Both are frozen: they are the files the paper was built from, unchanged. Where a field's meaning is narrower than its name suggests, this file says so.

Version 1.1 holds the 917 readings of version 1.0.0 and 1,610 readings of the replication (2,527 in all). A reading is one metered run. Run ids repeat between the two sources, so `uid` is the key.

- `features.parquet`: one row per reading (2,527 rows). The analysis script reads this file.
- `store_tables/cells.parquet`: one row per metered window (7,671 rows), with three windows per reading: `idle_before`, `active` and `idle_after`. This table holds the per-window telemetry and the server's counters.
- `cache_reuse.csv`: one row per reading with a cached context or prefix (1,559 rows): the in-window prefix-cache reuse the run achieved as a fraction of the expected (`cache_reuse_r`, blank for six early runs whose counters were not recorded), with the cached and fresh token counts used.
- `server_settings.csv`: one row per reading (2,527 rows): the server's own launch settings, read from each run's raw record by `tools/server_settings.py` (checkpoint, tensor-parallel size, memory offer, maximum context, request limit, KV-cache precision, prefix caching, speculative decoding, vLLM version). The analysis script takes each server's memory offer from this file; a run id names the offer only where it was a varied setting.
- `readings_left_out.csv`: every reading that is in no table of the paper, with its reason (410 rows). See "Readings left out" below.
- `detection/`: results of the two detection analyses (speculative decoding and the precision probe), computed from the per-request records.

## Readings left out

Some readings are left out of every table. None is deleted, and each has a recorded reason.

- 149 are rows of `features.parquet`. Their `disposition` is `excluded-cache`, `excluded-no-window`, `excluded-server-stopped` or `superseded`, and `dropped` holds the reason. The analysis script skips every row with a reason in `dropped`, and stops if a row is left out without one.
- 261 are not in `features.parquet`: copies of a reading already in the corpus, earlier attempts of a run that was run again, pilot runs, and readings superseded by a later measurement. Their records are in the Zenodo deposit under `raw/`. For 46 of them (Llama-3.3-70B nvfp4 at a memory offer of 0.8, whose server ran out of memory before any window ran) no run record exists; the server and campaign logs of their branch record the attempts, and `raw/ATTEMPTS_WITHOUT_RUN_RECORD.csv` in the deposit lists them.

`readings_left_out.csv` lists all 410 with source, results branch, run id, experiment, disposition and reason. `in_features_file` says which of the two groups a reading is in. The paper's appendix table "Readings left out of every table" gives the counts by reason.

To use the corpus without the left-out readings, keep the rows of `features.parquet` whose `dropped` is empty. To study them, keep the others.

The raw power samples, per-run records, counter qualification and server logs are in the Zenodo deposit, under `raw/`. They are not in this repository.

Energy is gross GPU-board energy: the board's cumulative energy counter at the end of the window minus its value at the start, summed over the GPUs serving the model. Nothing is subtracted.

## features.parquet

| Field | Meaning |
|---|---|
| `uid` | Unique key: source (`v1.0.0`) or results branch, a colon, and the run id. |
| `source`, `branch` | `v1.0.0` or `replication`, and the results branch the reading came from (replication only). |
| `experiment`, `series` | The paper's experiment the reading belongs to (c0 to c11; several run cells feed one experiment), and `main` (six models) or `size` (the two large-model runs). |
| `client_cap` | 100 for version 1.0.0 readings: the load client then held at most 100 connections, so a target batch above 100 offered 100 requests. Empty otherwise. |
| `disposition`, `marks` | `used`, `kept-marked` (used, with the quality mark given in `marks`), or one of the left-out dispositions above. |
| `run_id` | `accelerator/model/tpN/experiment/variant/rN`. The variant segment encodes the varied settings (target batch, cached context, context cap, fresh input, prefix length, control, memory offer). |
| `model`, `accelerator`, `n_gpus`, `tp` | Model, GPU type, GPUs serving the model, tensor-parallel degree. |
| `weight_precision`, `kv_precision` | Number formats of the weights and the KV cache (`fp8`, `nvfp4`). |
| `cell`, `repeat` | The cell the harness ran (version 1.0.0: c0 to c9b; replication: the block's cell name) and repeat number. |
| `pod_id`, `data_center` | Marketplace pod identifier and data center code. These are not hardware serial numbers. |
| `flags`, `n_flags` | Quality flags raised by the harness, such as idle drift (the idle power before and after the window differed by more than 5%). |
| `energy_j`, `duration_s`, `mean_power_w` | Gross board energy of the active window (J), its length (s), and the mean board power the harness computed from the power samples (W). `mean_power_w` differs from `energy_j / duration_s` by up to 0.6 W. The paper's energy figures (J per token or per request) use `energy_j`; its Board W and % of rated columns are means of `mean_power_w` over runs. |
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
| `q_W`, `q_KV`, `tau`, `W_batch_bytes`, `K_bytes`, `union_source` | Model-based predictors kept for later modelling: weight bytes per output token at the achieved batch, KV bytes attended per output token, seconds per output token, and their inputs. For MoE models the weight bytes use a routing estimate (`union_source`). These are computed from the model architecture, not measured. The paper uses only `W_batch_bytes` at target batch 1, as the weight bytes read per step in its speed-limit table (Table 11). |
| `dropped` | The reason a reading is left out of the tables. Empty for a reading in use. |

## store_tables/cells.parquet

| Field | Meaning |
|---|---|
| `run_id`, `window_id`, `kind` | Run, window, and window type (`idle_before`, `active`, `idle_after`). |
| `duration_s`, `energy_j_total`, `mean_power_w_total`, `n_gpus` | Window length, gross board energy summed over the GPUs, mean power. |
| `energy_sources` | The meter used (the NVML cumulative energy counter for every window in the paper). |
| `per_gpu_json` | Per GPU: energy, start and end times, sample count, power limit, clocks, temperature and throttle reasons at the window edges. |
| `variant_json`, `fixed_json` | The varied and fixed settings of the experiment. |
| `harness_commit` | The harness version the run used. `harness/RUNTIME.md` lists all 25: 13 in version 1.0 and 12 in the replication. |
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
| `replay_script` | Simulated agent (c8): the scripted steps. Its `cache_expected` labels show the *intended* schedule: a miss at step 1 and every fifth step. In version 1.0 a harness fault meant only steps 1 and 5 missed; the replication ran the script as intended. The prefix-cache counters confirm the executed schedule (paper, Appendix E). |
| `flags`, `n_flags`, `pod_id`, `data_center`, `model`, `accelerator`, `tp`, `cell`, `repeat` | As in `features.parquet`. |

## Checksums

```
c2743c20e9490a78ed8ac5d761e10af0b6fed514f4c6adb59e9e6c694ea9efde  data/features.parquet
6af2e0930b8e70b7b0d68711501622c5020115892f8e115490288186ac8de81f  data/store_tables/cells.parquet
bfb5a5cd0c070de2c19726d7a8bc20d94da75f94c08abcc1894c363865e1472b  data/cache_reuse.csv
fa7dde74a804a53ae2b90deaa4d554836addc20af565fa0332c3e559a2af0eb8  data/server_settings.csv
d7b8f06b873743b3600eaf1b94082c56b6bda66eeea0382d5be501ca7aced835  data/readings_left_out.csv
```
