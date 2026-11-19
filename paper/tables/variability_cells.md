**Run-to-run variability of the metric each experiment reports (energy per output token, per prompt token or per request; board power for c0). Repeat difference: the difference between the highest and lowest run of one configuration as a percentage of their mean (most configurations have two runs, placed on different pods where we had two). Between-pod difference: the same for the per-pod means, configurations run on two or more pods. Median and 90th percentile (p90) over configurations; n is the number of configurations. Configurations with fewer than two usable runs are excluded (Appendix C preamble). The last two rows split the decode configurations by whether both repeats ran on one pod; the two sets hold different configurations.**

| Experiment | Repeat difference median / p90 | Between-pod difference median / p90 | Configs with 2 pods | Configs on one pod |
|---|---|---|---|---|
| c0_static | 3.9% / 7.5% (n=18) | 3.9% / 7.5% (n=18) | 18 | 0 |
| c10_speculation | 2.7% / 9.7% (n=48) | 2.7% / 9.7% (n=48) | 48 | 0 |
| c1_decode_batch | 2.6% / 8.0% (n=126) | 2.7% / 8.5% (n=112) | 112 | 14 |
| c2_decode_context | 3.1% / 7.6% (n=234) | 3.3% / 7.7% (n=218) | 218 | 16 |
| c3_prefill_miss | 2.7% / 13.7% (n=204) | 2.7% / 13.7% (n=204) | 204 | 0 |
| c4_cached_prefix | 2.2% / 8.1% (n=274) | 2.2% / 8.1% (n=270) | 270 | 4 |
| c6_tensor_parallel | 2.5% / 9.8% (n=24) | 2.4% / 9.6% (n=24) | 24 | 0 |
| c7_precision | 3.2% / 10.4% (n=57) | 3.7% / 10.7% (n=49) | 49 | 8 |
| c7_precision_cached | 2.3% / 3.5% (n=32) | 2.5% / 3.5% (n=24) | 24 | 8 |
| c7_precision_prefill | 1.1% / 10.7% (n=24) | 1.6% / 11.4% (n=18) | 18 | 6 |
| c8_agent_replay | 1.4% / 5.5% (n=16) | 1.8% / 5.8% (n=12) | 12 | 4 |
| c9_kv_pressure | 11.8% / 62.6% (n=6) | 11.3% / 75.5% (n=6) | 6 | 0 |
| c9b_kv_pressure_diag | 11.2% / 36.3% (n=8) | 11.2% / 36.3% (n=8) | 8 | 0 |
| all decode experiments | 2.9% / 9.3% (n=503) | 3.2% / 9.4% (n=465) | 465 | 38 |
| decode, both repeats on one pod | 0.4% / 3.7% (n=38) | -- | 0 | 38 |
| decode, repeats on two pods | 3.2% / 9.5% (n=465) | 3.2% / 9.4% (n=465) | 465 | 0 |
