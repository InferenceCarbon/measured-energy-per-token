**Latency against batch (experiment c1, the runs of Table~\ref{tab:decode_batch}). Time to first token is measured by the client from sending a request to its first streamed token, pooled over the two runs' in-window requests (one copy of each run); at a target of 256 it includes waiting for one of the client's 100 connections and for a server slot, so it is a queueing time, not the server's prefill time. The interval between a request's output tokens is the achieved batch over the window's token rate, from the corpus, so it is a mean over the window rather than a per-request measurement.**

| Model | Acc. | Target B | Achieved B | Time to first token, median (s) | 90th percentile (s) | Interval between output tokens (ms) | Requests |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | B200 | 1 | 0.9 | 0.04 | 0.06 | 3.8 | 158 |
| DeepSeek-V2-Lite | B200 | 4 | 3.8 | 0.06 | 0.20 | 5.1 | 356 |
| DeepSeek-V2-Lite | B200 | 16 | 14.8 | 0.09 | 0.33 | 6.5 | 1077 |
| DeepSeek-V2-Lite | B200 | 64 | 54.4 | 0.29 | 0.67 | 8.3 | 3171 |
| DeepSeek-V2-Lite | B200 | 256 | 77.3 | 5.98 | 8.85 | 10.4 | 3826 |
| DeepSeek-V2-Lite | H100 | 1 | 0.9 | 0.05 | 0.06 | 3.7 | 158 |
| DeepSeek-V2-Lite | H100 | 4 | 3.7 | 0.12 | 0.15 | 4.1 | 424 |
| DeepSeek-V2-Lite | H100 | 16 | 14.3 | 0.18 | 0.28 | 4.7 | 1458 |
| DeepSeek-V2-Lite | H100 | 64 | 52.4 | 0.46 | 0.65 | 6.9 | 3608 |
| DeepSeek-V2-Lite | H100 | 256 | 84.0 | 6.00 | 9.24 | 13.5 | 3306 |
| Llama-3.3-70B | B200 | 1 | 1.0 | 0.05 | 0.06 | 20.8 | 158 |
| Llama-3.3-70B | B200 | 4 | 4.0 | 0.10 | 0.12 | 21.8 | 164 |
| Llama-3.3-70B | B200 | 16 | 15.5 | 0.27 | 0.34 | 21.3 | 352 |
| Llama-3.3-70B | B200 | 64 | 61.0 | 1.02 | 1.17 | 27.1 | 1152 |
| Llama-3.3-70B | B200 | 256 | 90.7 | 11.94 | 20.04 | 33.1 | 1714 |
| Llama-3.3-70B | H100 | 1 | 1.0 | 0.05 | 0.05 | 16.4 | 158 |
| Llama-3.3-70B | H100 | 4 | 3.9 | 0.10 | 0.14 | 17.3 | 164 |
| Llama-3.3-70B | H100 | 16 | 15.6 | 0.27 | 0.38 | 18.5 | 416 |
| Llama-3.3-70B | H100 | 64 | 60.0 | 0.96 | 1.35 | 26.1 | 1163 |
| Qwen3-30B-A3B | B200 | 1 | 1.0 | 0.03 | 0.04 | 3.7 | 158 |
| Qwen3-30B-A3B | B200 | 4 | 3.8 | 0.04 | 0.16 | 4.9 | 372 |
| Qwen3-30B-A3B | B200 | 16 | 15.2 | 0.08 | 0.26 | 6.2 | 1158 |
| Qwen3-30B-A3B | B200 | 64 | 55.6 | 0.24 | 0.48 | 8.1 | 3280 |
| Qwen3-30B-A3B | B200 | 256 | 79.0 | 6.05 | 9.17 | 11.1 | 3734 |
| Qwen3-30B-A3B | H100 | 1 | 1.0 | 0.04 | 0.04 | 6.6 | 158 |
| Qwen3-30B-A3B | H100 | 4 | 3.9 | 0.07 | 0.09 | 7.4 | 253 |
| Qwen3-30B-A3B | H100 | 16 | 14.8 | 0.09 | 0.16 | 9.5 | 752 |
| Qwen3-30B-A3B | H100 | 64 | 61.0 | 0.22 | 0.47 | 15.1 | 1996 |
| Qwen3-30B-A3B | H100 | 256 | 90.3 | 7.33 | 12.38 | 20.4 | 2538 |
| Qwen3-8B | B200 | 1 | 1.0 | 0.02 | 0.04 | 4.1 | 158 |
| Qwen3-8B | B200 | 4 | 3.8 | 0.04 | 0.07 | 4.1 | 446 |
| Qwen3-8B | B200 | 16 | 14.9 | 0.08 | 0.21 | 5.3 | 1354 |
| Qwen3-8B | B200 | 64 | 55.4 | 0.41 | 0.76 | 7.1 | 3672 |
| Qwen3-8B | B200 | 256 | 67.0 | 5.79 | 8.67 | 8.6 | 4036 |
| Qwen3-8B | H100 | 1 | 1.0 | 0.04 | 0.05 | 6.5 | 158 |
| Qwen3-8B | H100 | 4 | 3.9 | 0.06 | 0.07 | 6.2 | 296 |
| Qwen3-8B | H100 | 16 | 15.4 | 0.07 | 0.13 | 7.0 | 1056 |
| Qwen3-8B | H100 | 64 | 60.1 | 0.22 | 0.46 | 10.3 | 2825 |
| Qwen3-8B | H100 | 256 | 83.8 | 6.36 | 8.94 | 13.4 | 3340 |
