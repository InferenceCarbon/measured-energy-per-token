**Memory-bandwidth bound at one request at a time (experiment c1, target batch 1). Each output token needs one read of the weights the step uses, so the token rate cannot exceed the GPUs' HBM bandwidth (3.35 TB/s per H100, 8 TB/s per B200, NVIDIA's figures) divided by those bytes: the dense models' fp8 weights, or for the MoE models the expert and shared weights the batch touched (measured in the router pass). The bound ignores the KV cache and the fresh-input prefill, so it is generous and the share is a lower bound on how fully the bandwidth was used. Measured: mean over the two runs. Idle share: the loaded-idle draw over the window's mean board power.**

| Model | Acc. | GPUs | Weight bytes per step (GB) | Bound (tokens/s) | Measured (tokens/s) | Share of bound | Idle share of board power |
|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | B200 | 1 | 2.7 | 3,014 | 248 | 8% | 67% |
| DeepSeek-V2-Lite | H100 | 1 | 2.7 | 1,262 | 252 | 20% | 48% |
| Llama-3.3-70B | B200 | 1 | 70.6 | 113 | 48 | 42% | 36% |
| Llama-3.3-70B | H100 | 2 | 70.6 | 95 | 60 | 64% | 27% |
| Qwen3-30B-A3B | B200 | 1 | 3.3 | 2,409 | 266 | 11% | 65% |
| Qwen3-30B-A3B | H100 | 1 | 3.3 | 1,009 | 147 | 15% | 51% |
| Qwen3-8B | B200 | 1 | 8.2 | 976 | 242 | 25% | 48% |
| Qwen3-8B | H100 | 1 | 8.2 | 409 | 149 | 36% | 33% |
