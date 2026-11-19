**Weight precision on B200 (experiment c7): gross board energy per output token with nvfp4 (4-bit) weights over fp8 (8-bit), at the same design batch and cached context (200 fresh input, 256 output tokens). Series 'c7' is the experiment of version 1.0 with its later repeats; series 'ladder' is the replication's wider grid of five batches and two contexts. Each series compares its own fp8 and nvfp4 runs. Memory offer: the fraction of the board's memory each server was allowed; where the two sides differ, or where the harness cut the batch differently for the two servers, the row compares different batches (see the achieved B). The nvfp4 runs use 4-bit versions of the models made by third parties, so every row describes these two versions, not 4-bit weights in general. Brackets: the range of the per-pod ratios where both sides ran on the same pods, otherwise the lowest to the highest combination of the two sides' bands; throughput ratio is nvfp4 over fp8 tokens per second, mean over runs, no band.**

| Model | Acc. | Series | Design B | Cached context (tokens) | Memory offer fp8 ; nvfp4 | Achieved B fp8 ; nvfp4 | fp8, J per output token | nvfp4, J per output token | nvfp4 / fp8 | Throughput ratio | Pods nvfp4 / fp8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.3-70B | B200 | c7 | 8 | 1,024 | 0.9 ; 0.7 | 7.9 ; 7.7 | 1.97 [1.91, 2.02] | 1.42 [1.37, 1.54] | 0.724 [0.675, 0.808] | 1.65 | 2 / 2 |
| Llama-3.3-70B | B200 | c7 | 8 | 16,384 | 0.9 ; 0.7 | 7.8 ; 7.5 | 2.41 [2.28, 2.54] | 1.85 [1.79, 1.97] | 0.765 [0.703, 0.861] | 1.53 | 2 / 2 |
| Llama-3.3-70B | B200 | c7 | 64 | 1,024 | 0.9 ; 0.7 | 60.7 ; 57.8 | 0.382 [0.368, 0.396] | 0.259 [0.249, 0.278] | 0.678 [0.628, 0.757] | 1.58 | 2 / 2 |
| Llama-3.3-70B | B200 | ladder | 1 | 1,024 | 0.8 ; 0.7 | 1.0 ; 1.0 | 14.0 [13.8, 14.1] | 9.671 [9.668, 9.674] | 0.692 [0.685, 0.7] | 1.67 | 1 / 2 |
| Llama-3.3-70B | B200 | ladder | 1 | 16,384 | 0.8 ; 0.7 | 1.0 ; 1.0 | 14.6 [14.3, 14.8] | 10.16 [10.16, 10.17] | 0.698 [0.686, 0.71] | 1.65 | 1 / 2 |
| Llama-3.3-70B | B200 | ladder | 8 | 1,024 | 0.8 ; 0.7 | 7.9 ; 7.9 | 1.88 [1.87, 1.89] | 1.35 [1.35, 1.36] | 0.722 [0.716, 0.727] | 1.74 | 1 / 2 |
| Llama-3.3-70B | B200 | ladder | 8 | 16,384 | 0.8 ; 0.7 | 7.7 ; 7.9 | 2.356 [2.355, 2.356] | 1.76 [1.75, 1.76] | 0.745 [0.745, 0.746] | 1.66 | 1 / 2 |
| Llama-3.3-70B | B200 | ladder | 32 | 1,024 | 0.8 ; 0.7 | 30.6 ; 31.2 | 0.593 [0.59, 0.596] | 0.392 [0.391, 0.392] | 0.661 [0.657, 0.665] | 1.72 | 1 / 2 |
| Llama-3.3-70B | B200 | ladder | 32 | 16,384 | 0.8 ; 0.7 | 22.8 ; 23.9 | 1.2 (n=1) | 0.984 [0.983, 0.985] | 0.82 [0.819, 0.821] | 1.43 | 1 / 1 |
| Llama-3.3-70B | B200 | ladder | 64 | 1,024 | 0.8 ; 0.7 | 60.4 ; 61.7 | 0.371 [0.368, 0.374] | 0.244 [0.243, 0.245] | 0.658 [0.649, 0.667] | 1.67 | 1 / 2 |
| Llama-3.3-70B | B200 | ladder | 128 | 1,024 | 0.8 ; 0.7 | 119.7 ; 117.7 | 0.275 [0.275, 0.276] | 0.168 [0.168, 0.169] | 0.612 [0.609, 0.614] | 1.64 | 1 / 2 |
| Qwen3-8B | B200 | c7 | 8 | 1,024 | 0.9 ; 0.7/0.8 | 7.7 ; 7.3 | 0.305 [0.294, 0.316] | 0.256 [0.244, 0.279] | 0.838 [0.771, 0.95] | 1.18 | 2 / 2 |
| Qwen3-8B | B200 | c7 | 8 | 16,384 | 0.9 ; 0.7/0.8 | 7.4 ; 7.0 | 0.498 [0.482, 0.513] | 0.45 [0.43, 0.489] | 0.904 [0.838, 1.01] | 1.12 | 2 / 2 |
| Qwen3-8B | B200 | c7 | 64 | 1,024 | 0.9 ; 0.7/0.8 | 53.1 ; 35.2 | 0.0967 [0.0958, 0.0977] | 0.0618 [0.0588, 0.0677] | 0.639 [0.602, 0.707] | 1.09 | 2 / 2 |
| Qwen3-8B | B200 | c7 | 64 | 16,384 | 0.9 ; 0.7/0.8 | 57.8 ; 52.2 | 0.278 [0.271, 0.284] | 0.247 [0.239, 0.262] | 0.889 [0.844, 0.965] | 1.03 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 1 | 1,024 | 0.8 ; 0.8 | 1.0 ; 1.0 | 2.09 [2.05, 2.12] | 1.67 [1.64, 1.7] | 0.801 [0.799, 0.803] | 1.3 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 1 | 16,384 | 0.8 ; 0.8 | 0.9 ; 0.9 | 2.34 [2.3, 2.38] | 1.93 [1.89, 1.96] | 0.824 [0.822, 0.826] | 1.25 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 8 | 1,024 | 0.8 ; 0.8 | 7.7 ; 7.6 | 0.304 [0.298, 0.31] | 0.244 [0.239, 0.249] | 0.802 [0.802, 0.803] | 1.29 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 8 | 16,384 | 0.8 ; 0.8 | 7.4 ; 7.1 | 0.494 [0.491, 0.496] | 0.433 [0.429, 0.437] | 0.877 [0.874, 0.881] | 1.21 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 32 | 1,024 | 0.8 ; 0.8 | 28.5 ; 27.3 | 0.141 [0.14, 0.142] | 0.0839 [0.0819, 0.0859] | 0.595 [0.586, 0.603] | 1.5 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 32 | 16,384 | 0.8 ; 0.8 | 28.7 ; 27.6 | 0.326 [0.313, 0.339] | 0.275 [0.265, 0.286] | 0.845 [0.842, 0.848] | 1.17 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 64 | 1,024 | 0.8 ; 0.8 | 54.1 ; 49.4 | 0.0988 [0.0972, 0.1] | 0.057 [0.0566, 0.0574] | 0.578 [0.564, 0.591] | 1.49 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 64 | 16,384 | 0.8 ; 0.8 | 57.3 ; 57.1 | 0.279 [0.268, 0.29] | 0.241 [0.238, 0.244] | 0.867 [0.842, 0.891] | 1.15 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 128 | 1,024 | 0.8 ; 0.8 | 101.6 ; 79.0 | 0.0788 [0.0783, 0.0793] | 0.0425 [0.042, 0.043] | 0.54 [0.53, 0.549] | 1.5 | 2 / 2 |
| Qwen3-8B | B200 | ladder | 128 | 16,384 | 0.8 ; 0.8 | 95.4 ; 101.3 | 0.256 [0.249, 0.264] | 0.222 [0.221, 0.223] | 0.867 [0.839, 0.895] | 1.15 | 2 / 2 |
