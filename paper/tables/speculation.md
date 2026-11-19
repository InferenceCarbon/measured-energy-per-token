**Speculative decoding (experiment c10): gross board energy per output token of the decode workload (1k cached context, 200 fresh input, 256 output tokens) with speculative decoding by vLLM's n-gram method, guessing up to three tokens, over the same server without it. A ratio above 1 means speculation cost energy. Throughput ratio: tokens per second, speculative over plain, mean over runs, no band. Brackets: the range of the per-pod ratios. Pairing 'pod': the ratio is formed within each pod; 'none': the two sides ran on different pods, and the band runs from the lowest to the highest combination.**

| Model | Acc. | TP | Target B | Plain, J per output token | Speculative, J per output token | Speculative / plain | Throughput ratio | Pods | Pairing |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | B200 | 1 | 1 | 1.52 [1.52, 1.53] | 2.61 [2.12, 3.1] | 1.71 [1.39, 2.02] | 0.576 | 2 / 2 | pod |
| DeepSeek-V2-Lite | B200 | 1 | 8 | 0.401 [0.4, 0.403] | 1.56 [1.09, 2.03] | 3.9 [2.71, 5.08] | 0.179 | 2 / 2 | pod |
| DeepSeek-V2-Lite | B200 | 1 | 32 | 0.165 [0.164, 0.165] | 1.4 [1.05, 1.75] | 8.5 [6.34, 10.7] | 0.0624 | 2 / 2 | pod |
| DeepSeek-V2-Lite | H100 | 1 | 1 | 1.04 [1.04, 1.05] | 0.807 [0.755, 0.86] | 0.775 [0.729, 0.821] | 0.997 | 2 / 2 | pod |
| DeepSeek-V2-Lite | H100 | 1 | 8 | 0.163 [0.162, 0.163] | 0.761 [0.747, 0.776] | 4.68 [4.6, 4.75] | 0.16 | 2 / 2 | pod |
| DeepSeek-V2-Lite | H100 | 1 | 32 | 0.062 [0.0614, 0.0627] | 0.521 [0.512, 0.529] | 8.39 [8.17, 8.62] | 0.0925 | 2 / 2 | pod |
| Llama-3.3-70B | B200 | 1 | 1 | 14.6 [14.0, 15.2] | 13.0 [12.8, 13.2] | 0.887 [0.866, 0.909] | 1.09 | 2 / 2 | pod |
| Llama-3.3-70B | B200 | 1 | 8 | 2.02 [2.01, 2.03] | 2.03 [1.97, 2.09] | 1.01 [0.982, 1.03] | 0.964 | 2 / 2 | pod |
| Llama-3.3-70B | B200 | 1 | 32 | 0.608 [0.591, 0.625] | 0.904 [0.894, 0.915] | 1.49 [1.46, 1.51] | 0.68 | 2 / 2 | pod |
| Llama-3.3-70B | H100 | 2 | 1 | 15.4 [15.3, 15.4] | 14.2 [13.7, 14.8] | 0.924 [0.885, 0.963] | 1 | 2 / 2 | pod |
| Llama-3.3-70B | H100 | 2 | 8 | 2.219 [2.218, 2.221] | 2.07 [2.04, 2.09] | 0.932 [0.92, 0.943] | 0.99 | 2 / 2 | pod |
| Llama-3.3-70B | H100 | 2 | 32 | 0.775 [0.771, 0.779] | 0.736 [0.726, 0.745] | 0.949 [0.941, 0.957] | 0.995 | 2 / 2 | pod |
| Qwen3-30B-A3B | B200 | 1 | 1 | 1.46 [1.44, 1.47] | 1.4 [1.35, 1.45] | 0.963 [0.92, 1] | 1.08 | 2 / 2 | pod |
| Qwen3-30B-A3B | B200 | 1 | 8 | 0.359 [0.352, 0.365] | 0.422 [0.418, 0.426] | 1.18 [1.14, 1.21] | 0.888 | 2 / 2 | pod |
| Qwen3-30B-A3B | B200 | 1 | 32 | 0.152 [0.15, 0.153] | 0.231 [0.229, 0.232] | 1.52 [1.5, 1.54] | 0.721 | 2 / 2 | pod |
| Qwen3-30B-A3B | H100 | 1 | 1 | 1.64 [1.62, 1.65] | 1.72 [1.65, 1.79] | 1.05 [1.02, 1.08] | 1.08 | 2 / 2 | pod |
| Qwen3-30B-A3B | H100 | 1 | 8 | 0.38 [0.377, 0.382] | 0.396 [0.386, 0.406] | 1.04 [1.01, 1.08] | 1.06 | 2 / 2 | pod |
| Qwen3-30B-A3B | H100 | 1 | 32 | 0.189 [0.185, 0.193] | 0.185 [0.181, 0.189] | 0.979 [0.978, 0.979] | 1.08 | 2 / 2 | pod |
| Qwen3-8B | B200 | 1 | 1 | 2.08 [2.06, 2.1] | 1.8 [1.68, 1.93] | 0.867 [0.801, 0.934] | 1.09 | 2 / 2 | none |
| Qwen3-8B | B200 | 1 | 8 | 0.304 [0.3, 0.308] | 0.375 [0.367, 0.384] | 1.23 [1.19, 1.28] | 0.83 | 2 / 2 | none |
| Qwen3-8B | B200 | 1 | 32 | 0.135 [0.134, 0.135] | 0.182 [0.179, 0.184] | 1.35 [1.33, 1.37] | 0.748 | 2 / 2 | none |
| Qwen3-8B | H100 | 1 | 1 | 2.59 [2.54, 2.64] | 1.81 [1.77, 1.84] | 0.698 [0.669, 0.727] | 1.35 | 2 / 2 | pod |
| Qwen3-8B | H100 | 1 | 8 | 0.364 [0.359, 0.37] | 0.281 [0.278, 0.283] | 0.77 [0.766, 0.774] | 1.11 | 2 / 2 | pod |
| Qwen3-8B | H100 | 1 | 32 | 0.1279 [0.1275, 0.1282] | 0.105 [0.104, 0.106] | 0.821 [0.817, 0.826] | 1.05 | 2 / 2 | pod |
