**Weight precision on B200 (experiment c7): gross board energy per output token with nvfp4 (4-bit) weights over fp8 (8-bit), at matched design target batch and cached context (200 fresh input, 256 output tokens). The fp8 side is c7's own fp8 runs. At 16k context and design target 64 the feasibility rule resized the target differently for the two servers (31 for fp8 at a 90\% memory offer, 44 for nvfp4 at 70\%), so that row compares different batches. The nvfp4 points are eight windows per model from one pod each, using 4-bit versions of the models made by third parties and a smaller memory offer, so every row is indicative. Pairing and brackets as in Table \ref{tab:tp}; throughput ratio is nvfp4 over fp8 tokens per second, mean over runs, no band.**

| Model | Acc. | Design target B | Cached context (tokens) | Target as run fp8 / nvfp4 | Achieved B fp8 / nvfp4 | fp8, J per output token | nvfp4, J per output token | nvfp4 / fp8 | Throughput ratio | Pods nvfp4 / fp8 | Pairing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.3-70B | B200 | 8 | 1,024 | 8 / 8 | 7.9 / 7.8 | 1.97 [1.91, 2.02] | 1.37 [1.36, 1.37] | 0.694 [0.674, 0.717] | 1.66 | 1 / 2 | none |
| Llama-3.3-70B | B200 | 8 | 16,384 | 8 / 8 | 7.8 / 7.5 | 2.41 [2.28, 2.54] | 1.79 [1.78, 1.79] | 0.74 [0.7, 0.785] | 1.52 | 1 / 2 | none |
| Llama-3.3-70B | B200 | 64 | 1,024 | 64 / 64 | 60.7 / 57.9 | 0.382 [0.368, 0.396] | 0.249 [0.247, 0.251] | 0.652 [0.624, 0.682] | 1.62 | 1 / 2 | none |
| Qwen3-8B | B200 | 8 | 1,024 | 8 / 8 | 7.7 / 7.4 | 0.305 [0.294, 0.316] | 0.244 [0.235, 0.252] | 0.799 [0.744, 0.86] | 1.2 | 1 / 2 | none |
| Qwen3-8B | B200 | 8 | 16,384 | 8 / 8 | 7.4 / 7.0 | 0.498 [0.482, 0.513] | 0.43 [0.427, 0.433] | 0.864 [0.833, 0.898] | 1.13 | 1 / 2 | none |
| Qwen3-8B | B200 | 64 | 1,024 | 64 / 64 | 53.1 / 35.1 | 0.0967 [0.0958, 0.0977] | 0.0588 [0.0585, 0.0591] | 0.608 [0.599, 0.617] | 1.12 | 1 / 2 | none |
| Qwen3-8B | B200 | 64 | 16,384 | 64 / 64 | 57.8 / 51.5 | 0.278 [0.271, 0.284] | 0.239 [0.238, 0.24] | 0.863 [0.84, 0.886] | 1.03 | 1 / 2 | none |
