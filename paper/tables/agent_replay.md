**Simulated agentic workload (experiment c8): simulated agents of 30 steps (Llama-3.3-70B) or 11 steps (the 32k-context models) run concurrently, each step re-reading the growing context (10,240 tokens, plus 2,048 per step, capped at the context window) and writing 300 tokens, with cache misses at steps 1 and 5 only (Appendix E). Energy is the gross board energy of the whole run; J per step divides it by the steps of all agents, kJ per agent by the agents. Agents is the number in flight as run (8 was resized to 7 for Llama-3.3-70B on B200 and to 5 on two H100s). Wall seconds: mean over runs, no band. Brackets: the range over the two pods.**

| Model | Acc. | TP | Agents | Steps | Context at last step (tokens) | J per step | J per output token | kJ per agent | Wall seconds | Pods |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | B200 | 1 | 1 | 11 | 30,720 | 615 [602, 629] | 2.05 [2.01, 2.1] | 6.77 [6.62, 6.92] | 17.7 | 2 |
| DeepSeek-V2-Lite | B200 | 1 | 8 | 11 | 30,720 | 225 [210, 239] | 0.749 [0.702, 0.797] | 2.47 [2.32, 2.63] | 40.4 | 2 |
| DeepSeek-V2-Lite | H100 | 1 | 1 | 11 | 30,720 | 404.1 [404.1, 404.2] | 1.3471 [1.3469, 1.3474] | 4.45 [4.44, 4.45] | 16.7 | 2 |
| DeepSeek-V2-Lite | H100 | 1 | 8 | 11 | 30,720 | 128.0 [127.8, 128.2] | 0.427 [0.426, 0.427] | 1.408 [1.406, 1.41] | 37.0 | 2 |
| Llama-3.3-70B | B200 | 1 | 1 | 30 | 69,632 | 5,016 [4,766, 5,266] | 16.7 [15.9, 17.6] | 150 [143, 158] | 220 | 2 |
| Llama-3.3-70B | B200 | 1 | 7 | 30 | 69,632 | 1,343 [1,287, 1,399] | 4.48 [4.29, 4.66] | 40.3 [38.6, 42.0] | 366 | 2 |
| Llama-3.3-70B | H100 | 2 | 1 | 30 | 69,632 | 5,965 [5,929, 6,001] | 19.9 [19.8, 20.0] | 179 [178, 180] | 182 | 2 |
| Llama-3.3-70B | H100 | 2 | 5 | 30 | 69,632 | 2,119 [2,110, 2,128] | 7.06 [7.03, 7.09] | 63.6 [63.3, 63.8] | 300 | 2 |
| Qwen3-30B-A3B | B200 | 1 | 1 | 11 | 30,720 | 609 [606, 611] | 2.03 [2.02, 2.04] | 6.69 [6.66, 6.72] | 16.7 | 2 |
| Qwen3-30B-A3B | B200 | 1 | 8 | 11 | 30,720 | 245 [244, 247] | 0.818 [0.813, 0.822] | 2.7 [2.68, 2.71] | 37.4 | 2 |
| Qwen3-30B-A3B | H100 | 1 | 1 | 11 | 30,720 | 685 [681, 690] | 2.28 [2.27, 2.3] | 7.54 [7.49, 7.59] | 27.3 | 2 |
| Qwen3-30B-A3B | H100 | 1 | 8 | 11 | 30,720 | 289 [289, 290] | 0.964 [0.962, 0.966] | 3.18 [3.17, 3.19] | 59.9 | 2 |
| Qwen3-8B | B200 | 1 | 1 | 11 | 30,720 | 918 [915, 922] | 3.06 [3.05, 3.07] | 10.10 [10.06, 10.14] | 18.8 | 2 |
| Qwen3-8B | B200 | 1 | 8 | 11 | 30,720 | 316 [304, 328] | 1.05 [1.01, 1.09] | 3.47 [3.34, 3.61] | 41.4 | 2 |
| Qwen3-8B | H100 | 1 | 1 | 11 | 30,720 | 1,010 [997, 1,023] | 3.37 [3.32, 3.41] | 11.1 [11.0, 11.2] | 28.3 | 2 |
| Qwen3-8B | H100 | 1 | 8 | 11 | 30,720 | 320 [318, 322] | 1.07 [1.06, 1.07] | 3.52 [3.5, 3.54] | 56.8 | 2 |
