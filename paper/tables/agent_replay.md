**Simulated agentic workload (experiment c8): simulated agents of 30 steps (Llama-3.3-70B) or 11 steps (the 32k-context models) run concurrently, each step re-reading the growing context (10,240 tokens, plus 2,048 per step, capped at the context window) and writing 300 tokens, with a cache miss at step 1 and at every fifth step (Appendix E). Energy is the gross board energy of the whole run; J per step divides it by the steps of all agents, kJ per agent by the agents. Agents is the number in flight as run (8 was resized to 6 for Llama-3.3-70B on B200 and to 4 on two H100s, so that the last step fits the KV cache). The DeepSeek-V2-Lite runs were preceded by an unmetered replay of the script; the others were not (Appendix E). Wall seconds: mean over runs, no band. Brackets: the range over the two pods (one pod: the range of the repeats).**

| Model | Acc. | TP | Agents | Steps | Context at last step (tokens) | J per step | J per output token | kJ per agent | Wall seconds | Pods |
|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | B200 | 1 | 1 | 11 | 30,720 | 627 [626, 628] | 2.09 [2.087, 2.092] | 6.9 [6.89, 6.9] | 16.1 | 1 |
| DeepSeek-V2-Lite | B200 | 1 | 8 | 11 | 30,720 | 239 [238, 239] | 0.795 [0.793, 0.797] | 2.62 [2.62, 2.63] | 34.6 | 1 |
| DeepSeek-V2-Lite | H100 | 1 | 1 | 11 | 30,720 | 425 [419, 431] | 1.42 [1.4, 1.44] | 4.67 [4.61, 4.74] | 17.3 | 1 |
| DeepSeek-V2-Lite | H100 | 1 | 8 | 11 | 30,720 | 154.9 [154.7, 155.1] | 0.516 [0.516, 0.517] | 1.7 [1.7, 1.71] | 39.1 | 1 |
| Llama-3.3-70B | B200 | 1 | 1 | 30 | 69,632 | 5,973 [5,741, 6,204] | 19.9 [19.1, 20.7] | 179 [172, 186] | 247 | 2 |
| Llama-3.3-70B | B200 | 1 | 6 | 30 | 69,632 | 2,361 [2,292, 2,430] | 7.87 [7.64, 8.1] | 70.8 [68.8, 72.9] | 516 | 2 |
| Llama-3.3-70B | H100 | 2 | 1 | 30 | 69,632 | 7,182 [7,164, 7,200] | 23.9 [23.9, 24.0] | 215 [215, 216] | 213 | 2 |
| Llama-3.3-70B | H100 | 2 | 4 | 30 | 69,632 | 3,625 [3,620, 3,631] | 12.08 [12.07, 12.10] | 108.8 [108.6, 108.9] | 395 | 2 |
| Qwen3-30B-A3B | B200 | 1 | 1 | 11 | 30,720 | 642 [635, 649] | 2.14 [2.12, 2.16] | 7.06 [6.98, 7.14] | 16.6 | 2 |
| Qwen3-30B-A3B | B200 | 1 | 8 | 11 | 30,720 | 280 [275, 286] | 0.935 [0.917, 0.953] | 3.09 [3.03, 3.14] | 38.6 | 2 |
| Qwen3-30B-A3B | H100 | 1 | 1 | 11 | 30,720 | 716 [707, 726] | 2.39 [2.36, 2.42] | 7.88 [7.78, 7.98] | 28.9 | 2 |
| Qwen3-30B-A3B | H100 | 1 | 8 | 11 | 30,720 | 329 [328, 329] | 1.1 [1.09, 1.1] | 3.62 [3.61, 3.62] | 66.0 | 2 |
| Qwen3-8B | B200 | 1 | 1 | 11 | 30,720 | 1,021 [1,013, 1,028] | 3.4 [3.38, 3.43] | 11.2 [11.1, 11.3] | 19.3 | 2 |
| Qwen3-8B | B200 | 1 | 8 | 11 | 30,720 | 350 [341, 359] | 1.17 [1.14, 1.2] | 3.85 [3.75, 3.95] | 42.0 | 2 |
| Qwen3-8B | H100 | 1 | 1 | 11 | 30,720 | 1,024 [1,018, 1,031] | 3.41 [3.39, 3.44] | 11.3 [11.2, 11.3] | 28.4 | 2 |
| Qwen3-8B | H100 | 1 | 8 | 11 | 30,720 | 357.8 [357.5, 358.2] | 1.193 [1.192, 1.194] | 3.94 [3.93, 3.94] | 62.5 | 2 |
