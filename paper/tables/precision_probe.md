**Precision probe (experiment c11): the same 200 prompts sent to servers holding fp8 and nvfp4 weights of the same model on B200, and fp8 weights on H100 as a control (same format, other hardware). Replies were 64 tokens, always taking the most probable token, with the 20 most probable candidates and their log-probabilities recorded at each position. A threshold on any one of five features of a single reply (the gap between the two most probable tokens, the spread of probability over the candidates, the share of distinct values, the share of exact zeros and the mean log-probability), set on one model and tested on the other, labeled 46\% to 58\% of the 800 B200 replies correctly, where guessing gives 50\%.**

| Model | Pair | Same top token, share of positions | Identical replies | First differing token, mean position | Mean absolute change in log-probability |
|---|---|---|---|---|---|
| Llama-3.3-70B | fp8 against nvfp4, both on B200 | 17% | 0% | 8.1 | 4.47 |
| Qwen3-8B | fp8 against nvfp4, both on B200 | 5% | 0% | 1.4 | 6.40 |
| Llama-3.3-70B | fp8 on H100 against fp8 on B200 (control) | 24% | 1% | 12.5 | 4.37 |
| Qwen3-8B | fp8 on H100 against fp8 on B200 (control) | 34% | 12% | 20.6 | 2.21 |
