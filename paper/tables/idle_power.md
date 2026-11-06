**Loaded-idle board power (experiment c0: model loaded, no requests, default power management, 60 s window), summed over the GPUs serving the model. Mean over runs, with the range over the two pods in brackets. Share of rated: W per GPU over 700 W (H100) or 1,000 W (B200).**

| Model | Accelerator | GPUs | Board W, all GPUs | W per GPU | Share of rated | Pods |
|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite | B200 | 1 | 245 [239, 250] | 245 | 24% | 2 |
| DeepSeek-V2-Lite | H100 | 1 | 127 [126, 129] | 127 | 18% | 2 |
| Llama-3.3-70B | B200 | 1 | 244 [239, 249] | 244 | 24% | 2 |
| Llama-3.3-70B | H100 | 2 | 259 [249, 270] | 130 | 19% | 2 |
| Qwen3-30B-A3B | B200 | 1 | 245 [239, 250] | 245 | 24% | 2 |
| Qwen3-30B-A3B | H100 | 1 | 123 [121, 125] | 123 | 18% | 2 |
| Qwen3-8B | B200 | 1 | 244 [239, 249] | 244 | 24% | 2 |
| Qwen3-8B | H100 | 1 | 124 [122, 126] | 124 | 18% | 2 |
