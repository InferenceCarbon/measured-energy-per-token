**Two larger models: loaded-idle board power (experiment c0: model loaded, no requests, default power management, 60 s window), summed over the GPUs serving the model. Mean over runs, with the range over the pods in brackets (one pod: the range of the repeats). Share of rated: W per GPU over 700 W (H100, H200) or 1,000 W (B200).**

| Model | Accelerator | GPUs | Board W, all GPUs | W per GPU | Share of rated | Pods |
|---|---|---|---|---|---|---|
| Llama-3.1-405B | H200 | 4 | 475 [472, 478] | 119 | 17% | 1 |
| Qwen3-235B-A22B | B200 | 2 | 488 [488, 489] | 244 | 24% | 1 |
