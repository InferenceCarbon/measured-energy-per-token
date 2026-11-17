**Energy per output token at one request at a time and at the highest load each server reached (experiment c1, decode workload, 1k cached context, 256 output tokens; from Table \ref{tab:size_decode_batch}). Fold: the first over the second, as a ratio of central values. The last two columns give the lowest energy per token in the batch series and the achieved batch where it occurred; where that is not the highest load, energy per token had begun to rise again. Brackets: the range over the pods (one pod: the range of the repeats).**

| Model | Acc. | GPUs | One request at a time, J per token | Highest load: target B | achieved B | J per token | Fold | Lowest J per token | at achieved B | Pods at highest load |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-405B | H200 | 4 | 78.5 [78.1, 79.0] | 128 | 123 | 1.764 [1.763, 1.765] | 45 | 1.76 | 123 | 2 |
| Qwen3-235B-A22B | B200 | 2 | 8.35 [8.34, 8.36] | 128 | 120 | 0.476 [0.475, 0.477] | 18 | 0.476 | 120 | 1 |
