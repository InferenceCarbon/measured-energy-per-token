**Two larger models: energy per output token of the decode workload against batch (experiment c1: 1k cached context, 200 fresh input tokens, 256 output tokens; gross board energy including the idle draw). Target B is the number of requests the client kept in flight. Achieved B is the server's mean count of running requests over the middle 80\% of the window; at the highest targets the server ran fewer requests than were offered, and the rest waited. Mean KV use: the server's mean KV-cache occupancy over the same part of the window. Board W is the mean board power over the window summed over the GPUs in the TP column, and \% of rated divides it by 700 W per H100 or H200, or 1,000 W per B200. J per output token equals Board W divided by tokens per second to within 1\%; the small difference arises because each column is a mean over runs. Brackets: the range over the pods (one pod: the range of the repeats). TP: tensor-parallel degree (GPUs sharing the model).**

| Model | Acc. | TP | Target B | Achieved B | Mean KV use | Board W | % of rated | Tokens per second | J per output token | Pods |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-405B | H200 | 4 | 1 | 1.0 | 0% | 2,138 [2,134, 2,142] | 76% | 27.2 [27.1, 27.3] | 78.5 [78.1, 79.0] | 2 |
| Llama-3.1-405B | H200 | 4 | 8 | 8.0 | 4% | 2,199 [2,192, 2,205] | 79% | 196.99 [196.99, 197.00] | 11.2 [11.1, 11.2] | 1 |
| Llama-3.1-405B | H200 | 4 | 32 | 31.1 | 14% | 2,404 [2,401, 2,406] | 86% | 593.0 [592.9, 593.2] | 4.05 [4.05, 4.06] | 2 |
| Llama-3.1-405B | H200 | 4 | 64 | 60.7 | 27% | 2,665 [2,656, 2,673] | 95% | 1,110 [1,106, 1,114] | 2.4 [2.38, 2.42] | 1 |
| Llama-3.1-405B | H200 | 4 | 128 | 123 | 53% | 2,716 [2,703, 2,729] | 97% | 1,540 [1,533, 1,546] | 1.764 [1.763, 1.765] | 2 |
| Qwen3-235B-A22B | B200 | 2 | 1 | 1.0 | 0% | 824 [823, 824] | 41% | 98.6 [98.4, 98.8] | 8.35 [8.34, 8.36] | 1 |
| Qwen3-235B-A22B | B200 | 2 | 8 | 7.9 | 1% | 1,233 [1,226, 1,240] | 62% | 534 [531, 536] | 2.31 [2.29, 2.34] | 1 |
| Qwen3-235B-A22B | B200 | 2 | 32 | 31.6 | 5% | 1,513 [1,511, 1,514] | 76% | 1,555 [1,552, 1,558] | 0.973 [0.97, 0.975] | 1 |
| Qwen3-235B-A22B | B200 | 2 | 64 | 62.6 | 9% | 1,650 [1,649, 1,651] | 82% | 2,574 [2,567, 2,580] | 0.641 [0.639, 0.643] | 1 |
| Qwen3-235B-A22B | B200 | 2 | 128 | 120 | 18% | 1,718 [1,715, 1,720] | 86% | 3,608 [3,596, 3,619] | 0.476 [0.475, 0.477] | 1 |
