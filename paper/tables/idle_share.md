**Share of the decode window's mean board power that the pod drew while idle with the model loaded (the idle windows bracketing each run of experiment c1, 1k cached context). Gross energy per token includes this share. Median, minimum and maximum over the runs of all four models at that target batch; no band, as the quantity is a ratio of two means of the same run.**

| Accelerator | Target B | Idle share of board power, median | min to max |
|---|---|---|---|
| B200 | 1 | 57% | 35% to 68% |
| B200 | 4 | 50% | 34% to 59% |
| B200 | 16 | 40% | 32% to 46% |
| B200 | 64 | 34% | 28% to 41% |
| B200 | 256 | 35% | 26% to 46% |
| H100 | 1 | 39% | 27% to 51% |
| H100 | 4 | 34% | 26% to 47% |
| H100 | 16 | 30% | 23% to 44% |
| H100 | 64 | 24% | 20% to 37% |
| H100 | 256 | 23% | 19% to 30% |
