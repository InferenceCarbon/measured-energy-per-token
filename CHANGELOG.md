# Changelog

Versions follow the paper: each version of the paper gets a tag here and a Zenodo version under the same concept DOI.

## 1.0.1 (November 2026)

Corrections to version 1.0.0; the corpus file is unchanged. Zenodo: a new version under the same concept DOI.

- **Latency table (Table 10) and `latency_summary.json`.** `analysis/latency.py` pooled a run's request records twice when the run sat on both a shared campaign branch and its pod's own results branch, so 21 of the 40 rows had doubled request counts and shifted medians (every B200 row, and Qwen3-8B on H100 at a target of 4). It now reads one copy of each run. The abstract and Section 4.2 quote the corrected ranges (0.02 to 0.05 s at one request; 6 to 12 s at 256).
- **In-window cache-reuse check, new (`data/cache_reuse.csv`, Appendix C).** The harness confirmed the prefix cache was warm before each window but never during it. The server's counters show that in 17 of 523 runs with a cached context the context was recomputed on most requests. Eleven decode runs (c1, c2, c7) are excluded as not being the shape their configuration states; the six KV-pressure runs (c9, c9b) are kept and Table 12 prints their reuse. Consequences: Llama-3.3-70B on two H100s has no target-256 row (its highest reported load is 64, 2,300 tokens per second), the four `decode_context` configurations flagged in 1.0.0 as collapsed are explained by eviction and their broken pairs appear with n=1, the Llama-3.3-70B precision row at 16k and batch 64 is gone, the largest run-to-run differences fall from 120 % to 17 %, and the factor table's batch lever moves from a median 20.1 to 20.7. Section 4.10 now names eviction as the collapse mechanism.
- **Table 12 (`kv_pressure`).** The "Idle-drift flag" column was set from any harness flag rather than the idle-drift flag; the four "yes" entries were achieved-batch notes. Corrected, and a "Cache reuse" column added.
- Text: the corpus counts (902 usable runs; 438, 26 and 8 configurations with two, one and no usable runs; 433 in the variability tables), the variability figures, the batch and context factors, the prose figures of Sections 4.2, 4.3, 4.5, 4.8 and 4.10, and the glossary entries for collapse and cache reuse. The deposited factors row for context at batch 64 states which contexts kept their cache.
- `analysis/recompute/`: independent recomputation of every table from the corpus file, without reusing `analysis/build.py`, used to audit version 1.0.0 (it found the latency and idle-drift errors) and rerun on this version.

## 1.0.0 (6 November 2026)

First release. Zenodo deposit doi:10.5281/zenodo.24014980. The paper, its tables and figures, the corpus, the metering harness as it ran and the analysis code, as described in the paper's Section 6 and Appendix F.
