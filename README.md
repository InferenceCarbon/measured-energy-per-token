# Measured Energy per Token for Six Open Language Models on NVIDIA H100, H200 and B200

Paper source, analysis code, metering harness and corpus for:

> J. Manktelow, "Measured Energy per Token for Six Open Language Models on NVIDIA H100, H200 and B200, Analyzed by Batch, Context, Precision, Tensor Parallelism, Speculative Decoding and Prefix Caching," InferenceCarbon Ltd, 2026, doi:10.5281/zenodo.24324178.

**Status:** version 1.1 (`CHANGELOG.md`) follows version 1.0.0 of 6 November 2026 and its corrections, version 1.0.1. Cite the Zenodo record, [doi:10.5281/zenodo.24324178](https://doi.org/10.5281/zenodo.24324178), which holds the paper and the deposit; [doi:10.5281/zenodo.24014979](https://doi.org/10.5281/zenodo.24014979) always resolves to the latest version.

We metered the GPU-board energy of NVIDIA H100, H200 and B200 GPUs serving six open-weight models with vLLM, in 2,260 runs over 1,088 configurations on 51 rented pods in 13 data centers, and two larger models in 118 further runs. Every figure is gross board energy, including the idle draw, over a defined window of requests.

## Contents

| Path | What it is |
|---|---|
| `paper/` | LaTeX source (`paper.tex`, `refs.bib`), with the generated tables (`tables/`) and figures (`figs/`). |
| `analysis/build.py` | Regenerates every table and figure, and the numbers quoted in the text (`paper/tables/headline.json`), from the corpus. |
| `analysis/counter_check.py` | The counter check in Appendix E. It needs the raw power samples from the Zenodo deposit. |
| `analysis/latency.py` | The latency table (time to first token against batch). It needs the per-request records from the Zenodo deposit. |
| `analysis/recompute/` | Independent recomputation of every table from the corpus file, sharing no code with `build.py` (used to audit version 1.0.0; `check_summary.py` needs the per-request records for the latency table). |
| `data/` | The corpus: `features.parquet` (one row per run) and `store_tables/cells.parquet` (one row per window), with the cache-reuse check, each server's launch settings, the list of readings left out and the detection results. `data/README.md` is the data dictionary. |
| `harness/` | The metering harness as it ran at the end of version 1.0 (commit `e14783d`), with patches that rebuild each earlier version of that period; the harness of the replication at its last version (`harness/replication/`, with the omissions listed in `harness/RUNTIME.md`); and a list of the replication's versions and what each changed. |
| `tools/` | Review tooling: Markdown and Word exports with tracked changes, and the script that assembles the Zenodo deposit. |

## Rebuild

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
make tables      # every table and figure
make pdf         # paper/paper.pdf (needs tectonic)
make arxiv       # dist/arxiv.tar.gz
```

The rebuilt tables match the committed ones byte for byte.

## Readings left out

Some readings are left out of every table: runs with no usable window (no tokens returned), runs whose cached context was evicted during the window, runs whose server stopped during the window, and readings superseded by a later measurement. None is deleted. `data/readings_left_out.csv` lists each one with its reason, the paper's appendix gives the counts, and `data/README.md` explains how to include or exclude them. `analysis/build.py` skips every row of the corpus that carries a reason in `dropped`, and stops if a reading is left out without one. `tools/audit_v1_1.py` recomputes the main tables from the readings in use alone, sharing no code with `build.py`.

## Data and code availability

The corpus, the raw power samples, the per-run records, the counter qualification and the server logs are deposited on Zenodo: [doi:10.5281/zenodo.24324178](https://doi.org/10.5281/zenodo.24324178). `tools/build_deposit.py` assembles that deposit from the raw results.

## Licensing

Copyright 2026 InferenceCarbon Ltd. Code (`analysis/`, `harness/`, `tools/`): Apache License 2.0 (`LICENSE`). Corpus and deposited data: CC BY-NC 4.0 with express permissions for regulators, standards bodies, public authorities and auditors (`data/LICENSE`). Paper text: CC BY-NC-ND 4.0 in its preprint versions (`paper/LICENSE`). Appendix F of the paper states the full terms.

## Citation

See `CITATION.cff`.
