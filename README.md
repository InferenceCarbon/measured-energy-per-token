# Measured Energy per Token for Four Open Language Models on NVIDIA H100 and B200

Paper source, analysis code, metering harness and corpus for:

> J. Manktelow, "Measured Energy per Token for Four Open Language Models on NVIDIA H100 and B200, Analyzed by Batch, Context, Precision, Tensor Parallelism and Prefix Caching," InferenceCarbon Ltd, 2026, doi:10.5281/zenodo.24014980.

**Status:** version 1.0.1 (corrections; `CHANGELOG.md`) follows the release of 6 November 2026 as v1.0.0. Cite the Zenodo record, [doi:10.5281/zenodo.24014980](https://doi.org/10.5281/zenodo.24014980), which holds the paper and the deposit; the arXiv identifier will be added here once assigned. The PDF is attached to the [v1.0.0 release](https://github.com/InferenceCarbon/measured-energy-per-token/releases/tag/v1.0.0). The `COMMIT` file inside the Zenodo code archive names commit 3837fa5, from the repository's history as it stood at upload; that history was condensed afterwards. The archive's contents are identical to tag v1.0.0: both have tree hash `a94b0e54dcb54a4ed110b1fae9bfc2c04d0d115b`, which `git rev-parse v1.0.0^{tree}` reproduces.

We metered the GPU-board energy of NVIDIA H100 and B200 GPUs serving four open-weight models with vLLM, in 917 runs over 472 configurations on 13 rented pods. Every figure is gross board energy, including the idle draw, over a defined window of requests.

## Contents

| Path | What it is |
|---|---|
| `paper/` | LaTeX source (`paper.tex`, `refs.bib`), with the generated tables (`tables/`) and figures (`figs/`). |
| `analysis/build.py` | Regenerates every table and figure, and the numbers quoted in the text (`paper/tables/headline.json`), from the corpus. |
| `analysis/counter_check.py` | The counter check in Appendix E. It needs the raw power samples from the Zenodo deposit. |
| `analysis/latency.py` | The latency table (time to first token against batch). It needs the per-request records from the Zenodo deposit. |
| `analysis/recompute/` | Independent recomputation of every table from the corpus file, sharing no code with `build.py` (used to audit version 1.0.0; `check_summary.py` needs the per-request records for the latency table). |
| `data/` | The corpus: `features.parquet` (one row per run) and `store_tables/cells.parquet` (one row per window). `data/README.md` is the data dictionary. |
| `harness/` | The metering harness as it ran (commit `e14783d`), with patches that rebuild each earlier version used in the campaign (`harness/RUNTIME.md`). |
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

## Data and code availability

The corpus, the raw power samples, the per-run records, the counter qualification and the server logs are deposited on Zenodo: [doi:10.5281/zenodo.24014980](https://doi.org/10.5281/zenodo.24014980). `tools/build_deposit.py` assembles that deposit from the raw results store.

## Licensing

Copyright 2026 InferenceCarbon Ltd. Code (`analysis/`, `harness/`, `tools/`): Apache License 2.0 (`LICENSE`). Corpus and deposited data: CC BY-NC 4.0 with express permissions for regulators, standards bodies, public authorities and auditors (`data/LICENSE`). Paper text: CC BY-NC-ND 4.0 in its preprint versions (`paper/LICENSE`). Appendix F of the paper states the full terms.

## Citation

See `CITATION.cff`.
