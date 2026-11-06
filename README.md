# Measured Energy per Token for Four Open Language Models on NVIDIA H100 and B200

Paper source, analysis code, metering harness and corpus for:

> J. Manktelow, "Measured Energy per Token for Four Open Language Models on NVIDIA H100 and B200: Batch, Context, Precision, Tensor Parallelism and Prefix Caching," InferenceCarbon Ltd, 2026. Preprint in preparation.

**Status: not yet published.** This repository is private until the preprint and its Zenodo deposit go live on the same day. Please do not cite or circulate it before then.

We metered the GPU-board energy of NVIDIA H100 and B200 GPUs serving four open-weight models with vLLM, in 917 runs over 472 configurations on 13 rented pods. Every figure is gross board energy, including the idle draw, over a defined window of requests.

## Contents

| Path | What it is |
|---|---|
| `paper/` | LaTeX source (`paper.tex`, `refs.bib`), with the generated tables (`tables/`) and figures (`figs/`). |
| `analysis/build.py` | Regenerates every table and figure, and the numbers quoted in the text (`paper/tables/headline.json`), from the corpus. |
| `analysis/counter_check.py` | The counter check in Appendix E. It needs the raw power samples from the Zenodo deposit. |
| `analysis/latency.py` | The latency table (time to first token against batch). It needs the per-request records from the Zenodo deposit. |
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
