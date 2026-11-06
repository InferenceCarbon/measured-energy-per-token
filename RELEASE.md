# Release checklist

The order follows the publication strategy: the repository, the DOI and the preprint go live the same day, and no link in the paper may be dead on that day.

1. **Licensing:** done 6 Nov 2026. `LICENSE` (Apache-2.0), `data/LICENSE` (CC BY-NC 4.0 with express permissions) and `paper/LICENSE` (CC BY-NC-ND 4.0) are in place, and Appendix F of the paper states the terms.
2. **Reserve the DOI:** create a draft Zenodo upload (not the GitHub integration, which only mints a DOI after release) and reserve its DOI.
3. **Put the DOI in the paper:** in the `\thanks` on the title page and in Section 6 (`10.5281/zenodo.XXXXXXX`). Update the `Accessed` date of `harness2026` in `refs.bib`.
4. **Final build:** run `make tables`, `make latency STORE=<raw store>`, `make pdf` and `make arxiv`. Confirm the tables match the committed ones, and that the PDF and the arXiv bundle compile.
5. **Build the deposit:** done 6 Nov 2026 at tag v1.0.0: `dist/zenodo/` holds `corpus.tar.gz`, `raw.tar.gz` (592 MB), `code.tar.gz` and `SHA256SUMS` (6,150 files), for upload to Zenodo record 10.5281/zenodo.24014980. Original instructions: `python tools/build_deposit.py --store <raw store> --out deposit/`. It stops if it finds a credential in any log. Add `code/`, a `git archive` of this repository at the release tag. Upload the deposit, together with `SHA256SUMS`.
6. **Check links:** check every URL in the PDF.
7. **Tag and publish:** tag `v1.0.0`, make this repository public, and publish the Zenodo deposit.
8. **Submit to arXiv:** submit `dist/arxiv.tar.gz`. Suggested categories: cs.DC (primary), cross-listed to cs.LG and cs.PF. A first submission to a category may need an endorsement from an existing arXiv author, so arrange that before release day.
9. **After acceptance:** add the arXiv identifier to `CITATION.cff` and this README. The arXiv number does not go into the paper.

Known items before release:
- Fix the simulated-agent cache-miss fault in the harness used for future campaigns (the original lives in `inferencecarbon-meter`). This repository keeps the code as it ran.
- arXiv compiles with pdflatex. The bundle has been checked with tectonic only.
