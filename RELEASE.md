# Release checklist

The order follows the publication strategy: the repository, the DOI and the preprint go live the same day, and no link in the paper may be dead on that day.

1. **Licensing:** done 6 Nov 2026. `LICENSE` (Apache-2.0), `data/LICENSE` (CC BY-NC 4.0 with express permissions) and `paper/LICENSE` (CC BY-NC-ND 4.0) are in place, and Appendix F of the paper states the terms.
2. **Reserve the DOI:** create a draft Zenodo upload (not the GitHub integration, which only mints a DOI after release) and reserve its DOI.
3. **Put the DOI in the paper:** in the `\thanks` on the title page and in Section 6 (`10.5281/zenodo.XXXXXXX`). Update the `Accessed` date of `harness2026` in `refs.bib`.
4. **Final build:** run `make tables`, `make latency STORE=<raw store>`, `make pdf` and `make arxiv`. Confirm the tables match the committed ones, and that the PDF and the arXiv bundle compile.
5. **Build the deposit:** done 6 Nov 2026 at tag v1.0.0: `dist/zenodo/` holds `corpus.tar.gz`, `raw.tar.gz` (592 MB), `code.tar.gz` and `SHA256SUMS` (6,150 files), for upload to Zenodo record 10.5281/zenodo.24014980. Original instructions: `python tools/build_deposit.py --store <raw store> --out deposit/`. It stops if it finds a credential in any log. Add `code/`, a `git archive` of this repository at the release tag. Upload the deposit, together with `SHA256SUMS`.
6. **Check links:** check every URL in the PDF. Done 6 Nov: all resolve except the DOI and the repository, which go live at release.
6a. **The paper on Zenodo:** arXiv moderation may take weeks, so the Zenodo record is the paper's first public venue, not only the data's. Add the final PDF (`MeasuredEnergyperTokenFourOpenLanguageModels.pdf`, the build at the tag) as a fifth file in the draft, and attach the same PDF to the GitHub release as an asset. Until arXiv assigns an identifier, the citation is the Zenodo DOI; the README and CITATION.cff already carry it. Record type stays Dataset (the paper describes the data); if Zenodo offers a second resource type, add Publication: Preprint.
7. **Release day, in this order:** make this repository public and create the GitHub release from `v1.0.0` with the PDF attached; publish the Zenodo record and confirm the DOI resolves; submit to arXiv the same day. From this point the paper is public via the DOI even while arXiv moderation runs. (Tag `v1.0.0` already set, 6 Nov.)
8. **Submit to arXiv:** done 6 Nov 2026. Submitted `dist/arxiv.tar.gz` under cs.CY (the category this account is endorsed for; cs.LG was not available), licence CC BY-NC-ND 4.0, with the shortened 1,737-character abstract in `dist/arxiv_abstract.txt`. Submission submit/8341083, status processing at 6 Nov 18:30 UTC; the bundle was the layout-fixed build (commit 130dd51), text identical to v1.0.0. Awaiting moderation. Note: arXiv allows two submissions per calendar month on this account.
9. **After arXiv announces it:** add the arXiv identifier to `CITATION.cff`, the README and the Zenodo record's related works. The arXiv number does not go into the paper. If arXiv asks for changes to the source, that is version 1.0.1: a new tag, a new Zenodo version under the same concept DOI, and the DOI in the paper updated to the new version's.

Known items before release:
- Fix the simulated-agent cache-miss fault in the harness used for future campaigns (the original lives in `inferencecarbon-meter`). This repository keeps the code as it ran.
- arXiv compiles with pdflatex. The bundle has been checked with tectonic only.
