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

## Version 1.0.1 (corrections)

1. Review the redline (`tools/redline_docx.py` output) and merge the pull request.
2. Done 13 Nov 2026: doi:10.5281/zenodo.24201206 reserved. On Zenodo, open record 10.5281/zenodo.24014980, choose "New version", reserve the new version DOI, and put it in the paper's `\thanks` and Section 6 (the concept DOI 10.5281/zenodo.24014979 keeps resolving to the latest version).
3. `make tables`, `make latency STORE=<raw store>`, `make pdf`; confirm the committed tables match. Pass `PY=` the Python environment built from `requirements.txt` (matplotlib 3.11.2): another matplotlib version redraws the figures a few pixels different in size, and the last digit of one headline value can move. Done 13 Nov 2026: tables, latency table and figures identical to the committed ones; counter check reproduces the median absolute difference of 0.078 %.
4. Tag `v1.0.1`; build the deposit with `tools/build_deposit.py` at the tag: `corpus.tar.gz` (now with `cache_reuse.csv`), `code.tar.gz` and `SHA256SUMS` change; `raw.tar.gz` is unchanged and can be kept from the first version.
5. Upload the new PDF (`MeasuredEnergyperTokenFourOpenLanguageModels.pdf`) and the changed archives to the new Zenodo version; publish; attach the PDF to a GitHub release at the tag.
6. arXiv: once the submission is announced, replace it with the 1.0.1 source bundle (`make arxiv`).
