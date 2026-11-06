PY ?= python
TECTONIC ?= tectonic

.PHONY: tables pdf arxiv check clean

tables:            ## regenerate every table and figure from data/features.parquet
	$(PY) analysis/build.py

pdf: tables        ## compile paper/paper.pdf
	cd paper && $(TECTONIC) -X compile paper.tex

arxiv: tables      ## source bundle for arXiv: dist/arxiv.tar.gz (paper.tex, paper.bbl, tables, figures)
	rm -rf dist/arxiv && mkdir -p dist/arxiv
	cd paper && $(TECTONIC) -X compile --keep-intermediates --outdir ../dist/arxiv paper.tex
	cp paper/paper.tex paper/refs.bib dist/arxiv/ && cp -R paper/tables dist/arxiv/tables && mkdir -p dist/arxiv/figs && cp paper/figs/*.pdf dist/arxiv/figs/
	cd dist/arxiv && rm -f *.aux *.log *.out *.blg paper.pdf tables/*.md tables/*.csv tables/*.json
	tar -C dist/arxiv -czf dist/arxiv.tar.gz .
	@echo "dist/arxiv.tar.gz ready"

latency:           ## latency table from the per-request records: make latency STORE=/path/to/raw
	$(PY) analysis/latency.py --store $(STORE)

check:             ## counter check against the raw samples: make check STORE=/path/to/raw
	$(PY) analysis/counter_check.py --store $(STORE)

clean:
	rm -rf dist paper/paper.pdf paper/paper.md paper/paper.html paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out
