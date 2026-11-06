"""Reading copy of Paper 1 without a LaTeX installation.

    python tools/render.py

paper.tex -> paper.md (Markdown export, tables and figures inlined)
          -> paper.html (-> paper_reading.pdf with --pdf, via wkhtmltopdf)

paper.pdf is the compiled LaTeX (tectonic -X compile paper.tex) and is not
written by this script.

The converter understands the subset of LaTeX the paper uses: sections,
paragraphs, emphasis, \\cite, \\ref, \\input of the generated tables (their
Markdown twins are used), figure environments, simple tabular, itemize, the
abstract and the bibliography (rendered from refs.bib). It is a reading
copy, not the arXiv build; the arXiv build is pdflatex + bibtex on paper.tex.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "paper"
TEX = HERE / "paper.tex"
MD = HERE / "paper.md"
HTML = HERE / "paper.html"
PDF = HERE / "paper_reading.pdf"


FIRST_AUTHOR: dict[str, str] = {}


def _sentence_case(t: str) -> str:
    out, depth, cap = [], 0, True
    for i, c in enumerate(t):
        if c == "{":
            if depth == 0:
                cap = False      # a braced group in first position is kept as written and consumes the capital
            depth += 1
            continue
        if c == "}":
            depth -= 1
            continue
        if depth > 0:
            out.append(c)
        elif c.isalpha():
            out.append(c if cap else c.lower())   # BibTeX keeps, never raises, the first letter
            cap = False
        else:
            out.append(c)
        if c == ":" and depth == 0 and t[i + 1:i + 2].isspace():
            cap = True
    return "".join(out)


def bib_entries() -> dict[str, str]:
    """IEEE reference-list format (James, 31 Oct 2026): A. B. Author, "Title," *Venue*, vol., no., pp., year.
    doi / [Online]. Available: url. Accessed: date. Numbering is by first citation (see cite_num)."""
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", (HERE / "refs.bib").read_text(), re.S):
        kind, key, body = m.group(1), m.group(2), m.group(3)
        f = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S))
        if "title" in f:
            f["title"] = _sentence_case(f["title"])
        f = {k: re.sub(r"[{}]", "", v).replace("\\url", "").replace("\\texttt", "").replace("\\'", "").replace("\\~", "")
             .replace("--", "–").replace("\n", " ") for k, v in f.items()}
        names = []
        for a in f.get("author", "").split(" and "):
            a = a.strip()
            if a == "others":
                names.append("et al.")
            elif "," in a:
                last, first = [x.strip() for x in a.split(",", 1)]
                initials = " ".join("-".join(q[0] + "." for q in p.split("-") if q) for p in first.split() if p)
                names.append(f"{initials} {last}".strip())
            else:
                names.append(a)
        FIRST_AUTHOR[key] = names[0].split()[-1] if names else key
        if len(names) > 2 and names[-1] != "et al.":
            authors = ", ".join(names[:-1]) + ", and " + names[-1]
        elif len(names) == 2 and names[-1] != "et al.":
            authors = " and ".join(names)
        else:
            authors = ", ".join(names).replace(", et al.", " et al.")
        title = f.get("title", "")
        venue = f.get("journal") or f.get("booktitle") or f.get("howpublished", "")
        pages = (("pp. " if "–" in f["pages"] else "Art. no. ") + f["pages"]) if f.get("pages") else ""
        if f.get("booktitle"):
            parts = [f"in *{venue}*", f.get("year", ""), pages]
        else:
            parts = [f"*{venue}*" if f.get("journal") else venue]
            if f.get("volume"):
                parts.append("vol. " + f["volume"])
            if f.get("number"):
                parts.append("no. " + f["number"])
            parts += [pages, f.get("year", "")]
        s = f'{authors}, "{title}," {", ".join(p for p in parts if p)}'
        if f.get("doi"):
            s += f", doi: {f['doi']}"
        if f.get("note") and not f.get("url"):          # as IEEEtranN prints it: ", note."
            s += ", " + f.pop("note").rstrip(".")
        s += "."
        if f.get("url"):
            s += f" [Online]. Available: {f['url']}"
        if f.get("note"):
            s += (" " if s.endswith(".") else ". ") + f["note"].rstrip(".") + "."
        out[key] = s
    return out


def tex_to_md(tex: str) -> str:
    bib = bib_entries()
    cite_order: list[str] = []

    def cite_num(key: str) -> int:
        if key not in cite_order:
            cite_order.append(key)
        return cite_order.index(key) + 1

    labels: dict[str, str] = {}
    sec_counter = [0, 0]
    fig_n = tab_n = 0
    body = tex[tex.index(r"\begin{document}") + len(r"\begin{document}"):tex.index(r"\end{document}")]
    title = re.search(r"\\title\{(.*?)\}\n", tex, re.S).group(1).replace("\\\\", " ")
    date = re.search(r"\\date\{(.*?)\}", tex).group(1)
    appendix = False

    # first pass: number figures/tables/sections for \ref
    for m in re.finditer(r"\\(section|subsection|appendix|label|caption|input)\{?([^}]*)\}?", body):
        kind, arg = m.group(1), m.group(2)
        if kind == "appendix":
            appendix = True; sec_counter = [0, 0]
        elif kind == "section":
            sec_counter[0] += 1; sec_counter[1] = 0
        elif kind == "subsection":
            sec_counter[1] += 1
    # second pass with real rendering
    appendix = False; sec_counter = [0, 0]
    pending_label_target = None
    thanks = re.search(r"\\thanks\{(.*?)\}\\\\", tex)
    note = ""
    if thanks:
        note = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", thanks.group(1))
        note = re.sub(r"\\url\{([^}]*)\}", r"\1", note)
    out: list[str] = [f"# {title}", "", "James Manktelow", "", "InferenceCarbon Ltd", "",
                      "ORCID 0009-0003-6822-8300 · james.manktelow@inferencecarbon.ai", "", date, ""]
    if note:
        out += [note, ""]
    lines = body.splitlines()
    i = 0
    in_table = False; table_rows: list[list[str]] = []; table_caption = ""; table_label = ""

    def inline(s: str) -> str:
        s = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", s)
        s = re.sub(r"\\emph\{(.*?)\}", r"*\1*", s)
        s = re.sub(r"\\texttt\{(.*?)\}", r"`\1`", s)
        s = re.sub(r"\\url\{(.*?)\}", r"<\1>", s)
        s = re.sub(r"\\href\{(.*?)\}\{(.*?)\}", r"[\2](\1)", s)
        s = re.sub(r"\\path\{(.*?)\}", r"`\1`", s)
        def _cite(m):  # one pass so numbering follows document order whatever the command
            keys = [k.strip() for k in m.group(2).split(",")]
            if m.group(1) == "t":
                return ", ".join(FIRST_AUTHOR[k] + " et al. [" + str(cite_num(k)) + "]" for k in keys)
            return "[" + ", ".join(str(cite_num(k)) for k in keys) + "]"
        s = re.sub(r"\\cite([tp]?)\{(.*?)\}", _cite, s)
        s = re.sub(r"(Section|Figure|Table|Appendix)~\\ref\{(.*?)\}", lambda m: f"{m.group(1)} {labels.get(m.group(2), '?')}", s)
        s = re.sub(r"\\ref\{(.*?)\}", lambda m: labels.get(m.group(1), "?"), s)
        s = s.replace("\\%", "%").replace("\\&", "&").replace("\\_", "_").replace("\\#", "#").replace("~", " ")
        s = s.replace("``", "\u201c").replace("''", "\u201d").replace("---", "\u2014").replace("--", "\u2013")
        s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("{,}", ",").replace("{", "").replace("}", ""), s)
        s = re.sub(r"\\label\{.*?\}", "", s)
        s = s.replace("\\ldots", "...").replace("\\noindent", "").replace("\\\\", " ")
        return s

    # pre-compute labels by walking once
    fn = tn = 0; app = False; sc = [0, 0]
    for ln in lines:
        if "\\appendix" in ln:
            app = True; sc = [0, 0]
        m = re.match(r"\\section\{(.*?)\}(\\label\{(.*?)\})?", ln)
        if m:
            sc[0] += 1; sc[1] = 0
            if m.group(3):
                labels[m.group(3)] = (chr(64 + sc[0]) if app else str(sc[0]))
        m = re.match(r"\\subsection\{(.*?)\}(\\label\{(.*?)\})?", ln)
        if m:
            sc[1] += 1
            if m.group(3):
                labels[m.group(3)] = f"{chr(64 + sc[0]) if app else sc[0]}.{sc[1]}"
        if "\\begin{figure}" in ln:
            fn += 1
        m = re.search(r"\\label\{(fig:.*?)\}", ln)
        if m:
            labels[m.group(1)] = str(fn)
        if "\\begin{table}" in ln:
            tn += 1
        m = re.search(r"\\caption\{.*\}\\label\{(tab:.*?)\}", ln)
        if m and not m.group(1) in labels:
            labels[m.group(1)] = str(tn)
        m = re.match(r"\\input\{tables/(.*?)\.tex\}", ln)
        if m:
            tn += 1
            t = (HERE / "tables" / (m.group(1) + ".tex")).read_text()
            lm = re.search(r"\\label\{(tab:.*?)\}", t)
            if lm:
                labels[lm.group(1)] = str(tn)

    tn = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        i += 1
        if not ln or ln.startswith("%") or ln.startswith("\\maketitle") or ln.startswith("\\bibliographystyle"):
            if not ln:
                out.append("")
            continue
        if ln.startswith("\\appendix"):
            appendix = True; sec_counter = [0, 0]; out += ["", "---", "", "# Appendices", ""]
            continue
        if ln.startswith("\\begin{abstract}"):
            out += ["## Abstract", ""]; continue
        if ln.startswith("\\end{abstract}"):
            continue
        m = re.match(r"\\section\{(.*?)\}", ln)
        if m:
            sec_counter[0] += 1; sec_counter[1] = 0
            num = chr(64 + sec_counter[0]) if appendix else str(sec_counter[0])
            out += [f"## {num}. {inline(m.group(1))}", ""]; continue
        m = re.match(r"\\subsection\{(.*?)\}", ln)
        if m:
            sec_counter[1] += 1
            num = f"{chr(64 + sec_counter[0]) if appendix else sec_counter[0]}.{sec_counter[1]}"
            out += [f"### {num} {inline(m.group(1))}", ""]; continue
        m = re.match(r"\\paragraph\{(.*?)\}(.*)", ln)
        if m:
            out.append(f"**{inline(m.group(1))}** {inline(m.group(2))}"); continue
        if ln.startswith("\\begin{figure}"):
            fig = []
            while not lines[i].startswith("\\end{figure}"):
                fig.append(lines[i]); i += 1
            i += 1
            src = re.search(r"\\includegraphics\[.*?\]\{(.*?)\}", "\n".join(fig)).group(1)
            cap = re.search(r"\\caption\{(.*)\}\\label\{(.*?)\}", "\n".join(fig), re.S)
            out += [f"![Figure {labels.get(cap.group(2), '?')}](figs/{src}.png)", "", f"**Figure {labels.get(cap.group(2), '?')}.** {inline(cap.group(1))}", ""]
            continue
        if ln.startswith("\\begin{table}"):
            tn += 1
            blk = []
            while not lines[i].startswith("\\end{table}"):
                blk.append(lines[i]); i += 1
            i += 1
            cap = re.search(r"\\caption\{(.*)\}\\label\{(.*?)\}", "\n".join(blk), re.S)
            rows = [r for r in blk if "&" in r]
            out += [f"**Table {labels.get(cap.group(2), tn)}.** {inline(cap.group(1))}", ""]
            out += md_table(rows, inline) + [""]
            continue
        if ln.startswith("\\begin{small}") or ln.startswith("\\end{small}"):
            continue
        if ln.startswith("\\begin{tabular}"):
            rows = []
            while not lines[i].startswith("\\end{tabular}"):
                if "&" in lines[i]:
                    rows.append(lines[i])
                i += 1
            i += 1
            out += md_table(rows, inline) + [""]
            continue
        m = re.match(r"\\input\{tables/(.*?)\.tex\}", ln)
        if m:
            tn += 1
            md = (HERE / "tables" / (m.group(1) + ".md")).read_text().strip()
            first, rest = md.split("\n", 1)
            cap = re.sub(r"(Section|Figure|Table|Appendix)~?\s*\\ref\{(.*?)\}", lambda r: f"{r.group(1)} {labels.get(r.group(2), '?')}", first.strip("*"))
            out += [f"**Table {tn}.** " + cap, "", rest.strip(), ""]
            continue
        if ln.startswith("\\bibliography"):
            out += ["## References", ""]
            for n, k in enumerate(cite_order, 1):
                out.append(f"{n}. {bib[k]}")
            out.append("")
            continue
        if ln.startswith("\\begin{itemize}") or ln.startswith("\\end{itemize}"):
            continue
        if ln.startswith("\\item"):
            out.append("- " + inline(ln[5:].strip())); continue
        out.append(inline(ln))
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def md_table(rows: list[str], inline) -> list[str]:
    cells = [[inline(c.strip()) for c in r.replace("\\\\", "").split("&")] for r in rows]
    head, body = cells[0], cells[1:]
    return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(r) + " |" for r in body]


CSS = """
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.4; max-width: 7.2in; margin: 0 auto; color: #111; }
h1 { font-size: 18pt; line-height: 1.25; } h2 { font-size: 14pt; margin-top: 1.4em; } h3 { font-size: 12pt; }
table { border-collapse: collapse; font-size: 8pt; margin: 6pt 0 10pt 0; page-break-inside: auto; }
th, td { border-bottom: 1px solid #bbb; padding: 2pt 5pt; text-align: left; vertical-align: top; }
th { border-bottom: 1.5px solid #333; } tr { page-break-inside: avoid; }
img { width: 100%; max-width: 7in; display: block; margin: 8pt auto 4pt auto; page-break-inside: avoid; }
code { font-size: 9pt; } p { text-align: justify; }
"""


def main():
    md = tex_to_md(TEX.read_text())
    MD.write_text(md)
    import markdown
    html = markdown.markdown(md, extensions=["tables"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Paper 1</title><style>{CSS}</style></head><body>{html}</body></html>")
    import sys
    if "--pdf" not in sys.argv:
        print("wrote", MD, "and", HTML, "(pass --pdf for a wkhtmltopdf reading copy)")
        return
    wk = shutil.which("wkhtmltopdf")
    if not wk:
        print("wkhtmltopdf not found; wrote", MD, "and", HTML)
        return
    subprocess.run([wk, "--quiet", "--enable-local-file-access", "--page-size", "A4", "--margin-top", "18mm", "--margin-bottom", "18mm",
                    "--margin-left", "16mm", "--margin-right", "16mm", "--footer-center", "[page]", "--footer-font-size", "8",
                    str(HTML), str(PDF)], check=True)
    print("wrote", MD, PDF)


if __name__ == "__main__":
    main()
