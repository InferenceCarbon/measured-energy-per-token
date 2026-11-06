"""Word copy of Paper 1, set in Garamond and laid out like the LaTeX PDF.

    python tools/make_docx.py [paper/paper.md] [out.docx]

Needs pandoc and python-docx. Reads the Markdown reading copy written by
render.py (tables and figures inlined), converts it with pandoc, then sets
the page (US Letter, 1 in margins, centered page number in the footer, no
header), the type (Garamond: body 11 pt justified, sections 14 pt bold,
subsections 12 pt bold, captions and abstract 10 pt, tables 8 to 9 pt) and
the title block, so the Word copy matches the PDF closely enough to review
and redline. It is a review copy; the arXiv build is the LaTeX source.
"""
from __future__ import annotations

import copy
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent.parent / "paper"
FONT = "Garamond"
BLACK = RGBColor(0, 0, 0)


def set_font(style_or_run, size=None, bold=None, italic=None):
    f = style_or_run.font
    f.name = FONT
    f.color.rgb = BLACK
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run.element, "get_or_add_rPr") else style_or_run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), FONT)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rfonts.get(qn(a)) is not None:
            del rfonts.attrib[qn(a)]


def page_number_footer(section):
    section.header.is_linked_to_previous = False
    for p in section.header.paragraphs:
        p.text = ""
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)
    set_font(run, 11)


def table_borders(tbl):
    """Booktabs look: heavy top and bottom rules, a light rule under the header."""
    tblPr = tbl._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 12), ("left", 0), ("bottom", 12), ("right", 0), ("insideH", 0), ("insideV", 0)):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        b.append(e)
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    # schema order: ... tblW, jc, tblCellSpacing, tblInd, tblBorders, shd, tblLayout, tblCellMar, tblLook ...
    after = [qn(t) for t in ("w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription")]
    nxt = next((c for c in tblPr if c.tag in after), None)
    if nxt is not None:
        nxt.addprevious(b)
    else:
        tblPr.append(b)
    if tbl.rows:
        for cell in tbl.rows[0].cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tb = OxmlElement("w:tcBorders")
            e = OxmlElement("w:bottom")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
            tb.append(e)
            tcPr.append(tb)


def main(md: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.docx"
        subprocess.run(["pandoc", str(md), "-o", str(raw), f"--resource-path={md.parent}:{md.parent / 'figs'}"], check=True)
        doc = Document(str(raw))

    for s in doc.sections:
        s.orientation = WD_ORIENT.PORTRAIT
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
        s.footer_distance = s.header_distance = Inches(0.5)
        s.gutter = Inches(0)
        page_number_footer(s)

    st = doc.styles
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in [x.name for x in st]:
            set_font(st[name], 11)
            pf = st[name].paragraph_format
            pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(3), 1.0
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    st["Body Text"].paragraph_format.first_line_indent = Pt(15)
    st["First Paragraph"].paragraph_format.first_line_indent = Pt(0)
    for name, size, before, after in (("Heading 1", 17, 0, 12), ("Heading 2", 14, 14, 6), ("Heading 3", 12, 10, 4)):
        set_font(st[name], size, bold=name != "Heading 1", italic=False)
        pf = st[name].paragraph_format
        pf.space_before, pf.space_after, pf.keep_with_next = Pt(before), Pt(after), True
    for name in ("Table", "Image Caption", "Table Caption", "Captioned Figure", "Figure"):
        if name in [x.name for x in st]:
            set_font(st[name], 10)
    for name in [x.name for x in st if x.name.startswith("Hyperlink")]:
        set_font(st[name])

    paras = doc.paragraphs
    # title block: first Heading 1 is the title; author lines up to the Abstract heading
    title_i = next(i for i, p in enumerate(paras) if p.style.name == "Heading 1")
    abs_i = next(i for i, p in enumerate(paras) if p.style.name.startswith("Heading") and p.text.strip() == "Abstract")
    t = paras[title_i]
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_before = Pt(24)
    for p in paras[title_i + 1:abs_i]:
        p.style = st["Normal"]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        size = 12
        if p.text.startswith("ORCID"):
            size = 10
        elif p.text.startswith("Corpus, harness"):
            size = 9
        for r in p.runs:
            set_font(r, size)
    h = paras[abs_i]
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in h.runs:
        set_font(r, 10, bold=True)
    h.paragraph_format.space_before = Pt(18)
    ab = paras[abs_i + 1]
    ab.paragraph_format.left_indent = ab.paragraph_format.right_indent = Inches(0.5)
    ab.paragraph_format.first_line_indent = Pt(0)
    for r in ab.runs:
        set_font(r, 10)

    for p in paras:
        txt = p.text
        if re.match(r"^(Figure|Table) \d+\.", txt):          # caption paragraphs (bold label from render.py)
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_after = Pt(8)
            if txt.startswith("Table"):
                p.paragraph_format.keep_with_next = True
            for r in p.runs:
                set_font(r, 10)
        if p.style.name == "Heading 1" and txt.strip() == "Appendices":
            p.runs[0].add_break(WD_BREAK.PAGE) if False else None
            p.paragraph_format.page_break_before = True
        for r in p.runs:                                     # every run in Garamond, keep size from style
            set_font(r)
        for img in p._p.iter(qn("wp:extent")):               # figures no wider than the text block
            cx, cy = int(img.get("cx")), int(img.get("cy"))
            maxw = int(Inches(6.5))
            if cx > maxw:
                img.set("cx", str(maxw)); img.set("cy", str(int(cy * maxw / cx)))
        for ext in p._p.iter(qn("a:ext")):
            cx, cy = int(ext.get("cx")), int(ext.get("cy"))
            maxw = int(Inches(6.5))
            if cx > maxw:
                ext.set("cx", str(maxw)); ext.set("cy", str(int(cy * maxw / cx)))
        if any(True for _ in p._p.iter(qn("w:drawing"))):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.keep_with_next = True

    for tbl in doc.tables:
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        ncol = len(tbl.columns)
        size = 9 if ncol <= 6 else (8 if ncol <= 9 else 7)
        table_borders(tbl)
        for ri, row in enumerate(tbl.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.first_line_indent = Pt(0)
                    p.paragraph_format.space_after = Pt(0)
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        set_font(r, size, bold=True if ri == 0 else None)

    # pandoc turns the image alt text into a caption ("Figure 1"); render.py writes the real caption below it
    for p in list(doc.paragraphs):
        if p.style.name in ("Image Caption", "Caption") and re.fullmatch(r"Figure \d+", p.text.strip()):
            p._p.getparent().remove(p._p)

    # wide tables (9+ columns) go on landscape pages, as in the PDF
    body = doc.element.body
    base = doc.sections[-1]._sectPr

    def sect_par(landscape: bool):
        sp = copy.deepcopy(base)
        for tag in ("w:headerReference", "w:footerReference"):
            for e in sp.findall(qn(tag)):
                sp.remove(e)
        pg = sp.find(qn("w:pgSz"))
        w, h = 12240, 15840                          # US Letter in twentieths of a point
        if landscape:
            pg.set(qn("w:w"), str(h)); pg.set(qn("w:h"), str(w)); pg.set(qn("w:orient"), "landscape")
        else:
            pg.set(qn("w:w"), str(w)); pg.set(qn("w:h"), str(h))
            if pg.get(qn("w:orient")) is not None:
                del pg.attrib[qn("w:orient")]
        mar = sp.find(qn("w:pgMar"))
        m = "864" if landscape else "1440"          # 0.6 in on landscape pages, 1 in otherwise
        for side in ("w:top", "w:bottom", "w:left", "w:right"):
            mar.set(qn(side), m)
        for side, v in (("w:header", "720"), ("w:footer", "720"), ("w:gutter", "0")):
            if mar.get(qn(side)) is None:
                mar.set(qn(side), v)
        p = OxmlElement("w:p")
        ppr = OxmlElement("w:pPr")
        ppr.append(sp)
        p.append(ppr)
        return p

    # column widths in proportion to their content (pandoc gives equal widths), over the text block
    for tbl in doc.tables:
        ncol = len(tbl.columns)
        total = 9.8 if ncol >= 9 else 6.5
        need = []
        for j in range(ncol):
            texts = [row.cells[j].text for row in tbl.rows if j < len(row.cells)]
            body_len = max((len(x) for x in texts[1:]), default=4)
            head_len = max((len(w) for w in texts[0].split()), default=4) if texts else 4
            need.append(max(6, min(42, max(body_len, head_len)) + 2))   # +2: cell padding
        widths = [int(Inches(total * n / sum(need))) for n in need]
        grid = tbl._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w / 635)))          # EMU -> twentieths of a point
        for row in tbl.rows:
            for j, cell in enumerate(row.cells[:ncol]):
                cell.width = widths[j]
        tblPr = tbl._tbl.tblPr
        tw = tblPr.find(qn("w:tblW"))
        if tw is None:
            tw = OxmlElement("w:tblW")
            st_ = tblPr.find(qn("w:tblStyle"))
            st_.addnext(tw) if st_ is not None else tblPr.insert(0, tw)
        tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(int(sum(widths) / 635)))
        for e in tblPr.findall(qn("w:tblLayout")):
            tblPr.remove(e)
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
        nxt = next((c for c in tblPr if c.tag in (qn("w:tblCellMar"), qn("w:tblLook"), qn("w:tblCaption"), qn("w:tblDescription"))), None)
        nxt.addprevious(lay) if nxt is not None else tblPr.append(lay)

    for tbl in doc.tables:
        if len(tbl.columns) < 9:
            continue
        t = tbl._tbl
        cap = t.getprevious()
        start = cap if cap is not None and cap.tag == qn("w:p") and "".join(cap.itertext()).startswith("Table") else t
        start.addprevious(sect_par(landscape=False))
        t.addnext(sect_par(landscape=True))

    doc.core_properties.title = paras[title_i].text
    doc.core_properties.author = "James Manktelow"
    doc.save(str(out))


if __name__ == "__main__":
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "paper.md"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "paper.docx"
    main(md, out)
