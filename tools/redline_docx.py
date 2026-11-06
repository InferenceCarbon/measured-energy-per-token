"""Tracked-changes redline between two Word copies of Paper 1.

    python tools/redline_docx.py OLD.docx NEW.docx OUT.docx [--author NAME]

Both inputs should come from make_docx.py so their styles match. The output
is NEW.docx with every difference from OLD.docx recorded as a Word tracked
change (w:ins / w:del), so each change can be accepted or rejected in Word.

Paragraphs are aligned first (difflib over their text), then paired
paragraphs are compared word by word. Tables of the same shape are compared
cell by cell; a table whose shape changed is shown as the old table deleted
and the new one inserted. Figures are matched by their alt text; a changed
image file inside an unchanged figure is not tracked.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W, "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"}


def q(tag: str) -> str:
    p, t = tag.split(":")
    return "{%s}%s" % (NS[p], t)


class Ids:
    def __init__(self, author: str):
        self.n = 9000
        self.author = author
        self.date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def mark(self, tag: str):
        self.n += 1
        e = etree.Element(q(tag))
        e.set(q("w:id"), str(self.n)); e.set(q("w:author"), self.author); e.set(q("w:date"), self.date)
        return e


def text_of(el) -> str:
    return "".join(t.text or "" for t in el.iter(q("w:t")))


def key(el) -> str:
    if el.tag == q("w:tbl"):
        return "TBL|" + "|".join(text_of(tc) for tc in el.iter(q("w:tc")))
    d = el.find(".//wp:docPr", NS)
    if d is not None:
        return "IMG|" + (d.get("descr") or d.get("name") or "")
    return text_of(el)


def blocks(body):
    return [c for c in body if c.tag in (q("w:p"), q("w:tbl"))]


# ---------------------------------------------------------------- runs and tokens
TOK = re.compile(r"\s+|[A-Za-z0-9]+(?:[.,'’-][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


def run_tokens(p):
    """[(text, rPr, atomic_run)] for a paragraph; hyperlinks are flattened."""
    out = []
    for r in p.iter(q("w:r")):
        rpr = r.find(q("w:rPr"))
        if r.find(q("w:drawing")) is not None or r.find(q("w:fldChar")) is not None or r.find(q("w:instrText")) is not None:
            out.append(("￼" + etree.tostring(r, method="c14n").decode()[:200], rpr, r))
            continue
        for child in r:
            if child.tag == q("w:t"):
                for m in TOK.finditer(child.text or ""):
                    out.append((m.group(0), rpr, None))
            elif child.tag == q("w:tab"):
                out.append(("\t", rpr, None))
            elif child.tag == q("w:br"):
                out.append(("\n", rpr, None))
    return out


def make_run(text: str, rpr, deleted: bool):
    r = etree.Element(q("w:r"))
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    if text == "\t":
        r.append(etree.Element(q("w:tab")))
    elif text == "\n":
        r.append(etree.Element(q("w:br")))
    else:
        t = etree.SubElement(r, q("w:delText" if deleted else "w:t"))
        t.text = text
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return r


def emit(tokens, deleted: bool):
    """Runs for a token list, merging neighbours with the same formatting."""
    runs, buf, cur = [], "", None
    def flush():
        nonlocal buf
        if buf:
            runs.append(make_run(buf, cur, deleted))
            buf = ""
    for text, rpr, atom in tokens:
        if atom is not None:
            flush()
            if not deleted:
                runs.append(copy.deepcopy(atom))
            continue
        if text in ("\t", "\n"):
            flush(); runs.append(make_run(text, rpr, deleted)); continue
        sig = etree.tostring(rpr) if rpr is not None else b""
        if cur is not None and (etree.tostring(cur) if cur is not None else b"") != sig:
            flush()
        cur = rpr
        buf += text
    flush()
    return runs


def strip_runs(p):
    for c in list(p):
        if c.tag != q("w:pPr"):
            p.remove(c)


def para_mark(p, ids: Ids, tag: str):
    ppr = p.find(q("w:pPr"))
    if ppr is None:
        ppr = etree.Element(q("w:pPr")); p.insert(0, ppr)
    rpr = ppr.find(q("w:rPr"))
    if rpr is None:
        rpr = etree.SubElement(ppr, q("w:rPr"))
    rpr.insert(0, ids.mark(tag))


def readable(ops, a, b):
    """Make the diff read like a human redline: an unchanged stretch of a few
    short words between two edits is folded into one phrase replacement, and
    a paragraph mostly rewritten becomes one deletion and one insertion."""
    words = lambda toks: sum(1 for t, _, _ in toks if t.strip())
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in ops if op != "equal")
    total = max(len(a), len(b), 1)
    if changed / total > 0.8:
        return [("replace", 0, len(a), 0, len(b))]
    ops = list(ops)
    merged = True
    while merged:
        merged = False
        for k in range(1, len(ops) - 1):
            op, i1, i2, j1, j2 = ops[k]
            if op == "equal" and ops[k - 1][0] != "equal" and ops[k + 1][0] != "equal":
                stretch = a[i1:i2]
                if words(stretch) <= 2 and all(len(t) <= 4 for t, _, _ in stretch if t.strip()):
                    p, n = ops[k - 1], ops[k + 1]
                    ops[k - 1:k + 2] = [("replace", p[1], n[2], p[3], n[4])]
                    merged = True
                    break
    return ops


def diff_para(old_p, new_p, ids: Ids):
    a, b = run_tokens(old_p), run_tokens(new_p)
    sm = difflib.SequenceMatcher(None, [t for t, _, _ in a], [t for t, _, _ in b], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in readable(sm.get_opcodes(), a, b):
        if op == "equal":
            out += emit(b[j1:j2], False)
            continue
        if i2 > i1 and any(t for t, _, at in a[i1:i2] if at is None):
            d = ids.mark("w:del")
            for r in emit([x for x in a[i1:i2] if x[2] is None], True):
                d.append(r)
            out.append(d)
        if j2 > j1:
            ins = ids.mark("w:ins")
            for r in emit(b[j1:j2], False):
                ins.append(r)
            out.append(ins)
    strip_runs(new_p)
    for e in out:
        new_p.append(e)


def deleted_copy(old_el, ids: Ids):
    el = copy.deepcopy(old_el)
    for sp in el.iter(q("w:sectPr")):
        sp.getparent().remove(sp)
    paras = [el] if el.tag == q("w:p") else list(el.iter(q("w:p")))
    for p in paras:
        toks = [x for x in run_tokens(p) if x[2] is None]
        strip_runs(p)
        if toks:
            d = ids.mark("w:del")
            for r in emit(toks, True):
                d.append(r)
            p.append(d)
        para_mark(p, ids, "w:del")
    if el.tag == q("w:tbl"):
        for tr in el.findall(q("w:tr")):
            trpr = tr.find(q("w:trPr"))
            if trpr is None:
                trpr = etree.Element(q("w:trPr")); tr.insert(0, trpr)
            trpr.append(ids.mark("w:del"))
    return el


def mark_inserted(el, ids: Ids):
    paras = [el] if el.tag == q("w:p") else list(el.iter(q("w:p")))
    for p in paras:
        runs = [c for c in p if c.tag in (q("w:r"), q("w:hyperlink"))]
        if runs:
            ins = ids.mark("w:ins")
            runs[0].addprevious(ins)
            for r in runs:
                ins.append(r)
        para_mark(p, ids, "w:ins")
    if el.tag == q("w:tbl"):
        for tr in el.findall(q("w:tr")):
            trpr = tr.find(q("w:trPr"))
            if trpr is None:
                trpr = etree.Element(q("w:trPr")); tr.insert(0, trpr)
            trpr.append(ids.mark("w:ins"))


def same_shape(t1, t2) -> bool:
    r1, r2 = t1.findall(q("w:tr")), t2.findall(q("w:tr"))
    return len(r1) == len(r2) and all(len(a.findall(q("w:tc"))) == len(b.findall(q("w:tc"))) for a, b in zip(r1, r2))


def diff_block(o, n, ids: Ids):
    if o.tag == n.tag == q("w:p"):
        if text_of(o) != text_of(n):
            diff_para(o, n, ids)
    elif o.tag == n.tag == q("w:tbl") and same_shape(o, n):
        for ro, rn in zip(o.findall(q("w:tr")), n.findall(q("w:tr"))):
            for co, cn in zip(ro.findall(q("w:tc")), rn.findall(q("w:tc"))):
                for po, pn in zip(co.findall(q("w:p")), cn.findall(q("w:p"))):
                    if text_of(po) != text_of(pn):
                        diff_para(po, pn, ids)
    else:
        n.addprevious(deleted_copy(o, ids))
        if o.tag == n.tag == q("w:tbl"):
            spacer = etree.Element(q("w:p")); mark_inserted(spacer, ids)
            n.addprevious(spacer)
        mark_inserted(n, ids)


def similar(a, b) -> float:
    if (a.tag == q("w:tbl")) != (b.tag == q("w:tbl")):
        return 0.0
    return difflib.SequenceMatcher(None, key(a), key(b), autojunk=False).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--author", default="Claude (for J. Manktelow)")
    a = ap.parse_args()
    ids = Ids(a.author)
    with zipfile.ZipFile(a.old) as z:
        old_doc = etree.fromstring(z.read("word/document.xml"))
    with zipfile.ZipFile(a.new) as z:
        new_xml = z.read("word/document.xml")
    new_doc = etree.fromstring(new_xml)
    ob, nb = blocks(old_doc.find(q("w:body"))), blocks(new_doc.find(q("w:body")))
    sm = difflib.SequenceMatcher(None, [key(x) for x in ob], [key(x) for x in nb], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        olds, news = ob[i1:i2], nb[j1:j2]
        # pair blocks in order where they are similar enough to diff word by word
        pairs, oi = [], 0
        for nj, n in enumerate(news):
            best, bi = 0.0, None
            for k in range(oi, len(olds)):
                s = similar(olds[k], n)
                if s > best:
                    best, bi = s, k
            if bi is not None and best >= 0.45:
                pairs.append((bi, nj)); oi = bi + 1
        paired_o = {i for i, _ in pairs}; paired_n = {j for _, j in pairs}
        anchor = nb[j2] if j2 < len(nb) else None
        # deleted blocks go before the next paired or following new block, keeping order
        pi = 0
        for k, o in enumerate(olds):
            if k in paired_o:
                continue
            nxt = next((news[j] for i, j in pairs if i > k), anchor)
            dc = deleted_copy(o, ids)
            if nxt is not None:
                nxt.addprevious(dc)
            else:
                new_doc.find(q("w:body")).find(q("w:sectPr")).addprevious(dc)
        for i, j in pairs:
            diff_block(olds[i], news[j], ids)
        for j, n in enumerate(news):
            if j not in paired_n:
                mark_inserted(n, ids)

    tmp = Path(tempfile.mkdtemp())
    out = tmp / "out.docx"
    with zipfile.ZipFile(a.new) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = etree.tostring(new_doc, xml_declaration=True, encoding="UTF-8", standalone=True)
            zout.writestr(item, data)
    shutil.move(str(out), a.out)
    print(f"tracked changes: {ids.n - 9000}")


if __name__ == "__main__":
    main()
