"""Move a paragraph's w:rPr in front of its w:sectPr inside w:pPr (the schema's order), in place."""
import shutil, sys, zipfile
from lxml import etree
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
src = sys.argv[1]; tmp = src + ".tmp"
zin = zipfile.ZipFile(src); zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
n = 0
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "word/document.xml":
        root = etree.fromstring(data)
        for ppr in root.iter(f"{{{W}}}pPr"):
            sect, rpr = ppr.find(f"{{{W}}}sectPr"), ppr.find(f"{{{W}}}rPr")
            if sect is not None and rpr is not None and list(ppr).index(rpr) > list(ppr).index(sect):
                ppr.remove(rpr); sect.addprevious(rpr); n += 1
        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    zout.writestr(item, data)
zout.close(); zin.close(); shutil.move(tmp, src)
print("reordered", n)
