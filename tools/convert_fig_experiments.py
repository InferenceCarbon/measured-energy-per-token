"""Write paper/figs/fig_experiments.pdf and .png from paper/figs/src/fig_experiments.svg with headless Chrome.

    python tools/convert_fig_experiments.py
"""
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "paper" / "figs" / "src" / "fig_experiments.svg"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
s = SVG.read_text()
H = int(re.search(r'<svg[^>]*height="(\d+)"', s).group(1))
figs = ROOT / "paper" / "figs"
with tempfile.TemporaryDirectory() as td:
    html = Path(td) / "f.html"
    html.write_text(f"<html><head><style>@page{{size:820px {H}px;margin:0}}html,body{{margin:0;padding:0}}svg{{display:block}}</style></head><body>{s}</body></html>")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={figs / 'fig_experiments.pdf'}", html.as_uri()], check=True, capture_output=True)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size=820,{H}", f"--screenshot={figs / 'fig_experiments.png'}", html.as_uri()], check=True, capture_output=True)
print("written", H)
