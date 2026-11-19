"""Write paper/figs/fig_processes.pdf and .png (Figure 13) from paper/figs/src/fig_processes.html with headless Chrome.

    python tools/convert_fig_processes.py

The source is a hand-written page, 820 px wide, in the style of Figure 1. Its numbers restate the paper's tables;
edit the page when a figure it quotes changes.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "paper" / "figs" / "src" / "fig_processes.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
figs = ROOT / "paper" / "figs"
dom = subprocess.run([CHROME, "--headless", "--disable-gpu", "--virtual-time-budget=2000", "--dump-dom", SRC.as_uri()],
                     check=True, capture_output=True, text=True).stdout
H = int(re.search(r"<title>H=(\d+)", dom).group(1))
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size=820,{H}",
                f"--screenshot={figs / 'fig_processes.png'}", SRC.as_uri()], check=True, capture_output=True)
page = SRC.read_text().replace("<style>", f"<style>@page{{size:820px {H}px;margin:0}}", 1)
tmp = figs / "src" / "_fig_processes_print.html"
tmp.write_text(page)
try:
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={figs / 'fig_processes.pdf'}", tmp.as_uri()],
                   check=True, capture_output=True)
finally:
    tmp.unlink()
print("written", H)
