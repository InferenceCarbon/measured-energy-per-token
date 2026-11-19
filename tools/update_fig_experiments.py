"""Version 1.1 edit of the experiment graphic (paper/figs/src/fig_experiments.svg): twelve experiments.

    python tools/update_fig_experiments.py

Edits the SVG in place (run once on the version 1.0.1 file): new run counts and settings on the nine cards, and a
fourth row with cards for c10 (speculative decoding) and c11 (precision probe). Then writes fig_experiments.pdf and
.png with headless Chrome, as the version 1.0.1 figure was made.
"""
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "paper" / "figs" / "src" / "fig_experiments.svg"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DY, H0 = 178, 778

s = SVG.read_text()
assert 'height="778"' in s, "already edited, or not the version 1.0.1 file"
swaps = [(">The ten experiments<", ">The twelve experiments<"),
         (">16 runs · Section 4.1<", ">40 runs · Section 4.1<"), (">80 runs · Sections 4.2, 4.3<", ">278 runs · Sections 4.2, 4.3<"),
         (">141 runs · Section 4.4<", ">546 runs · Section 4.4<"), (">144 runs · Section 4.5<", ">484 runs · Section 4.5<"),
         (">384 runs · Section 4.6<", ">580 runs · Section 4.6<"), (">48 runs · Sections 4.3, 4.7<", ">60 runs · Sections 4.3, 4.7<"),
         (">48 runs · Section 4.8<", ">235 runs · Section 4.8<"), (">36 runs · Section 4.9<", ">32 runs · Section 4.9<"),
         (">20 runs · Section 4.10<", ">34 runs · Section 4.10<"),
         (">Target batch 1, 4, 16, 64, 256;<", ">Target batch 1 to 2,048;<"),
         (">Cached context 1k, 4k, 16k, model<", ">Cached context 1k to the model<"), (">maximum; target batch 8, 64<", ">maximum; target batch 1 to 64<"),
         (">Fresh prompt 1k, 4k, 16k, model<", ">Fresh prompt 1k to the model<"),
         (">fp8 or nvfp4; batch 8, 64; context<", ">fp8 or nvfp4; batch 1 to 128; context<"),
         (">One agent, or 5 to 8 together;<", ">One agent, or 4 to 8 together;<")]
for old, new in swaps:
    assert s.count(old) == 1, old
    s = s.replace(old, new)

a, b, c = s.index('<rect x="10" y="590"'), s.index('<rect x="281" y="590"'), s.index('<rect x="552" y="590"')
def shift(card: str) -> str:
    return re.sub(r'\b(y|y1|y2)="([\d.]+)"', lambda m: f'{m.group(1)}="{float(m.group(2)) + DY:g}"', card)
def retext(card: str, texts: list[str]) -> str:
    it = iter(texts)
    out = re.sub(r">([^<>]+)</text>", lambda m: ">" + next(it) + "</text>", card)
    assert next(it, None) is None
    return out
c10 = retext(shift(s[a:b]), ["c10", "Speculative decoding", "What does guessing tokens ahead", "save or cost?",
                             "Off, or n-gram with 3 tokens; batch", "1, 8, 32; H100 and B200", "96 runs · Section 4.11"])
c10 = c10.replace('width="25.0" height="17"', 'width="32.0" height="17"').replace('x="34.5"', 'x="38"').replace('x="54.0"', 'x="61.0"')
c11 = retext(shift(s[b:c]), ["c11", "Precision probe", "Can a model's replies reveal 4-bit", "or 8-bit weights?",
                             "fp8 or nvfp4; 200 fixed prompts;", "top-20 log-probabilities kept;", "two models", "6 runs · Section 4.12"])
c11 = c11.replace('width="25.0" height="17"', 'width="32.0" height="17"').replace('x="305.5"', 'x="309"').replace('x="325.0"', 'x="332.0"')
s = s.replace("</svg>", c10 + c11 + "</svg>")
s = s.replace(f'height="{H0}" viewBox="0 0 820 {H0}"', f'height="{H0 + DY}" viewBox="0 0 820 {H0 + DY}"').replace(f'<rect width="820" height="{H0}"', f'<rect width="820" height="{H0 + DY}"')
SVG.write_text(s)

H = H0 + DY
with tempfile.TemporaryDirectory() as td:
    html = Path(td) / "f.html"
    html.write_text(f"<html><head><style>@page{{size:820px {H}px;margin:0}}html,body{{margin:0;padding:0}}svg{{display:block}}</style></head><body>{s}</body></html>")
    figs = ROOT / "paper" / "figs"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={figs / 'fig_experiments.pdf'}", html.as_uri()], check=True, capture_output=True)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size=820,{H}", f"--screenshot={figs / 'fig_experiments.png'}", html.as_uri()], check=True, capture_output=True)
print("written", H)
