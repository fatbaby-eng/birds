"""Tight viewBox for birds-strip.svg from bird path geometry."""
import re
from pathlib import Path

from svgpathtools import parse_path

svg_path = Path("assets/birds.svg")
svg = svg_path.read_text(encoding="utf-8")

paths = re.findall(
    r'<g clip-path="url\([^)]+\)"><path fill="#000000" d="([^"]+)"',
    svg,
)
if not paths:
    raise SystemExit("no bird paths found")

min_x = min_y = float("inf")
max_x = max_y = float("-inf")
for d in paths:
    p = parse_path(d)
    x1, x2, y1, y2 = p.bbox()
    min_x, max_x = min(min_x, x1), max(max_x, x2)
    min_y, max_y = min(min_y, y1), max(max_y, y2)

pad = 8
min_x -= pad
min_y -= pad
max_x += pad
max_y += pad
w, h = max_x - min_x, max_y - min_y
vb = f'viewBox="{min_x:.2f} {min_y:.2f} {w:.2f} {h:.2f}"'
print(vb)

out = svg.replace(re.search(r'viewBox="[^"]+"', svg).group(0), vb, 1)
out = re.sub(r'\s+width="[^"]+"', "", out, count=1)
out = re.sub(r'\s+height="[^"]+"', "", out, count=1)
Path("assets/birds-strip.svg").write_text(out, encoding="utf-8")
print("wrote birds-strip.svg")
