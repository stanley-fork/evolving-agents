"""Overlays the title and the measured line on the generated plate (docs/img/IMAGES.md §6), so the words on the
social card are exactly these — an image model would misspell or invent them.

    python social_card.py <plate.png> social-preview.jpg      # Pillow
"""
import sys
from PIL import Image, ImageDraw, ImageFont

INK, DIM, SAGE = (33, 28, 21), (92, 83, 68), (90, 110, 70)
FONTS = ["/System/Library/Fonts/Supplemental/Georgia.ttf", "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]


def font(size, bold=False):
    names = (["/System/Library/Fonts/Supplemental/Georgia Bold.ttf"] if bold else []) + FONTS
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
w, h = im.size
t = 2.0
if w / h > t:
    nw = int(h * t); x = (w - nw) // 2; im = im.crop((x, 0, x + nw, h))
else:
    nh = int(w / t); y = (h - nh) // 2; im = im.crop((0, y, w, y + nh))
im = im.resize((1280, 640), Image.LANCZOS)
veil = Image.new("RGB", (470, 640), (243, 235, 221))
im.paste(Image.blend(im.crop((0, 0, 470, 640)), veil, 0.82), (0, 0))
d = ImageDraw.Draw(im)
x = 48
d.text((x, 70), "evolving-agents", font=font(50, True), fill=INK)
for i, line in enumerate(["A director that learns", "how your team works."]):
    d.text((x, 150 + i * 44), line, font=font(36), fill=INK)
for i, line in enumerate(["It plans by walking your CLAUDE.md", "and memory, delegates to Claude Code",
                          "or local models, and keeps each lesson", "as a gated, versioned patch."]):
    d.text((x, 262 + i * 30), line, font=font(22), fill=DIM)
d.text((x, 420), "0.929 vs 0.589", font=font(46, True), fill=SAGE)
d.text((x, 476), "planning by your rules, measured", font=font(22), fill=DIM)
d.text((x, 560), "github.com/EvolvingAgentsLabs/evolving-agents", font=font(18), fill=DIM)
im.save(dst, "JPEG", quality=86, optimize=True)
print(dst, im.size)
