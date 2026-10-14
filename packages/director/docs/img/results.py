"""Draws results.png from the numbers in docs/BRIEF-D0.md, BRIEF-D2.md and BRIEF-D3.md (copied here, each with
the run it comes from). A chart, never a generated image: an image model would invent the numbers.

    python results.py      # writes results.png next to this file (matplotlib)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, DIM, SEP = "#F4ECDB", "#211C15", "#5C5344", "#CDBF9F"
SLATE, SAGE, RUST, OCHRE = "#5B7186", "#8A9A6B", "#B5603A", "#C9A54A"

PANELS = [
    ("D0 — planning our way\n14 situations, blind judge", ["today", "today +\nwalk it", "director"],
     [0.589, 0.768, 0.929], [DIM, OCHRE, SAGE]),
    ("D2 — learning from tasks\n8 held-out lab probes", ["before", "after\nlearning", "guard:\nD0 after"],
     [0.0, 0.781, 0.964], [DIM, SAGE, SLATE]),
    ("D3 — learning the lane\npublic text → frontier", ["before", "after one\nmove", "control:\ninternal local"],
     [0.0, 1.0, 1.0], [DIM, SAGE, SLATE]),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 7), dpi=100)
fig.patch.set_facecolor(BG)
for ax, (title, labels, vals, cols) in zip(axes, PANELS):
    ax.set_facecolor(BG)
    bars = ax.bar(labels, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}".rstrip("0").rstrip(".") if v not in (0, 1) else f"{v:.1f}",
                ha="center", va="bottom", fontsize=17, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(colors=INK, labelsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SEP)
    ax.set_title(title, fontsize=16, color=INK, loc="left", pad=14)
fig.tight_layout(pad=2.0)
fig.savefig(Path(__file__).with_name("results.png"), facecolor=BG)
