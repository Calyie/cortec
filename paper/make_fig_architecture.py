"""Draw the deployment architecture (three trust zones, one arrow across the boundary) as a
figure for the arXiv paper: paper/figures/fig13_architecture.png."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "paper/figures/fig13_architecture.png"
INK, MUTED = "#1a1a1a", "#666666"
ZONE = {"1": "#f4e7e1", "2": "#e6eef7", "3": "#e7f2e8"}
EDGE = {"1": "#b5674d", "2": "#4a6fa5", "3": "#4f8a5b"}

fig, ax = plt.subplots(figsize=(13.0, 7.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 7.4); ax.axis("off")

def zone(x, y, w, h, key, title, subtitle):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=ZONE[key], ec=EDGE[key], lw=1.6))
    ax.text(x + 0.18, y + h - 0.28, title, fontsize=11.5, fontweight="bold", color=EDGE[key], va="top")
    ax.text(x + 0.18, y + h - 0.62, subtitle, fontsize=8.6, color=MUTED, va="top")

def box(x, y, w, h, title, lines, key, bold_title=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=EDGE[key], lw=1.2))
    ax.text(x + w / 2, y + h - 0.2, title, fontsize=9.6, fontweight="bold" if bold_title else "normal",
            ha="center", va="top", color=INK)
    ax.text(x + w / 2, y + h - 0.52, "\n".join(lines), fontsize=8.0, ha="center", va="top",
            color=INK, linespacing=1.35)

def arrow(x0, y0, x1, y1, label=None, color=INK, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.16, label, fontsize=8.2, ha="center", color=color)

# Zone 1
zone(0.2, 4.85, 12.6, 2.45, "1", "Trust zone 1: regulated",
     "the institution's own VPC or premises; private records never leave it")
box(0.45, 5.0, 2.3, 1.45, "System of record",
    ["EHR, core banking,", "registry", "PHI / PII"], "1")
box(3.2, 5.0, 2.6, 1.45, "Schema declaration",
    ["column list, public bounds,", "public bin edges, target,", "max rows per person"], "1")
box(6.25, 5.0, 3.1, 1.45, "Stage A: DP release engine",
    ["public stratification (ε = 0)", "DP histograms per cohort and class",
     "DP conditional table, DP sizes", "auto-configuration (charged)"], "1")
box(9.85, 5.0, 2.75, 1.45, "Release artifact R",
    ["noisy counts only", "audit trail: every query,", "ε, sensitivity, composition rule", "no private record"], "1")
arrow(2.75, 5.72, 3.2, 5.72, color=EDGE["1"]); arrow(5.8, 5.72, 6.25, 5.72, color=EDGE["1"])
arrow(9.35, 5.72, 9.85, 5.72, color=EDGE["1"])

# boundary
ax.plot([0.2, 12.8], [4.5, 4.5], color=INK, lw=2.2, ls=(0, (6, 3)))
ax.text(0.35, 4.57, "TRUST BOUNDARY: only R crosses; everything below is post-processing and adds nothing to ε",
        fontsize=9.2, fontweight="bold", color=INK, va="bottom")
arrow(11.22, 5.0, 11.22, 3.35, color=INK, lw=2.4)
ax.text(11.42, 4.2, "R", fontsize=11, fontweight="bold", color=INK, va="center")

# Zone 2
zone(0.2, 1.95, 12.6, 2.25, "2", "Trust zone 2: tenant-isolated model platform",
     "Bedrock, Azure OpenAI or Vertex AI in the institution's own account: private networking, no training on inputs, data residency, BAA")
box(0.45, 2.1, 4.85, 1.25, "Guardrails",
    ["hash-locked prompt · capability gate", "domain and category checks · coverage guards",
     "spend cap · rate-limit backoff · yield checkpoints"], "2")
box(5.6, 2.1, 7.0, 1.25, "Stage B: generation from R by a frozen model, reasoning enabled",
    ["exact-count batches per cohort  →  k× pool  →  selection to R  →  sub-bin redraw", "",
     "repeatable without limit; ε unchanged"], "2")

# Zone 3 (flow runs right to left so the synthetic dataset sits under Stage B)
zone(0.2, 0.15, 12.6, 1.55, "3", "Trust zone 3: synthetic data, freely shareable", "")
box(10.25, 0.28, 2.35, 0.8, "Synthetic dataset D̂", ["any size, unlimited redraws"], "3")
box(6.55, 0.28, 3.4, 0.8, "Stage C: utility transmission bound", ["a utility claim under DP; not a privacy audit"], "3")
box(3.75, 0.28, 2.5, 0.8, "Release package", ["D̂ + audit + bound + claim block"], "3")
box(0.45, 0.28, 3.0, 0.8, "Downstream use", ["training, vendor evaluation, sharing"], "3")
arrow(10.25, 0.68, 9.95, 0.68, color=EDGE["3"]); arrow(6.55, 0.68, 6.25, 0.68, color=EDGE["3"])
arrow(3.75, 0.68, 3.45, 0.68, color=EDGE["3"])
arrow(11.42, 2.1, 11.42, 1.08, color=INK, lw=1.8)

fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
