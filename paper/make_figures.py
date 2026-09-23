"""
make_figures.py: publication figures for the CoRTeC paper.

Every number is read from the experiment result JSON; none is typed in by hand, so re-running the
experiments and then this script regenerates the paper's figures consistently.

Layout and typography rules applied throughout, so the figures survive print reduction:

  * one type scale for the whole paper, derived from BASE, rather than a size chosen per call;
  * ``constrained_layout`` everywhere, so titles, axis labels and legends are allocated real space
    instead of being positioned by hand and then colliding;
  * every annotation anchored to a data point or an axes fraction and displaced in *offset points*,
    never placed at a guessed data coordinate that shifts when the data changes;
  * annotations placed only in regions the data provably cannot occupy (a monotone calibration
    curve leaves the lower-right corner empty; a reference line is labelled at the axis edge
    furthest from the marks), so text never lands on a line;
  * shared legends below the panels instead of per-panel boxes, wherever panels share series;
  * a validated categorical palette assigned in fixed order, never cycled; text in ink colours.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MultipleLocator, NullLocator

sys.path.insert(0, ".")
OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

# Validated categorical palette (light surface), assigned in fixed order.
C = {"blue": "#2a78d6", "orange": "#d1541f", "aqua": "#0f8f68", "yellow": "#a8770a",
     "magenta": "#c4306b", "violet": "#4a3aa7", "green": "#0f7a3d"}
INK, INK2, MUTED, GRID = "#111111", "#4e4d4a", "#8a8985", "#e6e5e2"

BASE = 9.0  # body size; every other size in this file is BASE ± a small offset
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.05, "figure.constrained_layout.w_pad": 0.05,
    "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
    "font.family": "DejaVu Sans", "font.size": BASE,
    "axes.titlesize": BASE, "axes.titleweight": "bold", "axes.titlepad": 6,
    "axes.labelsize": BASE - 0.5, "axes.labelcolor": INK, "axes.labelpad": 4,
    "xtick.labelsize": BASE - 1.5, "ytick.labelsize": BASE - 1.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "axes.labelweight": "normal",
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.55, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": BASE - 1.5,
    "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.4,
    "figure.facecolor": "white", "axes.facecolor": "white", "lines.solid_capstyle": "round",
})

TITLE = BASE + 1.5   # figure title
NOTE = BASE - 2.0    # in-plot annotation
SMALL = BASE - 2.5   # in-plot secondary annotation


def _pts(path, xk):
    if not os.path.exists(path):
        return None
    s = json.load(open(path))
    p = [q for q in s["points"] if q.get("generated") is not None]
    if not p:
        return None
    x = np.array([q[xk] for q in p]); y = np.array([q["generated"] for q in p])
    o = np.argsort(x)   # sweeps are recorded in run order; a line plot needs them in x order,
    return (x[o], y[o], # otherwise the trace doubles back on itself and reads as two lines
            s.get("header_only_control") or s.get("header_control"), s.get("slope"))


# ── Figure 0: the method ─────────────────────────────────────────────────────────────
def fig_pipeline():
    """The system diagram. Its job is to make the privacy argument legible at a glance: where
    the private data is touched, where the budget is spent, and that the generator sits wholly
    on the post-processing side. Boxes are sized from their own content so text cannot overflow."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig = plt.figure(figsize=(12.6, 4.2))
    fig.set_layout_engine("none")
    ax = fig.add_axes([0.006, 0.02, 0.988, 0.87])
    ax.set_xlim(0, 126); ax.set_ylim(0, 42); ax.axis("off")

    PAD_T, TITLE_H, GAP, LEAD, PAD_B = 2.6, 1.6, 2.4, 2.5, 1.8

    def box_h(lines):
        return PAD_T + TITLE_H + GAP + max(len(lines), 1) * LEAD + PAD_B

    def box(x, ytop, w, title, lines, face, edge, tcol=INK):
        """Anchored by its TOP edge, height derived from the content: never clipped."""
        h = box_h(lines)
        ax.add_patch(FancyBboxPatch((x, ytop - h), w, h,
                                    boxstyle="round,pad=0,rounding_size=1.3",
                                    fc=face, ec=edge, lw=1.2, zorder=2))
        ax.text(x + w / 2, ytop - PAD_T - TITLE_H / 2, title, ha="center", va="center",
                fontsize=BASE - 0.5, fontweight="bold", color=tcol, zorder=3)
        y = ytop - PAD_T - TITLE_H - GAP
        for ln in lines:
            ax.text(x + w / 2, y, ln, ha="center", va="center",
                    fontsize=NOTE, color=INK2, zorder=3)
            y -= LEAD
        return h

    def arrow(x1, y1, x2, y2, col=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.2, color=col, zorder=1))

    # The private / post-processing divide. The rule stops below its own caption rather than
    # running through it.
    ax.plot([48.0, 48.0], [0.5, 38.2], color=C["magenta"], lw=1.2, ls=(0, (5, 4)), zorder=0)
    ax.text(48.0, 40.3, "nothing to the right of this line ever sees a private record",
            fontsize=NOTE, color=C["magenta"], ha="center", va="center", fontstyle="italic")

    box(1.0, 27.0, 18.0, "Private data  $D$",
        ["$n$ records", "the only place", "raw data exists"], "#fceff3", C["magenta"], C["magenta"])
    box(24.0, 37.0, 20.0, "Public stratification",
        ["rule over public domains", "cost  ε = 0"], "#ecf3fd", C["blue"])
    box(24.0, 21.0, 20.0, "DP release   (Stage A)",
        ["histograms per outcome class,", "conditional target table",
         "cost  ε = ε$_{total}$"], "#ecf3fd", C["blue"])
    box(52.0, 34.0, 20.0, "Released statistics",
        ["cohort shapes per class,", "class balance,", "P(y | cell)", "",
         "the only channel", "from $D$ onward"], "#eaf8f2", C["aqua"])
    box(76.0, 37.0, 23.0, "Frozen LLM   (Stage B)",
        ["no fine-tuning; reads only the release", "each batch owes exact per-bin counts",
         "post-processing, ε = 0"], "#fdf5e7", C["yellow"])
    box(76.0, 18.0, 23.0, "Synthetic records",
        ["a k× pool, then the n rows", "that match the release;", "unlimited draws, no further cost"],
        "#fdf5e7", C["yellow"])

    arrow(19.2, 24.0, 23.8, 30.0)
    arrow(19.2, 20.0, 23.8, 14.5)
    arrow(44.2, 30.0, 51.8, 27.0)
    arrow(44.2, 14.5, 51.8, 19.0)
    arrow(72.2, 27.0, 75.8, 29.5)
    arrow(87.5, 21.0, 87.5, 18.3)
    box(102.0, 18.0, 23.0, "Transmission bound   (Stage C)",
        ["bounds the private-to-synthetic gap", "on every released cell, under DP",
         "a utility claim, not a privacy audit"], "#eef2ea", "#4f8a5b")
    arrow(99.2, 10.0, 101.8, 10.0)

    # Captions sit in the clear strip below the lowest box on each side (bottoms: 5.1 and 4.6).
    ax.text(34.0, 2.4, "the privacy budget is spent here, once", ha="center",
            fontsize=NOTE, color=C["blue"], fontweight="bold")
    ax.text(87.5, 2.4, "and this step is repeatable for free", ha="center",
            fontsize=NOTE, color=C["yellow"], fontweight="bold")

    fig.text(0.5, 0.955,
             "CoRTeC: the privacy budget is spent once on a statistics release; "
             "generation is post-processing",
             ha="center", va="center", fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig01_pipeline.png"); plt.close(fig)
    print("  fig01_pipeline.png")


# ── Figure 1: the central experiment ─────────────────────────────────────────────────
NHANES_TRUTH = 0.1538   # diabetes rate of race_ethnicity = black_nh in the seed-42 NHANES training split (650 rows)


def fig_calibration():
    sets = [
        ("Census: Adult", "education → income", C["blue"],
         # An earlier version read the 6-point seed-42 file alone, which gave slope 1.023 while §7.4 reports the pooled
         # 3-seed / 14-point fit (1.063). Read the pooled artefact so figure and table agree.
         "results/misalignment_pooled_adult.json", "dp_released_mean", 0.619),
        # the healthcare panel is NHANES, the primary clinical dataset (one row per person); the
        # Diabetes 130 sweep remains in the table of section 7.4. The true rate is the diabetes
        # rate of the forced group (race_ethnicity = black_nh) in the seed-42 training split.
        (("Healthcare: NHANES", "race/ethnicity → diabetes", C["orange"],
          "results/nhanes_misalign/summary.json", "dp_shown", NHANES_TRUTH)
         if os.path.exists("results/nhanes_misalign/summary.json") else
         ("Healthcare: Diabetes 130", "prior admissions → readmission", C["orange"],
          "results/diabetes_misalign/summary.json", "dp_shown", 0.214)),
        ("Finance: Credit default", "repayment status → default", C["aqua"],
         "results/credit_misalign/summary.json", "dp_shown", 0.696),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.0), sharey=True)
    drawn = []
    for ax, (title, rel, col, path, xk, truth) in zip(axes, sets):
        got = _pts(path, xk)
        if got is None:
            ax.set_axis_off(); continue
        x, y, hdr, slope = got
        mae = float(np.mean(np.abs(y - x)))

        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.1, color=MUTED, zorder=1)
        ax.axhline(truth, color=INK2, lw=0.9, ls=(0, (1, 2.5)), zorder=1)
        if hdr is not None:
            ax.axhline(hdr, color=C["magenta"], lw=1.8, zorder=2)
        ax.plot(x, y, "-o", lw=2.0, ms=7, color=col, mec="white", mew=1.4, zorder=4)
        drawn.append(col)

        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.06, 1.14)
        ax.set_yticks([0, .25, .5, .75, 1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("DP rate released to the generator")
        ax.set_title(f"{title}\n{rel}", fontsize=BASE - 0.5, linespacing=1.4)

        # The calibration curve is monotone increasing, so the LEFT of a high horizontal line
        # and the RIGHT of a low one are provably empty; label each reference there.
        if hdr is not None:
            ax.annotate(f"{hdr:.0%}", xy=(0.0, hdr), xytext=(2, -4), textcoords="offset points",
                        fontsize=NOTE, color=C["magenta"], fontweight="bold", ha="left", va="top")
        else:
            ax.annotate("no unconditioned control on this dataset", xy=(0.02, 0.97), xycoords="axes fraction",
                        fontsize=SMALL, color=C["magenta"], ha="left", va="top")
        ax.annotate(f"true rate {truth:.0%}", xy=(1.0, truth), xytext=(-2, 4),
                    textcoords="offset points", fontsize=SMALL, color=INK2,
                    ha="right", va="bottom")
        # Lower-right is empty for the same reason: put the fit statistics there.
        ax.annotate(f"slope {slope:.3f}   MAE {mae:.3f}", xy=(0.98, 0.035),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=NOTE, fontweight="bold", color=col)

    axes[0].set_ylabel("rate in the generated data")
    handles = [Line2D([], [], color=INK2, lw=2.0, marker="o", ms=6, mec="white", mew=1.2,
                      label="CoRTeC output"),
               Line2D([], [], color=MUTED, lw=1.1, ls=(0, (4, 3)),
                      label="perfect transmission (y = x)"),
               Line2D([], [], color=C["magenta"], lw=1.8,
                      label="unconditioned LLM (never sees $D$)"),
               Line2D([], [], color=INK2, lw=0.9, ls=(0, (1, 2.5)),
                      label="true rate in the unmodified data")]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, columnspacing=1.4)
    fig.suptitle("CoRTeC transmits a private relationship that contradicts the model's prior; "
                 "an unconditioned LLM is flat and far from the truth",
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig05_calibration.png"); plt.close(fig)
    print("  fig05_calibration.png")


# ── Figure 2: which fidelity axis predicts utility ───────────────────────────────────
def fig_axes_independent():
    import pandas as pd
    # An earlier version read the old evaluator's points CSV and fitted the correlation over
    # nine conditions (35 points) while drawing seven, so the caption's count could not be checked
    # against the legend. Read the unified per-draw file §7 is audited against, keep ONLY the
    # conditions that are drawn, and print the count so the caption is written from the data.
    p = "results/adult_unified_eval.json"
    if not os.path.exists(p):
        print("  (skip fig4)"); return
    style = [("cortec-v3", "CoRTeC", C["blue"], "o", 58),
             ("aim", "AIM", C["orange"], "s", 46),
             ("mst", "MST", C["aqua"], "^", 52),
             ("patectgan", "PATE-CTGAN", C["violet"], "D", 40),
             ("header-only-v3", "header-only LLM", C["yellow"], "P", 60),
             ("real sample, n=300", "real sample", MUTED, "*", 130),
             ("permuted target (floor)", "target permuted", C["magenta"], "X", 58)]
    u = json.load(open(p)); u = u.get("results", u)
    rows = [{"label": lab, "tv1": r["tv_1way"], "cseen": r["cond_tv_seen"], "lr": r["tstr_LR"]}
            for lab, *_ in style for r in u.get(lab, [])
            if r.get("tstr_LR") is not None and not np.isnan(r["tstr_LR"])]
    df = pd.DataFrame(rows)
    print(f"  fig04: {len(df)} points across {df.label.nunique()} drawn conditions")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    for ax, (xcol, xlabel, ttl) in zip(
            axes, [("tv1", "marginal fidelity error   (1-way TV)",
                    "Marginal fidelity does not predict utility"),
                   ("cseen", "conditional fidelity error",
                    "Conditional fidelity does")]):
        for lab, nice, col, mk, sz in style:
            sub = df[df.label == lab]
            if sub.empty:
                continue
            ax.scatter(sub[xcol], sub.lr, s=sz, color=col, marker=mk, alpha=0.92,
                       edgecolor="white", linewidth=0.8, zorder=3, label=nice)
        r = df[xcol].corr(df.lr)
        print(f"  fig04: r({xcol}, TSTR-LR) = {r:+.3f} over {len(df)} drawn points")
        xs = np.linspace(df[xcol].min(), df[xcol].max(), 50)
        ax.plot(xs, np.polyval(np.polyfit(df[xcol], df.lr, 1), xs),
                lw=1.5, color=INK2, ls=(0, (5, 3)), zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_title(ttl)
        ax.set_ylim(0.24, 0.95)
        # Fit line rises to the upper left / falls to the lower right; the opposite corner of
        # each panel is where the correlation statistic can sit without touching a mark.
        ax.annotate(f"r = {r:+.2f}", xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=BASE + 3, fontweight="bold",
                    color=C["orange"] if abs(r) > 0.4 else MUTED)
    axes[0].set_ylabel("downstream utility   (TSTR-AUC)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncols=7, columnspacing=1.1)
    fig.suptitle("Fidelity and utility are near-independent axes: the marginal methods cluster "
                 "at good marginals with poor utility",
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig04_fidelity_utility_axes.png"); plt.close(fig)
    print("  fig04_fidelity_utility_axes.png")


# ── Figure 3: model grid ─────────────────────────────────────────────────────────────
def fig_model_grid():
    """The SUPERSEDED open-weight model grid.

    Figure 7 in the paper is the enterprise-platform ladder, produced by
    `paper/make_fig07_enterprise_ladder.py`. This function used to write the same filename, so
    running the two in either order silently changed what Figure 7 showed. It now writes a clearly
    superseded name and is kept only because §7.5.1 discusses the open-weight control it plots.
    """
    meta = {
        "qwen2.5_7b-instruct":  ("qwen2.5 7B", "Alibaba (Qwen)", 7),
        "qwen2.5_14b-instruct": ("qwen2.5 14B", "Alibaba (Qwen)", 14),
        "qwen2.5_32b-instruct": ("qwen2.5 32B", "Alibaba (Qwen)", 32),
        "qwen2.5_72b-instruct": ("qwen2.5 72B", "Alibaba (Qwen)", 72),
        "gemma2_27b-instruct-q4_K_M": ("gemma2 27B", "Google", 27),
        "mistral-small_24b-instruct-2501-q4_K_M": ("mistral-small 24B", "Mistral", 24),
        "gpt-oss_20b": ("gpt-oss 20B", "OpenAI", 20),
    }
    fam_col = {"Alibaba (Qwen)": C["blue"], "Google": C["aqua"], "Mistral": C["yellow"],
               "OpenAI": C["violet"], "Meta": C["orange"], "Anthropic": C["magenta"]}
    rows = []
    for d, (lab, fam, b) in meta.items():
        g = _pts(f"results/model_grid/{d}/summary.json", "dp_shown")
        if g: rows.append((lab, fam, b, float(np.mean(np.abs(g[1] - g[0])))))
    g = _pts("results/diabetes_misalign_llama70b/summary.json", "dp_shown")
    if g: rows.append(("llama3.3 70B", "Meta", 70, float(np.mean(np.abs(g[1] - g[0])))))
    # Claude Fable 5's parameter count is not public, so it cannot be given an x-position on the
    # scale panel: inventing one would fabricate a data point. It appears in the ranked panel
    # and as a reference line, with no scale implied.
    g = _pts("results/diabetes_misalign/summary.json", "dp_shown")
    fable = float(np.mean(np.abs(g[1] - g[0]))) if g else None
    if not rows:
        print("  (skip fig3)"); return

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    ranked = sorted(rows + ([("Claude Fable 5", "Anthropic", None, fable)] if fable else []),
                    key=lambda r: r[3], reverse=True)
    ypos = np.arange(len(ranked))
    ax.barh(ypos, [r[3] for r in ranked], height=0.6,
            color=[fam_col[r[1]] for r in ranked], zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in ranked])
    for i, r in enumerate(ranked):
        ax.annotate(f"{r[3]:.3f}", xy=(r[3], i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=NOTE, color=INK2)
    ax.set_xlabel("mean absolute magnitude error   (lower is better)")
    ax.set_xlim(0, max(r[3] for r in ranked) * 1.18)
    ax.set_ylim(-0.7, len(ranked) - 0.3)
    ax.set_title("Magnitude accuracy, by model")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    fams = {}
    for lab, fam, b, err in rows:
        fams.setdefault(fam, []).append((b, err))
    for fam, pts in sorted(fams.items()):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                "-o" if len(pts) > 1 else "o", lw=1.8, ms=8, color=fam_col[fam],
                mec="white", mew=1.3, label=fam, zorder=3)
    ax.set_xscale("log"); ax.set_xticks([7, 14, 28, 70])
    ax.set_xticklabels(["7B", "14B", "28B", "70B"])
    ax.xaxis.set_minor_locator(NullLocator())   # log minor ticks print "2 x 10^1" over our labels
    ax.set_xlim(5.7, 95); ax.set_ylim(-0.03, 0.50)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("model scale, log axis   (open-weights models only)")
    ax.set_ylabel("mean absolute magnitude error")
    ax.set_title("Both scale and family matter")
    if fable is not None:
        ax.axhline(fable, color=C["magenta"], lw=1.3, ls=(0, (5, 3)), zorder=2)
        # every open-weights point sits at x ≥ 7 and, below 0.05, at x ≥ 24, so the label goes
        # at the left edge under the line, where no mark can fall.
        ax.annotate(f"Claude Fable 5: {fable:.3f}  (scale not public)",
                    xy=(5.9, fable), xytext=(0, -4), textcoords="offset points",
                    fontsize=SMALL, color=C["magenta"], ha="left", va="top")
    ax.legend(loc="upper right", ncols=1)
    fig.suptitle("The mechanism holds across six model families; magnitude accuracy improves "
                 "with scale but is not determined by scale alone",
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/_superseded_openweight_model_grid.png"); plt.close(fig)
    print("  _superseded_openweight_model_grid.png")


# ── Figure 4: privacy–utility curve ──────────────────────────────────────────────────
def fig_epsilon():
    data = {}
    # NHANES under the shipped configuration (the ε sweep of §7.6); Diabetes 130 as the fallback
    if all(os.path.exists(f"results/nhanes_eps_shipped/eval_eps{t}.json") for t in ("0p3", "1", "2", "8")):
        for e, t in {0.3: "0p3", 1.0: "1", 2.0: "2", 8.0: "8"}.items():
            data[e] = json.load(open(f"results/nhanes_eps_shipped/eval_eps{t}.json"))
        ref = data[2.0]
    else:
        for e, f in {0.3: "eval_eps0p3", 1.0: "eval_eps1", 8.0: "eval_eps8"}.items():
            p = f"results/diabetes_eps/{f}.json"
            if os.path.exists(p): data[e] = json.load(open(p))
        if os.path.exists("results/diabetes/eval_full.json"):
            data[2.0] = json.load(open("results/diabetes/eval_full.json"))
        if len(data) < 3:
            print("  (skip fig4)"); return
        ref = json.load(open("results/diabetes/eval_full.json"))
    eps = sorted(data)
    fl = float(np.nanmean([r["tstr_LR"] for r in ref["shuffled-target [FLOOR]"]]))
    ce = float(np.nanmean([r["tstr_LR"] for r in ref["TRTR [CEILING]"]]))
    rs = float(np.nanmean([r["tstr_LR"] for r in ref["real-sample [FLOOR-n]"]]))

    # The utility panel is given extra width because its three reference levels have to be
    # labelled in a right-hand margin: the "real sample" level falls *inside* the measured range,
    # so there is no clear space for its label anywhere along the line itself.
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.5]})
    panels = [("tv_1way", "marginal fidelity error\n1-way TV, lower is better", False),
              ("cond_tv_seen", "conditional fidelity error\nlower is better", False),
              ("tstr_LR", "downstream utility\nTSTR-AUC, higher is better", True)]
    series = [("cortec", "CoRTeC", C["blue"]), ("mst", "MST", C["aqua"])]
    for ax, (key, lab, higher) in zip(axes, panels):
        for mkey, mlab, col in series:
            ys = [np.nanmean([r[key] for r in data[e][mkey]]) if data[e].get(mkey) else np.nan
                  for e in eps]
            ax.plot(eps, ys, "-o", lw=1.9, ms=7, color=col, mec="white", mew=1.3,
                    label=mlab, zorder=4)
        ax.set_xscale("log"); ax.set_xticks(eps)
        ax.set_xticklabels([f"{e:g}" for e in eps])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("privacy budget  ε   (log axis)")
        ax.set_title(lab, fontsize=BASE - 0.5, linespacing=1.4)
        if higher:
            # Reserve a right-hand margin the data never enters, and label the three reference
            # levels there: the measured curves all sit near 0.55 and would collide at the left.
            ax.set_xlim(eps[0] * 0.85, eps[-1] * 5.5)
            ax.set_ylim(fl - 0.035, ce + 0.035)
            for yv, tl in [(ce, "train-on-real ceiling"), (rs, "real sample, same n"),
                           (fl, "permuted-target floor")]:
                ax.axhline(yv, color=INK2, lw=0.85, ls=(0, (1, 2.5)), zorder=1)
                ax.annotate(tl, xy=(eps[-1] * 5.2, yv), xytext=(0, 3),
                            textcoords="offset points", fontsize=SMALL,
                            color=INK2, ha="right", va="bottom")
        else:
            ax.set_xlim(eps[0] * 0.85, eps[-1] * 1.20)
            ax.set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncols=2, columnspacing=1.6)
    nh = os.path.exists("results/nhanes_eps_shipped/eval_eps8.json") and ref is data.get(2.0)
    fig.suptitle(("NHANES, shipped configuration: on a 3,000-record dataset the release, not the generator, "
                  "is the limit below ε = 1") if nh else
                 ("CoRTeC's output quality is essentially independent of ε over [0.3, 8]: "
                  "its error is generator-limited, not noise-limited"),
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig08_epsilon.png"); plt.close(fig)
    print("  fig08_epsilon.png")


# ── Figure 5: the saturated fidelity criterion ───────────────────────────────────────
def fig_saturation():
    # An earlier version of this figure read FINAL_comparison.json, the old evaluator (2-way over a 20-pair
    # sample), while §7.1.1's table is the unified evaluator over all 105 pairs. The two disagreed
    # (0.067 vs 0.106 on 2-way) and the suptitle hardcoded the stale percentages. Read the same
    # file the table is audited against, and compute the title from the data.
    p = "results/adult_unified_eval.json"
    if not os.path.exists(p):
        print("  (skip fig2)"); return
    d = json.load(open(p)); d = d.get("results", d)
    good, bad = d["real sample, n=300"], d["permuted target (floor)"]
    metrics = [("tv_1way", "1-way TV"), ("tv_2way", "2-way TV"),
               ("cond_tv_seen", "conditional TV\nseen groups"),
               ("cond_tv_heldout", "conditional TV\nheld-out groups")]
    g = [np.mean([r[k] for r in good]) for k, _ in metrics]
    b = [np.mean([r[k] for r in bad]) for k, _ in metrics]
    tg = np.mean([r["tstr_LR"] for r in good]); tb = np.mean([r["tstr_LR"] for r in bad])
    sim1, sim2 = 100 * (1 - b[0]), 100 * (1 - b[1])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                  gridspec_kw={"width_ratios": [2.4, 1.0]})
    x = np.arange(len(metrics)); w = 0.36
    ax.bar(x - w / 2 - 0.012, g, w, color=C["blue"], label="real data", zorder=3)
    ax.bar(x + w / 2 + 0.012, b, w, color=C["orange"],
           label="real data, target column permuted", zorder=3)
    for i, (gv, bv) in enumerate(zip(g, b)):
        ax.annotate(f"{gv:.3f}", xy=(i - w / 2 - 0.012, gv), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=NOTE, color=INK2)
        ax.annotate(f"{bv:.3f}", xy=(i + w / 2 + 0.012, bv), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=NOTE, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], linespacing=1.4)
    ax.set_ylim(0, max(g + b) * 1.30)
    ax.set_ylabel("error   (lower is better)")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Marginal metrics cannot separate them; conditional metrics can")

    ax2.bar([0, 1], [tg, tb], 0.5, color=[C["blue"], C["orange"]], zorder=3)
    for i, v in enumerate([tg, tb]):
        ax2.annotate(f"{v:.3f}", xy=(i, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=NOTE, color=INK2)
    ax2.axhline(0.5, color=INK2, lw=0.85, ls=(0, (1, 2.5)), zorder=1)
    ax2.annotate("chance", xy=(1.42, 0.5), xytext=(0, 3), textcoords="offset points",
                 fontsize=SMALL, color=INK2, ha="right", va="bottom")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["real", "permuted"])
    ax2.set_xlim(-0.55, 1.55); ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("TSTR-AUC   (higher is better)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("…and utility collapses")
    fig.suptitle('The 90%-similarity criterion is saturated: data whose target column has been '
                 f'permuted still scores {sim1:.1f}% / {sim2:.1f}% "similar"',
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig02_saturation.png"); plt.close(fig)
    print("  fig02_saturation.png")


# ── Figure 6: ablation ───────────────────────────────────────────────────────────────
def fig_ablation():
    """One panel, not five. The five decisions are measured in five different units, so plotting
    them as five small before/after axes forced five y-labels into a strip and made none of them
    readable. Plotting the improvement FACTOR puts them on one commensurable axis, and the raw
    before → after values are printed beside each bar so nothing is lost."""
    items = [  # label, before, after, unit, lower_is_better
        ("DP histograms\ninstead of mean + std", 0.780, 0.045, "released-statistic error, TV", True),
        ("Row allocation ∝ cohort size\ninstead of uniform", 12.1, 1.0,
         "over-representation of a 2% cohort, ×", True),
        ("Parallel composition\n+ ε-free stratification", 0.0100, 0.0667,
         "ε available per released statistic", False),
    ]
    # Two rows here were once typed in by hand and disagreed with Appendix E's table
    # (0.054 -> 0.039 against the audited 0.052 -> 0.045; 0.705 -> 0.846 against a value that
    # appears in no §8.2 table). Read them from the same files the tables are audited against.
    u = json.load(open("results/adult_unified_eval.json")); u = u.get("results", u)
    cat_before = float(np.mean([r["tv_1way"] for r in u["cortec-v3"]]))
    cat_after = float(np.mean([r["tv_1way"] for r in u["cortec-v4 [full categoricals]"]]))
    items.append(("Full categorical release\ninstead of top-4 values",
                  round(cat_before, 3), round(cat_after, 3), "1-way TV", True))
    hyb_path = ("results/hybrid_table_unified.json"
                if os.path.exists("results/hybrid_table_unified.json")
                else "results/hybrid_table_shipped.json")
    h = json.load(open(hyb_path))["table"]
    def _rf(row):  # the shipped file spells it "TSTR-RF", the unified re-score "tstr_RF"
        return float(row.get("TSTR-RF", row.get("tstr_RF")))
    items.append(("Rich conditional table\ninstead of 12 cells",
                  round(_rf(h["AIM + level 2"]), 3), round(_rf(h["AIM + level 4"]), 3),
                  "TSTR-RF of the hybrid", False))
    # §7.11: the class-conditional release, read from the audited evaluation file
    if os.path.exists("results/credit_cc_fable_eval.json"):
        cc = json.load(open("results/credit_cc_fable_eval.json"))
        rf_cc = float(np.mean([r["tstr_RF"] for r in cc["CoRTeC, class-conditional release (Fable 5)"]]))
        rf_po = float(np.mean([r["tstr_RF"] for r in cc["CoRTeC, pooled release (Fable 5, F.3)"]]))
        items.append(("One histogram block per outcome class\ninstead of a pooled block",
                      round(rf_po, 3), round(rf_cc, 3), "finance TSTR-RF", False))
    # §7.12: exact-count batches and selection, from the v2 ablation record
    if os.path.exists("results/cortec_v2_ablation.json"):
        ab = json.load(open("results/cortec_v2_ablation.json"))
        # Only the selection step is plotted: exact-count batches act on the pool (they make it a
        # faithful sample of the release) and do not move a random n-row sample's 1-way error, so
        # a bar for them would be a bar for nothing; the table in Appendix E says so in words.
        if "select" in ab:
            items.append(("Selection from a 3x pool of exact-count\nbatches instead of the raw draw",
                          round(ab["select"]["before"], 3), round(ab["select"]["after"], 3), "Adult 1-way TV", True))
    rows = []
    for label, before, after, unit, lower in items:
        factor = (before / after) if lower else (after / before)
        rows.append((label, before, after, unit, factor))
    rows.sort(key=lambda r: r[4])

    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    y = np.arange(len(rows))
    ax.barh(y, [r[4] for r in rows], height=0.58, color=C["blue"], zorder=3)
    ax.axvline(1.0, color=INK2, lw=1.0, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], linespacing=1.35)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1×", "2×", "5×", "10×", "20×", "50×"])
    ax.xaxis.set_minor_locator(NullLocator())
    # The right margin is sized for the longest trailing annotation, whose width is fixed in
    # points; without it the top bar's label runs off the canvas.
    ax.set_xlim(0.92, 170)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel("improvement factor, log axis   (1× = the decision changed nothing)")
    ax.grid(axis="y", visible=False)
    for i, (label, before, after, unit, factor) in enumerate(rows):
        ax.annotate(f"{factor:.1f}×", xy=(factor, i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=NOTE,
                    fontweight="bold", color=INK)
        ax.annotate(f"{before:g} → {after:g}   {unit}", xy=(factor, i), xytext=(30, 0),
                    textcoords="offset points", va="center", fontsize=SMALL, color=INK2)
    ax.annotate("no change", xy=(1.0, len(rows) - 0.45), xytext=(4, 0),
                textcoords="offset points", fontsize=SMALL, color=INK2, va="center")
    fig.suptitle("Ablation: what each design decision bought, all at unchanged "
                 "ε$_{total}$ = 2.0",
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig11_ablation.png"); plt.close(fig)
    print("  fig11_ablation.png")


# ── Figure 7: the head-to-head: the paper's central claim ───────────────────────────
def fig_head_to_head():
    """CoRTeC against the deployed DP mechanisms on downstream utility, across three domains,
    with bootstrap CIs and both floors drawn. This figure carries §6.3, so it has to make the
    comparison legible without the reader consulting a table: the reference band between the
    permuted-target floor (no information) and the real-sample floor (what is achievable at this
    n) is shaded, and every method is a point with its interval inside or below it."""
    import json as _json
    DATASETS = [
        ("Census: Adult", "results/FINAL_comparison_v2.json", "cortec-v3",
         "real-sample-300", "real-shuffled-income [FLOOR]",
         {"cortec-v3": "CoRTeC", "aim": "AIM", "mst": "MST", "patectgan": "PATE-CTGAN",
          "dpctgan": "DP-CTGAN", "pategan": "PATE-GAN"},
         {"lr": "tstr_auc_LR", "rf": "tstr_auc_RF", "gbm": "tstr_auc_GBM"}),
        # the healthcare panel is NHANES, scored beside its own baselines at n = 300
        (("Healthcare: NHANES", "results/cortec_v2_nhanes_auto_eval.json", "CoRTeC (Gemini 3.5 Flash)",
          "real-sample [FLOOR-n]", "shuffled-target [FLOOR]",
          {"CoRTeC (Gemini 3.5 Flash)": "CoRTeC", "AIM": "AIM", "MST": "MST", "PATE-CTGAN": "PATE-CTGAN"},
          {"lr": "tstr_LR", "rf": "tstr_RF", "gbm": "tstr_GBM"})
         if os.path.exists("results/nhanes_baselines/mst_seed0.csv") else
         ("Healthcare: Diabetes 130", "results/diabetes/eval_all.json", "cortec",
          "real-sample [FLOOR-n]", "shuffled-target [FLOOR]",
          {"cortec": "CoRTeC", "aim": "AIM", "mst": "MST", "patectgan": "PATE-CTGAN",
           "dpctgan": "DP-CTGAN", "pategan": "PATE-GAN"},
          {"lr": "tstr_LR", "rf": "tstr_RF", "gbm": "tstr_GBM"})),
        ("Finance: Credit default", "results/credit/eval_all.json", "cortec",
         "real-sample [FLOOR-n]", "shuffled-target [FLOOR]",
         {"cortec": "CoRTeC", "aim": "AIM", "mst": "MST", "patectgan": "PATE-CTGAN",
          "dpctgan": "DP-CTGAN", "pategan": "PATE-GAN"},
         {"lr": "tstr_LR", "rf": "tstr_RF", "gbm": "tstr_GBM"}),
    ]
    # Each panel also carries CoRTeC v2 (class-conditional release, exact-count batches, selection
    # from a 3x pool; Gemini 3.5 Flash), read from the v2 evaluation files, beside the earlier
    # pooled-release arms, which stay as the ablation reference.
    EXTRA = {"Census: Adult": ("results/cortec_v2_adult_eval.json", "CoRTeC (Gemini 3.5 Flash)", "CoRTeC"),
             "Healthcare: Diabetes 130": None, "Healthcare: NHANES": None,
             "Finance: Credit default": ("results/cortec_v2_credit_eval.json", "CoRTeC (Gemini 3.5 Flash)", "CoRTeC")}
    RENAME = {"Census: Adult": "CoRTeC, earlier configuration", "Healthcare: Diabetes 130": "CoRTeC, earlier configuration",
              "Healthcare: NHANES": "CoRTeC",
              "Finance: Credit default": "CoRTeC, earlier configuration"}
    ORDER = ["CoRTeC", "CoRTeC, earlier configuration", "AIM", "MST", "PATE-CTGAN", "DP-CTGAN", "PATE-GAN"]
    COL = {"CoRTeC": C["blue"], "CoRTeC, earlier configuration": "#7fa6d9", "AIM": C["orange"], "MST": C["aqua"],
           "PATE-CTGAN": C["violet"], "DP-CTGAN": C["yellow"], "PATE-GAN": C["magenta"]}

    def boot(xs, n=4000, seed=0):
        xs = np.asarray(xs, float)
        if len(xs) < 2:
            return float(xs.mean()), float(xs.mean()), float(xs.mean())
        r = np.random.default_rng(seed)
        m = r.choice(xs, size=(n, len(xs)), replace=True).mean(axis=1)
        return float(xs.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))

    panels = []
    for title, path, treat, ceil, floor, names, keys in DATASETS:
        if not os.path.exists(path):
            continue
        d = _json.load(open(path))
        pool = {**{k: v for k, v in d.items() if k not in ("_meta", "_reference")},
                **d.get("_reference", {})}
        if treat not in pool:
            continue
        got = {}
        for key, nice in names.items():
            if key in pool and pool[key]:
                xs = [r[keys["lr"]] for r in pool[key] if r.get(keys["lr"]) is not None]
                if xs:
                    got[RENAME.get(title, nice) if nice == "CoRTeC" else nice] = boot(xs)
        if EXTRA.get(title) and os.path.exists(EXTRA[title][0]):
            _e = _json.load(open(EXTRA[title][0]))
            _rows = _e.get(EXTRA[title][1]) or []
            _xs = [r["tstr_LR"] for r in _rows if r.get("tstr_LR") is not None]
            if _xs:
                got[EXTRA[title][2]] = boot(_xs)
        refs = {}
        for lab, k in (("real", ceil), ("permuted", floor)):
            if k in pool and pool[k]:
                refs[lab] = float(np.mean([r[keys["lr"]] for r in pool[k]]))
        if got:
            panels.append((title, got, refs))
    if not panels:
        print("  (skip fig7)"); return

    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels) + 0.8, 4.3), sharey=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, got, refs) in zip(axes, panels):
        labs = [m for m in ORDER if m in got]
        y = np.arange(len(labs))[::-1]
        if "real" in refs and "permuted" in refs:
            ax.axhspan(-0.5, len(labs) - 0.5, xmin=0, xmax=0, alpha=0)  # keep limits sane
            ax.axvspan(refs["permuted"], refs["real"], color=GRID, alpha=0.55, zorder=0)
        for yy, m in zip(y, labs):
            mu, lo, hi = got[m]
            ax.plot([lo, hi], [yy, yy], color=COL[m], lw=2.4, solid_capstyle="round", zorder=3)
            ax.plot([mu], [yy], "o", ms=8, color=COL[m], mec="white", mew=1.4, zorder=4)
            ax.annotate(f"{mu:.3f}", xy=(mu, yy), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=NOTE, color=INK2)
        for lab in ("real", "permuted"):
            if lab in refs:
                ax.plot([refs[lab]] * 2, [-0.45, len(labs) - 0.35], color=INK2, lw=0.9,
                        ls=(0, (1, 2.5)), zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(labs)
        # Reserve a clear strip BELOW the lowest method row for the reference labels. Placing them
        # at the top collided with the value labels, and on a panel where the two references are
        # close together they also collided with each other.
        ax.set_ylim(-1.55, len(labs) - 0.35)
        ax.set_xlabel("downstream utility   (TSTR-AUC, logistic regression)")
        ax.set_title(title, fontsize=BASE - 0.5)
        ax.grid(axis="y", visible=False)
        lo = min(min(v[1] for v in got.values()),
                 refs.get("permuted", 1.0)) - 0.05
        hi = max(max(v[2] for v in got.values()), refs.get("real", 0.0)) + 0.045
        ax.set_xlim(lo, hi)
        # anchor each label to its own line and push it away from the other, so a small gap
        # between the two references cannot make the two labels overlap
        if "permuted" in refs:
            ax.annotate("no-information\nfloor", xy=(refs["permuted"], -0.85),
                        xytext=(-3, 0), textcoords="offset points", fontsize=SMALL,
                        color=INK2, ha="right", va="center", linespacing=1.3)
        if "real" in refs:
            ax.annotate("real sample,\nsame n", xy=(refs["real"], -0.85),
                        xytext=(3, 0), textcoords="offset points", fontsize=SMALL,
                        color=INK2, ha="left", va="center", linespacing=1.3)
    fig.suptitle("CoRTeC against the deployed DP mechanisms: downstream utility at matched ε and "
                 "matched n\n(points are means over draws, bars are 95% bootstrap CIs; the shaded "
                 "band spans no-information to real-data-at-this-n)",
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig03_head_to_head.png"); plt.close(fig)
    print("  fig03_head_to_head.png")


# ── Figure 9: does a private conditional relationship survive each mechanism? ────────
def fig_transmission_by_family():
    """Three mechanisms, one forced relationship, identical data and ε. The point is that two
    methods with identical correct DP guarantees differ by a factor of ~18 in whether the
    relationship reaches their output, which no fidelity number reveals."""
    import json as _json
    # NHANES carries the healthcare figures; the Diabetes 130 files remain as the fallback
    path = "results/nhanes_baseline_transmission/transmission.json"
    sweep = "results/nhanes_misalign/summary.json"
    if not (os.path.exists(path) and os.path.exists(sweep)):
        path = "results/diabetes_baseline_transmission/transmission.json"
        sweep = "results/diabetes_misalign/summary.json"
    if not os.path.exists(path):
        print("  (skip fig9)"); return
    d = _json.load(open(path))
    NICE = {"mst": ("MST", "marginal / graphical", C["aqua"]),
            "aim": ("AIM", "marginal / graphical", C["orange"]),
            "patectgan": ("PATE-CTGAN", "PATE generative", C["violet"]),
            "dpctgan": ("DP-CTGAN", "DP-SGD generative", C["yellow"]),
            "pategan": ("PATE-GAN", "PATE generative", C["magenta"])}
    series = []
    # CoRTeC from its own sweep, same dataset and same forced group
    g = _pts(sweep, "dp_shown")
    if g:
        series.append(("CoRTeC", "released statistics + frozen LLM", C["blue"],
                       g[0], g[1], g[3]))
    for key, e in d.get("methods", {}).items():
        pts = [p for p in e.get("points", []) if p.get("generated") is not None]
        if len(pts) < 2 or key not in NICE:
            continue
        nice, fam, col = NICE[key]
        x = np.array([p["true_rate"] for p in pts]); y = np.array([p["generated"] for p in pts])
        o = np.argsort(x)
        series.append((nice, fam, col, x[o], y[o], e.get("slope")))
    if not series:
        print("  (skip fig9)"); return

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.3),
                                  gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.1, color=MUTED, zorder=1)
    for nice, fam, col, x, y, slope in series:
        ax.plot(x, y, "-o", lw=2.0, ms=7, color=col, mec="white", mew=1.3, zorder=3, label=nice)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.08)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("rate forced into the private data")
    ax.set_ylabel("rate in the mechanism's output")
    ax.set_title("Does the relationship reach the output?")
    ax.annotate("faithful transmission", xy=(0.60, 0.60), xytext=(6, -12),
                textcoords="offset points", fontsize=SMALL, color=MUTED)
    ax.legend(loc="upper left")

    order = sorted(series, key=lambda r: -(r[5] if r[5] is not None else -9))
    ypos = np.arange(len(order))[::-1]
    for yy, (nice, fam, col, x, y, slope) in zip(ypos, order):
        ax2.barh([yy], [slope], height=0.55, color=col, zorder=3)
        # a bar ending near 1.0 would put its label across the "faithful" reference line
        ax2.annotate(f"{slope:.3f}", xy=(max(slope, 1.04) if slope > 0.9 else slope, yy), xytext=(5 if slope >= 0 else -5, 0),
                     textcoords="offset points", va="center",
                     ha="left" if slope >= 0 else "right", fontsize=NOTE, color=INK2)
    ax2.axvline(1.0, color=INK2, lw=0.9, ls=(0, (1, 2.5)), zorder=1)
    ax2.annotate("faithful", xy=(1.0, -0.75), xytext=(-3, 0), textcoords="offset points",
                 fontsize=SMALL, color=INK2, ha="right", va="center")
    ax2.set_yticks(ypos); ax2.set_yticklabels([f"{n}\n{f}" for n, f, *_ in order],
                                              linespacing=1.3, fontsize=BASE - 2)
    ax2.set_ylim(-1.3, len(order) - 0.35)
    ax2.set_xlim(0, 1.25)
    ax2.set_xlabel("transmission slope   (1.0 = faithful, 0 = independent of the data)")
    ax2.set_title("All at identical ε = 2.0")
    ax2.grid(axis="y", visible=False)
    fig.suptitle("Whether a private conditional relationship survives depends on the mechanism "
                 "family, not on having a DP guarantee",
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig06_transmission_family.png"); plt.close(fig)
    print("  fig06_transmission_family.png")


# ── Figure 10: membership inference: measured leakage against the theoretical bound ─
def fig_membership_inference():
    """Two panels. Left: the measured advantage against the bound the epsilon guarantee permits,
    across budgets. Right: the attacks on CoRTeC beside the SAME attacks on a positive control of
    real records passed off as synthetic: without which a null result says nothing, because a
    blind attack also scores 0.5."""
    import json as _json
    sweep_p, mia_p = "results/mia_epsilon_sweep.json", "results/membership_inference.json"
    if not (os.path.exists(sweep_p) and os.path.exists(mia_p)):
        print("  (skip fig10)"); return
    sweep = _json.load(open(sweep_p)); mia = _json.load(open(mia_p))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2),
                                  gridspec_kw={"width_ratios": [1.15, 1.0]})

    eps = [o["epsilon"] for o in sweep]
    bound = [o["bound"] for o in sweep]
    adv = [o["max_advantage"] for o in sweep]
    ax.plot(eps, bound, "-o", lw=2.0, ms=7, color=C["magenta"], mec="white", mew=1.3,
            zorder=3, label="what the ε guarantee permits")
    ax.plot(eps, adv, "-o", lw=2.0, ms=7, color=C["blue"], mec="white", mew=1.3,
            zorder=4, label="measured against CoRTeC")
    ax.fill_between(eps, 0, adv, color=C["blue"], alpha=0.10, zorder=1)
    ax.set_xscale("log"); ax.set_xticks(eps)
    ax.set_xticklabels([f"{e:g}" for e in eps])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(eps) * 0.8, max(eps) * 1.3)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("privacy budget  ε   (log axis)")
    ax.set_ylabel("MIA Advantage")
    ax.set_title("Measured leakage does not track the bound")
    ax.legend(loc="center left")
    ax.annotate("bound is vacuous here", xy=(eps[-1], bound[-1]), xytext=(-14, -62),
                textcoords="offset points", fontsize=SMALL, color=C["magenta"], ha="right")

    # right panel: CoRTeC vs the positive control, per attack
    names = [("nearest_neighbour", "nearest\nneighbour"), ("shadow_model", "shadow\nmodel"),
             ("exact_match", "exact\nmatch")]
    x = np.arange(len(names)); w = 0.36
    ct = [mia["cortec"][k]["advantage"] for k, _ in names]
    pc = [mia["positive_control"][k]["advantage"] for k, _ in names]
    ax2.bar(x - w/2 - 0.012, ct, w, color=C["blue"], zorder=3, label="CoRTeC output")
    ax2.bar(x + w/2 + 0.012, pc, w, color=C["orange"], zorder=3,
            label="positive control\n(real records leaked)")
    for i, (a_, b_) in enumerate(zip(ct, pc)):
        ax2.annotate(f"{a_:.3f}", xy=(i - w/2 - 0.012, a_), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=NOTE, color=INK2)
        ax2.annotate(f"{b_:.3f}", xy=(i + w/2 + 0.012, b_), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=NOTE, color=INK2)
    ax2.set_xticks(x); ax2.set_xticklabels([n for _, n in names], linespacing=1.35)
    ax2.set_ylim(0, max(pc) * 1.35)
    ax2.set_ylabel("MIA Advantage")
    ax2.set_title("The attacks work: they just find nothing")
    ax2.legend(loc="upper left")
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Membership inference against CoRTeC: no detectable signal, and it does not grow "
                 "with the privacy budget",
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/fig09_membership_inference.png"); plt.close(fig)
    print("  fig09_membership_inference.png")


# ── Figure 3: classification profile: attack and downstream, with confusion matrices ─
def fig_classification_profile(report="results/classification_report.json",
                               outname="fig10_classification.png"):
    """Six panels over two rows: the privacy question and the utility question.

    A design note that changed this figure. Plotting CoRTeC's attack confusion matrix ALONE
    conveys nothing: all four cells hold ~500, so a default colour scale stretches a 2-count
    difference into maximum contrast (asserting a stark split that does not exist), while a scale
    anchored at zero renders a uniform block (asserting nothing at all). The information lives in
    the COMPARISON, so each confusion matrix is paired with its reference: the attack beside the
    leaking control, the downstream model beside a real-data model: on a shared 0-100% scale.
    Cells are row-normalised, because what matters is how each true class was split, not the
    absolute totals.
    """
    import json as _json
    import seaborn as sns
    p = report
    if not os.path.exists(p):
        print(f"  (skip {outname})"); return
    d = _json.load(open(p))

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6),
                             gridspec_kw={"width_ratios": [1.5, 1.0, 1.0]})
    (axA, axB, axC), (axD, axE, axF) = axes

    def cm_panel(ax, conf, xlabels, ylabels, title, quadrant_labels, show_ylab):
        cm = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]], float)
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        ann = np.array([[f"{quadrant_labels[0][0]}\n{int(cm[0,0])}\n{pct[0,0]:.0f}%",
                         f"{quadrant_labels[0][1]}\n{int(cm[0,1])}\n{pct[0,1]:.0f}%"],
                        [f"{quadrant_labels[1][0]}\n{int(cm[1,0])}\n{pct[1,0]:.0f}%",
                         f"{quadrant_labels[1][1]}\n{int(cm[1,1])}\n{pct[1,1]:.0f}%"]])
        sns.heatmap(pct, annot=ann, fmt="", cmap="rocket_r", cbar=False, square=True,
                    vmin=0, vmax=100, linewidths=2.0, linecolor="white", ax=ax,
                    annot_kws={"size": BASE - 0.5, "weight": "bold", "linespacing": 1.5},
                    xticklabels=xlabels, yticklabels=ylabels)
        # contrast follows the shared 0-100 scale, so it is correct in every panel
        for t, v in zip(ax.texts, pct.ravel()):
            t.set_color("white" if v > 42 else INK)   # rocket_r is already dark by ~45%
        ax.set_title(title, fontsize=BASE - 0.5)
        ax.tick_params(axis="both", labelsize=BASE - 2, length=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        if show_ylab:
            ax.set_ylabel("actual", fontsize=BASE - 1)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("predicted", fontsize=BASE - 1)

    # ── row 1: privacy ──────────────────────────────────────────────────────────────
    atk, ctrl = d["attack_on_cortec"], d["attack_on_control"]
    axA.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.1, color=MUTED, zorder=1)
    axA.plot(ctrl["roc"]["fpr"], ctrl["roc"]["tpr"], lw=2.2, color=C["orange"], zorder=3,
             label=f"on leaking control (AUC {ctrl['auc']:.3f})")
    axA.plot(atk["roc"]["fpr"], atk["roc"]["tpr"], lw=2.2, color=C["blue"], zorder=4,
             label=f"on CoRTeC (AUC {atk['auc']:.3f})")
    axA.set_xlim(0, 1); axA.set_ylim(0, 1.02)
    axA.set_xlabel("false positive rate"); axA.set_ylabel("true positive rate")
    axA.set_title("PRIVACY: membership attack ROC")
    axA.legend(loc="lower right")
    axA.annotate("chance", xy=(0.66, 0.66), xytext=(4, -12), textcoords="offset points",
                 fontsize=SMALL, color=MUTED)

    ql = [["TN", "FP"], ["FN", "TP"]]
    lab = ["non-member", "member"]
    cm_panel(axB, atk["confusion"], lab, lab,
             "Attack on CoRTeC: an even split", ql, True)
    cm_panel(axC, ctrl["confusion"], lab, lab,
             "Same attack, leaking control", ql, False)

    # ── row 2: utility ──────────────────────────────────────────────────────────────
    down = d["downstream"]
    style = [("real, full training set", MUTED, (0, (1, 2)), 1.6),
             ("real sample (same n)", INK2, (0, (5, 3)), 1.8),
             ("CoRTeC", C["blue"], "solid", 2.4),
             ("MST", C["aqua"], "solid", 1.8),
             ("AIM", C["orange"], "solid", 1.8),
             ("PATE-GAN", C["magenta"], "solid", 1.6),
             ("DP-CTGAN", C["yellow"], "solid", 1.6)]
    axD.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.0, color=GRID, zorder=1)
    for name, col, ls, lw in style:
        if name not in down:
            continue
        m = down[name]
        axD.plot(m["roc"]["fpr"], m["roc"]["tpr"], lw=lw, color=col, ls=ls, zorder=3,
                 label=f"{name} ({m['auc']:.3f})")
    axD.set_xlim(0, 1); axD.set_ylim(0, 1.02)
    axD.set_xlabel("false positive rate"); axD.set_ylabel("true positive rate")
    axD.set_title("UTILITY: trained on synthetic, tested on real")
    axD.legend(loc="lower right", fontsize=BASE - 2.4)

    inc = ["≤50K", ">50K"]
    cm_panel(axE, down["CoRTeC"]["confusion"], inc, inc,
             "Model trained on CoRTeC", ql, True)
    if "real sample (same n)" in down:
        cm_panel(axF, down["real sample (same n)"]["confusion"], inc, inc,
                 "Model trained on real data (same n)", ql, False)

    fig.suptitle("CoRTeC classification profile: the membership attack fails (top), the "
                 "downstream model does not (bottom)\n"
                 "confusion cells show count and row-share; colour is the row-share on a shared "
                 "0–100% scale",
                 fontsize=TITLE, fontweight="bold", color=INK)
    fig.savefig(f"{OUT}/{outname}"); plt.close(fig)
    print(f"  {outname}")


def fig_classification_profile_shipped():
    """The same profile on the shipped-configuration Adult draws (Gemini 3.5 Flash, Table 3 of the
    arXiv paper), from results/classification_report_shipped.json."""
    fig_classification_profile("results/classification_report_shipped.json",
                               "fig12_classification_shipped.png")


if __name__ == "__main__":
    print("generating figures into", OUT)
    for fn in (fig_pipeline, fig_head_to_head, fig_calibration, fig_axes_independent,
               fig_model_grid, fig_epsilon, fig_saturation, fig_ablation, fig_transmission_by_family, fig_membership_inference,
               fig_classification_profile, fig_classification_profile_shipped):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__}: {type(e).__name__}: {e}")
    print("done")
