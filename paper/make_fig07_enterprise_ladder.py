"""Figure 7: enterprise-platform generators on one DP release.

`make_figures.py` once wrote a different figure (open-weight models against parameter scale) to
the same filename, so whichever script ran last decided what Figure 7 showed, and for a while the
paper carried this figure under a caption describing the other one. `make_figures.py` now writes
that figure under a superseded name; see the note there.
"""

import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"blue": "#2a78d6", "orange": "#d1541f", "green": "#0f8f68", "yellow": "#a8770a",
     "magenta": "#c4306b", "violet": "#4a3aa7", "grey": "#6b7280"}

def mae(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    p = [(x["dp_shown"], x["generated"]) for x in d["points"] if x.get("generated") is not None]
    return float(np.mean([abs(g - s) for s, g in p])) if p else None

# transmission: every entry measured on the identical sweep
T = [("Gemini 3.1 Pro",  "results/apiladder_gemini31pro/summary.json", "on"),
     ("Gemini 3.5 Flash","results/transmission_gemini/summary.json",   "on"),
     ("GPT-5",           "results/apiladder_gpt5/summary.json",        "on"),
     ("Claude Fable 5",  "results/apiladder_fable5_matched/summary.json","on"),
     ("Claude Sonnet 5", "results/apiladder_sonnet5/summary.json",     "on"),
     ("Claude Opus 5",   "results/apiladder_opus5/summary.json",       "on"),
     ("Claude Haiku 4.5","results/apiladder_haiku45/summary.json",     "none")]
T = [(n, mae(p), r) for n, p, r in T]
T = sorted([t for t in T if t[1] is not None], key=lambda x: x[1])

# generation, same DP release
G = json.load(open("results/api_model_ladder.json"))
def gm(lab, k="cond_tv_seen"):
    r = G["results"].get(lab)
    return float(np.nanmean([x[k] for x in r])) if r else None
GEN = [("Gemini 3.1 Pro",   "CoRTeC (Gemini 3.1 Pro)", "on"),
       ("GPT-5",            "CoRTeC (GPT-5, default) +2", "on"),
       ("Claude Opus 5",    "CoRTeC (Claude Opus 5)", "on"),
       ("Claude Fable 5",   "CoRTeC (Claude Fable 5)", "on"),
       # The MATCHED reasoning arm (same seed, same DP release). The unmatched run scored 0.091
       # and an earlier draw of it 0.084; using either here would contradict the table in 7.5.
       ("Claude Sonnet 5",  "CoRTeC (Sonnet 5, reasoning on, matched)", "on"),
       ("Claude Sonnet 5",  "CoRTeC (Claude Sonnet 5, suppressed)", "suppressed"),
       ("Claude Haiku 4.5", "CoRTeC (Claude Haiku 4.5)", "none"),
       ("GPT-5",            "CoRTeC (GPT-5, minimal)", "suppressed")]
GEN = [(n, gm(k), r) for n, k, r in GEN]
GEN = sorted([g for g in GEN if g[1] is not None], key=lambda x: x[1])
floor = gm("permuted target (no-information floor)")
realn = gm("real sample, n=300")

col = {"on": C["blue"], "suppressed": C["orange"], "none": C["grey"]}
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3),
                         gridspec_kw={"width_ratios": [1.0, 1.15]})

ax = axes[0]
y = np.arange(len(T))[::-1]
ax.barh(y, [t[1] for t in T], color=[col[t[2]] for t in T], height=0.62, zorder=3)
for yy, (n, v, r) in zip(y, T):
    ax.text(v + 0.0022, yy, f"{v:.3f}", va="center", fontsize=8.4, color="#222")
ax.set_yticks(y); ax.set_yticklabels([t[0] for t in T], fontsize=9)
ax.set_xlabel("transmission magnitude error  (mean |generated − released|, lower is better)", fontsize=8.6)
ax.set_title("Transmission: does the released rate reach the output?", fontsize=9.8, pad=8)
ax.set_xlim(0, max(t[1] for t in T) * 1.30)
ax.grid(axis="x", alpha=0.25, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

ax = axes[1]
y2 = np.arange(len(GEN))[::-1]
ax.barh(y2, [g[1] for g in GEN], color=[col[g[2]] for g in GEN], height=0.62, zorder=3)
for yy, (n, v, r) in zip(y2, GEN):
    ax.text(v + 0.004, yy, f"{v:.3f}", va="center", fontsize=8.4, color="#222")
lbl = [f"{g[0]}" + ("" if g[2] == "on" else f"  ({g[2]})") for g in GEN]
ax.set_yticks(y2); ax.set_yticklabels(lbl, fontsize=9)
ax.axvline(floor, color=C["magenta"], ls="--", lw=1.6, zorder=1)
ax.axvline(realn, color=C["green"], ls=":", lw=1.6, zorder=1)
# Reference labels go ABOVE the plot area, horizontal: rotated in-panel labels collided with the
# bars and with the value annotations, which is the whole reason the old figure was rebuilt.
ymax = ax.get_ylim()[1]
ax.annotate("real sample, n=300", xy=(realn, ymax), xytext=(realn, ymax + 0.42),
            color=C["green"], fontsize=8.0, ha="center", va="bottom",
            annotation_clip=False)
ax.annotate("no-information floor", xy=(floor, ymax), xytext=(floor, ymax + 0.42),
            color=C["magenta"], fontsize=8.0, ha="center", va="bottom",
            annotation_clip=False)
ax.set_ylim(ax.get_ylim()[0], ymax + 1.1)
ax.set_xlabel("conditional fidelity error on full-dataset generation (lower is better)", fontsize=8.6)
ax.set_title("Generation: does that structure survive a whole dataset?", fontsize=9.8, pad=8)
ax.set_xlim(0, max(max(g[1] for g in GEN), floor) * 1.18)
ax.grid(axis="x", alpha=0.25, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=col["on"], label="reasoning enabled"),
                    Patch(facecolor=col["suppressed"], label="reasoning suppressed"),
                    Patch(facecolor=C["grey"], label="no reasoning mode")],
           loc="lower center", ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Enterprise-platform generators, same DP release: transmission and generation are different axes",
             fontsize=10.6, y=0.99)
fig.tight_layout(rect=[0, 0.09, 1, 0.93])
fig.savefig("paper/figures/fig07_model_grid.png", dpi=190)
print("  wrote paper/figures/fig07_model_grid.png")
print(f"  transmission: {[(n, round(v,3)) for n,v,_ in T]}")
print(f"  generation  : {[(n+('' if r=='on' else f' [{r}]'), round(v,3)) for n,v,r in GEN]}")
print(f"  floor {floor:.3f} | real sample {realn:.3f}")
