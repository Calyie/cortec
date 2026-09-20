"""
report_epsilon_sweep.py — assemble the per-epsilon evaluations into a privacy-utility curve.

The question is not "who wins at each epsilon" but **how fast each method degrades as the budget
tightens**, because that determines which method survives at the epsilon a cautious regulator
would actually impose. MST publishes at eps in {0.3, 1.0, 8.0}; AIM's practical regime is
[0.1, 10]; everything in this project until now was a single point at eps = 2.0.

Reported per epsilon and per method: marginal fidelity, conditional fidelity (on the groups the
method is told about and on strictly held-out groups), and utility under three students —
always alongside the two references that make the numbers readable, the real-sample floor at the
same n and the shuffled-target floor.
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path

import numpy as np


def load(root: Path) -> dict:
    out = {}
    for p in sorted(glob.glob(str(root / "eval_eps*.json"))):
        m = re.search(r"eval_eps([0-9p]+)\.json$", p)
        if not m:
            continue
        eps = float(m.group(1).replace("p", "."))
        out[eps] = json.load(open(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/diabetes_eps")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="eps=path/to/eval.json for epsilons evaluated outside the sweep")
    ap.add_argument("--dataset", default="diabetes")
    a = ap.parse_args()

    data = load(Path(a.root))
    for e in a.extra:
        eps, path = e.split("=", 1)
        if Path(path).exists():
            data[float(eps)] = json.load(open(path))

    if not data:
        raise SystemExit(f"no eval_eps*.json under {a.root}")

    metrics = [("tv_1way", "1-way TV ↓"), ("cond_tv_seen", "cond seen ↓"),
               ("cond_tv_heldout", "cond held ↓"), ("tstr_LR", "TSTR-LR ↑"),
               ("tstr_RF", "TSTR-RF ↑")]
    methods = sorted({m for d in data.values() for m in d})
    order = [m for m in ["cortec", "mst", "aim", "patectgan", "header-only"] if m in methods]
    order += [m for m in methods if m not in order and "FLOOR" not in m and "CEILING" not in m]
    refs = [m for m in methods if "FLOOR" in m or "CEILING" in m]

    def val(d, meth, key):
        rows = d.get(meth)
        if not rows:
            return None
        v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(v)) if v else None

    print(f"\n{'=' * 104}")
    print(f"PRIVACY-UTILITY CURVE — {a.dataset}   (epsilon_total; MST publishes at 0.3/1.0/8.0)")
    print("=" * 104)
    eps_list = sorted(data)
    for key, label in metrics:
        print(f"\n{label}")
        print(f"  {'method':26s} " + " ".join(f"{'eps=' + f'{e:g}':>11s}" for e in eps_list)
              + f" {'  degradation':>16s}")
        for meth in order + refs:
            cells, first, last = [], None, None
            for e in eps_list:
                v = val(data[e], meth, key)
                cells.append("—" if v is None else f"{v:.3f}")
                if v is not None:
                    last = v
                    if first is None:
                        first = v
            if all(c == "—" for c in cells):
                continue
            deg = ""
            if first is not None and last is not None and first != last:
                # tightest -> loosest epsilon: how much worse is the strictest budget?
                if key.startswith("tstr"):
                    deg = f"{last - first:+.3f} AUC"
                else:
                    deg = f"x{first / last:.1f}" if last > 0 else ""
            print(f"  {meth:26s} " + " ".join(f"{c:>11s}" for c in cells) + f" {deg:>16s}")

    print("\nDegradation column: for error metrics, how many times worse the tightest epsilon is")
    print("than the loosest; for TSTR, the AUC change from tightest to loosest.")
    print("A method whose error barely moves across the sweep is one whose quality is set by")
    print("something other than the privacy budget.")


if __name__ == "__main__":
    main()
