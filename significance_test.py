"""
significance_test.py — the utility comparison in §6.3, as a script rather than an ad-hoc
calculation.

It answers one question: at matched n and matched epsilon, is CoRTeC's downstream utility
separated from AIM's and MST's by more than draw-to-draw noise? Two tests are reported for every
comparison, because they fail differently:

  * Welch's t-test, which does not assume equal variances but does assume approximate normality
    of the draw means — reasonable here, but it is the *parametric* claim;
  * Mann-Whitney U, which assumes nothing about the distribution but has very little power at
    small group sizes.

The second one carries a trap this paper hit once already. With n_a draws against n_b, the
smallest two-sided p-value Mann-Whitney can *possibly* return is 2 / C(n_a + n_b, n_a), whatever
the effect size: at 2 vs 5 that floor is 0.095, so the test cannot reach significance even if the
two groups are perfectly separated. Reporting "p = 0.095, not significant" without that floor
invites a reader to conclude the effect is absent when the test simply could not have detected it.
Every row therefore carries the achievable floor next to the p-value, and the script says plainly
whether the comparison was powered at all.
"""
from __future__ import annotations
import argparse, json, math
from itertools import combinations

import numpy as np
from scipy import stats

STUDENTS = [("tstr_auc_LR", "LR"), ("tstr_auc_RF", "RF"), ("tstr_auc_GBM", "GBM")]


def mw_floor(na: int, nb: int) -> float:
    """Smallest attainable two-sided Mann-Whitney p at these group sizes."""
    if na < 1 or nb < 1:
        return float("nan")
    return min(1.0, 2.0 / math.comb(na + nb, na))


def compare(a: list[float], b: list[float]) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"n_a": len(a), "n_b": len(b),
           "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "sd_a": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
           "sd_b": float(b.std(ddof=1)) if len(b) > 1 else float("nan"),
           "diff": float(a.mean() - b.mean()),
           "mw_floor": mw_floor(len(a), len(b))}
    if len(a) > 1 and len(b) > 1:
        out["welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        out["mw_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        # Hedges-corrected Cohen's d: the effect size is the durable quantity here, since the
        # p-values are the part limited by draw count.
        sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                       / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp if sp > 0 else float("nan")
        J = 1 - 3 / (4 * (len(a) + len(b)) - 9)
        out["hedges_g"] = float(d * J)
    else:
        out["welch_p"] = out["mw_p"] = out["hedges_g"] = float("nan")
    out["powered"] = bool(out["mw_floor"] <= 0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", default="results/FINAL_comparison.json")
    ap.add_argument("--treatment", default="cortec-v3")
    ap.add_argument("--baselines", nargs="+", default=["aim", "mst"])
    ap.add_argument("--out", default="results/significance.json")
    a = ap.parse_args()

    d = json.load(open(a.comparison))
    if a.treatment not in d:
        raise SystemExit(f"{a.treatment!r} not in {a.comparison}; have {list(d)}")

    results, rows = {}, []
    for base in a.baselines:
        if base not in d:
            print(f"  (skip {base}: absent from {a.comparison})")
            continue
        for key, name in STUDENTS:
            t = [r[key] for r in d[a.treatment] if r.get(key) is not None]
            b = [r[key] for r in d[base] if r.get(key) is not None]
            if not t or not b:
                continue
            c = compare(t, b)
            results[f"{a.treatment}_vs_{base}_{name}"] = c
            rows.append((f"{a.treatment} − {base}", name, c))

    w = 104
    print("=" * w)
    print(f"DOWNSTREAM UTILITY (TSTR-AUC): {a.treatment} against each baseline")
    print("=" * w)
    print(f"{'comparison':24s} {'student':>7s} {'treat':>7s} {'base':>7s} {'diff':>8s} "
          f"{'Welch p':>9s} {'MW p':>8s} {'MW floor':>9s} {'g':>7s}")
    for label, name, c in rows:
        print(f"{label:24s} {name:>7s} {c['mean_a']:>7.3f} {c['mean_b']:>7.3f} "
              f"{c['diff']:>+8.3f} {c['welch_p']:>9.4f} {c['mw_p']:>8.3f} "
              f"{c['mw_floor']:>9.3f} {c['hedges_g']:>7.2f}")

    if rows:
        na, nb = rows[0][2]["n_a"], rows[0][2]["n_b"]
        floor = rows[0][2]["mw_floor"]
        print("-" * w)
        print(f"draws: {na} {a.treatment} vs {nb} per baseline.")
        if floor > 0.05:
            print(f"** UNDERPOWERED: with {na} vs {nb} draws the smallest two-sided Mann-Whitney")
            print(f"   p-value attainable is {floor:.3f}, so the rank test CANNOT reach p < 0.05")
            print(f"   however large the effect. Report the effect size and the consistency of")
            print(f"   the sign across students and datasets, not the rank-test p-value.")
        else:
            print(f"   Mann-Whitney is powered at these group sizes (attainable floor "
                  f"{floor:.4f} < 0.05).")
    json.dump({"draws": {a.treatment: rows[0][2]["n_a"] if rows else 0},
               "comparisons": results}, open(a.out, "w"), indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
