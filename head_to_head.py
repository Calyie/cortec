"""
head_to_head.py — CoRTeC against the DP synthesisers a regulated institution would actually
deploy, on fidelity and on downstream utility, with the statistics stated properly.

This is the comparison the paper turns on, so the statistical treatment is deliberately
conservative:

  * **Every point estimate carries a bootstrap CI**, not a bare mean over a handful of draws.
  * **Both a parametric and a rank test**, with the rank test's *attainable floor* reported
    beside its p-value. At small group sizes the smallest two-sided Mann-Whitney p is
    2/C(n_a+n_b, n_a) whatever the effect size; quoting "p = 0.095, not significant" without that
    floor invites a reader to conclude an effect is absent when the design could not have
    detected it.
  * **Holm-Bonferroni correction across the whole family of comparisons**, because we run three
    students against several baselines on several datasets and an uncorrected minimum p-value
    over ~30 tests is not evidence of anything.
  * **Hedges' g with its CI**, since with few draws the effect size is the durable quantity and
    the p-value is the one limited by draw count.
  * **Floors and ceilings in every table.** A method is judged against a real sample at matched n
    (what it could achieve) and a target-permuted floor (what "no information" scores), never in
    isolation.

Usage:
    python head_to_head.py --comparison results/FINAL_comparison_v2.json --treatment cortec-v3
"""
from __future__ import annotations
import argparse, json, math

import numpy as np
from scipy import stats

# Two evaluation scripts in this project emit different key names for the same quantities
# (the Adult suite and the spec-driven generic evaluator). Resolve them per file rather than
# forcing one, so a comparison never silently drops a metric because of a naming mismatch.
_SCHEMAS = [
    {"tv1": "marginal_tv_1way", "tv2": "marginal_tv_2way", "cs": "cond_tv_seen",
     "ch": "cond_tv_heldout", "lr": "tstr_auc_LR", "rf": "tstr_auc_RF", "gbm": "tstr_auc_GBM"},
    {"tv1": "tv_1way", "tv2": "tv_2way", "cs": "cond_tv_seen", "ch": "cond_tv_heldout",
     "lr": "tstr_LR", "rf": "tstr_RF", "gbm": "tstr_GBM"},
]
LABELS = {"tv1": "1-way TV", "tv2": "2-way TV", "cs": "cond. TV (seen)",
          "ch": "cond. TV (held-out)", "lr": "TSTR-LR", "rf": "TSTR-RF", "gbm": "TSTR-GBM"}


def resolve_schema(pool):
    """Pick the key naming this results file actually uses."""
    sample = next((v[0] for v in pool.values() if isinstance(v, list) and v), {})
    for sch in _SCHEMAS:
        if sch["lr"] in sample:
            return sch
    raise SystemExit(f"unrecognised metric keys: {sorted(sample)[:12]}")


def boot_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))


def mw_floor(na, nb):
    return min(1.0, 2.0 / math.comb(na + nb, na)) if na >= 1 and nb >= 1 else float("nan")


def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan")
    sp = math.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    if sp == 0:
        return float("nan"), float("nan"), float("nan")
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3/(4*(na+nb) - 9)
    g = d * J
    se = math.sqrt((na+nb)/(na*nb) + g**2/(2*(na+nb-2)))
    return float(g), float(g - 1.96*se), float(g + 1.96*se)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down. Returns {key: adjusted p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)      # enforce monotonicity
        adj[k] = running
    return adj


def compare(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "diff": float(a.mean() - b.mean()), "mw_floor": mw_floor(len(a), len(b))}
    if len(a) > 1 and len(b) > 1:
        out["welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        out["mw_p"] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        g, lo, hi = hedges_g(a, b)
        out.update(hedges_g=g, g_lo=lo, g_hi=hi)
    else:
        out.update(welch_p=float("nan"), mw_p=float("nan"),
                   hedges_g=float("nan"), g_lo=float("nan"), g_hi=float("nan"))
    out["powered"] = bool(out["mw_floor"] <= 0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", default="results/FINAL_comparison_v2.json")
    ap.add_argument("--treatment", default="cortec-v3")
    ap.add_argument("--baselines", nargs="+",
                    default=["aim", "mst", "patectgan", "dpctgan", "pategan", "mwem"])
    ap.add_argument("--floor", default="real-shuffled-income [FLOOR]")
    ap.add_argument("--ceiling", default="real-sample-300")
    ap.add_argument("--label", default="Adult")
    ap.add_argument("--out", default="results/head_to_head.json")
    a = ap.parse_args()

    d = json.load(open(a.comparison))
    ref = d.get("_reference", {})
    pool = {**{k: v for k, v in d.items() if k not in ("_meta", "_reference")}, **ref}
    if a.treatment not in pool:
        raise SystemExit(f"{a.treatment!r} not found; have {sorted(pool)}")

    sch = resolve_schema(pool)
    FIDELITY = [(sch["tv1"], LABELS["tv1"], False), (sch["tv2"], LABELS["tv2"], False),
                (sch["cs"], LABELS["cs"], False), (sch["ch"], LABELS["ch"], False)]
    UTILITY = [(sch["lr"], LABELS["lr"], True), (sch["rf"], LABELS["rf"], True),
               (sch["gbm"], LABELS["gbm"], True)]
    present = [b for b in a.baselines if b in pool and pool[b]]
    W = 108
    print("=" * W)
    print(f"HEAD-TO-HEAD — {a.label}: {a.treatment} against deployed DP synthesisers")
    print("=" * W)

    # ── descriptive table with bootstrap CIs ───────────────────────────────────────
    rows = [a.treatment] + present + [k for k in (a.ceiling, a.floor) if k in pool]
    for metrics, title in ((FIDELITY, "FIDELITY (lower is better)"),
                           (UTILITY, "DOWNSTREAM UTILITY (higher is better)")):
        print(f"\n{title}")
        print(f"{'condition':30s} {'n':>3s} " +
              " ".join(f"{lab:>22s}" for _, lab, _ in metrics))
        for name in rows:
            vals = pool[name]
            cells = []
            for key, _, _ in metrics:
                xs = [r[key] for r in vals if r.get(key) is not None]
                if not xs:
                    cells.append(f"{'--':>22s}"); continue
                m, lo, hi = boot_ci(xs)
                cells.append(f"{m:8.3f} [{lo:.3f},{hi:.3f}]" if len(xs) > 1
                             else f"{m:8.3f} {'(1 draw)':>13s}")
            print(f"{name:30s} {len(vals):>3d} " + " ".join(cells))

    # ── inferential comparisons, Holm-corrected across the whole family ────────────
    raw, cells = {}, {}
    for base in present:
        for key, lab, higher in UTILITY + FIDELITY:
            t = [r[key] for r in pool[a.treatment] if r.get(key) is not None]
            b = [r[key] for r in pool[base] if r.get(key) is not None]
            if len(t) < 2 or len(b) < 2:
                continue
            c = compare(t, b)
            k = f"{base}|{lab}"
            cells[k] = c
            raw[k] = c["welch_p"]
    adj = holm(raw) if raw else {}

    print(f"\n{'='*W}")
    print("INFERENCE — Welch p, Holm-adjusted across the whole family of "
          f"{len(raw)} comparisons")
    print("(MW floor = smallest two-sided Mann-Whitney p attainable at these group sizes)")
    print("=" * W)
    print(f"{'baseline':12s} {'metric':22s} {'CoRTeC':>8s} {'base':>8s} {'diff':>9s} "
          f"{'welch p':>9s} {'holm p':>8s} {'MW p':>7s} {'floor':>7s} {'g [95% CI]':>22s}")
    for k, c in cells.items():
        base, lab = k.split("|")
        gtxt = (f"{c['hedges_g']:6.1f} [{c['g_lo']:5.1f},{c['g_hi']:5.1f}]"
                if np.isfinite(c["hedges_g"]) else f"{'--':>22s}")
        star = "*" if adj.get(k, 1) < 0.05 else " "
        print(f"{base:12s} {lab:22s} {c['mean_a']:8.3f} {c['mean_b']:8.3f} {c['diff']:+9.3f} "
              f"{c['welch_p']:9.4f} {adj.get(k, float('nan')):8.4f}{star} {c['mw_p']:7.3f} "
              f"{c['mw_floor']:7.3f} {gtxt}")

    n_sig = sum(1 for k in cells if adj.get(k, 1) < 0.05)
    print("-" * W)
    print(f"{n_sig} of {len(cells)} comparisons significant at Holm-adjusted alpha = 0.05.")
    unpowered = {k for k, c in cells.items() if not c["powered"]}
    if unpowered:
        print(f"NOTE: {len(unpowered)} comparison(s) cannot reach a significant RANK test at "
              f"their group sizes whatever the effect size; read their Welch p as descriptive.")

    json.dump({"label": a.label, "treatment": a.treatment,
               "comparisons": cells, "holm_adjusted": adj}, open(a.out, "w"),
              indent=2, default=str)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
