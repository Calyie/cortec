"""
explain_utility_gap.py — why does a method with good marginals train a bad model?

The head-to-head shows MST attaining the best 1-way marginal fidelity in our comparison while
training models 0.15 AUC worse than CoRTeC. That is a claim about an *outcome*; this script asks
for the mechanism, because "marginal methods lose on utility" is only interesting if we can say
what they lose.

It decomposes a synthetic dataset's departure from real data into three levels, all measured
against the same held-out real data and all comparable across methods:

  1. **Per-feature target association.** For each feature, the mutual information I(X_j ; Y) and
     the signed lift of P(Y=1 | X_j = v) against the base rate. A method can match the marginal
     of X_j and the marginal of Y exactly while destroying their association — this is precisely
     what permuting the target column does, and it is invisible to 1-way TV.

  2. **Pairwise feature dependence.** Mutual information I(X_j ; X_k) over feature pairs. A model
     exploits redundancy and interaction between features; a synthesiser that renders features
     independent gives a downstream model less to work with even when every 1-way marginal is
     perfect.

  3. **Conditional-structure realisability.** How much of the real data's predictive signal is
     recoverable at all: we fit the same student on the synthetic data and read its learned
     coefficients/importances against those learned on real data, so a method that inverts or
     flattens a relationship is visible as a sign flip or a collapse toward zero.

The output is a per-method profile, so a claim like "MST preserves 1-way marginals and the
selected pairwise structure but loses the feature-target association that the student needs"
becomes a measurement rather than an assertion.
"""
from __future__ import annotations
import argparse, glob, json, sys, warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


def _bin_frame(df: pd.DataFrame, num_cols, cat_cols, edges: dict) -> pd.DataFrame:
    """Discretise to a common public grid so every method is compared on identical cells."""
    out = pd.DataFrame(index=df.index)
    for c in num_cols:
        e = np.asarray(edges[c], dtype=float)
        v = pd.to_numeric(df[c], errors="coerce").fillna(e[0]).clip(e[0], e[-1])
        out[c] = np.clip(np.digitize(v, e) - 1, 0, len(e) - 2).astype(int).astype(str)
    for c in cat_cols:
        out[c] = df[c].astype(str).str.strip()
    return out


def target_association(binned: pd.DataFrame, y: np.ndarray, cols) -> dict:
    """I(X_j ; Y) per feature."""
    return {c: float(mutual_info_score(binned[c].values, y)) for c in cols}


def pairwise_dependence(binned: pd.DataFrame, cols, max_pairs=60, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    pairs = list(combinations(cols, 2))
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    return {f"{a}|{b}": float(mutual_info_score(binned[a].values, binned[b].values))
            for a, b in pairs}


def student_coefficients(binned: pd.DataFrame, y: np.ndarray, cols, enc) -> np.ndarray:
    X = enc.transform(binned[cols])
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1])
    m = LogisticRegression(max_iter=2000, C=1.0)
    m.fit(X, y)
    return m.coef_.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--spec-json", required=True, help='{label: [globs]}')
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--matched-n", type=int, default=300,
                    help="reference real-sample size; MI is only comparable at matched n")
    ap.add_argument("--n-reference-samples", type=int, default=25)
    a = ap.parse_args()

    from src.dataset_spec import get_spec
    import src.datasets_extra  # noqa: F401
    from sklearn.model_selection import train_test_split

    spec = get_spec(a.dataset)
    df = spec.loader(); spec.validate(df)
    train, test = train_test_split(df, test_size=0.2, random_state=a.seed,
                                   stratify=df[spec.target_col])
    num, cat = spec.numerical_cols, spec.categorical_cols
    cols = num + cat
    edges = {c: np.asarray(spec.public_bins[c], dtype=float) for c in num}

    real_b = _bin_frame(train, num, cat, edges)
    real_y = spec.is_positive(train).astype(int).values
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(real_b[cols])
    base_rate = float(real_y.mean())

    # The decision rule is learned on ALL real training data: that is the target a synthetic set
    # is trying to reproduce, and a logistic fit on 26k rows is stable enough to compare against.
    real_coef = student_coefficients(real_b, real_y, cols, enc)

    # Mutual information is NOT comparable across sample sizes — the plug-in estimator is
    # upward-biased and the bias grows as n shrinks. Comparing a 300-row synthetic set against a
    # 26,048-row reference therefore measures sample size, not method quality: on a first run a
    # genuine 300-row REAL sample scored 2.7x "retention", which is impossible and was the signal
    # that the reference was wrong. The reference is instead the mean over repeated REAL samples
    # at the same n, so retention of 1.0 means "indistinguishable from real data of this size".
    n_ref = int(a.matched_n)
    rng = np.random.default_rng(a.seed)
    ref_mi_y, ref_mi_xx = [], []
    for _ in range(a.n_reference_samples):
        idx = rng.choice(len(train), size=min(n_ref, len(train)), replace=False)
        sb = real_b.iloc[idx]
        sy = real_y[idx]
        ref_mi_y.append(target_association(sb, sy, cols))
        ref_mi_xx.append(pairwise_dependence(sb, cols, seed=a.seed))
    real_mi_y = {c: float(np.mean([r[c] for r in ref_mi_y])) for c in cols}
    real_mi_xx = {k: float(np.mean([r[k] for r in ref_mi_xx])) for k in ref_mi_xx[0]}

    spec_map = json.load(open(a.spec_json))
    rows = []
    for label, globs in spec_map.items():
        files = sorted(f for g in globs for f in glob.glob(g))
        if not files:
            print(f"  (skip {label}: no files)"); continue
        per = []
        for f in files:
            s = pd.read_csv(f, keep_default_na=False)
            if not set(cols + [spec.target_col]).issubset(s.columns):
                continue
            sb = _bin_frame(s, num, cat, edges)
            sy = spec.is_positive(s).astype(int).values
            if len(np.unique(sy)) < 2:
                continue
            mi_y = target_association(sb, sy, cols)
            mi_xx = pairwise_dependence(sb, cols, seed=a.seed)
            coef = student_coefficients(sb, sy, cols, enc)

            # how much of the real feature->target association is retained, per feature
            ret = np.array([mi_y[c] / real_mi_y[c] if real_mi_y[c] > 1e-9 else np.nan
                            for c in cols], dtype=float)
            # pairwise dependence retention
            retx = np.array([mi_xx[k] / real_mi_xx[k] if real_mi_xx[k] > 1e-9 else np.nan
                             for k in real_mi_xx], dtype=float)
            # agreement of the learned decision rule with the one learned on real data
            denom = (np.linalg.norm(coef) * np.linalg.norm(real_coef))
            cos = float(coef @ real_coef / denom) if denom > 0 else np.nan
            signflip = float(np.mean(np.sign(coef) != np.sign(real_coef)))

            per.append({"file": f, "assoc_retention": float(np.nanmean(ret)),
                        "assoc_retention_median": float(np.nanmedian(ret)),
                        "pair_dep_retention": float(np.nanmean(retx)),
                        "rule_cosine": cos, "coef_sign_flip_rate": signflip,
                        "pos_rate": float(sy.mean())})
        if not per:
            continue
        agg = {k: float(np.nanmean([p[k] for p in per]))
               for k in ("assoc_retention", "assoc_retention_median", "pair_dep_retention",
                         "rule_cosine", "coef_sign_flip_rate", "pos_rate")}
        agg.update(label=label, n_draws=len(per))
        rows.append(agg)

    W = 104
    print("=" * W)
    print(f"WHY THE UTILITY GAP — {spec.name}: what each synthesiser preserves")
    print(f"real base rate {base_rate:.3f}. Retention is measured against REAL samples of the "
          f"same size (n={a.matched_n}, {a.n_reference_samples} draws),")
    print("so 1.00 = indistinguishable from real data of this size. Mutual information is not "
          "comparable across n.")
    print("=" * W)
    print(f"{'method':30s} {'n':>2s} {'I(X;Y) ret':>11s} {'median':>8s} {'I(X;X) ret':>11s} "
          f"{'rule cos':>9s} {'sign flip':>10s} {'pos rate':>9s}")
    for r in sorted(rows, key=lambda r: -r["rule_cosine"]):
        print(f"{r['label']:30s} {r['n_draws']:>2d} {r['assoc_retention']:>11.3f} "
              f"{r['assoc_retention_median']:>8.3f} {r['pair_dep_retention']:>11.3f} "
              f"{r['rule_cosine']:>9.3f} {r['coef_sign_flip_rate']:>10.3f} {r['pos_rate']:>9.3f}")
    print("-" * W)
    print("I(X;Y) ret   how much of the real feature->target association survives (1.0 = all)")
    print("I(X;X) ret   how much of the real feature-feature dependence survives")
    print("rule cos     cosine between the decision rule learned on synthetic vs on real data")
    print("sign flip    fraction of learned coefficients whose SIGN disagrees with real data")
    print("             (uninformative at this n: a real 300-row sample flips ~half of them too)")

    out = a.out or f"results/{a.dataset}_utility_gap.json"
    json.dump({"dataset": spec.name, "base_rate": base_rate, "methods": rows},
              open(out, "w"), indent=2, default=str)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
