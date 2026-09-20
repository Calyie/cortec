"""
workload_error.py — AIM's own utility metric, so CoRTeC can be compared on AIM's home turf.

Everything this project has measured so far (1-way / 2-way TV) is NOT the metric AIM is
designed for or evaluated on. AIM (McKenna et al., VLDB 2022, §6) evaluates workloads of
**3-way** marginal queries under this definition:

    Definition 2 (Workload Error).  A workload W is marginal queries r_1..r_k with weights c_i.
        Error(D, D_hat) = 1/(k |D|) * sum_i c_i * || M_ri(D) - M_ri(D_hat) ||_1

i.e. the mean over the workload of the L1 distance between the true and synthetic marginal
COUNT vectors, normalised by the dataset size. With both sides normalised to distributions
(which is what dividing by |D| does when |D_hat| = |D|), that L1 distance is exactly 2x the
total-variation distance, so this is directly comparable to the TV numbers used elsewhere in
the project — but computed over 3-way marginals rather than 1- and 2-way.

Their three workloads:
  all-3way  every 3-way marginal.
  target    every 3-way marginal containing the target attribute (income for Adult). This is
            the workload most relevant to CoRTeC's Goal 2 story.
  skewed    3-way marginals sampled with attribute weights from a squared-exponential, so a
            few attributes dominate. Reproduced here with a fixed seed.

SAMPLE SIZE IS PART OF THE COMPARISON. AIM's paper generates |D_hat| = |D|. A 3-way marginal
over binned Adult columns has hundreds of cells, so at n=300 almost every cell is empty and the
error is dominated by sampling noise rather than by the mechanism. `real_sample` at the same n
is therefore reported as the floor: no synthetic method can beat a real sample of its own size.
Always compare methods at MATCHED n, against that floor.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd


def discretize_for_workload(df: pd.DataFrame, cols, numerical_cols, public_bins,
                            feature_bounds) -> pd.DataFrame:
    """Map every column to a discrete domain, matching how AIM's pipeline is fed."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in numerical_cols:
            edges = np.asarray(public_bins[c], dtype=float)
            lo, hi = feature_bounds[c]
            v = pd.to_numeric(df[c], errors="coerce").fillna(lo).clip(lo, hi)
            out[c] = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2).astype(int).astype(str)
        else:
            out[c] = df[c].astype(str).str.strip()
    return out


def build_workload(cols, kind="target", target_col=None, n_skewed=256, seed=0):
    cols = list(cols)
    if kind == "all3way":
        return [tuple(t) for t in itertools.combinations(cols, 3)]
    if kind == "target":
        if target_col is None:
            raise ValueError("target workload needs target_col")
        others = [c for c in cols if c != target_col]
        return [(target_col, a, b) for a, b in itertools.combinations(others, 2)]
    if kind == "skewed":
        rng = np.random.RandomState(seed)
        w = rng.randn(len(cols)) ** 2
        w = w / w.sum()
        seen, out = set(), []
        guard = 0
        while len(out) < n_skewed and guard < n_skewed * 200:
            guard += 1
            t = tuple(sorted(rng.choice(len(cols), 3, replace=False, p=w)))
            if t not in seen:
                seen.add(t)
                out.append(tuple(cols[i] for i in t))
        return out
    raise ValueError(kind)


def _marginal(dd: pd.DataFrame, attrs) -> dict:
    key = dd[attrs[0]].astype(str)
    for a in attrs[1:]:
        key = key + "|" + dd[a].astype(str)
    vc = key.value_counts(normalize=True)
    return {k: float(v) for k, v in vc.items()}


def workload_error(real_disc: pd.DataFrame, synth_disc: pd.DataFrame, workload,
                   weights=None) -> float:
    """Mean over the workload of || p_real - p_synth ||_1  (= 2 x TV per query).

    This is AIM's Definition 2 with both marginals normalised to distributions, which is what
    their 1/|D| factor amounts to when the synthetic set is generated at the real size.
    """
    weights = weights or [1.0] * len(workload)
    tot, wsum = 0.0, 0.0
    for attrs, c in zip(workload, weights):
        p = _marginal(real_disc, attrs)
        q = _marginal(synth_disc, attrs)
        keys = set(p) | set(q)
        l1 = sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
        tot += c * l1
        wsum += c
    return float(tot / wsum) if wsum else float("nan")


def workload_error_report(real_raw, synth_sets: dict, cols, numerical_cols, public_bins,
                          feature_bounds, target_col, kinds=("target", "all3way", "skewed"),
                          seed=0):
    """Score every synthetic set on every workload. Returns {label: {kind: error}}."""
    real_disc = discretize_for_workload(real_raw, cols, numerical_cols, public_bins, feature_bounds)
    workloads = {k: build_workload(cols, k, target_col=target_col, seed=seed) for k in kinds}
    res = {}
    for label, dfs in synth_sets.items():
        per_kind = {}
        for k, wl in workloads.items():
            errs = []
            for d in dfs:
                sd = discretize_for_workload(d, cols, numerical_cols, public_bins, feature_bounds)
                errs.append(workload_error(real_disc, sd, wl))
            per_kind[k] = (float(np.mean(errs)), float(np.std(errs)), len(errs))
        res[label] = per_kind
    return res, {k: len(v) for k, v in workloads.items()}
