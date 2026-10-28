"""Correct sequential sampler for an mbi 1.1 MarkovRandomField. mbi 1.1.0's MarkovRandomField.synthetic_data indexes
each conditional table by the parents' values in the order it stacked them, while the projected Factor's axes follow
the Factor's own attribute order; when the two orders differ the conditionals are read against the wrong axes and the
sampled joints are scrambled although every 1-way marginal stays right (verified 2026-10-21 on Adult: the model's own
P(income | fnlwgt) is flat, the sampled one is 0.00 .. 0.90). This sampler transposes every projected marginal to the
exact (parents..., child) order before conditioning. i.i.d. sampling (method 'sample')."""
import numpy as np
from mbi import junction_tree, marginal_oracles, Dataset

def sample_mrf(model, rows, rng):
    domain = model.domain
    cliques = [set(cl) for cl in model.cliques]
    jtree, elimination_order = junction_tree.make_junction_tree(domain, cliques)
    potentials = model.potentials.expand(list(jtree.nodes))
    marginals = marginal_oracles.message_passing_stable(potentials)
    order = list(elimination_order[::-1])
    data = {}
    import sys
    def aligned(proj):
        f = marginals.project(tuple(proj))
        arr = np.asarray(f.datavector(flatten=False), dtype=float)
        attrs = list(f.domain.attrs)
        arr = np.transpose(arr, [attrs.index(a) for a in proj])
        bad = ~np.isfinite(arr); neg = arr < 0
        if bad.any() or neg.any():
            print(f"[mrf_sampler] marginal {tuple(proj)}: {int(bad.sum())} non-finite and {int(neg.sum())} negative cells "
                  f"(negative mass {float(arr[neg].sum()) if neg.any() else 0.0:.4g} of total {float(np.nansum(np.abs(arr))):.4g}); clipped to 0", file=sys.stderr, flush=True)
            arr = np.where(bad, 0.0, arr); arr = np.clip(arr, 0.0, None)
        return arr
    col = order[0]
    p = aligned((col,)); p = p / p.sum() if p.sum() > 0 else np.full(p.size, 1.0 / p.size)
    data[col] = rng.choice(p.size, rows, True, p)
    used = [col]
    for col in order[1:]:
        relevant = [cl for cl in cliques if col in cl]
        parents = [a for a in used if a in set().union(*relevant)]
        if not parents:
            p = aligned((col,)); p = p / p.sum() if p.sum() > 0 else np.full(p.size, 1.0 / p.size); data[col] = rng.choice(p.size, rows, True, p); used.append(col); continue
        marg = aligned(tuple(parents) + (col,))
        parent_tot = marg.sum(axis=-1, keepdims=True)
        cond = np.divide(marg, parent_tot, out=np.zeros_like(marg), where=parent_tot != 0)
        # rows whose parent combination has zero model mass fall back to the child's 1-way marginal
        p1 = aligned((col,)); p1 = p1 / p1.sum() if p1.sum() > 0 else np.full(p1.size, 1.0 / p1.size)
        idx = tuple(data[a] for a in parents)
        cp = cond[idx]                      # rows x child
        zero = cp.sum(axis=1) <= 0
        if zero.any(): cp[zero] = p1
        cdf = cp.cumsum(axis=1); u = rng.random(rows)
        data[col] = np.minimum((u[:, None] > cdf).sum(axis=1), cp.shape[1] - 1)
        used.append(col)
    import pandas as pd
    return Dataset(pd.DataFrame({a: data[a].astype(int) for a in domain.attrs}), domain)
