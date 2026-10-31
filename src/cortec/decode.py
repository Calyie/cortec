"""Stage B without a language model: decode a release with the maximum-entropy table.

The release fixes a set of counts. Among every table that agrees with those counts, the
maximum-entropy one assumes nothing the release did not say. Private-PGM (the estimator AIM and MST
use) finds it from the released counts and their noise scales alone. This module fits that table,
samples a pool of `pool_factor * n_rows` records from it and returns the `n_rows` of them that the
selection step (`cortec.select.select_to_release`) rakes to the released cells. It reads the release
and the public schema only: post-processing, no privacy cost, no model, CPU only. In the paper it is
the configuration labelled "CoRTeC, maximum-entropy decoder" (Table 3). There it came close to the
best language-model configuration on single columns and on the conditionals the release names.

Needs the optional dependency: `pip install '.[pgm]'` in a clone of the repository (private-pgm's `mbi` 1.1, with JAX).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .release import Release, _cell_keys, _cohort_keys
from .schema import Schema


class DecodeError(RuntimeError):
    """The release could not be decoded. The message names what to change."""


def _require_pgm():
    try:
        import jax  # noqa: F401
        import mbi  # noqa: F401
        from mbi import Domain, LinearMeasurement  # noqa: F401
        import mbi.estimation  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra installed
        raise DecodeError(
            "decoding without a language model needs private-pgm (mbi 1.1) and JAX: "
            "pip install '.[pgm]' in a clone of the repository") from e


def _epsilons(schema: Schema, release: Release) -> tuple[float, float | None, dict[int, float]]:
    """The per-query budgets behind the release, read from its audit trail (the noise scale of a
    count of sensitivity 1 at budget e is 1/e). A release without an audit block gets the package's
    default split, with a warning."""
    groups = (release.audit or {}).get("groups") or {}
    nq = len(schema.numerical) + len(schema.categorical) + 1

    def total(name: str) -> float:
        return float((groups.get(name) or {}).get("group_total", 0.0) or 0.0)

    levels = [int(lv["level"]) for lv in release.conditional_levels]
    if total("marginals") > 0:
        eps_q = total("marginals") / nq
        eps_coh = total("published_counts_cohorts") or None
        eps_lv = {li: (total(f"conditional_L{li}") or None) for li in levels}
        if all(v for v in eps_lv.values()):
            return eps_q, eps_coh, eps_lv
    import warnings
    warnings.warn("the release carries no usable audit block. The decoder assumes the package's "
                  "default budget split: 0.2 for the conditional table, 0.1 for counts and the rest for marginals")
    e = float(release.epsilon_total)
    return 0.7 * e / nq, 0.05 * e, {li: 0.2 * e / max(len(levels), 1) for li in levels}


@dataclass
class ReleaseModel:
    """A maximum-entropy table fitted to a release, ready to be sampled any number of times."""
    schema: Schema
    release: Release
    model: object = field(repr=False)
    columns: list[str]
    values: dict[str, list]
    edges: dict[str, np.ndarray]
    known_total: float
    measurements: list[dict]
    consistency_mean_tv: float
    consistency_worst: list[tuple]
    attempts: int
    seconds: float

    def sample(self, n_rows: int, *, seed: int | None = None) -> pd.DataFrame:
        """`n_rows` records from the fitted table, values drawn uniformly inside their bins
        (integers where the public bounds and edges are integral). No selection. See
        `decode_release` for the full path."""
        rng = np.random.default_rng(seed)
        idx = _sample_mrf(self.model, int(n_rows), rng)
        return _undiscretise(self.schema, idx, self.values, self.edges, rng)


def _representative_rows(schema: Schema, cols: list[str], values: dict, edges: dict) -> tuple[pd.DataFrame, list[tuple]]:
    """One row per combination of domain values of `cols`, carrying a representative raw value
    (a bin's left edge, a category), so that the package's own cohort and cell rules can be
    applied to domain indices exactly as they are applied to records."""
    shape = [len(values[c]) for c in cols]
    flats = list(np.ndindex(*shape)) if cols else [()]
    rows = {}
    for c in cols:
        if c in schema.numerical:
            rows[c] = [float(edges[c][f[cols.index(c)]]) for f in flats]
        else:
            rows[c] = [str(values[c][f[cols.index(c)]]) for f in flats]
    return pd.DataFrame(rows, index=range(len(flats))), flats


def fit_release_model(schema: Schema, release: Release, *, iters: int = 3000, seed: int = 42,
                      verbose: bool = True) -> ReleaseModel:
    """Fit the maximum-entropy table to exactly the released counts, each at the noise scale its
    budget implies. These are the cohort sizes, the class balances, the per-cohort histograms (per
    class where the release carries class blocks) and the conditional cells. Private-PGM's mirror
    descent on the junction tree. A fit with a non-finite or collapsed marginal is rejected and
    retried from another seed, up to three times."""
    _require_pgm()
    import jax.numpy as jnp
    from mbi import Domain, LinearMeasurement
    import mbi.estimation as est
    from . import report

    if release.schema_name != schema.name:
        raise DecodeError(f"release is for schema {release.schema_name!r}, not {schema.name!r}")
    if not release.cohorts:
        raise DecodeError("the release holds no cohort")
    t0 = time.time()
    eff = replace(release.effective_schema(schema), coarsen=dict(release.coarsen or schema.coarsen or {}))
    cols = list(schema.numerical) + list(schema.categorical) + [schema.target]
    first = release.cohorts[0]
    edges = {c: np.asarray(first["numerical"][c]["bin_edges"], float) for c in schema.numerical}
    values: dict[str, list] = {}
    for c in schema.numerical:
        values[c] = list(range(len(edges[c]) - 1))
    for c in schema.categorical:
        values[c] = [str(v) for v in schema.categorical[c]]
    tv = [str(schema.positive), str(schema.negative)]
    values[schema.target] = tv
    domain = Domain(cols, [len(values[c]) for c in cols])

    # cohorts: the released ones, indexed over the stratification columns' domain values
    strat_cols = [col for col, _ in (eff.stratify or [])]
    for col, bands in (eff.stratify or []):
        if col in schema.numerical and bands:
            cuts = sorted({float(b.lo) for b in bands} | {float(b.hi) for b in bands})
            for i in range(len(edges[col]) - 1):
                lo, hi = edges[col][i], edges[col][i + 1]
                if any(lo < e < hi for e in cuts):
                    import warnings
                    warnings.warn(f"{col}: public bin [{lo:g}, {hi:g}) straddles a cohort band edge. "
                                  f"The decoder assigns the bin to the band of its left edge.")
    rep, flats = _representative_rows(schema, strat_cols, values, edges)
    names = [str(c["cohort_name"]) for c in release.cohorts]
    keys = _cohort_keys(eff, rep).astype(str).tolist() if strat_cols else ["all"] * len(flats)
    strat_shape = [len(values[c]) for c in strat_cols]
    coh_index = np.full(strat_shape, -1, dtype=int) if strat_cols else np.array(-1)
    for f, k in zip(flats, keys):
        if k in names:
            if strat_cols:
                coh_index[f] = names.index(k)
            else:
                coh_index = np.array(names.index(k))
    if int((coh_index >= 0).sum()) == 0:
        raise DecodeError("no released cohort maps onto the schema's stratification. The release "
                          "and the schema disagree on the cohort rule")
    sizes = [float(c["cohort_size"]) for c in release.cohorts]
    known_total = float(sum(sizes))
    eps_q, eps_coh, eps_lv = _epsilons(schema, release)

    measurements, summary = [], []

    def add(clique: list[str], cell_index: np.ndarray, y: list[float], eps: float, label: str) -> None:
        clique_t = tuple(clique)
        shape = [domain[c] for c in clique_t]
        M = np.zeros((len(y), int(np.prod(shape))), dtype=np.float32)
        for flat, pos in np.ndenumerate(cell_index):
            if pos >= 0:
                M[pos, np.ravel_multi_index(flat, shape)] = 1.0
        Mj = jnp.asarray(M)

        def query(fct, clique_t=clique_t, Mj=Mj):
            arr = fct.datavector(flatten=False)
            attrs = list(fct.domain.attrs)
            arr = jnp.transpose(arr, [attrs.index(c) for c in clique_t])
            return Mj @ arr.reshape(-1)

        sd = float(math.sqrt(2.0) / eps)
        measurements.append(LinearMeasurement(jnp.asarray(np.asarray(y, dtype=np.float32)), clique_t,
                                              stddev=sd, query=query))
        summary.append({"label": label, "clique": list(clique_t), "cells": int(len(y)), "epsilon": float(eps),
                        "stddev": sd, "total": float(np.sum(y))})

    def cohort_of(flat: tuple, clique: list[str]) -> int:
        if not strat_cols:
            return int(coh_index)
        return int(coh_index[tuple(flat[clique.index(c)] for c in strat_cols)])

    # (1) cohort sizes
    if strat_cols:
        add(strat_cols, coh_index, sizes, eps_coh or eps_q, "cohort sizes")
    # (2) class balance per cohort
    cl = strat_cols + [schema.target]
    shape = [len(values[c]) for c in cl]
    ci = np.full(shape, -1, dtype=int); y = []
    for k, coh in enumerate(release.cohorts):
        for t, cls in enumerate(tv):
            row = len(y); y.append(sizes[k] * float((coh.get("class_balance") or {}).get(cls, 0.5)))
            for flat in np.ndindex(*shape):
                if cohort_of(flat, cl) == k and flat[-1] == t:
                    ci[flat] = row
    add(cl, ci, y, eps_q, "class balance")

    # (3) the column blocks: pooled where the cohort has no class blocks, per class where it has
    def block_counts(blk: dict, col: str, size: float) -> np.ndarray:
        if col in schema.numerical:
            h = np.asarray(blk["numerical"][col]["proportions"], float)
            return size * h / max(h.sum(), 1e-12)
        pr = blk["categorical"][col]; tot = sum(float(v) for v in pr.values()) or 1.0
        return np.array([size * float(pr.get(v, 0.0)) / tot for v in values[col]], float)

    for col in cols[:-1]:
        base = [c for c in strat_cols if c != col]
        pooled = [k for k, c in enumerate(release.cohorts) if not c.get("by_class")]
        byc = [k for k, c in enumerate(release.cohorts) if c.get("by_class")]
        if pooled:
            cl = base + [col]; shape = [len(values[c]) for c in cl]
            ci = np.full(shape, -1, dtype=int); y = []
            for k in pooled:
                cnt = block_counts(release.cohorts[k], col, sizes[k]); start = len(y)
                for flat in np.ndindex(*shape):
                    if cohort_of(flat, cl) == k:
                        ci[flat] = start + flat[-1]
                y += list(cnt)
            add(cl, ci, y, eps_q, f"pooled block {col} ({len(pooled)} cohorts)")
        if byc:
            cl = base + [schema.target, col]; shape = [len(values[c]) for c in cl]
            ci = np.full(shape, -1, dtype=int); y = []
            for k in byc:
                coh = release.cohorts[k]
                for t, cls in enumerate(tv):
                    blk = (coh.get("by_class") or {}).get(cls)
                    if blk is None:
                        continue
                    share = float((coh.get("class_balance") or {}).get(cls, 0.5))
                    cnt = block_counts(blk, col, sizes[k] * share); start = len(y)
                    for flat in np.ndindex(*shape):
                        if cohort_of(flat, cl) == k and flat[cl.index(schema.target)] == t:
                            ci[flat] = start + flat[-1]
                    y += list(cnt)
            add(cl, ci, y, eps_q, f"class blocks {col} ({len(byc)} cohorts x 2 classes)")

    # (4) the conditional cells: positives and negatives per released cell, at the cell's own
    #     released support (the cells of one level partition the data)
    pos_i = tv.index(str(schema.positive))
    for lv in release.conditional_levels:
        li = int(lv["level"]); lcols = [str(c) for c in lv["columns"]]; cells = lv["cells"]
        rep_l, flats_l = _representative_rows(schema, lcols, values, edges)
        keys_l = _cell_keys(rep_l, tuple(lcols), eff).astype(str).tolist() if lcols else ["*"] * len(flats_l)
        cl = lcols + [schema.target]; shape = [len(values[c]) for c in cl]
        ci = np.full(shape, -1, dtype=int); y = []; rows: dict[str, int] = {}
        for f, key in zip(flats_l, keys_l):
            cell = cells.get(key)
            if cell is None:
                continue
            if key not in rows:
                rows[key] = len(y); sup = float(cell["support"]); rate = float(cell["rate"])
                y += [0.0, 0.0]; y[rows[key] + pos_i] = sup * rate; y[rows[key] + 1 - pos_i] = sup * (1.0 - rate)
            for t in range(len(tv)):
                ci[tuple(f) + (t,)] = rows[key] + t
        if y:
            add(cl, ci, y, eps_lv.get(li) or eps_q, f"conditional level {li} ({len(rows)} of {len(cells)} cells mapped)")
            if len(rows) < len(cells):
                import warnings
                warnings.warn(f"conditional level {li}: {len(cells) - len(rows)} released cells have no "
                              f"counterpart in the schema's domain and were not used")

    if verbose:
        report.emit(report.note(f"decoder: {len(measurements)} measurement sets over "
                                f"{sum(s['cells'] for s in summary)} released counts, known total {known_total:,.0f}"))

    def problem(model) -> str | None:
        for c in model.domain.attrs:
            m = np.asarray(model.project((c,)).datavector(), dtype=float)
            if not np.all(np.isfinite(m)):
                return f"{c}: non-finite marginal"
            if m.sum() <= 0:
                return f"{c}: zero mass"
            if m.size > 1 and m.max() / m.sum() > 0.995:
                return f"{c}: {m.max() / m.sum():.4f} of the mass on one value"
        return None

    model = None; attempts = 0
    for attempt in range(3):
        attempts += 1
        np.random.seed(int(seed) + attempt); t1 = time.time()
        try:
            m = est.mirror_descent(domain, measurements, known_total=known_total, iters=int(iters))
            pr = problem(m)
        except Exception as e:  # a numerical failure inside the library is a rejected attempt
            m, pr = None, f"raised {type(e).__name__}: {str(e)[:120]}"
        if verbose:
            report.emit(report.note(f"decoder fit {attempt + 1}: {'valid' if pr is None else 'rejected, ' + pr} "
                                    f"({time.time() - t1:.0f} s)"))
        if pr is None:
            model = m; break
    if model is None:
        raise DecodeError("no valid maximum-entropy fit in three attempts. Raise iters, or check that the "
                          "release's bins and cohorts match the schema")

    # consistency: the fitted per-cohort column shares against the released ones
    worst = []
    for k, coh in enumerate(release.cohorts):
        for col in cols[:-1]:
            cl = [c for c in strat_cols if c != col] + [col]
            fct = model.project(tuple(cl)); arr = np.asarray(fct.datavector(flatten=False), float)
            attrs = list(fct.domain.attrs); arr = np.transpose(arr, [attrs.index(c) for c in cl])
            got = np.zeros(arr.shape[-1])
            for flat in np.ndindex(*arr.shape):
                if cohort_of(flat, cl) == k:
                    got[flat[-1]] += arr[flat]
            got = got / max(got.sum(), 1e-12)
            want = block_counts(coh, col, 1.0); want = want / max(want.sum(), 1e-12)
            worst.append((float(0.5 * np.abs(got - want).sum()), str(coh["cohort_name"]), col))
    worst.sort(reverse=True)
    mean_tv = float(np.mean([w[0] for w in worst])) if worst else 0.0
    if verbose:
        report.emit(report.note(f"decoder consistency with the release (mean per-cohort column TV): {mean_tv:.4f}. "
                                f"The worst is {worst[0][1]} / {worst[0][2]} at {worst[0][0]:.3f}" if worst else
                                "decoder consistency: nothing to compare"))
    return ReleaseModel(schema=schema, release=release, model=model, columns=cols, values=values, edges=edges,
                        known_total=known_total, measurements=summary, consistency_mean_tv=mean_tv,
                        consistency_worst=worst[:5], attempts=attempts, seconds=time.time() - t0)


def _sample_mrf(model, rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sequential sampling of the fitted table, each column from its conditional given the parents
    already drawn, with every projected marginal transposed to the exact (parents..., child) order
    before conditioning. (mbi 1.1's own sampler reads the conditionals against the factor's
    attribute order, which scrambles the joints when the two orders differ.)"""
    from mbi import junction_tree, marginal_oracles
    domain = model.domain
    cliques = [set(cl) for cl in model.cliques]
    jtree, elimination_order = junction_tree.make_junction_tree(domain, cliques)
    potentials = model.potentials.expand(list(jtree.nodes))
    marginals = marginal_oracles.message_passing_stable(potentials)
    order = list(elimination_order[::-1])

    def aligned(proj: tuple) -> np.ndarray:
        fct = marginals.project(tuple(proj))
        arr = np.asarray(fct.datavector(flatten=False), dtype=float)
        attrs = list(fct.domain.attrs)
        arr = np.transpose(arr, [attrs.index(a) for a in proj])
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return np.clip(arr, 0.0, None)

    def one_way(col: str) -> np.ndarray:
        p = aligned((col,))
        return p / p.sum() if p.sum() > 0 else np.full(p.size, 1.0 / p.size)

    data: dict[str, np.ndarray] = {}
    col = order[0]
    data[col] = rng.choice(one_way(col).size, rows, True, one_way(col))
    used = [col]
    for col in order[1:]:
        relevant = [cl for cl in cliques if col in cl]
        parents = [a for a in used if a in set().union(*relevant)] if relevant else []
        if not parents:
            data[col] = rng.choice(one_way(col).size, rows, True, one_way(col)); used.append(col); continue
        marg = aligned(tuple(parents) + (col,))
        parent_tot = marg.sum(axis=-1, keepdims=True)
        cond = np.divide(marg, parent_tot, out=np.zeros_like(marg), where=parent_tot != 0)
        cp = cond[tuple(data[a] for a in parents)]
        zero = cp.sum(axis=1) <= 0
        if zero.any():
            cp[zero] = one_way(col)
        cdf = cp.cumsum(axis=1); u = rng.random(rows)
        data[col] = np.minimum((u[:, None] > cdf).sum(axis=1), cp.shape[1] - 1)
        used.append(col)
    return pd.DataFrame({a: data[a].astype(int) for a in domain.attrs})


def _integer_valued(schema: Schema, col: str, edges: np.ndarray) -> bool:
    lo, hi = schema.numerical[col]
    return all(float(x).is_integer() for x in [lo, hi, *edges])


def _undiscretise(schema: Schema, idx: pd.DataFrame, values: dict, edges: dict, rng: np.random.Generator) -> pd.DataFrame:
    out = pd.DataFrame(index=idx.index)
    for c in schema.numerical:
        e = edges[c]; nb = len(e) - 1; i = np.clip(idx[c].to_numpy().astype(int), 0, nb - 1)
        left, right = e[i], e[i + 1]
        lo, hi = schema.numerical[c]
        if _integer_valued(schema, c, e):
            # integers inside [left, right); the last bin is closed at the declared maximum
            top = np.where(i == nb - 1, right, right - 1)
            v = np.floor(left + rng.random(len(i)) * (top - left + 1))
            out[c] = np.clip(v, lo, hi).astype(int)
        else:
            v = left + rng.random(len(i)) * (right - left)
            out[c] = np.clip(v, lo, hi)
    for c in schema.categorical:
        vals = np.array(values[c], dtype=object)
        out[c] = vals[np.clip(idx[c].to_numpy().astype(int), 0, len(vals) - 1)]
    tv = np.array(values[schema.target], dtype=object)
    out[schema.target] = tv[np.clip(idx[schema.target].to_numpy().astype(int), 0, len(tv) - 1)]
    return out[list(schema.numerical) + list(schema.categorical) + [schema.target]]


def decode_release(schema: Schema, release: Release, n_rows: int, *, pool_factor: int = 3,
                   seed: int = 42, iters: int = 3000, select: bool = True, sub_bin: str = "release",
                   verbose: bool = True, model: ReleaseModel | None = None) -> pd.DataFrame:
    """Decode `n_rows` synthetic records from `release` with no language model. The maximum-entropy
    table that fits the released counts is sampled at `pool_factor * n_rows`, and the selection step
    chooses from that pool (`select=False` returns `n_rows` samples of the table as they are).
    Post-processing of the release. Repeatable at no privacy cost. Pass a fitted `model` (from
    `fit_release_model`) to draw again without refitting."""
    from .select import select_to_release
    if n_rows < 1:
        raise DecodeError("n_rows must be at least 1")
    if pool_factor < 1:
        raise DecodeError("pool_factor must be at least 1")
    if sub_bin not in ("release", "generator"):
        raise DecodeError("sub_bin must be 'release' (default) or 'generator'")
    rm = model or fit_release_model(schema, release, iters=iters, seed=seed, verbose=verbose)
    if not select or pool_factor == 1:
        return rm.sample(n_rows, seed=seed)
    pool = rm.sample(int(pool_factor) * int(n_rows), seed=seed)
    return select_to_release(schema, release, pool, int(n_rows), seed=seed, sub_bin=sub_bin)


def main(argv: list[str] | None = None, prog: str = "cortec decode") -> int:
    """Stage B without a language model: decode a release with Private-PGM and the selection step. cortec decode --help"""
    import argparse
    from .report import emit, kv_block, note, install_guard
    from .run import load_schema
    install_guard()
    ap = argparse.ArgumentParser(prog=prog, description=(
        "Decode a release with no language model. The decoder fits the maximum-entropy table that agrees with "
        "every released count (Private-PGM), samples it at a multiple of the requested size and rakes the "
        "sample to the released cells with the selection step. Reads the release and the schema only. Spends no privacy budget. Needs cortec[pgm]."))
    ap.add_argument("--schema", required=True, help="a Python file that defines SCHEMA = Schema(...)")
    ap.add_argument("--release", required=True, help="the release (release.json written by cortec run or Release.to_json)")
    ap.add_argument("--n-rows", type=int, default=300, help="synthetic rows per draw (default 300)")
    ap.add_argument("--draws", type=int, default=1, help="how many draws to write (default 1)")
    ap.add_argument("--pool-factor", type=int, default=3, help="pool size as a multiple of --n-rows (default 3)")
    ap.add_argument("--seed", type=int, default=42, help="the fit's and the sampler's seed (default 42)")
    ap.add_argument("--iters", type=int, default=3000, help="mirror-descent iterations (default 3000)")
    ap.add_argument("--no-select", action="store_true", help="return samples of the fitted table without the selection step")
    ap.add_argument("--sub-bin", default="release", choices=("release", "generator"),
                    help="values below bin resolution: the release's own (default) or the sampler's")
    ap.add_argument("--out", default=None, help="output folder. Default results/<schema name>_decoded")
    a = ap.parse_args(argv)
    schema = load_schema(a.schema)
    release = Release.from_json(a.release)
    out = a.out or os.path.join("results", f"{schema.name}_decoded")
    os.makedirs(out, exist_ok=True)
    emit(kv_block([("schema", f"{schema.name} ({a.schema})"), ("release", a.release),
                   ("rows", f"{a.n_rows:,} per draw, {a.draws} draw(s)" +
                    ("" if a.no_select else f", selected from a pool of {a.n_rows * a.pool_factor:,}")),
                   ("model", "none: the maximum-entropy table of the release (Private-PGM)" +
                    ("" if a.no_select else " + the selection step")),
                   ("privacy cost", "0 (post-processing of the release)"), ("output", os.path.abspath(out))]))
    t0 = time.time()
    rm = fit_release_model(schema, release, iters=a.iters, seed=a.seed)
    files = []
    for k in range(a.draws):
        df = decode_release(schema, release, a.n_rows, pool_factor=a.pool_factor, seed=a.seed + k,
                            select=not a.no_select, sub_bin=a.sub_bin, model=rm, verbose=False)
        path = os.path.join(out, f"decoded_draw{k}.csv"); df.to_csv(path, index=False); files.append(path)
        emit(note(f"draw {k}: {len(df):,} rows -> {path}"))
    manifest = {"schema": schema.name, "release": os.path.abspath(a.release), "n_rows": a.n_rows, "draws": a.draws,
                "pool_factor": a.pool_factor, "selected": not a.no_select, "sub_bin": a.sub_bin, "seed": a.seed,
                "iters": a.iters, "known_total": rm.known_total, "fit_attempts": rm.attempts,
                "fit_seconds": round(rm.seconds, 1), "consistency_mean_tv": rm.consistency_mean_tv,
                "consistency_worst": rm.consistency_worst, "measurements": rm.measurements,
                "private_data_read": False, "privacy_cost": 0.0, "files": files, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, "decode_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    emit(note(f"manifest -> {os.path.join(out, 'decode_manifest.json')}"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
