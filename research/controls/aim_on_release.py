"""AIM-on-release: fit a maximum-entropy graphical model (private-pgm / mbi 1.1 mirror descent, the same inference
engine and the same corrected sampler as the reference AIM and MST baselines) to EXACTLY the measurements CoRTeC
releases in Stage A, with their noise scales, and sample rows from it. Nothing private is read: the inputs are the
release JSON and the public dataset spec. It isolates the release design (cohort marginals, class-conditional
blocks, conditional-outcome tables, noisy cohort sizes) from the LLM: whatever CoRTeC scores above this model is
what the language model adds beyond a maximum-entropy fill of the same numbers.

Measurements (all noisy COUNTS; stddev = sqrt(2) / eps of the Laplace count query that produced them):
  cohort sizes            clique = stratification columns          eps = cohort_count_query
  pooled column blocks    clique = strat + (col,)                   eps = per_query  (cohorts without by_class)
  class-conditional blocks clique = strat + (target, col)           eps = per_query  (cohorts with by_class)
  class balance           clique = strat + (target,)                eps = per_query
  conditional tables      clique = level bands' columns + (target,) eps = per_cond_cell_query; cell sizes derived
                          from the released cohort statistics (post-processing), rate x size = positives
Usage: python aim_on_release.py --dataset adult --release results/adult_v3_gemini/cohort_statistics.json
                                --n-out 1000 --draws 3 --seed 42 --outdir DIR [--iters 3000]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
ROOT = os.environ.get('CORTEC_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True); ap.add_argument('--release', required=True)
ap.add_argument('--n-out', type=int, required=True); ap.add_argument('--draws', type=int, default=3); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--outdir', required=True); ap.add_argument('--iters', type=int, default=3000)
ap.add_argument('--prefix', default='aim_on_release')
a = ap.parse_args()
from src.dataset_spec import get_spec
import src.datasets_extra, src.datasets_regulated, src.datasets_synthetic, src.datasets_clinical, src.datasets_auto  # noqa: F401
import jax.numpy as jnp
from mbi import Domain, LinearMeasurement
import mbi.estimation as est
from mrf_sampler import sample_mrf
spec = get_spec(a.dataset); cols = spec.column_names
release = json.load(open(a.release)); eps = release[0]['_eps']
bins = {c: np.asarray(spec.public_bins[c], dtype=float) for c in spec.numerical_cols}
# ---- public domain: numeric bin index over the full public bins; categorical = the release's declared category list
dom_vals = {}
for c in cols:
    if c in spec.numerical_cols: dom_vals[c] = list(range(len(bins[c]) - 1))
    elif c == spec.target_col: dom_vals[c] = sorted([spec.positive_class, spec.negative_class])
    else:
        keys = set()
        for r in release:
            keys |= set(r['categorical'].get(c, {}).keys())
            for blk in (r.get('by_class') or {}).values(): keys |= set(blk['categorical'].get(c, {}).keys())
        dom_vals[c] = sorted(keys)
domain = Domain(cols, [len(dom_vals[c]) for c in cols])
print(f'domain sizes {dict(zip(cols, domain.shape))}', flush=True)
# ---- band labels per domain value (bands are public; the public bins refine every band, asserted)
def band_labels(band, col):
    vals = dom_vals[col]
    if band.edges is not None:
        edges = list(band.edges); labs = band.labels or [f"{edges[i]:g}-{edges[i+1]:g}" for i in range(len(edges) - 1)]
        out = []
        for i in vals:
            lo, hi = bins[col][i], bins[col][i + 1]
            k = max(j for j in range(len(edges) - 1) if lo >= edges[j])
            last = (i == len(vals) - 1)
            assert (hi <= edges[k + 1]) or (last and hi <= edges[k + 1] + 1e-9) or (last and edges[k + 1] == float('inf')), \
                f'{col} public bin [{lo},{hi}) straddles band edge {edges[k + 1]}'
            out.append(labs[k])
        return out
    if band.groups:
        m = {v: k for k, vs in band.groups.items() for v in vs}; return [m.get(v, 'other') for v in vals]
    return [str(v) for v in vals]
strat_cols = [b.col for b in spec.stratify]
strat_labels = {b.col: band_labels(b, b.col) for b in spec.stratify}
def cohort_of(idx_tuple):   # idx per strat col -> cohort_name
    return ' & '.join(strat_labels[c][i] for c, i in zip(strat_cols, idx_tuple))
levels = [int(k) for k in release[0]['conditional_target_levels']]
measurements = []; summary = []
def add(clique, cell_index, y, eps_q, label):
    """clique: tuple of attrs (order = my canonical order); cell_index: dict mapping a flat domain index tuple
    (over clique attrs in that order) -> row of y, or -1 for cells outside any released cell; y: counts."""
    clique = tuple(clique); shape = [domain[c] for c in clique]
    M = np.zeros((len(y), int(np.prod(shape))), dtype=np.float32)
    for flat, pos in np.ndenumerate(cell_index):
        if pos >= 0: M[pos, np.ravel_multi_index(flat, shape)] = 1.0
    Mj = jnp.asarray(M)
    def query(f, clique=clique, Mj=Mj):
        arr = f.datavector(flatten=False); attrs = list(f.domain.attrs)
        arr = jnp.transpose(arr, [attrs.index(c) for c in clique]); return Mj @ arr.reshape(-1)
    sd = float(np.sqrt(2.0) / eps_q)
    measurements.append(LinearMeasurement(jnp.asarray(np.asarray(y, dtype=np.float32)), clique, stddev=sd, query=query))
    summary.append({'label': label, 'clique': list(clique), 'cells': int(len(y)), 'eps': eps_q, 'stddev': sd, 'total': float(np.sum(y))})
# cohort index tensor over the strat columns
strat_shape = [domain[c] for c in strat_cols]
released = {r['cohort_name']: r for r in release}; names = list(released)
coh_index = np.full(strat_shape, -1, dtype=int)
for flat in np.ndindex(*strat_shape):
    nm = cohort_of(flat); coh_index[flat] = names.index(nm) if nm in released else -1
N_total = float(sum(r['cohort_size'] for r in release))
# (1) cohort sizes
add(strat_cols, coh_index, [released[n]['cohort_size'] for n in names], eps['cohort_count_query'], 'cohort sizes')
# (2) class balance per cohort
tv = dom_vals[spec.target_col]
ci = np.full(strat_shape + [len(tv)], -1, dtype=int); y = []
for k, n in enumerate(names):
    for t, cls in enumerate(tv):
        ci[..., t][coh_index == k] = len(y); y.append(released[n]['cohort_size'] * released[n]['class_balance'][cls])
add(strat_cols + [spec.target_col], ci, y, eps['per_query'], 'class balance')
# (3) column blocks: pooled for cohorts without by_class, per class for the others
def block_counts(blk, col, size):
    if col in spec.numerical_cols: h = np.asarray(blk['numerical'][col]['dp_hist'], float); return size * h / max(h.sum(), 1e-12)
    pr = blk['categorical'][col]; return np.array([size * pr.get(v, 0.0) for v in dom_vals[col]], float)
for col in [c for c in cols if c != spec.target_col]:
    cl = [c for c in strat_cols if c != col] + [col]           # a strat column measured within its own cohorts stays one axis
    pooled = [n for n in names if not released[n].get('by_class')]; byc = [n for n in names if released[n].get('by_class')]
    if pooled:
        shape = [domain[c] for c in cl]; ci = np.full(shape, -1, dtype=int); y = []
        for n in pooled:
            k = names.index(n); cnt = block_counts(released[n], col, released[n]['cohort_size'])
            for flat in np.ndindex(*shape):
                sidx = tuple(flat[cl.index(c)] for c in strat_cols)
                if coh_index[sidx] == k: ci[flat] = len(y) + flat[-1]
            y += list(cnt)
        add(cl, ci, y, eps['per_query'], f'pooled block {col} ({len(pooled)} cohorts)')
    if byc:
        cl2 = [c for c in strat_cols if c != col] + [spec.target_col, col]
        shape = [domain[c] for c in cl2]; ci = np.full(shape, -1, dtype=int); y = []
        for n in byc:
            k = names.index(n); r = released[n]
            for t, cls in enumerate(tv):
                blk = r['by_class'].get(cls)
                if blk is None: continue
                cnt = block_counts(blk, col, r['cohort_size'] * r['class_balance'][cls])
                for flat in np.ndindex(*shape):
                    sidx = tuple(flat[cl2.index(c)] for c in strat_cols)
                    if coh_index[sidx] == k and flat[cl2.index(spec.target_col)] == t: ci[flat] = len(y) + flat[-1]
                y += list(cnt)
        add(cl2, ci, y, eps['per_query'], f'class blocks {col} ({len(byc)} cohorts x 2 classes)')
# (4) conditional-outcome tables: positives and negatives per released cell; cell size derived from the cohort blocks
pos_i = tv.index(spec.positive_class)
for li in levels:
    tbl = release[0]['conditional_target_levels'][str(li)]; bands = spec.conditional_levels[li]
    if not bands:
        rate = tbl.get('*'); cnt = [0.0, 0.0]; cnt[pos_i] = N_total * rate; cnt[1 - pos_i] = N_total * (1 - rate)
        ci = np.arange(len(tv)); add([spec.target_col], ci, cnt, eps['per_cond_cell_query'], 'conditional level 0 (global rate)'); continue
    lcols = [b.col for b in bands]; labs = {b.col: band_labels(b, b.col) for b in bands}
    cl = list(dict.fromkeys(strat_cols + lcols)) + [spec.target_col]; shape = [domain[c] for c in cl]
    # P(cell | cohort) from the cohort's pooled blocks, one factor per band column (independent within cohort: post-processing approximation)
    def p_band(r, col, lab):
        vals = [i for i, l in enumerate(labs[col]) if l == lab]
        if col in spec.numerical_cols: h = np.asarray(r['numerical'][col]['dp_hist'], float); h = h / max(h.sum(), 1e-12); return float(h[vals].sum())
        pr = r['categorical'][col]; return float(sum(pr.get(dom_vals[col][i], 0.0) for i in vals))
    ci = np.full(shape, -1, dtype=int); y = []; cellmap = {}
    for flat in np.ndindex(*shape):
        sidx = tuple(flat[cl.index(c)] for c in strat_cols); k = coh_index[sidx]
        if k < 0: continue
        key = '|'.join(labs[c][flat[cl.index(c)]] for c in lcols)
        if key not in tbl: continue
        ck = (k, key)
        if ck not in cellmap:
            r = released[names[k]]; size = r['cohort_size'] * float(np.prod([p_band(r, c, labs[c][flat[cl.index(c)]]) for c in lcols]))
            if size < 1.0: continue
            rate = tbl[key]; cellmap[ck] = len(y); y += [0.0, 0.0]; y[cellmap[ck] + pos_i] = size * rate; y[cellmap[ck] + 1 - pos_i] = size * (1 - rate)
        ci[flat] = cellmap[ck] + flat[-1]
    if y: add(cl, ci, y, eps['per_cond_cell_query'], f'conditional level {li} ({len(cellmap)} cohort x cell pairs of {len(tbl)} cells)')
print('measurements:'); [print(f"  {s['label']}: clique {s['clique']}, {s['cells']} counts, stddev {s['stddev']:.1f}, total {s['total']:.0f}") for s in summary]
# ---- fit (validated as in pgm_baselines.py: reject non-finite or collapsed models, retry with a new seed)
os.makedirs(a.outdir, exist_ok=True)
def problem(model):
    for c in model.domain.attrs:
        m = np.asarray(model.project((c,)).datavector(), dtype=float)
        if not np.all(np.isfinite(m)): return f'{c}: non-finite marginal'
        if m.sum() <= 0: return f'{c}: zero mass'
        if m.size > 1 and m.max() / m.sum() > 0.995: return f'{c}: {m.max() / m.sum():.4f} of the mass on one value'
    return None
model = None
for attempt in range(3):
    np.random.seed(a.seed + attempt); t0 = time.time()
    try:
        m = est.mirror_descent(domain, measurements, known_total=N_total, iters=a.iters); pr = problem(m)
    except Exception as e:
        m, pr = None, f'raised {type(e).__name__}: {str(e)[:100]}'
    print(f'  fit attempt {attempt}: {"valid" if pr is None else "rejected, " + pr} ({time.time() - t0:.0f}s)', flush=True)
    if pr is None: model = m; break
if model is None: raise SystemExit('no valid fit')
# ---- consistency check against the release: per-cohort column marginals of the fitted model vs released shares
def model_marg(clique):
    f = model.project(tuple(clique)); arr = np.asarray(f.datavector(flatten=False), float); attrs = list(f.domain.attrs)
    return np.transpose(arr, [attrs.index(c) for c in clique])
worst = []
for n in names:
    k = names.index(n)
    for col in [c for c in cols if c != spec.target_col]:
        cl = [c for c in strat_cols if c != col] + [col]; arr = model_marg(cl)
        got = np.zeros(arr.shape[-1])
        for flat in np.ndindex(*arr.shape):
            sidx = tuple(flat[cl.index(c)] for c in strat_cols)
            if coh_index[sidx] == k: got[flat[-1]] += arr[flat]
        got = got / max(got.sum(), 1e-12)
        want = block_counts(released[n], col, 1.0); want = want / max(want.sum(), 1e-12)
        worst.append((0.5 * np.abs(got - want).sum(), n, col))
worst.sort(reverse=True)
print(f'release consistency (TV between fitted and released per-cohort column shares): mean {np.mean([w[0] for w in worst]):.4f}, worst ' + ', '.join(f'{w[1]}/{w[2]} {w[0]:.3f}' for w in worst[:3]), flush=True)
# ---- sample and decode exactly as pgm_baselines.py does
def undisc(df, rng):
    o = pd.DataFrame(index=df.index)
    for c in cols:
        if c in spec.numerical_cols:
            nb = len(bins[c]) - 1; i = np.clip(df[c].astype(int).values, 0, nb - 1); left, right = bins[c][i], bins[c][i + 1]
            v = left + rng.random_sample(len(i)) * (right - left); v = np.minimum(np.floor(v), np.ceil(right) - 1); o[c] = np.clip(v, *spec.feature_bounds[c])
        else: o[c] = np.array(dom_vals[c], dtype=object)[np.clip(df[c].astype(int).values, 0, len(dom_vals[c]) - 1)]
    return o[cols]
for k in range(a.draws):
    s = sample_mrf(model, a.n_out, np.random.default_rng(200 + k)).df
    undisc(s.reset_index(drop=True), np.random.RandomState(200 + k)).to_csv(os.path.join(a.outdir, f'{a.prefix}_seed{k}.csv'), index=False)
    print(f'    draw {k}: {len(s)} rows', flush=True)
json.dump({'dataset': a.dataset, 'release': a.release, 'release_md5': __import__('hashlib').md5(open(a.release, 'rb').read()).hexdigest()[:12],
           'eps': eps, 'n_out': a.n_out, 'draws': a.draws, 'seed': a.seed, 'iters': a.iters, 'known_total': N_total,
           'measurements': summary, 'consistency_mean_tv': float(np.mean([w[0] for w in worst])), 'consistency_worst': [(float(w[0]), w[1], w[2]) for w in worst[:5]],
           'private_data_read': False}, open(os.path.join(a.outdir, f'{a.prefix}_manifest.json'), 'w'), indent=1)
print('AIM_ON_RELEASE_DONE', flush=True)
