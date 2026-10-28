"""Reference-implementation AIM and MST baselines (Ryan McKenna's private-pgm mechanisms, mbi 1.1) on exactly the
same discretised table and the same uniform-within-bin decoding as run_dataset.py's baselines stage (spec public
bins; rare values handled by the mechanisms themselves). Written because smartnoise-synth 1.0.8's MST (and possibly
AIM) returns a wrong joint distribution for columns with rare categories even at epsilon = 1000 (verified
2026-10-21). Usage:
  python pgm_baselines.py --train real_train.csv --epsilon 1.0 --delta 1e-9 --n-out 2000 --draws 3 --seed 42
                          --methods mst aim --outdir DIR [--max-model-size 80] [--degree 2] [--max-iters 1000]
Column names may use '-' or '_'; the output uses the spec's names. Each draw k samples the fitted model with seed
200 + k and decodes with RandomState(200 + k), as the baselines stage does. The mechanism itself is seeded once
with --seed (np.random.seed) before each fit so that reruns are reproducible."""
import argparse, itertools, os, sys, time
import numpy as np, pandas as pd
ROOT = os.environ.get('CORTEC_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
MECH = os.environ.get('PGM_MECHANISMS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'privatepgm_mechanisms'))
sys.path.insert(0, MECH); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--train', required=True); ap.add_argument('--dataset', default='adult')
ap.add_argument('--epsilon', type=float, required=True); ap.add_argument('--delta', type=float, default=1e-9)
ap.add_argument('--n-out', type=int, required=True); ap.add_argument('--draws', type=int, default=3); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--methods', nargs='+', default=['mst', 'aim']); ap.add_argument('--outdir', required=True)
ap.add_argument('--max-model-size', type=float, default=80); ap.add_argument('--degree', type=int, default=2); ap.add_argument('--max-iters', type=int, default=1000)
ap.add_argument('--roundtrip', action='store_true', help='also write real-roundtrip draws (real rows binned and decoded)')
a = ap.parse_args()
from src.dataset_spec import get_spec
import src.datasets_extra, src.datasets_regulated, src.datasets_synthetic, src.datasets_clinical, src.datasets_auto  # noqa: F401
from mbi import Dataset, Domain
import mst as MSTM
from aim import AIM
from mrf_sampler import sample_mrf
from mbi import estimation
spec = get_spec(a.dataset); cols = spec.column_names
bins = {c: np.asarray(spec.public_bins[c], dtype=float) for c in spec.numerical_cols}
tr = pd.read_csv(a.train, skipinitialspace=True); tr.columns = [c.strip().replace('-', '_') for c in tr.columns]
tr = tr[cols]
def disc(df):
    d = df.copy()
    for c in spec.numerical_cols:
        lo, hi = spec.feature_bounds[c]; v = pd.to_numeric(d[c], errors='coerce').fillna(lo).clip(lo, hi)
        d[c] = np.clip(np.digitize(v, bins[c]) - 1, 0, len(bins[c]) - 2).astype(int)
    for c in spec.categorical_cols + [spec.target_col]:
        d[c] = d[c].astype(str).str.strip().str.rstrip('.')
    return d[cols]
td = disc(tr)
# integer coding for the mechanisms: numerical columns keep their bin index over the FULL public domain (so the
# domain is public, not data-dependent); categorical columns use the spec's public category lists when available,
# else the sorted observed values (recorded in the manifest).
codes, inv, dom = {}, {}, {}
for c in cols:
    if c in spec.numerical_cols:
        n = len(bins[c]) - 1; codes[c] = td[c].astype(int).values; inv[c] = None; dom[c] = n
    else:
        cats = list(getattr(spec, 'categories', {}).get(c, [])) if hasattr(spec, 'categories') else []
        cats = [str(x).strip() for x in cats] if cats else sorted(td[c].unique().tolist())
        lut = {v: i for i, v in enumerate(cats)}; miss = set(td[c].unique()) - set(cats)
        if miss: cats = cats + sorted(miss); lut = {v: i for i, v in enumerate(cats)}
        codes[c] = td[c].map(lut).astype(int).values; inv[c] = np.array(cats, dtype=object); dom[c] = len(cats)
data = Dataset(pd.DataFrame(codes, columns=cols), Domain(cols, [dom[c] for c in cols]))
print(f'train {len(td)} rows; domain sizes {dict(zip(cols, [dom[c] for c in cols]))}', flush=True)
def undisc(df, rng):
    o = pd.DataFrame(index=df.index)
    for c in cols:
        if c in spec.numerical_cols:
            nb = len(bins[c]) - 1; i = np.clip(df[c].astype(int).values, 0, nb - 1)
            left, right = bins[c][i], bins[c][i + 1]
            v = left + rng.random_sample(len(i)) * (right - left)
            v = np.minimum(np.floor(v), np.ceil(right) - 1); o[c] = np.clip(v, *spec.feature_bounds[c])
        else:
            o[c] = inv[c][np.clip(df[c].astype(int).values, 0, len(inv[c]) - 1)]
    return o[cols]
os.makedirs(a.outdir, exist_ok=True)

# ---- fit validation (uses only the fitted model, never the data): mbi 1.1.0's mirror descent sometimes diverges at
# larger budgets, either to NaN potentials (AIM then raises inside its exponential mechanism) or to a collapsed model
# (a column with all its mass on one value). Each fit is checked and retried with a new seed, the third attempt with
# a halved mirror-descent step; the attempt count is printed and kept in the output folder.
import mbi.estimation as _est
_orig_md = _est.mirror_descent
_LAST = {}  # the measurements of the latest mirror-descent call, for the consistency gate below
from mbi import marginal_oracles as _mo
_ORACLE = os.environ.get('PGM_ORACLE', 'fast')  # 'stable' = log-space message passing (mbi's message_passing_stable) instead of the default fast oracle
def _oracle_kw(kw):
    if _ORACLE == 'stable' and 'marginal_oracle' not in kw: kw['marginal_oracle'] = _mo.message_passing_stable
    return kw
def _capture_md(domain, loss_fn, **kw):
    _LAST['measurements'] = loss_fn if isinstance(loss_fn, list) else None
    return _orig_md(domain, loss_fn, **_oracle_kw(kw))
def _scaled_md(scale):
    def md(domain, loss_fn, **kw):
        if kw.get('stepsize') is None:
            # mbi's default is alpha = 2 / (L * known_total); it is not exposed, so the retry asks for a smaller
            # explicit step via the total: 2 / (L * total) with L = len(measurements) is the documented default
            total = kw.get('known_total')
            if total is None:
                try: total = float(max(1.0, loss_fn[0].noisy_measurement.sum()))
                except Exception: total = None
            L = len(loss_fn) if hasattr(loss_fn, '__len__') else 1
            if total: kw['stepsize'] = scale * 2.0 / (L * total)
        _LAST['measurements'] = loss_fn if isinstance(loss_fn, list) else None
        return _orig_md(domain, loss_fn, **_oracle_kw(kw))
    return md
def _model_problem(model):
    for c in model.domain.attrs:
        m = np.asarray(model.project((c,)).datavector(), dtype=float)
        if not np.all(np.isfinite(m)): return f'{c}: non-finite marginal'
        if m.sum() <= 0: return f'{c}: zero mass'
        if m.size > 1 and m.max() / m.sum() > 0.995: return f'{c}: {m.max() / m.sum():.4f} of the mass on one value'
    return None
def _consistency(model, measurements):
    """Mean and worst TV between the fitted model's marginals and the noisy measurements it was fitted to (clipped
    at zero and normalised). Uses only the DP outputs. A fit that diverged or stalled near uniform shows a mean
    of 0.2 or more; a converged fit at these budgets sits well below 0.1."""
    if not measurements: return None, None
    tvs = []
    for M in measurements:
        cl = tuple(M.clique); y = np.clip(np.asarray(M.noisy_measurement, dtype=float), 0, None)
        if y.sum() <= 0: continue
        f = np.asarray(model.project(cl).datavector(), dtype=float)
        if f.size != y.size or f.sum() <= 0: continue
        tvs.append(0.5 * np.abs(f / f.sum() - y / y.sum()).sum())
    return (float(np.mean(tvs)), float(np.max(tvs))) if tvs else (None, None)
GATE = float(os.environ.get('PGM_CONSISTENCY_GATE', '0.15'))
def fit_validated(fit_fn, label, outdir):
    """Attempt policy v2 (2026-10-24): four standard-step attempts with fresh seeds, then two half-step attempts
    with three times the iterations; every candidate must pass the marginal checks AND the consistency gate. v1
    (one standard retry, then a half step with the same iterations, no quality gate) accepted a near-uniform model
    on the NHANES seed-43 and seed-44 splits."""
    plan = [(k, 1.0, 1) for k in range(4)] + [(4, 0.5, 3), (5, 0.5, 3)]
    for attempt, scale, itx in plan:
        np.random.seed(a.seed + attempt); _LAST['measurements'] = None
        _est.mirror_descent = _capture_md if scale == 1.0 else _scaled_md(scale)
        saved_iters = a.max_iters; a.max_iters = saved_iters * itx
        try:
            model = fit_fn(); problem = _model_problem(model)
            if problem is None:
                mean_tv, max_tv = _consistency(model, _LAST.get('measurements'))
                if mean_tv is not None and mean_tv > GATE: problem = f'consistency mean TV {mean_tv:.3f} (max {max_tv:.3f}) above the gate {GATE}'
                else: problem = None; cons = f'consistency mean TV {mean_tv:.3f} max {max_tv:.3f}' if mean_tv is not None else 'consistency not computed'
        except Exception as e:
            problem = f'raised {type(e).__name__}: {str(e)[:80]}'; model = None
        a.max_iters = saved_iters; _est.mirror_descent = _orig_md
        tag = f'seed {a.seed + attempt}' + (f', step x{scale}, iters x{itx}' if scale != 1.0 else '')
        msg = f'  [{label}] attempt {attempt} ({tag}): ' + (f'valid, {cons}' if problem is None else f'rejected, {problem}')
        print(msg, flush=True); open(os.path.join(outdir, f'{label}_attempts.log'), 'a').write(msg + '\n')
        if problem is None:
            try:
                ms = _LAST.get('measurements') or []; rows_ = []
                for M in ms:
                    cl = tuple(M.clique); y = np.clip(np.asarray(M.noisy_measurement, dtype=float), 0, None); f = np.asarray(model.project(cl).datavector(), dtype=float)
                    tv = 0.5 * float(np.abs(f / f.sum() - y / y.sum()).sum()) if (y.sum() > 0 and f.sum() > 0 and f.size == y.size) else None
                    rows_.append({'clique': list(cl), 'stddev': float(M.stddev), 'cells': int(y.size), 'tv_fit_vs_measured': tv, 'has_target': spec.target_col in cl})
                import json as _json
                _json.dump({'attempt': attempt, 'seed': a.seed + attempt, 'oracle': _ORACLE, 'n_measurements': len(rows_), 'target_measurements': sum(r['has_target'] for r in rows_), 'measurements': sorted(rows_, key=lambda r: -(r['tv_fit_vs_measured'] or 0))}, open(os.path.join(outdir, f'{label}_measurements.json'), 'w'), indent=1)
            except Exception as e: print(f'  [{label}] diagnostics not written: {e}', flush=True)
            return model
    raise SystemExit(f'{label}: no valid fit in {len(plan)} attempts')
if a.roundtrip:
    for k in range(a.draws):
        s = data.df.sample(n=min(a.n_out, len(td)), random_state=100 + k)
        undisc(s, np.random.RandomState(100 + k)).to_csv(os.path.join(a.outdir, f'real-roundtrip_seed{k}.csv'), index=False)
    print(f'wrote {a.draws} real-roundtrip draws', flush=True)
for m in a.methods:
    t0 = time.time(); print(f'\n── {m.upper()} reference (eps={a.epsilon}, delta={a.delta}) ──', flush=True)
    if m == 'mst':
        # the reference MST body, with the model's rows drawn by the axis-aligned sampler instead of mbi 1.1.0's
        rho = MSTM.cdp_rho(a.epsilon, a.delta); sigma = np.sqrt(3 / (2 * rho)); state = {}
        def fit_mst():
            log1 = MSTM.measure(data, [(c,) for c in data.domain], sigma)
            data2, log1b, undo = MSTM.compress_domain(data, log1)
            cliques = MSTM.select(data2, rho / 3.0, log1b); log2 = MSTM.measure(data2, cliques, sigma)
            state['undo'] = undo; state['cliques'] = cliques
            return _est.mirror_descent(data2.domain, log1b + log2, iters=10000)
        est = fit_validated(fit_mst, 'mst', a.outdir); undo = state['undo']; cliques = state['cliques']
        print(f'  fitted in {time.time()-t0:.0f}s; tree edges {cliques}', flush=True)
        for k in range(a.draws):
            s = undo(sample_mrf(est, a.n_out, np.random.default_rng(200 + k))).df
            undisc(s.reset_index(drop=True), np.random.RandomState(200 + k)).to_csv(os.path.join(a.outdir, f'mst_seed{k}.csv'), index=False)
            print(f'    draw {k}: {len(s)} rows', flush=True)
    elif m == 'aim':
        workload = [(cl, 1.0) for cl in itertools.combinations(cols, a.degree)]
        def fit_aim():
            mech = AIM(a.epsilon, a.delta, max_model_size=a.max_model_size, max_iters=a.max_iters)
            model, synth = mech.run(data, workload, num_synth_rows=a.n_out); return model
        model = fit_validated(fit_aim, 'aim', a.outdir)
        print(f'  fitted in {time.time()-t0:.0f}s', flush=True)
        for k in range(a.draws):
            s = sample_mrf(model, a.n_out, np.random.default_rng(200 + k)).df
            undisc(s.reset_index(drop=True), np.random.RandomState(200 + k)).to_csv(os.path.join(a.outdir, f'aim_seed{k}.csv'), index=False)
            print(f'    draw {k}: {len(s)} rows', flush=True)
print('PGM_BASELINES_DONE', flush=True)
