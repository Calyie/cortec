"""The decoder-with-selection control of the paper (Table 3, Section 6).

The maximum-entropy decoder is sampled at pool_factor times the requested size, each pool is passed through
CoRTeC-Select (Algorithm 2: rake, refine, the sub-bin rule), integer columns are floored under the integer rule,
and the draws are scored with the evaluator beside the plain decoder. The script reads the release and public
facts only: post-processing of the release, no privacy cost, CPU only.

Usage:
    python3 controls/decoder_select.py --dataset adult --release <cohort_statistics.json> --sizes 300 1000 \
        --draws 3 --pool-factor 3 --seed 42 --method decoder_select_eps2 --compare aim_on_release_eps2
"""
import argparse, glob, json, os, subprocess, sys
import numpy as np, pandas as pd
ROOT = os.environ.get('CORTEC_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from src.dataset_spec import get_spec
import src.datasets_auto  # noqa: F401  (registers the auto-configured datasets)
from src.generic_pipeline import select_to_release

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True); ap.add_argument('--release', required=True)
ap.add_argument('--sizes', type=int, nargs='+', default=[300, 1000]); ap.add_argument('--draws', type=int, default=3)
ap.add_argument('--pool-factor', type=int, default=3); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--method', default='decoder_select')
ap.add_argument('--compare', default=None, help='a scored arm to show beside this one, e.g. aim_on_release_eps2')
a = ap.parse_args()
spec = get_spec(a.dataset); release = json.load(open(a.release))
OUT = os.path.join(ROOT, 'results', 'baselines_modern', a.method)
RAW = os.path.join(ROOT, 'results', 'baselines_modern', 'raw', a.method)
os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)


def integer_columns(pool: pd.DataFrame) -> list:
    """The integer rule: integral public bounds and bin edges, and a majority of integral pool values."""
    cols = []
    for c in spec.numerical_cols:
        lo, hi = spec.feature_bounds[c]; edges = list(spec.public_bins.get(c, []))
        if not all(float(x).is_integer() for x in [lo, hi, *edges]):
            continue
        v = pd.to_numeric(pool[c], errors='coerce').dropna()
        if len(v) and (v == np.floor(v)).mean() > 0.5:
            cols.append(c)
    return cols


for n in a.sizes:
    pooldir = os.path.join(RAW, f'n{n}_pools')
    if not os.path.exists(os.path.join(pooldir, f'pool_seed{a.draws - 1}.csv')):
        subprocess.run([sys.executable, os.path.join(HERE, 'aim_on_release.py'), '--dataset', a.dataset,
                        '--release', a.release, '--n-out', str(a.pool_factor * n), '--draws', str(a.draws),
                        '--seed', str(a.seed), '--outdir', pooldir, '--prefix', 'pool'], check=True, cwd=ROOT)
    for k in range(a.draws):
        pool = pd.read_csv(os.path.join(pooldir, f'pool_seed{k}.csv'), keep_default_na=False)
        sel = select_to_release(spec, release, pool, n, seed=a.seed + k)
        for c in integer_columns(pool):
            sel[c] = np.floor(pd.to_numeric(sel[c], errors='coerce')).astype('Int64')
        sel.to_csv(os.path.join(OUT, f'n{n}_draw{k}.csv'), index=False)
        print(f'n={n} draw {k}: pool {len(pool)} rows -> selected {len(sel)}', flush=True)
    files = sorted(glob.glob(os.path.join(OUT, f'n{n}_draw[0-9].csv')))
    cmd = [sys.executable, os.path.join(HERE, 'score_draws.py'), '--method', a.method, '--n', str(n),
           '--dataset', a.dataset, '--files', *files]
    if a.compare:
        cmd += ['--compare', f"{a.compare}={os.path.join(ROOT, 'results', 'baselines_modern', a.compare, f'n{n}_draw*.csv')}"]
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f'SCORED n={n}', flush=True)
print('DECODER_SELECT_DONE')
