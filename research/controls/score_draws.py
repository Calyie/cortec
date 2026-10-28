"""Score a method whose draws come from separate runs (one file per draw): take the first n rows of each file after a
fixed shuffle, write results/baselines_modern/<method>/n<N>_draw<k>.csv, and run OUR evaluator on them.
Usage: python score_draws.py --method tabpe_eps2_pop1000 --n 300 --files f0.csv f1.csv f2.csv [--compare label=glob ...]
"""
import argparse, json, os, subprocess, sys
import pandas as pd
ROOT = os.environ.get('CORTEC_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from src.data_loader import COLUMN_NAMES
ap = argparse.ArgumentParser()
ap.add_argument('--method', required=True); ap.add_argument('--n', type=int, required=True)
ap.add_argument('--files', nargs='+', required=True); ap.add_argument('--compare', nargs='*', default=[])
ap.add_argument('--dataset', default='adult'); ap.add_argument('--seed', type=int, default=42, help='split seed passed to the evaluator (default 42, the shipped split)')
a = ap.parse_args()
out = os.path.join(ROOT, 'results', 'baselines_modern', a.method); os.makedirs(out, exist_ok=True)
paths = []
if a.dataset == 'adult':
    cols, target = COLUMN_NAMES, 'income'
else:
    from src.dataset_spec import get_spec
    import src.datasets_extra, src.datasets_regulated, src.datasets_synthetic, src.datasets_clinical, src.datasets_auto  # noqa: F401
    _spec = get_spec(a.dataset); cols, target = list(_spec.column_names), _spec.target_col
for k, f in enumerate(a.files):
    df = pd.read_csv(f, keep_default_na=False).rename(columns=lambda c: c.replace('-', '_'))[cols]
    df[target] = df[target].astype(str).str.strip().str.rstrip('.')
    df = df.sample(frac=1.0, random_state=20261021).reset_index(drop=True).iloc[:a.n]
    if len(df) < a.n:
        print(f'  NOTE: {f} holds {len(df)} rows, {a.n - len(df)} short of n={a.n}; scored as delivered', flush=True)
    p = f'{out}/n{a.n}_draw{k}.csv'; df.to_csv(p, index=False); paths.append(p)
spec = {a.method: [f'{out}/n{a.n}_draw*.csv']}
for item in a.compare:
    lab, g = item.split('=', 1); spec[lab] = [g]
sp = f'{out}/spec_n{a.n}.json'; json.dump(spec, open(sp, 'w'))
print(f'{a.method}: n={a.n}, {len(paths)} draws -> evaluator', flush=True)
subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'evaluate_generic.py'), '--dataset', a.dataset, '--spec-json', sp, '--out', f'{out}/eval_n{a.n}.json', '--seed', str(a.seed)], cwd=ROOT, check=True)
