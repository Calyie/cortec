"""The one-shot control of the paper (Table 3, Appendix F.3).

Rows written by a frozen model from the column names alone, with no release and no private data: the research
pipeline's header-only prompt gives the dataset's one-line description, the column names in order, the two labels
of the target and a request for rows_per_call realistic rows. Integer columns are floored under the integer rule
and the draws are scored as delivered. The script spends API calls under a hard cap; the model's spend is written
to the manifest.

Usage:
    GEMINI_API_KEY=... python3 controls/oneshot_control.py --dataset adult --backend gemini \
        --model gemini-3.5-flash --sizes 300 1000 --draws 3 --cap-usd 6 --method oneshot_gemini36
"""
import argparse, glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd
ROOT = os.environ.get('CORTEC_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from src.dataset_spec import get_spec
import src.datasets_auto  # noqa: F401
from src.llm_generator import LLMSyntheticGenerator

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True); ap.add_argument('--backend', default='gemini'); ap.add_argument('--model', required=True)
ap.add_argument('--sizes', type=int, nargs='+', default=[300, 1000]); ap.add_argument('--draws', type=int, default=3)
ap.add_argument('--rows-per-call', type=int, default=25); ap.add_argument('--cap-usd', type=float, default=6.0)
ap.add_argument('--method', default='oneshot')
a = ap.parse_args()
spec = get_spec(a.dataset)
gen = LLMSyntheticGenerator(backend=a.backend, model=a.model, rows_per_call=a.rows_per_call, spec=spec)
gen.budget_usd = a.cap_usd
OUT = os.path.join(ROOT, 'results', 'baselines_modern', a.method)
RAW = os.path.join(ROOT, 'results', 'baselines_modern', 'raw', a.method)
os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)


def spend():
    s = getattr(gen, 'spend_usd', None)
    return float(s()) if callable(s) else s


def integer_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in spec.numerical_cols:
        lo, hi = spec.feature_bounds[c]; edges = list(spec.public_bins.get(c, []))
        if not all(float(x).is_integer() for x in [lo, hi, *edges]):
            continue
        v = pd.to_numeric(df[c], errors='coerce').dropna()
        if len(v) and (v == np.floor(v)).mean() > 0.5:
            cols.append(c)
    return cols


log = []
for n in a.sizes:
    for k in range(a.draws):
        raw = os.path.join(RAW, f'n{n}_draw{k}.csv')
        if os.path.exists(raw):
            continue
        t0 = time.time(); df = gen.generate_header_only(n_total=n); df.to_csv(raw, index=False)
        log.append({'n': n, 'draw': k, 'rows': len(df), 'seconds': round(time.time() - t0), 'cumulative_spend_usd': spend()})
        print(log[-1], flush=True)
        out = df.copy()
        for c in integer_columns(out):
            out[c] = np.floor(pd.to_numeric(out[c], errors='coerce')).astype('Int64')
        out.to_csv(os.path.join(OUT, f'n{n}_draw{k}.csv'), index=False)
json.dump({'backend': a.backend, 'model': a.model, 'rows_per_call': a.rows_per_call, 'cap_usd': a.cap_usd,
           'draws': log, 'spend_usd': spend()}, open(os.path.join(RAW, 'manifest.json'), 'w'), indent=1, default=str)
for n in a.sizes:
    files = sorted(glob.glob(os.path.join(OUT, f'n{n}_draw[0-9].csv')))
    subprocess.run([sys.executable, os.path.join(HERE, 'score_draws.py'), '--method', a.method, '--n', str(n),
                    '--dataset', a.dataset, '--files', *files], check=True, cwd=ROOT)
    print(f'SCORED n={n}', flush=True)
print('ONESHOT_DONE')
