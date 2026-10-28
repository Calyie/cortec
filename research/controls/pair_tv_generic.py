"""Per-pair two-way TV diagnosis for any registered dataset: mean pair TV over all column pairs (10 quantile bins per
numeric from the seed's training split), averaged over draws, for the labelled arms, against the training split.
Usage: python pair_tv_generic.py --dataset bank_marketing_auto --seed 42 --n 300 --arms cortec_bank_qwen3_14b_eps2 aim_on_release_bank_eps2 aim_ref_bank_s42"""
import argparse, sys, glob, itertools, numpy as np, pandas as pd
ROOT = os.environ.get('CORTEC_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--n', type=int, required=True); ap.add_argument('--arms', nargs='+', required=True); ap.add_argument('--top', type=int, default=8)
a = ap.parse_args()
from run_dataset import split
from src.dataset_spec import get_spec
import src.datasets_extra, src.datasets_regulated, src.datasets_synthetic, src.datasets_clinical, src.datasets_auto  # noqa: F401
spec = get_spec(a.dataset); cols = list(spec.column_names); NUM = list(spec.numerical_cols)
tr, te = split(spec, seed=a.seed)
edges = {c: np.unique(np.quantile(pd.to_numeric(tr[c], errors='coerce').dropna().astype(float), np.linspace(0, 1, 11))) for c in NUM}
def binned(df):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in NUM:
            v = pd.to_numeric(df[c], errors='coerce').astype(float); e = edges[c]
            out[c] = np.clip(np.searchsorted(e[1:-1], v, side='right'), 0, max(len(e) - 2, 0))
        else: out[c] = df[c].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    return out
R = binned(tr)
def pair_tv(syn):
    Sb = binned(syn); res = {}
    for x, y in itertools.combinations(cols, 2):
        p = R.groupby([x, y]).size() / len(R); q = Sb.groupby([x, y]).size() / len(Sb)
        res[f'{x}~{y}'] = 0.5 * float(p.subtract(q, fill_value=0).abs().sum())
    return res
res = {}
for lab in a.arms:
    fs = sorted(glob.glob(f'{ROOT}/results/baselines_modern/{lab}/n{a.n}_draw[0-9].csv'))
    per = [pair_tv(pd.read_csv(f, keep_default_na=False)) for f in fs]
    res[lab] = {p: float(np.mean([d[p] for d in per])) for p in per[0]}
    print(f'{lab:32s} {len(fs)} draws  mean pair TV {np.mean(list(res[lab].values())):.3f}')
A, Bl = a.arms[0], a.arms[1]
d = sorted(((res[A][p] - res[Bl][p], p) for p in res[A]), reverse=True)
print(f'\npairs where {A} is WORSE than {Bl} by most:'); [print(f'  {p:34s} ' + '  '.join(f'{l.split("_")[0]} {res[l][p]:.3f}' for l in a.arms)) for v, p in d[:a.top]]
print(f'pairs where {A} is BETTER than {Bl} by most:'); [print(f'  {p:34s} ' + '  '.join(f'{l.split("_")[0]} {res[l][p]:.3f}' for l in a.arms)) for v, p in d[-a.top:][::-1]]
