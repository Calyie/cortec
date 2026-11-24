"""Tab-PE paper protocol metrics (Tran et al., ICML 2026, Appendix C.2, Table 8 for Adult): downstream TabICL (fit on
the synthetic table, evaluated on the real test split: accuracy and macro F1), k-way Wasserstein distance (average over
all k-attribute subsets, label included, between the synthetic table and the PRIVATE training table, POT), and
precision / recall (Sajjadi et al. 2018, k = 5) in the embedding of an autoencoder trained on the private table.
Choices the paper does not state, made here and recorded in the output: numerical attributes scaled to [0, 1] by the
training range and categorical attributes one-hot encoded before the Euclidean ground cost; WD computed by exact EMD
on random subsamples of at most --wd-n rows per side (uniform weights); the autoencoder is a 2-layer MLP (64-16-64)
on the same encoding, trained on the private table; precision / recall use at most --pr-n real rows.
Usage: python tabpe_metrics.py --train real_train.csv --test real_test.csv --synthetic syn.csv [...] --out out.json
"""
import argparse, itertools, json, sys, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('--train', required=True); ap.add_argument('--test', required=True); ap.add_argument('--synthetic', nargs='+', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--max-k', type=int, default=3); ap.add_argument('--wd-n', type=int, default=2000)
ap.add_argument('--pr-n', type=int, default=5000); ap.add_argument('--target', default='income'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--skip-tabicl', action='store_true')
a = ap.parse_args(); rng = np.random.default_rng(a.seed)
def load(p):
    df = pd.read_csv(p, skipinitialspace=True, keep_default_na=False).rename(columns=lambda c: c.replace('-', '_'))
    for c in df.columns:
        if df[c].dtype == object: df[c] = df[c].astype(str).str.strip().str.rstrip('.')
    return df
tr, te = load(a.train), load(a.test); cols = list(tr.columns)
num = [c for c in cols if pd.api.types.is_numeric_dtype(tr[c])]; cat = [c for c in cols if c not in num]
lo = {c: float(tr[c].min()) for c in num}; hi = {c: float(tr[c].max()) for c in num}
cats = {c: sorted(tr[c].astype(str).unique()) for c in cat}
def encode(df, subset=None):
    """[0,1]-scaled numericals and one-hot categoricals for the chosen attribute subset (default all)."""
    blocks = []
    for c in (subset or cols):
        if c in num:
            v = pd.to_numeric(df[c], errors='coerce').astype(float).fillna(lo[c]).to_numpy()
            blocks.append(((v - lo[c]) / max(hi[c] - lo[c], 1e-9)).reshape(-1, 1))
        else:
            m = {v: i for i, v in enumerate(cats[c])}; idx = df[c].astype(str).map(lambda v: m.get(v, -1)).to_numpy()
            oh = np.zeros((len(df), len(cats[c]))); ok = idx >= 0; oh[np.arange(len(df))[ok], idx[ok]] = 1.0; blocks.append(oh)
    return np.hstack(blocks)
def kway_wd(real, syn, k):
    import ot
    vals = []
    for combo in itertools.combinations(cols, k):
        R = encode(real.sample(min(a.wd_n, len(real)), random_state=int(rng.integers(1e9))), list(combo))
        Sy = encode(syn.sample(min(a.wd_n, len(syn)), random_state=int(rng.integers(1e9))), list(combo))
        M = ot.dist(R, Sy, metric='euclidean'); vals.append(float(ot.emd2(np.ones(len(R)) / len(R), np.ones(len(Sy)) / len(Sy), M)))
    return float(np.mean(vals)), len(vals)
def autoencoder_embed(fit_df):
    from sklearn.neural_network import MLPRegressor
    X = encode(fit_df); ae = MLPRegressor(hidden_layer_sizes=(64, 16, 64), max_iter=200, random_state=a.seed).fit(X, X)
    W, b = ae.coefs_, ae.intercepts_
    def embed(df):
        h = encode(df)
        for i in range(2):  # input -> 64 -> 16 (the bottleneck)
            h = np.maximum(h @ W[i] + b[i], 0.0)
        return h
    return embed
def precision_recall(emb_real, emb_syn, k=5):
    from sklearn.neighbors import NearestNeighbors
    def cover(A, B):   # fraction of A within the k-NN radius of some B point (radius measured within B)
        nnB = NearestNeighbors(n_neighbors=k + 1).fit(B); rB = nnB.kneighbors(B)[0][:, -1]
        d, j = NearestNeighbors(n_neighbors=1).fit(B).kneighbors(A); return float(np.mean(d[:, 0] <= rB[j[:, 0]]))
    return cover(emb_syn, emb_real), cover(emb_real, emb_syn)   # precision, recall
def tabicl_scores(train_df):
    from tabicl import TabICLClassifier
    from sklearn.metrics import accuracy_score, f1_score
    X = train_df[[c for c in cols if c != a.target]].copy(); Xt = te[[c for c in cols if c != a.target]].copy()
    for c in cat:
        if c != a.target: X[c] = X[c].astype('category'); Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)
    y = (train_df[a.target].astype(str) == '>50K').astype(int).values; yt = (te[a.target].astype(str) == '>50K').astype(int).values
    clf = TabICLClassifier(random_state=a.seed).fit(X, y); p = clf.predict(Xt)
    return float(accuracy_score(yt, p)), float(f1_score(yt, p, average='macro'))
embed = autoencoder_embed(tr)
real_sub = tr.sample(min(a.pr_n, len(tr)), random_state=a.seed); emb_real = embed(real_sub)
res = {'protocol': 'tabpe_paper', 'wd_reference': 'private training table', 'wd_subsample': a.wd_n, 'n_train': len(tr), 'n_test': len(te), 'synthetic': {}}
for p in a.synthetic:
    syn = load(p); r = {'rows': len(syn)}
    for k in range(1, a.max_k + 1):
        r[f'wd_{k}way'], n = kway_wd(tr, syn, k); print(f'{p}: {k}-WD {r[f"wd_{k}way"]:.4f} over {n} subsets', flush=True)
    r['precision'], r['recall'] = precision_recall(emb_real, embed(syn.sample(min(a.pr_n, len(syn)), random_state=a.seed)))
    print(f'{p}: precision {r["precision"]:.3f} recall {r["recall"]:.3f}', flush=True)
    if not a.skip_tabicl:
        r['tabicl_acc'], r['tabicl_macro_f1'] = tabicl_scores(syn); print(f'{p}: TabICL accuracy {r["tabicl_acc"]:.4f} macro F1 {r["tabicl_macro_f1"]:.4f}', flush=True)
    res['synthetic'][p] = r
ref = {'rows': len(tr)}
for k in range(1, a.max_k + 1): ref[f'wd_{k}way'], _ = kway_wd(tr, tr, k)
ref['precision'], ref['recall'] = precision_recall(emb_real, embed(tr.sample(min(a.pr_n, len(tr)), random_state=a.seed + 1)))
if not a.skip_tabicl:
    ref['tabicl_acc'], ref['tabicl_macro_f1'] = tabicl_scores(tr.sample(min(2000, len(tr)), random_state=a.seed))
    print(f'reference (2,000 real training rows): TabICL accuracy {ref["tabicl_acc"]:.4f} macro F1 {ref["tabicl_macro_f1"]:.4f} (paper UB 84.41 / 75.68 on the full set)')
res['training_reference'] = ref; json.dump(res, open(a.out, 'w'), indent=2); print('written', a.out)
