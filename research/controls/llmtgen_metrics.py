"""DP-LLMTGen paper protocol metrics (Tran & Xiong, arXiv 2406.01457, Table 1 and Appendix C): k-way total variation
distance for k = 1..5 between the synthetic table and the TEST split ("D and D' represent the synthetic and testing
datasets", App. C.1) over every k-subset of the attributes, numerical features quantiled into 20 groups, and XGBoost
accuracy / AUC trained on the synthetic table and tested on the test split, with the paper's grid search (5-fold CV
over n_estimators {100, 200, 300}, max_depth {3, 5, 10, 20}, learning rate {0.01, 0.05, 0.1}). Quantile edges are
fitted on the real training table (the paper does not say which set defines them).
Usage: python llmtgen_metrics.py --train real_train.csv --test real_test.csv --synthetic syn.csv [...] --out out.json
"""
import argparse, os, itertools, json, sys
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('--train', required=True); ap.add_argument('--test', required=True)
ap.add_argument('--synthetic', nargs='+', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--bins', type=int, default=20); ap.add_argument('--max-k', type=int, default=5)
ap.add_argument('--target', default='income'); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
def load(p):
    df = pd.read_csv(p, skipinitialspace=True, keep_default_na=False)
    df = df.rename(columns={c: c.replace('-', '_') for c in df.columns})
    for c in df.columns:
        if df[c].dtype == object: df[c] = df[c].astype(str).str.strip().str.rstrip('.')
    return df
tr, te = load(a.train), load(a.test)
cols = list(tr.columns); num = [c for c in cols if pd.api.types.is_numeric_dtype(tr[c])]; cat = [c for c in cols if c not in num]
edges = {c: np.unique(np.quantile(tr[c].astype(float), np.linspace(0, 1, a.bins + 1))) for c in num}
def disc(df):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in num:
            v = pd.to_numeric(df[c], errors='coerce').astype(float); e = edges[c]
            out[c] = np.clip(np.searchsorted(e, v, side='right') - 1, 0, len(e) - 2)
        else:
            out[c] = df[c].astype(str)
    return out
def kway_tvd(dr, ds, k):
    vals = []
    for combo in itertools.combinations(cols, k):
        pr = dr.groupby(list(combo), observed=True).size(); ps = ds.groupby(list(combo), observed=True).size()
        pr = pr / pr.sum(); ps = ps / ps.sum()
        idx = pr.index.union(ps.index)
        vals.append(0.5 * float((pr.reindex(idx, fill_value=0) - ps.reindex(idx, fill_value=0)).abs().sum()))
    return float(np.mean(vals)), len(vals)
def xgb_scores(train_df, test_df):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score
    def xy(df, ref):
        X = df[[c for c in cols if c != a.target]].copy()
        for c in cat:
            if c == a.target: continue
            X[c] = pd.Categorical(X[c], categories=sorted(ref[c].astype(str).unique()))
        for c in num: X[c] = pd.to_numeric(X[c], errors='coerce')
        y = (df[a.target].astype(str) == '>50K').astype(int).values
        return X, y
    Xtr, ytr = xy(train_df, tr); Xte, yte = xy(test_df, tr)
    from sklearn.model_selection import GridSearchCV
    base = xgb.XGBClassifier(enable_categorical=True, tree_method='hist', random_state=a.seed, n_jobs=int(os.environ.get('N_JOBS', '8')))
    grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 10, 20], 'learning_rate': [0.01, 0.05, 0.1]}
    clf = GridSearchCV(base, grid, cv=5, scoring='roc_auc', n_jobs=1).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return float(accuracy_score(yte, (p >= 0.5).astype(int))), float(roc_auc_score(yte, p))
dr = disc(te); res = {'protocol': 'dpllmtgen_paper', 'tvd_reference': 'test split', 'bins': a.bins, 'n_train': len(tr), 'n_test': len(te), 'synthetic': {}}
for p in a.synthetic:
    syn = load(p); ds = disc(syn); r = {'rows': len(syn)}
    for k in range(1, a.max_k + 1):
        r[f'tvd_{k}way'], n = kway_tvd(dr, ds, k); print(f'{p}: {k}-way TVD {r[f"tvd_{k}way"]:.4f} over {n} subsets', flush=True)
    r['xgb_acc'], r['xgb_auc'] = xgb_scores(syn, te); print(f'{p}: XGBoost ACC {r["xgb_acc"]:.4f} AUC {r["xgb_auc"]:.4f}', flush=True)
    res['synthetic'][p] = r
ref = {'rows': len(tr)}
for k in range(1, a.max_k + 1): ref[f'tvd_{k}way'], _ = kway_tvd(dr, disc(tr), k)
ref['xgb_acc'], ref['xgb_auc'] = xgb_scores(tr, te); res['training_set_reference'] = ref
print(f'training-set reference: XGBoost ACC {ref["xgb_acc"]:.4f} AUC {ref["xgb_auc"]:.4f} (paper: 0.872 / 0.928)')
json.dump(res, open(a.out, 'w'), indent=2); print('written', a.out)
