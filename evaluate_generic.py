"""
evaluate_generic.py — the full metric suite, driven by a DatasetSpec.

Mirrors evaluate_suite.py (which is Adult-only through its evaluate_fidelity imports) so the
healthcare and financial datasets are scored on exactly the same quantities:

  marginal fidelity   1-way and 2-way TV distance, binned with the spec's PUBLIC bins
  conditional fidelity  P(target | group) error on the groups the method IS told about
                        (the spec's stratification) and on `heldout_groups`, which share no
                        column with the stratification or the conditional table
  pMSE ratio          PATE-CTGAN's own fidelity metric
  3-way workload error  AIM's own metric, all three of its workloads
  utility             TSTR under logistic regression, random forest and gradient boosting,
                      against the train-on-real ceiling

Every conclusion in this project has depended on reporting fidelity AND utility together and on
checking a metric's floor against its ceiling, so `real-sample` and `shuffled-target` rows are
computed automatically for every dataset rather than left to the caller.
"""
from __future__ import annotations
import argparse, glob, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset_spec import get_spec, DatasetSpec
import src.datasets_extra       # noqa: F401
import src.datasets_regulated   # noqa: F401  (six untouched regulated benchmarks)
import src.datasets_clinical    # noqa: F401  (nhanes, mimic3_demo)
import src.datasets_synthetic   # noqa: F401  (renal_registry, the contamination control)
import src.datasets_auto        # noqa: F401  (auto-configured *_auto variants)
from workload_error import workload_error_report

RNG = 42


def _encoders(spec: DatasetSpec, real: pd.DataFrame) -> dict:
    enc = {}
    for c in spec.categorical_cols + [spec.target_col]:
        cats = sorted(real[c].astype(str).str.strip().unique().tolist())
        enc[c] = {v: i for i, v in enumerate(cats)}
        enc[c]["__mode__"] = real[c].astype(str).str.strip().mode()[0]
    return enc


def encode(spec: DatasetSpec, df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in spec.categorical_cols:
        m = enc[c]; mode_i = m[m["__mode__"]]
        out[c] = df[c].astype(str).str.strip().map(lambda v: m.get(v, mode_i)).astype(int)
    for c in spec.numerical_cols:
        lo, hi = spec.feature_bounds[c]
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(lo).clip(lo, hi).astype(float)
    if spec.target_col in df.columns:
        m = enc[spec.target_col]; mode_i = m[m["__mode__"]]
        out[spec.target_col] = df[spec.target_col].astype(str).str.strip() \
            .map(lambda v: m.get(v, mode_i)).astype(int)
    return out


def _tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _norm_categorical(s: pd.Series) -> pd.Series:
    """Strip a trailing '.0' from integer-like categorical values.

    Several ID-coded columns (admission_type_id, discharge_disposition_id, ...) hold integers
    stored as strings. A generator that emits them as floats writes "1.0" where the real data has
    "1", and every such column then scores total variation 1.000 against a distribution it in fact
    matched almost exactly — Gemini Flash's 1-way TV read 0.188 instead of 0.035 for this reason
    alone, which would have led us to reject a generator that is the most accurate we measured.
    Normalising here keeps the comparison about distributions rather than about formatting.
    """
    v = s.astype(str).str.strip()
    return v.str.replace(r"^(-?\d+)\.0$", r"\1", regex=True)


def _dist(spec, s: pd.Series, col: str) -> dict:
    if col in spec.numerical_cols:
        edges = np.asarray(spec.public_bins[col], dtype=float)
        v = pd.to_numeric(s, errors="coerce").dropna()
        # CLIP, do not drop. `pd.cut` maps anything outside the outer edges to NaN and
        # `value_counts` then discards it, so a generator emitting an age of 200 would have those
        # rows vanish from the comparison and the rest renormalised -- the error would make the
        # distribution look BETTER the more out-of-range values it produced. Clipping puts them in
        # the boundary bin, where they show up as excess mass and are scored. Bounds are public, so
        # this costs nothing. (Verified: no file in this project currently has out-of-range values,
        # because generation enforces domain bounds -- this guards the metric, not a live bug.)
        v = v.clip(edges[0], edges[-1])
        cats = pd.cut(v, bins=edges, include_lowest=True)
    else:
        cats = _norm_categorical(s)
    vc = cats.value_counts(normalize=True)
    return {str(k): float(v) for k, v in vc.items()}


# Above this many column pairs the 2-way metric subsamples; at or below it, every pair is scored.
# 15 columns give 105 pairs and 19 give 171, so in practice every dataset here is scored complete.
MAX_PAIRS_EXHAUSTIVE = 200


def marginal_tv(spec, real, synth, order=1, n_pairs=None) -> float:
    """Marginal total variation. The 2-way metric scores EVERY column pair, not a subsample.

    It used to sample 20 pairs by indexing into the column list, which made the result depend on
    the order the columns happen to be declared in. Two evaluators in this project list them
    differently and, from the same seed, scored pair sets overlapping in only 2 of 20 -- so the
    same synthetic data returned 0.079 under one and 0.115 under the other. Measured across 400
    random 20-pair draws the subsample carries a standard deviation of 0.005-0.009, which is the
    same size as the between-method differences it was being used to compare.

    With 15 columns there are 105 pairs, so there is no reason to subsample. `n_pairs` is retained
    for datasets wide enough to need it, and is applied only above MAX_PAIRS_EXHAUSTIVE.
    """
    cols = spec.column_names
    if order == 1:
        return float(np.mean([_tv(_dist(spec, real[c], c), _dist(spec, synth[c], c)) for c in cols]))
    import itertools
    total = len(cols) * (len(cols) - 1) // 2
    if n_pairs is None and total <= MAX_PAIRS_EXHAUSTIVE:
        pairs = set(itertools.combinations(sorted(cols), 2))
    else:
        k = min(n_pairs or 20, total)
        rng = np.random.RandomState(RNG)
        pairs = set()
        while len(pairs) < k:
            a, b = rng.choice(len(cols), 2, replace=False)
            pairs.add(tuple(sorted((cols[a], cols[b]))))

    def joint(df, a, b):
        ka = (pd.cut(pd.to_numeric(df[a], errors="coerce"),
                     bins=np.asarray(spec.public_bins[a], float), include_lowest=True).astype(str)
              if a in spec.numerical_cols else _norm_categorical(df[a]))
        kb = (pd.cut(pd.to_numeric(df[b], errors="coerce"),
                     bins=np.asarray(spec.public_bins[b], float), include_lowest=True).astype(str)
              if b in spec.numerical_cols else _norm_categorical(df[b]))
        vc = (ka + "|" + kb).value_counts(normalize=True)
        return {str(k): float(v) for k, v in vc.items()}
    return float(np.mean([_tv(joint(real, a, b), joint(synth, a, b)) for a, b in sorted(pairs)]))


def conditional_tv(spec, real, synth, cols=None, bands=None, min_n=30) -> float:
    """P(target | group) error, frequency-weighted by the REAL data's group sizes.

    `bands` groups by the spec's Band objects (the banded form the method is actually told
    about). `cols` groups by raw column values and is only appropriate for low-cardinality
    columns — grouping raw on a numerical column like LIMIT_BAL (81 distinct values) shatters
    it into groups that all fall below min_n and makes the metric meaningless.
    """
    def _col_key(df, c):
        # A numerical column grouped by raw value shatters into singleton groups that all fall
        # below min_n, which silently makes the metric meaningless (the docstring warned about it,
        # so numerical columns simply could not be used as held-out families). Bin it with the
        # spec's PUBLIC bin edges: public information, no privacy cost, and it makes the strongest
        # columns usable — on NHANES waist_cm and bmi carry 5-10x the conditional signal of any
        # categorical column.
        if c in getattr(spec, "numerical_cols", []) and c in (getattr(spec, "public_bins", {}) or {}):
            edges = np.asarray(spec.public_bins[c], dtype=float)
            v = pd.to_numeric(df[c], errors="coerce")
            idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
            return pd.Series([f"{c}_b{i}" for i in idx], index=df.index)
        return df[c].astype(str).str.strip()

    def key(df):
        parts = ([b.apply(df).astype(str) for b in bands] if bands
                 else [_col_key(df, c) for c in cols])
        out = parts[0]
        for p in parts[1:]:
            out = out + "|" + p
        return out
    rk, sk = key(real), key(synth)
    ry, sy = spec.is_positive(real), spec.is_positive(synth)
    num = den = 0.0
    for g, idx in rk.groupby(rk).groups.items():
        if len(idx) < min_n:
            continue
        w = len(idx) / len(rk)
        p_real = float(ry.loc[idx].mean())
        m = (sk == g)
        num += w * (p_real if m.sum() == 0 else abs(float(sy[m].mean()) - p_real))
        den += w
    return float(num / den) if den else float("nan")


def score(spec, synth, real_train, real_test_enc, enc) -> dict:
    se = encode(spec, synth, enc)
    feat = spec.feature_cols
    y = se[spec.target_col].values.astype(int)
    res = {"n": len(synth), "pos_rate": float(spec.is_positive(synth).mean()),
           "tv_1way": marginal_tv(spec, real_train, synth, 1),
           "tv_2way": marginal_tv(spec, real_train, synth, 2),
           "cond_tv_seen": conditional_tv(spec, real_train, synth, bands=spec.stratify),
           "cond_tv_heldout": float(np.nanmean(
               [conditional_tv(spec, real_train, synth, cols=fam)
                for fam in spec.heldout_groups]))}
    yte = real_test_enc[spec.target_col].values.astype(int)
    Xte = real_test_enc[feat].values
    students = {"LR": make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000, class_weight="balanced")),
                "RF": RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1,
                                             class_weight="balanced"),
                "GBM": HistGradientBoostingClassifier(random_state=RNG)}
    for k, m in students.items():
        if len(np.unique(y)) < 2:
            res[f"tstr_{k}"] = float("nan"); continue
        m.fit(se[feat].values, y)
        res[f"tstr_{k}"] = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--spec-json", required=True, help="JSON: {label: [globs]}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workloads", action="store_true", help="also compute AIM's 3-way workloads")
    a = ap.parse_args()

    spec = get_spec(a.dataset)
    df = spec.loader(); spec.validate(df)
    tr, te = train_test_split(df, test_size=0.2, random_state=a.seed, stratify=df[spec.target_col])
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    enc = _encoders(spec, tr); te_enc = encode(spec, te, enc)

    # A "held-out" family stops being held out if the release conditions on one of its columns —
    # the generator was then TOLD that relationship, and condHeld silently becomes a training
    # measurement rather than a generalisation one. autoconfig chooses conditional columns from the
    # data, so this can change without anyone editing a spec.
    _cond_cols = {b.col for lv in (spec.conditional_levels or []) for b in (lv or [])}
    for _fam in (spec.heldout_groups or []):
        _bad = _cond_cols & set(_fam)
        if _bad:
            print(f"  !! held-out family {_fam} overlaps RELEASED conditional column(s) "
                  f"{sorted(_bad)} — condHeld for it is not a generalisation test")
    if not spec.heldout_groups:
        print("  note: no held-out conditional families declared — condHeld will be nan, which is "
              "an ABSENT measurement, not a good score")

    conditions = json.load(open(a.spec_json))
    sets: dict[str, list[pd.DataFrame]] = {}
    n_ref = None
    for lab, globs in conditions.items():
        paths = sorted({p for g in globs for p in glob.glob(g)})
        if not paths:
            print(f"  !! {lab}: no files matched {globs}"); continue
        sets[lab] = [pd.read_csv(p) for p in paths]
        n_ref = n_ref or len(sets[lab][0])

    # Fidelity metrics are n-SENSITIVE: on diabetes a real sample scores 1-way TV 0.0181 at
    # n=1000, 0.0145 at n=1525 and 0.0128 at n=2000 — a 30%-of-floor swing from sample size alone.
    # The floors below are built at n_ref (the FIRST condition's size), so any condition with a
    # materially different n is being compared against a floor that does not apply to it.
    _sizes = {lab: len(dfs[0]) for lab, dfs in sets.items()}
    if _sizes and (max(_sizes.values()) > 1.1 * min(_sizes.values())):
        print("  !! conditions differ in size — fidelity is NOT comparable across them, and the "
              "floors are built at n_ref only:")
        for lab, n in sorted(_sizes.items(), key=lambda kv: -kv[1]):
            print(f"       {lab:34s} n={n}")

    n_ref = n_ref or 300
    sets["real-sample [FLOOR-n]"] = [tr.sample(min(n_ref, len(tr)), random_state=1000 + i)
                                     for i in range(3)]
    shuf = []
    for i in range(3):
        d = tr.sample(min(n_ref, len(tr)), random_state=3000 + i).copy()
        d[spec.target_col] = np.random.RandomState(i).permutation(d[spec.target_col].values)
        shuf.append(d)
    sets["shuffled-target [FLOOR]"] = shuf
    sets["TRTR [CEILING]"] = [tr.sample(min(4000, len(tr)), random_state=1)]

    results = {}
    for lab, dfs in sets.items():
        results[lab] = [score(spec, d, tr, te_enc, enc) for d in dfs]

    print("\n" + "=" * 116)
    print(f"{spec.name.upper()} — fidelity AND utility  (n={n_ref}/draw; "
          f"real positive rate {spec.is_positive(tr).mean():.1%})")
    print("=" * 116)
    print(f"{'condition':32s} {'draws':>5s} {'1wayTV↓':>9s} {'2wayTV↓':>9s} {'condSeen↓':>10s} "
          f"{'condHeld↓':>10s} {'TSTR-LR↑':>9s} {'TSTR-RF↑':>9s} {'TSTR-GBM↑':>10s}")
    for lab, rows in results.items():
        m = lambda k: float(np.nanmean([r[k] for r in rows]))
        # TRTR is the UTILITY ceiling only. Its fidelity columns compare a sample of the training
        # set against the training set, and on every dataset with <= 4000 training rows the sample
        # IS the whole training set — so the numbers were 0.000 by construction (6 of our 11
        # datasets, NHANES included) and meant something different on the large ones. A
        # tautology must not be printed in the same column as a measurement; the real-sample
        # [FLOOR-n] row is the like-for-like real-data fidelity reference.
        if lab.startswith("TRTR"):
            print(f"{lab:32s} {len(rows):>5d} {'—':>9s} {'—':>9s} {'—':>10s} {'—':>10s} "
                  f"{m('tstr_LR'):>9.3f} {m('tstr_RF'):>9.3f} {m('tstr_GBM'):>10.3f}")
            continue
        print(f"{lab:32s} {len(rows):>5d} {m('tv_1way'):>9.3f} {m('tv_2way'):>9.3f} "
              f"{m('cond_tv_seen'):>10.3f} {m('cond_tv_heldout'):>10.3f} "
              f"{m('tstr_LR'):>9.3f} {m('tstr_RF'):>9.3f} {m('tstr_GBM'):>10.3f}")

    if a.workloads:
        wl, nq = workload_error_report(tr, sets, spec.column_names, spec.numerical_cols,
                                       spec.public_bins, spec.feature_bounds, spec.target_col)
        print(f"\nAIM's 3-way workload error (target={nq['target']}, all3way={nq['all3way']} queries)")
        print(f"{'condition':32s} {'target↓':>10s} {'all-3way↓':>11s} {'skewed↓':>10s}")
        for lab, v in wl.items():
            print(f"{lab:32s} {v['target'][0]:>10.3f} {v['all3way'][0]:>11.3f} {v['skewed'][0]:>10.3f}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2, default=str)
        print(f"\nsaved → {a.out}")


if __name__ == "__main__":
    main()
