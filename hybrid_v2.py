"""
hybrid_v2.py — AIM for Goal 1 (marginals), CoRTeC's DP conditional structure for Goal 2.

This supersedes the first hybrid script, whose two strategies both failed (technical report,
section 8.2): `hard` destroyed the income signal carried by every feature outside
education/hours (in-sample AUC 0.676 -> 0.527), and `soft` bought a small AUC gain by halving
the positive rate. Crucially, the NON-PRIVATE oracle of the same rule did no better, so the
limit was the rule, not the DP noise.

Two changes, each aimed at one of those failures.

1. A RICHER conditional table, which is nearly free.
   The cells of a conditional table are a DISJOINT partition of the private data, so under
   parallel composition every cell receives the FULL conditional budget — the table costs one
   query's worth of epsilon no matter how many cells it has. The Laplace scale on a cell's
   rate is 1/(n_cell * eps_cond), so a cell holding 150 people is accurate to about 2 points
   at eps_cond = 0.33. Richness is therefore limited by cell SUPPORT, not by privacy budget.
   Cells with too little support back off to their parent (coarser) cell rather than being
   released noisily — backoff is a deterministic function of released quantities.

2. RANK-PRESERVING rate matching instead of i.i.d. resampling.
   Sampling each row's label independently from its cell rate throws away AIM's own ordering
   of who is likely to earn more. Instead, within each cell we keep AIM's ranking and flip only
   as many rows as needed to hit the DP-released rate: score every row by a propensity model
   fitted to AIM's own output (post-processing of an already-DP artifact, no privacy cost),
   then label the top k of the cell positive, where k = round(dp_rate * n_cell).
   This matches CoRTeC's conditional rates EXACTLY while preserving AIM's within-cell signal.

Everything here is post-processing of two already-DP artifacts, so the hybrid's privacy cost
is exactly that of the AIM run plus the CoRTeC release — zero additional budget.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, ".")
from src.data_loader import NUMERICAL_COLS, TARGET_COL

try:
    from diffprivlib.mechanisms import Laplace
except ImportError:
    from src.dp_primitives import Laplace


# ── cell definitions, from coarse to fine ────────────────────────────────────────────
def edu_band(df):
    v = pd.to_numeric(df["education_num"], errors="coerce").fillna(9)
    return pd.cut(v, [0, 9, 12, 13, 16], labels=["<=HS", "SomeCol", "Bach", "Grad"]).astype(str)


def hrs_band(df):
    v = pd.to_numeric(df["hours_per_week"], errors="coerce").fillna(40)
    return pd.cut(v, [0, 39, 40, 99], labels=["<40", "40", ">40"]).astype(str)


def age_band(df):
    v = pd.to_numeric(df["age"], errors="coerce").fillna(38)
    return pd.cut(v, [16, 30, 40, 50, 90], labels=["17-30", "31-40", "41-50", "51+"]).astype(str)


def marital(df):
    s = df["marital_status"].astype(str).str.strip()
    return np.where(s == "Married-civ-spouse", "Married", "NotMarried")


def sex(df):
    return df["sex"].astype(str).str.strip()


def occ(df):
    s = df["occupation"].astype(str).str.strip()
    high = {"Exec-managerial", "Prof-specialty", "Tech-support", "Protective-serv", "Sales"}
    return np.where(s.isin(high), "occHigh", "occOther")


# Nested: each level adds one factor to the previous level's key, so a fine cell always has a
# well-defined parent to back off to.
LEVELS = [
    ("L0_global", []),
    ("L1_edu", [edu_band]),
    ("L2_edu_hrs", [edu_band, hrs_band]),
    ("L3_edu_hrs_marital", [edu_band, hrs_band, marital]),
    ("L4_edu_hrs_marital_age", [edu_band, hrs_band, marital, age_band]),
    ("L5_edu_hrs_marital_age_sex", [edu_band, hrs_band, marital, age_band, sex]),
    ("L6_plus_occupation", [edu_band, hrs_band, marital, age_band, sex, occ]),
]


def cell_keys(df: pd.DataFrame, factors) -> pd.Series:
    if not factors:
        return pd.Series(["*"] * len(df), index=df.index)
    parts = [pd.Series(np.asarray(f(df)), index=df.index).astype(str) for f in factors]
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def build_table(private: pd.DataFrame, level_idx: int, *, epsilon: float | None,
                n_min: int = 150, seed: int = 0, backoff_levels: tuple[int, ...] | None = None) -> dict:
    """Release P(>50K | cell) at the finest level, plus coarse fallback levels for thin cells.

    PRIVACY ACCOUNTING — read before changing this.
    Within ONE level the cells are a disjoint partition, so they fall under parallel
    composition: every cell gets the FULL per-level epsilon, however many cells there are.
    That is what makes a rich table cheap.
    ACROSS levels there is no such luck — L1 and L6 describe the SAME people, so separate
    levels compose SEQUENTIALLY. An earlier version of this function released every level
    0..level_idx while giving each the full epsilon, which under-charged the budget by a
    factor of (level_idx + 1). It now releases only the levels in `backoff_levels` plus the
    finest one, and splits `epsilon` equally among them.

    epsilon=None builds the NON-PRIVATE oracle table (diagnostic only — it is not a method).
    """
    y = (private[TARGET_COL].astype(str).str.strip() == ">50K")

    # Default: the finest level, plus the global rate as the only fallback (2 levels).
    levels_to_release = sorted(set((backoff_levels or (0,)) + (level_idx,)))
    eps_per_level = None if epsilon is None else epsilon / len(levels_to_release)

    tables = []          # coarse -> fine, so lookup can walk back up
    for li in levels_to_release:
        _, factors = LEVELS[li]
        keys = cell_keys(private, factors)
        tbl = {}
        for cell, idx in keys.groupby(keys).groups.items():
            n = len(idx)
            if n < n_min:
                continue
            rate = float(y.loc[idx].mean())
            if eps_per_level is None:
                tbl[cell] = (rate, n)
            else:
                # Sensitivity of a mean over n people is 1/n. This is the mechanism the reported
                # section 8.2 table was produced with (the report's regeneration script for that
                # table calls this function). Its scale depends on the private cell size, so under
                # add/remove-one adjacency it is not pure eps-DP; section 4.3 of the technical
                # report computes the (eps, delta) guarantee it carries (delta = 1.5e-7 at
                # n_min = 150). The shipped cortec-hybrid package releases the positive count and
                # the noised support as two sensitivity-1 queries instead
                # (cortec_hybrid.core.release_conditional_table).
                noisy = Laplace(epsilon=eps_per_level, sensitivity=1.0 / n).randomise(rate)
                tbl[cell] = (float(np.clip(noisy, 0.0, 1.0)), n)
        tables.append((li, tbl))
    return {"level": level_idx, "tables": tables, "n_min": n_min,
            "levels_released": levels_to_release, "eps_per_level": eps_per_level,
            "eps_total": epsilon}


def lookup(table: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (rate, level_used) per row, backing off from the finest level that has the cell."""
    n = len(df)
    rate = np.full(n, np.nan)
    used = np.full(n, -1)
    for li, tbl in reversed(table["tables"]):          # finest first
        if not tbl:
            continue
        _, factors = LEVELS[li]
        keys = cell_keys(df, factors).values
        need = np.isnan(rate)
        if not need.any():
            break
        for i in np.where(need)[0]:
            hit = tbl.get(keys[i])
            if hit is not None:
                rate[i], used[i] = hit[0], li
    return rate, used


# ── relabeling strategies ────────────────────────────────────────────────────────────
def _propensity(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Rank rows by how likely AIM's OWN output says they are to earn >50K.

    Fitted on the synthetic data itself, so this is post-processing of an already-DP artifact
    and costs no privacy budget. It exists only to order rows within a cell.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from src.data_loader import CATEGORICAL_COLS as CC
    X = df.copy()
    for c in CC:
        X[c] = X[c].astype(str).str.strip().astype("category").cat.codes
    for c in NUMERICAL_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    feats = NUMERICAL_COLS + CC
    y = (df[TARGET_COL].astype(str).str.strip() == ">50K").astype(int).values
    if len(np.unique(y)) < 2:
        return np.random.RandomState(seed).random_sample(len(df))
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    try:
        p = cross_val_predict(clf, X[feats].values, y, cv=5, method="predict_proba")[:, 1]
    except Exception:
        clf.fit(X[feats].values, y)
        p = clf.predict_proba(X[feats].values)[:, 1]
    # tiny jitter breaks ties deterministically
    return p + np.random.RandomState(seed).random_sample(len(df)) * 1e-6


def relabel(df: pd.DataFrame, table: dict, strategy: str, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    rate, _ = lookup(table, df)
    rng = np.random.RandomState(seed)

    if strategy == "iid":
        # the old `hard` rule: sample every label independently from its cell rate
        lab = np.where(rng.random_sample(len(df)) < np.nan_to_num(rate, nan=0.5), ">50K", "<=50K")
        keep = np.isnan(rate)
        lab[keep] = df[TARGET_COL].astype(str).str.strip().values[keep]
        out[TARGET_COL] = lab
        return out

    if strategy == "shipped":
        # What cortec-hybrid actually ships (`relabel(..., preserve_ranking=False)`): an EXACT
        # per-cell count with stochastic rounding, assigned over a random permutation of the cell.
        # Neither `iid` (per-row Bernoulli, the old rule) nor `rank` (exact count, but ordered by a
        # propensity score) is this, and the paper's hybrid table must report the shipped path.
        lab = df[TARGET_COL].astype(str).str.strip().values.copy()
        key = pd.Series(np.round(np.nan_to_num(rate, nan=-1.0), 6), index=df.index)
        for r, idx in key.groupby(key).groups.items():
            pos = np.array([df.index.get_loc(i) for i in idx])
            if r < 0:
                continue                       # no table entry: leave the base labels alone
            # stochastic rounding: plain round() biases every rate below 0.5 toward zero
            exact = float(r) * len(pos)
            k = int(np.floor(exact)) + (1 if rng.random_sample() < (exact - np.floor(exact)) else 0)
            order = pos[rng.permutation(len(pos))]
            lab[order[:k]] = ">50K"
            lab[order[k:]] = "<=50K"
        out[TARGET_COL] = lab
        return out

    if strategy == "rank":
        prop = _propensity(df, seed=seed)
        lab = df[TARGET_COL].astype(str).str.strip().values.copy()
        # group rows by the exact rate they were assigned (i.e. by cell)
        key = pd.Series(np.round(np.nan_to_num(rate, nan=-1.0), 6), index=df.index)
        for r, idx in key.groupby(key).groups.items():
            pos = np.array([df.index.get_loc(i) for i in idx])
            if r < 0:
                continue                       # no table entry: leave AIM's own labels alone
            k = int(round(r * len(pos)))
            order = pos[np.argsort(-prop[pos])]
            lab[order[:k]] = ">50K"
            lab[order[k:]] = "<=50K"
        out[TARGET_COL] = lab
        return out

    raise ValueError(f"unknown strategy {strategy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", nargs="+", required=True,
                    help="synthetic CSV(s) to relabel (e.g. AIM draws)")
    ap.add_argument("--private-source", default="real",
                    help="'real' = derive the conditional table from the real private data")
    ap.add_argument("--level", type=int, default=2, help="index into LEVELS")
    ap.add_argument("--epsilon", type=float, default=0.333,
                    help="conditional-table budget; cells are disjoint so each gets all of it")
    ap.add_argument("--oracle", action="store_true", help="NON-PRIVATE table (diagnostic only)")
    ap.add_argument("--strategy", default="shipped", choices=["shipped", "rank", "iid"])
    ap.add_argument("--n-min", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    from src.data_loader import load_adult
    raw_train, *_ = load_adult()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)

    table = build_table(raw_train, a.level, epsilon=None if a.oracle else a.epsilon,
                        n_min=a.n_min, seed=a.seed)
    lvl_name = LEVELS[a.level][0]
    n_cells = len(table["tables"][-1][1])
    print(f"level={lvl_name} cells_released={n_cells} "
          f"({'ORACLE (not DP)' if a.oracle else f'DP eps={a.epsilon}'}) n_min={a.n_min}")

    for p in a.base_csv:
        df = pd.read_csv(p)
        newdf = relabel(df, table, a.strategy, seed=a.seed)
        stem = Path(p).stem
        outp = Path(a.outdir) / f"{stem}__{lvl_name}_{a.strategy}" \
                                f"{'_oracle' if a.oracle else ''}.csv"
        newdf.to_csv(outp, index=False)
        before = (df[TARGET_COL].astype(str).str.strip() == ">50K").mean()
        after = (newdf[TARGET_COL].astype(str).str.strip() == ">50K").mean()
        print(f"  {stem}: >50K {before:.3f} -> {after:.3f}  ->  {outp}")


if __name__ == "__main__":
    main()
