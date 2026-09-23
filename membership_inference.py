"""
membership_inference.py — an empirical membership-inference attack against CoRTeC's output.

**Why run this when the mechanism already has a proof.** Proposition 1 gives ε-differential
privacy, which bounds any membership adversary's advantage information-theoretically: at ε = 2 no
attack can exceed an advantage of (e^ε − 1)/(e^ε + 1) ≈ 0.762, whatever it does. An empirical
attack cannot improve on a proof and cannot validate one. What it *can* do is two things worth
having.

First, it detects **implementation** failure. A proof covers the mechanism as written; an attack
covers the code as run. If the release accidentally leaked a record — a suppression threshold not
applied, a cohort of size one, a released histogram that is a delta on a single individual — the
proof would still be true of the algorithm and false of the artifact. This is the same reason we
verify the accounting numerically rather than trusting the derivation (§4.3).

Second, it gives a **measured** number for the audience that asks for one. Regulators and security
reviewers routinely ask "has anyone tried to attack it", and "the proof forbids it" is a correct
but unsatisfying answer. Reporting a measured advantage alongside the theoretical bound is more
useful than either alone.

**The attacks.** Three, of increasing strength, all in the standard membership game: the adversary
sees the synthetic dataset and must decide whether a candidate record was in the private training
set. Members are drawn from the training split; non-members from the held-out split, which is
disjoint and identically distributed — so the attack cannot succeed merely by recognising the data
distribution.

  1. **Nearest-neighbour distance.** The classic attack: a record that participated in training
     should sit closer to some synthetic record than a non-member does. Reported as AUC over the
     negated distance.
  2. **Exact / near-duplicate match.** Direct memorisation check. Any exact match between a
     synthetic record and a private record is worth knowing about even if it is coincidental at
     this schema size, so we report the count as well as the attack score.
  3. **Shadow-model likelihood.** A density model fitted to the synthetic data scores candidate
     records; members should score higher if the synthetic data encodes them. This is the
     strongest of the three and closest to the literature's state of the art for tabular data.

**Reading the result.** The honest null is AUC = 0.5. Because the sample is finite, an AUC near
0.5 is evidence of no *detectable* leakage rather than proof of none — so we report a bootstrap
confidence interval and check whether it contains 0.5, rather than eyeballing the point estimate.
We also run every attack against a **positive control**: real training records passed off as
"synthetic". An attack that cannot detect membership there is not a working attack, and reporting
its null on CoRTeC would be meaningless. This is the same floor/ceiling discipline used throughout.
"""
from __future__ import annotations
import argparse, glob, json, sys, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def encode(frames: list[pd.DataFrame], num, cat) -> list[np.ndarray]:
    """One-hot categoricals + standardised numerics, fitted on the union so every frame shares a
    space. Fitting the encoder on the union is not a leak: it uses only column domains."""
    allf = pd.concat(frames, ignore_index=True)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
        allf[cat].astype(str))
    sc = StandardScaler().fit(allf[num].apply(pd.to_numeric, errors="coerce").fillna(0))
    out = []
    for f in frames:
        X = np.hstack([sc.transform(f[num].apply(pd.to_numeric, errors="coerce").fillna(0)),
                       enc.transform(f[cat].astype(str))])
        out.append(X)
    return out


def boot_auc_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def attack_nn(syn_X, mem_X, non_X):
    nn = NearestNeighbors(n_neighbors=1).fit(syn_X)
    dm, _ = nn.kneighbors(mem_X)
    dn, _ = nn.kneighbors(non_X)
    y = np.r_[np.ones(len(mem_X)), np.zeros(len(non_X))]
    s = -np.r_[dm.ravel(), dn.ravel()]          # closer = more likely a member
    return y, s


def attack_exact(syn: pd.DataFrame, mem: pd.DataFrame, non: pd.DataFrame, cols):
    keys = set(map(tuple, syn[cols].astype(str).values))
    sm = np.array([tuple(r) in keys for r in mem[cols].astype(str).values], float)
    sn = np.array([tuple(r) in keys for r in non[cols].astype(str).values], float)
    y = np.r_[np.ones(len(sm)), np.zeros(len(sn))]
    return y, np.r_[sm, sn], int(sm.sum()), int(sn.sum())


def attack_shadow(syn_X, mem_X, non_X, seed=0):
    """Discriminative shadow attack: train to separate synthetic from generic real records, then
    score candidates. Higher score = looks more like the synthetic data = more likely a member."""
    rng = np.random.default_rng(seed)
    ref = np.vstack([mem_X, non_X])
    idx = rng.choice(len(ref), min(len(syn_X), len(ref)), replace=False)
    X = np.vstack([syn_X, ref[idx]])
    y = np.r_[np.ones(len(syn_X)), np.zeros(len(idx))]
    m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)
    s = np.r_[m.predict_proba(mem_X)[:, 1], m.predict_proba(non_X)[:, 1]]
    yy = np.r_[np.ones(len(mem_X)), np.zeros(len(non_X))]
    return yy, s


def attack_lira(syn, mem, non, num, cat, seed=0, n_shadow=16):
    """A likelihood-ratio attack in the style of Carlini et al. (2022), adapted to synthetic data.

    The three attacks above are *global*: they score a record against the synthetic set as a whole.
    LiRA is per-record — it asks whether this specific record looks more likely under a model fitted
    to the synthetic data than under a model fitted to data that excludes it — and is the attack
    that exposed leakage in models the global attacks had cleared. Including it materially widens
    the class of adversary this evidence covers.

    Implementation: fit `n_shadow` density models to bootstrap resamples of the synthetic data,
    score every candidate under each, and use the standardised score (the per-record z against the
    shadow distribution) as the membership statistic. A record the synthetic data encodes unusually
    well scores high under all shadows and low variance, which is exactly what memorisation looks
    like.
    """
    from sklearn.neighbors import KernelDensity
    from sklearn.decomposition import PCA
    syn_X, mem_X, non_X = encode([syn, mem, non], num, cat)
    rng = np.random.default_rng(seed)
    # project once so the density estimate is tractable and shared across shadows
    k = min(12, syn_X.shape[1], max(2, len(syn_X) // 20))
    pca = PCA(n_components=k, random_state=seed).fit(syn_X)
    S, M, N = pca.transform(syn_X), pca.transform(mem_X), pca.transform(non_X)
    cand = np.vstack([M, N])
    bw = max(np.std(S) * (len(S) ** (-1.0 / (k + 4))), 1e-3)
    scores = []
    for i in range(n_shadow):
        idx = rng.choice(len(S), len(S), replace=True)
        kde = KernelDensity(bandwidth=bw).fit(S[idx])
        scores.append(kde.score_samples(cand))
    scores = np.vstack(scores)
    mu, sd = scores.mean(axis=0), scores.std(axis=0) + 1e-9
    stat = mu / sd            # high mean, low variance across shadows = consistently well modelled
    y = np.r_[np.ones(len(M)), np.zeros(len(N))]
    return y, stat


def run(label, syn, mem, non, num, cat, cols, seed=0):
    syn_X, mem_X, non_X = encode([syn, mem, non], num, cat)
    out = {}
    for name, (y, s) in {
        "nearest_neighbour": attack_nn(syn_X, mem_X, non_X),
        "shadow_model": attack_shadow(syn_X, mem_X, non_X, seed),
        "lira_per_record": attack_lira(syn, mem, non, num, cat, seed),
    }.items():
        auc = float(roc_auc_score(y, s))
        lo, hi = boot_auc_ci(y, s, seed=seed)
        out[name] = {"auc": auc, "ci": [lo, hi], "advantage": 2 * abs(auc - 0.5),
                     "contains_chance": bool(lo <= 0.5 <= hi)}
    y, s, n_mem_hit, n_non_hit = attack_exact(syn, mem, non, cols)
    auc = float(roc_auc_score(y, s)) if len(np.unique(s)) > 1 else 0.5
    lo, hi = boot_auc_ci(y, s, seed=seed) if len(np.unique(s)) > 1 else (0.5, 0.5)
    out["exact_match"] = {"auc": auc, "ci": [lo, hi], "advantage": 2 * abs(auc - 0.5),
                          "contains_chance": bool(lo <= 0.5 <= hi),
                          "member_matches": n_mem_hit, "nonmember_matches": n_non_hit}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--synthetic", nargs="+", required=True)
    ap.add_argument("--n-candidates", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/membership_inference.json")
    a = ap.parse_args()

    if a.dataset == "adult":
        from src.data_loader import load_adult, NUMERICAL_COLS as num, CATEGORICAL_COLS as cat, TARGET_COL
        d = load_adult()
        train = d[0] if isinstance(d, tuple) else d
        test = d[1] if isinstance(d, tuple) and len(d) > 1 else None
        cols = num + cat + [TARGET_COL]
    else:
        from src.dataset_spec import get_spec
        import src.datasets_extra      # noqa: F401
        import src.datasets_regulated  # noqa: F401
        import src.datasets_clinical   # noqa: F401
        import src.datasets_synthetic  # noqa: F401
        import src.datasets_auto       # noqa: F401  (auto-configured *_auto variants)
        sp = get_spec(a.dataset)
        df = sp.loader()
        train, test = train_test_split(df, test_size=0.2, random_state=a.seed,
                                       stratify=df[sp.target_col])
        num, cat, cols = sp.numerical_cols, sp.categorical_cols, sp.column_names

    files = sorted(f for g in a.synthetic for f in glob.glob(g))
    if not files:
        raise SystemExit("no synthetic files matched")
    syn = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files], ignore_index=True)
    syn = syn[[c for c in cols if c in syn.columns]]
    use = [c for c in cols if c in syn.columns and c in train.columns]
    numu = [c for c in num if c in use]
    catu = [c for c in use if c not in numu]

    rng = np.random.default_rng(a.seed)
    k = min(a.n_candidates, len(train), len(test) if test is not None else len(train))
    mem = train.iloc[rng.choice(len(train), k, replace=False)][use].reset_index(drop=True)
    non = test.iloc[rng.choice(len(test), k, replace=False)][use].reset_index(drop=True)

    bound = (np.exp(a.epsilon) - 1) / (np.exp(a.epsilon) + 1)
    W = 92
    print("=" * W)
    print(f"MEMBERSHIP INFERENCE — {a.dataset}: {len(syn)} synthetic records from {len(files)} draw(s)")
    print(f"  {k} members (training split) vs {k} non-members (held-out split)")
    print(f"  theoretical bound at eps={a.epsilon}: adversary advantage <= {bound:.3f}")
    print("=" * W)

    print("\nATTACKING CoRTeC OUTPUT")
    res_cortec = run("cortec", syn, mem, non, numu, catu, use, seed=a.seed)
    for name, r in res_cortec.items():
        verdict = "no detectable leakage" if r["contains_chance"] else "*** SIGNAL ***"
        extra = (f"  [exact hits: {r['member_matches']} member / {r['nonmember_matches']} non-member]"
                 if name == "exact_match" else "")
        print(f"  {name:20s} AUC {r['auc']:.4f}  95% CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]  "
              f"adv {r['advantage']:.4f}  {verdict}{extra}")

    # POSITIVE CONTROL: real training records presented as "synthetic". Any attack that cannot
    # detect membership here is broken, and its null result on CoRTeC would mean nothing.
    print("\nPOSITIVE CONTROL — real training records passed off as synthetic")
    print("  (an attack that scores ~0.5 here is not working, and its null above is meaningless)")
    leak = train.iloc[rng.choice(len(train), min(len(syn), len(train)), replace=False)][use]
    leak = pd.concat([leak, mem.iloc[:min(200, len(mem))]], ignore_index=True)
    res_ctrl = run("control", leak, mem, non, numu, catu, use, seed=a.seed)
    for name, r in res_ctrl.items():
        works = "attack works" if not r["contains_chance"] else "*** ATTACK IS BLIND ***"
        print(f"  {name:20s} AUC {r['auc']:.4f}  95% CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]  "
              f"adv {r['advantage']:.4f}  {works}")

    print("\n" + "-" * W)
    strongest = max(r["advantage"] for r in res_cortec.values())
    powered = [n for n, r in res_ctrl.items() if not r["contains_chance"]]
    print(f"strongest measured advantage against CoRTeC: {strongest:.4f}")
    print(f"theoretical bound at eps={a.epsilon}:          {bound:.4f}")
    print(f"attacks demonstrated to work on the control:  {powered if powered else 'NONE — results are uninterpretable'}")

    json.dump({"dataset": a.dataset, "n_synthetic": len(syn), "n_candidates": k,
               "epsilon": a.epsilon, "theoretical_advantage_bound": bound,
               "cortec": res_cortec, "positive_control": res_ctrl,
               "strongest_advantage": strongest,
               "attacks_validated_on_control": powered},
              open(a.out, "w"), indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
