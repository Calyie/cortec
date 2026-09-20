"""
classification_report.py — full classification metrics for the two questions a security-and-privacy
reader asks about CoRTeC, computed and reported separately because they are different questions.

**1. The attack's classification performance (the privacy question).** Membership inference is a
binary classifier: given a candidate record, decide member or non-member. Reporting only AUC hides
how it behaves at an operating point, so we report accuracy, precision, recall, F1, the ROC curve
and a confusion matrix. For a privacy result the *desirable* outcome is a classifier that performs
at chance — precision near the base rate, F1 near the chance value, ROC on the diagonal.

Two traps this file is written to avoid. First, a balanced membership game has a 50% base rate, so
"accuracy 0.51" sounds meaningful and is not; every attack metric is therefore reported beside its
chance value. Second, a classifier that predicts one class for everything can post a respectable
F1 while being useless, so the confusion matrix is reported rather than summarised away.

**2. The downstream classifier's performance (the utility question).** A model trained on CoRTeC's
synthetic records and evaluated on real held-out data — the TSTR protocol. Here the reference is
not chance but a model trained on *real* records of the same size, and the question is how much is
lost by substituting synthetic data. The same metric set is reported so the two tables are read
the same way.

Both write JSON for the paper and leave plotting to `paper/make_figures.py`.
"""
from __future__ import annotations
import argparse, glob, json, sys, warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             log_loss, brier_score_loss,
                             roc_auc_score, roc_curve, confusion_matrix,
                             precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def tpr_at_fpr(y_true, score, targets=(0.001, 0.01, 0.1)) -> dict:
    """True-positive rate at fixed low false-positive rates.

    This is the metric membership-inference work should lead with (Carlini et al., 2022).
    Average-case summaries — AUC, accuracy, F1 — can hide an attack that is *confidently* right
    about a small number of individuals, which is the case that actually harms someone. An attack
    that is useless on average but identifies 1% of members at a 0.1% false-positive rate is a real
    attack; AUC would report it as ~0.5.
    """
    y_true = np.asarray(y_true).astype(int)
    fpr, tpr, _ = roc_curve(y_true, np.asarray(score, float))
    out = {}
    for t in targets:
        i = int(np.searchsorted(fpr, t, side="right") - 1)
        out[f"tpr_at_fpr_{t:g}"] = float(tpr[max(i, 0)])
    return out


def metrics_at(y_true, score, threshold=None, *, split_threshold=False, seed=0) -> dict:
    """Full metric set at an operating point, with the chance reference alongside.

    `split_threshold=True` selects the operating point on one half of the candidates and evaluates
    on the other. This matters: choosing a threshold by maximising Youden's J on the *same* data
    being scored is optimistically biased, and when the ROC is a diagonal — as it is for an attack
    that has found nothing — the maximum lands at an arbitrary point. On our first run that
    arbitrary point labelled 61% of all candidates "member", inflating recall and producing an F1
    of 0.566 that read as a signal when J was 0.039, i.e. zero. Selecting the threshold out of
    sample removes that artefact.
    """
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, float)
    y_full, s_full = y_true, score          # threshold-free metrics stay on the full sample
    if threshold == "balanced":
        # Predict "member" for exactly half the candidates. On a diagonal ROC the J-maximising
        # threshold is undefined — any point is as good as any other — so the argmax lands
        # arbitrarily and can label 61% of everything "member", inflating recall and producing an
        # F1 that reads as signal. At the balanced point the predicted-positive rate is 0.5 by
        # construction, so F1 reflects precision, and precision at chance gives F1 at chance.
        # This is a REPORTING choice, not a weakening of the attack: the threshold-free metrics
        # (AUC, average precision, TPR at fixed FPR) are unaffected and are the primary evidence.
        threshold = float(np.median(score))
    elif threshold is None and split_threshold:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y_true))
        half = len(idx) // 2
        sel, ev = idx[:half], idx[half:]
        f, t, thr = roc_curve(y_true[sel], score[sel])
        threshold = float(thr[np.argmax(t - f)])
        # Evaluate THRESHOLD-DEPENDENT metrics on the untouched half. Threshold-FREE metrics
        # (AUC, average precision, log loss, Brier) do not depend on the operating point, so
        # halving their sample only adds variance — keep them on the full set.
        y_eval, s_eval = y_true[ev], score[ev]
        y_true, score = y_eval, s_eval
    elif threshold is None:                       # Youden's J on the same data (optimistic)
        fpr, tpr, thr = roc_curve(y_true, score)
        threshold = float(thr[np.argmax(tpr - fpr)])
    pred = (score >= threshold).astype(int)
    base = float(y_true.mean())
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        # Log loss and Brier score are the calibration half of the picture: a model can rank well
        # (high AUC) while being badly calibrated, and a clinical deployment reads probabilities,
        # not just ranks.
        "log_loss": float(log_loss(y_full, np.clip(s_full, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_full, np.clip(s_full, 0.0, 1.0))),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_full, s_full)),
        "average_precision": float(average_precision_score(y_full, s_full)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "predicted_positive_rate": float(pred.mean()),
        "youden_j": float(recall_score(y_true, pred, zero_division=0)
                          - (fp / max(fp + tn, 1))),
        "base_rate": base,
        # what each metric would be for a coin flip at this base rate, so a reader can see
        # immediately whether a number is meaningful or an artefact of the class balance
        "chance": {"accuracy": max(base, 1 - base), "precision": base,
                   "recall": 0.5, "f1": 2 * base * 0.5 / (base + 0.5) if base > 0 else 0.0,
                   "auc": 0.5},
    }


def roc_points(y_true, score, n=200):
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(score, float))
    idx = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    return {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}


def encode(frames, num, cat):
    allf = pd.concat(frames, ignore_index=True)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(allf[cat].astype(str))
    sc = StandardScaler().fit(allf[num].apply(pd.to_numeric, errors="coerce").fillna(0))
    return [np.hstack([sc.transform(f[num].apply(pd.to_numeric, errors="coerce").fillna(0)),
                       enc.transform(f[cat].astype(str))]) for f in frames]


# ── 1. the attack ───────────────────────────────────────────────────────────────────

def attack_scores(syn, mem, non, num, cat, seed=0):
    """Shadow-model attack — the strongest of the three in §4.4, so the one worth profiling."""
    syn_X, mem_X, non_X = encode([syn, mem, non], num, cat)
    rng = np.random.default_rng(seed)
    ref = np.vstack([mem_X, non_X])
    idx = rng.choice(len(ref), min(len(syn_X), len(ref)), replace=False)
    X = np.vstack([syn_X, ref[idx]])
    y = np.r_[np.ones(len(syn_X)), np.zeros(len(idx))]
    m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(X, y)
    s = np.r_[m.predict_proba(mem_X)[:, 1], m.predict_proba(non_X)[:, 1]]
    return np.r_[np.ones(len(mem_X)), np.zeros(len(non_X))], s


# ── 2. the downstream classifier ────────────────────────────────────────────────────

def tstr_scores(train_df, test_df, num, cat, target, positive, seed=0):
    """Train on `train_df`, score real held-out `test_df`. Returns (y_true, score)."""
    tr_X, te_X = encode([train_df, test_df], num, cat)
    ytr = (train_df[target].astype(str).str.strip() == positive).astype(int).values
    yte = (test_df[target].astype(str).str.strip() == positive).astype(int).values
    if len(np.unique(ytr)) < 2:
        return yte, np.full(len(yte), 0.5)
    m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(tr_X, ytr)
    return yte, m.predict_proba(te_X)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortec", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="label=glob pairs, e.g. MST=results/.../mst_*.csv")
    ap.add_argument("--n-candidates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--out", default="results/classification_report.json")
    a = ap.parse_args()

    if a.dataset == "adult":
        from src.data_loader import (load_adult, NUMERICAL_COLS as num,
                                     CATEGORICAL_COLS as cat, TARGET_COL as target)
        d = load_adult()
        train, test = d[0], d[1]
        positive, negative = ">50K", "<=50K"
    else:
        from src.dataset_spec import get_spec
        import src.datasets_extra      # noqa: F401
        import src.datasets_regulated  # noqa: F401
        import src.datasets_clinical   # noqa: F401
        import src.datasets_synthetic  # noqa: F401
        import src.datasets_auto       # noqa: F401  (auto-configured *_auto variants)
        import src.datasets_extra       # noqa: F401
        import src.datasets_regulated   # noqa: F401
        import src.datasets_clinical    # noqa: F401  (nhanes, mimic3_demo)
        import src.datasets_synthetic   # noqa: F401  (renal_registry)
        sp = get_spec(a.dataset)
        df = sp.loader(); sp.validate(df)
        train, test = train_test_split(df, test_size=0.2, random_state=a.seed,
                                       stratify=df[sp.target_col])
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
        num, cat, target = sp.numerical_cols, sp.categorical_cols, sp.target_col
        positive, negative = sp.positive_class, sp.negative_class
    cols = num + cat + [target]

    syn = pd.concat([pd.read_csv(f, keep_default_na=False)
                     for g in a.cortec for f in sorted(glob.glob(g))], ignore_index=True)
    syn = syn[[c for c in cols if c in syn.columns]]
    use = [c for c in cols if c in syn.columns and c in train.columns]
    numu = [c for c in num if c in use]
    catu = [c for c in use if c not in numu and c != target]

    out = {"n_synthetic": len(syn)}
    W = 96

    # ── attack ──────────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(a.seed)
    k = min(a.n_candidates, len(train), len(test))
    mem = train.iloc[rng.choice(len(train), k, replace=False)][use].reset_index(drop=True)
    non = test.iloc[rng.choice(len(test), k, replace=False)][use].reset_index(drop=True)

    y, s = attack_scores(syn, mem, non, numu, catu, seed=a.seed)
    atk = metrics_at(y, s, threshold="balanced")
    atk["roc"] = roc_points(y, s)
    atk.update(tpr_at_fpr(y, s))
    # kept for comparison so the artefact is documented rather than quietly dropped
    atk["youden_same_data"] = metrics_at(y, s)
    atk["youden_heldout"] = metrics_at(y, s, split_threshold=True, seed=a.seed)
    out["attack_on_cortec"] = atk

    # positive control: real records passed off as synthetic
    leak = pd.concat([train.iloc[rng.choice(len(train), min(len(syn), len(train)),
                                            replace=False)][use],
                      mem.iloc[:200]], ignore_index=True)
    yc, sc = attack_scores(leak, mem, non, numu, catu, seed=a.seed)
    ctrl = metrics_at(yc, sc, threshold="balanced")
    ctrl["roc"] = roc_points(yc, sc)
    ctrl.update(tpr_at_fpr(yc, sc))
    ctrl["youden_same_data"] = metrics_at(yc, sc)
    ctrl["youden_heldout"] = metrics_at(yc, sc, split_threshold=True, seed=a.seed)
    out["attack_on_control"] = ctrl

    print("=" * W)
    print("1. MEMBERSHIP-INFERENCE ATTACK AS A CLASSIFIER  (chance is the DESIRABLE outcome)")
    print("   operating point: BALANCED (predict 'member' for half) — on a diagonal ROC the")
    print("   J-maximising threshold is arbitrary; threshold-free metrics are below and unaffected")
    print("=" * W)
    hdr = f"{'target':28s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'AUC':>7s} {'AP':>7s}"
    print(hdr)
    for lab, m in (("CoRTeC output", atk), ("positive control (leaking)", ctrl)):
        print(f"{lab:28s} {m['accuracy']:7.3f} {m['precision']:7.3f} {m['recall']:7.3f} "
              f"{m['f1']:7.3f} {m['auc']:7.3f} {m['average_precision']:7.3f}")
    ch = atk["chance"]
    print(f"{'chance at this base rate':28s} {ch['accuracy']:7.3f} {ch['precision']:7.3f} "
          f"{ch['recall']:7.3f} {ch['f1']:7.3f} {ch['auc']:7.3f} {atk['base_rate']:7.3f}")
    print(f"\n{'':28s} {'TPR@FPR=0.1%':>13s} {'TPR@FPR=1%':>11s} {'TPR@FPR=10%':>12s}")
    for lab, m in (("CoRTeC output", atk), ("positive control (leaking)", ctrl)):
        print(f"{lab:28s} {m['tpr_at_fpr_0.001']:13.4f} {m['tpr_at_fpr_0.01']:11.4f} "
              f"{m['tpr_at_fpr_0.1']:12.4f}")
    print(f"{'chance':28s} {0.001:13.4f} {0.01:11.4f} {0.1:12.4f}")
    print("(low-FPR TPR is the metric MIA work should lead with — average-case scores hide an "
          "attack\n that is confidently right about a few individuals)")
    print(f"\npredicted-positive rate: CoRTeC {atk['predicted_positive_rate']:.3f}, "
          f"control {ctrl['predicted_positive_rate']:.3f}  (0.5 = balanced)")
    print(f"Youden J: CoRTeC {atk['youden_j']:+.4f}, control {ctrl['youden_j']:+.4f}  "
          f"(0 = no separation)")
    c = atk["confusion"]
    print(f"\nconfusion matrix, attack on CoRTeC (rows: true non-member/member):")
    print(f"    predicted:   non-member   member")
    print(f"    non-member   {c['tn']:10d} {c['fp']:8d}")
    print(f"    member       {c['fn']:10d} {c['tp']:8d}")

    # ── downstream ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("2. DOWNSTREAM CLASSIFIER TRAINED ON SYNTHETIC DATA, TESTED ON REAL HELD-OUT DATA")
    print("=" * W)
    conds = {"CoRTeC": syn}
    for pair in a.baselines:
        if "=" not in pair:
            continue
        lab, g = pair.split("=", 1)
        fs = sorted(glob.glob(g))
        if fs:
            b = pd.concat([pd.read_csv(f, keep_default_na=False) for f in fs], ignore_index=True)
            if set(use).issubset(b.columns):
                conds[lab] = b[use]
    rs = train.iloc[rng.choice(len(train), min(len(syn), len(train)), replace=False)][use]
    conds["real sample (same n)"] = rs.reset_index(drop=True)
    conds["real, full training set"] = train[use]

    print(hdr)
    down = {}
    for lab, tr_df in conds.items():
        yt, st = tstr_scores(tr_df, test[use], numu, catu, target, positive, seed=a.seed)
        # `metrics_at(y, s)` with no threshold picks the operating point by maximising Youden's J
        # on the SAME data it then scores — the file's own docstring calls that optimistic and the
        # ATTACK metrics above already guard against it with split_threshold. The downstream
        # clinical metrics did not, so accuracy/precision/recall/F1 were reported at a threshold
        # fitted to the test set. Same guard, same file, now on both sides.
        m = metrics_at(yt, st, split_threshold=True, seed=a.seed)
        m["roc"] = roc_points(yt, st)
        down[lab] = m
        print(f"{lab:28s} {m['accuracy']:7.3f} {m['precision']:7.3f} {m['recall']:7.3f} "
              f"{m['f1']:7.3f} {m['auc']:7.3f} {m['average_precision']:7.3f}")
    out["downstream"] = down
    b = down["CoRTeC"]["base_rate"]
    print(f"{'chance (base rate ' + f'{b:.3f})':28s} {max(b,1-b):7.3f} {b:7.3f} "
          f"{0.5:7.3f} {2*b*0.5/(b+0.5):7.3f} {0.5:7.3f} {b:7.3f}")
    c = down["CoRTeC"]["confusion"]
    # The class names are per-DATASET. These were hard-coded to Adult's income labels, so an
    # NHANES report printed "true <=50K / >50K" for a diabetes outcome — wrong in a way that would
    # have gone straight into the paper's clinical table.
    _neg, _pos = str(negative), str(positive)
    _w = max(len(_neg), len(_pos), 7)
    print(f"\nconfusion matrix, model trained on CoRTeC "
          f"(rows: true {_neg} / {_pos}):")
    print(f"    predicted:  {_neg:>{_w}} {_pos:>{_w}}")
    print(f"    {_neg:<{_w}}    {c['tn']:9d} {c['fp']:8d}")
    print(f"    {_pos:<{_w}}    {c['fn']:9d} {c['tp']:8d}")

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
