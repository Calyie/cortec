"""
run_baselines_v2.py — regenerate the DP baselines (MST / AIM / PATE-CTGAN) under better
discretization, with multiple draws per method, and with the control that isolates how much
of their measured weakness is OUR preprocessing rather than the method.

Why this exists. `run_baselines.py` bins every numerical column into 12 UNIFORM bins over the
public range and reconstructs values at the bin midpoint. For UCI Adult that is brutal:
`capital_gain` spans [0, 99999] and is zero for ~92% of people, so a uniform 12-bin grid puts
almost everyone in bin 0 and reconstructs a strong income predictor as a constant. If AIM's
TSTR-AUC of 0.694 is mostly that, then "CoRTeC beats AIM on utility" is a statement about our
binning, not about either method — the boring explanation has to be ruled out before the
comparison means anything (project rule 4).

Three things are added:
  1. `--binning public`  — the domain-informed, non-uniform edges in data_loader.PUBLIC_BINS
     (a bin boundary at capital_gain=1 so "zero" survives), instead of 12 uniform bins.
  2. `--reconstruct uniform` — reconstruct a binned value by sampling uniformly inside its bin
     rather than collapsing every row to the midpoint.
  3. `real-roundtrip` — the decisive control. Real training data, discretized and reconstructed
     with NO DP and NO synthesis at all. Its TSTR is the ceiling any method can reach under a
     given binning. If real-roundtrip drops to ~AIM's score, the binning is the whole story.

Output: results/baselines_v2/<method>_<binning>_seed<k>.csv
"""
from __future__ import annotations
import argparse, os, sys, time, traceback
import numpy as np, pandas as pd

sys.path.insert(0, ".")
from src.data_loader import (load_adult, NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL,
                             FEATURE_BOUNDS, COLUMN_NAMES, PUBLIC_BINS)

INT_COLS = {"age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"}



def _create_synth(Synthesizer, method, epsilon):
    """Construct a smartnoise synthesizer, tolerating per-method constructor differences.

    PATE-GAN's __init__ does not accept `verbose`, and passing it raises a TypeError at
    construction that is easy to misread as "this method is unavailable" rather than a signature
    mismatch. Try the richer signature, then fall back.
    """
    try:
        return Synthesizer.create(method, epsilon=epsilon, verbose=False)
    except TypeError:
        return Synthesizer.create(method, epsilon=epsilon)

def make_bins(mode: str, n_uniform: int = 12) -> dict:
    if mode == "public":
        return {c: np.array(PUBLIC_BINS[c], dtype=float) for c in NUMERICAL_COLS}
    return {c: np.linspace(FEATURE_BOUNDS[c][0], FEATURE_BOUNDS[c][1], n_uniform + 1)
            for c in NUMERICAL_COLS}


def discretize(df: pd.DataFrame, bins: dict) -> pd.DataFrame:
    d = df.copy()
    for c in NUMERICAL_COLS:
        lo, hi = FEATURE_BOUNDS[c]
        v = pd.to_numeric(d[c], errors="coerce").fillna(lo).clip(lo, hi)
        nb = len(bins[c]) - 1
        d[c] = np.clip(np.digitize(v, bins[c]) - 1, 0, nb - 1).astype(str)
    for c in CATEGORICAL_COLS + [TARGET_COL]:
        d[c] = d[c].astype(str).str.strip()
    return d[COLUMN_NAMES]


def undiscretize(d: pd.DataFrame, bins: dict, mode: str, rng: np.random.RandomState) -> pd.DataFrame:
    out = d.copy()
    for c in NUMERICAL_COLS:
        nb = len(bins[c]) - 1
        i = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int).clip(0, nb - 1).values
        left, right = bins[c][i], bins[c][i + 1]
        if mode == "uniform":
            v = left + rng.random_sample(len(i)) * (right - left)
        else:
            v = (left + right) / 2.0
        if c in INT_COLS:
            # FLOOR, not round. Bins are half-open [left, right), so an integer column's bin
            # [40, 41) means exactly 40. Rounding a uniform draw from that interval sends ~half
            # the mass to 41 — a value that barely exists in real Adult — which showed up as a
            # spurious 0.26 TV penalty against AIM/MST on hours_per_week (2026-09-04). Flooring
            # keeps every reconstructed value inside the bin it actually came from.
            v = np.floor(v)
            v = np.minimum(v, np.ceil(right) - 1)
        out[c] = np.clip(v, FEATURE_BOUNDS[c][0], FEATURE_BOUNDS[c][1])
    return out[COLUMN_NAMES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--n-out", type=int, default=300)
    ap.add_argument("--binning", default="public", choices=["public", "uniform12", "uniform20"])
    ap.add_argument("--reconstruct", default="uniform", choices=["uniform", "midpoint"])
    ap.add_argument("--methods", nargs="+", default=["mst", "aim", "patectgan"])
    ap.add_argument("--draws", type=int, default=5, help="synthetic draws per fitted model")
    ap.add_argument("--outdir", default="results/baselines_v2")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    nb = 20 if a.binning == "uniform20" else 12
    bins = make_bins(a.binning, nb)
    print(f"binning={a.binning} reconstruct={a.reconstruct} eps={a.epsilon} "
          f"bins/col={ {c: len(b)-1 for c, b in bins.items()} }")

    raw_train, raw_test, *_ = load_adult()
    train_disc = discretize(raw_train, bins)

    # ── the control: real data through the SAME binning, no DP, no synthesis ──────────
    rng = np.random.RandomState(0)
    for k in range(a.draws):
        samp = train_disc.sample(n=a.n_out, random_state=100 + k)
        out = undiscretize(samp, bins, a.reconstruct, np.random.RandomState(100 + k))
        out.to_csv(f"{a.outdir}/real-roundtrip_{a.binning}_seed{k}.csv", index=False)
    print(f"wrote {a.draws} real-roundtrip draws (the ceiling under this binning)")

    from snsynth import Synthesizer
    for method in a.methods:
        t0 = time.time()
        try:
            print(f"\n── {method.upper()} ──", flush=True)
            synth = _create_synth(Synthesizer, method, a.epsilon)
            synth.fit(train_disc, categorical_columns=COLUMN_NAMES, preprocessor_eps=0.0)
            print(f"  fitted in {time.time()-t0:.0f}s; drawing {a.draws} samples", flush=True)
            for k in range(a.draws):
                sample = synth.sample(a.n_out)
                out = undiscretize(sample, bins, a.reconstruct, np.random.RandomState(200 + k))
                out[TARGET_COL] = out[TARGET_COL].astype(str).str.strip()
                path = f"{a.outdir}/{method}_{a.binning}_seed{k}.csv"
                out.to_csv(path, index=False)
                print(f"    draw {k}: {len(out)} rows -> {path}", flush=True)
        except Exception as e:
            print(f"  {method}: FAILED after {time.time()-t0:.0f}s -> {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print("\nbaselines_v2 done")


if __name__ == "__main__":
    main()
