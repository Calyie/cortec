"""
run_baseline_transmission.py — do the DP synthesisers actually in industrial use transmit a
conditional relationship present in the private data?

This is the experiment that places CoRTeC among the methods a regulated institution would
realistically deploy. It asks one question of every method on equal terms:

    If the private data says P(target | group) = r, does the synthetic data say r?

The protocol is identical for every method and gives none of them an advantage:

  1. force the target rate for a chosen group to r in the private training data;
  2. hand that modified data to the method with the SAME epsilon and the SAME public binning;
  3. measure the realised rate for that group in the method's synthetic output;
  4. regress realised rate on true rate over the schedule of r.

    slope ~ 1, MAE ~ 0   the method carries the private conditional relationship
    slope ~ 0            the method's output is independent of that relationship

Both numbers are needed. A method exact at the endpoints and wrong in between scores slope ~ 1
with a large MAE, so slope alone can flatter a method badly.

Why this is the right comparison, and why an unconditioned language model is not. A regulated
institution does not choose between CoRTeC and "ask an LLM without any privacy accounting" — the
latter has no guarantee and would never clear review. It chooses among mechanisms that carry a
formal epsilon: marginal/graphical methods (MST, AIM, MWEM), DP-SGD-trained generative models
(DP-CTGAN, DP-GAN), and PATE-based ones (PATE-CTGAN, PATE-GAN). The question that decides between
them is whether the structure a downstream model needs survives the mechanism. That is what this
measures, at matched epsilon, on the same data, with the same evaluation.

Note on what "conditional relationship" means for a marginal method. MST and AIM select a set of
low-order marginals to measure. If the (group, target) marginal is selected, the relationship can
survive; if the budget goes elsewhere, it cannot, and no amount of sampling from the fitted model
will recover it. This experiment does not assume either outcome — it measures which happens.
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.dataset_spec import get_spec
import src.datasets_auto  # noqa: F401  (registers the auto-configured datasets, e.g. nhanes_auto)
import src.datasets_extra  # noqa: F401
from sklearn.model_selection import train_test_split



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

def group_mask(spec, df: pd.DataFrame, group_col: str, group_values: list[str]) -> pd.Series:
    """Membership of the group being manipulated.

    For a NUMERIC column the group has to be defined at the granularity the mechanism actually
    sees, which is the public bin. Selecting raw values 2..10 while the public bins are
    [0,1,2,3,21] would force the target for 2..10 but leave 11..21 untouched *inside the same
    bin*, so the mechanism could not represent the manipulated group even in principle and every
    method would score a spuriously low slope. Matching on bin index removes that confound and
    is also what makes the comparison fair across methods.
    """
    if group_col in spec.numerical_cols:
        edges = np.asarray(spec.public_bins[group_col], dtype=float)
        want = np.unique(np.clip(np.digitize(np.asarray([float(v) for v in group_values]),
                                             edges) - 1, 0, len(edges) - 2))
        v = pd.to_numeric(df[group_col], errors="coerce").fillna(edges[0])
        idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
        return pd.Series(np.isin(idx, want), index=df.index)
    return df[group_col].astype(str).str.strip().isin([str(v) for v in group_values])


def force_rate(spec, df: pd.DataFrame, group_col: str, group_values: list[str],
               rate: float, seed: int) -> tuple[pd.DataFrame, float, int]:
    """Force the target rate to `rate` for every record in the (bin-aligned) group."""
    out = df.copy()
    mask = group_mask(spec, out, group_col, group_values)
    rng = np.random.RandomState(seed)
    before = float(spec.is_positive(out[mask]).mean())
    out.loc[mask, spec.target_col] = np.where(
        rng.random_sample(int(mask.sum())) < rate, spec.positive_class, spec.negative_class)
    return out, before, int(mask.sum())


def realised_rate(spec, syn: pd.DataFrame, group_col: str, group_values: list[str]) -> tuple:
    m = group_mask(spec, syn, group_col, group_values)
    if m.sum() == 0:
        return None, 0
    return float(spec.is_positive(syn[m]).mean()), int(m.sum())


def discretise(spec, df):
    """Public binning, identical to the one every other baseline run in this project uses."""
    d = df.copy()
    for c in spec.numerical_cols:
        lo, hi = spec.feature_bounds[c]
        edges = np.asarray(spec.public_bins[c], dtype=float)
        v = pd.to_numeric(d[c], errors="coerce").fillna(lo).clip(lo, hi)
        d[c] = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2).astype(int).astype(str)
    for c in spec.categorical_cols + [spec.target_col]:
        d[c] = d[c].astype(str).str.strip()
    return d[spec.column_names]


def undiscretise(spec, d, rng):
    o = d.copy()
    for c in spec.numerical_cols:
        edges = np.asarray(spec.public_bins[c], dtype=float)
        nb = len(edges) - 1
        i = pd.to_numeric(o[c], errors="coerce").fillna(0).astype(int).clip(0, nb - 1).values
        left, right = edges[i], edges[i + 1]
        v = left + rng.random_sample(len(i)) * (right - left)
        # floor, not round: bins are half-open, so [40,41) means exactly 40 for an integer column
        v = np.minimum(np.floor(v), np.ceil(right) - 1)
        o[c] = np.clip(v, *spec.feature_bounds[c])
    return o[spec.column_names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--group-values", nargs="+", required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--methods", nargs="+",
                    default=["mst", "aim", "patectgan", "dpctgan", "pategan"])
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--n-out", type=int, default=2000,
                    help="sample enough rows that the group is well represented")
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit-timeout", type=int, default=5400)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    from snsynth import Synthesizer

    spec = get_spec(a.dataset)
    out = Path(a.outdir or f"results/{a.dataset}_baseline_transmission")
    out.mkdir(parents=True, exist_ok=True)

    df = spec.loader()
    spec.validate(df)
    train, _ = train_test_split(df, test_size=0.2, random_state=a.seed,
                                stratify=df[spec.target_col])
    train = train.reset_index(drop=True)

    print("=" * 96)
    print(f"BASELINE TRANSMISSION — {spec.name}")
    print(f"  P({spec.target_col}={spec.positive_class} | {a.group_col} in {a.group_values})")
    print(f"  forced to {a.rates} in the private data; eps={a.epsilon} for every method")
    print("=" * 96, flush=True)

    results: dict = {"dataset": spec.name, "group_col": a.group_col,
                     "group_values": list(a.group_values), "epsilon": a.epsilon,
                     "rates": list(a.rates), "methods": {}}

    for method in a.methods:
        pts = []
        print(f"\n{'='*96}\n=== {method.upper()}\n{'='*96}", flush=True)
        for r in a.rates:
            mod, before, n_group = force_rate(spec, train, a.group_col, a.group_values,
                                              r, seed=a.seed)
            true_rate = float(spec.is_positive(
                mod[mod[a.group_col].astype(str).str.strip().isin(
                    [str(v) for v in a.group_values])]).mean())
            td = discretise(spec, mod)
            t0 = time.time()
            try:
                syn = _create_synth(Synthesizer, method, a.epsilon)
                syn.fit(td, categorical_columns=spec.column_names, preprocessor_eps=0.0)
                fit_s = time.time() - t0
                got = []
                for k in range(a.draws):
                    s = undiscretise(spec, syn.sample(a.n_out), np.random.RandomState(700 + k))
                    g, n = realised_rate(spec, s, a.group_col, a.group_values)
                    if g is not None:
                        got.append(g)
                    s.to_csv(out / f"{method}_r{int(r*100):03d}_draw{k}.csv", index=False)
                gen = float(np.mean(got)) if got else None
                sd = float(np.std(got, ddof=1)) if len(got) > 1 else 0.0
                pts.append({"target": r, "true_rate": true_rate, "generated": gen,
                            "sd": sd, "n_group_private": n_group, "fit_seconds": round(fit_s, 1)})
                print(f"  target {r:.0%} | true {true_rate:.3f} | generated "
                      f"{'--' if gen is None else f'{gen:.3f}'} (sd {sd:.3f}) | fit {fit_s:.0f}s",
                      flush=True)
            except Exception as e:
                print(f"  target {r:.0%}: FAILED after {time.time()-t0:.0f}s -> "
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                pts.append({"target": r, "true_rate": true_rate, "generated": None,
                            "error": f"{type(e).__name__}: {str(e)[:200]}"})

        ok = [p for p in pts if p.get("generated") is not None]
        entry = {"points": pts, "n_ok": len(ok)}
        if len(ok) >= 2:
            x = np.array([p["true_rate"] for p in ok])
            y = np.array([p["generated"] for p in ok])
            slope, intercept = np.polyfit(x, y, 1)
            entry.update(slope=float(slope), intercept=float(intercept),
                         mae=float(np.mean(np.abs(y - x))),
                         spread=float(y.max() - y.min()))
            print(f"  -> slope {slope:.3f}  MAE {entry['mae']:.3f}  "
                  f"output spread {entry['spread']:.3f}")
        else:
            print(f"  -> insufficient successful points to fit a slope")
        results["methods"][method] = entry
        json.dump(results, open(out / "transmission.json", "w"), indent=2, default=str)

    # ── report ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("TRANSMISSION OF A PRIVATE CONDITIONAL RELATIONSHIP, at matched epsilon")
    print("slope ~1 = the relationship survives the mechanism; ~0 = the output ignores it")
    print("=" * 96)
    print(f"{'method':14s} {'slope':>8s} {'MAE':>8s} {'spread':>8s} {'pts':>5s}  outputs")
    for m, e in results["methods"].items():
        if "slope" not in e:
            print(f"{m:14s} {'--':>8s} {'--':>8s} {'--':>8s} {e['n_ok']:>5d}  did not complete")
            continue
        outs = ", ".join("--" if p.get("generated") is None else f"{p['generated']:.2f}"
                         for p in e["points"])
        print(f"{m:14s} {e['slope']:>8.3f} {e['mae']:>8.3f} {e['spread']:>8.3f} "
              f"{e['n_ok']:>5d}  [{outs}]")
    print(f"\nsaved -> {out}/transmission.json")


if __name__ == "__main__":
    main()
