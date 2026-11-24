"""Stage C of Appendix E.8 on a release made by the released package. One noisy draw of the private cell rates,
charged once at epsilon_cert, scores three datasets: the synthetic draws, a real hold-out sample of the same size
(the ceiling) and that sample with its outcome permuted (the floor). The tolerance is derived from the release.
CPU only, no model, no API. Each call spends epsilon_cert on the training split again.
Usage (from research/):
  python controls/stage_c_on_release.py --dataset nhanes --seed 42 \\
      --release results/rerun_fresh/rerun_fresh_nhanes_s42/release.json \\
      --synthetic 'results/rerun_fresh/rerun_fresh_nhanes_s42/raw_n300_draw*.csv'"""
import argparse, glob, os, sys
ROOT = os.environ.get("CORTEC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["adult", "nhanes", "bank_marketing"])
ap.add_argument("--seed", type=int, default=42, help="the split seed; also picks the hold-out rows and the permutation")
ap.add_argument("--release", required=True, help="a release.json written by the package, for example by rerun_fresh_configuration.py")
ap.add_argument("--synthetic", nargs="+", required=True, help="draw files or patterns, pooled into one dataset")
ap.add_argument("--epsilon-cert", type=float, default=1.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
import pandas as pd
from src.dataset_spec import get_spec
import src.datasets_auto  # noqa: F401  (registers every base dataset)
from sklearn.model_selection import train_test_split
from cortec import Schema, Release, bound_with_controls
spec = get_spec(a.dataset); df = spec.loader(); spec.validate(df)
tr, te = train_test_split(df, test_size=0.2, random_state=a.seed, stratify=df[spec.target_col])
schema = Schema(name=a.dataset, target=spec.target_col, positive=spec.positive_class, negative=spec.negative_class,
                numerical={c: (float(spec.feature_bounds[c][0]), float(spec.feature_bounds[c][1])) for c in spec.numerical_cols},
                categorical={c: sorted(df[c].astype(str).str.strip().unique().tolist()) for c in spec.categorical_cols},
                bins={c: [float(x) for x in spec.public_bins[c]] for c in spec.numerical_cols})


def _clean(frame):
    frame = frame[spec.column_names].copy()
    for c in spec.categorical_cols + [spec.target_col]:
        frame[c] = frame[c].astype(str).str.strip()
    return frame


paths = sorted({p for g in a.synthetic for p in glob.glob(g)})
if not paths:
    raise SystemExit(f"no files matched {a.synthetic}")
synthetic = _clean(pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))
report = bound_with_controls(schema, Release.from_json(a.release), _clean(tr), synthetic, _clean(te),
                             epsilon_cert=a.epsilon_cert, tolerance="auto", seed=a.seed)
print(f"{len(paths)} draw file(s), {len(synthetic)} synthetic rows")
print(report.summary())
out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.release)), "stage_c.json")
report.to_json(out)
print(f"written to {out}")
