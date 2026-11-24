"""The rerun of Appendix E.9: NHANES or bank marketing through the released package with a freshly derived
configuration. The package's Stage A derives the configuration with unseeded selection noise counted inside
epsilon = 2, `decode_release` decodes it with the maximum-entropy decoder and the selection step at n = 300 and
1,000 (three draws each), and score_draws.py scores the draws on the same split. CPU only, no model, no API.
Usage (from research/): python controls/rerun_fresh_configuration.py --dataset nhanes --seed 42 [--compare LABEL]
LABEL names an existing scored arm under results/baselines_modern/ to score beside the rerun (for example the
seeded configuration's plain decoder). Each call is a new private derivation on that split."""
import argparse, glob, os, subprocess, sys, time
ROOT = os.environ.get("CORTEC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=["nhanes", "bank_marketing"])
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--compare", default=None)
a = ap.parse_args()
from src.dataset_spec import get_spec
import src.datasets_auto  # noqa: F401  (registers every base dataset)
from sklearn.model_selection import train_test_split
from cortec import Schema, release_statistics, decode_release, fit_release_model
spec = get_spec(a.dataset); df = spec.loader(); spec.validate(df)
tr, _ = train_test_split(df, test_size=0.2, random_state=a.seed, stratify=df[spec.target_col])
schema = Schema(name=a.dataset, target=spec.target_col, positive=spec.positive_class, negative=spec.negative_class,
                numerical={c: (float(spec.feature_bounds[c][0]), float(spec.feature_bounds[c][1])) for c in spec.numerical_cols},
                categorical={c: sorted(df[c].astype(str).str.strip().unique().tolist()) for c in spec.categorical_cols},
                bins={c: [float(x) for x in spec.public_bins[c]] for c in spec.numerical_cols})
label = f"rerun_fresh_{a.dataset}_s{a.seed}"; out = os.path.join(ROOT, "results", "rerun_fresh", label); os.makedirs(out, exist_ok=True)
train = tr[spec.column_names].copy()
for c in spec.categorical_cols + [spec.target_col]: train[c] = train[c].astype(str).str.strip()
t0 = time.time()
rel = release_statistics(schema, train, epsilon_total=2.0, n_min=150, max_rows_per_person=1)
rel.to_json(os.path.join(out, "release.json"))
print(f"{label}: release with {len(rel.cohorts)} cohorts and {len(rel.conditional_levels)} levels in {time.time() - t0:.0f} s", flush=True)
model = fit_release_model(schema, rel, iters=3000, seed=42)
for n in (300, 1000):
    files = []
    for k in range(3):
        d = decode_release(schema, rel, n, pool_factor=3, seed=42 + k, model=model, verbose=False)
        f = os.path.join(out, f"raw_n{n}_draw{k}.csv"); d.to_csv(f, index=False); files.append(f)
    cmp = []
    if a.compare:
        g = os.path.join(ROOT, "results", "baselines_modern", a.compare, f"n{n}_draw*.csv")
        if glob.glob(g): cmp = ["--compare", f"{a.compare}={g}"]
    subprocess.run([sys.executable, os.path.join(ROOT, "controls", "score_draws.py"), "--dataset", f"{a.dataset}_auto",
                    "--method", label, "--n", str(n), "--seed", str(a.seed), "--files", *files] + cmp, check=True, cwd=ROOT)
print(f"{label}: done", flush=True)
