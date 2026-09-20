"""
run_dataset.py — the whole CoRTeC pipeline on ANY registered dataset.

Stage A (DP release) -> Stage B (generation) -> baselines -> evaluation, all driven by a
DatasetSpec. This is what makes the multi-dataset validation possible: every Adult-specific
assumption now lives in the spec (src/dataset_spec.py, src/datasets_extra.py).

  python run_dataset.py --dataset diabetes --stage release
  python run_dataset.py --dataset diabetes --stage generate --n-synthetic 300 --budget-usd 2.5
  python run_dataset.py --dataset diabetes --stage baselines --n-out 300
  python run_dataset.py --dataset diabetes --stage evaluate
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from src.dataset_spec import get_spec, available
import src.datasets_extra  # noqa: F401  (registers diabetes + credit)
try:                                  # internal contamination variant; not in the public tree
    import src.datasets_obfuscated  # noqa: F401  (registers adult_obf)
except ImportError:
    pass
import src.datasets_regulated  # noqa: F401  (six untouched regulated benchmarks)
import src.datasets_synthetic  # noqa: F401  (renal_registry — the contamination control)
import src.datasets_clinical  # noqa: F401  (NHANES + MIMIC-III demo)
import src.datasets_auto  # noqa: F401  (auto-configured variants, see src/autoconfig.py)
from src.generic_pipeline import release_statistics
from sklearn.model_selection import train_test_split



def _create_synth(Synthesizer, method, epsilon, aim_max_model_size=None):
    """Construct a smartnoise synthesizer, tolerating per-method constructor differences.

    PATE-GAN's __init__ does not accept `verbose`, and passing it raises a TypeError at
    construction that is easy to misread as "this method is unavailable" rather than a signature
    mismatch. Try the richer signature, then fall back.

    `aim_max_model_size` (MB, smartnoise default 80) caps the junction tree AIM is willing to
    build. It is the parameter that governs whether AIM converges in bounded time on a wide
    schema, so it is exposed here to test whether our non-convergence on the two regulated
    datasets is a model-size problem rather than a domain-size one.
    """
    kw = {"epsilon": epsilon}
    if method == "aim" and aim_max_model_size is not None:
        kw["max_model_size"] = aim_max_model_size
    try:
        return Synthesizer.create(method, verbose=False, **kw)
    except TypeError:
        return Synthesizer.create(method, **kw)

def split(spec, seed=42, test_size=0.2):
    df = spec.loader()
    spec.validate(df)
    tr, te = train_test_split(df, test_size=test_size, random_state=seed,
                              stratify=df[spec.target_col])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


# ── stages ──────────────────────────────────────────────────────────────────────────
def stage_release(spec, a, out: Path):
    tr, _ = split(spec, seed=a.seed)
    np.random.seed(a.seed)
    cs = release_statistics(spec, tr, epsilon_stats=a.epsilon_total, cond_frac=a.cond_frac,
                            n_min=a.n_min, cond_levels=tuple(a.cond_levels),
                            class_conditional=a.class_conditional)
    json.dump(cs, open(out / "cohort_statistics.json", "w"), indent=2, default=str)
    print(f"  saved → {out}/cohort_statistics.json")
    return cs


def stage_generate(spec, a, out: Path):
    from src.llm_generator import LLMSyntheticGenerator
    stats_p = out / "cohort_statistics.json"
    if not stats_p.exists():
        # A fresh --outdir means a fresh DP release, i.e. another full eps_total spent on the
        # same private data. That is almost never what is wanted: extra draws are supposed to be
        # post-processing of ONE release, at zero additional cost. Say so loudly — this silently
        # double-charged the credit dataset once before it was caught.
        print("!" * 78)
        print(f"!! NO RELEASE IN {out} — performing a NEW DP release at eps={a.epsilon_total}.")
        print("!! This spends the budget AGAIN on the same data. Extra draws from an existing")
        print("!! release are free: copy that release's cohort_statistics.json into this")
        print("!! directory first, and the draws will cost no additional privacy budget.")
        print("!" * 78, flush=True)
        stage_release(spec, a, out)
    else:
        print(f"  reusing DP release at {stats_p} — no additional privacy budget spent",
              flush=True)
    cs = json.load(open(stats_p))

    gen = LLMSyntheticGenerator(backend=a.backend, model=a.model,
                                rows_per_call=a.rows_per_call, spec=spec,
                                ollama_url=a.ollama_url)
    gen.quota = bool(getattr(a, "quota", False)); gen.quota_seed = a.seed
    if getattr(a, "anthropic_max_tokens", None):
        gen.anthropic_max_tokens = int(a.anthropic_max_tokens)
    if a.ollama_num_predict:
        gen.ollama_num_predict = a.ollama_num_predict
        gen.ollama_num_ctx = max(gen.ollama_num_ctx, a.ollama_num_predict + 4096)
    gen.budget_usd = a.budget_usd
    manifest = {"dataset": spec.name, "args": vars(a), "draws": []}
    t0 = time.time()
    try:
        for d in range(a.draws):
            print(f"\n===== {spec.name} draw {d + 1}/{a.draws} =====")
            if a.skip_cortec:
                # the CoRTeC arm already exists for this release; regenerating it would spend
                # money to reproduce a number we hold, and would not be the same draw anyway
                manifest["draws"].append({"draw": d, "cortec_csv": None, "rows": 0,
                                          "note": "skipped, reusing an existing CoRTeC arm"})
            elif a.by_cell and a.pool_factor <= 1.0:
                # the pool path below is cohort-wise (the shipped configuration); a pool factor
                # therefore takes precedence over the cell-wise default, whose fallback would
                # otherwise generate a plain n-row draw and silently ignore the pool
                df = gen.generate_cortec_by_cell(cs, n_total=a.n_synthetic, seed=a.seed + d)
                p = out / f"synthetic_cortec_draw{d}.csv"
                df.to_csv(p, index=False)
                manifest["draws"].append({"draw": d, "cortec_csv": str(p), "rows": len(df)})
                print(f"  saved {len(df)} rows → {p}")
            elif a.pool_factor > 1.0:
                # pool-and-rake selection (src.generic_pipeline.select_to_release): generate a
                # pool_factor x n pool, keep the n rows whose marginals match the release.
                # Post-processing of the release; costs pool_factor x the generation spend, no ε.
                from src.generic_pipeline import select_to_release
                pool = gen.generate_cortec(cs, n_total=int(round(a.pool_factor * a.n_synthetic)))
                pp = out / f"synthetic_cortec_pool{d}.csv"
                pool.to_csv(pp, index=False)
                df = select_to_release(spec, cs, pool, a.n_synthetic, seed=a.seed + d)
                p = out / f"synthetic_cortec_draw{d}.csv"
                df.to_csv(p, index=False)
                manifest["draws"].append({"draw": d, "cortec_csv": str(p), "rows": len(df),
                                          "pool_csv": str(pp), "pool_rows": len(pool),
                                          "pool_factor": a.pool_factor})
                print(f"  saved {len(df)} rows selected from a pool of {len(pool)} → {p}")
            else:
                df = gen.generate_cortec(cs, n_total=a.n_synthetic)
                p = out / f"synthetic_cortec_draw{d}.csv"
                df.to_csv(p, index=False)
                manifest["draws"].append({"draw": d, "cortec_csv": str(p), "rows": len(df)})
                print(f"  saved {len(df)} rows → {p}")
            if not a.skip_header_only:
                dh = (gen.generate_matched_header_only(cs, n_total=a.n_synthetic)
                      if a.matched_header_only
                      else gen.generate_header_only(n_total=a.n_synthetic))
                ph = out / f"synthetic_header_only_draw{d}.csv"
                dh.to_csv(ph, index=False)
                manifest["draws"][-1].update(header_csv=str(ph), header_rows=len(dh))
    except LLMSyntheticGenerator.GenerationAborted as e:
        print(f"\n!!! {e}")
        manifest["aborted"] = str(e)
        part = gen.partial_dataframe()
        if len(part):
            pp = out / "partial_generated_rows.csv"
            part.to_csv(pp, index=False)
            manifest["partial_rows"] = len(part)
            print(f"  kept {len(part)} already-generated rows → {pp}")
    finally:
        # keep whatever was generated on ANY exit, not only on a guarded abort: a KeyError in a
        # post-processing hook once discarded 465 paid rows
        try:
            part = gen.partial_dataframe()
            if len(part) and not (out / "partial_generated_rows.csv").exists():
                part.to_csv(out / "partial_generated_rows.csv", index=False)
                manifest["partial_rows"] = len(part)
        except Exception:
            pass
        manifest.update(api_calls=gen._n_calls, parse_ok=gen._n_parse_ok,
                        rows=gen._total_rows, spend_usd=round(gen.spend_usd(), 4),
                        wall_seconds=round(time.time() - t0, 1))
        json.dump(manifest, open(out / "generation_manifest.json", "w"), indent=2, default=str)
        print(f"\nAPI calls {gen._n_calls} | rows {gen._total_rows} | ${gen.spend_usd():.3f}")


def stage_baselines(spec, a, out: Path):
    """AIM / MST / PATE-CTGAN at matched n, using the spec's public bins."""
    from snsynth import Synthesizer
    tr, _ = split(spec, seed=a.seed)
    cols = spec.column_names
    bins = {c: np.asarray(spec.public_bins[c], dtype=float) for c in spec.numerical_cols}

    def disc(df):
        d = df.copy()
        for c in spec.numerical_cols:
            lo, hi = spec.feature_bounds[c]
            v = pd.to_numeric(d[c], errors="coerce").fillna(lo).clip(lo, hi)
            d[c] = np.clip(np.digitize(v, bins[c]) - 1, 0, len(bins[c]) - 2).astype(int).astype(str)
        for c in spec.categorical_cols + [spec.target_col]:
            d[c] = d[c].astype(str).str.strip()
        return d[cols]

    def undisc(d, rng):
        o = d.copy()
        for c in spec.numerical_cols:
            nb = len(bins[c]) - 1
            i = pd.to_numeric(o[c], errors="coerce").fillna(0).astype(int).clip(0, nb - 1).values
            left, right = bins[c][i], bins[c][i + 1]
            v = left + rng.random_sample(len(i)) * (right - left)
            # floor, not round: bins are half-open, so [40,41) means exactly 40 for an
            # integer column. Rounding scatters half the mass into the next bin.
            v = np.minimum(np.floor(v), np.ceil(right) - 1)
            o[c] = np.clip(v, *spec.feature_bounds[c])
        return o[cols]

    td = disc(tr)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(a.draws):
        s = td.sample(n=min(a.n_out, len(td)), random_state=100 + k)
        undisc(s, np.random.RandomState(100 + k)).to_csv(
            out / f"real-roundtrip_seed{k}.csv", index=False)
    print(f"  wrote {a.draws} real-roundtrip draws (ceiling under this binning)")

    for m in a.methods:
        t0 = time.time()
        try:
            print(f"\n── {m.upper()} (eps={a.epsilon_total}) ──", flush=True)
            syn = _create_synth(Synthesizer, m, a.epsilon_total, a.aim_max_model_size)
            syn.fit(td, categorical_columns=cols, preprocessor_eps=0.0)
            print(f"  fitted in {time.time()-t0:.0f}s", flush=True)
            for k in range(a.draws):
                o = undisc(syn.sample(a.n_out), np.random.RandomState(200 + k))
                o.to_csv(out / f"{m}_seed{k}.csv", index=False)
                print(f"    draw {k}: {len(o)} rows", flush=True)
        except Exception as e:
            print(f"  {m}: FAILED after {time.time()-t0:.0f}s -> {type(e).__name__}: {e}",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=available() + ["diabetes", "credit"])
    ap.add_argument("--stage", required=True,
                    choices=["release", "generate", "baselines"])
    ap.add_argument("--backend", default="anthropic")
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-num-predict", type=int, default=None,
                    help="raise for reasoning models whose chain of thought eats the budget")
    ap.add_argument("--n-synthetic", type=int, default=300)
    ap.add_argument("--n-out", type=int, default=300)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--rows-per-call", type=int, default=25)
    ap.add_argument("--epsilon-total", type=float, default=2.0)
    ap.add_argument("--cond-frac", type=float, default=0.2)
    ap.add_argument("--n-min", type=int, default=150)
    ap.add_argument("--cond-levels", type=int, nargs="+", default=[0, 3])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--skip-header-only", action="store_true")
    ap.add_argument("--skip-cortec", action="store_true",
                    help="generate only the control arm, reusing an existing CoRTeC arm")
    ap.add_argument("--matched-header-only", action="store_true",
                    help="control = CoRTeC's prompt with the released arrays deleted (§7.1.2)")
    ap.add_argument("--no-class-conditional", dest="class_conditional", action="store_false",
                    default=True,
                    help="release POOLED cohort histograms only (the pre-2026-09-18 release); the "
                         "default releases one histogram block per (cohort, class) where both "
                         "classes clear n_min, at the same epsilon per query")
    ap.add_argument("--by-cell", dest="by_cell", action="store_true", default=True,
                    help="generate per released cell with an exact positive COUNT (default: on). "
                         "Asking for a rate saturates; asking for 'exactly k of n' does not.")
    ap.add_argument("--anthropic-max-tokens", type=int, default=None,
                    help="output budget per Anthropic call (thinking counts against it); above "
                         "8192 the call streams")
    ap.add_argument("--quota", action="store_true",
                    help="cohort-wise path: give every batch exact per-column counts apportioned "
                         "from the release instead of shares to match")
    ap.add_argument("--pool-factor", type=float, default=1.0,
                    help="cohort-wise path only: generate this many times n rows and select the n "
                         "whose marginals match the release (pool-and-rake selection, §7.12)")
    ap.add_argument("--no-by-cell", dest="by_cell", action="store_false",
                    help="use the original per-cohort rate-based path")
    ap.add_argument("--methods", nargs="+", default=["mst", "aim", "patectgan"])
    ap.add_argument("--aim-max-model-size", type=int, default=None,
                    help="AIM junction-tree cap in MB (smartnoise default 80). Lower "
                         "values trade fidelity for convergence time.")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    spec = get_spec(a.dataset)
    # `--cond-levels` defaults to (0, 3), which assumes the spec has at least four conditional
    # levels. Autoconfig derives a VARIABLE number: nhanes_auto picks 3 columns (4 levels, index 3
    # valid) while adult_auto picks 2 (3 levels, index 3 out of range) — and the pipeline died with
    # a bare IndexError from `conditional_levels[level]`. Clamp to what the spec actually has, and
    # say so, rather than requiring the caller to know the derived depth in advance.
    _max_level = len(spec.conditional_levels) - 1
    _asked = sorted(set(a.cond_levels))
    _clamped = sorted({min(lv, _max_level) for lv in _asked if lv >= 0})
    if _clamped != _asked:
        print(f"  note: --cond-levels {_asked} clamped to {_clamped}; this spec has "
              f"{_max_level + 1} conditional level(s) (0..{_max_level})")
        a.cond_levels = _clamped
    out = Path(a.outdir or f"results/{a.dataset}")
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"  {spec.name} | stage={a.stage} | eps_total={a.epsilon_total} | n_min={a.n_min}")
    print(f"  {spec.description[:100]}")
    print("=" * 78)

    {"release": stage_release, "generate": stage_generate,
     "baselines": stage_baselines}[a.stage](spec, a, out)


if __name__ == "__main__":
    main()
