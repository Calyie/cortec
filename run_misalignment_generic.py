"""
run_misalignment_generic.py — the prior-misalignment test on ANY registered dataset.

This is the experiment that separates the two ways a generator can produce useful synthetic
data: because it transmitted THIS dataset's relationships, or because its own prior happened to
be right. On Adult and on credit the LLM's prior IS roughly right (degrees→income,
late payments→default), so an unconditioned header-only baseline scores well and the two
explanations are indistinguishable. On the diabetes data the effect is even starker: header-only
trains a BETTER readmission model than real data does, while having held-out conditional error
five times worse than a shuffled-target floor.

A regulated deployment is the opposite case — institution-specific patterns that contradict
general expectations — and that is what this sweeps. For a chosen group, the target rate in the
private data is forced to r, Stage A re-releases from that modified data, and CoRTeC regenerates
only the affected cohorts. The slope of generated-rate against released-rate is the quantity:

    slope ~ 1  the private relationship is transmitted at full magnitude
    slope ~ 0  generation follows the model's prior and ignores the private data

The header-only control needs no run per point: its prompt never sees the private data, so one
draw is the control for every r.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.dataset_spec import get_spec
import src.datasets_auto  # noqa: F401  (registers the auto-configured datasets, e.g. nhanes_auto)
import src.datasets_extra  # noqa: F401
from src.generic_pipeline import release_statistics
from sklearn.model_selection import train_test_split


def _in_group(series, group_values) -> "pd.Series":
    """Membership test that survives a model formatting an integer as a float.

    A 7B model emitted `number_inpatient` as '2.0'/'3.0'/'10.0' while the group was declared as
    '2','3','10'. Exact string matching found NOTHING, the measured group came back empty, and the
    sweep reported `generated None (n=0)` — a silently VOID arm, not an error. This is the same
    defect as the evaluator's `_norm_categorical` fix (a model writing 1.0 for 1); that fix was
    never carried into this harness. Compare numerically when both sides are numeric, and fall
    back to normalised strings otherwise.
    """
    want_num = pd.to_numeric(pd.Series([str(v) for v in group_values]), errors="coerce")
    got_num = pd.to_numeric(series, errors="coerce")
    if want_num.notna().all() and got_num.notna().any():
        return got_num.isin(set(want_num.tolist()))
    norm = lambda s: s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return norm(series).isin({norm(pd.Series([v]))[0] for v in group_values})


def invert(spec, df: pd.DataFrame, group_col: str, group_values: list[str], rate: float,
           seed: int) -> pd.DataFrame:
    """Force the target rate to `rate` for rows whose group_col is in group_values."""
    out = df.copy()
    mask = _in_group(out[group_col], group_values)
    rng = np.random.RandomState(seed)
    before = spec.is_positive(out[mask]).mean()
    out.loc[mask, spec.target_col] = np.where(
        rng.random_sample(int(mask.sum())) < rate, spec.positive_class, spec.negative_class)
    print(f"  [INVERSION] {group_col} in {group_values}: {int(mask.sum())} rows, "
          f"true rate {before:.1%} -> forced {rate:.0%}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--group-values", nargs="+", required=True)
    ap.add_argument("--cohort-filter", nargs="+", required=True,
                    help="substrings selecting the cohorts to regenerate")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--n-synthetic", type=int, default=75)
    ap.add_argument("--rows-per-call", type=int, default=25)
    # gemini/openai were missing although `llm_generator` supports both — so the transmission
    # sweep, which is what `cortec.models.PROFILES` is built from, could not be run on the API
    # models this project actually uses. That is why PROFILES has no Gemini entry.
    ap.add_argument("--backend", default="anthropic",
                    choices=["anthropic", "ollama", "gemini", "openai", "mock"])
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-num-predict", type=int, default=None,
                    help="raise for reasoning models whose chain of thought consumes the budget")
    ap.add_argument("--budget-usd", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--header-control", default=None)
    ap.add_argument("--outdir", default=None)
    # The PUBLISHED capability ladder used the per-cohort RATE path, whose saturation is exactly
    # what the cell-wise path fixes (magnitude error 0.155 -> 0.002 on the registry, same model and
    # release). So the ladder may be measuring the prompt rather than the models. Default stays
    # False to keep the published experiment reproducible; pass --by-cell for the corrected arm.
    # The finest declared level gives 54 (cohort, cell) pairs on diabetes; at n=75 that is ~1.4
    # rows per cell and the per-cell prompt asks for a single record, which collapses parsing.
    # Exposing the level lets the rate-vs-cell comparison run where the cell path is viable.
    ap.add_argument("--cond-levels", type=int, nargs="+", default=[0, 3],
                    help="conditional levels to release (default 0 3, the published setting)")
    ap.add_argument("--by-cell", dest="by_cell", action="store_true", default=False,
                    help="generate per released cell with an exact positive COUNT")
    a = ap.parse_args()

    spec = get_spec(a.dataset)
    out = Path(a.outdir or f"results/{a.dataset}_misalign")
    out.mkdir(parents=True, exist_ok=True)
    df = spec.loader(); spec.validate(df)
    tr, _ = train_test_split(df, test_size=0.2, random_state=a.seed,
                             stratify=df[spec.target_col])
    tr = tr.reset_index(drop=True)

    def group_rate(d: pd.DataFrame):
        m = _in_group(d[a.group_col], a.group_values)
        return (float(spec.is_positive(d[m]).mean()), int(m.sum())) if m.sum() else (None, 0)

    points, spent = [], 0.0
    from src.llm_generator import LLMSyntheticGenerator

    for r in a.rates:
        tag = f"r{int(round(r*100)):03d}"
        odir = out / tag
        odir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*74}\n=== {spec.name}: {a.group_col}={a.group_values} -> {r:.0%}\n{'='*74}")
        np.random.seed(a.seed)
        inv = invert(spec, tr, a.group_col, a.group_values, r, a.seed)
        cs = release_statistics(spec, inv, epsilon_stats=2.0, cond_frac=0.5, n_min=150,
                                cond_levels=tuple(a.cond_levels))
        json.dump(cs, open(odir / "cohort_statistics.json", "w"), indent=2, default=str)
        keep = [c for c in cs if any(s.lower() in c["cohort_name"].lower()
                                     for s in a.cohort_filter)]
        if not keep:
            print(f"  !! cohort filter {a.cohort_filter} matched none of "
                  f"{[c['cohort_name'] for c in cs]}"); break
        print(f"  regenerating {len(keep)} cohorts: {[c['cohort_name'] for c in keep]}")

        gen = LLMSyntheticGenerator(backend=a.backend, model=a.model,
                                    rows_per_call=a.rows_per_call, spec=spec,
                                    ollama_url=a.ollama_url)
        if a.ollama_num_predict:
            gen.ollama_num_predict = a.ollama_num_predict
            gen.ollama_num_ctx = max(gen.ollama_num_ctx, a.ollama_num_predict + 4096)
        # Ollama is self-hosted: no per-token cost, so the dollar cap does not apply.
        gen.budget_usd = None if a.backend == "ollama" else max(a.budget_usd - spent, 0.05)
        try:
            d = (gen.generate_cortec_by_cell(keep, n_total=a.n_synthetic, seed=a.seed)
                 if a.by_cell else gen.generate_cortec(keep, n_total=a.n_synthetic))
        except LLMSyntheticGenerator.GenerationAborted as e:
            print(f"  !!! {e}")
            d = gen.partial_dataframe()
        spent += gen.spend_usd()
        if not len(d):
            print("  no rows; stopping"); break
        p = odir / "synthetic_cortec.csv"
        d.to_csv(p, index=False)

        # the DP rate the generator was actually shown for the affected cells
        tbl = cs[0].get("conditional_target", {})
        shown = [v for k, v in tbl.items()
                 if any(str(g).lower() in k.lower() for g in a.cohort_filter)]
        dp_shown = float(np.mean(shown)) if shown else float("nan")
        got, n = group_rate(d)
        points.append({"rate": r, "dp_shown": dp_shown, "generated": got, "n": n,
                       "spend": gen.spend_usd()})
        print(f"  -> target {r:.0%} | DP shown {dp_shown:.3f} | generated "
              f"{got if got is None else f'{got:.3f}'} (n={n}) | cumulative ${spent:.2f}")
        if a.backend != "ollama" and spent >= a.budget_usd:
            print("  budget reached; stopping"); break

    hdr = None
    if a.header_control and Path(a.header_control).exists():
        hdr, hn = group_rate(pd.read_csv(a.header_control))

    print("\n" + "=" * 90)
    print(f"PRIOR-MISALIGNMENT SWEEP — {spec.name}: "
          f"P({spec.target_col}={spec.positive_class} | {a.group_col} in {a.group_values})")
    print("=" * 90)
    if hdr is not None:
        print(f"header-only control (never sees the private data): {hdr:.1%}")
    print(f"\n{'target':>8s} {'DP shown':>10s} {'generated':>11s} {'n rows':>8s}")
    for p in points:
        g = "—" if p["generated"] is None else f"{p['generated']:.1%}"
        print(f"{p['rate']:>8.0%} {p['dp_shown']:>10.3f} {g:>11s} {p['n']:>8d}")
    fit = [p for p in points if p["generated"] is not None and np.isfinite(p["dp_shown"])]
    if len(fit) >= 2:
        x = np.array([p["dp_shown"] for p in fit]); y = np.array([p["generated"] for p in fit])
        slope, inter = np.polyfit(x, y, 1)
        print(f"\nslope = {slope:.3f}  intercept = {inter:.3f}  "
              f"mean|generated-shown| = {np.mean(np.abs(y-x)):.3f}")
        json.dump({"points": points, "slope": float(slope), "intercept": float(inter),
                   "header_control": hdr}, open(out / "summary.json", "w"), indent=2, default=str)
    print(f"\ntotal spend: ${spent:.2f}")


if __name__ == "__main__":
    main()
