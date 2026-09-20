"""Stage C — a UTILITY TRANSMISSION BOUND on how much released conditional structure reached the
synthetic output. Computed under DP. It bounds utility, never privacy.

WHAT THE BOUND COVERS, precisely. For each released cell `c` the private data has a true positive
rate `p_c` and the synthetic data has a rate `q_c`. `q_c` is a function of the synthetic output
alone, so it is PUBLIC and free. `p_c` is private. We release a Laplace estimate of it and turn the
noise into a one-sided confidence bound, giving a statement of the form

    with probability at least 1 - alpha, simultaneously over all released cells,
        |p_c - q_c|  <=  |p_hat_c - q_c| + b_c * ln(k / alpha)

where `b_c = (1/n_min) / eps_cell` is the Laplace scale -- the sensitivity of a released cell's
rate is bounded by 1/n_min, the PUBLIC floor, never by the private cell size -- and `k` is the number of cells (a union
bound over the cells, so the guarantee is SIMULTANEOUS rather than per-cell — reporting a per-cell
95% bound over 20 cells and calling it simultaneous would be wrong).

Why a confidence bound rather than a p-value. A p-value answers "could this difference be chance?".
The claim the paper needs is the opposite direction: "the difference is provably no larger than X".
A one-sided bound states exactly that, and it degrades gracefully — a wide bound is an honest
"cannot bound this tightly at this epsilon", not a false negative.

The test costs privacy budget and is accounted for. It is NOT free: `p_hat_c` reads the private
data. Bounding at eps_certify means the deployment spent eps_release + eps_certify in total, and
this module refuses to pretend otherwise.

FLOOR AND CEILING. `certify()` is meaningless unless a known-good input passes and a known-bad one
fails, so `certify_with_controls()` runs three conditions: the synthetic data, a real hold-out
sample (should clear the tolerance tightly), and the same real sample with its target permuted
(must NOT clear it). A run where the permuted control also clears means the tolerance is too loose
to discriminate and the result should be discarded — the §3 saturation lesson, applied here.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

# The research pipeline has no dependency on the reference package, so the threshold is defined
# once here and a test asserts the two agree -- a per-person epsilon that is vacuous in the tool
# and acceptable in the paper's own Stage C report would be the worst of both.
VACUOUS_EPSILON_PER_PERSON = 10.0

import datetime as _dt
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset_spec import get_spec
import src.datasets_extra       # noqa: F401
import src.datasets_regulated   # noqa: F401
import src.datasets_clinical    # noqa: F401
import src.datasets_synthetic   # noqa: F401
import src.datasets_auto        # noqa: F401


# ── standards alignment ─────────────────────────────────────────────────────────────
# NIST SP 800-226 (Guidelines for Evaluating Differential Privacy Guarantees) requires a DP claim
# to state the parameters below, because "epsilon = 2" alone is not a guarantee — it is a number
# whose meaning depends entirely on the unit, the neighbouring relation and the variant. Most
# published DP synthetic-data work omits several of these. They are emitted with every bound report
# so a reviewer or auditor can check the claim rather than take it on trust.
#
# Standards THIS REPORT is aligned to. The list is deliberately short, and it used to be longer.
#   NIST SP 800-226   parameter reporting, and documenting where the guarantee does NOT hold
#   NIST SP 800-188   de-identifying government datasets; disclosure-review framing
#
# It formerly also claimed ISO/IEC 27559, ISO/IEC 20889, GDPR Art. 25 and HIPAA Expert
# Determination. Those belong to OTHER controls in the architecture, not to this bound:
# re-identification risk is carried by the aggregation floor and the measured MIA evidence, data
# protection by design by the fact that private data never leaves the regulated zone. HIPAA Expert
# Determination is the sharpest case -- it requires a statistical assessment of RE-IDENTIFICATION
# RISK, and this report bounds a UTILITY quantity, so listing it here is a category error with
# compliance consequences. They are now named explicitly in STANDARDS_NOT_CLAIMED, because a
# compliance reader who sees a short list cannot tell a considered exclusion from an oversight.

# Below this many synthetic rows in a cell, q_c is too coarse to be meaningful (with 1 row it can
# only be 0 or 1). Such cells are reported, not silently trusted.
MIN_SYNTH_ROWS_PER_CELL = 20

# A bare standard name in a compliance artefact is an invitation to assume the broadest reading of
# it. Each entry says what it is claimed FOR, so "SP 800-188" cannot be read as "this performed a
# disclosure review" -- it did not; it documents fitness for use.
STANDARDS_ALIGNED = {
    "NIST SP 800-226": "parameter reporting for the DP claim block, and documenting where the "
                       "guarantee does not hold",
    "NIST SP 800-188": "governance and documentation of the release package. This is the DOCUMENTED "
                       "UTILITY CLAIM attached to a release -- fitness for use. It is NOT a "
                       "disclosure review: disclosure risk is assessed by the aggregation floor and "
                       "the measured re-identification evidence, not by this bound.",
}

# Named rather than merely omitted: an absent entry reads as an oversight, an explicit refusal reads
# as a decision. Each says where the weight actually sits, so a reader is not left looking for it.
STANDARDS_NOT_CLAIMED = {
    "HIPAA Expert Determination":
        "NOT claimed. Expert Determination requires a statistical assessment of RE-IDENTIFICATION "
        "RISK. This report bounds a UTILITY quantity -- how much released conditional structure "
        "reached the output -- and mapping it to that route would be a category error with "
        "compliance consequences. That route is carried by the DP release itself (with epsilon "
        "stated per PERSON) together with measured re-identification evidence, not by this bound.",
    "ISO/IEC 27559 / ISO/IEC 20889":
        "NOT claimed. These frame re-identification risk and de-identification technique. The "
        "corresponding evidence is the aggregation floor (cells of at least n_min) and the measured "
        "membership-inference results, neither of which is this report.",
    "GDPR Art. 25 / Recital 26":
        "NOT claimed. Data protection by design is carried by the architecture -- private data "
        "never leaves the regulated zone and the prompt carries only the DP release -- not by a "
        "utility bound computed downstream of it.",
}


DP_CLAIM = {
    "variant": "pure epsilon-DP (central / trusted-curator model)",
    "delta": 0.0,
    "neighbouring_relation": "add or remove one record (UNBOUNDED DP)",
    "privacy_unit": "one dataset row",
    "composition": "parallel across disjoint cells; sequential across overlapping query groups",
    "mechanism": "Laplace",
    "sensitivity": "1/n_min per released cell rate: the PUBLIC size floor, not the private cell "
                   "size. A scale set from the true cell size is data-dependent under add/remove "
                   "adjacency and is not pure epsilon-DP; the bound is conservative by n/n_min.",
}

# Where the guarantee does NOT hold — NIST SP 800-226 asks for this explicitly and it is the part
# most often left out.
DP_CAVEATS = [
    "The privacy unit is one ROW, not one person. Under group privacy a person contributing k "
    "rows receives k*epsilon, so the per-person guarantee is weaker than the headline number "
    "whenever rows repeat per individual. This is NOT hypothetical here: Diabetes 130 has 101,766 "
    "rows over 71,518 patients (mean 1.42, MAXIMUM 40), so its worst-case per-person guarantee at "
    "a row-level epsilon of 2.0 is epsilon_person = 80. --max-rows-per-person is REQUIRED for "
    "exactly this reason: this report is not produced without a declared privacy unit. Reduce k "
    "before Stage A (cap contributions, or aggregate to one row per person) if a per-person "
    "guarantee is needed.",
    "Noise is sampled with a floating-point Laplace generator. Naive floating-point Laplace "
    "sampling is vulnerable to the Mironov (2012) attack, which can leak the exact value through "
    "the low-order bits. A deployment handling real PHI should use a secure/discrete sampler "
    "(e.g. discrete Laplace / snapping); this implementation does not, and that is a known gap.",
    "The guarantee covers the RELEASED STATISTICS and this bound's own query. It says nothing "
    "about the generator's pretraining data, which is outside the DP boundary entirely.",
    "This is a UTILITY TRANSMISSION BOUND computed under DP. It bounds how far the synthetic "
    "conditional structure can differ from the private one; it does NOT certify privacy, says "
    "nothing about re-identification risk, and must never be presented as a privacy audit.",
    "WHICH cells enter the bound depends on private counts (a cell is included only if it "
    "holds at least n_min records). That selection is data-dependent and is NOT charged to "
    "epsilon here, exactly as the research pipeline treats cohort sizes as public (paper "
    "limitation 5). A deployment that cannot make that assumption must charge the suppression "
    "decision separately, as tools/cortec does via spend_suppression_counts.",
    "Cells for which the synthetic data supplies NO rows are scored at the trivial bound of 1.0. "
    "A bound therefore cannot be obtained by covering a convenient subset of the cells.",
]

def _secure_laplace(value: float, scale: float) -> float:
    """Laplace noise from a cryptographically secure source, never from a seeded generator.

    Uses the same mechanism as the release path when diffprivlib is available. The fallback is
    `secrets.SystemRandom`, which is unseedable by construction -- the point is that no caller,
    and no recorded command line, can reproduce the draw.
    """
    if scale <= 0:
        return float(value)
    try:
        from diffprivlib.mechanisms import Laplace as _L
        return float(_L(epsilon=1.0, sensitivity=float(scale)).randomise(float(value)))
    except Exception:
        import secrets
        u = secrets.SystemRandom().random() - 0.5
        return float(value) - scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def _cell_keys(spec, df: pd.DataFrame, level: int) -> pd.Series:
    return spec.level_keys(df, level)


def certify(spec, real: pd.DataFrame, synth: pd.DataFrame, *, level: int,
            epsilon: float, alpha: float = 0.05, n_min: int = 150,
            rng: np.random.Generator | None = None) -> dict:
    """One-sided simultaneous DP bound on |p_c - q_c| across released cells.

    The `rng` argument controls only EXPERIMENTAL reproducibility; it must never be used for the
    DP noise itself. This mechanism reads `p_true` from the private data and publishes the noised
    `p_hat`, so a seeded draw makes the whole spend void: an adversary who knows the seed -- and it
    was a logged command-line argument -- regenerates the identical Laplace value, subtracts it and
    recovers `p_true` exactly. Verified: recovery error 0.00e+00. The noise is now drawn from the
    same secure mechanism as the release path.
    """
    _ = rng   # retained for signature compatibility; deliberately NOT used for the DP noise
    rk = _cell_keys(spec, real, level)
    sk = _cell_keys(spec, synth, level)
    ry = spec.is_positive(real)
    sy = spec.is_positive(synth)

    groups = {str(c): idx for c, idx in rk.groupby(rk).groups.items() if len(idx) >= n_min}
    if not groups:
        return {"level": level, "epsilon_spent": epsilon, "alpha": alpha, "n_cells": 0,
                "n_cells_covered": 0, "n_cells_uncovered": 0, "n_cells_thin": 0,
                "worst_case_bound": 1.0, "mean_bound": 1.0,
                "reason": f"no cell reached n_min={n_min}", "cells": []}

    k = len(groups)
    # Cells PARTITION the private data, so the k rate queries are disjoint and compose in parallel:
    # the total cost is max over cells, not the sum, and each cell may therefore spend the full
    # `epsilon`. (Splitting it k ways would be a k-fold accuracy loss for no privacy gain — the
    # same mistake as the conditional-level split fixed in tools/cortec.)
    eps_cell = epsilon
    # The union bound below is over the k independent Laplace draws. For Laplace(b),
    # P(|X| > b*ln(k/alpha)) = alpha/k, so k cells give alpha overall — a SIMULTANEOUS guarantee.

    out = []
    for cell, idx in groups.items():
        n_c = len(idx)
        p_true = float(ry.loc[idx].mean())
        # Sensitivity bound is 1/n_min, NOT 1/n_c (review A1). Under add/remove-one adjacency the
        # cell size is private, so a scale set from n_c is data-dependent and the mechanism is not
        # pure eps-DP. For a cell released in both neighbouring datasets both sizes are >= n_min,
        # so removing one record moves the mean by at most 1/(n-1) <= 1/n_min. n_min is public.
        # This is conservative -- a cell of size n pays n/n_min more noise than the old scheme --
        # and it keeps the clean Laplace tail the union bound below relies on.
        b = (1.0 / n_min) / eps_cell                     # Laplace scale, sensitivity <= 1/n_min
        p_hat = float(np.clip(_secure_laplace(p_true, b), 0.0, 1.0))
        m = (sk == cell)
        halfwidth = b * np.log(k / alpha)
        if m.sum() == 0:
            # The synthetic data has NO rows in this cell, so it offers no evidence about p_c and
            # the only honest bound on |p_c - q_c| is the trivial one. Scoring the worst case over
            # COVERED cells only rewarded ignoring cells: a dataset covering 1 of 4 cells scored a
            # TIGHTER bound (0.0549) than the full dataset (0.0727) and would have cleared it.
            q, bound = float("nan"), 1.0
        else:
            q = float(sy[m].mean())
            bound = abs(p_hat - q) + halfwidth
        out.append({"cell": cell, "n": n_c, "p_hat": round(p_hat, 4),
                    "q_synth": None if np.isnan(q) else round(q, 4),
                    "noise_halfwidth": round(halfwidth, 4),
                    "bound": round(bound, 4),
                    "synth_rows": int(m.sum())})
    covered = [c for c in out if c["synth_rows"] > 0]
    thin = [c for c in out if 0 < c["synth_rows"] < MIN_SYNTH_ROWS_PER_CELL]
    # worst case over ALL released cells, uncovered ones included at their trivial bound of 1.0
    worst = max(c["bound"] for c in out)
    return {"level": level, "epsilon_spent": epsilon, "alpha": alpha, "n_cells": k,
            "n_cells_covered": len(covered),
            "n_cells_uncovered": k - len(covered),
            "n_cells_thin": len(thin),
            "worst_case_bound": round(worst, 4),
            "mean_bound": round(float(np.mean([c["bound"] for c in out])), 4),
            "cells": out}


def certify_with_controls(spec, train, test, synth, *, level, epsilon, alpha=0.05,
                          n_min=150, seed=0, tolerance=0.15) -> dict:
    """Bound the synthetic data, plus a real-sample ceiling and a permuted-target floor."""
    rng = np.random.default_rng(seed)
    n = len(synth)
    # The ceiling must be INDEPENDENT of the data `p_hat` is computed from. Drawing it from `train`
    # compares the training set against its own subset, which understates the bound a genuine real
    # sample would achieve — the same tautology as the TRTR ceiling defect in evaluate_generic.
    # `test` is the held-out split and was previously accepted as a parameter and never used.
    real_ctrl = test.sample(min(n, len(test)), random_state=seed)
    perm = real_ctrl.copy()
    perm[spec.target_col] = np.random.RandomState(seed).permutation(
        perm[spec.target_col].values)

    conds = {"synthetic": synth, "real-sample [CEILING]": real_ctrl,
             "permuted-target [FLOOR]": perm}
    res = {}
    for lab, d in conds.items():
        r = certify(spec, train, d, level=level, epsilon=epsilon, alpha=alpha,
                    n_min=n_min, rng=np.random.default_rng(seed))
        # bool(), not the numpy bool the comparison returns: json.dump(default=str) wrote that as
        # the STRING "False", which is truthy to every reader that does not compare it to a literal.
        # A compliance tool reading the report would have taken a failed verdict for a passed one.
        r["within_bound"] = bool(r.get("worst_case_bound") is not None
                                 and r["worst_case_bound"] <= tolerance)
        res[lab] = r

    floor_ok = not res["permuted-target [FLOOR]"]["within_bound"]
    ceil_ok = res["real-sample [CEILING]"]["within_bound"]
    res["_discriminating"] = bool(floor_ok and ceil_ok)
    res["_tolerance"] = tolerance
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--synthetic", nargs="+", required=True)
    ap.add_argument("--level", type=int, default=None, help="conditional level (default: finest)")
    ap.add_argument("--epsilon", type=float, default=0.5,
                    help="budget for the transmission bound (eps_cert)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tolerance", type=float, default=0.15)
    ap.add_argument("--n-min", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    # REQUIRED, with no default. Defaulting to 1 is the silent per-person claim itself: the
    # ledger refuses to assume it, and this report -- the artefact a compliance reader actually
    # reads -- must not assume it either. Making it optional is not enough: an omitted argument
    # produces a report that simply does not mention the privacy unit, and a reader who never sees
    # the field cannot notice it is missing. Forcing the declaration is what puts the question in
    # front of the person running the tool, which is the only place it can be answered.
    ap.add_argument("--max-rows-per-person", type=int, required=True,
                    help="REQUIRED. Worst-case rows contributed by one individual (Diabetes 130: "
                         "40; pass 1 if the data is one row per person). The per-person guarantee "
                         "is this times the row-level epsilon, and this report will not state a "
                         "guarantee without it. Count it on your own data before running this.")
    ap.add_argument("--epsilon-release", type=float, default=2.0,
                    help="what the RELEASE already spent, so the report can state the TOTAL")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = get_spec(a.dataset)
    df = spec.loader()
    train, test = train_test_split(df, test_size=0.2, random_state=a.seed,
                                   stratify=df[spec.target_col])
    paths = sorted({p for g in a.synthetic for p in glob.glob(g)})
    if not paths:
        raise SystemExit(f"no files matched {a.synthetic}")
    synth = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    level = a.level if a.level is not None else len(spec.conditional_levels) - 1

    print("=" * 100)
    print(f"STAGE C — UTILITY TRANSMISSION BOUND | {spec.name} | level {level} | "
          f"eps_cert={a.epsilon} | alpha={a.alpha} | tolerance={a.tolerance}")
    print("=" * 100)
    print("This SPENDS privacy budget: the deployment total is eps_release + "
          f"{a.epsilon}, not eps_release.\n")

    res = certify_with_controls(spec, train, test, synth, level=level, epsilon=a.epsilon,
                                alpha=a.alpha, n_min=a.n_min, seed=a.seed,
                                tolerance=a.tolerance)
    print(f"{'condition':26s} {'cells':>6s} {'cov':>5s} {'uncov':>6s} {'thin':>5s} "
          f"{'mean':>7s} {'worst':>7s} {'within bound':>13s}")
    for lab in ("synthetic", "real-sample [CEILING]", "permuted-target [FLOOR]"):
        r = res[lab]
        print(f"{lab:26s} {r.get('n_cells', 0):6d} {r.get('n_cells_covered', 0):5d} "
              f"{r.get('n_cells_uncovered', 0):6d} {r.get('n_cells_thin', 0):5d} "
              f"{str(r.get('mean_bound')):>7s} {str(r.get('worst_case_bound')):>7s} "
              f"{'YES' if r.get('within_bound') else 'no':>13s}")
    if res["synthetic"].get("n_cells_uncovered"):
        print(f"\n  note: {res['synthetic']['n_cells_uncovered']} released cell(s) have NO synthetic "
              f"rows and are scored at the trivial bound 1.0 — the synthetic data supplies no "
              f"evidence for them.")
    if res["synthetic"].get("n_cells_thin"):
        print(f"  note: {res['synthetic']['n_cells_thin']} cell(s) have fewer than "
              f"{MIN_SYNTH_ROWS_PER_CELL} synthetic rows, so their rate is coarse.")

    if not res["_discriminating"]:
        print("\n  !! THIS TEST DID NOT DISCRIMINATE. A utility bound is only meaningful when the "
              "real-sample ceiling clears the bound AND the permuted-target floor does NOT. Adjust "
              "tolerance or epsilon; do not report the synthetic result from this run.")
    else:
        v = res["synthetic"]
        print(f"\n  Test discriminates (ceiling clears the bound, floor does not).")
        print(f"  VERDICT: synthetic data {'IS' if v['within_bound'] else 'is NOT'} within bound at "
              f"tolerance {a.tolerance} with simultaneous confidence {1 - a.alpha:.0%} "
              f"over {v['n_cells_covered']} cells, at eps_cert={a.epsilon}.")

    # ── the formal bound report, in the form NIST SP 800-226 asks for ───────────────
    total_eps = a.epsilon_release + a.epsilon
    print("\n" + "=" * 100)
    print("DIFFERENTIAL PRIVACY CLAIM (NIST SP 800-226 reporting fields)")
    print("=" * 100)
    for key, val in DP_CLAIM.items():
        print(f"  {key:24s} {val}")
    print(f"  {'epsilon_release':24s} {a.epsilon_release}")
    print(f"  {'epsilon_transmission_bound':26s} {a.epsilon}")
    print(f"  {'EPSILON TOTAL (per row)':24s} {total_eps}")
    k = a.max_rows_per_person                      # required: never None
    if k < 1:
        sys.exit(f"--max-rows-per-person must be >= 1, got {k}")
    eps_person = total_eps * k
    vacuous = eps_person > VACUOUS_EPSILON_PER_PERSON
    if k > 1:
        print(f"  {'max rows per person':24s} {k}")
        print(f"  {'EPSILON PER PERSON':24s} {eps_person}   <- the number a deployment must "
              f"report; group privacy, k*epsilon")
    else:
        print(f"  {'max rows per person':24s} 1 (declared)")
        print(f"  {'EPSILON PER PERSON':24s} {eps_person}   <- equals the per-row epsilon")
    if vacuous:
        print(f"\n  !! EPSILON PER PERSON = {eps_person:.1f} IS VACUOUS (past "
              f"{VACUOUS_EPSILON_PER_PERSON:.0f}). This report bounds utility transmitted under a "
              f"ROW-level guarantee. No per-person privacy claim may be made from it. Aggregate to "
              f"one row per person before Stage A if a per-person guarantee is required.")
    print("\n  WHERE THIS GUARANTEE DOES NOT HOLD:")
    for c in DP_CAVEATS:
        print(f"    - {c}")
    print("\n  STANDARDS ALIGNED, AND WHAT FOR:")
    for name, why in STANDARDS_ALIGNED.items():
        print(f"    - {name}: {why}")
    print("  STANDARDS EXPLICITLY NOT CLAIMED BY THIS REPORT:")
    for name, why in STANDARDS_NOT_CLAIMED.items():
        print(f"    - {name}: {why}")

    # FIRST key in the saved artefact, because a compliance reader opens the file and reads from
    # the top. The report heading says UTILITY TRANSMISSION BOUND; the file must say it too, or the
    # rename stops at the console and the artefact still reads as a privacy certificate.
    res["_what_this_is"] = {
        "artifact": "utility transmission bound",
        "bounds": "how far the synthetic conditional structure can differ from the private one, "
                  "over the released cells, at a stated confidence and a stated epsilon",
        "is_not": "a privacy audit, a re-identification risk assessment, or any form of privacy "
                  "certificate. It does not certify privacy and must not be presented as doing so.",
        "privacy_evidence_lives_elsewhere": "the DP release itself (see _dp_claim) and the measured "
                                            "re-identification evidence, not this bound",
    }
    res["_dp_claim"] = dict(DP_CLAIM)
    res["_dp_claim"].update({"epsilon_release": a.epsilon_release,
                             "epsilon_transmission_bound": a.epsilon,
                             "epsilon_total_per_row": total_eps,
                             "max_rows_per_person": k,
                             "epsilon_per_person": eps_person,
                             "privacy_unit_declared": True,
                             "epsilon_per_person_vacuous": vacuous,
                             "per_person_claim_permitted": not vacuous,
                             "vacuous_threshold": VACUOUS_EPSILON_PER_PERSON})
    res["_dp_caveats"] = DP_CAVEATS
    res["_standards"] = dict(STANDARDS_ALIGNED)
    res["_standards_not_claimed"] = dict(STANDARDS_NOT_CLAIMED)
    if a.out:
        # Provenance. The earlier certificates recorded none, so the inputs that produced a
        # published bound had to be reverse-engineered from cell counts -- and one stored
        # certificate turned out to predate a fix the paper already described, which nothing
        # detected because the audit checked two of its three numbers.
        res["_meta"] = {
            "dataset": a.dataset, "level": level, "epsilon_transmission_bound": a.epsilon,
            "alpha": a.alpha, "tolerance": a.tolerance, "n_min": a.n_min, "seed": a.seed,
            "synthetic_files": paths, "n_synthetic_rows": int(len(synth)),
            "epsilon_release": a.epsilon_release,
            "noise_source": "cryptographically secure, unseeded — this bound is ONE draw "
                            "and re-running will not reproduce these bounds",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
        ordered = {"_what_this_is": res.pop("_what_this_is"), **res}
        json.dump(ordered, open(a.out, "w"), indent=2, default=str)
        print(f"\n  saved -> {a.out}")


if __name__ == "__main__":
    main()
