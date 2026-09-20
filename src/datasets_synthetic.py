"""datasets_synthetic.py — a registry with NO public presence, for the contamination control.

Limitation 12 asks whether CoRTeC's results are inherited from the generator having seen UCI Adult,
Diabetes 130 and the Taiwan credit data in pretraining. The paper's existing evidence against that
is the inversion test: CoRTeC emits 0.000 for a group its prior places at 0.737, which recitation
cannot do. That rules out gross memorisation but not recall of the marginals, and the decisive
experiment named in the limitation is a run on data with no public presence.

Obtaining a post-cutoff public dataset is awkward and an obfuscated version of an existing one was
refused by the generator (§8, and the obfuscation route was abandoned). So we construct one. Every
value here comes from a generative process defined in this file: the joint distribution has never
existed anywhere, cannot have been memorised, and its conditional structure is known exactly rather
than estimated.

Design requirements, and why each matters:

  * **Plausible domain and column names.** A schema of opaque codes is what the obfuscation attempt
    used and the generator refused it outright. A renal-registry schema reads as a legitimate
    clinical task, so a refusal cannot be confused with a fidelity result.
  * **Realistic marginals.** Ages, albumin levels and session counts fall in clinically sensible
    ranges, so the model cannot detect that the data is fabricated and treat it differently.
  * **One deliberately counterintuitive relationship.** `transport_mode` is the strongest predictor
    of hospitalisation here, above every clinical variable. That is not what a prior would guess,
    so a generator relying on domain knowledge rather than on the released statistics will get it
    wrong in a measurable direction.

The diagnostic is NOT CoRTeC's absolute score. It is the gap between CoRTeC and the header-only
control, compared against the same gap on Adult. If the prior is doing the work on Adult, the gap
should be markedly wider here, where there is no prior to draw on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset_spec import DatasetSpec, register

N_RECORDS = 20_000
SEED = 20260907

NUM_BOUNDS = {
    "patient_age": (18, 95),
    "sessions_per_week": (1, 7),
    "comorbidity_count": (0, 9),
    "albumin_g_dl": (2.0, 5.5),
    "months_on_dialysis": (0, 240),
}
CAT_LEVELS = {
    "vascular_access": ["fistula", "graft", "catheter"],
    "transport_mode": ["self", "family", "ambulance", "community_van"],
    "facility_size": ["small", "medium", "large"],
    "shift": ["morning", "afternoon", "evening"],
}


def _generate() -> pd.DataFrame:
    """The whole joint distribution, defined here and nowhere else."""
    rng = np.random.default_rng(SEED)
    n = N_RECORDS
    age = np.clip(rng.normal(63, 14, n), *NUM_BOUNDS["patient_age"]).round()
    sessions = rng.choice([2, 3, 4, 5], n, p=[0.10, 0.68, 0.18, 0.04])
    comorb = np.clip(rng.poisson(2.3, n), *NUM_BOUNDS["comorbidity_count"])
    albumin = np.clip(rng.normal(3.8, 0.45, n), *NUM_BOUNDS["albumin_g_dl"]).round(2)
    months = np.clip(rng.gamma(2.0, 18.0, n), *NUM_BOUNDS["months_on_dialysis"]).round()
    access = rng.choice(CAT_LEVELS["vascular_access"], n, p=[0.58, 0.17, 0.25])
    transport = rng.choice(CAT_LEVELS["transport_mode"], n, p=[0.42, 0.28, 0.12, 0.18])
    facility = rng.choice(CAT_LEVELS["facility_size"], n, p=[0.30, 0.45, 0.25])
    shift = rng.choice(CAT_LEVELS["shift"], n, p=[0.45, 0.38, 0.17])

    # Log-odds of hospitalisation within 90 days. `transport_mode` dominates deliberately: it
    # carries more signal than any clinical variable, which is not what domain knowledge predicts.
    lo = (-2.35
          + 1.45 * (transport == "ambulance")
          + 0.85 * (transport == "community_van")
          - 0.40 * (transport == "family")
          + 0.55 * (access == "catheter")
          + 0.18 * (comorb - 2.3)
          - 0.62 * (albumin - 3.8)
          + 0.010 * (age - 63)
          + 0.22 * (facility == "small")
          - 0.15 * (sessions >= 4))
    p = 1.0 / (1.0 + np.exp(-lo))
    y = np.where(rng.random(n) < p, "YES", "NO")

    return pd.DataFrame({
        "patient_age": age.astype(int), "sessions_per_week": sessions.astype(int),
        "comorbidity_count": comorb.astype(int), "albumin_g_dl": albumin,
        "months_on_dialysis": months.astype(int), "vascular_access": access,
        "transport_mode": transport, "facility_size": facility, "shift": shift,
        "hospitalised_90d": y,
    })


@register("renal_registry")
def renal_registry_spec() -> DatasetSpec:
    bins = {
        "patient_age": [18, 45, 55, 65, 75, 95],
        "sessions_per_week": [1, 3, 4, 8],
        "comorbidity_count": [0, 1, 2, 3, 5, 10],
        "albumin_g_dl": [2.0, 3.2, 3.6, 4.0, 4.4, 5.5],
        "months_on_dialysis": [0, 12, 24, 48, 96, 241],
    }
    return DatasetSpec(
        name="renal_registry",
        description=("A regional renal-replacement-therapy registry: patients receiving "
                     "maintenance haemodialysis, with treatment, laboratory and logistic "
                     "attributes. Target: unplanned hospitalisation within 90 days."),
        numerical_cols=list(NUM_BOUNDS), categorical_cols=list(CAT_LEVELS),
        target_col="hospitalised_90d", positive_class="YES", negative_class="NO",
        feature_bounds=dict(NUM_BOUNDS), public_bins=bins,
        stratify=[], conditional_levels=[[]], heldout_groups=[], loader=_generate,
    )
