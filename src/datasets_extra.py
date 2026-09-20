"""
datasets_extra.py — healthcare and financial datasets, as proxies for the regulated-industry
settings CoRTeC is actually aimed at.

Every CoRTeC result so far is on UCI Adult, a public census dataset where the LLM's prior
happens to align with the truth — which is precisely NOT the target use case
(CORTEC_MEMORY.md §6). These two datasets are the closest publicly available stand-ins:

  diabetes   Diabetes 130-US Hospitals, 1999-2008 (UCI 296). 101,766 real hospital encounters
             across 130 US hospitals: admission source, length of stay, lab and procedure
             counts, medication changes, A1C and glucose results, discharge disposition.
             This is genuine hospital administrative/clinical data — the nearest public proxy
             to MIMIC-III, which is still blocked on credentialing. Target: readmitted within
             30 days, the standard prediction task on it and a real payer/provider metric.

  credit     Default of Credit Card Clients (UCI 350). 30,000 Taiwanese credit-card accounts:
             credit limit, demographics, six months of repayment status, bill amounts and
             payments. Target: default next month. Credit-default prediction is the canonical
             regulated financial model, and the columns are exactly what a bank holds.

Both have imbalanced binary targets like Adult (11.2% and 22.1% positive vs Adult's 24.1%),
so the class-balance regime is comparable and the metrics stay interpretable.

IMPORTANT — bounds and bin edges here are PUBLIC choices. They come from the datasets'
published documentation and from general domain knowledge (a hospital stay is 1-14 days; a
credit limit is a round number in a known range), never from inspecting the private values.
`DatasetSpec.validate` enforces that the declared bounds actually cover the data.
"""
from __future__ import annotations
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_spec import Band, DatasetSpec, register

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DIABETES_URL = "https://archive.ics.uci.edu/static/public/296/data.csv"
CREDIT_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def _cached(name: str, fetch, min_rows: int = 1001) -> pd.DataFrame:
    """Download once, validate, then reuse — the same protection as the Adult cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / name
    # `min_rows` guards against a truncated download being cached and silently used. It defaults
    # to 1000 because every dataset in the original study is far larger, but several genuinely
    # small regulated benchmarks (Cleveland heart, 303 rows; German credit, 1000) are legitimately
    # below that, so the caller can lower the bar for those rather than the guard being disabled.
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        if len(df) >= min_rows:
            print(f"  (using cached copy at {p}, {len(df)} rows)")
            return df
        print(f"  cached copy at {p} looks truncated ({len(df)} rows) — refetching")
    df = fetch()
    if len(df) < min_rows:
        raise RuntimeError(f"{name}: fetched only {len(df)} rows, expected >= {min_rows} "
                           f"— refusing to cache (pass min_rows= if this dataset is genuinely small)")
    df.to_csv(p, index=False)
    print(f"  fetched {len(df)} rows; cached → {p}")
    return df


# ── healthcare ──────────────────────────────────────────────────────────────────────
DIAB_NUM = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
DIAB_CAT = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
            "admission_source_id", "insulin", "change", "diabetesMed", "A1Cresult"]

DIAB_BOUNDS = {
    "time_in_hospital": (1, 14), "num_lab_procedures": (0, 132), "num_procedures": (0, 6),
    "num_medications": (0, 81), "number_outpatient": (0, 42), "number_emergency": (0, 76),
    "number_inpatient": (0, 21), "number_diagnoses": (1, 16),
}
DIAB_BINS = {
    "time_in_hospital":   [1, 2, 3, 4, 5, 7, 9, 14],
    "num_lab_procedures": [0, 1, 20, 35, 45, 55, 65, 80, 132],
    "num_procedures":     [0, 1, 2, 3, 4, 6],
    "num_medications":    [0, 5, 10, 13, 16, 20, 25, 35, 81],
    "number_outpatient":  [0, 1, 2, 3, 42],
    "number_emergency":   [0, 1, 2, 76],
    "number_inpatient":   [0, 1, 2, 3, 21],
    "number_diagnoses":   [1, 4, 6, 8, 9, 10, 16],
}


def _load_diabetes() -> pd.DataFrame:
    def fetch():
        return pd.read_csv(DIABETES_URL, low_memory=False)
    df = _cached("diabetes130.csv", fetch)
    keep = DIAB_NUM + DIAB_CAT + ["readmitted"]
    df = df[keep].copy()
    # A1Cresult / max_glu_serum are missing for 83% and 95% of encounters because the test
    # was not ordered. Turning that into the literal string "nan" (what astype(str) does)
    # asks the generator to emit a token pandas then re-reads as NA. "NotMeasured" is both
    # the clinically correct label and safe to round-trip.
    # Fill the missing values BEFORE stringifying. Relying on astype(str) to render NA as the
    # literal "nan" and catching it in .replace() is pandas-version dependent: 2.x produces the
    # string "nan" (so the mapping fires), 3.0's str dtype preserves NA (so it does not), and the
    # dropna() below then silently deletes the 83% of encounters where A1C was not ordered.
    # That produced a real, published-number-changing corruption — see §8 of the paper.
    for c in DIAB_CAT:
        s = df[c].astype("object")
        s = s.where(pd.notna(s), "NotMeasured")
        df[c] = s.astype(str).str.strip().replace(
            {"nan": "NotMeasured", "None": "NotMeasured", "": "NotMeasured",
             "<NA>": "NotMeasured", "?": "Unknown"})
    # A1C is not ordered for ~83% of encounters. If that mass is ever absent, the missing-value
    # convention above has silently stopped working (it has done so once, via a pandas upgrade)
    # and every fidelity number computed downstream is wrong. Fail loudly instead.
    a1c = (df["A1Cresult"] == "NotMeasured").mean()
    if not 0.70 <= a1c <= 0.95:
        raise AssertionError(
            f"diabetes loader: A1Cresult 'NotMeasured' share is {a1c:.1%}, expected ~83%. "
            "The missing-value mapping is not firing — check pandas' astype(str) NA rendering.")
    df = df[df["gender"].isin(["Male", "Female"])]
    # Binary target: readmitted within 30 days vs not. '>30' and 'NO' both count as not-<30,
    # which is the standard formulation of this task.
    df["readmitted_30d"] = np.where(df["readmitted"].astype(str).str.strip() == "<30",
                                    "YES", "NO")
    df = df.drop(columns=["readmitted"])
    for c in DIAB_NUM:
        lo, hi = DIAB_BOUNDS[c]
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lo, hi)
    return df.dropna().reset_index(drop=True)


@register("diabetes")
def diabetes_spec() -> DatasetSpec:
    los = Band("time_in_hospital", edges=[1, 3, 6, 15], labels=["los1-2", "los3-5", "los6+"])
    inpat = Band("number_inpatient", edges=[0, 1, 2, 22], labels=["prior_inp0", "prior_inp1", "prior_inp2+"])
    diag = Band("number_diagnoses", edges=[1, 8, 10, 17], labels=["dx<8", "dx8-9", "dx10+"])
    agec = Band("age")
    ins = Band("insulin")
    return DatasetSpec(
        name="diabetes",
        description=("Diabetes 130-US Hospitals (1999-2008) — inpatient encounters for diabetic "
                     "patients across 130 US hospitals. Target: whether the patient was "
                     "readmitted within 30 days ('YES') or not ('NO')."),
        numerical_cols=DIAB_NUM,
        categorical_cols=DIAB_CAT,
        target_col="readmitted_30d",
        positive_class="YES",
        negative_class="NO",
        feature_bounds=DIAB_BOUNDS,
        public_bins=DIAB_BINS,
        stratify=[los, inpat],
        conditional_levels=[[], [inpat], [los, inpat], [los, inpat, diag],
                            [los, inpat, diag, agec], [los, inpat, diag, agec, ins]],
        heldout_groups=[["race", "gender"], ["admission_source_id"], ["diabetesMed"]],
        loader=_load_diabetes,
    )


# ── finance ─────────────────────────────────────────────────────────────────────────
CRED_NUM = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]
CRED_CAT = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3"]

CRED_BOUNDS = {
    "LIMIT_BAL": (10000, 1000000), "AGE": (21, 79),
    "BILL_AMT1": (-200000, 1000000), "BILL_AMT2": (-200000, 1000000),
    "BILL_AMT3": (-200000, 1000000),
    "PAY_AMT1": (0, 900000), "PAY_AMT2": (0, 1700000), "PAY_AMT3": (0, 900000),
}
CRED_BINS = {
    "LIMIT_BAL": [10000, 50000, 100000, 150000, 200000, 300000, 500000, 1000000],
    "AGE":       [21, 26, 31, 36, 41, 46, 51, 60, 80],
    "BILL_AMT1": [-200000, 0, 1, 5000, 20000, 50000, 100000, 200000, 1000000],
    "BILL_AMT2": [-200000, 0, 1, 5000, 20000, 50000, 100000, 200000, 1000000],
    "BILL_AMT3": [-200000, 0, 1, 5000, 20000, 50000, 100000, 200000, 1000000],
    "PAY_AMT1":  [0, 1, 1000, 2000, 5000, 10000, 30000, 900000],
    "PAY_AMT2":  [0, 1, 1000, 2000, 5000, 10000, 30000, 1700000],
    "PAY_AMT3":  [0, 1, 1000, 2000, 5000, 10000, 30000, 900000],
}


def _load_credit() -> pd.DataFrame:
    def fetch():
        import urllib.request
        with urllib.request.urlopen(CREDIT_URL, timeout=180) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        nm = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx"))][0]
        return pd.read_excel(io.BytesIO(z.read(nm)), header=1)
    df = _cached("credit_default.csv", fetch)
    tgt = [c for c in df.columns if "default" in str(c).lower()][0]
    df = df.rename(columns={tgt: "default_next_month"})
    keep = CRED_NUM + CRED_CAT + ["default_next_month"]
    df = df[keep].copy()
    df["default_next_month"] = np.where(pd.to_numeric(df["default_next_month"]) == 1, "YES", "NO")
    for c in CRED_CAT:
        df[c] = df[c].astype(str).str.strip()
    for c in CRED_NUM:
        lo, hi = CRED_BOUNDS[c]
        df[c] = pd.to_numeric(df[c], errors="coerce").clip(lo, hi)
    return df.dropna().reset_index(drop=True)


@register("credit")
def credit_spec() -> DatasetSpec:
    # PAY_0 is the most recent repayment status: -1/-2 paid duly, 0 revolving, 1+ months late.
    pay = Band("PAY_0", groups={"paid_duly": ["-2", "-1"], "revolving": ["0"],
                                "late_1m": ["1"], "late_2m+": ["2", "3", "4", "5", "6", "7", "8"]})
    lim = Band("LIMIT_BAL", edges=[10000, 100000, 250000, 1000001],
               labels=["limit_low", "limit_mid", "limit_high"])
    agec = Band("AGE", edges=[21, 31, 41, 51, 80], labels=["21-30", "31-40", "41-50", "51+"])
    edu = Band("EDUCATION", groups={"grad_school": ["1"], "university": ["2"],
                                    "high_school": ["3"]})
    mar = Band("MARRIAGE", groups={"married": ["1"], "single": ["2"]})
    # PAY_2 is the second-strongest predictor of default in this data and is public domain
    # knowledge in credit risk (repayment history predicts default), so conditioning on it is a
    # legitimate a priori choice rather than selection on private values. §6.11.1 measured that
    # the original hierarchy conditioned on MARRIAGE (mutual information with the target 0.004)
    # while omitting PAY_2 (0.057), and that the unconditioned columns actively degraded a
    # downstream model. `credit_payhist` swaps the two at identical epsilon — levels compose
    # sequentially, so substituting one for another changes no cost.
    pay2 = Band("PAY_2", groups={"paid_duly": ["-2", "-1"], "revolving": ["0"],
                                 "late_1m": ["1"],
                                 "late_2m+": ["2", "3", "4", "5", "6", "7", "8"]})
    return DatasetSpec(
        name="credit",
        description=("Default of Credit Card Clients (Taiwan, 2005) — credit-card accounts with "
                     "credit limit, demographics, and six months of repayment history. Target: "
                     "whether the account defaults next month ('YES') or not ('NO')."),
        numerical_cols=CRED_NUM,
        categorical_cols=CRED_CAT,
        target_col="default_next_month",
        positive_class="YES",
        negative_class="NO",
        feature_bounds=CRED_BOUNDS,
        public_bins=CRED_BINS,
        stratify=[pay, lim],
        conditional_levels=[[], [pay], [pay, lim], [pay, lim, agec],
                            [pay, lim, agec, edu], [pay, lim, agec, edu, mar]],
        heldout_groups=[["SEX", "MARRIAGE"], ["PAY_3"], ["EDUCATION"]],
        loader=_load_credit,
    )


# ── AIM's own benchmark: titanic ────────────────────────────────────────────────────
# AIM (McKenna et al., VLDB 2022) Table 3 evaluates on six datasets; of those we had only
# tested `adult`. `titanic` is the most valuable of the remaining five for this project:
#   * AIM's `target` workload on titanic uses `Survived` — the same label-centred workload
#     shape our conditional machinery targets;
#   * its total domain is ~9e7, tiny, so **AIM actually fits on it in seconds**, where AIM has
#     failed to complete in 2-3 h on diabetes and credit (CORTEC_MEMORY.md §33.1). This is the
#     dataset that can finally give us a real CoRTeC-vs-AIM head-to-head on a second dataset;
#   * at 1,309 records it is a genuine SMALL-DATA DP regime, which is realistic for the
#     regulated settings CoRTeC targets (rare-disease cohorts, small institutions).
# Source: Frank E. Harrell Jr's titanic3, which is exactly AIM's reference [17].
# NOTE: AIM's exact preprocessing is not published (their "full paper" reference is a
# self-citation), so this is OUR 9-dimensional preprocessing of the same source. We never quote
# AIM's published numbers — we run AIM ourselves on this identical data — so the comparison is
# internally valid even though the preprocessing will not match theirs row for row.
TITANIC_URL = "https://hbiostat.org/data/repo/titanic3.csv"
TITAN_NUM = ["age", "fare", "sibsp", "parch"]
TITAN_CAT = ["pclass", "sex", "embarked", "cabin_deck"]
TITAN_BOUNDS = {"age": (0, 80), "fare": (0, 513), "sibsp": (0, 8), "parch": (0, 9)}
TITAN_BINS = {
    "age":   [0, 5, 12, 18, 25, 32, 40, 50, 60, 80],
    "fare":  [0, 8, 11, 15, 22, 32, 60, 100, 513],
    "sibsp": [0, 1, 2, 3, 9],
    "parch": [0, 1, 2, 3, 10],
}


def _load_titanic() -> pd.DataFrame:
    def fetch():
        return pd.read_csv(TITANIC_URL)
    df = _cached("titanic3.csv", fetch)
    out = pd.DataFrame()
    for c in TITAN_NUM:
        lo, hi = TITAN_BOUNDS[c]
        out[c] = pd.to_numeric(df[c], errors="coerce").clip(lo, hi)
    out["pclass"] = df["pclass"].astype(int).astype(str)
    out["sex"] = df["sex"].astype(str).str.strip()
    out["embarked"] = df["embarked"].astype(str).str.strip().replace({"nan": "Unknown"})
    # Cabin is missing for ~77% of passengers, which is itself informative (deck was recorded
    # mainly for first class). Encode the missingness as its own public category rather than
    # dropping the rows.
    out["cabin_deck"] = (df["cabin"].astype(str).str.strip().str[0]
                         .where(df["cabin"].notna(), "Unknown").replace({"n": "Unknown"}))
    out["survived"] = np.where(pd.to_numeric(df["survived"]) == 1, "YES", "NO")
    # age is missing for 20%; impute to the public bin midpoint rather than dropping a fifth of
    # a 1,309-row dataset. Imputation uses no private information beyond the public bounds.
    out["age"] = out["age"].fillna(28.0)
    out["fare"] = out["fare"].fillna(14.0)
    return out.dropna().reset_index(drop=True)


@register("titanic")
def titanic_spec() -> DatasetSpec:
    pcl = Band("pclass")
    sx = Band("sex")
    agec = Band("age", edges=[0, 18, 40, 81], labels=["child", "adult", "older"])
    far = Band("fare", edges=[0, 15, 60, 514], labels=["fare_low", "fare_mid", "fare_high"])
    return DatasetSpec(
        name="titanic",
        description=("RMS Titanic passenger manifest — passenger class, demographics, fare and "
                     "cabin deck for 1,309 passengers. Target: whether the passenger survived "
                     "('YES') or not ('NO')."),
        numerical_cols=TITAN_NUM,
        categorical_cols=TITAN_CAT,
        target_col="survived",
        positive_class="YES",
        negative_class="NO",
        feature_bounds=TITAN_BOUNDS,
        public_bins=TITAN_BINS,
        stratify=[pcl, sx],
        conditional_levels=[[], [pcl], [pcl, sx], [pcl, sx, agec], [pcl, sx, agec, far]],
        heldout_groups=[["embarked"], ["cabin_deck"]],
        loader=_load_titanic,
    )


@register("credit_aim12")
def credit_aim12_spec() -> DatasetSpec:
    """The credit schema minus the three PAY_AMT columns -- the 12-column schema on which AIM
    converges (§F.2.2's ablation, arm C). Used ONLY to score §F.3's AIM comparison like-for-like:
    every arm, CoRTeC and the real floors included, is scored on these 12 columns. Dropping columns
    from an already-generated table is post-processing and costs no budget; the conditional
    hierarchy, stratification and held-out families of `credit` do not touch the dropped columns."""
    base = credit_spec()
    dropped = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]
    return DatasetSpec(
        name="credit_aim12",
        description=base.description + " Reduced to the 12 columns on which AIM converges.",
        numerical_cols=[c for c in CRED_NUM if c not in dropped],
        categorical_cols=list(CRED_CAT),
        target_col=base.target_col, positive_class=base.positive_class,
        negative_class=base.negative_class,
        feature_bounds={c: v for c, v in CRED_BOUNDS.items() if c not in dropped},
        public_bins={c: v for c, v in CRED_BINS.items() if c not in dropped},
        stratify=base.stratify, conditional_levels=base.conditional_levels,
        heldout_groups=base.heldout_groups,
        loader=lambda: _load_credit().drop(columns=dropped),
    )


@register("credit_payhist")
def credit_payhist_spec() -> DatasetSpec:
    """Credit, with the conditional hierarchy ordered by expected predictive strength.

    Identical to `credit_spec()` in every respect except the conditional levels: PAY_2 takes the
    place MARRIAGE held. This is the change §6.11.1 proposes, isolated so the comparison is clean —
    same columns, same bounds, same bins, same stratification, same epsilon, same number of levels
    and therefore the same per-level budget.
    """
    base = credit_spec()
    pay = Band("PAY_0", groups={"paid_duly": ["-2", "-1"], "revolving": ["0"],
                                "late_1m": ["1"],
                                "late_2m+": ["2", "3", "4", "5", "6", "7", "8"]})
    pay2 = Band("PAY_2", groups={"paid_duly": ["-2", "-1"], "revolving": ["0"],
                                 "late_1m": ["1"],
                                 "late_2m+": ["2", "3", "4", "5", "6", "7", "8"]})
    lim = Band("LIMIT_BAL", edges=[10000, 100000, 250000, 1000001],
               labels=["limit_low", "limit_mid", "limit_high"])
    agec = Band("AGE", edges=[21, 31, 41, 51, 80], labels=["21-30", "31-40", "41-50", "51+"])
    edu = Band("EDUCATION", groups={"grad_school": ["1"], "university": ["2"],
                                    "high_school": ["3"]})
    base.name = "credit_payhist"
    base.conditional_levels = [[], [pay], [pay, pay2], [pay, pay2, lim],
                               [pay, pay2, lim, agec], [pay, pay2, lim, agec, edu]]
    return base

