"""datasets_clinical.py — two genuinely clinical sources, both obtainable without an application.

Limitation 11 recorded MIMIC-III as unavailable: its credentialing (CITI training plus a data-use
agreement) is a real barrier, and Diabetes 130 was carrying the paper's clinical validation alone.
Diabetes 130 is hospital *administrative* data — encounters, billing categories, discharge codes —
so a second source covering a different clinical modality strengthens the claim rather than merely
repeating it.

  nhanes      NHANES 2017-2018 (CDC/NCHS). A national health examination survey: physical
              measurements and laboratory assays on real people, released publicly with no
              application of any kind. Target: HbA1c >= 6.5%, the diagnostic threshold for
              diabetes, which makes this an undiagnosed-diabetes screening task on genuine
              laboratory values rather than on billing codes.

  mimic3_demo MIMIC-III Clinical Database Demo v1.4 (PhysioNet). 100 ICU patients, OPEN access —
              the demo subset carries no credentialing requirement. Far too small to synthesise
              under DP (roughly 80 training records), so it is included as a STRUCTURAL check
              only: it exercises the pipeline against the real MIMIC-III schema and content so the
              paper can say that honestly, rather than claiming a result the sample cannot support.

Bounds and bin edges below are public clinical knowledge — adult age ranges, the WHO BMI
categories, standard blood-pressure bands, the ADA HbA1c threshold — not quantiles of these files.
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
import pandas as pd

from src.dataset_spec import DatasetSpec, register
from src.datasets_extra import _cached

NHANES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
MIMIC_DEMO = "https://physionet.org/files/mimiciii-demo/1.4"


def _xpt(name: str) -> pd.DataFrame:
    req = urllib.request.Request(f"{NHANES}/{name}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return pd.read_sas(io.BytesIO(r.read()), format="xport")


@register("nhanes")
def nhanes_spec() -> DatasetSpec:
    NUM = {"age_years": (18, 80), "bmi": (14.0, 60.0), "waist_cm": (55.0, 175.0),
           "systolic_bp": (70, 220), "diastolic_bp": (30, 130)}
    CAT = ["sex", "race_ethnicity", "education", "income_bracket"]

    SEX = {1: "male", 2: "female"}
    RACE = {1: "mexican_american", 2: "other_hispanic", 3: "white_nh", 4: "black_nh",
            6: "asian_nh", 7: "other_multi"}
    EDU = {1: "lt_9th", 2: "9_11th", 3: "hs_grad", 4: "some_college", 5: "college_grad"}

    def load():
        def fetch():
            d = _xpt("DEMO_J.xpt")[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3",
                                    "DMDEDUC2", "INDFMPIR"]]
            for f, cols in (("GHB_J.xpt", ["SEQN", "LBXGH"]),
                            ("BMX_J.xpt", ["SEQN", "BMXBMI", "BMXWAIST"]),
                            ("BPX_J.xpt", ["SEQN", "BPXSY1", "BPXDI1"])):
                d = d.merge(_xpt(f)[cols], on="SEQN", how="inner")
            return d
        raw = _cached("nhanes_2017.csv", fetch, min_rows=3000).copy()

        df = pd.DataFrame({
            "age_years": pd.to_numeric(raw["RIDAGEYR"], errors="coerce"),
            "bmi": pd.to_numeric(raw["BMXBMI"], errors="coerce"),
            "waist_cm": pd.to_numeric(raw["BMXWAIST"], errors="coerce"),
            "systolic_bp": pd.to_numeric(raw["BPXSY1"], errors="coerce"),
            "diastolic_bp": pd.to_numeric(raw["BPXDI1"], errors="coerce"),
            "sex": pd.to_numeric(raw["RIAGENDR"], errors="coerce").map(SEX),
            "race_ethnicity": pd.to_numeric(raw["RIDRETH3"], errors="coerce").map(RACE),
            "education": pd.to_numeric(raw["DMDEDUC2"], errors="coerce").map(EDU),
        })
        # Family income-to-poverty ratio -> public bands. 7/9 are refused/don't-know codes.
        pir = pd.to_numeric(raw["INDFMPIR"], errors="coerce")
        df["income_bracket"] = pd.cut(pir, [-0.01, 1.0, 2.0, 4.0, 5.1],
                                      labels=["under_1x", "1_2x", "2_4x", "over_4x"]).astype(object)
        # ADA diagnostic threshold: HbA1c >= 6.5% indicates diabetes.
        a1c = pd.to_numeric(raw["LBXGH"], errors="coerce")
        df["diabetes"] = np.where(a1c >= 6.5, "YES", "NO")
        # A participant with NO HbA1c measurement is not a negative — their status is unknown, and
        # `NaN >= 6.5` is False, so the comparison above silently labels them "NO". The final
        # dropna() cannot catch it either, because the column now holds a non-null string. That
        # put 135 fabricated negatives into 3,884 records (3.5%) and diluted the reported base
        # rate from 14.19% to 13.70%. Mark the unmeasured rows missing so they are dropped with
        # every other incomplete case.
        df.loc[a1c.isna(), "diabetes"] = np.nan
        df = df[df["age_years"].between(18, 80)]
        for c in NUM:
            df[c] = df[c].clip(*NUM[c])
        return df.dropna().reset_index(drop=True)

    return DatasetSpec(
        name="nhanes",
        description=("NHANES 2017-2018: a national health examination survey with physical "
                     "measurements and laboratory assays. Target: glycated haemoglobin at or "
                     "above the 6.5% diagnostic threshold for diabetes."),
        numerical_cols=list(NUM), categorical_cols=CAT,
        target_col="diabetes", positive_class="YES", negative_class="NO",
        feature_bounds=dict(NUM),
        public_bins={"age_years": [18, 30, 40, 50, 60, 70, 80],
                     "bmi": [14, 18.5, 25, 30, 35, 40, 60],          # WHO categories
                     "waist_cm": [55, 80, 94, 102, 120, 175],
                     "systolic_bp": [70, 100, 120, 130, 140, 160, 220],   # ACC/AHA stages
                     "diastolic_bp": [30, 60, 80, 90, 100, 130]},
        stratify=[],
        conditional_levels=[[]],
        # Held-out conditional families: relationships the release does NOT carry, used to test
        # whether the generator transmits structure beyond what it was told. These were empty, so
        # `condHeld` printed nan on NHANES — our PRIMARY healthcare dataset — where it reads like a
        # measurement but is an absent one. Chosen from columns autoconfig is unlikely to select
        # (it ranks by mutual information with the target, and these rank low); the evaluator now
        # warns if a family overlaps the released conditional columns.
        # Chosen for SIGNAL, not convenience: a held-out family with no conditional target
        # variation makes the floors coincide and the diagnostic powerless. Measured weighted
        # spread of P(diabetes|level) on the training split: waist_cm 0.208, bmi 0.165,
        # education 0.163 — against sex 0.017 and income_bracket 0.028, which were the first
        # (bad) choice. None of these is selected by autoconfig under either richness rule.
        heldout_groups=[["waist_cm"], ["bmi"], ["education"]],
        loader=load)


@register("mimic3_demo")
def mimic3_demo_spec() -> DatasetSpec:
    """The open-access 100-patient demo. A STRUCTURAL check, not a result: see the module docstring."""
    NUM = {"age_at_admission": (18, 95), "los_days": (0, 60), "n_prior_admissions": (0, 20)}
    CAT = ["admission_type", "admission_location", "insurance", "marital_status", "ethnicity_grp"]

    def load():
        def fetch():
            a = pd.read_csv(f"{MIMIC_DEMO}/ADMISSIONS.csv")
            p = pd.read_csv(f"{MIMIC_DEMO}/PATIENTS.csv")
            return a.merge(p[["subject_id", "gender", "dob"]], on="subject_id", how="left")
        raw = _cached("mimic3_demo.csv", fetch, min_rows=100).copy()
        # MIMIC de-identifies by shifting every date into 2100-2200, and back-dates patients over
        # 89 by ~300 years so their true age cannot be recovered. Subtracting those two directly
        # overflows int64 within pandas' datetime range, so ages are computed on the year fields.
        adm = pd.to_datetime(raw["admittime"], errors="coerce")
        dis = pd.to_datetime(raw["dischtime"], errors="coerce")
        dob = pd.to_datetime(raw["dob"], errors="coerce")
        age = (adm.dt.year - dob.dt.year).astype("float64")
        df = pd.DataFrame({
            # MIMIC shifts dates and codes ages over 89 as ~300; clip to the declared adult range.
            "age_at_admission": age.clip(18, 95),
            "los_days": (dis - adm).dt.total_seconds().div(86400).clip(0, 60),
            "n_prior_admissions": raw.groupby("subject_id").cumcount().clip(0, 20),
            "admission_type": raw["admission_type"].astype(str).str.strip(),
            "admission_location": raw["admission_location"].astype(str).str.strip().str[:24],
            "insurance": raw["insurance"].astype(str).str.strip(),
            "marital_status": raw["marital_status"].fillna("UNKNOWN").astype(str).str.strip(),
            "ethnicity_grp": raw["ethnicity"].astype(str).str.strip().str[:20],
            "died_in_hospital": np.where(pd.to_numeric(raw["hospital_expire_flag"],
                                                       errors="coerce") == 1, "YES", "NO"),
        })
        return df.dropna().reset_index(drop=True)

    return DatasetSpec(
        name="mimic3_demo",
        description=("MIMIC-III Clinical Database Demo: intensive-care admissions with "
                     "administrative and outcome fields. Target: in-hospital mortality."),
        numerical_cols=list(NUM), categorical_cols=CAT,
        target_col="died_in_hospital", positive_class="YES", negative_class="NO",
        feature_bounds=dict(NUM),
        public_bins={"age_at_admission": [18, 40, 55, 65, 75, 95],
                     "los_days": [0, 2, 5, 10, 20, 60],
                     "n_prior_admissions": [0, 1, 2, 4, 21]},
        stratify=[], conditional_levels=[[]], heldout_groups=[], loader=load)
