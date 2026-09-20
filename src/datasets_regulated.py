"""datasets_regulated.py — six additional regulated-domain datasets, declared but NOT tuned.

These exist to test one specific claim: that `src/autoconfig.py` derives a working CoRTeC
configuration from a schema and a budget, without the per-dataset engineering every result in this
paper has needed so far. That claim is only testable on datasets we have not already fitted to, so
each spec below declares **only what a deploying institution would know without looking at its own
data**: the column list, public bounds, and a target.

Deliberately absent from every spec here: any conditional hierarchy, any stratification chosen by
inspecting the data, any `n_min` tuned to the outcome. `conditional_levels` is left as the trivial
[[ ]] and the auto-configurator is expected to supply the rest. If a dataset needs hand-tuning to
work, that is the finding.

Bin edges are quantile-free: they come from the declared public bounds, so they use no private
information. Where a column's plausible range is genuinely public knowledge (age, blood pressure,
a credit line in a known currency) the bounds are set from that knowledge, not from the data.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src.dataset_spec import Band, DatasetSpec, register
from src.datasets_extra import _cached

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"


def _lin(lo: float, hi: float, k: int = 8) -> list[float]:
    return list(np.linspace(float(lo), float(hi), k + 1))


def _spec(name, desc, df_loader, num_bounds, cat_cols, target, pos, neg) -> DatasetSpec:
    """Build a spec with NO conditional hierarchy — autoconfig must supply it."""
    return DatasetSpec(
        name=name, description=desc,
        numerical_cols=list(num_bounds), categorical_cols=list(cat_cols),
        target_col=target, positive_class=pos, negative_class=neg,
        feature_bounds=dict(num_bounds),
        public_bins={c: _lin(*b) for c, b in num_bounds.items()},
        stratify=[], conditional_levels=[[]], heldout_groups=[], loader=df_loader,
    )


# ── health ─────────────────────────────────────────────────────────────────────────

@register("heart_cleveland")
def heart_cleveland_spec() -> DatasetSpec:
    NUM = {"age": (20, 80), "trestbps": (80, 220), "chol": (100, 600),
           "thalach": (60, 220), "oldpeak": (0, 7)}
    CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

    def load():
        cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                "exang", "oldpeak", "slope", "ca", "thal", "num"]
        df = _cached("heart_cleveland.csv",
                     lambda: pd.read_csv(f"{UCI}/heart-disease/processed.cleveland.data",
                                         names=cols, na_values="?"),
                     min_rows=300)
        df = df.dropna(subset=["num"]).copy()
        df["disease"] = np.where(pd.to_numeric(df["num"], errors="coerce") > 0, "YES", "NO")
        for c in CAT:
            df[c] = df[c].astype(str).str.strip()
        for c in NUM:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(*NUM[c])
        return df[list(NUM) + CAT + ["disease"]].dropna().reset_index(drop=True)

    return _spec("heart_cleveland", "Cleveland heart-disease records. Target: presence of "
                 "angiographic heart disease.", load, NUM, CAT, "disease", "YES", "NO")


@register("cervical_cancer")
def cervical_spec() -> DatasetSpec:
    NUM = {"Age": (13, 85), "Number of sexual partners": (0, 30),
           "First sexual intercourse": (10, 40), "Num of pregnancies": (0, 15),
           "Smokes (years)": (0, 40)}
    CAT = ["Smokes", "Hormonal Contraceptives", "IUD", "STDs"]

    def load():
        df = _cached("cervical_cancer.csv",
                     lambda: pd.read_csv(f"{UCI}/00383/risk_factors_cervical_cancer.csv",
                                         na_values="?"),
                     min_rows=850)
        df = df.copy()
        df["biopsy_positive"] = np.where(pd.to_numeric(df["Biopsy"], errors="coerce") > 0,
                                         "YES", "NO")
        for c in CAT:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).astype(str)
        for c in NUM:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(*NUM[c])
        return df[list(NUM) + CAT + ["biopsy_positive"]].dropna().reset_index(drop=True)

    return _spec("cervical_cancer", "Cervical-cancer risk-factor screening records. Target: "
                 "positive biopsy.", load, NUM, CAT, "biopsy_positive", "YES", "NO")


@register("retinopathy")
def retinopathy_spec() -> DatasetSpec:
    NUM = {f"f{i}": (0.0, 1.0) for i in range(3, 9)}
    NUM.update({"f16": (0.0, 1.0), "f17": (0.0, 1.0)})
    CAT = ["f0", "f1", "f18"]

    def load():
        def fetch():
            raw = urllib.request.urlopen(f"{UCI}/00329/messidor_features.arff",
                                         timeout=60).read().decode()
            rows = [l for l in raw.splitlines() if l and not l.startswith(("@", "%"))]
            return pd.DataFrame([r.split(",") for r in rows],
                                columns=[f"f{i}" for i in range(20)])
        df = _cached("retinopathy.csv", fetch).copy()
        df["dr_present"] = np.where(pd.to_numeric(df["f19"], errors="coerce") > 0, "YES", "NO")
        for c in CAT:
            df[c] = df[c].astype(str).str.strip()
        for c in NUM:
            v = pd.to_numeric(df[c], errors="coerce")
            # DATA-DEPENDENT, and the comment here used to claim the opposite. This rescales each
            # column by ITS OWN observed min and max, so the declared bounds of (0, 1) hold only
            # because the data was fitted to them. Extremes are the most disclosive statistic a
            # column has -- a single record determines them -- so a release computed on top of
            # this is not protected by epsilon alone, exactly as `n_min` suppression is not
            # (§4.4). The Messidor features are confidence-level microaneurysm counts with no
            # published range, so there is no public constant to substitute; rather than invent
            # one by looking at the data, we keep the transform and declare it. This is the only
            # loader in the project that does anything data-dependent, and retinopathy is an
            # auxiliary dataset -- it carries no headline result.
            df[c] = ((v - v.min()) / max(v.max() - v.min(), 1e-9)).clip(0, 1)
        return df[list(NUM) + CAT + ["dr_present"]].dropna().reset_index(drop=True)

    return _spec("retinopathy", "Messidor diabetic-retinopathy screening features. Target: "
                 "referable retinopathy present.", load, NUM, CAT, "dr_present", "YES", "NO")


# ── finance ────────────────────────────────────────────────────────────────────────

@register("german_credit")
def german_credit_spec() -> DatasetSpec:
    NUM = {"duration": (4, 72), "credit_amount": (250, 20000), "age": (18, 80),
           "installment_rate": (1, 4), "existing_credits": (1, 4)}
    CAT = ["status", "credit_history", "purpose", "savings", "employment",
           "personal_status", "property", "housing", "job"]

    def load():
        names = ["status", "duration", "credit_history", "purpose", "credit_amount", "savings",
                 "employment", "installment_rate", "personal_status", "debtors", "residence",
                 "property", "age", "other_plans", "housing", "existing_credits", "job",
                 "dependents", "telephone", "foreign", "target"]
        df = _cached("german_credit.csv",
                     lambda: pd.read_csv(f"{UCI}/statlog/german/german.data",
                                         sep=r"\s+", names=names),
                     min_rows=1000)
        df = df.copy()
        df["bad_credit"] = np.where(pd.to_numeric(df["target"], errors="coerce") == 2, "YES", "NO")
        for c in CAT:
            df[c] = df[c].astype(str).str.strip()
        for c in NUM:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(*NUM[c])
        return df[list(NUM) + CAT + ["bad_credit"]].dropna().reset_index(drop=True)

    return _spec("german_credit", "German credit applications. Target: applicant classified as "
                 "bad credit risk.", load, NUM, CAT, "bad_credit", "YES", "NO")


@register("australian_credit")
def australian_spec() -> DatasetSpec:
    NUM = {"A2": (10, 90), "A3": (0, 30), "A7": (0, 30), "A10": (0, 70),
           "A13": (0, 2000), "A14": (0, 100000)}
    CAT = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"]

    def load():
        names = [f"A{i}" for i in range(1, 15)] + ["target"]
        df = _cached("australian_credit.csv",
                     lambda: pd.read_csv(f"{UCI}/statlog/australian/australian.dat",
                                         sep=r"\s+", names=names),
                     min_rows=680)
        df = df.copy()
        df["approved"] = np.where(pd.to_numeric(df["target"], errors="coerce") == 1, "YES", "NO")
        for c in CAT:
            df[c] = df[c].astype(str).str.strip()
        for c in NUM:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(*NUM[c])
        return df[list(NUM) + CAT + ["approved"]].dropna().reset_index(drop=True)

    return _spec("australian_credit", "Australian credit-card applications, anonymised attributes. "
                 "Target: application approved.", load, NUM, CAT, "approved", "YES", "NO")


@register("bank_marketing")
def bank_marketing_spec() -> DatasetSpec:
    NUM = {"age": (18, 95), "balance": (-3000, 100000), "duration": (0, 3000),
           "campaign": (1, 50), "pdays": (-1, 900)}
    CAT = ["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]

    def load():
        def fetch():
            z = zipfile.ZipFile(io.BytesIO(
                urllib.request.urlopen(f"{UCI}/00222/bank.zip", timeout=60).read()))
            with z.open("bank-full.csv") as f:
                return pd.read_csv(f, sep=";")
        df = _cached("bank_marketing.csv", fetch).copy()
        df["subscribed"] = np.where(df["y"].astype(str).str.strip() == "yes", "YES", "NO")
        for c in CAT:
            df[c] = df[c].astype(str).str.strip()
        for c in NUM:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(*NUM[c])
        return df[list(NUM) + CAT + ["subscribed"]].dropna().reset_index(drop=True)

    return _spec("bank_marketing", "Retail-bank direct-marketing contacts. Target: client "
                 "subscribed a term deposit.", load, NUM, CAT, "subscribed", "YES", "NO")
