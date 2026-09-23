"""
data_loader.py
Loads and preprocesses the UCI Adult Income dataset for the CoRTeC viability test.

Two representations are maintained throughout:
    raw_df:     original strings + integers — used for LLM prompts
    encoded_df: all-numeric label-encoded  — used for DP k-means and ML models
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ── Dataset constants ─────────────────────────────────────────────────────────
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

NUMERICAL_COLS = ["age", "fnlwgt", "education_num",
                  "capital_gain", "capital_loss", "hours_per_week"]

CATEGORICAL_COLS = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race", "sex", "native_country"
]

TARGET_COL = "income"

# Explicit bounds for DP mechanisms (from dataset documentation)
FEATURE_BOUNDS = {
    "age":           (17, 90),
    "fnlwgt":        (12285, 1490400),
    "education_num": (1, 16),
    "capital_gain":  (0, 99999),
    "capital_loss":  (0, 4356),
    "hours_per_week":(1, 99),
}

# PUBLIC bin edges for the DP histogram release (Stage A).
#
# Releasing a histogram costs the SAME privacy budget as releasing a single mean: one person
# contributes to exactly one bin, so the L1 sensitivity of the whole histogram is 1, however
# many bins it has. Releasing mean and std separately cost two queries and — as measured on
# 2026-09-04 — produced unusable std values at every realistic cohort size. The histogram
# carries the whole shape instead, and mean/std can be derived from it for free by
# post-processing. See the technical report, section 3 (histograms, not moments).
#
# The edges are chosen from the PUBLIC schema and general domain knowledge (e.g. capital gains
# are zero for most people, so a bin boundary at 1 is worth having), never from the private
# data — choosing them costs no privacy budget.
PUBLIC_BINS = {
    "age":            [17, 25, 30, 35, 40, 45, 50, 55, 60, 70, 90],
    "fnlwgt":         [12285, 50000, 100000, 150000, 200000, 250000, 300000, 400000, 600000, 1490400],
    "education_num":  [1, 5, 9, 10, 11, 12, 13, 14, 15, 16],
    "capital_gain":   [0, 1, 1000, 3000, 5000, 7500, 10000, 20000, 50000, 99999],
    "capital_loss":   [0, 1, 500, 1000, 1500, 2000, 2500, 3000, 4356],
    "hours_per_week": [1, 10, 20, 30, 35, 40, 41, 45, 50, 60, 99],
}

# Human-readable description used in LLM prompts
DATASET_DESCRIPTION = (
    "UCI Adult Income dataset — records of US census respondents. "
    "Target: whether annual income exceeds $50K ('>50K') or not ('<=50K')."
)


ADULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "adult.data"


def _read_adult_csv() -> pd.DataFrame:
    """Read the real Adult data, caching the download locally.

    Every script in the project calls load_adult(), so without a cache each one re-downloads
    ~4 MB from archive.ics.uci.edu. Two concurrent runs were enough to get a truncated
    response (IncompleteRead) on 2026-09-04, which would silently change the reference data
    an experiment is scored against. The cache is written only after the row count is
    validated, so a truncated download is never persisted.
    """
    read_kw = dict(header=None, names=COLUMN_NAMES, na_values=" ?", skipinitialspace=True)

    if ADULT_CACHE.exists():
        df = pd.read_csv(ADULT_CACHE, **read_kw)
        if len(df) >= 32000:
            print(f"  (using cached copy at {ADULT_CACHE}, {len(df)} rows)")
            return df
        print(f"  cached copy at {ADULT_CACHE} looks truncated ({len(df)} rows) — re-downloading")

    last_err = None
    for attempt in range(3):
        try:
            df = pd.read_csv(ADULT_URL, **read_kw)
            if len(df) < 32000:
                raise ValueError(f"download returned only {len(df)} rows — expected ~32561")
            ADULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(ADULT_CACHE, header=False, index=False)
            print(f"  downloaded {len(df)} rows; cached → {ADULT_CACHE}")
            return df
        except Exception as e:                      # network flake / truncated chunked response
            last_err = e
            print(f"  download attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"could not obtain the real Adult dataset after 3 attempts: {last_err}")


def load_adult(test_size: float = 0.20, random_state: int = 42,
               invert_grad_income: bool = False, invert_grad_rate: float | None = None,
               invert_hours_income: bool = False):
    """
    Download and prepare the Adult dataset.

    Parameters
    ----------
    invert_grad_income : if True, run the COUNTERFACTUAL test — force advanced-degree holders
        (education_num >= 14: Masters/Prof-school/Doctorate) to income '<=50K'. This makes the private
        data's education→income relationship *contradict* the LLM's common-sense prior, so we can test
        whether CoRTeC follows the private DP statistics or silently substitutes its own world-knowledge.

    Returns
    -------
    raw_train, raw_test   : pd.DataFrame  (original string values)
    enc_train, enc_test   : pd.DataFrame  (all-numeric, label-encoded)
    label_encoders        : dict          {column → LabelEncoder}  for decoding
    """
    print("Loading UCI Adult Income dataset …")

    df = _read_adult_csv()
    df = df.dropna().reset_index(drop=True)

    # Strip whitespace from string columns
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        df[col] = df[col].str.strip()

    _inv_rng = np.random.RandomState(random_state)
    if invert_grad_income or invert_grad_rate is not None:
        grad = df["education_num"] >= 14                      # Masters / Prof-school / Doctorate
        rate = 0.0 if invert_grad_rate is None else float(invert_grad_rate)
        before = (df.loc[grad, TARGET_COL] == ">50K").mean()
        df.loc[grad, TARGET_COL] = np.where(_inv_rng.random(int(grad.sum())) < rate, ">50K", "<=50K")
        print(f"  [INVERSION grad→{rate:.0%}] set {int(grad.sum())} advanced-degree rows (true rate {before:.0%}) "
              f"— counterintuitive private truth.")
    if invert_hours_income:
        hrs = pd.to_numeric(df["hours_per_week"], errors="coerce") > 40
        before = (df.loc[hrs, TARGET_COL] == ">50K").mean()
        df.loc[hrs, TARGET_COL] = "<=50K"                     # long-hours workers now earn LESS
        print(f"  [INVERSION hours>40→0%] set {int(hrs.sum())} long-hours rows to '<=50K' (true rate {before:.0%}).")

    # ── Encode categoricals ─────────────────────────────────────────────────
    label_encoders = {}
    enc_df = df.copy()

    for col in CATEGORICAL_COLS + [TARGET_COL]:
        le = LabelEncoder()
        enc_df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # ── Train / test split ───────────────────────────────────────────────────
    raw_train, raw_test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COL], random_state=random_state
    )
    enc_train, enc_test = enc_df.loc[raw_train.index], enc_df.loc[raw_test.index]

    print(
        f"  Train: {len(raw_train)} rows | Test: {len(raw_test)} rows | "
        f"Positive rate (>50K): {(df[TARGET_COL] == '>50K').mean():.1%}"
    )

    return (
        raw_train.reset_index(drop=True),
        raw_test.reset_index(drop=True),
        enc_train.reset_index(drop=True),
        enc_test.reset_index(drop=True),
        label_encoders,
    )


def get_X_y(enc_df: pd.DataFrame, label_encoders: dict):
    """Split encoded DataFrame into feature matrix X and binary label vector y."""
    y = enc_df[TARGET_COL].values
    X = enc_df.drop(columns=[TARGET_COL]).values.astype(float)
    return X, y


def get_numerical_X(enc_df: pd.DataFrame) -> np.ndarray:
    """Return only the numerical feature columns as a NumPy array (for DP k-means)."""
    return enc_df[NUMERICAL_COLS].values.astype(float)


def decode_encoded_df(enc_df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """Convert a label-encoded DataFrame back to human-readable strings."""
    raw = enc_df.copy()
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        if col in raw.columns and col in label_encoders:
            raw[col] = label_encoders[col].inverse_transform(raw[col].astype(int))
    return raw
