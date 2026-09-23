"""
dataset_spec.py — everything that is dataset-specific, in one object.

The pipeline was written against UCI Adult and hard-coded it in several places: the column lists
and bounds in `data_loader`, the education/hours stratification and the income conditional table
of the original Adult release path, the income wording in `prompts`, the label validation in
`llm_generator`, the group families of the original Adult evaluator, and the cell levels in
`hybrid_v2`. Every early CoRTeC result was therefore a result about one dataset, which was the
single largest threat to the claims (technical report, section 10).

A DatasetSpec captures all of it so the same pipeline can run on any tabular dataset with a
binary target. The Adult spec is built from the existing `data_loader` constants, so the
existing pipeline and every published number stay reproducible.

Two fields deserve explanation because they carry the method's assumptions:

  `stratify`   the PUBLIC rule that forms cohorts. Membership must be a deterministic function
               of each individual's own record and must not depend on the private data, or the
               zero-cost claim for cohort formation breaks (technical report, section 4.3).

  `heldout_groups`  group families used to measure conditional fidelity that share NO column
               with `stratify` or with the conditional-table levels. Without this the
               generalisation metric is contaminated and a method that merely parrots its
               released table scores well (technical report, section 6.4).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import pandas as pd


@dataclass
class Band:
    """A public, data-independent grouping of one column into named bands."""
    col: str
    # numerical: ordered edges -> bands [e0,e1), [e1,e2), ...   categorical: {label: [values]}
    edges: list[float] | None = None
    groups: dict[str, list[str]] | None = None
    labels: list[str] | None = None

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if self.edges is not None:
            v = pd.to_numeric(df[self.col], errors="coerce")
            lab = self.labels or [f"{self.edges[i]:g}-{self.edges[i+1]:g}"
                                  for i in range(len(self.edges) - 1)]
            return pd.cut(v, bins=self.edges, labels=lab, include_lowest=True,
                          right=False).astype(str)
        s = df[self.col].astype(str).str.strip()
        if self.groups:
            m = {v: k for k, vals in self.groups.items() for v in vals}
            return s.map(lambda x: m.get(x, "other"))
        return s


@dataclass
class DatasetSpec:
    name: str
    description: str
    numerical_cols: list[str]
    categorical_cols: list[str]
    target_col: str
    positive_class: str
    negative_class: str
    feature_bounds: dict
    public_bins: dict
    stratify: list[Band]
    conditional_levels: list[list[Band]]
    heldout_groups: list[list[str]]
    loader: Callable[[], pd.DataFrame] = field(repr=False, default=None)

    @property
    def column_names(self) -> list[str]:
        return self.numerical_cols + self.categorical_cols + [self.target_col]

    @property
    def feature_cols(self) -> list[str]:
        return self.numerical_cols + self.categorical_cols

    def validate(self, df: pd.DataFrame) -> None:
        """Fail loudly on a mis-specified dataset rather than silently producing nonsense."""
        missing = [c for c in self.column_names if c not in df.columns]
        if missing:
            raise ValueError(f"{self.name}: columns missing from the loaded data: {missing}")
        for c in self.numerical_cols:
            if c not in self.feature_bounds:
                raise ValueError(f"{self.name}: no feature_bounds for numerical column {c}")
            if c not in self.public_bins:
                raise ValueError(f"{self.name}: no public_bins for numerical column {c}")
            lo, hi = self.feature_bounds[c]
            v = pd.to_numeric(df[c], errors="coerce")
            if v.min() < lo or v.max() > hi:
                raise ValueError(
                    f"{self.name}: column {c} has values outside its declared PUBLIC bounds "
                    f"[{lo}, {hi}] (observed [{v.min()}, {v.max()}]). Bounds must be public and "
                    f"must cover the data, or clipping leaks information about the private range.")
        vals = set(df[self.target_col].astype(str).str.strip().unique())
        if not {self.positive_class, self.negative_class} <= vals:
            raise ValueError(f"{self.name}: target {self.target_col} has values {vals}, expected "
                             f"{self.positive_class!r} and {self.negative_class!r}")
        strat_cols = {b.col for b in self.stratify}
        for fam in self.heldout_groups:
            if strat_cols & set(fam):
                raise ValueError(
                    f"{self.name}: held-out group family {fam} shares a column with the "
                    f"stratification {sorted(strat_cols)}. It would not be held out.")

    def stratum_keys(self, df: pd.DataFrame) -> pd.Series:
        # No stratification means ONE cohort covering every record. This is a legitimate,
        # declared configuration — every spec in datasets_regulated.py ships `stratify=[]` and
        # expects autoconfig to supply the hierarchy — but the empty list used to reach `parts[0]`
        # and raise IndexError, so those specs could not be released without autoconfig.
        if not self.stratify:
            return pd.Series(["*"] * len(df), index=df.index)
        parts = [b.apply(df) for b in self.stratify]
        out = parts[0]
        for p in parts[1:]:
            out = out + " & " + p
        return out

    def level_keys(self, df: pd.DataFrame, level: int) -> pd.Series:
        bands = self.conditional_levels[level]
        if not bands:
            return pd.Series(["*"] * len(df), index=df.index)
        parts = [b.apply(df) for b in bands]
        out = parts[0]
        for p in parts[1:]:
            out = out + "|" + p
        return out

    def is_positive(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target_col].astype(str).str.strip() == self.positive_class


# ── registry ────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, Callable[[], DatasetSpec]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_spec(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)


# ── UCI Adult — built from the existing constants so current results stay reproducible ──
@register("adult")
def adult_spec() -> DatasetSpec:
    from src.data_loader import (NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL, FEATURE_BOUNDS,
                                 PUBLIC_BINS, DATASET_DESCRIPTION, _read_adult_csv)
    edu = Band("education_num", edges=[1, 10, 13, 14, 17],
               labels=["edu<=HS", "edu=SomeCol/Assoc", "edu=Bachelors", "edu=Graduate"])
    hrs = Band("hours_per_week", edges=[0, 40, 41, 100], labels=["hours<40", "hours=40", "hours>40"])
    age = Band("age", edges=[17, 31, 41, 51, 91], labels=["17-30", "31-40", "41-50", "51+"])
    mar = Band("marital_status", groups={"Married": ["Married-civ-spouse"]})
    sex = Band("sex")

    def _load():
        df = _read_adult_csv().dropna().reset_index(drop=True)
        for c in CATEGORICAL_COLS + [TARGET_COL]:
            df[c] = df[c].astype(str).str.strip()
        return df

    return DatasetSpec(
        name="adult",
        description=DATASET_DESCRIPTION,
        numerical_cols=list(NUMERICAL_COLS),
        categorical_cols=list(CATEGORICAL_COLS),
        target_col=TARGET_COL,
        positive_class=">50K",
        negative_class="<=50K",
        feature_bounds=dict(FEATURE_BOUNDS),
        public_bins=dict(PUBLIC_BINS),
        stratify=[edu, hrs],
        conditional_levels=[[], [edu], [edu, hrs], [edu, hrs, mar], [edu, hrs, mar, age],
                            [edu, hrs, mar, age, sex]],
        heldout_groups=[["workclass", "race"], ["relationship"], ["native_country"]],
        loader=_load,
    )
