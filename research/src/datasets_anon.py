"""
datasets_anon.py -- the ANONYMISED-SCHEMA control for the backbone study (2026-10-23).

`adult_anon` is UCI Adult with every column name, every category label, every band label and the dataset
description replaced by neutral tokens (n1..n6, k1..k8, y; k3_v07; n3_b2 ...). Numeric values, public bounds,
public bins, the stratification rule, the conditional levels and the held-out groups are unchanged, so a release
of `adult` maps one-to-one onto a release of `adult_anon` (see research/controls/anonymise_release.py in the published repository) and the
generator sees byte-identical prompt TEMPLATES with the semantics removed. It measures what the language model
adds from world knowledge or from memory of the public Adult table: whatever fidelity survives anonymisation
comes from the released numbers and the method, not from the names.

Mapping rule (deterministic, public): numerical columns in spec order -> n1..n6, categorical -> k1..k8, target ->
y with classes A (negative, '<=50K') and B (positive, '>50K'); each categorical column's category list, sorted,
-> <col>_v01, <col>_v02, ...; numeric band labels -> <col>_b1, <col>_b2, ...; categorical band groups -> <col>_g1
... and 'other' stays 'other' (it is the spec's own token).
"""
from __future__ import annotations
import pandas as pd
from src.dataset_spec import DatasetSpec, Band, register


def _adult():
    from src.dataset_spec import get_spec
    return get_spec("adult")


def anon_maps() -> dict:
    """Every mapping needed to anonymise a release or a prompt, and to map draws back."""
    base = _adult()
    df = base.loader()
    cols = {c: f"n{i + 1}" for i, c in enumerate(base.numerical_cols)}
    cols.update({c: f"k{i + 1}" for i, c in enumerate(base.categorical_cols)})
    cols[base.target_col] = "y"
    cats = {}
    for c in base.categorical_cols:
        vals = sorted(df[c].astype(str).str.strip().unique().tolist())
        cats[c] = {v: f"{cols[c]}_v{i + 1:02d}" for i, v in enumerate(vals)}
    cats[base.target_col] = {base.negative_class: "A", base.positive_class: "B"}
    bands = {}
    for b in [x for lv in base.conditional_levels for x in lv] + base.stratify:
        if b.edges is not None and b.labels:
            bands.update({lab: f"{cols[b.col]}_b{i + 1}" for i, lab in enumerate(b.labels)})
        elif b.groups:
            bands.update({g: f"{cols[b.col]}_g{i + 1}" for i, g in enumerate(b.groups)})
    return {"cols": cols, "cats": cats, "bands": bands,
            "cols_inv": {v: k for k, v in cols.items()},
            "cats_inv": {cols[c]: {v: k for k, v in m.items()} for c, m in cats.items()}}


def _anon_band(b: Band, m: dict) -> Band:
    col = m["cols"][b.col]
    if b.edges is not None:
        labs = [m["bands"][l] for l in b.labels] if b.labels else None
        return Band(col, edges=list(b.edges), labels=labs)
    if b.groups:
        return Band(col, groups={m["bands"][g]: [m["cats"][b.col][v] for v in vs] for g, vs in b.groups.items()})
    return Band(col)


@register("adult_anon")
def adult_anon_spec() -> DatasetSpec:
    base = _adult(); m = anon_maps()
    num = [m["cols"][c] for c in base.numerical_cols]; cat = [m["cols"][c] for c in base.categorical_cols]

    def _load():
        df = base.loader().copy()
        for c in base.categorical_cols + [base.target_col]:
            df[c] = df[c].astype(str).str.strip().map(m["cats"][c])
        return df.rename(columns=m["cols"])

    return DatasetSpec(
        name="adult_anon",
        description=("A tabular dataset with 15 columns: n1 to n6 are numerical, k1 to k8 are categorical codes. "
                     "Target: the binary column y with values 'B' or 'A'."),
        numerical_cols=num, categorical_cols=cat, target_col="y", positive_class="B", negative_class="A",
        feature_bounds={m["cols"][c]: v for c, v in base.feature_bounds.items()},
        public_bins={m["cols"][c]: v for c, v in base.public_bins.items()},
        stratify=[_anon_band(b, m) for b in base.stratify],
        conditional_levels=[[_anon_band(b, m) for b in lv] for lv in base.conditional_levels],
        heldout_groups=[[m["cols"][c] for c in fam] for fam in base.heldout_groups],
        loader=_load,
    )
