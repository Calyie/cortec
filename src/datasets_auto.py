"""datasets_auto.py — register auto-configured variants of existing datasets.

`run_dataset.py` resolves a spec by name, so a derived configuration has to be registered under a
name before the normal pipeline can generate from it. Each `<name>_auto` spec is the base spec with
its stratification and conditional hierarchy replaced by what `src/autoconfig.py` derived — nothing
else changes, so a comparison against the hand-tuned run isolates the configuration.

The derivation spends privacy budget (`epsilon_selection`) on a query over the private data, so it
CANNOT be re-run casually and it cannot be seeded: a DP mechanism with a fixed public seed is
deterministic, its output distribution is a point mass, and the budget buys nothing. This module
previously derived with a hardcoded `AUTO_SEED = 0`, which is exactly that failure.

The configuration is therefore treated the same way as the DP release itself — **derived once and
persisted as an artefact**. `get_spec("<name>_auto")` loads the stored configuration when one
exists and only derives (and saves) a fresh one when it does not. That keeps a run reproducible
against the configuration it actually used, without pretending the derivation is repeatable.
"""
from __future__ import annotations

import os

from sklearn.model_selection import train_test_split

from src.dataset_spec import DatasetSpec, get_spec, register
import src.datasets_extra      # noqa: F401
import src.datasets_regulated  # noqa: F401
import src.datasets_synthetic  # noqa: F401
import src.datasets_clinical  # noqa: F401
from src.autoconfig import apply_to_spec, autoconfigure

AUTO_SEED = 0
AUTO_EPSILON = 2.0
AUTO_N_RECORDS = 1000


# 0 = the n_min-BLIND richness rule, i.e. size the conditional table against n_records only.
# This looks wrong (it selects tables whose cells the release will suppress — NHANES released 4 of
# 27 cells, 77% coverage) and the n_min-aware alternative was built to "fix" it. The NHANES
# ablation says do NOT adopt that fix as the default: the coarser, fully-covered table lost on
# EVERY clinical metric and was worse than the shuffled-target floor on held-out relationships.
#   richer / partial coverage : acc 0.740  F1 0.303  AUC 0.728  condHeld 0.035
#   coarser / full coverage   : acc 0.450  F1 0.272  AUC 0.711  condHeld 0.059  (floor 0.053)
# (acc/F1 are at an OUT-OF-SAMPLE threshold. An earlier version of this comment quoted 0.624/0.357
#  and 0.538/0.331, which came from a threshold fitted on the same data it was scored on. AUC and
#  condHeld are threshold-free and are unchanged -- which is the tell that only the thresholded
#  metrics moved. The conclusion is the same either way: A leads B on acc, F1 and AUC.)
# Coverage is not the objective; transmitted conditional structure is. Set CORTEC_AUTO_NMIN=150 to
# get the coarser rule for the ablation. See the technical report, Appendix F.1.
AUTO_N_MIN = 0


AUTOCONFIG_DIR = os.environ.get("CORTEC_AUTOCONFIG_DIR", "results/autoconfig")


def _cfg_path(base_name: str, n_min: int) -> str:
    return os.path.join(AUTOCONFIG_DIR, f"{base_name}_nmin{n_min}.json")


def _load_or_derive(base, tr, n_min: int):
    """Load the persisted configuration for this dataset, or derive and persist one.

    Deriving costs privacy budget and, now that the selection noise is unseeded, produces a
    different configuration each time. Persisting makes the configuration an artefact -- the same
    discipline the release follows -- so a result can be tied to the configuration that produced
    it instead of to a seed that never really controlled it.
    """
    import json
    from dataclasses import asdict, fields
    from src.autoconfig import AutoConfig

    path = _cfg_path(base.name, n_min)
    if os.path.exists(path):
        raw = json.load(open(path))
        known = {f.name for f in fields(AutoConfig)}
        cfg = AutoConfig(**{k: v for k, v in raw.items() if k in known})
        cfg.ranking = [tuple(r) for r in cfg.ranking]
        return cfg

    cfg = autoconfigure(base, tr, epsilon_total=AUTO_EPSILON,
                        n_records=AUTO_N_RECORDS, n_min=n_min)
    os.makedirs(AUTOCONFIG_DIR, exist_ok=True)
    json.dump(asdict(cfg), open(path, "w"), indent=2, default=str)
    print(f"  derived and persisted a NEW auto-configuration -> {path} "
          f"(this spent epsilon_selection={cfg.epsilon_selection:.3f} and cannot be reproduced)")
    return cfg


def _make(base_name: str):
    def factory() -> DatasetSpec:
        base = get_spec(base_name)
        df = base.loader()
        tr, _ = train_test_split(df, test_size=0.2, random_state=42,
                                 stratify=df[base.target_col])
        tr = tr.reset_index(drop=True)
        # `n_min` is the release's suppression threshold. Richness must respect it: sizing the
        # conditional table only against n_records produces cells that are guaranteed to be
        # suppressed (NHANES released 10 of 30 cells and lost 22.6% of its records). Overridable
        # by env var so the two configurations can be compared without editing code.
        _n_min = int(os.environ.get("CORTEC_AUTO_NMIN", AUTO_N_MIN))
        cfg = _load_or_derive(base, tr, _n_min)
        spec = apply_to_spec(base, tr, cfg)
        spec.autoconfig = cfg          # carried for reporting; not used by the pipeline
        return spec
    return factory


for _n in ("adult", "diabetes", "credit", "bank_marketing", "retinopathy",
           "renal_registry", "nhanes"):
    register(f"{_n}_auto")(_make(_n))
