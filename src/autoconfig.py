"""autoconfig.py — derive a CoRTeC configuration from a schema and a budget, privately.

Every result in this project so far used a hand-tuned configuration: which columns the conditional
table conditions on, how many levels it carries, the cohort stratification, `n_min`. Two problems
follow, and they are the same problem seen from two sides.

**Scientifically**, we chose those columns after inspecting mutual information in the private data
(limitation 4). A deployment cannot legitimately do that, so the reported configuration is not one
an institution could have arrived at.

**Operationally**, it means CoRTeC is not deployable without a statistician in the loop. The finance
utility gap was entirely a configuration defect — the released hierarchy conditioned on MARRIAGE
(MI 0.004) while omitting PAY_2 (0.049) — and nothing in the method warned us.

This module removes both. The institution supplies a schema, a target, an ε and the number of
records it wants; everything else is derived.

**How the column ranking stays private.** We do not compute mutual information on the raw data.
Each candidate column is scored from ONE noised contingency table of (its bands) x target — a
marginal table, not a conditional one. A record occupies exactly one cell of such a table, so its
L1 sensitivity is 1 however many cells it has, the same argument that makes histogram bin count
free (§4.2). Those tables are *not* disjoint across candidates (every record appears in all of
them), so they compose SEQUENTIALLY: m candidates at epsilon/m each. Everything after the noised
counts — the mutual-information computation, the coarsening and the ranking — is post-processing
and free.

**Ranking is by MARGINAL mutual information, and the obvious alternative was tried and reverted.**
Greedy forward selection by CONDITIONAL MI — having chosen a set S, score each remaining candidate
by I(c ; target | S) — was implemented and removed because it made autoconfig WORSE (diabetes TSTR
0.613 -> 0.535). A budget sweep ruled out noise as the cause: on diabetes BOTH arms selected the
same columns and scored identically at eps_sel 0.10 / 0.30 / 0.60, so that gap cannot be noise; on
Adult the greedy arm's selection moved with the budget but every selection it made scored below the
marginal arm at all three. The mechanism is a cardinality bias in the
conditional-MI estimator — conditioning on S shatters the contingency table, and a high-cardinality
candidate scores well simply for having more cells (on Adult it selected `native_country`, 42
levels), after which the richness rule stops early.

Marginal MI has its own redundancy problem — on Adult it picks `marital_status` and `relationship`,
two columns encoding almost the same fact, giving a ceiling of 0.770 against the hand-tuned 0.825.
What fixes that here is not a different objective but explicit COARSENING (`_coarsen_map`,
MAX_LEVELS_PER_COLUMN): wide columns are grouped into at most four bands by their noisy positive
rate, computed from the table already released for the ranking, so it costs nothing further. That
combination reaches parity with expert hand-tuning on all three studied datasets (adult -0.013,
diabetes +0.012, credit -0.005) and is what runs here.

**How richness is chosen without touching the data at all.** §6.6.1 established that a conditional
table's usable resolution is set by how many records land in each cell: too coarse wastes capacity
at large n, too fine fragments at small n and the §8 granularity defect dominates. Cell counts are
a function of the *public* schema cardinalities, and the number of records requested is a public
choice, so the rule needs no private input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Target records per conditional cell. Below ~8 the granularity defect of §8 dominates (a cell
# receiving 0.65 rows can only emit 0 or 1, which is what made the calibration curve saturate);
# above ~40 the table is coarser than the data supports and leaves resolution unused. The Adult
# richness sweep put the optimum at 24 cells for n=300 (12 records/cell) and the healthcare sweep
# was flat from 85 to 3151 cells at n>=1000, both consistent with this band.
MIN_RECORDS_PER_CELL = 8
TARGET_RECORDS_PER_CELL = 20


@dataclass
class AutoConfig:
    """A configuration derived from public schema facts plus one DP selection step."""
    conditional_columns: list[str]
    n_levels: int
    cells_at_finest: int
    records_per_cell: float
    epsilon_selection: float
    epsilon_release: float
    coarsen: dict[str, dict[str, str]] = field(default_factory=dict)
    ranking: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"conditional columns : {self.conditional_columns}",
                 f"levels              : {self.n_levels}  ({self.cells_at_finest} cells at finest)",
                 f"records per cell    : {self.records_per_cell:.1f}",
                 f"epsilon             : {self.epsilon_selection:.3f} selection "
                 f"+ {self.epsilon_release:.3f} release"]
        lines += [f"  ranked: " + ", ".join(f"{c}={s:.4f}" for c, s in self.ranking[:6])]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _band(spec, df: pd.DataFrame, col: str) -> pd.Series:
    """Discretise a column using only PUBLIC information (declared bins / declared levels)."""
    if col in spec.numerical_cols:
        edges = np.asarray(spec.public_bins[col], dtype=float)
        return pd.cut(pd.to_numeric(df[col], errors="coerce"),
                      bins=edges, include_lowest=True).astype(str)
    return df[col].astype(str).str.strip()


def _levels(spec, df: pd.DataFrame, col: str, coarsen: dict | None = None) -> list[str]:
    if coarsen and col in coarsen:
        return sorted(set(coarsen[col].values()))
    if col in spec.numerical_cols:
        return [str(x) for x in pd.IntervalIndex.from_breaks(
            np.asarray(spec.public_bins[col], dtype=float))]
    return sorted(df[col].astype(str).str.strip().unique().tolist())


def _mi_from_noisy_table(tab: np.ndarray) -> float:
    """Mutual information from a (possibly noised) contingency table. Post-processing."""
    tab = np.clip(tab, 0.0, None)                   # counts cannot be negative
    total = tab.sum()
    if total <= 0:
        return 0.0
    p = tab / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p * np.log(np.where(p > 0, p / np.maximum(px * py, 1e-12), 1.0))
    return float(np.nansum(terms))


MAX_LEVELS_PER_COLUMN = 4   # coarsen anything wider; see _coarsen_map
# A column is only ranked if its expected cell count exceeds this multiple of the Laplace scale
# applied to it. Below that the noise dominates the counts and the ranking is meaningless.
NOISE_FLOOR_MULTIPLE = 1.0


def _coarsen_map(tab: np.ndarray, levels: list[str], target_levels: int) -> dict[str, str]:
    """Group a wide column's levels into at most `target_levels` bands, by their noisy rate.

    Hand-tuned specs collapse `education`'s 16 values into four ordered bands, and that coarsening
    is doing real work: with raw levels a two-column table already exhausts the records-per-cell
    budget, so the auto-configurator could never use the three columns the hand-tuned hierarchy
    used. Grouping by positive rate reproduces the same effect without a human choosing the cuts.

    The grouping is computed from the ALREADY-NOISED contingency table released for the ranking, so
    it is post-processing and costs nothing further.
    """
    total = np.clip(tab, 0.0, None).sum(axis=1)
    pos = np.clip(tab, 0.0, None)[:, -1]
    rate = np.divide(pos, np.maximum(total, 1e-9))
    order = np.argsort(rate)                       # ascending by noisy positive rate
    out: dict[str, str] = {}
    per = max(1, int(np.ceil(len(levels) / target_levels)))
    for rank, idx in enumerate(order):
        out[levels[idx]] = f"g{rank // per}"
    return out


class _SecureRNG:
    """Laplace draws from an unseedable source, with numpy's call signature.

    Mirrors `certify._secure_laplace`: uses diffprivlib when present (the same mechanism the
    release path uses) and falls back to `secrets.SystemRandom`. The point is that no caller and
    no recorded constant can reproduce the draw.
    """

    def laplace(self, loc=0.0, scale=1.0, size=None):
        n = 1 if size is None else int(np.prod(size))
        try:
            from diffprivlib.mechanisms import Laplace as _L
            mech = _L(epsilon=1.0, sensitivity=float(scale))
            vals = np.array([mech.randomise(0.0) for _ in range(n)], dtype=float)
        except Exception:
            import math
            import secrets
            sr = secrets.SystemRandom()
            vals = np.empty(n, dtype=float)
            for i in range(n):
                u = sr.random() - 0.5
                vals[i] = -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))
        vals = vals + loc
        return vals[0] if size is None else vals.reshape(size)


def private_column_ranking(spec, train: pd.DataFrame, *, epsilon: float,
                           candidates: list[str] | None = None,
                           seed: int | None = None,
                           insecure_seed: bool = False) -> list[tuple[str, float]]:
    """Rank columns by MARGINAL mutual information with the target, differentially privately.

    History, because the obvious reading of this function is the algorithm we removed. Greedy
    forward selection by CONDITIONAL MI — score each remaining candidate by I(c ; target | S) —
    was implemented and then REVERTED: it made autoconfig worse, not better (diabetes TSTR 0.613
    -> 0.535). A budget sweep showed the cause was not noise but a cardinality bias in the
    conditional-MI estimator: conditioning on S shatters the contingency table, and a
    high-cardinality candidate scores well simply for having more cells. Marginal MI plus explicit
    coarsening (`_coarsen_map`, MAX_LEVELS_PER_COLUMN) handles the redundancy problem better and
    is what runs here.

    Accounting: one contingency table per candidate at epsilon/m, m = number of candidates. A
    record occupies exactly one cell of a table, so L1 sensitivity is 1 regardless of cell count —
    the same argument that makes histogram bin count free (§4.2). The m tables are NOT disjoint
    (every record appears in every one), so they compose SEQUENTIALLY: m x (epsilon/m) = epsilon.
    Everything after the noised counts is post-processing.
    """
    cols = list(candidates or [c for c in spec.column_names if c != spec.target_col])
    m = max(1, len(cols))
    scale = m / float(epsilon)                     # sensitivity 1 at epsilon/m
    # The selection noise MUST NOT be seeded. This step charges eps_sel out of the release budget
    # and its output -- which columns condition the table -- is a function of the private data. With
    # a fixed seed the mechanism is deterministic, so the output distribution is a point mass and
    # the guarantee is vacuous: the budget was spent for nothing. This ran with a hardcoded
    # AUTO_SEED = 0 for the life of the project, which is why every *_auto configuration reproduced
    # exactly. `seed` is retained for callers that need a reproducible *experiment*, and it is
    # honoured only when explicitly opted into via `insecure_seed`, which is never used in a
    # deployment path.
    rng = (np.random.default_rng(seed) if (seed is not None and insecure_seed)
           else _SecureRNG())
    y = train[spec.target_col].astype(str).str.strip()
    y_levels = sorted(y.unique().tolist())

    scored: list[tuple[str, float]] = []
    coarsen: dict[str, dict[str, str]] = {}
    for c in cols:
        x = _band(spec, train, c)
        levels = sorted(x.unique().tolist())
        tab = pd.crosstab(x, y).reindex(index=levels, columns=y_levels,
                                        fill_value=0).to_numpy(dtype=float)
        tab = tab + rng.laplace(0.0, scale, size=tab.shape)
        # Refuse to rank a column whose cells are smaller than the noise added to them. With a
        # few hundred training rows spread over a dozen candidates each table receives a tiny
        # slice of epsilon, the Laplace scale exceeds the counts, and the "ranking" is of noise.
        # On one small dataset that produced a configuration scoring BELOW no conditioning at all
        # (0.419 against a 0.486 base rate) — worse than useless, and silently so.
        expected_cell = len(train) / max(1, len(levels) * len(y_levels))
        if expected_cell < NOISE_FLOOR_MULTIPLE * scale:
            continue
        scored.append((c, _mi_from_noisy_table(tab)))
        if len(levels) > MAX_LEVELS_PER_COLUMN:
            coarsen[c] = _coarsen_map(tab, levels, MAX_LEVELS_PER_COLUMN)
    scored.sort(key=lambda kv: -kv[1])
    private_column_ranking.last_coarsen = coarsen   # consumed by autoconfigure
    return scored


def choose_richness(spec, df: pd.DataFrame, ranked_cols: list[str], n_records: int,
                    coarsen: dict | None = None, n_min: int = 0) -> tuple[int, int]:
    """Pick how many conditioning columns to use, from PUBLIC cardinalities and the requested n.

    Returns (n_levels, cells_at_finest). Adds columns while each cell still expects at least
    MIN_RECORDS_PER_CELL records, preferring the depth closest to TARGET_RECORDS_PER_CELL.

    Two DIFFERENT quantities constrain the table and both must hold:

    * `n_records` — how many rows we intend to GENERATE. Too many cells relative to it and each
      cell gets too few synthetic rows to be worth conditioning on.
    * `n_min` over `len(df)` — how many PRIVATE records a cell must hold to be RELEASED at all.
      The release suppresses any cell below `n_min`, so a table sized only against `n_records`
      silently produces cells that are guaranteed to be suppressed.

    Ignoring the second is why NHANES released 10 of 30 cells and lost 22.6% of its records: the
    richness rule targeted 20 records per cell while the release demanded 150. The rule now
    requires both, so on small data it selects a coarser table that actually survives.
    """
    # Cohorts PARTITION the data, so a conditional table is estimated within a cohort, not over
    # the whole release: each sees roughly n_records / COHORT_GROUPS records. Sizing the table by
    # n_records alone over-provisions by that factor, and on one dataset every cell then fell below
    # the release threshold and the finest table came back EMPTY.
    effective_n = max(1.0, n_records / float(COHORT_GROUPS))
    # Seed `best` with the ACTUAL cell count of the shallowest table, not 1. When the loop breaks
    # on its first iteration the function used to report `cells=1`, which reads as "no conditional
    # structure" in every downstream note and audit — while `n_levels=1` meant one column with its
    # real level count (4 on retinopathy, 2 on heart_cleveland) was in fact being used.
    _first = max(1, len(_levels(spec, df, ranked_cols[0], coarsen))) if ranked_cols else 1
    best = (1, _first, float("inf"))
    cells = 1
    for k, c in enumerate(ranked_cols, start=1):
        cells *= max(1, len(_levels(spec, df, c, coarsen)))
        per_cell = effective_n / cells
        if per_cell < MIN_RECORDS_PER_CELL:
            break
        # release viability: expected PRIVATE records per cell, within a cohort
        if n_min > 0:
            private_per_cell = (len(df) / float(COHORT_GROUPS)) / cells
            if private_per_cell < n_min:
                break
        gap = abs(per_cell - TARGET_RECORDS_PER_CELL)
        if gap < best[2]:
            best = (k, cells, gap)
    return best[0], best[1]


SELECTION_FRACTION_MIN = 0.05
SELECTION_FRACTION_MAX = 0.30


def _selection_fraction(spec, train: pd.DataFrame) -> tuple[float, str]:
    """How much of the budget selection needs, from PUBLIC quantities only.

    A fixed 5% is the wrong rule. Selection is meaningful only while a column's expected cell count
    exceeds the Laplace noise added to it, and that comparison depends on the dataset: with ~550
    training rows over a dozen candidates, 5% left each table eps ~0.008 and the ranking was of
    noise — on one dataset it produced a configuration worse than no conditioning at all.

    Requiring  n / (L * |Y|)  >=  NOISE_FLOOR_MULTIPLE * (m / eps_sel)  and solving for eps_sel:

        eps_sel >= NOISE_FLOOR_MULTIPLE * m * L * |Y| / n

    where m is the candidate count, L a representative level count and |Y| the number of target
    classes. All three come from the declared schema, and n from the dataset size, so the rule
    reads nothing private. Small datasets therefore spend MORE on selection — which is right, since
    a smaller release needs less budget to describe.
    """
    cols = [c for c in spec.column_names if c != spec.target_col]
    m = max(1, len(cols))
    lv = [len(_levels(spec, train, c)) for c in cols] or [2]
    L = float(np.median(lv))
    ny = max(2, train[spec.target_col].nunique())
    need = NOISE_FLOOR_MULTIPLE * m * L * ny / max(1, len(train))
    frac = float(np.clip(need, SELECTION_FRACTION_MIN, SELECTION_FRACTION_MAX))
    why = (f"m={m}, median levels={L:.0f}, |Y|={ny}, n={len(train)} -> selection needs "
           f"eps>={need:.3f} of the total; using {frac:.0%}")
    return frac, why


def autoconfigure(spec, train: pd.DataFrame, *, epsilon_total: float, n_records: int,
                  selection_fraction: float | None = None, seed: int | None = None,
                  insecure_seed: bool = False,
                  n_min: int = 0) -> AutoConfig:
    """Derive a full CoRTeC configuration. The only private step is the column ranking."""
    auto_note = None
    if selection_fraction is None:
        selection_fraction, auto_note = _selection_fraction(spec, train)
    eps_sel = epsilon_total * selection_fraction
    eps_rel = epsilon_total - eps_sel
    ranking = private_column_ranking(spec, train, epsilon=eps_sel, seed=seed,
                                     insecure_seed=insecure_seed)
    coarsen = getattr(private_column_ranking, "last_coarsen", {}) or {}
    ranked_cols = [c for c, _ in ranking]
    n_levels, cells = choose_richness(spec, train, ranked_cols, n_records, coarsen, n_min=n_min)
    chosen = ranked_cols[:n_levels]

    notes = []
    if auto_note:
        notes.append(auto_note)
    if cells and n_records / cells < TARGET_RECORDS_PER_CELL / 2:
        notes.append(f"cells are fine relative to n ({n_records/cells:.1f} records/cell); "
                     f"consider generating more records rather than coarsening")
    if not ranking:
        notes.append("no column had cells large enough to survive the selection noise at this "
                     "epsilon and sample size — conditioning on nothing, which is the safe "
                     "fallback; raise epsilon_selection or supply more records")
    if len(ranking) > 1 and ranking[0][1] <= 0:
        notes.append("no column showed positive mutual information after noising — the target may "
                     "be weakly predictable, or epsilon_selection may be too small")
    return AutoConfig(conditional_columns=chosen,
                      coarsen={c: m for c, m in coarsen.items() if c in chosen},
                      n_levels=n_levels, cells_at_finest=cells,
                      records_per_cell=n_records / max(cells, 1),
                      epsilon_selection=eps_sel, epsilon_release=eps_rel,
                      ranking=ranking, notes=notes)


# ── turning a derived configuration into a runnable spec ────────────────────────────

def bands_for(spec, train: pd.DataFrame, cfg: "AutoConfig") -> list:
    """Convert a derived configuration into the `Band` objects the release pipeline consumes.

    Numerical columns band on their declared public bins. Categorical columns band on the coarsening
    map when one was produced (post-processing of the ranking tables, §48.1), and on their raw
    levels otherwise. Nothing here reads the private data beyond what the ranking already released.
    """
    from src.dataset_spec import Band
    out = []
    for c in cfg.conditional_columns:
        if c in spec.numerical_cols:
            # Outermost edges are widened to +-inf. A declared public bound can still be exceeded
            # by a value the loader clipped to exactly the bound, and pd.cut then returns NaN for
            # it, which surfaced as a spurious cohort literally named "nan".
            edges = [float(x) for x in spec.public_bins[c]]
            edges = [-np.inf] + edges[1:-1] + [np.inf]
            out.append(Band(c, edges=edges,
                            labels=[f"{c}_b{i}" for i in range(len(edges) - 1)]))
        elif c in cfg.coarsen:
            groups: dict[str, list[str]] = {}
            for level, band in cfg.coarsen[c].items():
                groups.setdefault(band, []).append(level)
            out.append(Band(c, groups=groups))
        else:
            levels = sorted(train[c].astype(str).str.strip().unique().tolist())
            out.append(Band(c, groups={v: [v] for v in levels}))
    return out


def apply_to_spec(spec, train: pd.DataFrame, cfg: "AutoConfig"):
    """Return a copy of `spec` whose stratification and conditional levels are the derived ones.

    The cohort grid uses the first band (coarse, so cohorts stay populated) and the conditional
    hierarchy nests the rest, coarse to fine — the same shape the hand-tuned specs declare, but
    derived rather than chosen by a human reading the data.
    """
    import copy
    from src.dataset_spec import Band
    bands = bands_for(spec, train, cfg)
    out = copy.copy(spec)
    out.name = f"{spec.name}_auto"

    # The cohort grid must stay COARSE. Stratifying on a full 8-bin numerical band splits the data
    # into eight cohorts, most of which fall under n_min and are suppressed — on one dataset that
    # left a conditional table with zero cells. Cohorts are a partition of the whole dataset, so
    # each must retain enough records to describe; the conditional table is where resolution
    # belongs, and it is nested underneath.
    out.stratify = [_coarse_band(bands[0], target_groups=COHORT_GROUPS)] if bands else []
    out.conditional_levels = [bands[:i] for i in range(len(bands) + 1)]
    return out


COHORT_GROUPS = 3


def _coarse_band(b, target_groups: int = COHORT_GROUPS):
    """Collapse a Band to at most `target_groups` groups, so cohorts stay populated."""
    from src.dataset_spec import Band
    if b.edges is not None:
        inner = [e for e in b.edges[1:-1]]
        if len(inner) + 1 <= target_groups:
            return b
        keep = [inner[int(round(i * (len(inner) - 1) / (target_groups - 2)))]
                for i in range(target_groups - 1)] if target_groups > 2 else [inner[len(inner) // 2]]
        edges = [b.edges[0]] + sorted(set(keep)) + [b.edges[-1]]
        return Band(b.col, edges=edges, labels=[f"{b.col}_c{i}" for i in range(len(edges) - 1)])
    if b.groups and len(b.groups) > target_groups:
        items = sorted(b.groups.items())
        per = max(1, len(items) // target_groups)
        merged: dict[str, list[str]] = {}
        for i, (_, vals) in enumerate(items):
            merged.setdefault(f"{b.col}_c{min(i // per, target_groups - 1)}", []).extend(vals)
        return Band(b.col, groups=merged)
    return b
