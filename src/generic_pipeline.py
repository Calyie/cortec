"""
generic_pipeline.py — Stage A and the prompt builders, driven by a DatasetSpec.

These mirror `dp_cohorts.release_cohort_statistics` and `prompts.build_cortec_prompt` exactly,
including every mechanism decision established on 2026-09-04 (CORTEC_MEMORY.md §13, §17, §23):

  * cohorts come from the spec's PUBLIC stratification rule, so forming them costs eps = 0;
  * queries compose in PARALLEL across the disjoint cohorts, not by dividing by k;
  * each numerical column is released as ONE DP histogram (L1 sensitivity 1 regardless of bin
    count), with mean/std derived from it as free post-processing;
  * the conditional target table is released for each level of the spec, and the levels are NOT
    disjoint from one another, so the conditional budget is SPLIT across the levels released;
  * every released categorical proportion is shown in the prompt, not just the top few — the
    truncation to 4 categories accounted for essentially all of CoRTeC's fidelity gap.

The Adult-specific modules are left untouched so existing Adult results stay reproducible;
`test_generic_matches_adult` in the test suite pins the two paths together.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from src.dataset_spec import DatasetSpec

try:
    from diffprivlib.mechanisms import Laplace
except ImportError:
    from src.dp_primitives import Laplace


# Share of the total budget reserved for publishing per-cohort record counts. Small because a count
# has sensitivity 1 and the numbers are in the hundreds-to-thousands, so even a tight epsilon leaves
# the published size accurate to a fraction of a percent -- but it must be non-zero, because the
# alternative is publishing the exact count and calling the release private.
COUNT_FRAC = 0.02

# Categories whose noisy share lands at or below CATEGORY_FLOOR are kept at a floor rather than
# deleted, sharing FLOOR_TOTAL of the column's mass between them. Capping the total bounds the
# distortion on high-cardinality columns: Adult's native_country has 41 levels with a genuinely
# tiny tail, and giving each a fixed floor would have invented ~8% of spurious mass.
CATEGORY_FLOOR = 0.005
FLOOR_TOTAL = 0.01


def _num_block(props, edges, lo, hi) -> dict:
    """A released numerical histogram with the moments derived from it (post-processing)."""
    props = np.asarray(props, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    m = float((props * centers).sum())
    s = float(np.sqrt(max(0.0, (props * (centers - m) ** 2).sum())))
    return {"dp_mean": round(m, 2), "dp_std": round(s, 2), "bounds": [float(lo), float(hi)],
            "bin_edges": [float(e) for e in edges], "dp_hist": [round(float(p), 4) for p in props]}


def release_statistics(spec: DatasetSpec, raw_train: pd.DataFrame, *, epsilon_stats: float = 2.0,
                       cond_frac: float = 0.2, n_min: int = 150,
                       cond_levels: tuple[int, ...] | None = None,
                       class_conditional: bool = True) -> list[dict]:
    """Stage A for any dataset. Returns one dict per surviving cohort."""
    # Two guards ported from the reference implementation (tools/cortec). Both catch failures that
    # are otherwise SILENT — the release looks populated and is not, and nothing downstream says so.
    if n_min < 50:
        raise ValueError(
            f"n_min={n_min} is below the safe floor of 50. A conditional rate over fewer than ~50 "
            f"records is dominated by its own Laplace noise, so the release looks populated while "
            f"carrying noise. Raise n_min, or coarsen the conditional levels so cells have support.")

    # The per-cohort RECORD COUNT is a private quantity and has to be paid for like any other.
    # It was previously published exact, while the prompt described every released number as
    # "approximate, privacy-perturbed" and the accounting block reported a clean total. That is
    # the same defect the shipped tool was caught with (a real DP violation, not a wrong number),
    # and it survived here because nothing asserted that released quantities actually move when
    # the noise seed moves. Carve the cost OUT of the declared budget rather than adding to it,
    # so the epsilon we report stays the epsilon we spend. Cohorts are disjoint, so all cohort
    # counts together cost eps_counts once under parallel composition, not once per cohort.
    eps_counts = epsilon_stats * COUNT_FRAC
    _eps_rest = epsilon_stats - eps_counts
    eps_marginal = _eps_rest * (1.0 - cond_frac)
    eps_cond = _eps_rest * cond_frac

    n_q = len(spec.numerical_cols) + len(spec.categorical_cols) + 1
    eps_per_query = eps_marginal / n_q                     # parallel composition across cohorts

    # Which conditional levels to release. They overlap each other, so the budget is split.
    levels = tuple(sorted(set(cond_levels if cond_levels is not None
                              else range(len(spec.conditional_levels)))))
    eps_per_level = eps_cond / max(len(levels), 1)

    strata = spec.stratum_keys(raw_train)
    y = spec.is_positive(raw_train)
    out: list[dict] = []
    _n_blocks, _n_class_conditional = 0, 0   # histogram blocks released; cohorts split by class

    for cid, (sname, idx) in enumerate(sorted(strata.groupby(strata).groups.items())):
        n_c = len(idx)
        if n_c < n_min:
            print(f"  cohort {sname!r} too small ({n_c} < n_min={n_min}) — skipping")
            continue
        sub = raw_train.loc[idx]
        # Noised count. n_min gating above uses the TRUE count, which is a data-dependent choice
        # of what to release; that is unchanged here and is accounted for where the release
        # documents it. What must not happen is publishing the true count itself.
        # The published size is clamped at n_min: a cohort is released only when it holds at least
        # n_min records, so any lower published value is impossible under the release's own rule,
        # and clamping is post-processing. Without it, at eps = 0.3 on NHANES the 18-30 band's
        # noised size clipped to 0, it was allocated 1 of 600 rows, and the youngest age band
        # vanished from the output while every per-row check passed.
        _n_pub = int(max(n_min, round(Laplace(epsilon=eps_counts, sensitivity=1.0).randomise(float(n_c)))))
        stats = {"cohort_id": cid, "cohort_name": str(sname), "cohort_size": _n_pub,
                 "numerical": {}, "categorical": {}, "class_balance": {}}

        def _blocks(rows: pd.DataFrame) -> tuple[dict, dict, dict]:
            """One noised histogram per column over `rows`, each at eps_per_query (sensitivity 1).
            Returns (numerical, categorical, floored-categories)."""
            num_b, cat_b, floored_b = {}, {}, {}
            for col in spec.numerical_cols:
                lo, hi = spec.feature_bounds[col]
                edges = np.asarray(spec.public_bins[col], dtype=float)
                vals = pd.to_numeric(rows[col], errors="coerce").fillna(lo).clip(lo, hi).values
                counts, _ = np.histogram(vals, bins=edges)
                mech = Laplace(epsilon=eps_per_query, sensitivity=1.0)
                noisy = np.array([max(0.0, mech.randomise(float(c))) for c in counts])
                tot = noisy.sum()
                props = noisy / tot if tot > 0 else np.full(len(counts), 1.0 / len(counts))
                num_b[col] = _num_block(props, edges, lo, hi)
            for col in spec.categorical_cols:
                cats = sorted(raw_train[col].astype(str).str.strip().unique().tolist())
                true_counts = rows[col].astype(str).str.strip().value_counts()
                mech = Laplace(epsilon=eps_per_query, sensitivity=1.0)
                noisy = {c: max(0.0, mech.randomise(float(true_counts.get(c, 0)))) for c in cats}
                tot = sum(noisy.values())
                props = ({c: v / tot for c, v in noisy.items()} if tot > 0
                         else {c: 1.0 / len(cats) for c in cats})
                # A declared category must never DISAPPEAR from the release. Dropping every category
                # whose noisy share fell below 0.5% erased real subpopulations: measured on the
                # releases behind this project's results, it removed race_ethnicity='other_multi'
                # from one health-survey cohort, race='Amer-Indian-Eskimo' from both Adult releases
                # and race='Asian' from all three hospital releases. A category at 3.7% of a cohort
                # (20 people, Laplace scale 10.2) had an 8.6% chance of erasure on any given run.
                # Once a category is absent from the release, nothing downstream can generate it, and
                # no fidelity or utility metric reports its absence.
                #
                # The category LIST is declared in the schema and is public, so keeping every declared
                # key costs no privacy -- only the proportions are private, and they are already paid
                # for by the histogram query above. Categories whose noisy share lands at or below the
                # floor share a small fixed mass so they stay generatable, rather than being deleted.
                # The total is capped so a high-cardinality column (41 countries, a genuinely tiny
                # tail) is not inflated: the floor costs at most FLOOR_TOTAL of the column's mass
                # however many categories fall into it.
                small = [c for c, v in props.items() if v <= CATEGORY_FLOOR]
                big = {c: v for c, v in props.items() if v > CATEGORY_FLOOR}
                bs = sum(big.values())
                if small and bs > 0:
                    each = FLOOR_TOTAL / len(small)
                    filt = {c: v / bs * (1.0 - FLOOR_TOTAL) for c, v in big.items()}
                    filt.update({c: each for c in small})
                elif bs > 0:
                    filt = {c: v / bs for c, v in big.items()}
                else:
                    filt = {c: 1.0 / len(cats) for c in cats}
                filt = {c: round(v, 5) for c, v in filt.items()}
                if small:
                    floored_b[col] = sorted(small)
                cat_b[col] = dict(sorted(filt.items(), key=lambda x: -x[1]))
            return num_b, cat_b, floored_b

        ys = y.loc[idx]
        cls_mech = Laplace(epsilon=eps_per_query, sensitivity=1.0)
        pos = max(0.0, cls_mech.randomise(float(ys.sum())))
        neg = max(0.0, cls_mech.randomise(float((~ys).sum())))
        t = pos + neg
        stats["class_balance"] = ({spec.positive_class: round(pos / t, 4),
                                   spec.negative_class: round(neg / t, 4)} if t > 0
                                  else {spec.positive_class: 0.5, spec.negative_class: 0.5})

        # CLASS-CONDITIONAL HISTOGRAMS. The pooled cohort histogram tells the generator what the
        # cohort looks like and nothing about how any feature relates to the target outside the
        # conditional hierarchy; the generator fills that in from its prior, and on finance the
        # filled-in columns measurably DEGRADED the downstream model (paper §F.3.1). Partitioning
        # the cohort by the target and releasing one histogram per (cohort, class) hands it the
        # class-conditional marginals a classifier needs, for every column. The two class blocks
        # are disjoint, so they compose in PARALLEL: each spends the same eps_per_query per column
        # the pooled block did, and the pooled histogram is their noisy mixture -- post-processing,
        # released for free. Measured with a naive independent decoder and no model at all, this
        # release reaches the real-sample floor on two of three students where the pooled one
        # trails it by 0.03-0.06 (paper/audit/release_sufficiency.py). Where either class falls
        # below n_min the cohort keeps the pooled block, a suppression decision of the kind the
        # accounting already documents.
        n_pos_true, n_neg_true = int(ys.sum()), int((~ys).sum())
        if class_conditional and n_pos_true >= n_min and n_neg_true >= n_min:
            p_num, p_cat, p_fl = _blocks(sub[ys.values])
            q_num, q_cat, q_fl = _blocks(sub[~ys.values])
            w = pos / t if t > 0 else 0.5
            stats["by_class"] = {spec.positive_class: {"numerical": p_num, "categorical": p_cat},
                                 spec.negative_class: {"numerical": q_num, "categorical": q_cat}}
            for col in spec.numerical_cols:
                lo, hi = spec.feature_bounds[col]
                edges = np.asarray(spec.public_bins[col], dtype=float)
                h = w * np.asarray(p_num[col]["dp_hist"]) + (1.0 - w) * np.asarray(q_num[col]["dp_hist"])
                stats["numerical"][col] = _num_block(h / h.sum() if h.sum() > 0 else h, edges, lo, hi)
            for col in spec.categorical_cols:
                keys = list(dict.fromkeys(list(p_cat[col]) + list(q_cat[col])))
                mix = {c: round(w * p_cat[col].get(c, 0.0) + (1.0 - w) * q_cat[col].get(c, 0.0), 5)
                       for c in keys}
                stats["categorical"][col] = dict(sorted(mix.items(), key=lambda x: -x[1]))
            fl = {c: sorted(set(p_fl.get(c, [])) | set(q_fl.get(c, [])))
                  for c in set(p_fl) | set(q_fl)}
            if fl:
                stats["_floored_categories"] = fl
                # which block each floor placeholder sits in, so an audit can tell a floored
                # (public, constant) share from an unnoised one per class block
                stats["_floored_categories_by_class"] = {spec.positive_class: p_fl,
                                                         spec.negative_class: q_fl}
            _n_blocks += 2
            _n_class_conditional += 1
        else:
            stats["numerical"], stats["categorical"], fl = _blocks(sub)
            if fl:
                stats["_floored_categories"] = fl
            _n_blocks += 1
        out.append(stats)

    if not out:
        raise ValueError(f"{spec.name}: no cohort survived n_min={n_min}; nothing would be "
                         f"released and generation would fall back to an unconditioned prompt.")

    # ── global conditional target tables, one per released level ────────────────────
    cond_tables = {}
    for li in levels:
        keys = spec.level_keys(raw_train, li)
        tbl = {}
        for cell, idx in keys.groupby(keys).groups.items():
            n_sub = len(idx)
            if n_sub < n_min:
                continue
            # The rate is NOT noised directly. A bounded mean over |c| records has sensitivity
            # 1/|c| only when |c| is public; under add/remove-one adjacency |c| differs between
            # neighbouring datasets, so Lap(1/(|c|·ε)) has a DATA-DEPENDENT scale and is not pure
            # ε-DP (two Laplace densities with different scales have an unbounded ratio in one
            # tail). Release two counting queries instead -- positives and cell size -- each of
            # sensitivity exactly 1 under add/remove, at a scale that depends on nothing private,
            # and derive the rate as post-processing. The two queries touch the same records, so
            # they compose SEQUENTIALLY and split eps_per_level; cells within a level are disjoint,
            # so the level still costs eps_per_level in total (parallel across cells).
            eps_half = eps_per_level / 2.0
            pos_noisy = max(0.0, Laplace(epsilon=eps_half, sensitivity=1.0)
                            .randomise(float(y.loc[idx].sum())))
            n_noisy = Laplace(epsilon=eps_half, sensitivity=1.0).randomise(float(n_sub))
            # Algorithm 1, line 10': the denominator is floored at the PUBLIC n_min, not at 1 -- a
            # cell is released only if it holds n_min records, so a noisy count below that floor is
            # noise, and dividing by it would amplify the rate's error for no privacy reason.
            tbl[str(cell)] = round(float(np.clip(pos_noisy / max(n_noisy, float(n_min)), 0.0, 1.0)), 4)
        cond_tables[li] = tbl

    finest = max(levels)
    for s in out:
        s["conditional_target"] = cond_tables[finest]
        s["conditional_target_levels"] = {str(k): v for k, v in cond_tables.items()}
        # The accounting block must enumerate EVERY charged family, or it becomes a statement
        # about what we remembered to charge rather than what we spent.
        s["_eps"] = {"per_query": eps_per_query, "per_cond_level": eps_per_level,
                     # each cell spends its level share on two sensitivity-1 counts (positives,
                     # size) at this epsilon each; their ratio is the released rate
                     "per_cond_cell_query": eps_per_level / 2.0,
                     "levels_released": list(levels), "cohort_count_query": eps_counts,
                     # partition structure of the marginal family: one histogram block per
                     # cohort, or two disjoint class blocks (parallel, same eps per query)
                     "marginal_blocks": _n_blocks, "class_conditional_cohorts": _n_class_conditional,
                     "queries_per_cohort": n_q,
                     "accounted": round(eps_marginal + eps_cond + eps_counts, 10),
                     "total": epsilon_stats}

    _acct = eps_marginal + eps_cond + eps_counts
    assert abs(_acct - epsilon_stats) < 1e-9, (
        f"budget accounting does not close: charged {_acct} against a declared {epsilon_stats}")
    print(f"  {spec.name}: released {len(out)} cohorts | eps/query={eps_per_query:.4f} "
          f"| {len(levels)} conditional levels @ eps={eps_per_level:.4f} each "
          f"| finest table has {len(cond_tables[finest])} cells")
    # A release whose finest conditional table is empty carries NO conditional structure: the
    # generator would produce unconditioned output while every log line still reads normally. This
    # happened on one dataset (autoconfig sized the table by total n while cohorts partition the
    # data) and was caught only by inspecting cell counts by hand.
    # The FINEST table specifically — not the max across levels. A coarse level can be non-empty
    # while the finest one is empty, and it is the finest table the generator conditions on.
    _finest = len(cond_tables.get(finest, {}) or {})
    if _finest == 0:
        raise ValueError(
            "the conditional target table is empty: no cell reached n_min. This table is the "
            "channel that carries conditional structure to the generator, so an empty one means "
            "the output would be unconditioned. Coarsen the conditional levels, lower n_min, or "
            "request more records so cells have support.")
    return out


def build_matched_header_only_prompt(spec: DatasetSpec, cohort_stats: dict,
                                     n_rows: int = 25) -> str:
    """CoRTeC's own prompt with the released statistical arrays deleted and NOTHING else changed.

    §7.1.2's header-only control shares the system prompt, model, decoding, batching, parser and
    repair path with CoRTeC, but its USER prompt is not CoRTeC's minus the arrays: it also drops the
    cohort framing sentence and closes with a different instruction ("realistic ... use varied
    values" against "faithfully reflect the statistics above"). That residual cannot be removed by
    rewording, because an instruction to reflect statistics is incoherent in a prompt with none.

    It CAN be removed by deletion, which is what this builds. The prompt is produced by taking
    `build_cortec_prompt` output and removing exactly the blocks carrying released quantities --
    cohort size, class balance, numerical features, categorical features, conditional target table.
    Every other line, including the framing sentence and the closing instruction, survives
    byte-identically. The dangling references to "the statistics above" are deliberate: keeping them
    is what makes this an ablation of the STATISTICS rather than of the instructions.

    This is the control §7.1.2 names as the one it still lacked.
    """
    full = build_cortec_prompt(spec, cohort_stats, n_rows=n_rows)

    # WHITELIST, not blacklist. A blacklist of block headers fails open: the first attempt here
    # dropped the "NUMERICAL FEATURES" header but kept every numeric distribution under it, because
    # that block's second line is unindented and ended the skip immediately. An ablation that
    # silently retains the arrays it claims to remove is worse than no ablation, so keep only the
    # lines known to carry no released quantity and drop everything else.
    KEEP = ("Dataset:", "You are generating rows", "Column schema (in order):",
            f"Generate {n_rows} synthetic rows", "Output only CSV")
    out = [ln for ln in full.splitlines() if any(ln.startswith(k) for k in KEEP)]

    text = "\n\n".join(out) + "\n"

    # FAIL CLOSED. Every released statistic this pipeline emits is rendered as a percentage or a
    # decimal, so either surviving means a leak. Raise rather than return a contaminated control.
    leaked = re.findall(r"\d+\.\d+|\d+%", text)
    if leaked:
        raise RuntimeError(
            f"matched header-only prompt still carries released quantities {leaked[:5]} — "
            f"refusing to return a contaminated control")
    return text


def build_header_only_prompt(spec: DatasetSpec, n_rows: int = 25) -> str:
    cols = ", ".join(spec.column_names)
    return (f"Dataset: {spec.description}\n\n"
            f"Column schema (in order):\n{cols}\n\n"
            f"Generate {n_rows} realistic synthetic rows that could plausibly appear in this "
            f"dataset. Use varied values. For the '{spec.target_col}' column use exactly "
            f"'{spec.positive_class}' or '{spec.negative_class}'.\n\n"
            f"Output only CSV with header row.")


def _render_feature_blocks(lines: list, num: dict, cat: dict, *, who: str = "this cohort") -> None:
    """Append the NUMERICAL / CATEGORICAL sections for one histogram block."""
    if num:
        lines += [f"NUMERICAL FEATURES — approximate distribution of {who}.",
                  "Match the SHAPE below, not just the average: the percentages are the share of "
                  "rows that must fall in each range."]
        for col, v in num.items():
            lo, hi = v["bounds"]
            lines.append(f"  {col}: mean≈{v['dp_mean']:.1f}, std≈{v['dp_std']:.1f} "
                         f"(valid range {lo:g}–{hi:g})")
            edges, hist = v.get("bin_edges"), v.get("dp_hist")
            if edges and hist:
                parts = [f"{edges[i]:g}–{edges[i+1]:g}: {p:.0%}" for i, p in enumerate(hist) if p >= 0.005]
                if parts:
                    lines.append(f"      distribution → {', '.join(parts)}")
    if cat:
        lines += ["", "CATEGORICAL FEATURES (approximate proportions — reproduce ALL of these, "
                      "including the less common ones, in roughly these shares):"]
        for col, props in cat.items():
            lines.append(f"  {col}: " + ", ".join(f"{c}({p:.1%})" for c, p in props.items()))


def _apportion(probs, n: int, rng) -> np.ndarray:
    """Integer counts summing to n from probabilities, by largest remainder with random ties."""
    p = np.asarray(probs, float); p = p / max(p.sum(), 1e-12)
    raw = p * n; base = np.floor(raw).astype(int); rem = raw - base
    short = int(n - base.sum())
    if short > 0:
        order = np.lexsort((rng.random(len(p)), -rem))
        base[order[:short]] += 1
    return base


def batch_quotas(spec: DatasetSpec, cohort_stats: dict, n_rows: int, rng) -> dict:
    """Exact per-column counts for one batch of `n_rows`, apportioned from the released
    histograms: the number of positives from the class balance, and for every column the number
    of rows per bin (numerical, public edges) or per category. Where the release carries class
    blocks, the positives' counts come from the positive block and the negatives' from the
    negative block, and the batch total per column is their sum. Reads only the release, so
    handing these counts to the generator is post-processing."""
    cb = cohort_stats["class_balance"]
    k = int(_apportion([float(cb.get(spec.positive_class, 0.5)), float(cb.get(spec.negative_class, 0.5))], n_rows, rng)[0])
    by_class = cohort_stats.get("by_class") or {}
    parts = ([(k, by_class[spec.positive_class]), (n_rows - k, by_class[spec.negative_class])]
             if by_class and spec.positive_class in by_class and spec.negative_class in by_class
             else [(n_rows, cohort_stats)])
    quotas = {"positives": k, "n": n_rows, "numerical": {}, "categorical": {}}
    for col in spec.numerical_cols:
        edges = cohort_stats["numerical"][col]["bin_edges"]
        counts = np.zeros(len(edges) - 1, int)
        for m, blk in parts:
            if m > 0:
                counts += _apportion(blk["numerical"][col]["dp_hist"], m, rng)
        quotas["numerical"][col] = [(f"{edges[i]:g}–{edges[i+1]:g}", int(c)) for i, c in enumerate(counts) if c > 0]
    for col in spec.categorical_cols:
        cats = list(cohort_stats["categorical"][col].keys())
        counts = np.zeros(len(cats), int)
        for m, blk in parts:
            if m > 0:
                pr = blk["categorical"][col]
                counts += _apportion([float(pr.get(c, 0.0)) for c in cats], m, rng)
        quotas["categorical"][col] = [(c, int(v)) for c, v in zip(cats, counts) if v > 0]
    return quotas


def cohort_quota_targets(spec: DatasetSpec, cohort_stats: dict, n_rows: int, rng) -> dict:
    """Integer target counts for a WHOLE cohort's n_rows: positives, and per column the rows per
    bin / per category, apportioned once from the release. Batches then request the remaining
    counts, so a batch that returns more or fewer valid rows than asked does not break the
    cohort's totals: the next batch absorbs the difference."""
    q = batch_quotas(spec, cohort_stats, n_rows, rng)
    return {"n": n_rows, "positives": q["positives"],
            "numerical": {c: dict(v) for c, v in q["numerical"].items()},
            "categorical": {c: dict(v) for c, v in q["categorical"].items()}}


def emitted_counts(spec: DatasetSpec, cohort_stats: dict, df: pd.DataFrame) -> dict:
    """What a batch actually contributed, in the same keys as cohort_quota_targets."""
    # a batch can come back without a column the parser could not find; count what is there and
    # let the next batch's remaining counts absorb the rest (a missing target once crashed a run)
    pos = int(spec.is_positive(df).sum()) if spec.target_col in df.columns else 0
    out = {"n": len(df), "positives": pos, "numerical": {}, "categorical": {}}
    for c in spec.numerical_cols:
        if c not in df.columns:
            continue
        edges = cohort_stats["numerical"][c]["bin_edges"]
        e = np.asarray(edges, float); v = pd.to_numeric(df[c], errors="coerce").fillna(e[0]).values
        idx = np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2)
        cnt = np.bincount(idx, minlength=len(e) - 1)
        out["numerical"][c] = {f"{edges[i]:g}–{edges[i+1]:g}": int(k) for i, k in enumerate(cnt) if k}
    for c in spec.categorical_cols:
        if c not in df.columns:
            continue
        out["categorical"][c] = {str(k): int(v) for k, v in df[c].astype(str).str.strip().value_counts().items()}
    return out


def remaining_quotas(spec: DatasetSpec, target: dict, done: dict, b: int, rng) -> dict:
    """Per-batch quotas for the next b rows, apportioned from what the cohort still owes."""
    def _rem(t: dict, d: dict) -> dict:
        return {k: max(int(v) - int(d.get(k, 0)), 0) for k, v in t.items()}
    def _scale(rem: dict) -> list:
        tot = sum(rem.values())
        if tot <= 0 or b <= 0:
            return []
        keys = list(rem); counts = _apportion([rem[k] for k in keys], b, rng)
        return [(k, int(c)) for k, c in zip(keys, counts) if c > 0]
    pos_rem = max(target["positives"] - done["positives"], 0); rows_rem = max(target["n"] - done["n"], 0)
    k = int(round(b * pos_rem / rows_rem)) if rows_rem > 0 else 0
    q = {"n": b, "positives": min(max(k, 0), b), "numerical": {}, "categorical": {}}
    for c, t in target["numerical"].items():
        q["numerical"][c] = _scale(_rem(t, done["numerical"].get(c, {})))
    for c, t in target["categorical"].items():
        q["categorical"][c] = _scale(_rem(t, done["categorical"].get(c, {})))
    return q


def build_cortec_prompt(spec: DatasetSpec, cohort_stats: dict, n_rows: int = 25, quotas: dict | None = None) -> str:
    num, cat = cohort_stats["numerical"], cohort_stats["categorical"]
    lines = [
        f"Dataset: {spec.description}", "",
        f"You are generating rows for a specific subpopulation "
        f"(cohort: {cohort_stats.get('cohort_name', cohort_stats['cohort_id'])}) identified by a "
        f"privacy-preserving rule. The statistics below describe this cohort (approximate, "
        f"privacy-perturbed — treat them as distributional targets, not exact values).", "",
        f"Cohort size (approximate): {cohort_stats['cohort_size']} individuals", "",
        "CLASS BALANCE:",
    ]
    for cls, p in cohort_stats["class_balance"].items():
        lines.append(f"  {spec.target_col}={cls}: {p:.1%}")

    by_class = cohort_stats.get("by_class") or {}
    if by_class:
        # The release carries one histogram block per outcome class. Telling the model what
        # positives and negatives each look like is what lets it reproduce the feature->target
        # structure for EVERY column, instead of inventing it from its prior for the columns the
        # conditional hierarchy does not name (paper §F.3.1: on finance those invented columns
        # measurably degraded the downstream model).
        lines += ["", "FEATURE DISTRIBUTIONS BY OUTCOME. Rows with different outcomes have DIFFERENT "
                      "feature distributions in this cohort. First decide each row's "
                      f"'{spec.target_col}' (class balance above, and the TARGET LIKELIHOOD table "
                      "below), then draw ALL of its other columns from the distributions given for "
                      "THAT outcome."]
        for cls, blk in by_class.items():
            lines += ["", f"=== rows with {spec.target_col} = {cls} ==="]
            _render_feature_blocks(lines, blk.get("numerical", {}), blk.get("categorical", {}),
                                   who=f"rows with {spec.target_col} = {cls}")
    else:
        lines.append("")
        _render_feature_blocks(lines, num, cat)

    cond = cohort_stats.get("conditional_target", {})
    if cond:
        lines += ["", f"TARGET LIKELIHOOD BY GROUP (from the private data, privacy-preserving). "
                      f"These are the GROUND TRUTH for this dataset — set each row's "
                      f"'{spec.target_col}' to match the rate for that row's group. This dataset "
                      f"may DIFFER from general expectations; follow THESE numbers, not any "
                      f"outside assumption:"]
        for g, r in cond.items():
            lines.append(f"  P({spec.target_col}='{spec.positive_class}' | {g}) ≈ {r:.0%}")

    if quotas:
        # Exact counts turn "match these shares" into a counting task the model can satisfy
        # exactly, the same move that made conditional rates exact cell-wise (§3.3). The counts
        # are apportioned from the released histograms for THIS batch, so they carry no more
        # than the release does.
        lines += ["", f"EXACT COUNTS FOR THIS BATCH OF {quotas['n']} ROWS. These are not targets; "
                      f"they are requirements. Every column's counts below sum to {quotas['n']}. "
                      f"Plan the {quotas['n']} rows so that EVERY line is satisfied exactly, then "
                      f"write them out:",
                  f"  {spec.target_col}: exactly {quotas['positives']} rows '{spec.positive_class}' "
                  f"and {quotas['n'] - quotas['positives']} rows '{spec.negative_class}'"]
        for col, items in quotas["numerical"].items():
            lines.append(f"  {col}: " + ", ".join(f"{b}: {c} rows" for b, c in items))
        for col, items in quotas["categorical"].items():
            lines.append(f"  {col}: " + ", ".join(f"{c}: {v}" for c, v in items))
        lines += ["", f"Column schema (in order): {', '.join(spec.column_names)}", "",
                  f"Generate exactly {n_rows} synthetic rows that satisfy EVERY count above, with "
                  f"realistic correlations between columns and with each row's "
                  f"'{spec.target_col}' assigned so that the TARGET LIKELIHOOD table holds within "
                  f"each group as far as the exact counts allow. Do NOT use general world-knowledge "
                  f"about this outcome; use only the numbers given above.", "",
                  "Output only CSV with header row."]
        return "\n".join(lines)
    lines += ["", f"Column schema (in order): {', '.join(spec.column_names)}", "",
              f"Generate {n_rows} synthetic rows whose distributions faithfully reflect the "
              f"statistics above, including correlations between columns. Set the "
              f"'{spec.target_col}' column ('{spec.positive_class}' or "
              f"'{spec.negative_class}') to MATCH the TARGET LIKELIHOOD table for each row's "
              f"group — even if it seems counterintuitive. Do NOT use general world-knowledge "
              f"about this outcome; use only the rates given above.", "",
              "Output only CSV with header row."]
    return "\n".join(lines)



def _readable_cell(cell_key: str, cond_bands: tuple = (), cohort_stats: dict | None = None) -> str:
    """Turn a cell key into something a model can actually act on.

    Cell keys are built from band LABELS — 'patient_age_b0|ambulance'. A label is an internal
    identifier: nothing in the prompt told the model that `patient_age_b0` means 18-45, so it was
    being asked to condition on a token it could not interpret, and the emitted age distribution
    drifted accordingly. Translate each part back to the constraint it encodes.
    """
    parts = [p.strip() for p in str(cell_key).split("|") if p.strip()]
    out = []
    for i, part in enumerate(parts):
        b = cond_bands[i] if i < len(cond_bands) else None
        col = getattr(b, "col", None)
        edges = getattr(b, "edges", None) if b is not None else None
        labels = getattr(b, "labels", None) if b is not None else None
        groups = getattr(b, "groups", None) if b is not None else None
        if col and groups and part in groups:
            vals = list(groups[part])
            if len(vals) == 1:
                out.append(f"{col} = {vals[0]}")
            else:
                # Naming the members without their frequencies pins WHICH values are legal and
                # says nothing about how often each occurs, and the model then spreads them
                # roughly uniformly. Measured on a HIPAA discharge-disposition column coarsened
                # into 7-value groups: the dominant category fell from a real 0.592 to 0.143 and
                # the column's total variation went 0.050 -> 0.641, while cohort-wise generation,
                # which receives the full distribution, reproduced it at 0.600. The within-group
                # shares are a renormalisation of proportions already in the release, so supplying
                # them is post-processing and costs no privacy budget.
                shares = ((cohort_stats or {}).get("categorical") or {}).get(col) or {}
                sub = [(v, float(shares.get(str(v), shares.get(v, 0.0)))) for v in vals]
                tot = sum(x for _, x in sub)
                if tot > 0:
                    sub = sorted(((v, x / tot) for v, x in sub), key=lambda t: -t[1])
                    pretty = ", ".join(f"{v} {p:.0%}" for v, p in sub if p >= 0.005)
                    out.append(f"{col} is one of: {pretty} "
                               f"(these percentages are the shares WITHIN this group — match them)")
                else:
                    out.append(f"{col} is one of: {', '.join(map(str, vals))}")
        elif col and edges and labels and part in labels:
            j = list(labels).index(part)
            lo, hi = edges[j], edges[j + 1]
            if lo == float("-inf"):
                out.append(f"{col} is under {hi:g}")
            elif hi == float("inf"):
                out.append(f"{col} is {lo:g} or above")
            else:
                out.append(f"{col} is between {lo:g} and {hi:g}")
        elif col:
            out.append(f"{col} = {part}")
        else:
            out.append(part)
    return " and ".join(out) if out else str(cell_key)


def build_cell_prompt(spec, cohort_stats: dict, cell_key: str, n_rows: int,
                      n_positive: int, cond_cols: tuple[str, ...] = (),
                      cond_bands: tuple = ()) -> str:
    """Prompt for ONE conditional cell, stating an exact positive COUNT rather than a rate.

    Asking a model to hit a *rate* saturates: §8's calibration defect showed released rates at or
    above ~0.24 pulled toward 1.0, because a cell receiving a couple of rows can only emit 0 or 1
    and the model resolves the ambiguity by rounding up. Asking for "exactly k of n records" removes
    the ambiguity — the quantity requested is one the model can satisfy exactly — and moved
    magnitude error from 0.253 to 0.037 in the reference implementation.

    The count itself is produced by stochastic rounding upstream, so its expectation equals the
    released rate and the error is bounded by one row per cell rather than accumulating.
    """
    n_negative = int(n_rows) - int(n_positive)
    readable = _readable_cell(cell_key, cond_bands, cohort_stats)
    lines = [f"Dataset: {spec.description}", "",
             f"Generate exactly {n_rows} synthetic records, all of which belong to this group.",
             "", "GROUP (every record must match this exactly):", f"  {readable}", "",
             "OUTCOME REQUIREMENT — this is a hard count, not a probability. Follow it exactly.",
             f"Of the {n_rows} records, EXACTLY {n_positive} must have "
             f"{spec.target_col} = \"{spec.positive_class}\", and the remaining {n_negative} must "
             f"have {spec.target_col} = \"{spec.negative_class}\".",
             "This count was measured on the real data for this group under differential privacy.",
             "It may contradict what you would expect; generate the counts as stated regardless.",
             ""]

    # A column named in the GROUP is already pinned for every record in this cell. Also printing
    # its cohort-wide distribution contradicts the GROUP outright — an ambulance-only cell was
    # being told "transport_mode: self 43%, family 29%, ambulance 13%" in the same prompt.
    _fixed = {c for c in cond_cols if c}
    by_class = cohort_stats.get("by_class") or {}
    if by_class:
        # per-outcome blocks: the k positives follow the positive-class distributions, the
        # n-k negatives the negative-class ones (see build_cortec_prompt for why)
        lines += ["FEATURE DISTRIBUTIONS BY OUTCOME — the records with different outcomes have "
                  "DIFFERENT feature distributions. Draw each record's other columns from the "
                  "distributions of ITS outcome.", ""]
        for cls, blk in by_class.items():
            n_cls = n_positive if cls == spec.positive_class else n_negative
            lines.append(f"=== the {n_cls} records with {spec.target_col} = \"{cls}\" ===")
            _render_feature_blocks(
                lines,
                {k: v for k, v in (blk.get("numerical", {}) or {}).items() if k not in _fixed},
                {k: v for k, v in (blk.get("categorical", {}) or {}).items() if k not in _fixed},
                who=f"records with {spec.target_col} = {cls}")
            lines.append("")
    num = {k: v for k, v in (cohort_stats.get("numerical", {}) or {}).items() if k not in _fixed}
    if num and not by_class:
        # The released DP histogram is the channel that carries SHAPE. An earlier version of this
        # prompt sent only mean and std; the model then emitted a smooth continuous spread, giving
        # 90% non-integer ages against integer-valued real data and a worse 1-way TV than the
        # rate-based path it replaced. Mean and std do not determine a distribution — send the bins.
        lines += ["NUMERICAL FEATURES — approximate distribution for this cohort.",
                  "Match the SHAPE below, not just the average: the percentages are the share of "
                  "rows that must fall in each range."]
        for col, st in num.items():
            lo, hi = st.get("bounds", (None, None))
            lines.append(f"  {col}: mean≈{st.get('dp_mean')}, std≈{st.get('dp_std')} "
                         f"(valid range {lo:g}–{hi:g})" if isinstance(lo, (int, float))
                         else f"  {col}: mean≈{st.get('dp_mean')}, std≈{st.get('dp_std')}")
            edges, hist = st.get("bin_edges"), st.get("dp_hist")
            if edges and hist:
                parts = [f"{edges[i]:g}–{edges[i+1]:g}: {p:.0%}"
                         for i, p in enumerate(hist) if p >= 0.005]
                if parts:
                    lines.append(f"      distribution → {', '.join(parts)}")
        lines.append("")
    cat = {k: v for k, v in (cohort_stats.get("categorical", {}) or {}).items()
           if k not in _fixed}
    if cat and not by_class:
        lines.append("CATEGORICAL FEATURES (approximate proportions — reproduce ALL of these):")
        for col, dist in cat.items():
            parts = ", ".join(f"{k}: {v:.0%}" for k, v in list(dist.items())[:12])
            lines.append(f"  {col}: {parts}")
        lines.append("")

    lines += [f"Column schema (in order): {', '.join(spec.column_names)}", "",
              "Rules:",
              "  - every record must match the GROUP above",
              f"  - exactly {n_positive} records have {spec.target_col} = "
              f"\"{spec.positive_class}\"; no more, no fewer",
              "  - every numeric value must lie within its stated range, and must follow the "
              "per-range distribution given above",
              "  - use the same granularity the ranges imply: if a column's ranges are whole "
              "numbers, emit whole numbers, not decimals",
              "  - do not repeat identical rows; vary records the way real data varies",
              f"  - output nothing except the header and the {n_rows} data rows", "",
              "Output only CSV with a header row."]
    return "\n".join(lines)


# ── pool-and-rake selection: post-processing that returns a pool to the release ──────────
MIN_EXPECTED_ROWS = 1.0   # a released cell expecting fewer rows than this cannot be met by selection


def _selection_cells(spec: DatasetSpec, df: pd.DataFrame, release: list) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["_cohort"] = spec.stratum_keys(df).values
    out["_pos"] = spec.is_positive(df).values
    for c in spec.numerical_cols:
        edges = np.asarray(release[0]["numerical"][c]["bin_edges"], float)
        v = pd.to_numeric(df[c], errors="coerce").fillna(edges[0]).values
        out[c] = np.clip(np.digitize(v, edges[1:-1], right=False), 0, len(edges) - 2)
    for c in spec.categorical_cols:
        out[c] = df[c].astype(str).str.strip().values
    return out


def selection_weights(spec: DatasetSpec, release: list, pool: pd.DataFrame, n: int, *,
                      rounds: int = 40, min_expected_rows: float = MIN_EXPECTED_ROWS) -> np.ndarray:
    """Inclusion weights in [0, 1], summing to n, whose sums over every released cell match the
    release's expected counts at size n: cohort mass x class share x bin probability, per
    (cohort, class, column) where the release carries class blocks and per (cohort, column) where
    it carries a pooled block. Iterative proportional fitting with the weights capped at 1.
    Reads only the release and the pool: post-processing, no privacy cost."""
    cells = _selection_cells(spec, pool, release)
    total = sum(float(c["cohort_size"]) for c in release) or 1.0
    w = np.full(len(pool), min(1.0, n / max(len(pool), 1)))
    cons: list = []

    def hist_cons(mask, block, scale):
        for c in spec.numerical_cols:
            h = np.asarray(block["numerical"][c]["dp_hist"], float); h = h / max(h.sum(), 1e-12)
            for b, p in enumerate(h):
                cons.append((mask & (cells[c].values == b), scale * float(p)))
        for c in spec.categorical_cols:
            pr = block["categorical"][c]; tot = sum(float(v) for v in pr.values()) or 1.0
            for k, p in pr.items():
                cons.append((mask & (cells[c].values == str(k)), scale * float(p) / tot))
            cons.append((mask & ~cells[c].isin([str(k) for k in pr]).values, 0.0))  # suppressed

    for coh in release:
        m_c = cells["_cohort"].values == coh["cohort_name"]
        mass = n * float(coh["cohort_size"]) / total
        cons.append((m_c, mass))
        cb = coh.get("class_balance") or {}
        by_class = coh.get("by_class") or {}
        for is_pos, cls in ((True, spec.positive_class), (False, spec.negative_class)):
            share = float(cb.get(cls, 0.5))
            m_cy = m_c & (cells["_pos"].values == is_pos)
            cons.append((m_cy, mass * share))
            if cls in by_class:
                hist_cons(m_cy, by_class[cls], mass * share)
        if not by_class:
            hist_cons(m_c, coh, mass)   # a pooled block describes the cohort, not each class
    cons = [(m, t) for m, t in cons if t == 0.0 or t >= min_expected_rows]
    for _ in range(rounds):
        for mask, t in cons:
            s_ = w[mask].sum()
            if s_ > 0:
                w[mask] *= (t / s_) if t > 0 else 0.0
        np.clip(w, 0.0, 1.0, out=w)
        tot = w.sum()
        if tot > 0:
            w *= n / tot
            np.clip(w, 0.0, 1.0, out=w)
    return w


def _release_cell_matrix(spec: DatasetSpec, release: list, pool: pd.DataFrame, n: int,
                         w_pooled: float = 3.0):
    """Indicator matrix (rows x released cells) with each cell's target count at size `n` and its
    weight. Two families: the per-(cohort, class, column-bin) counts the release states, and the
    POOLED marginals those imply, which is the quantity 1-way total variation actually measures.
    Weighting the pooled family above 1 is what moves marginal error to the release's own floor."""
    cells = _selection_cells(spec, pool, release)
    total = sum(float(c["cohort_size"]) for c in release) or 1.0
    cols, targ, wts = [], [], []

    def add(mask, t, w=1.0):
        cols.append(mask); targ.append(t); wts.append(w)

    coh_of, pos_of = cells["_cohort"].values, cells["_pos"].values
    for coh in release:
        m_c = coh_of == coh["cohort_name"]; mass = n * float(coh["cohort_size"]) / total
        add(m_c, mass)
        cb = coh.get("class_balance") or {}; by = coh.get("by_class") or {}
        for is_pos, cls in ((True, spec.positive_class), (False, spec.negative_class)):
            share = float(cb.get(cls, 0.5)); m_cy = m_c & (pos_of == is_pos)
            add(m_cy, mass * share)
            blk = by.get(cls)
            if blk is None:
                continue
            for c in spec.numerical_cols:
                h = np.asarray(blk["numerical"][c]["dp_hist"], float); h = h / max(h.sum(), 1e-12)
                for b, p in enumerate(h):
                    add(m_cy & (cells[c].values == b), mass * share * float(p))
            for c in spec.categorical_cols:
                pr = blk["categorical"][c]; tot = sum(float(v) for v in pr.values()) or 1.0
                for k, p in pr.items():
                    add(m_cy & (cells[c].values == str(k)), mass * share * float(p) / tot)
        if not by:
            for c in spec.numerical_cols:
                h = np.asarray(coh["numerical"][c]["dp_hist"], float); h = h / max(h.sum(), 1e-12)
                for b, p in enumerate(h):
                    add(m_c & (cells[c].values == b), mass * float(p))
            for c in spec.categorical_cols:
                pr = coh["categorical"][c]; tot = sum(float(v) for v in pr.values()) or 1.0
                for k, p in pr.items():
                    add(m_c & (cells[c].values == str(k)), mass * float(p) / tot)
    for c in spec.numerical_cols:
        e = np.asarray(release[0]["numerical"][c]["bin_edges"], float); p = np.zeros(len(e) - 1)
        for coh in release:
            h = np.asarray(coh["numerical"][c]["dp_hist"], float)
            p += float(coh["cohort_size"]) / total * h / max(h.sum(), 1e-12)
        for b, pp in enumerate(p):
            add(cells[c].values == b, n * float(pp), w_pooled)
    for c in spec.categorical_cols:
        pm: dict = {}
        for coh in release:
            pr = coh["categorical"][c]; tot = sum(float(v) for v in pr.values()) or 1.0
            for k, v in pr.items():
                pm[str(k)] = pm.get(str(k), 0.0) + float(coh["cohort_size"]) / total * float(v) / tot
        for k, pp in pm.items():
            add(cells[c].values == str(k), n * float(pp), w_pooled)
        # a category the release suppressed has zero mass: any row carrying it costs the objective
        add(~cells[c].isin(list(pm)).values, 0.0, 10.0 * w_pooled)
    return np.array(cols, dtype=np.float32).T, np.array(targ, float), np.array(wts, float)


def refine_selection(spec: DatasetSpec, release: list, pool: pd.DataFrame, chosen: np.ndarray, *,
                     seed: int | None = None, max_rounds: int = 40, w_pooled: float = 3.0) -> np.ndarray:
    """Swap rows in and out of `chosen` while the weighted L1 distance to the released cell counts
    falls. The rake gives fractional weights that systematic sampling can only approximate at
    integer resolution; this closes the remainder. Post-processing: the objective is built from the
    release, the candidates from the pool."""
    A, t, w = _release_cell_matrix(spec, release, pool, int(chosen.sum()), w_pooled)
    cur = A[chosen].sum(0); L = float((w * np.abs(cur - t)).sum())
    # SEEDED on purpose, and safely: this generator only orders the candidate rows of the swap
    # search, which is post-processing of the release. Every DP noise draw in this file comes
    # from diffprivlib's Laplace mechanism objects above, which take no seed.
    rng = np.random.default_rng(seed)
    for _ in range(max_rounds):
        moved = 0
        ins = np.where(~chosen)[0]; rng.shuffle(ins)
        for i in ins:
            outs = np.where(chosen)[0]
            cand = cur + A[i] - A[outs]
            losses = (w * np.abs(cand - t)).sum(1)
            j = int(np.argmin(losses))
            if losses[j] < L - 1e-9:
                chosen[i] = True; chosen[outs[j]] = False
                cur = cand[j]; L = float(losses[j]); moved += 1
        if not moved:
            break
    return chosen


def release_subbin_values(spec: DatasetSpec, release: list, df: pd.DataFrame, *,
                          seed: int | None = None) -> pd.DataFrame:
    """Make the output's structure below bin resolution the release's own wherever the release
    states the class-conditional shape.

    A released histogram fixes how many rows fall in each public bin and nothing finer, so a
    value's position inside its bin is never released information. Where a cohort carries
    class-conditional blocks, the class shape is the release's at bin resolution and whatever the
    generator does below it is prior; on NHANES that prior separated the classes about twice as
    far as the data inside each bin and cost the linear student 0.044 AUC (Appendix H.17). In
    those cohorts every numeric value is redrawn uniformly inside its released bin, intersected
    with the row's stratification band so cohort membership never moves. Where a cohort carries
    only a pooled block the generator's placement is the only carrier of the class signal and it
    is kept untouched. Reads public edges, the release and the rows: post-processing, no ε.
    Integer-valued columns draw integers from [ceil(lo), ceil(hi) - 1], so every bin count and
    every binned metric is unchanged by construction."""
    rng = np.random.default_rng(seed); out = df.copy()
    cells = _selection_cells(spec, df, release); coh = cells["_cohort"].values
    scoped = np.isin(coh, [r["cohort_name"] for r in release if r.get("by_class")])
    bands = {b.col: np.asarray(b.edges, float) for b in spec.stratify if getattr(b, "edges", None) is not None}
    for c in spec.numerical_cols:
        edges = np.asarray(spec.public_bins[c], float)
        v = pd.to_numeric(out[c], errors="coerce").values.astype(float); fin = np.isfinite(v)
        if not (scoped & fin).any():
            continue
        is_int = bool(np.all(np.mod(v[fin], 1) == 0))
        b = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
        lo = edges[b].copy(); hi = edges[b + 1].copy()
        if c in bands:
            be = bands[c]; k = np.clip(np.digitize(v, be[1:-1]), 0, len(be) - 2)
            lo = np.maximum(lo, be[k]); hi = np.minimum(hi, be[k + 1])
        dlo, dhi = spec.feature_bounds[c]
        lo = np.maximum(lo, dlo); hi = np.minimum(hi, dhi + (1.0 if is_int else 0.0))
        m = scoped & fin & (hi > lo)
        new = v.copy()
        if is_int:
            ilo = np.ceil(lo[m]); ihi = np.ceil(hi[m]) - 1          # integers inside [lo, hi)
            ihi = np.maximum(ihi, ilo)
            new[m] = np.floor(rng.uniform(ilo, ihi + 1))
        else:
            new[m] = rng.uniform(lo[m], hi[m])
        new = np.clip(new, dlo, dhi)
        out[c] = new.astype(int) if is_int and np.isfinite(new).all() else new
    return out


def select_to_release(spec: DatasetSpec, release: list, pool: pd.DataFrame, n: int, *,
                      seed: int | None = None, rounds: int = 40, refine: bool = True,
                      w_pooled: float = 3.0, sub_bin: str = "release") -> pd.DataFrame:
    """Choose n rows of `pool` whose marginals match `release`, by systematic sampling on the
    inclusion weights with rows grouped by cell, so each released cell's count lands within one row
    of its expected value rather than carrying multinomial noise."""
    if len(pool) < n:
        raise ValueError(f"pool holds {len(pool)} rows, fewer than the {n} requested")
    # a pool that stopped early leaves later cohorts unfilled; selection cannot repair that, so say so
    cells0 = _selection_cells(spec, pool, release); total0 = sum(float(c["cohort_size"]) for c in release) or 1.0
    short = [(c["cohort_name"], int((cells0["_cohort"].values == c["cohort_name"]).sum()), n * float(c["cohort_size"]) / total0)
             for c in release]
    short = [(k, h, o) for k, h, o in short if o >= 1 and h < 1.1 * o]
    if short:
        import warnings
        warnings.warn("pool does not cover every released cohort: " +
                      "; ".join(f"{k} has {h} rows for {o:.0f} owed" for k, h, o in short) +
                      ". Those cohorts will come out short of their released share.")
    w = selection_weights(spec, release, pool, n, rounds=rounds)
    cells = _selection_cells(spec, pool, release)
    order = ["_cohort", "_pos"] + list(spec.categorical_cols) + list(spec.numerical_cols)
    idx = np.lexsort([cells[c].astype(str).values for c in order[::-1]])
    ww = w[idx]; tot = ww.sum()
    if tot <= 0:
        return pool.iloc[:0].copy()
    ww = ww * n / tot
    cum = np.cumsum(ww); u = np.random.default_rng(seed).random()
    picks = np.floor(cum - u).astype(int)
    take = np.r_[picks[0] >= 0, np.diff(picks) > 0]
    keep = np.sort(idx[take][:n])
    if refine and len(keep) == n and len(pool) > n:
        chosen = np.zeros(len(pool), bool); chosen[keep] = True
        chosen = refine_selection(spec, release, pool, chosen, seed=seed, w_pooled=w_pooled)
        keep = np.where(chosen)[0]
    out = pool.iloc[keep].reset_index(drop=True)
    if sub_bin == "release":
        out = release_subbin_values(spec, release, out, seed=seed)
    elif sub_bin != "generator":
        raise ValueError("sub_bin must be 'release' (default) or 'generator'")
    return out
