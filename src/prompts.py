"""
prompts.py
Prompt templates for the two generation conditions being compared:

  Condition A — HEADER-ONLY:
      LLM knows only column names and the dataset context.
      This is the informal "just ask an LLM" baseline many practitioners use.

  Condition B — CoRTeC:
      LLM sees DP-released per-cohort statistics as grounding signal.
      The hypothesis: richer conditioning → rows that better match the
      private joint distribution.
"""

from __future__ import annotations
from src.data_loader import COLUMN_NAMES, DATASET_DESCRIPTION


SYSTEM_PROMPT = (
    "You are a synthetic tabular data generator. "
    "Generate realistic rows exactly matching the column schema described. "
    "Output ONLY valid CSV: one header row, then the requested number of data rows. "
    "No explanations, no markdown code blocks, no extra text — only CSV."
)


def build_header_only_prompt(n_rows: int = 20) -> str:
    """
    Condition A: LLM knows column names and dataset description only.
    No distributional information beyond what column names imply.
    """
    cols_str = ", ".join(COLUMN_NAMES)
    return (
        f"Dataset: {DATASET_DESCRIPTION}\n\n"
        f"Column schema (in order):\n{cols_str}\n\n"
        f"Generate {n_rows} realistic synthetic rows that could plausibly appear "
        f"in this dataset. Use varied values. "
        f"For the 'income' column use exactly '>50K' or '<=50K'.\n\n"
        f"Output only CSV with header row."
    )


def build_matched_header_only_prompt(cohort_stats: dict, n_rows: int = 20) -> str:
    """CoRTeC's own prompt with the released arrays deleted and NOTHING else changed.

    The legacy (spec-free) twin of `generic_pipeline.build_matched_header_only_prompt`; see that
    docstring for why this control exists. Built by STRIPPING `build_cortec_prompt` output, so every
    surviving line is byte-identical to CoRTeC's, and fail-closed: any surviving percentage or
    decimal is a released quantity and raises rather than returning a contaminated control.
    """
    import re as _re
    full = build_cortec_prompt(cohort_stats, n_rows=n_rows)
    KEEP = ("Dataset:", "You are generating rows", "Column schema (in order):",
            f"Generate {n_rows} synthetic rows", "Output only CSV")
    out = [ln for ln in full.splitlines() if any(ln.startswith(k) for k in KEEP)]
    text = "\n\n".join(out) + "\n"
    leaked = _re.findall(r"\d+\.\d+|\d+%", text)
    if leaked:
        raise RuntimeError(
            f"matched header-only prompt still carries released quantities {leaked[:5]} — "
            f"refusing to return a contaminated control")
    return text


def build_cortec_prompt(cohort_stats: dict, n_rows: int = 20) -> str:
    """
    Condition B: LLM conditioned on DP-released per-cohort statistics.
    The statistics are the ONLY information from the private dataset that reaches
    the generator — everything else is post-processing of already-DP-protected values.
    """
    cid     = cohort_stats["cohort_id"]
    n_c     = cohort_stats["cohort_size"]
    num     = cohort_stats["numerical"]
    cat     = cohort_stats["categorical"]
    balance = cohort_stats["class_balance"]

    lines = [
        f"Dataset: {DATASET_DESCRIPTION}",
        f"",
        f"You are generating rows for a specific subpopulation (Cohort {cid}) "
        f"identified by privacy-preserving clustering. "
        f"The following statistics describe this cohort (approximate, "
        f"privacy-perturbed — treat them as distributional targets, not exact values).",
        f"",
        f"Cohort size (approximate): {n_c} individuals",
        f"",
        f"CLASS BALANCE:",
    ]

    for cls, prop in balance.items():
        lines.append(f"  income={cls}: {prop:.1%}")

    lines += ["", "NUMERICAL FEATURES — approximate distribution of this cohort.",
              "Match the SHAPE below, not just the average: the percentages are the share of rows "
              "that must fall in each range."]
    for col, v in num.items():
        lo, hi = v["bounds"]
        lines.append(f"  {col}: mean≈{v['dp_mean']:.1f}, std≈{v['dp_std']:.1f} (valid range {lo}–{hi})")
        edges, hist = v.get("bin_edges"), v.get("dp_hist")
        if edges and hist:
            parts = []
            for i, p in enumerate(hist):
                if p < 0.005:
                    continue
                a, b = edges[i], edges[i + 1]
                a_s = f"{a:.0f}"; b_s = f"{b:.0f}"
                parts.append(f"{a_s}–{b_s}: {p:.0%}")
            if parts:
                lines.append(f"      distribution → {', '.join(parts)}")

    lines += ["", "CATEGORICAL FEATURES (approximate proportions — reproduce ALL of these, "
                  "including the less common ones, in roughly these shares):"]
    for col, props in cat.items():
        # Show every released category, not just the top 4. Stage A releases ~10 occupation
        # categories and the prompt was showing 4, which measured as 105% of CoRTeC's excess
        # 1-way TV over AIM on 2026-09-04 (occupation TV 0.161 vs AIM's 0.055). These
        # proportions are already DP-released, so displaying more of them is post-processing
        # and costs no additional privacy budget.
        cats = ", ".join(f"{c}({p:.1%})" for c, p in props.items())
        lines.append(f"  {col}: {cats}")

    cond = cohort_stats.get("conditional_income", {})
    if cond:
        lines += [
            "",
            "INCOME LIKELIHOOD BY GROUP (from the private data, privacy-preserving). "
            "These are the GROUND TRUTH for this dataset — set each row's 'income' to match the rate for "
            "that row's education/hours group. This dataset may DIFFER from general expectations; follow "
            "THESE numbers, not any outside assumption about who earns more:",
        ]
        for g, r in cond.items():
            lines.append(f"  P(income='>50K' | {g}) ≈ {r:.0%}")

    # The joint table is more specific than the two marginal families and removes the need to
    # combine them by hand. Added 2026-09-04 alongside (not instead of) the marginal table —
    # the marginal table and the "even if counterintuitive" instruction are unchanged.
    joint = cohort_stats.get("conditional_income_joint", {})
    if joint:
        lines += [
            "",
            "INCOME LIKELIHOOD BY EXACT GROUP (education band AND hours band together — use this "
            "table in preference to the one above whenever a row's exact group appears here):",
        ]
        for g, r in joint.items():
            lines.append(f"  P(income='>50K' | {g}) ≈ {r:.0%}")

    cols_str = ", ".join(COLUMN_NAMES)
    lines += [
        "",
        f"Column schema (in order): {cols_str}",
        "",
        f"Generate {n_rows} synthetic rows whose distributions faithfully reflect the statistics above, "
        f"including correlations (occupation consistent with education and hours-per-week). "
        f"Set the 'income' column ('>50K' or '<=50K') to MATCH the INCOME LIKELIHOOD table for each row's "
        f"education and hours group — even if it seems counterintuitive. Do NOT use general world-knowledge "
        f"about income; use only the rates given above.",
        "",
        "Output only CSV with header row.",
    ]

    return "\n".join(lines)
