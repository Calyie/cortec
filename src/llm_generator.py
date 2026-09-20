"""
llm_generator.py
Calls a frozen LLM to generate synthetic tabular rows under two conditions:
  - HEADER-ONLY    (Condition A — baseline)
  - CoRTeC         (Condition B — conditioned on DP cohort statistics)

Supports three backends:
  --backend anthropic   Use Anthropic claude-sonnet-4-6 via API
  --backend ollama      Use a local Ollama model (no API costs)
  --backend mock        Rule-based mock — for pipeline testing without any LLM

The LLM is NEVER fine-tuned and NEVER sees raw private data.
All private information arrives only through already-DP-protected statistics.
"""

from __future__ import annotations

import io
import os
import re
import time
import random
from typing import Literal

import numpy as np
import pandas as pd

from src.prompts import (
    SYSTEM_PROMPT,
    build_header_only_prompt,
    build_cortec_prompt,
)
from src.data_loader import COLUMN_NAMES, NUMERICAL_COLS, CATEGORICAL_COLS, TARGET_COL


BackendType = Literal["anthropic", "ollama", "mock"]


def _level_bands(spec, cohort_stats: dict) -> tuple:
    """The Band objects of the finest released conditional level, in cell-key order."""
    lv = cohort_stats.get("conditional_target_levels") or {}
    try:
        fi = max((int(k) for k in lv), default=0)
        return tuple(spec.conditional_levels[fi] or [])
    except Exception:
        return ()


def _hist_mass_between(stat: dict, lo: float, hi: float) -> float:
    """Share of a released DP histogram's mass falling in [lo, hi), bins treated as uniform."""
    edges, hist = stat.get("bin_edges"), stat.get("dp_hist")
    if not edges or not hist:
        return 0.0
    tot = 0.0
    for i, pi in enumerate(hist):
        a, b = float(edges[i]), float(edges[i + 1])
        if b <= a:
            continue
        ov = max(0.0, min(b, hi) - max(a, lo))
        tot += float(pi) * (ov / (b - a))
    return tot


def _part_mass(cohort_stats: dict, band, part: str) -> float:
    """Released marginal mass of ONE cell-key token against its own conditioning band.

    Cell keys are positional: the i-th token belongs to the i-th band, and the token's FORM depends
    on the band. A numeric band yields its own label; a categorical band coarsened by autoconfig
    yields `g0`, `g1`, ... which do not contain the column name at all. Any code that identifies a
    token by looking for the column name inside it is therefore blind to coarsened categoricals --
    which is exactly the kind of column a protected attribute becomes.
    """
    num = cohort_stats.get("numerical", {}) or {}
    cat = cohort_stats.get("categorical", {}) or {}
    col = getattr(band, "col", None)
    if col is None:
        return 0.0
    edges = getattr(band, "edges", None)
    labels = getattr(band, "labels", None)
    groups = getattr(band, "groups", None)
    if edges and labels and part in labels and col in num:
        j = list(labels).index(part)
        return _hist_mass_between(num[col], float(edges[j]), float(edges[j + 1]))
    if groups and part in groups and col in cat:
        return float(sum(cat[col].get(v, 0.0) for v in groups[part]))
    if col in cat:
        return float(cat[col].get(part, 0.0))
    return 0.0


def _released_band_coverage(cohort_stats_list: list[dict], spec) -> dict:
    """Per conditioning column: share of released marginal mass lying in RELEASED bands.

    This is the quantity that predicts cell-wise damage. Cell-wise emits rows only for released
    cells, so mass in a band no released cell names is produced at essentially rate zero -- measured
    at roughly a tenth of what belongs there. Aggregate cell coverage does NOT predict it: two
    releases 34 points apart in aggregate coverage did identical damage because both released one
    band of the same column, and two releases an aggregate rule refused were in fact better than
    the alternative.

    Tokens are matched POSITIONALLY against the band list, the same way `_cell_mass` reads them,
    because a coarsened categorical token (`g0`) carries no column name.
    """
    out: dict[str, list[float]] = {}
    tot = sum(max(cs.get("cohort_size", 1), 1) for cs in cohort_stats_list) or 1
    for cs in cohort_stats_list:
        lv = cs.get("conditional_target_levels") or {}
        table = lv.get(str(max((int(k) for k in lv), default=0)), {}) if lv \
            else (cs.get("conditional_target") or {})
        if not table:
            continue
        bands = _level_bands(spec, cs)
        if not bands:
            continue
        w = max(cs.get("cohort_size", 1), 1) / tot
        seen: dict[int, set] = {}
        for key in table:
            for i, part in enumerate(p.strip() for p in str(key).split("|") if p.strip()):
                seen.setdefault(i, set()).add(part)
        for i, band in enumerate(bands):
            col = getattr(band, "col", None)
            if col is None:
                continue
            covered = sum(_part_mass(cs, band, p) for p in seen.get(i, ()))
            acc = out.setdefault(col, [0.0, 0.0])
            acc[0] += w * min(covered, 1.0)
            acc[1] += w
    return {k: v[0] / max(v[1], 1e-9) for k, v in out.items()}


def _cell_mass(cohort_stats: dict, cell_key: str, bands: tuple) -> float:
    """Released marginal mass of one conditional cell.

    Product of each conditioning level's own released probability — categorical proportions read
    straight off the release, numerical bands integrated over the released histogram. Assumes the
    conditioning columns are independent within the cohort, which is an approximation, but a far
    better one than "every cell is equally likely", which is what the uniform weighting asserted.
    """
    parts = [p.strip() for p in str(cell_key).split("|") if p.strip()]
    num = cohort_stats.get("numerical", {}) or {}
    cat = cohort_stats.get("categorical", {}) or {}
    m = 1.0
    for i, part in enumerate(parts):
        b = bands[i] if i < len(bands) else None
        col = getattr(b, "col", None)
        if col is None:
            continue
        edges = getattr(b, "edges", None)
        labels = getattr(b, "labels", None)
        groups = getattr(b, "groups", None)
        if edges and labels and part in labels and col in num:
            j = list(labels).index(part)
            m *= _hist_mass_between(num[col], float(edges[j]), float(edges[j + 1]))
        elif groups and part in groups and col in cat:
            # autoconfig coarsens a categorical column into groups labelled g0, g1, ...; the
            # RELEASE is still keyed by the original category values. Looking up "g0" in it
            # returns 0, which silently collapsed every cell to the floor mass and put the
            # allocation straight back to uniform on most datasets.
            m *= float(sum(cat[col].get(v, 0.0) for v in groups[part]))
        elif col in cat:
            m *= float(cat[col].get(part, 0.0))
    return max(m, 1e-6)   # never let a cell drop out entirely on a noisy zero


# Minimum share of the population that released conditional cells must cover before the cell-wise
# path is safe. Cell-wise emits rows only for released cells, so uncovered population is generated
# at rate zero. Measured coverage on the two datasets where both paths were run head to head on a
# shared release: adult 99.9% (cell-wise mildly better), NHANES 25.4% (cell-wise erased four of six
# racial groups). Two points do not fix a threshold precisely; 0.90 sits in the wide empty gap
# between them and errs toward the path that cannot silently delete a subpopulation.
CELL_COVERAGE_FLOOR = 0.90
# Coverage is a ratio of floats, so a release sitting exactly ON the floor computes as
# 0.8999999... and would be refused. A release that exactly meets the bar must pass.
_FLOOR_EPS = CELL_COVERAGE_FLOOR - 1e-9

MIN_ROWS_PER_CELL = 8   # below this the per-cell prompt asks for 1-2 rows and parsing collapses


def split_cell_into_batches(n_rows: int, n_positive: int, rows_per_call: int) -> list[tuple[int, int]]:
    """Split one cell's (rows, positives) into per-call batches that sum EXACTLY to both.

    The positive count is a property of the CELL, but prompts are issued per BATCH and
    `_generate_batched` repeats one prompt across batches — so handing every batch the cell's full
    count multiplied it (a 40-row cell asking for 14 positives emitted 28). Each batch gets its
    proportional share, floored so the remainder stays satisfiable.
    """
    remaining, rem_pos, out = int(n_rows), int(n_positive), []
    while remaining > 0:
        b = min(int(rows_per_call), remaining)
        bp = int(round(rem_pos * b / remaining)) if remaining else 0
        bp = max(0, min(b, bp, rem_pos))
        bp = max(bp, rem_pos - (remaining - b))   # leave no impossible remainder
        out.append((b, bp))
        remaining -= b
        rem_pos -= bp
    return out


def normalise_categorical(col: pd.Series) -> pd.Series:
    """Categorical cells as the real data's strings: an integer-valued float (2.0) becomes "2",
    other values are stripped text, NaN stays NaN so the caller can drop the row."""
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        t = str(v).strip()
        if re.fullmatch(r"-?\d+\.0+", t):
            return str(int(float(t)))
        return t if t else np.nan
    return col.map(one)


class LLMSyntheticGenerator:
    """
    Generates synthetic rows using a frozen LLM under header-only and CoRTeC conditions.

    Parameters
    ----------
    backend         : 'anthropic' | 'ollama' | 'mock'
    model           : model name (used for anthropic/ollama backends)
    rows_per_call   : how many rows to request per LLM call (keep ≤25 for reliability)
    max_retries     : retry parse failures this many times per call
    ollama_url      : base URL for Ollama API
    """

    def __init__(
        self,
        backend: BackendType = "anthropic",
        model: str = "claude-sonnet-4-6",
        rows_per_call: int = 20,
        max_retries: int = 3,
        ollama_url: str = "http://localhost:11434",
        spec=None,
    ):
        # `spec` is an optional DatasetSpec. When None the generator keeps the original
        # Adult-only behaviour, so every existing Adult result stays reproducible; when set,
        # the schema, target column, label values and prompts all come from the spec.
        self.spec          = spec
        self.backend       = backend
        self.model         = model
        self.rows_per_call = rows_per_call
        self.max_retries   = max_retries
        self.ollama_url    = ollama_url
        # Reasoning models served through Ollama (gpt-oss, deepseek-r1, qwen3-thinking...) put
        # their chain of thought in a separate `thinking` field that still consumes num_predict.
        # gpt-oss:20b spent all 3072 tokens reasoning and returned content="" with
        # done_reason="length" — it looked like a model that could not follow the CSV contract
        # when it simply ran out of budget. Budget for reasoning + output, and size the context
        # to hold prompt + both.
        self.ollama_num_predict = 8192
        self.ollama_num_ctx = 16384

        # Fable 5 always uses thinking (can't be disabled) and bills those tokens; give headroom so the
        # thinking + CSV never truncates. Bigger rows_per_call amortises the per-call thinking overhead.
        # 3072 was not enough headroom: on the Adult release two consecutive calls each spent the
        # ENTIRE 3072 on reasoning and returned no CSV at all, which the parse guard correctly
        # caught but which reads as "the model cannot follow the schema". This is the §8 defect-8
        # failure mode recurring on a frontier model rather than a local reasoning model, so the
        # budget has to cover reasoning *plus* the rows, not the rows alone.
        self.anthropic_max_tokens = 8192
        self._anthropic_extra: dict = {}
        # Per-vendor request extras, set by the runner (e.g. reasoning effort). Kept separate
        # because the parameter names are not portable across vendors.
        self._openai_extra: dict = {}
        self._gemini_extra: dict = {}

        # ── API spend guard (see CORTEC_MEMORY.md §13) ──────────────────────────────
        # Results are inspected at every CHECK_EVERY-th call. If the calls are not
        # producing what we expect, the run aborts rather than continuing to spend
        # credits on output we are going to throw away.
        # Inspect results after every 2 calls, not 5 (PI instruction, 2026-09-04): a schema or
        # parsing fault should be caught on the second call, not the fifth. The diabetes
        # schema bug burned $3.34 before the 5-call window closed.
        self.CHECK_EVERY = 2
        self.MIN_PARSE_RATE = 0.40          # over the calls since the last checkpoint
        self.MIN_CALLS_BEFORE_ABORT = 2     # never abort on a single unlucky call
        self.MAX_CONSECUTIVE_API_ERRORS = 3
        # A call can "succeed" while most of its rows are discarded downstream. That is how
        # the diabetes schema bug burned $3.34 at a 100% call-success rate, so yield is
        # guarded separately from parse success.
        self.MIN_YIELD_RATE = 0.40
        # a yield ratio is only actionable once this many rows have been requested
        self.MIN_REQUESTED_BEFORE_YIELD_ABORT = 60

        # Hard spend cap. Thinking tokens bill as output on every vendor here, so a generation
        # batch is dominated by output cost and is easy to under-estimate. The budget is enforced
        # from measured `usage`, not guesses.
        #
        # These are APPROXIMATE list rates used to drive the *guard*, not to report cost. Charging
        # every model at Fable 5's $10/$50 (as this did originally) makes the guard ~8x too strict
        # for a cheap model, which aborts healthy runs; charging too little would let one overrun.
        # Report tokens, not these dollars, in the paper.
        _PRICES = {           # (input $/Mtok, output $/Mtok)
            "claude-fable":  (10.0, 50.0),
            "claude-opus":   (10.0, 50.0),
            "claude-sonnet":  (3.0, 15.0),
            "claude-haiku":   (1.0,  5.0),
            "gpt-5":          (1.25, 10.0),
            "gpt-4.1":        (2.0,  8.0),
            "gpt-4o":         (2.5, 10.0),
            "gemini-3-flash": (0.3,  2.5),
            "gemini-3.1-flash": (0.3, 2.5),
            "gemini-3.5-flash": (0.3, 2.5),
            "gemini-3.6-flash": (0.3, 2.5),
            "gemini-3.7-flash": (0.3, 2.5),
            "gemini-3":       (2.0, 12.0),
            "gemini-2.5-pro": (1.25, 10.0),
            "gemini-2.5-flash": (0.3, 2.5),
        }
        pin, pout = 10.0, 50.0          # conservative default for an unrecognised model
        for stem, (i, o) in sorted(_PRICES.items(), key=lambda kv: -len(kv[0])):
            if str(model).startswith(stem):
                pin, pout = i, o
                break
        self.PRICE_IN_PER_MTOK = pin
        self.PRICE_OUT_PER_MTOK = pout
        self.budget_usd: float | None = None
        self._tok_in = 0
        self._tok_out = 0
        self._tok_think = 0
        # Reasoning observability. `_tok_think` is NOT trustworthy on its own: the streaming path
        # returns no token details at all, so a streamed run accumulates 0 while possibly having
        # reasoned throughout. These two counters make that distinguishable.
        self._n_calls_with_thinking_block = 0
        self._n_calls_think_unmeasured = 0
        self._n_calls = 0
        self._n_parse_ok = 0
        self._window_calls = 0
        self._window_ok = 0
        self._consecutive_api_errors = 0
        self._last_api_error: str | None = None
        self._total_rows = 0
        self._total_rows_requested = 0   # denominator for the yield guard
        self._n_rows_out_of_bounds = 0   # rows rejected for leaving the public domain
        # Every successfully parsed frame is kept here so that rows we have already paid for
        # survive an abort. Losing them would waste exactly the credits the guard exists to save.
        self._partial_frames: list[pd.DataFrame] = []

        if backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        elif backend == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif backend == "gemini":
            from google import genai
            from google.genai import types as _gt
            # Without an explicit timeout the client waits on a hung request forever. One call
            # stalled a 1000-row run for 11 minutes with zero rows written while a direct probe of
            # the same model answered in 1.7 s — the request was hung, not slow. Bound it so a bad
            # connection fails and retries instead of blocking the whole run. (Milliseconds.)
            # Which SURFACE serves the model: the public developer API (what the paper's arms
            # used) or Gemini on Vertex AI inside a Google Cloud project -- the enterprise surface
            # the paper recommends. Same model, same request; only the client's auth and endpoint
            # differ. CORTEC_GEMINI_SURFACE=vertex selects Vertex (Application Default Credentials,
            # GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION; the newest Gemini models are served from
            # the `global` location). §6.2 of the paper measures the two surfaces against each
            # other on one release.
            self.surface = os.environ.get("CORTEC_GEMINI_SURFACE", "public").lower()
            if self.surface not in ("public", "vertex"):
                raise ValueError(f"CORTEC_GEMINI_SURFACE must be 'public' or 'vertex', got {self.surface!r}")
            _surface_kw = ({"vertexai": True, "project": os.environ["GOOGLE_CLOUD_PROJECT"],
                            "location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global")}
                           if self.surface == "vertex" else
                           {"api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")})
            print(f"  gemini surface: {self.surface}")
            self._client = genai.Client(
                **_surface_kw,
                # `HttpOptions.timeout` alone is NOT honoured by google-genai 2.22.0: the
                # constructed httpx client still carries Timeout(timeout=None), so a hung request
                # blocks forever and the run stalls with no error and no rows. `client_args` is
                # forwarded to the httpx.Client constructor and is the one that actually binds.
                # Keep both. (HttpOptions is MILLISECONDS; httpx is SECONDS.)
                http_options=_gt.HttpOptions(timeout=180_000,
                                             client_args={"timeout": 180.0}))
        elif backend == "ollama":
            import httpx
            # Ollama's allowedHostsMiddleware 403s any Host header that is not loopback, and it
            # only skips that check when the listener is bound to a non-loopback address
            # (OLLAMA_HOST=0.0.0.0). Two remote setups therefore need OPPOSITE client behaviour:
            #   * Ollama bound to 0.0.0.0, reached directly  -> send the real Host (a loopback
            #     override is fine too).
            #   * Ollama on 127.0.0.1 behind `tailscale serve` -> serve routes BY Host and
            #     forwards it unchanged, so overriding the Host breaks routing (404) while
            #     leaving it intact gets Ollama's 403.
            # So the header cannot be fixed at construction time. Send the natural Host, and
            # retry once with the loopback override only if the server answers 403.
            self._http = httpx.Client(base_url=ollama_url, timeout=600.0)
            self._ollama_host_override = None
        # mock: no client needed

    # ── schema helpers (spec-aware) ────────────────────────────────────────────

    @property
    def _cols(self):
        return self.spec.column_names if self.spec else COLUMN_NAMES

    @property
    def _numcols(self):
        return self.spec.numerical_cols if self.spec else NUMERICAL_COLS

    @property
    def _target(self):
        return self.spec.target_col if self.spec else TARGET_COL

    @property
    def _labels(self):
        if self.spec:
            return [self.spec.positive_class, self.spec.negative_class]
        return [">50K", "<=50K"]

    # ── Public methods ─────────────────────────────────────────────────────────

    def generate_header_only(self, n_total: int = 200) -> pd.DataFrame:
        """
        Condition A: generate n_total rows using only the column schema.
        Returns a raw string DataFrame (same dtypes as raw_train).
        """
        print(f"\n[Condition A] Header-only generation — {n_total} rows …")
        if self.spec:
            from src.generic_pipeline import build_header_only_prompt as _hp
            prompt = _hp(self.spec, n_rows=self.rows_per_call)
        else:
            prompt = build_header_only_prompt(n_rows=self.rows_per_call)
        return self._generate_batched(prompt, n_total, label="header-only")

    @staticmethod
    def _positives_for_cell(rate: float, n_rows: int, rng) -> int:
        """Stochastic rounding: floor(rate*n), plus one more with probability the fractional part.

        Deterministic rounding biases every small cell the same way — round-half-up turns a released
        0.554 over one row into 1.000 every time, which is precisely the saturation this path exists
        to remove. Stochastic rounding keeps the expected count equal to rate*n, so the error is
        bounded by a single row and averages out across cells and draws instead of accumulating.
        """
        exact = float(rate) * int(n_rows)
        base = int(np.floor(exact))
        if rng.random() < (exact - base):
            base += 1
        return int(min(max(base, 0), int(n_rows)))

    def _report_incomplete_batches(self, combined) -> None:
        """Say so when batches arrived without every column.

        Silence here is the dangerous case: the NaN those batches introduce is later filled with
        each column's LOWER BOUND, which biases the marginal without any error being raised.
        """
        n = getattr(self, "_n_missing_col_batches", 0)
        if not n:
            return
        nan_cells = int(combined.isna().sum().sum()) if len(combined) else 0
        print(f"  !! {n} batch(es) were missing columns {getattr(self, '_missing_cols_seen', [])}; "
              f"{nan_cells} NaN cells in the assembled output. These are filled with each "
              f"column's LOWER BOUND downstream and will bias that column — treat the affected "
              f"marginals as unreliable.", flush=True)

    def generate_cortec_by_cell(self, cohort_stats_list: list[dict], n_total: int = 200,
                                seed: int | None = None) -> pd.DataFrame:
        """CoRTeC generation driven by exact per-cell counts rather than per-cohort rates.

        `generate_cortec` gives the model a cohort's conditional table and asks it to honour the
        rates. That saturates: on the constructed registry of §9 it preserved the ORDERING of a
        planted relationship while overstating its spread (0.741 emitted against a released 0.340
        for the strongest group, 0.009 against 0.075 for the weakest). This path instead asks, per
        released cell, for "exactly k of n records" — a quantity the model can satisfy exactly.
        """
        rng = np.random.default_rng(seed)
        cells: list[tuple[dict, str, float, float]] = []
        for cs in cohort_stats_list:
            lv = cs.get("conditional_target_levels") or {}
            table = lv.get(str(max((int(k) for k in lv), default=0)), {}) if lv \
                else (cs.get("conditional_target") or {})
            if not table:
                continue
            # Allocating cohort_size/len(table) to every cell weights all cells EQUALLY, which
            # forces each conditioning column's marginal flat: the registry's transport_mode came
            # back 25/25/25/25 against a real 43/29/15/13, and that alone dominated the 1-way TV.
            # Weight each cell by its released marginal mass instead. This reads only statistics
            # already in the release, so it is post-processing and costs no privacy budget.
            base = max(cs.get("cohort_size", 1), 1)
            _bands = _level_bands(self.spec, cs)
            for key, rate in table.items():
                cells.append((cs, key, float(rate), base * _cell_mass(cs, key, _bands)))
        # which columns the finest conditional level pins — these are stated in the GROUP line and
        # must not also appear as free distributions in the prompt
        try:
            _lv = cohort_stats_list[0].get("conditional_target_levels") or {}
            _fi = max((int(k) for k in _lv), default=0)
            cond_bands = tuple(self.spec.conditional_levels[_fi] or [])
            cond_cols = tuple(b.col for b in cond_bands)
        except Exception:
            cond_bands, cond_cols = (), ()
        if not cells:
            print("  no released conditional cells — falling back to cohort-level generation", flush=True)
            return self.generate_cortec(cohort_stats_list, n_total=n_total)

        # COVERAGE GUARD -- gated PER CONDITIONING COLUMN, not on aggregate cell mass.
        #
        # Cell-wise generation emits rows only for released cells, so any band of a conditioning
        # column that no released cell names is produced at essentially rate zero. Measured on one
        # release at four conditional depths, roughly 90% of the mass belonging to unreleased bands
        # simply never appears: 40.1% of the population sat in unreleased blood-pressure bands and
        # 3.7% was emitted; at 8.4% unreleased, 1.0% was emitted; at 2.1% unreleased, none. So a
        # column's damage equals the released-marginal mass its released bands fail to cover, and
        # that is the quantity to gate on.
        #
        # We gated on AGGREGATE cell coverage first, and it was wrong in both directions. Two
        # releases 34 points apart in aggregate coverage (25.4% and 59.7%) did identical damage,
        # because both released exactly one blood-pressure band. And two releases the aggregate
        # rule refused (82.5%, 87.0%) were in fact fine: at 87.0% cell-wise beat cohort-wise on
        # every aggregate metric (1-way TV 0.037 vs 0.047, TSTR-LR 0.760 vs 0.708), so refusing it
        # would have cost real quality for no safety. Per-column coverage separates all five
        # releases measured, in both directions. Everything here is read from marginals already in
        # the release, so the check costs no privacy budget.
        _percol_cov = _released_band_coverage(cohort_stats_list, self.spec)
        _worst_col, _worst = min(_percol_cov.items(), key=lambda kv: kv[1]) if _percol_cov else ("", 1.0)
        _short = ", ".join(f"{c} ({v:.0%} of its mass)" for c, v in
                           sorted(_percol_cov.items(), key=lambda kv: kv[1]) if v < CELL_COVERAGE_FLOOR)

        # A researcher measuring what the guard prevents has to be able to run the bad path on
        # purpose. This override is deliberately awkward -- an environment variable, not a
        # parameter -- so it cannot be set by accident and shows up in the run's environment.
        _override = os.environ.get("CORTEC_ALLOW_LOW_COVERAGE") == "1"
        if _override and _worst < _FLOOR_EPS:
            print(f"\n  !! CORTEC_ALLOW_LOW_COVERAGE=1: proceeding although '{_worst_col}' has only "
                  f"{_worst:.1%} of its mass in released bands. The rest will be essentially absent "
                  f"from the output. For measurement, not for use.", flush=True)
        if _worst < _FLOOR_EPS and not _override:
            print(f"\n  !! cell-wise generation is unsafe on this release: {_short} in released "
                  f"bands (floor {CELL_COVERAGE_FLOOR:.0%} per conditioning column). Rows outside "
                  f"those bands are emitted at essentially rate zero, so that share of the "
                  f"population would be absent from the output while aggregate fidelity and "
                  f"utility still looked healthy. Falling back to cohort-level generation, which "
                  f"reproduced every subpopulation to within 0.005 where this was measured.\n"
                  f"     To use the cell path, lower n_min or coarsen the conditional level so "
                  f"more bands of '{_worst_col}' survive release.", flush=True)
            return self.generate_cortec(cohort_stats_list, n_total=n_total)

        weights = np.array([c[3] for c in cells], dtype=float)
        weights = weights / weights.sum()
        alloc = np.floor(weights * n_total).astype(int)
        while alloc.sum() < n_total:
            alloc[int(np.argmax(weights * n_total - alloc))] += 1

        _per_cell = n_total / max(len(cells), 1)
        if _per_cell < MIN_ROWS_PER_CELL:
            print(f"\n  !! cell-wise generation with {n_total} rows across {len(cells)} cells is "
                  f"{_per_cell:.1f} rows per cell. Below ~{MIN_ROWS_PER_CELL} the model is asked "
                  f"for one or two rows at a time and parsing collapses — a 7B model aborted at 0% "
                  f"parse in exactly this regime. Raise n_total to >= {int(MIN_ROWS_PER_CELL*len(cells))}, "
                  f"or use a coarser conditional level.", flush=True)
        print(f"\n[Condition B] CoRTeC cell-wise generation — {n_total} rows across "
              f"{len(cells)} released cells (exact counts, stochastic rounding)", flush=True)
        frames = []
        for (cs, key, rate, _), n_rows in zip(cells, alloc):
            if n_rows <= 0:
                continue
            npos = self._positives_for_cell(rate, int(n_rows), rng)
            from src.generic_pipeline import build_cell_prompt
            # The positive count is a property of the CELL, but the prompt is issued per BATCH.
            # Building one prompt for min(rows_per_call, n_rows) and letting _generate_batched
            # repeat it multiplies the positives by the batch count — a 40-row cell asking for 14
            # positives emitted 28. Split the cell into batches here and hand each batch its own
            # share of the count, so the totals are exactly n_rows and npos however the cell splits.
            parts = []
            for b, b_pos in split_cell_into_batches(int(n_rows), int(npos), self.rows_per_call):
                prompt = build_cell_prompt(self.spec, cs, key, n_rows=b, n_positive=b_pos,
                                           cond_cols=cond_cols, cond_bands=cond_bands)
                d = self._generate_batched(prompt, n_rows=b, label=str(key)[:34])
                if len(d):
                    parts.append(d)
            if parts:
                df = pd.concat(parts, ignore_index=True)
                df["_cohort_id"] = cs["cohort_id"]
                frames.append(df)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self._report_incomplete_batches(combined)
        print(f"  cell-wise total: {len(combined)} rows", flush=True)
        return combined

    def generate_matched_header_only(self, cohort_stats_list: list[dict],
                                     n_total: int = 200) -> pd.DataFrame:
        """§7.1.2's missing control: CoRTeC's own run with the released arrays deleted.

        Shares `generate_cortec`'s cohort loop, row allocation, batching and parsing exactly -- the
        ONLY difference is the prompt builder, so nothing but the statistical content varies between
        the two conditions. The ordinary `generate_header_only` control differs additionally in its
        task instruction, which this removes.
        """
        if self.spec:
            from src.generic_pipeline import build_matched_header_only_prompt as _b
            builder = _b
        else:
            # the legacy Adult path (run_experiment.py, and therefore the inversion harness)
            # carries no DatasetSpec, so it needs the spec-free twin
            from src.prompts import build_matched_header_only_prompt as _b
            builder = (lambda _spec, cs, n_rows: _b(cs, n_rows=n_rows))
        return self.generate_cortec(cohort_stats_list, n_total=n_total,
                                    prompt_builder=builder,
                                    label_prefix="matched-header-only")

    def generate_cortec(
        self,
        cohort_stats_list: list[dict],
        n_total: int = 200,
        prompt_builder=None,
        label_prefix: str = "",
    ) -> pd.DataFrame:
        """
        Condition B: generate n_total rows spread across cohorts, each batch
        conditioned on that cohort's DP statistics.

        `prompt_builder` exists so the matched header-only control of §7.1.2 can reuse this loop
        verbatim -- same allocation, same batching, same parser -- and differ in the prompt alone.
        """
        _cond = label_prefix or "CoRTeC"
        print(f"\n[Condition B] {_cond} generation — {n_total} rows across "
              f"{len(cohort_stats_list)} cohorts …")

        # Allocate rows PROPORTIONAL to each cohort's released size. The old code gave every
        # cohort an equal share regardless of size, which over-represented a 2%-of-population
        # cohort by 12x and made the synthetic set a sample from a deliberately wrong mixture
        # of the private population — the thing 1-way/2-way TV distance measures.
        # Cohort sizes are already in the released statistics, so using them is post-processing.
        sizes = np.array([max(cs.get("cohort_size", 1), 1) for cs in cohort_stats_list], dtype=float)
        weights = sizes / sizes.sum()
        alloc = np.floor(weights * n_total).astype(int)
        # Guarantee every cohort is represented at all, then hand the remainder to the largest.
        alloc = np.maximum(alloc, 1)
        while alloc.sum() > n_total and alloc.max() > 1:
            alloc[int(np.argmax(alloc))] -= 1
        if alloc.sum() < n_total:
            alloc[int(np.argmax(sizes))] += n_total - alloc.sum()

        print("  Row allocation (proportional to released cohort size):")
        for cs, w, a in zip(cohort_stats_list, weights, alloc):
            nm = cs.get("cohort_name", f"cohort-{cs['cohort_id']}")
            print(f"    {nm:34s} n={cs.get('cohort_size', 0):>6d}  share={w:>6.1%}  rows={a}")

        frames = []
        for ci, (cs, n_rows) in enumerate(zip(cohort_stats_list, alloc)):
            if n_rows <= 0:
                continue
            if prompt_builder is not None:
                prompt = prompt_builder(self.spec, cs, n_rows=min(self.rows_per_call, int(n_rows)))
            elif self.spec and getattr(self, "quota", False):
                # exact per-column counts: the cohort's totals are apportioned once, and every
                # batch asks for what the cohort still owes after the rows accepted so far, so a
                # batch that returns more or fewer valid rows than asked cannot break the totals
                from src.generic_pipeline import (build_cortec_prompt as _cp, cohort_quota_targets as _ct,
                                                  remaining_quotas as _rq, emitted_counts as _ec)
                _rng = np.random.default_rng(10_000 * (ci + 1) + int(getattr(self, "quota_seed", 0)))
                _target = _ct(self.spec, cs, int(n_rows), _rng)
                _done = {"n": 0, "positives": 0, "numerical": {}, "categorical": {}}

                def _absorb(df, _cs=cs, _done=_done):
                    e = _ec(self.spec, _cs, df)
                    _done["n"] += e["n"]; _done["positives"] += e["positives"]
                    for kind in ("numerical", "categorical"):
                        for c, cnt in e[kind].items():
                            d = _done[kind].setdefault(c, {})
                            for k, v in cnt.items():
                                d[k] = d.get(k, 0) + v
                self._quota_absorb = _absorb
                prompt = lambda b, _cs=cs, _t=_target, _d=_done: _cp(self.spec, _cs, n_rows=b, quotas=_rq(self.spec, _t, _d, b, _rng))
            elif self.spec:
                from src.generic_pipeline import build_cortec_prompt as _cp
                prompt = _cp(self.spec, cs, n_rows=min(self.rows_per_call, int(n_rows)))
            else:
                prompt = build_cortec_prompt(cs, n_rows=min(self.rows_per_call, int(n_rows)))
            if not (self.spec and getattr(self, "quota", False)):
                self._quota_absorb = None
            _lbl = cs.get("cohort_name", f"cohort-{cs['cohort_id']}")
            df = self._generate_batched(
                prompt,
                n_rows=int(n_rows),
                label=f"{label_prefix}/{_lbl}" if label_prefix else _lbl,
            )
            if len(df):
                df = df.copy()
                df["_cohort_id"] = cs["cohort_id"]
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        print(f"  CoRTeC total valid rows: {len(combined)}")
        return combined

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _generate_batched(
        self,
        prompt: str,
        n_rows: int,
        label: str,
    ) -> pd.DataFrame:
        """Repeatedly call the LLM until we accumulate n_rows valid rows."""
        frames: list[pd.DataFrame] = []
        attempts = 0
        max_attempts = (n_rows // self.rows_per_call + 1) * self.max_retries

        while sum(len(f) for f in frames) < n_rows and attempts < max_attempts:
            attempts += 1
            # what this particular call asks the model for — the yield guard scores against this,
            # never against rows_per_call (see _checkpoint)
            still_needed = n_rows - sum(len(f) for f in frames)
            b = min(self.rows_per_call, max(int(still_needed), 1))
            self._total_rows_requested += b
            # a callable prompt is rebuilt per call (exact per-batch counts); a string is reused
            raw_text = self._call_llm(prompt(b) if callable(prompt) else prompt)
            df = self._parse_csv(raw_text)
            self._n_calls += 1
            self._window_calls += 1
            if df is not None and len(df) > 0:
                frames.append(df)
                if callable(prompt) and getattr(self, "_quota_absorb", None):
                    self._quota_absorb(df)      # the next batch asks for what is still owed
                tagged = df.copy()
                tagged["_source"] = label
                self._partial_frames.append(tagged)
                self._n_parse_ok += 1
                self._window_ok += 1
                self._total_rows += len(df)
                print(f"  [{label}] call {attempts}: got {len(df)} valid rows "
                      f"(total {sum(len(f) for f in frames)})")
            else:
                print(f"  [{label}] call {attempts}: parse failed — retrying …")
            if self._window_calls >= self.CHECK_EVERY:
                self._checkpoint()
            time.sleep(0.5)  # polite rate limiting

        if not frames:
            print(f"  WARNING [{label}]: no valid rows produced after {attempts} attempts")
            return pd.DataFrame(columns=self._cols)

        result = pd.concat(frames, ignore_index=True)
        return result.head(n_rows)

    class GenerationAborted(RuntimeError):
        """Raised when the spend guard stops a run so credits are not wasted."""

    def partial_dataframe(self) -> pd.DataFrame:
        """Everything successfully generated so far, including after an abort."""
        if not self._partial_frames:
            return pd.DataFrame(columns=self._cols)
        return pd.concat(self._partial_frames, ignore_index=True)

    def spend_usd(self) -> float:
        return (self._tok_in / 1e6) * self.PRICE_IN_PER_MTOK + \
               (self._tok_out / 1e6) * self.PRICE_OUT_PER_MTOK

    def _checkpoint(self) -> None:
        """Inspect results every CHECK_EVERY calls; abort the run if they are not what we expect."""
        rate = self._window_ok / max(self._window_calls, 1)
        overall = self._n_parse_ok / max(self._n_calls, 1)
        spend = self.spend_usd()
        per_call = spend / max(self._n_calls, 1)
        budget_note = ""
        if self.budget_usd is not None:
            budget_note = (f" | budget ${spend:.2f}/${self.budget_usd:.2f} "
                           f"(~{int((self.budget_usd - spend) / per_call) if per_call > 0 else 0} calls left)")
        print(f"  ── API checkpoint @ call {self._n_calls}: last {self._window_calls} calls "
              f"{self._window_ok} ok ({rate:.0%}) | overall {overall:.0%} | rows {self._total_rows} "
              f"| ${spend:.2f} (${per_call:.3f}/call, {self._tok_think} thinking tok){budget_note}")
        self._window_calls = 0
        self._window_ok = 0
        if self.budget_usd is not None and spend >= self.budget_usd:
            raise self.GenerationAborted(
                f"ABORTED — spend cap reached: ${spend:.2f} of ${self.budget_usd:.2f} after "
                f"{self._n_calls} calls ({self._total_rows} rows). Raise --budget-usd to continue."
            )
        # Yield must be measured against what each call actually ASKED FOR, not against
        # rows_per_call. A cohort needing 7 rows prompts for 7; scoring its 7 returned rows out of
        # 25 reads as a 28% yield and aborts a run that is behaving perfectly. Datasets whose
        # smallest cohorts sort first (credit) tripped this on call 4 of a healthy run.
        requested = max(self._total_rows_requested, 1)
        yield_rate = self._total_rows / requested
        # A yield ratio needs a denominator big enough to be worth acting on. Cohorts are
        # generated smallest-first on some datasets, so at call 2 the denominator can be 13 rows —
        # one unlucky response then reads as a systematic schema failure and aborts a healthy run.
        # The parse-rate guard below still fires at call 2; it is the one that catches API and
        # schema errors early. This guard is about rows being silently dropped, which is a
        # sustained effect and shows up unmistakably by the time this many rows have been asked for.
        enough = self._total_rows_requested >= self.MIN_REQUESTED_BEFORE_YIELD_ABORT
        if enough and yield_rate < self.MIN_YIELD_RATE \
                and self._n_calls >= self.MIN_CALLS_BEFORE_ABORT:
            raise self.GenerationAborted(
                f"ABORTED at call {self._n_calls}: {self._total_rows} usable rows returned "
                f"against {requested} requested ({yield_rate:.0%} yield, threshold "
                f"{self.MIN_YIELD_RATE:.0%}). Calls are SUCCEEDING but most rows are being "
                f"dropped — check schema/parsing before spending more credits."
            )
        if rate < self.MIN_PARSE_RATE and self._n_calls >= self.MIN_CALLS_BEFORE_ABORT:
            raise self.GenerationAborted(
                f"ABORTED at call {self._n_calls}: only {rate:.0%} of the last {self.CHECK_EVERY} "
                f"calls produced usable rows (threshold {self.MIN_PARSE_RATE:.0%}). "
                f"Last API error: {self._last_api_error or 'none — parsing, not the API'}. "
                f"Fix the cause before spending more credits."
            )

    # A rate limit is not a failed call. Two transient Vertex 429s in a row ("Resource exhausted.
    # Please try again later.") aborted the third finance draw of the shipped configuration at
    # call 70 of ~84 under the yield rule, which counts a call that returned nothing as a parse
    # failure. Such an error is waited out with exponential backoff (15 s doubling to 4 min, six
    # attempts, ~8 min in total) before it is allowed to count against the run.
    RATE_LIMIT_RETRIES = 6
    RATE_LIMIT_FIRST_DELAY_S = 15.0
    _RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "resource exhausted", "rate limit", "rate_limit",
                           "ratelimit", "try again later", "overloaded", "too many requests", "503")
    _RATE_LIMIT_NOT = ("billing", "credit", "insufficient_quota", "spending cap", "api key", "unauthorized",
                       "permission", "authentication", "daily", "monthly")   # a daily/monthly cap is not transient

    class _RateLimited(Exception):
        """The vendor asked us to wait; raised by _handle_api_error, caught by _call_llm."""

    def _is_rate_limit(self, e: Exception) -> bool:
        low = str(e).lower()
        return any(m in low for m in self._RATE_LIMIT_MARKERS) and not any(f in low for f in self._RATE_LIMIT_NOT)

    def _call_llm(self, user_prompt: str) -> str:
        """Dispatch to the configured backend. Returns raw response text."""
        delay = self.RATE_LIMIT_FIRST_DELAY_S
        for attempt in range(self.RATE_LIMIT_RETRIES + 1):
            try:
                return self._dispatch(user_prompt)
            except self._RateLimited as rl:
                if attempt >= self.RATE_LIMIT_RETRIES:
                    # persisted for the whole backoff window: count it the ordinary way
                    self._consecutive_api_errors += 1
                    self._last_api_error = str(rl)
                    print(f"  rate limit persisted through {attempt} waits; counting it as an API error")
                    if self._consecutive_api_errors >= self.MAX_CONSECUTIVE_API_ERRORS:
                        raise self.GenerationAborted(
                            f"ABORTED — {self._consecutive_api_errors} consecutive API errors. Last: {str(rl)[:300]}")
                    return ""
                print(f"  rate limited (wait {attempt + 1}/{self.RATE_LIMIT_RETRIES}, {delay:.0f}s): {str(rl)[:140]}")
                time.sleep(delay)
                delay = min(delay * 2, 240.0)
        return ""

    def _dispatch(self, user_prompt: str) -> str:
        if self.backend == "anthropic":
            return self._call_anthropic(user_prompt)
        elif self.backend == "openai":
            return self._call_openai(user_prompt)
        elif self.backend == "gemini":
            return self._call_gemini(user_prompt)
        elif self.backend == "ollama":
            return self._call_ollama(user_prompt)
        else:
            return self._call_mock(user_prompt)

    # Vendors differ in how they signal "this will not fix itself". Sharing one classifier keeps
    # the abort discipline identical across backends: a credit/auth/quota failure must stop the run
    # rather than burn the remaining budget retrying.
    _FATAL_MARKERS = ("credit", "quota", "billing", "authentication", "permission",
                      "invalid_api_key", "invalid api key", "usage limit", "exceeded your current",
                      "insufficient_quota", "api key not valid", "unauthorized")

    def _handle_api_error(self, vendor: str, e: Exception):
        msg = f"{type(e).__name__}: {e}"
        if self._is_rate_limit(e):
            raise self._RateLimited(msg)      # waited out by _call_llm, not counted
        self._last_api_error = msg
        self._consecutive_api_errors += 1
        print(f"  {vendor} API error ({self._consecutive_api_errors} in a row): {msg[:300]}")
        low = str(e).lower()
        for marker in self._FATAL_MARKERS:
            if marker in low:
                raise self.GenerationAborted(
                    f"ABORTED — the API rejected the call for a reason that will not resolve by "
                    f"retrying ({marker}): {msg[:300]}. Resume once this is cleared.")
        if self._consecutive_api_errors >= self.MAX_CONSECUTIVE_API_ERRORS:
            raise self.GenerationAborted(
                f"ABORTED — {self._consecutive_api_errors} consecutive API errors. Last: {msg[:300]}")
        return ""

    def _call_openai(self, user_prompt: str) -> str:
        try:
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_prompt}],
                **self._openai_extra,
            )
            u = getattr(r, "usage", None)
            if u is not None:
                self._tok_in += getattr(u, "prompt_tokens", 0) or 0
                self._tok_out += getattr(u, "completion_tokens", 0) or 0
                det = getattr(u, "completion_tokens_details", None)
                if det is not None:
                    self._tok_think += getattr(det, "reasoning_tokens", 0) or 0
            self._consecutive_api_errors = 0
            return r.choices[0].message.content or ""
        except Exception as e:
            return self._handle_api_error("OpenAI", e)

    def _call_gemini(self, user_prompt: str) -> str:
        try:
            from google.genai import types
            r = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, **self._gemini_extra),
            )
            u = getattr(r, "usage_metadata", None)
            if u is not None:
                self._tok_in += getattr(u, "prompt_token_count", 0) or 0
                # `thoughts_token_count` is reported SEPARATELY from `candidates_token_count` and
                # is billed at the output rate, so it must be added to the output total or the
                # budget guard undercounts by ~2x. (Anthropic and OpenAI both fold reasoning into
                # their output count already, which is why only Gemini needs this.)
                thoughts = getattr(u, "thoughts_token_count", 0) or 0
                self._tok_out += (getattr(u, "candidates_token_count", 0) or 0) + thoughts
                self._tok_think += thoughts
            self._consecutive_api_errors = 0
            # .text is None when the response carried no text part (e.g. a safety block or a
            # response that spent its budget on thoughts) — return "" so the caller's parse
            # failure path and yield guard handle it, exactly as for the other backends.
            return r.text or ""
        except Exception as e:
            return self._handle_api_error("Gemini", e)

    def _call_anthropic(self, user_prompt: str) -> str:
        try:
            kw = dict(
                model=self.model,
                max_tokens=self.anthropic_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                **self._anthropic_extra,
            )
            # The SDK refuses a non-streaming request whose max_tokens implies it could run past
            # ten minutes, which is exactly the configuration needed to give a reasoning model
            # enough headroom that thinking does not starve the CSV. Stream above the threshold
            # and collect the assembled message.
            if self.anthropic_max_tokens > 8192:
                with self._client.messages.stream(**kw) as stream:
                    response = stream.get_final_message()
            else:
                response = self._client.messages.create(**kw)
            # Reasoning models (e.g. Fable 5) return ThinkingBlock(s) before the TextBlock — grab the text only.
            u = getattr(response, "usage", None)
            if u is not None:
                self._tok_in += getattr(u, "input_tokens", 0) or 0
                self._tok_out += getattr(u, "output_tokens", 0) or 0
                det = getattr(u, "output_tokens_details", None)
                if det is not None:
                    self._tok_think += getattr(det, "thinking_tokens", 0) or 0
                else:
                    # The STREAMING path returns usage with output_tokens_details=None, so the
                    # thinking-token count is simply unavailable -- not zero. Recording it as zero
                    # is worse than recording nothing: a run that streamed (max_tokens > 8192,
                    # which is exactly the configuration used to give a reasoning model headroom)
                    # reports "0 thinking tokens" and reads as evidence that reasoning did not
                    # fire, when in fact nothing was measured. Count the calls instead, so the
                    # absence is visible as an absence.
                    self._n_calls_think_unmeasured += 1
            # Presence of a thinking block is observable on BOTH paths and does not depend on the
            # usage object, so it is the reliable signal for "did reasoning actually run?".
            if any(getattr(b, "type", None) == "thinking" for b in response.content):
                self._n_calls_with_thinking_block += 1
            parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            self._consecutive_api_errors = 0
            return "".join(parts)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            self._last_api_error = msg
            self._consecutive_api_errors += 1
            print(f"  Anthropic API error ({self._consecutive_api_errors} in a row): {msg[:300]}")
            low = str(e).lower()
            # Credit/quota/auth problems will not fix themselves — stop immediately rather than
            # hammering the API. Usage-limit errors are surfaced so the run can be resumed later.
            for marker in ("credit", "quota", "billing", "authentication", "permission",
                           "invalid_api_key", "usage limit"):
                if marker in low:
                    raise self.GenerationAborted(
                        f"ABORTED — the API rejected the call for a reason that will not resolve "
                        f"by retrying ({marker}): {msg[:300]}. Resume once this is cleared."
                    )
            if self._consecutive_api_errors >= self.MAX_CONSECUTIVE_API_ERRORS:
                raise self.GenerationAborted(
                    f"ABORTED — {self._consecutive_api_errors} consecutive API errors. "
                    f"Last: {msg[:300]}"
                )
            return ""

    def _call_ollama(self, user_prompt: str) -> str:
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
                # num_ctx MUST be set explicitly. Ollama defaults to a 4096-token context, but
                # a CoRTeC prompt is ~1150 tokens and num_predict asks for 3072 more, so the
                # default silently truncates — and the DP conditional table sits ~70% through
                # the prompt, exactly where truncation bites. That would make a backend look
                # like it "ignores the DP statistics" when it was simply never shown them.
                # Found 2026-09-04 while interpreting the qwen2.5:7b misalignment result.
                "options": {"num_predict": self.ollama_num_predict, "temperature": 0.7,
                            "num_ctx": self.ollama_num_ctx},
            }
            hdrs = {"Host": self._ollama_host_override} if self._ollama_host_override else None
            # A reverse proxy in front of Ollama (e.g. `tailscale serve`) can impose its own
            # request timeout — measured at ~60 s on the RunPod A100 pod, where a 25-row call
            # (58 s) returned 502 while a 15-row call (35 s) succeeded. A transient 5xx from the
            # proxy is not a reason to fail the whole run, so retry briefly with backoff. Keep
            # rows_per_call small enough that a single call stays well under the proxy timeout.
            r = None
            for attempt in range(3):
                r = self._http.post("/api/chat", json=payload, headers=hdrs)
                if r.status_code < 500:
                    break
                wait = 5 * (attempt + 1)
                print(f"  Ollama proxy {r.status_code}; retry {attempt + 1}/3 in {wait}s")
                time.sleep(wait)
            if r.status_code == 403 and self._ollama_host_override is None:
                # Ollama rejected the forwarded Host; retry once with the loopback override and
                # remember the choice for the rest of the run.
                self._ollama_host_override = "localhost:11434"
                r = self._http.post("/api/chat", json=payload,
                                    headers={"Host": self._ollama_host_override})
                if r.status_code == 403:
                    raise self.GenerationAborted(
                        "ABORTED — Ollama returns 403 both with and without a loopback Host "
                        "header. The server is bound to 127.0.0.1 and its host check is active. "
                        "Fix on the server: OLLAMA_HOST=0.0.0.0:11434, then restart Ollama.")
            r.raise_for_status()
            d = r.json()
            msg = d.get("message", {})
            content = msg.get("content", "") or ""
            if not content.strip() and msg.get("thinking"):
                print(f"  [ollama] empty content but {len(msg['thinking'])} chars of reasoning "
                      f"(done_reason={d.get('done_reason')}, eval_count={d.get('eval_count')}) — "
                      f"the reasoning budget was exhausted before any output; raise "
                      f"ollama_num_predict (currently {self.ollama_num_predict}).")
            return content
        except Exception as e:
            # Record it. The Anthropic and Gemini paths both set `_last_api_error`; this one did
            # not, so the abort diagnostic reported "none — parsing, not the API" while the log
            # right above it read "Ollama error: timed out". That misattribution sent three
            # separate investigations at the parser instead of the transport.
            self._last_api_error = f"{type(e).__name__}: {e}"
            print(f"  Ollama error: {e}")
            return ""

    def _call_mock(self, user_prompt: str) -> str:
        """Rule-based mock. With a spec it emits rows matching THAT schema, so a mock run is a
        real end-to-end smoke test of a new dataset's parsing path before any credits are spent.
        It still ignores the prompt, so it is not expected to score well on any metric."""
        if self.spec is not None:
            rng = random.Random(0)
            rows = []
            cats = getattr(self, "_mock_cats", None)
            for _ in range(self.rows_per_call):
                row = []
                for c in self.spec.column_names:
                    if c == self.spec.target_col:
                        row.append(rng.choice(self._labels))
                    elif c in self.spec.numerical_cols:
                        lo, hi = self.spec.feature_bounds[c]
                        row.append(int(rng.uniform(lo, hi)))
                    else:
                        vals = (cats or {}).get(c) or ["A", "B", "C"]
                        row.append(rng.choice(vals))
                rows.append(row)
            header = ",".join(self.spec.column_names)
            return "\n".join([header] + [",".join(str(v) for v in r) for r in rows])

        """
        Rule-based mock that generates plausible Adult-like rows.
        Used for pipeline testing without any LLM calls or API costs.
        The mock ignores the prompt — it generates from fixed Adult-like priors.
        Real LLMs should do better than this mock on CoRTeC prompts if the
        hypothesis holds.
        """
        workclasses = ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"]
        educations  = ["Bachelors", "Some-college", "HS-grad", "Masters", "Assoc-voc", "Prof-school"]
        marital     = ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"]
        occupations = ["Exec-managerial", "Prof-specialty", "Craft-repair", "Adm-clerical",
                       "Sales", "Other-service", "Transport-moving", "Machine-op-inspct"]
        relationships = ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife"]
        races       = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
        sexes       = ["Male", "Female"]
        countries   = ["United-States", "Mexico", "Philippines", "Germany", "Canada", "Cuba"]
        incomes     = [">50K", "<=50K"]

        rows = []
        for _ in range(self.rows_per_call):
            age      = random.randint(18, 70)
            fnlwgt   = random.randint(50000, 400000)
            edu_num  = random.randint(8, 16)
            cap_gain = random.choices([0, random.randint(1000, 10000)], weights=[0.9, 0.1])[0]
            cap_loss = random.choices([0, random.randint(100, 2000)],  weights=[0.93, 0.07])[0]
            hours    = random.randint(35, 55)
            income   = random.choices(incomes, weights=[0.24, 0.76])[0]

            rows.append([
                age,
                random.choice(workclasses),
                fnlwgt,
                random.choice(educations),
                edu_num,
                random.choice(marital),
                random.choice(occupations),
                random.choice(relationships),
                random.choice(races),
                random.choice(sexes),
                cap_gain,
                cap_loss,
                hours,
                random.choice(countries),
                income,
            ])

        # Return as CSV string
        header = ",".join(COLUMN_NAMES)
        data_rows = [",".join(str(v) for v in row) for row in rows]
        return "\n".join([header] + data_rows)

    def _parse_csv(self, text: str) -> pd.DataFrame | None:
        """
        Parse LLM output into a validated DataFrame.
        Handles markdown code fences, extra whitespace, and malformed rows.
        """
        if not text or not text.strip():
            return None

        # Strip markdown code fences if present
        text = re.sub(r"```[a-z]*\n?", "", text).strip()

        # Find the CSV block: first line containing our expected column name
        lines = text.splitlines()
        start_idx = None
        # A real CSV header/row has many commas; require that so a prose sentence
        # mentioning "age"/"income" cannot hijack header detection.
        first_col = str(self._cols[0]).lower()
        last_col = str(self._cols[-1]).lower()
        min_commas = max(4, len(self._cols) - 5)
        for i, line in enumerate(lines):
            low = line.lower()
            if first_col in low and last_col in low and line.count(",") >= min_commas:
                start_idx = i
                break
        if start_idx is None:                       # fall back to the first comma-rich line
            for i, line in enumerate(lines):
                if line.count(",") >= min_commas:
                    start_idx = i
                    break
        if start_idx is None:
            start_idx = 0

        csv_text = "\n".join(lines[start_idx:])

        try:
            # keep_default_na=False is essential: pandas' default NA strings include
            # 'None', 'NA' and 'nan', which are LEGITIMATE CATEGORY VALUES in real datasets
            # (e.g. A1Cresult='None' means the test was not ordered). Without this, those
            # cells became NaN and the dropna() below deleted the whole row — silently
            # destroying 83% of generated rows on the diabetes dataset (2026-09-04).
            df = pd.read_csv(
                io.StringIO(csv_text),
                skipinitialspace=True,
                on_bad_lines="skip",
                keep_default_na=False,
                na_values=[""],
            )
        except Exception:
            return None

        # Normalise column names
        df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]

        # Map to expected column names
        cols = self._cols
        col_map = {c.replace("-", "_").lower(): c for c in cols}
        df = df.rename(columns=col_map)

        # Keep only expected columns that are present
        present = [c for c in cols if c in df.columns]
        if len(present) < len(cols) * 0.7:
            return None  # too many columns missing
        # A batch missing SOME columns is still accepted (throwing away usable rows is worse), but
        # it must not pass silently: concatenating it with a complete batch introduces NaN in the
        # missing columns, and `evaluate_generic.encode` fills NaN numerics with the LOWER BOUND —
        # so those rows quietly become minimum-valued records that bias every numeric marginal.
        # Count it here and report at assembly.
        _missing = [c for c in cols if c not in df.columns]
        if _missing:
            self._n_missing_col_batches = getattr(self, "_n_missing_col_batches", 0) + 1
            self._missing_cols_seen = sorted(
                set(getattr(self, "_missing_cols_seen", set())) | set(_missing))
        df = df[present]

        # Validate the target column against the schema's own label values
        tcol, labels = self._target, self._labels
        if tcol in df.columns:
            valid = df[tcol].astype(str).str.strip().isin(labels)
            df = df[valid]
            df[tcol] = df[tcol].astype(str).str.strip()

        # Coerce numerical columns
        for col in self._numcols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalise categorical cells to the strings the real data carries. A batch in which the
        # model wrote an integer-coded category with a decimal point ("2.0" for SEX = 2) made
        # pandas type that column float, and concatenating it with the other batches re-typed
        # EVERY row of the pool, so the whole pool was written as "2.0"; the evaluator and the
        # selection step then saw a category the real data does not have (finance, draw 2 of the
        # shipped configuration: conditional error 0.22 against 0.01, TSTR-LR 0.60 against 0.69,
        # from a pool whose generation had been fine). Integer-valued floats become the integer's
        # string; everything else is stripped text; a missing categorical cell drops the row.
        catcols = [c for c in cols if c not in self._numcols and c != tcol and c in df.columns]
        for col in catcols:
            df[col] = normalise_categorical(df[col])
        if catcols:
            bad = df[catcols].isna().any(axis=1)
            self._n_missing_cat_rows = getattr(self, "_n_missing_cat_rows", 0) + int(bad.sum())
            df = df[~bad]

        # Drop only rows with a genuinely unusable NUMERIC cell. Categorical cells are kept
        # verbatim — an unexpected category is a fidelity error to be measured, not a reason
        # to silently discard the row.
        numeric_present = [c for c in self._numcols if c in df.columns]
        df = df.dropna(subset=numeric_present) if numeric_present else df

        # Reject rows outside the schema's PUBLIC domain bounds. These bounds are public by
        # assumption (§3.1), so filtering on them costs no privacy budget. A degenerate response
        # once produced LIMIT_BAL=34 against a documented floor of 10,000 and AGE=0 against a
        # floor of 21; without this the row entered the synthetic dataset silently, and an
        # out-of-domain value is not a fidelity error to be measured, it is a malformed record.
        bounds = getattr(self.spec, "feature_bounds", None) if self.spec else None
        if bounds:
            keep = pd.Series(True, index=df.index)
            for col, (lo, hi) in bounds.items():
                if col in df.columns and col in numeric_present:
                    keep &= df[col].between(lo, hi)
            dropped = int((~keep).sum())
            if dropped:
                self._n_rows_out_of_bounds += dropped
                print(f"    dropped {dropped} row(s) with values outside the public domain "
                      f"bounds", flush=True)
            df = df[keep]
        return df if len(df) > 0 else None
