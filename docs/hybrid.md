# cortec-hybrid

A correction of the target column for marginal DP synthesisers such as AIM, MST and PrivBayes. It
needs no language model. Its heaviest dependency is scikit-learn.

**It does not generate synthetic data.** The name can mislead, so this is the most important point.
`cortec-hybrid` contains no synthesiser. It never calls AIM, MST or anything else, and it will not
produce a synthetic dataset for you. Nor does it depend on smartnoise-synth. It is a post-processor.
It releases a DP conditional target table from your private data. It then rewrites only the target
column of synthetic data you have already generated, so that each cell's rate matches the table.
Every feature column comes out exactly as you put it in. The tool reports the shift in the feature
marginals, which is zero. The test suite checks that the columns are equal.

If you do not already run a marginal synthesiser, you do not need this tool. `cortec decode` in the
`cortec` package decodes a CoRTeC release with no language model, using the same estimator as AIM
and MST. Every column is then carried by the release, and the budget is spent once. The hybrid is
for an institution that keeps AIM or MST and wants the named conditional rates added to its output.

```
your private data ──► DP conditional table ──┐
                                             ├──► relabel target column only ──► corrected output
your AIM/MST output ─────────────────────────┘
```

| Step | What you do |
|---|---|
| 1. Install | Install `cortec` first, then this package |
| 2. What you need | Your private table, your synthesiser's output and a public schema |
| 3. Run the correction | One call, or the four steps it is made of |
| 4. Read the result | What `worth_it` measures, and what it does not |
| 5. Before you run: feasibility | Why `check_feasible()` refuses some schemas |
| 6. Privacy | The ledger, `n_min` and the privacy unit |
| 7. Why rate matching alone | The measurement that turned rank preservation off |

## 1. Install

`cortec-hybrid` reuses `cortec`'s audited privacy accounting, so install `cortec` first. It needs
only `cortec`'s `Schema` and `PrivacyLedger` and no vendor SDK. You need Python 3.10 or later. The
packages are not on PyPI:

```bash
git clone https://github.com/Calyie/cortec
cd cortec
pip install .                                # both packages: cortec and cortec_hybrid
pip install '.[dev]'                         # pytest, for the tests
python3 -m pytest tests -k hybrid -q         # the hybrid's tests, offline
cortec-hybrid --help                         # the help page: what you can run, with the arguments
                                             #   (or python3 -m cortec_hybrid --help)
```

**Run it.** Installing the package puts the `cortec-hybrid` command on your PATH. `cortec-hybrid
run` runs the correction from any directory. It takes the private table, the synthetic table to
correct, the schema file and the cell columns. The schema file holds `SCHEMA = Schema(...)`, as in
step 4 of the setup guide (`docs/setup.md`). With `--holdout`, it also scores the table before and
after beside a real sample and a shuffled floor. That is the number `worth_it` does not give.

```bash
cortec-hybrid run --data private.csv --synthetic synthetic.csv \
    --schema my_schema.py --columns admission_type,a1c_result --epsilon 0.5 --holdout holdout.csv
cortec-hybrid run --help
```

`python3 -m cortec_hybrid run ...` is the same command for an environment whose scripts directory is
not on PATH.

## 2. What you need

1. **Your private data** as one pandas DataFrame with the columns the schema declares. Use one row
   per person (section 6 explains why).
2. **Synthetic data you already generated** with whatever DP synthesiser you use (AIM, MST,
   PrivBayes or a vendor product). It must have the same columns. This tool does not produce it.
3. **A `Schema`** from the `cortec` package. It holds the column list, public bounds, public bin
   edges, the target and its two labels, all declared from public knowledge only. The setup guide of
   `cortec` (`setup.md`, step 4) explains each field.

## 3. Run the correction

The one-call form:

```python
from cortec import Schema
from cortec_hybrid import check_feasible, correct

check_feasible(schema, n_rows=len(private_df))   # refuses BEFORE you spend hours on a schema AIM cannot fit

# `synthetic_df` is YOUR synthesiser's output. This tool does not produce it.
corrected, table, gain = correct(
    schema, private_df, synthetic_df,
    columns=("admission_type", "a1c_result"),   # the cells: richer is better and costs no extra ε
    epsilon=0.5,
)
print(gain["verdict"])
# -> "the correction materially improved conditional calibration"
#    or "the correction changed little; ... consider keeping the budget instead"
```

| Parameter of `correct()` | Default | What it does |
|---|---|---|
| `columns` | required | the columns whose cells the table is released over. Numeric columns are binned by the schema's public edges, categoricals by declared value or by the schema's `coarsen` map |
| `epsilon` | required | the budget for the table. Cells partition the data, so the whole table costs this once, however many cells it holds |
| `n_min` | `150` | cells with fewer private records are not released. The floor is 50 |
| `seed` | `None` | tests only: a seeded release is reproducible and carries no guarantee |

When the correction is refused, it raises `CorrectionError` before any budget is spent. Reasons
include no cell reaching `n_min`, a column the schema does not declare or a meaningless privacy
unit. `cortec`'s `install_guard()` or `with guard():` prints the refusal in the standard layout and
exits with status 2 instead of showing a traceback (see section 10 of the setup guide, `setup.md`).
`cortec-hybrid --help` prints this package's help page.

`correct()` is four steps, each available on its own:

1. `check_feasible(schema, n_rows=...)` refuses a schema AIM is unlikely to converge on (section 5).
2. `release_conditional_table(schema, private_df, columns, epsilon=..., n_min=...,
   max_rows_per_person=...)` spends the budget and returns a `ConditionalTable`. It holds `cells`
   (the rate per cell), `support` (the noised size per cell), `epsilon` and `audit`. Store it. It
   contains no private record.
3. `relabel(schema, synthetic_df, table)` rewrites the target column so that each cell matches its
   released rate. It uses stochastic rounding of the count and i.i.d. assignment within the cell. It
   is pure post-processing of two already-DP artefacts and costs nothing beyond the table.
4. `estimate_gain(schema, synthetic_df, corrected, table)` measures what changed, using released
   quantities only.

## 4. Read the result

`estimate_gain()` returns a dictionary:

| Key | Meaning |
|---|---|
| `conditional_error_before`, `conditional_error_after`, `conditional_error_reduction` | mean absolute gap between each cell's rate in the data and the released rate, before and after |
| `feature_marginal_shift` | mean total-variation change of the categorical marginals between input and output (it should be zero, since features are untouched) |
| `n_cells_corrected` | how many released cells were matched |
| `worth_it`, `verdict` | true when conditional error fell by more than 0.02, with a sentence saying so |
| `worth_it_measures`, `worth_it_does_not_predict` | what the criterion is, and what it is not |

`cortec-hybrid terms` explains each word in the printed output, such as conditional error, feature
marginal shift, worth it, ceiling, floor and TSTR. `cortec-hybrid <term>` explains just one.

`cortec`'s output standard applies here too. `show(correct(...))` prints the result as a stage
header, the named values, the released table, the warnings and the verdict. It returns a record
whose `save()` writes JSON, Markdown, CSV and a before-and-after figure. Section 11 of `cortec`'s
setup guide and `result-format.md` in the same folder describe the layout and the record.

```python
from cortec import show

rec = show(correct(schema, private_df, synthetic_df, columns=("admission_type",), epsilon=0.5),
           schema=schema.name)
rec.save("results", "correction")
```

**Read `worth_it` for what it is.** It is a calibration criterion. It means that conditional error
against the released table fell by more than 0.02. It does not predict downstream utility, and we
measured the two coming apart. Across 11 datasets, the correction improved calibration in nearly
every case, while downstream AUC moved in both directions. Of 9 runs flagged `worth_it`, downstream
AUC fell on 6 in one run and on 3 of 10 in a repeat. Three datasets were reliably negative (a renal
registry, hospital readmission and cervical cancer): calibration improved and downstream AUC dropped
every time. `worth_it_does_not_predict` says exactly this. Check the correction on your own held-out
data before adopting it. Keep your budget if the gain is small.

**What to expect.** Marginal DP synthesisers are excellent at the statistic they optimise, and their
own papers evaluate them on it. They do not release a conditional target table. On UCI Adult at ε =
2 and n = 300, the reference AIM's seen-conditional error is 0.055 and the reference MST's is 0.140
against 0.062 for a real sample (the paper's Table 3). The reference AIM's downstream utility is at
the real sample's level. At n = 1,000, it scores higher than the fixed-rule CoRTeC release. It is
level with or below the auto-configured release decoded with no model and the selection step
(`cortec decode`). The measurements in this section were made during development against AIM and MST
at a library's defaults. That library's sampler was later found to be defective (the paper's
Appendix D). The large utility gap those rows showed came from the sampler, and the paper reports
the reference implementations instead.

This tool spends a small, separately accounted part of the budget on the one thing those mechanisms
do not target. That is a conditional target table, `P(y | cell)`, over a partition of the data. It
then relabels the output to match the table. Because the cells partition the data, the whole table
costs one query's worth of ε however many cells it holds. Accuracy depends on how many records each
cell holds, not on how many cells there are. This is what makes a rich table affordable. It is not
"CoRTeC without the language model", nor is it a general improvement over its input. A coarse
12-cell table is the obvious design. In our runs on Adult, it fixes calibration but adds little
utility (+0.025 AUC on AIM and +0.037 on MST). The large gain came only with a richer table and
against the library-default rows. Use the tool in one specific situation: when your marginal
synthesiser's output has weak conditional target structure and you need a calibrated rate.

## 5. Before you run: the feasibility check

`check_feasible()` refuses a schema AIM is unlikely to converge on, so you learn in the first second
rather than the third hour. Pass `n_rows=len(private_df)`, because the row count is one of the three
factors it uses. Pass `raise_on_fail=False` to receive the report and proceed anyway.

It used to gate on domain size, until our own ablation showed that this was wrong. UCI Adult has the
largest domain of every schema we measured (4.4 × 10¹³, which is 6.5 million times larger than a
healthcare prefix that times out), and yet Adult is the one that fits. Within a single dataset, the
run with the larger domain fit 425 seconds faster than the run with the smaller one. Domain volume
does not predict convergence, so gating on it refuses the wrong schemas. Three factors did predict
it, each demonstrated by a controlled flip:

- **Attribute count.** A 19-attribute schema times out at a row count where a 15-attribute one fits.
- **Row count.** The same schema at the same domain moves from a 26-minute fit to a timeout purely
  by going from 26,048 to 81,410 rows.
- **High-cardinality correlated numeric columns.** One dataset has Adult's width, fewer rows and a
  30× smaller domain, yet it still fails. Dropping any three columns does not help, but dropping
  three amount columns does. A width-matched control that drops three demographic columns instead
  still times out. That is the only way to separate the two causes.

The report includes the full observation table it is derived from. It also gives the estimated
domain size for reference, which is explicitly not used to decide. This is a heuristic from ten fits
under one AIM implementation, not a theory.

## 6. Privacy

The tool reuses `cortec`'s audited `PrivacyLedger`, so the composition rules are implemented and
audited once. `table.audit` is the same machine-readable trail: every query with its epsilon,
sensitivity, composition rule and partition key. The cell sizes it publishes are noised and charged
in their own group beside the rates. Validation runs before any budget is spent. `n_min` has a hard
floor of 50, because a rate over fewer records is dominated by its own Laplace noise.

**ε protects one ROW, and this tool is the one most likely to be used on data where a row is not a
person.** It corrects a synthesiser you already run, so it tends to be pointed at the table your
institution already has. That often includes encounter-level clinical data, where one patient
contributes many rows. Under group privacy, a person who contributes `k` rows receives `k · ε`. On a
real hospital dataset whose heaviest patient has 40 encounters, a declared ε of 2.0 is an ε of 80
for that patient. That carries no meaningful guarantee.

Declare the privacy unit with `max_rows_per_person`. If you declare one whose ε per person is
meaningless, the release is refused rather than annotated:

```python
# refused: eps_person = 80
release_conditional_table(schema, df, ("grp",), epsilon=2.0, max_rows_per_person=40)

# proceed with a ROW-level guarantee, acknowledgement recorded in the audit trail
tbl = release_conditional_table(schema, df, ("grp",), epsilon=2.0, max_rows_per_person=40,
                                acknowledge_vacuous_privacy_unit=True)
tbl.audit["epsilon_per_person"]          # -> 80.0
tbl.audit["epsilon_per_person_vacuous"]  # -> True
```

Leave it out, and the audit records `"UNDECLARED"` rather than assuming 1. If your data has more
than one row per person, aggregate before releasing or report `ε_person` rather than the per-row
number.

## 7. Why rate matching alone

Rank-preserving assignment orders the rows within a cell by a model of the synthetic data's own
target. We assumed it was necessary for this correction to be worth its cost. Testing this package
disproved that in one regime, and the code changed as a result: `preserve_ranking=False` is the
default of `relabel()`.

Rank preservation helps only when the input synthetic data already carries structure within each
cell. When its target is close to noise (exactly the case this tool targets), a ranking model fitted
on that target learns the noise. Ordering by it then adds false structure that a downstream model
mistakes for signal. On the renal registry, that cost 0.068 AUC against plain random assignment
(0.546 against 0.614 in the table below).

The guard we first wrote for this did not work, and the reason is worth knowing. `_rank_scores`
gated on whether the synthetic target is predictable from the synthetic features (cross-validated
AUC against `RANK_INFORMATIVENESS_FLOOR = 0.55`). But a synthesiser generates its target as a
function of those features, so that score is almost always near-perfect: 0.981 on one dataset whose
real downstream AUC is 0.631. It measures self-consistency, not validity. It is highest exactly when
the synthesiser has confidently learned a wrong relationship. It therefore selected for the failure
mode it was built to prevent.

Over three datasets at three synthesiser seeds each, i.i.d. assignment scored higher than rank
preservation in 9 of 9 runs:

| dataset | base synthesiser | + rank-preserving | + i.i.d. within cell |
|---|---|---|---|
| renal registry | 0.631 | 0.546 (−0.085) | **0.614 (−0.017)** |
| diabetes | 0.581 | 0.528 (−0.053) | **0.549 (−0.032)** |
| adult | 0.705 | 0.788 (+0.083) | **0.821 (+0.116)** |

Preserving the within-cell ordering requires evidence that the ordering is real, and nothing
available to a post-processing tool provides that. The transferable lesson is not specific to this
tool. A check fitted on a system's own output measures self-consistency. It cannot detect that the
system is confidently wrong. When confidence and wrongness go together, it will actively recommend
the wrong action.

## Licence

Apache-2.0.
