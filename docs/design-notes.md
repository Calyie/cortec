# Design notes: what was measured, and why the defaults are what they are

This document records the measurements behind the package's defaults and guardrails. The setup
guide (`setup.md`) says how to run the package; this is the evidence. The numbers of sections 2
and 11 are those of the paper (`paper/CoRTeC_arxiv.pdf`); the measurements of sections 3 to 10 are
development measurements made before the paper's campaign (on Adult, NHANES and a finance benchmark, with
Gemini 3.5 Flash and Claude Fable 5, against AIM and MST at a library's defaults) and are kept because each
explains a default or a guard. They are marked "development measurement" below; where the paper
supersedes one, the section says so.

## 1. When to use the package, and when not

**What the package delivers** is the release: an auditable artefact of counts built for the downstream classifier, ahead of the reference AIM on the conditional structure it budgets for and behind it on pairs, which anyone can decode at no further privacy cost, with Private-PGM or with a frozen language model (section 11).

**Use it** when your private data plausibly departs from what a general-purpose model would assume,
such as institution-specific coding, local case mix or proprietary product structure, and when a
downstream model or a calibrated rate is the deliverable.

**Do not use it** for a published marginal table when a marginal synthesiser will do. The reference
AIM, fitted the way its authors fit it, is ahead of this package on pairwise fidelity at every seed
and on downstream utility at n = 1,000 (section 11); the marginal synthesisers pay once and sample
freely. What this package offers that they do not is the architecture: the budget is spent once on
a small release, the model never sees a row, every draw is post-processing, and the backbone can
change at no privacy cost.

**Do not use an unconditioned LLM** for any regulated quantity. Asked for synthetic hospital records
with no conditioning, a frontier model reported a 30-day readmission rate of 98.5% for a group whose
true rate was 21.4%, and it reported that same 98.5% whatever the private data said. No rank-based
utility metric reveals that, because the ordering is right and only the magnitude is wrong.

## 2. How close to real data it gets

This is the number the method is usually adopted for, so here is the boundary rather than the best
case. On UCI Adult at ε = 2 and n = 300 (the paper's Table 3, seed 42, three draws), models trained
on the package's output are within 0.025 AUC of models trained on a real sample of the same size on
every student: 0.807, 0.862 and 0.833 on Qwen3-14B and 0.805, 0.863 and 0.838 on Gemini against the
sample's 0.830, 0.870 and 0.840. The single-column error is 0.046 on Qwen3-14B and 0.027 on Gemini
against the sample's 0.043; the seen-conditional error is 0.027 and 0.013 against the sample's
0.062. Replicated on three splits with three fresh releases, the ordering holds (Figure 3 and
Appendix E.2 of the paper). At n = 1,000 the students are 0.03 AUC below the real sample's (0.827, 0.860, 0.859 against
0.855, 0.892, 0.883): a fixed release does not get richer as you ask for more records. On the
clinical survey NHANES the students are 0.07 to 0.10 below a real sample's at n = 300 (0.678, 0.657,
0.645 against 0.773, 0.728, 0.716) and the model-free decode of the same release is ahead of them
(section 11). A development measurement found an equivalence result at n = 300 under one generator
and a finance dataset; the finance dataset is not in the paper, and the comparison against the
library-default AIM and MST that accompanied it is not reported, because of the sampler defect
(section 11). Do not assume parity.
Measure it on your own data against a real sample of matched size, and read the conditional
measures beside the aggregate ones.

## 3. The release: class-conditional histograms

The release used to carry one pooled feature histogram per cohort plus the target rate inside the
conditional hierarchy's cells, and nothing about how any other feature relates to the target. The
generator filled that in from its prior, and on the finance benchmark the filled-in columns
measurably degraded the downstream model (development measurement). The release carries one histogram block per
(cohort, outcome) wherever both outcomes clear `n_min`.

- **Same ε.** The two outcome blocks of a cohort are disjoint, so they compose in parallel at the
  same ε per query the pooled block spent. The pooled histogram is their mixture under the released
  class balance, which is post-processing and no query. The ledger charges each record the class
  balance plus one block, through nested partition keys (`cohort/outcome`), and `epsilon_accounted`
  is unchanged. Where either outcome falls short of `n_min` the cohort keeps a pooled block.
- **What the model sees.** "Rows with `readmitted_30d = YES` look like this; rows with `NO` look
  like that" for every column, with the instruction to decide each row's outcome first and draw its
  other columns from that outcome's distributions. Cell-wise prompts show the same per-outcome
  blocks for the k positives and n − k negatives they ask for.
- **Measured.** Decoded by a naive independent sampler with no model at all, the class-conditional
  release reaches the real-sample floor on two of three downstream students on both finance and
  Adult, where the pooled release trails it by 0.03–0.06 AUC, and held-out conditional error
  improves by 31–39% (development measurement). Through the generator on the finance benchmark, three draws per
  release, TSTR is 0.680 / 0.712 / 0.704 (LR / RF / GBM) from the class-conditional release against
  0.651 / 0.662 / 0.662 from the pooled one with Gemini 3.5 Flash (Welch p = 0.007 / 0.017 /
  0.029), and 0.670 / 0.723 / 0.716 against 0.652 / 0.673 / 0.664 with Claude Fable 5, where the
  tree students reach the real-sample floor of 0.695 / 0.727 / 0.717 and held-out conditional error
  rises 0.008. On Adult two Fable 5 draws score 0.846 / 0.885 / 0.873 against a real sample's
  0.830 / 0.870 / 0.840 (development measurement).
- `release_statistics(..., class_conditional=False)` restores the pooled release.

## 4. Exact counts and selection: how the output reaches the release's own fidelity

Two steps, both post-processing of the release, both on by default.

- **Exact-count batches.** Every batch is told the number of rows it owes per bin of each numeric
  column, per category of each categorical column, and per outcome. The counts are apportioned from
  the released histograms for the cohort as a whole and updated after each accepted batch, so a
  batch returning more or fewer valid rows than asked cannot leave the cohort short. A frontier
  model given these counts reproduces them exactly; given shares, it matches about two thirds of
  them. In a cohort with class blocks the counts are stated per outcome (the positives' rows per
  bin from the positive block, the negatives' from the negative block), so each class's marginals
  are enforced rather than read from the blocks; `Generator(quota_by_class=False)` states the
  totals only. `Generator(quota=False)` turns the counts off.
- **Selection from a pool.** `generate_selected(release, n_rows, pool_factor=3)` asks for three
  times the rows and keeps the `n_rows` whose cell counts match the release: inclusion weights by
  iterative proportional fitting over every released cell, systematic sampling, then a greedy
  exchange of rows that lowers the weighted distance to the released counts, with the pooled
  marginals weighted above the per-cell counts and any suppressed category given zero mass. A
  cohort whose release carries only a pooled block is constrained at cohort level, not per class,
  because imposing a pooled histogram on each class erases the feature-to-target structure; that is
  a regression test. The release carries the cohort rule it was built under, so selection works on
  auto-configured releases whose cohorts the caller's schema does not name.
- **A pool that stopped early is refused.** Selection can only choose among rows that exist. A
  generation run that hit its spend cap or lost its connection leaves its later cohorts unfilled,
  and no reweighting can repair a cohort share from rows that are not there. `select_to_release`
  checks every released cohort's rows against the rows it owes (`pool_coverage` reports them) and
  raises, naming the short cohorts, unless `allow_short_pool=True`.
- **Categorical cells are checked against the declared set.** A model that writes an integer-coded
  category with a decimal point (`2.0` for a column declared as `1`/`2`) would otherwise re-type
  the whole pool as floats on concatenation, and the selection step and every consumer would then
  see a category the data does not have. The parser normalises integer-valued floats to the
  declared string and drops rows whose value is not declared, counted in
  `stats.rows_undeclared_category`, exactly as an out-of-domain number is dropped.
- **Values inside a bin are redrawn from the release.** A released histogram fixes bin counts and
  nothing finer, so where a value lies inside its bin is never released information. In cohorts
  whose release carries class-conditional blocks, `generate_selected` redraws every numeric value
  uniformly inside its released bin, never crossing the row's stratification band, so the output's
  structure below bin resolution is the release's rather than the generator's. In cohorts with only
  a pooled block the generator's values are kept, because there they are the only carrier of the
  class signal. Measured on every stored development arm, this lifted NHANES's logistic-regression
  student from 0.729 to 0.752 against a real sample's 0.773 and changed no binned measure anywhere.
  `sub_bin="generator"` disables it.

The `mock` backend honours the exact-count block, so the whole generate-and-select loop runs end to
end with no API and no cost; the offline test in `tests/test_selection.py` does exactly that.

- **Integer columns stay integer.** Whether a numerical column is integer-valued is decided
  from the public bin edges and bounds (all integral) together with the majority of the pool's
  values, not from every value. Under an all-values rule a single model-written `45.0` or `37.5` re-typed the
  whole column, so the redraw inside the bin produced real numbers and delivered rows carried
  fractional ages, hours and capital values, which an evaluator that compares exact values treats
  as far from the data. The fix delivers every value of such a column as an integer, including the
  kept values of pooled-only cohorts. The test `tests/test_selection.py` plants three fractions in
  a pool of 3,000 rows and checks that none is delivered.

Measured during development on Adult and finance with Gemini 3.5 Flash: 1-way marginal error
below a real sample of the same size, conditional error about a quarter of a real sample's, and
downstream utility at the real-sample level; the comparison with the library-default AIM and MST
that accompanied it is not reported, because of the sampler defect (section 11). The ceiling is
the release's own distance from the truth, which is why the conditional table's share of the budget
defaults to a fifth (`conditional_fraction=0.2`): the histograms take the rest, and the release's
marginal error halves for no measurable loss elsewhere. The cost is `pool_factor` times the
generation spend, and roughly twice the thinking per call on models that reason over the counts.

## 5. Guardrails you cannot turn off by accident

Each guardrail corresponds to a defect that produced a plausible but wrong conclusion during the
research behind this tool.

| Guardrail | The wrong conclusion it prevents |
|---|---|
| Model capability gating | A 7B model scored a transmission slope of 0.21 while emitting perfectly valid CSV. It fails silently, so it is refused rather than warned about. |
| Hash-locked prompts | Removing the conditional table or the "even if counterintuitive" instruction collapses the mechanism to an unconditioned model, while the pipeline keeps producing plausible rows. |
| Context-fit check | Ollama's 4096-token default silently truncated the prompt, cutting off the conditional table, which looked like "this model family ignores the statistics". |
| Empty-content detection | A reasoning model spent its whole output budget on hidden chain-of-thought and returned `content=""`, which looked like "this model cannot follow the schema". |
| `keep_default_na=False` | pandas reads the string `'None'` as missing. `A1Cresult='None'` means the test was not ordered; the default deleted 83% of rows at a 100% call-success rate. |
| Domain-bounds rejection | A degenerate response produced a credit limit of 34 against a declared floor of 10,000; without the check it entered the dataset silently. |
| Declared-category check | One batch wrote an integer-coded category as `2.0`; concatenation re-typed the whole pool, and the selection step and the evaluator then saw a category the data does not have (conditional error 0.22 against 0.01 from a pool whose generation was fine). |
| Pool-coverage guard | A run that stopped at its spend cap left two cohorts with 7 rows each; selection cannot fill a cohort from rows that do not exist, and the output's cohort shares were wrong while every per-row check passed. |
| Bands tile the domain | Half-open autoconfig bands left the records at the domain maximum in an `oob` cohort that was released like any other and that no generated row could ever join. |
| Published cohort size floored at `n_min` | At ε = 0.3 a 469-record cohort's noised size clipped to zero and the generator gave it 1 of 600 rows: a whole age band missing from the output while every per-row check passed. A cohort is released only because it holds `n_min` records, so the published size is clamped there (post-processing). |
| Published cell support floored at `n_min` | The same rule, for the conditional table. A stored NHANES release carried a level-1 cell with rate 0.078 and support 0, and cell-wise generation allocates rows in proportion to support, so that cell would have received none. A cell is released only because it holds `n_min` records, so its published support is clamped there (post-processing of the noised count). `cortec-hybrid`'s table has the same clamp. |
| Proportional row allocation | Uniform allocation over-represented a 2%-of-population cohort by 12×, so every marginal measured afterwards described a deliberately wrong mixture. |
| Validation before spending | Privacy budget cannot be refunded, so schema conflicts are raised before the first query. |
| `n_min` floor | A conditional rate over too few records is dominated by its own Laplace noise. |

Run them with `pytest tests/`. Each test is named after the defect it prevents.

## 6. Cell-wise generation: accurate absolute rates, and the coverage guard

`Generator.generate()` asks the model for a mixed batch of records and lets it allocate them across
the released conditional cells, which caps how finely any one cell's rate can be expressed. On UCI
Adult with 12 rows per call, `education = Masters` receives 0.65 rows per call, so it can only emit
0% or 100%; its released rate of 0.554 exceeds a half, so it rounds to 100% every time. Measured end
to end this gave MAE 0.253 and slope 1.64: rates below about 0.19 were accurate and rates above
about 0.24 saturated towards 1.0. The apparent rate threshold was an artefact: on this dataset the
rare education values happen to be the high-income ones.

`Generator.generate_by_cell()` removes the coupling. Each call covers exactly one released cell, so
the row count is known, and the prompt states the outcome as an integer count ("exactly 7 of these
12 records must have income = '>50K'") rather than a probability. The model has no discretion over
the target column, and the residual error is bounded by the stochastic rounding of a single row.

| | released | `generate()` | `generate_by_cell()` |
|---|---|---|---|
| Masters | 0.554 | 1.000 | **0.625** |
| Bachelors | 0.419 | 0.968 | **0.408** |
| Assoc-acdm | 0.238 | 0.818 | **0.300** |
| Some-college | 0.190 | 0.189 | 0.206 |
| HS-grad | 0.159 | 0.148 | 0.155 |
| | | MAE **0.253**, slope 1.64 | MAE **0.037**, slope 1.19, r 0.99 |

These figures are one development run on UCI Adult with Claude Fable 5 (12 rows per call, one
draw) and are not among the paper's audited numbers. The paper's audited measurement of the same
change is on the constructed registry, where conditional magnitude error fell from 0.155 to 0.002
(Appendix E). Cell-wise generation costs more calls for the same output size (34 calls for 300 rows
here, against 25). Two caveats remain. Rounding is stochastic, so a cell allocated very few rows
still carries up to half a row of error in a single draw; draw more records, which is free. And
this was measured on one dataset with one model, so run a small live batch against your own schema
and compare the per-cell rates of the output with the release before relying on the numbers.

**The coverage guard, and what it does not cover.** Cell-wise generation emits rows only for
released cells, so any band of a conditioning column that no released cell names is produced at
essentially rate zero. On one health-survey release that erased four of six racial groups to
exactly 0.000 while aggregate fidelity and downstream utility both stayed healthy. So
`generate_by_cell()` refuses when any conditioning column has less than 90% of its released
marginal mass inside released bands. It names the offending column and points at the settings that
change it (`n_min`, conditional depth). The check reads the release only, so it costs no budget.

The guard answers one question soundly and for free: which bands of a column did the released cells
span? It does not answer the one next door: what is inside a band they did span? On a coarsened
high-cardinality categorical those differ, and a column scoring 100% coverage was still destroyed.
Its cells were keyed on an opaque group label, the generator was told which values were legal and
not how often each occurs, and it spread them near-uniformly: a discharge code at 59.2% of real
records came out at 14.3%, which is 1/7 on a seven-member group. The tool now expands every
coarsened group into its members with their within-group shares, a renormalisation of proportions
already in the release that costs no budget, and that repairs it on both vendors' models we tested.
A fix for one named failure mode is not a guarantee about the mapping in general, which is why the
The setup guide calls the guard a heuristic mitigation and the general question an open problem.

## 7. Reasoning: the largest single effect, and the trap in its economics

On one model tested against itself, with the same release, the same prompts, one flag changed and
three draws per arm, conditional error moved from 0.173 ± 0.032 to 0.045 ± 0.007, a 3.8×
improvement (Cohen's d = 5.48). That takes it from worse than a no-information control (0.152) to
among the best generators measured. `Generator(..., reasoning="on")` is the default.

The trap is the economics. The suppressed setting is 14.5× cheaper, and across the same three draws
per arm downstream utility does not separate at all: TSTR-LR reads 0.786 ± 0.050 suppressed against
0.802 ± 0.025 enabled, well inside one standard deviation. So a team optimising cost against a
standard TSTR check does not see a small difference and accept it. It sees no measurable
difference, concludes correctly on that evidence, and ships a pipeline carrying none of their data's
conditional structure. Only a conditional measure reveals it. At one draw reasoning appeared to
improve downstream utility too; that did not survive replication, and no such claim is made.

Setting the flag is not evidence that it fired. We once read a model's reasoning as intermittent, a
large count in one batch and none in another, and it was our instrument: the two batches ran on
different transports, and one of them reports no count at all. A count that was never reported is
not a count of zero, which is why `gen.stats` keeps `calls_without_reasoning` and
`calls_reasoning_unmeasured` apart.

## 8. The sweep-only tier, and a claim that did not hold

The capability table in the setup guide is scored on a transmission sweep, which asks whether a model
reproduces a released rate: one large, salient signal in a short prompt. Two self-hosted 70B models
pass it inside the frontier band, and on that basis alone a capability table would call them
validated and recommend them. Re-measured on full-dataset generation at three draws,
`llama3.3:70b-instruct` closes 12% of the conditional-error gap to an unconditioned prompt, closes
nothing at all on 2-way total variation (0.238 against the control's identical 0.238), and is worse
than that control on 1-way total variation. On the metrics this method exists to improve, its
output is close to indistinguishable from asking a model for plausible records with no
conditioning, and no output check reveals it. Passing the sweep is necessary, not sufficient. These
models are refused by default and run only under `acknowledge_insufficient=True`.

Capability is family-dependent, not just size-dependent: at comparable scale a 24B model recorded
lower error than a 32B one, so a size threshold alone would be wrong.

Some models are measured but not listed, and the gate tells you so. GPT-5 and Gemini 3.1 Pro were
measured on full-dataset generation (conditional error 0.045 and 0.041, both inside the band of the
validated frontier models and below what a real 300-record sample achieves) but not on the
transmission sweep the tiers are scored against. Those are different experiments and their numbers
are not interchangeable: a model can track a forced rate in three cohorts while attending weakly to
a full multi-level table, and nothing in the output says which is happening. So they are still
gated, and the refusal message reports exactly what we do and do not know.

**A claim that did not hold.** The transmission sweep alone had suggested that 70B-class models are "validated
and self-hostable, so a regulated deployment needs no external API and no data leaves the
institution." The first half does not hold: that validation was measured on the transmission sweep only.
On full-dataset generation the same class of model closes only 12% of the gap between an
unconditioned prompt and a frontier model on conditional-seen error, 20% on held-out, nothing at all
on 2-way total variation, and is worse than an unconditioned prompt on 1-way. The failure is silent
and the marginal metrics do not reveal it. The second half was never the load-bearing part. Because
the prompt carries a DP release and never a record, an enterprise endpoint under your own tenancy
is available to you, and the generator choice is a quality decision. If you do run local weights,
verify conditional fidelity on your own data first.

## 9. The trust boundary, in full

No private record is ever sent to the model. The prompt carries only released statistics: noised
histograms, class balances and a conditional table. Generation is post-processing of a DP release,
so by post-processing immunity whoever runs the model learns nothing beyond what the release
already discloses, and the ε guarantee does not depend on where that computation happens.

The usual objection to an LLM-based method, that data is being sent to a third party, therefore does
not apply. That is not because the endpoint is trustworthy but because there is no private data in
the request. The decoder may be a tenant-isolated enterprise endpoint (Bedrock in your own VPC,
Azure OpenAI, Vertex AI) under a BAA, or an air-gapped model on your own hardware with no network
egress; the argument is the same. What differs between those is capability, not trust, which is why
the deployment recommendation names enterprise platforms even though the privacy argument permits
local weights.

The guarantee says nothing about the model's pretraining corpus. If a private record was in it,
that happened before this tool ran. The sharper version of the concern is that conditioning on true
marginals for a narrow stratum resembles a prompt-based extraction attack: it tells the model which
region to sample from, and a memorised record there becomes more reachable. DP does not exclude
this, because such a record is not a function of your release. In the research behind this tool,
four membership-inference attacks (nearest-neighbour, exact-match, shadow-model, and a per-record
likelihood-ratio test, each validated on a positive control) found no advantage above chance and
zero exact matches on any dataset. That is evidence about the published output, not a proof about
the corpus, and it is the strongest statement available.

## 10. The ledger and the suppression charge

Every privatised query is recorded, and `PrivacyLedger.spend()` is the only function in the
package that returns a noise scale, so nothing can be released without appearing in the audit
trail. `release.audit` records every query with its epsilon, sensitivity, composition rule and
partition key, the per-group totals, and the composition rules applied. `cortec/accounting.py` is
written to be read end to end by someone auditing it, independently of the rest of the package.

Which cohorts, class blocks and cells appear is decided by a noisy threshold: every count that
gates a release is itself a charged, noised count, and the comparison with the public floor
`n_min` is post-processing of it. The release queries every cohort and every cell of the public
domains, present in the data or not, so the set of queried groups is a function of the schema
alone. A separate `suppression` entry charged while the decision still reads the exact count would
not make a deterministic function of private data private; that budget share strengthens the
count queries instead (about 10% of the available budget at the defaults, against 5% without it). The price is that a group near the floor can fall on
either side of it: a group of true size `n >= n_min` is dropped with probability
`0.5 * exp(-(n - n_min) / b)`, where `b` is the count's Laplace scale, and a group below the floor
is released with the mirror-image probability. Pass `charge_suppression=False` to read the exact
counts, the treatment of the marginal-synthesis literature; the stated guarantee does not cover
those decisions and the audit records the choice. Auto-configuration ranks conditioning columns by mutual information and has no concept of
a person, so it can select a column that is itself a proxy for how many rows someone contributes:
on the hospital dataset of the setup guide's privacy-unit example it chose `number_inpatient`,
correlation 0.733 with encounter count. That does not change ε, but it means the release is
stratified by contribution count.

## 11. The paper's measurements: the release on its own, and what the backbone adds

The paper measured the mechanism with the controls a reviewer asks for, and
three of its findings bear on this package's defaults.

**The release carries the result.** The same release decoded with no language model at all, by the
maximum-entropy table that fits every released count (Private-PGM fed exactly the release, no
further noise, no further query), trains students within 0.012 AUC of the package's output on Adult
at n = 300 and within 0.005 at n = 1,000 over three seeds, is ahead of it on pairs and on every
student on NHANES, and is at a real sample's floor on single columns on bank marketing. The
language model's one consistent gain over that decode is on the conditionals the release names
(seen-conditional error 0.027 against the decode's 0.082 on Adult at n = 300, lower at 3 of 3
seeds), and that gain is the selection step's: the decode's own pool passed through
`select_to_release` reads 0.017 on the seen conditionals and 0.035 on single columns at n = 300 over
three splits (0.007 and 0.027 at n = 1,000), with students above every CoRTeC arm's. Per pair, the model's error is below the decode's on the pairs it recognises as logical
(education with education number, marital status with relationship) and above it on pairs it
expects and the data lacks (four near-independent pairs on bank marketing). The decode is free and
is the first thing to compare your output against.

**The model's edge is the column names.** With every column, category and label renamed to a
neutral token, the same releases and prompts give, on Gemini, pair error 0.148 against 0.110 with
the real names (the decode's level, 0.146) and students 0.745, 0.840, 0.798 against 0.805, 0.863,
0.838; the single-column error stays at 0.030 against the decode's 0.058, because that edge is the
selection step raking the pool to the released marginals, which works whatever the columns are
called. On Qwen3-14B every measure is worse without names. An institution whose schema the model
does not know should expect the decode's result, and can have the selection step's single-column
and seen-conditional precision by passing the decode's pool through `select_to_release`. Without
any release the same backbone recites: asked for rows from the column names alone, it copied the
public Adult file's first 25 rows into about 70 % of its output. The package therefore reads the column names as
part of the prompt on purpose, and the setup guide says what to expect when they carry no meaning.

**The package decodes without a model.** `decode_release` and `cortec decode`
fit the maximum-entropy table to the released counts (Private-PGM's mirror descent on the junction
tree, each count at the noise scale its budget implies, a corrected sequential sampler), sample a
pool of `pool_factor × n_rows` records and pass it through `select_to_release`. It is the paper's
decoder-with-selection control: on Adult at ε = 2 the research pipeline's release gave 0.033, 0.118
and 0.014 on single columns, pairs and the seen conditionals at n = 300 (0.027, 0.089 and 0.008 at
n = 1,000) with students at 0.824, 0.866, 0.862 and 0.836, 0.873, 0.874, one split. The package's own
release of the same split (its Stage A with the auto-configurator, which chose three age cohorts and a
relationship-by-marital-status hierarchy, at the package's budget split), decoded the same way, reads
0.019, 0.108 and 0.026 on single columns, pairs and held-out conditionals at n = 300 with students at
0.826, 0.873, 0.860, and 0.016, 0.081 and 0.016 at n = 1,000 with students at 0.841, 0.884, 0.884, which is
level with the reference AIM's students at the larger size; its seen-conditional error (0.053 and 0.026)
is measured on the research schema's cohort bands, which that release does not condition on. Over the
three splits of the paper's replication (seeds 42, 43 and 44, a fresh release each; on seed 43 the
auto-configurator chose education-number cohorts and a relationship-by-native-country hierarchy instead)
the means are 0.020, 0.109
and 0.028 at n = 300 with students 0.830, 0.870, 0.859, and 0.016, 0.082 and 0.017 at n = 1,000 with
students 0.840, 0.879, 0.881, against the reference AIM's 0.052, 0.120, 0.048 and 0.832, 0.866, 0.839, and
0.031, 0.074, 0.031 and 0.847, 0.882, 0.876: ahead of AIM on single columns at both sizes, level with or ahead of it on the students, behind on
pairs at the larger size only; its conditional cells are not compared, because the evaluator's seen
bands are the fixed rule's cohorts and two of its three held-out families are columns this release
conditions on (over the one clean family it is ahead at both sizes). Three splits, three releases; in the paper as
the "auto-configured" rows of Tables 3 and 6 and Appendix E.2, beside the rows of the schema's own cohort rule
fixed before the campaign. The path spends no budget and
needs no key; the `pgm` extra installs Private-PGM and JAX.

**Capability improves single columns and formatting, not utility.** One Adult release decoded
through Qwen2.5-7B, Qwen3-8B, Qwen3-14B, Qwen3-32B (4-bit AWQ) and Gemini under the same prompts and
flags: 1-way total variation 0.110, 0.068, 0.046, 0.049 and 0.027 at n = 300, saturating at 14
billion parameters; downstream AUC 0.80 to 0.82, 0.85 to 0.86 and 0.82 to 0.84 on every rung, with
the model-free decode inside that band. The 7B model wrote its own prior for marital status and
education over the released distributions; the 8B model writes range labels into numerical columns
at times. The capability table's Qwen3 rows (section 8 of the setup guide) come from this ladder.
The same Qwen3-14B fine-tuned on the private rows under DP-SGD at the same ε reads 0.100 and 0.172
on single columns and pairs at n = 1,000 against 0.036 and 0.108 as a frozen decoder of the
release: where the budget is spent decides more than the backbone does.

**The reference baselines, and a comparison that is not reported.** During development the package was
compared against AIM and MST at the smartnoise-synth defaults, which gave a utility margin of 0.15 AUC. An
audit found that the library's sampler (mbi 1.1, used by the wrapper and by the authors' reference
code) read conditional tables against the wrong axes when its column order differed from the
table's, so single-column counts survived and the joint structure of some columns was scrambled.
The paper's rows are the authors' private-pgm mechanisms at their defaults with a corrected sampler
and the stable fitting routine; the reference AIM is at the real sample's floor on Adult utility at
n = 300, ahead of the package on pairs at every seed and, against the fixed cohort rule, on utility at
n = 1,000 (level with or behind the auto-configured release decoded with no model and the selection step, section 11), level on
single columns, and behind on the seen conditionals (0.055 against 0.027). That margin was the sampler's and is not reported anywhere. Seen from AIM's side, the release is a fixed workload (class-conditional 1-way marginals inside cohorts and the conditional cells) decoded by the estimator AIM uses; AIM's adaptive selection gives the pairs, the fixed workload gives the named conditionals, an inspectable artefact and a release that takes seconds.

**Privacy of the draws, measured.** Four membership attacks on 3,000 candidates, validated on a
positive control: the strongest advantage is 0.022 on Qwen3-14B and 0.035 on Qwen3-32B on Adult, and 0.049 on
the 32B on NHANES, against a permitted 0.762, with no exact match in any arm; no synthetic row copies a training row, and no arm
is closer to training rows than to held-out rows (share 0.47 to 0.54 against 0.50 expected). The
served Gemini model reproduces the first 25 rows of the public Adult file from its header and none
of them appears in any draw.

**Cost.** A 300-row Gemini draw with a threefold pool is 41 calls and 1.56 dollars at list price
(0.75 and 3.75 dollars per million input and output tokens on 28 October 2026); the same draw on
Qwen3-14B is 42 calls and about 50 minutes on one rented A40.
