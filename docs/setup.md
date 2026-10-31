# cortec

A differentially private release for synthetic tabular data, decodable by any generator. Whether the
decoder is a frozen model or the maximum-entropy decoder with no model, it reads only DP statistics.

The privacy budget is spent once on a release of statistics (Stage A). A frozen model, or the
maximum-entropy decoder with no model at all (`cortec decode`), then generates records from that
release (Stage B). It never sees a private record, so generation is post-processing. It adds no
privacy cost, and you can draw as many datasets as you like from one release. An optional Stage C
attaches a utility bound to the output at a budget of its own.

```
private data ──► DP release (Stage A) ──►│──► frozen LLM (Stage B) ──► synthetic records
                 the budget is spent     │     reads only the release, ε = 0
                 here, once              │     repeatable for free
                                    nothing right of this line
                                    ever sees a private record
```

This document is the setup guide. Read it in order the first time. Each step names the call it uses,
the settings that matter and what to check before moving on.

| Step | What you do |
|---|---|
| 1. Install and run | Install the package, then run the whole pipeline on your table with one command |
| 2. Choose where the model runs | Pick the surface and set its credentials |
| 3. Prepare the data | One table, one row per person, string categoricals |
| 4. Declare the schema | Public facts only: columns, bounds, bins, target |
| 5. Stage A | Spend the budget once, check the audit, store the release |
| 6. Stage B | Generate from the release and check the run |
| 7. Stage C | Bound how far the output is from the private data |
| 8. Evaluate | Compare against a real sample and a permuted floor |
| 9. Trust boundary | What the model sees, and what the guarantee does not cover |
| 10. Troubleshooting | What each refusal means and what to change |
| 11. Output and exports | One printed layout and one exportable record for every stage |

Two further documents may help. The [deployment guide](../paper/CoRTeC_deployment_guide.pdf) is the
manual for a regulated deployment. It covers the architecture, the parameters, the full prompt, a
checklist, the controls mapped to standards, the failure modes with their guards and the cost.
[`design-notes.md`](design-notes.md) records the measurements behind the defaults.

## 1. Install and run

You need Python 3.10 or later. The package depends on numpy and pandas. Each vendor SDK is an
optional extra. The package is not on PyPI, so install it from a clone of this repository:

```bash
git clone https://github.com/Calyie/cortec
cd cortec
pip install '.[anthropic]'     # Claude
pip install '.[openai]'        # GPT, and any server with an OpenAI-compatible interface (vLLM)
pip install '.[gemini]'        # Gemini
pip install '.[ollama]'        # local models through Ollama, for scientific controls only, NOT a deployment path (step 2)
pip install '.[pgm]'           # decode a release with NO language model (Private-PGM and JAX, a large install)
```

One install gives both packages (`cortec` and `cortec_hybrid`) and both commands (`cortec` and
`cortec-hybrid`). The enterprise surfaces have their own extras: `bedrock` (adds request signing for
AWS), `azure` and `vertex`. Install `'.[dev]'` as well to run the tests:

```bash
python3 -m pytest tests -q     # offline, each test named after the defect it prevents
```

**Run it.** Installing the package puts the `cortec` command on your PATH. `cortec run` runs the
whole pipeline on your own table from any directory: Stage A (the release), Stage B (generation),
Stage C (the bound) and the evaluation. It prints each stage in the standard layout and writes every
file into one folder, which it names at the end. The run opens with a banner that says what is about
to run. It closes with a results panel on one screen, which is also saved as `RESULTS.md`. The panel
shows the marginal fidelity and the downstream utility. Each is placed against the range that real
samples of the same size span on the same holdout. The panel also gives the Stage C verdict, the
privacy spent and the cost. The command needs two inputs. The first is the private table as a CSV
with one row per person (step 3). The second is a Python file that declares the schema from public
facts only (step 4). For example:

```python
# my_schema.py
from cortec import Schema

SCHEMA = Schema(
    name="encounters",
    numerical={"age": (18, 95), "length_of_stay": (1, 30)},
    categorical={"admission_type": ["Emergency", "Elective", "Urgent"],
                 "a1c_result": ["None", "Norm", ">7", ">8"]},
    target="readmitted_30d", positive="YES", negative="NO",
    bins={"age": [18, 40, 55, 70, 95], "length_of_stay": [1, 3, 6, 10, 30]},
)
```

```bash
# offline first: no key, no spend, the whole pipeline on the mock backend
cortec run --data private.csv --schema my_schema.py --backend mock

# then a real model, with a spend cap. The run keeps the rows completed within it
export ANTHROPIC_API_KEY=...
cortec run --data private.csv --schema my_schema.py \
    --backend anthropic --model claude-fable-5 --n-rows 300 --budget-usd 5

# every setting, with its default
cortec run --help
```

Without `--holdout`, one fifth of the data is set aside before Stage A for the Stage C ceiling and
the evaluation. The output folder (by default `results/<schema name>_<backend>`) holds `RESULTS.md`
(the results panel), `synthetic.csv`, `bound.json` and one JSON record for each stage and for the
panel. It also holds `release.json`. Later runs into the same folder reuse it, because a release is
made only once. `--exports all` also writes each record's Markdown, CSV and figure.
`RunRecord.from_json` can produce these later from the JSON. The sections below explain each stage
and every setting. The same calls are available from Python for your own scripts. `cortec-hybrid
run` runs the correction. In a clone whose scripts directory is not on PATH, `python3 run_cortec.py`
in this directory is the same command.

For a full tour of the three stages with no API key and no data, run the bundled example. It builds
a small synthetic table in code and uses the offline `mock` backend. The release and the bound are
real, and only the generated numbers are placeholders:

```bash
python3 example.py
```

To list everything the package can run, print the help page. It shows each function in order, with
its arguments and defaults read from the code. Give it a function's name for that function's full
documentation. The `cortec` command does nothing else. The package is a library, and the runs are
Python calls.

```bash
cortec --help                        # the commands, the run order, functions, classes, refusals
cortec bound_with_controls           # one name in full
cortec-hybrid --help                 # the same page for cortec-hybrid
python3 -m cortec --help             # the same, when pip's scripts directory is not on PATH
```

## 2. Choose where the model runs, and set the credentials

The generator is called through one of four backends. Three of them can reach either the vendor's
public API or an enterprise surface inside your own cloud account. The `model` name is the same on
both. The surface changes only the client and the credentials.

| Backend | Surface | Set before running | `Generator(...)` arguments |
|---|---|---|---|
| `anthropic` | public API | `ANTHROPIC_API_KEY` | `backend="anthropic", model="claude-fable-5"` |
| `anthropic` | Claude on AWS Bedrock (recommended) | AWS credentials from the standard chain (environment, profile or instance role) and `AWS_REGION`. The `bedrock` extra | `surface="bedrock", surface_model="<Bedrock model id or inference-profile ARN>"` |
| `openai` | public API | `OPENAI_API_KEY` | `backend="openai", model="gpt-5"` |
| `openai` | a self-hosted server with an OpenAI-compatible interface (vLLM) | `OPENAI_BASE_URL` (for example `http://127.0.0.1:8000/v1`) and any value in `OPENAI_API_KEY` | `backend="openai", model="Qwen/Qwen3-14B"`. The paper's Qwen3 arms ran this way on one rented graphics card. `spend_usd` reads 0 for such a model (the card is paid by the hour), so the budget cap does not apply, and `describe_surface()` reports the surface as `public` because the package cannot tell a local server from the vendor's |
| `openai` | GPT on Azure OpenAI (recommended) | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, optionally `AZURE_OPENAI_API_VERSION` | `surface="azure", surface_model="<your deployment name>"` |
| `gemini` | public API | `GEMINI_API_KEY` | `backend="gemini", model="gemini-3.5-flash"` |
| `gemini` | Gemini on Google Vertex AI (recommended) | Application Default Credentials, `GOOGLE_CLOUD_PROJECT`, optionally `GOOGLE_CLOUD_LOCATION` (default `global`) | `surface="vertex"` |
| `ollama` | a local Ollama server | `ollama_url` (default `http://localhost:11434`) | **not an enterprise surface, out of deployment scope**. The capability gate refuses these models by default |
| `mock` | none | nothing | offline dry runs of the whole pipeline |

A reasoning model served through Ollama needs three settings that `cortec run` exposes. An example
is OpenAI's open-weight `gpt-oss:120b`, standing in for a model an institution hosts itself. Set
`--effort` (`low`, `medium` or `high`) for its thinking level. Set `--max-output-tokens` high enough
that the reasoning never uses up the whole budget before the CSV is written (32,768 at the low
level). Set `--request-timeout` to several minutes per call. The capability gate has not measured
that model, so `--allow-unvalidated` is required and the output says so.

**Which endpoint to point this at.** For regulated data, use the enterprise-hosted, tenant-isolated
surface in your own cloud account (Bedrock, Azure OpenAI or Vertex AI). It should have private
networking, data residency, a contract that excludes training on your inputs and a BAA where HIPAA
applies. The vendors' public developer APIs are supported but are not recommended for regulated
deployment. The weights are the same but the contract and the network path are not, and those are
what a compliance review assesses. The privacy argument does not depend on the endpoint because the
request carries only the DP release (step 9). The compliance argument does. The paper's own
measurements were made on the public APIs and on self-hosted open models. The deployment guide says
which results that affects.

**What is verified.** The tests construct each SDK's real client class offline and replace only the
transport. An SDK change therefore fails in CI rather than at your first call.
**Vertex AI is verified live.** We ran the `gemini-3.5-flash` model name (served by Gemini 3.6 Flash
since 8
October 2026) through a service account that holds only `roles/aiplatform.user`. Both generation
paths worked and reasoning was reported on every call. The newest Gemini models are served from the
`global` location and return 404 in every regional one. A model 404 is treated as fatal and is not
retried. **What is not verified:** a live round trip on Bedrock or Azure OpenAI, because no account
was available to us. Treat your first run there as that test: generate a small batch, check
`gen.stats.calls_with_reasoning_block` and `gen.stats.thinking_tokens`, and compare the output with
the release before generating at scale.

If a setting is missing or the surface does not serve the chosen backend, constructing the
`Generator` fails before any client exists. `gen.describe_surface()` returns one line for your audit
trail. A run on a public surface says so in `gen.stats.warnings`.

## 3. Prepare the data

Stage A reads one pandas DataFrame with exactly the columns the schema declares.

- Categorical columns hold strings. Integer codes are fine as strings (`"1"`, `"2"`). If a real
  category is spelled `None`, `NA` or `nan`, read the file with `keep_default_na=False`. Otherwise
  pandas turns it into a missing value.
- Numeric columns hold numbers inside the declared bounds. Values outside are clipped and the
  validation report says how many. If that number surprises you, the declared bounds are wrong.
- The target column holds exactly two labels: the ones declared as `positive` and `negative`.
- **One row per person.** ε protects one row. If a person contributes `k` rows, that person's
  guarantee is `k · ε`. Count the rows per person before Stage A. If the maximum is above 1, cap
  each person's rows or aggregate to one row per person. Step 5 explains the declaration.

Validation runs before any budget is spent and reports every problem at once.

## 4. Declare the schema

A `Schema` is a public description of the table. Everything in it must be knowledge you have without
looking at the private file. That means the column list, domain bounds from documentation or
convention, bin edges from clinical or regulatory practice and the target. Never derive a bound or
an edge from the data's own quantiles. A schema inferred from private values leaks through its
bounds. This is why `Schema.from_dataframe()` does not exist.

```python
from cortec import Schema, Band

schema = Schema(
    name="encounters",
    numerical={"age": (18, 95), "length_of_stay": (1, 30)},
    categorical={"admission_type": ["Emergency", "Elective", "Urgent"],
                 "a1c_result": ["None", "Norm", ">7", ">8"]},   # 'None' is a real category
    target="readmitted_30d", positive="YES", negative="NO",
    bins={"age": [18, 40, 55, 70, 95], "length_of_stay": [1, 3, 6, 10, 30]},
    stratify=[("length_of_stay", [Band("los1-2", 1, 3), Band("los3-5", 3, 6),
                                  Band("los6+", 6, 31)])],
    conditional=[("admission_type",), ("admission_type", "a1c_result")],
)
```

| Field | What it declares | Notes |
|---|---|---|
| `name` | an identifier used in file names and the audit trail | |
| `numerical` | `{column: (lo, hi)}`, the public domain of each numeric column | values outside are clipped |
| `categorical` | `{column: [allowed values]}` | every declared value stays in the release, even when rare |
| `target`, `positive`, `negative` | the binary target and its two labels | the target is not listed as a feature |
| `bins` | `{column: [edges]}`, public histogram edges | default: 10 equal-width bins over the bounds |
| `stratify` | `[(column, [Band(name, lo, hi), ...])]`, the public rule that forms cohorts | the bands must tile the declared domain with no gap. The top band is closed at its upper edge |
| `conditional` | the levels of the conditional target table, coarse to fine, as column tuples | leave it out to have it derived (below) |
| `coarsen` | `{column: {declared value: group}}`, grouping a wide categorical for conditioning | derived by auto-configuration. May also be supplied |

**Leave `conditional` empty and the hierarchy is derived for you.** Auto-configuration ranks the
columns by their mutual information with the target, using one DP-noised contingency table per
column. From the public cardinalities and the number of records you will request, it chooses how
many columns to condition on. It groups wide categorical columns into a few bands. It forms cohorts
on the highest-ranked numeric column when the data can support them. The selection spends a declared
share of `epsilon_total` (between 5% and 30%, depending on the schema size and the row count). It is
charged to the same ledger. If you declare `conditional` yourself, it is used unchanged.
Auto-configuration has no concept of a person. On data with several rows per person it can select a
column that stands in for how many rows someone contributes. Declare the columns yourself if that
matters.

## 5. Stage A: release the statistics

```python
from cortec import release_statistics

release = release_statistics(schema, private_df, epsilon_total=2.0, n_min=150,
                             max_rows_per_person=1)
print(release.audit["epsilon_accounted"])      # 2.0
release.to_json("release.json")                # contains no private record
```

This is the only call that reads the private data. It forms the cohorts from the public rule. It
releases one Laplace-noised histogram per column and cohort (or one per outcome class where both
classes clear `n_min`). It also releases the class balance, the noised cohort and cell sizes and the
conditional target table. Every query is recorded in a ledger.

| Parameter | Default | What it does |
|---|---|---|
| `epsilon_total` | `2.0` | the whole row-level budget. Every query is charged to it |
| `n_min` | `150` | cohorts and cells with fewer records are suppressed. The floor is 50, below which a rate is dominated by its own noise |
| `conditional_fraction` | `0.2` | the share of the budget for the conditional table. The histograms take the rest |
| `max_rows_per_person` | `None` | the privacy unit: the most rows one person contributes. If it is left out, the audit records `UNDECLARED` |
| `acknowledge_vacuous_privacy_unit` | `False` | required to proceed when `max_rows_per_person × epsilon_total` exceeds 10 |
| `class_conditional` | `True` | one histogram block per (cohort, outcome) where both outcomes clear `n_min`. `False` restores a pooled block |
| `autoconfig` | `True` | derive the hierarchy when `schema.conditional` is empty |
| `n_records` | `1000` | how many records you intend to generate. Auto-configuration sizes the table against it |
| `charge_suppression` | `True` | decide which cohorts, class blocks and cells appear by comparing their noised, charged counts with `n_min` (a noisy threshold, so the decision is post-processing). `False` reads the exact counts, the literature-standard treatment, which the stated guarantee does not cover. The choice is recorded |
| `seed` | `None` | tests only: a seeded release is reproducible, so its noise can be subtracted and it carries no guarantee. The audit says so |

**The privacy unit.** ε protects one ROW, and on some data that is not one person. Under group
privacy a person who contributes `k` rows receives `k · ε`. On encounter-level hospital data, with
one admission per row and several admissions per patient, this matters a great deal. In a real
dataset of 101,766 encounters from 71,518 patients, the heaviest patient contributes 40 encounters.
A declared ε of 2.0 is then an ε of 80 for that patient, far outside any range normally considered
meaningful. This package is therefore meant for one-row-per-person data. `k` cannot be computed for
you without spending budget, so you declare it. A declaration whose ε per person is meaningless is
refused, not just annotated:

```python
# refused: eps_person = 80
release_statistics(schema, df, epsilon_total=2.0, max_rows_per_person=40)
# ReleaseError: REFUSED: max_rows_per_person=40 at epsilon_total=2.0 gives
#   epsilon_per_person=80.0, past the 10 beyond which a per-person guarantee carries no
#   meaning ... Either aggregate to one row per person before Stage A ...

# proceed anyway, with the acknowledgement recorded in the audit trail
release = release_statistics(schema, df, epsilon_total=2.0, max_rows_per_person=40,
                             acknowledge_vacuous_privacy_unit=True)
release.audit["epsilon_per_person"]           # -> 80.0
release.audit["epsilon_per_person_vacuous"]   # -> True
```

If your data has more than one row per person, aggregate to one row per person or cap each person's
rows before Stage A. Otherwise, report `ε_person` rather than the per-row number.

**Check the release before generating.**

- `release.audit["epsilon_accounted"]` equals what you declared and `within_budget` is true. The
  audit lists every query with its epsilon, sensitivity, composition rule and partition key. A
  privacy engineer can therefore recompute the total by hand. `PrivacyLedger.spend()` is the only
  function in the package that returns a noise scale, so nothing is released without appearing
  there.
- `release.audit.get("warnings", [])` is empty, or you have read each entry. A seeded release and
  an undeclared or meaningless privacy unit are recorded there. Two further conditions are printed
  at release time rather than recorded. One is an under-spent conditional budget, where some
  declared levels released no cell. The other is a noise-dominated conditional table, where the
  noise on a level's rates is at least as large as their spread. Read the console.
- `release.conditional_levels` is non-empty and its finest level has cells. An empty table is
  refused because the output would be unconditioned.
- Store `release.json` and hash it. Generation reads only this file, so every later draw and every
  comparison must reuse it. A release cannot be regenerated identically because its noise is
  unseeded. Re-running Stage A also spends the budget again.

## 6. Stage B: generate

```python
from cortec import Generator

gen = Generator(schema, backend="anthropic", model="claude-fable-5", reasoning="on",
                budget_usd=25.0)
synthetic = gen.generate_selected(release, n_rows=5000, pool_factor=3)
```

| Parameter | Default | What it does |
|---|---|---|
| `backend`, `model` | `"anthropic"`, `"claude-fable-5"` | the vendor and the model. The model must pass the capability gate (below) |
| `surface`, `surface_model` | `"public"`, `None` | the enterprise surface and the identifier it expects (step 2) |
| `reasoning` | `"on"` | `"suppressed"` is measured to be 3.8× worse on conditional error. The tool warns if you choose it |
| `rows_per_call` | `25` | rows requested per model call |
| `quota` | `True` | give every batch the exact per-bin, per-category and per-outcome counts it owes |
| `quota_by_class` | `True` | state those counts per outcome, so each class's marginals are enforced rather than read from the class blocks. `False` states the totals only |
| `budget_usd` | `None` | a spend cap from the vendor's reported token counts at list prices. The run stops when it is reached and keeps what it has |
| `request_timeout`, `max_retries` | `120.0`, `3` | the per-call timeout in seconds, and the retries per batch |
| `allow_unvalidated` | `False` | run a model not in the measured set. Verify transmission on your own data first |
| `acknowledge_insufficient` | `False` | run a model measured as insufficient or sweep-only, for reproducing that finding rather than for use |

There are three generation methods:

- **`generate_selected(release, n_rows, pool_factor=3)`** is the default recommendation. It
  generates `pool_factor × n_rows` records cohort by cohort and keeps the `n_rows` whose cell counts
  match the release. In the cohorts that carry class-conditional blocks, it then redraws each
  numeric value inside its released bin (`sub_bin="generator"` keeps the model's values instead).
  Both steps read only the release, so they cost no privacy budget. The cost is `pool_factor`
  times the generation spend.
- **`generate(release, n_rows)`** is the same without the pool. It is adequate when relative
  ordering is all you need.
- **`generate_by_cell(release, n_rows, level=0)`** makes one call per released conditional cell with
  an integer positive count. Use it when absolute conditional rates matter, such as a readmission
  probability or a default rate. It refuses when any conditioning column has less than 90% of its
  released marginal mass inside released bands. Otherwise rows outside those bands would be missing
  from the output while every aggregate check still looked healthy. `allow_low_coverage=True`
  overrides the refusal for measurement, not for use. Treat the guard as a
heuristic mitigation, not a solved problem. It checks which bands the released cells cover, but not
what happens inside a
  band. How to pass a DP conditional table to a language model without silent loss is still an
open problem. Compare the output column by column with the release anyway.

**Which model.** Capability is checked before any call. A model that is too weak to follow the
release still produces well-formed, plausible rows. Those rows carry none of the release's
structure, and no output check will reveal that.

| Model | Tier | Measured on |
|---|---|---|
| Claude Fable 5 / Opus / Sonnet | validated | transmission sweep, magnitude error 0.009 |
| Gemini Flash (the `gemini.*flash` pattern. `gemini-3.5-flash` is served by 3.6 Flash) | validated | the sweep (0.008) and full-dataset generation (1-way total variation 0.027 on Adult at n = 300) |
| Qwen3-14B, Qwen3-32B (4-bit AWQ), served through vLLM | validated | full-dataset generation (1-way 0.046 and 0.049, seen-conditional error 0.027 and 0.031 on Adult at n = 300) |
| Qwen3-8B | adequate | full-dataset generation (1-way 0.068. Writes range labels into numerical columns at times) |
| llama3.3:70b-instruct | **sweep-only, refused** | the sweep (0.020). Fails full-dataset generation |
| qwen2.5:72b-instruct | **sweep-only, refused** | the sweep (0.030). Fails full-dataset generation |
| mistral-small:24b, gemma2:27b, gpt-oss:20b | adequate | the sweep (0.049–0.078) |
| qwen2.5:32b, qwen2.5:14b | adequate | the sweep (0.098–0.105) |
| qwen2.5:7b | **insufficient, refused** | the sweep (0.406). On full-dataset generation it wrote its own prior over the released distributions |

`print(cortec.models.recommended_table())` lists the models we recommend, measured on full-dataset
generation. A sweep-only model reproduces one forced rate in a short prompt. On full-dataset
generation, however, it is close to indistinguishable from an unconditioned prompt. The design notes
give the measurements. On a ladder of open models decoding one release, single-column fidelity
improved with capability and stopped improving at 14 billion parameters. Downstream utility did not
change with the model at all (design notes, section 11). An unlisted model is refused unless you
pass `allow_unvalidated=True`.

**Enable reasoning and verify that it fired.** Reasoning is the largest single effect measured. On
one model tested against itself, changing that one flag moved the conditional error from 0.173 ±
0.032 to 0.045 ± 0.007. Downstream utility did not separate (0.786 ± 0.050 against 0.802 ± 0.025). A
cost comparison on utility alone would therefore choose the 14.5× cheaper setting and ship output
carrying none of the data's conditional structure. Setting the flag is not evidence that it fired.
Some transports report no reasoning-token count at all. A count that was never reported is not a
count of zero. The stats keep the states apart:

```python
gen.stats.thinking_tokens              # total reported reasoning tokens
gen.stats.calls_without_reasoning      # calls that REPORTED zero while reasoning="on"
gen.stats.calls_reasoning_unmeasured   # calls on which the vendor reported no count at all
gen.stats.calls_with_reasoning_block   # calls where a reasoning block was present (Anthropic)
gen.stats.positives_off_by_more_than_one   # cell-wise calls that missed the "exactly k" by > 1 row
gen.stats.warnings                     # one explicit warning for each condition above
```

**After the run**, read `gen.stats`. It holds `calls`, `parse_ok`, `rows`, `rows_requested`,
`rows_out_of_bounds`, `rows_undeclared_category`, `rate_limit_waits` and `spend_usd`. Rows outside
the declared domain or with an undeclared category are dropped and counted. They are never kept. The
output carries one provenance column that names the released unit each row came from: `_cohort`, or
`_cell` for cell-wise generation. Drop it if your consumer expects only the schema's columns.

**Decoding with no language model.** The same release can be decoded without any model. The decoder
fits the maximum-entropy table that agrees with every released count, using Private-PGM (the
estimator AIM and MST use). It samples that table at `pool_factor × n_rows` and passes the rows
through the same selection step. It reads only the release and the schema, spends no budget, needs
no key and runs on a CPU. Install the `pgm` extra to use it.

```python
from cortec import decode_release
synthetic = decode_release(schema, release, n_rows=5000, pool_factor=3)   # no model, no API
```

or `cortec decode --schema my_schema.py --release results/<name>/release.json --n-rows 5000`. In the
paper this path is the decoder-with-selection control (Table 3). On Adult at ε = 2 and n = 300 it
reads 0.033 on single columns and 0.014 on the conditionals the release names. This matches the best
language-model configuration. Its students score 0.824, 0.866 and 0.862, above every language-model
configuration (one split, three draws). Beyond this, a language model improves only the pairs it
recognises from the column names and adds nothing to aggregate utility (design notes, section 11).
Decode this way first. Add a model only where you need that improvement on pairs and only for a
schema the model knows.

**Dry run first.** `backend="mock"` needs no key and honours the exact-count block. The whole
generate-and-select loop therefore runs offline from start to finish. `tests/test_selection.py` does
exactly that.

## 7. Stage C: attach a utility bound

For each released cell Stage C releases a Laplace estimate of the private target rate at a declared
budget, `epsilon_cert`. It then turns the noise into a one-sided bound on the gap between the
private and synthetic rates. The bound holds for every cell at once with probability at least `1 −
alpha`. It is a utility bound computed under DP. It is not a privacy audit.

```python
from cortec import bound_with_controls

report = bound_with_controls(schema, release, private_df, synthetic, holdout_df,
                             epsilon_cert=1.0, alpha=0.05)          # tolerance derived from the release
print(report.summary())
report.to_json("bound.json")       # readable by a third party without the private data
```

- `holdout_df` is real data that was not used to build the release. It supplies the ceiling (a real
  sample, which should clear the tolerance) and the floor (the same sample with its target permuted,
  which must not). If the ceiling fails or the floor clears, the test did not discriminate.
  `report.verdict` is then `None` and that run is not reported.
- `epsilon_cert` is spent once through the ledger. The deployment total is `epsilon_release +
  epsilon_cert`, stated per row and per person. Each cell's rate is taken over `max(|c|, n_min)`,
  and the Laplace scale uses the public floor `n_min`. The sensitivity bound therefore holds for
  every cell whatever its true size, and the bound is pure ε-DP.
- A released cell that the synthetic data never covers scores the trivial bound of 1.0. A bound
  therefore cannot be obtained by covering only a convenient subset of cells. Cells with fewer than
  20 synthetic rows are reported as thin. The noise cannot be seeded. The report's `_what_this_is`
  and `_standards_not_claimed` blocks say what it is and which privacy frameworks it must not be
  mapped to.

## 8. Evaluate the output

Never against a bare threshold. Score the synthetic data beside two references built from real data
of the same size. The first is a real sample: the ceiling a synthetic method can reach at that size.
The second is the same sample with its target column shuffled. It is a floor that carries no usable
information. Report fidelity and utility together, because a method can score well on one and poorly
on the other. Read the conditional measures beside the aggregate ones. Also compare each declared
categorical column's full set of categories with the release. A category that is present in the
release but missing from the output is a representativeness failure, and no aggregate metric reports
it.

The package ships this comparison. `evaluate()` scores every table you give it beside a real sample
of the same size drawn from `train` and beside that sample with its target shuffled. It uses the
paper's measures: 1-way total variation on the public bins and TSTR AUC under logistic regression, a
random forest and gradient boosting on the holdout. It also counts the declared categories missing
from each table. It needs scikit-learn, which the `dev` extra installs.

```python
from cortec import evaluate
from cortec.report import record_results

rec = evaluate(schema, {"cortec": synthetic}, train=private_df, holdout=holdout_df)
rec.show()                      # the paper's table layout, reference rows marked
rec.save("results", "evaluation")
panel = record_results(evaluation=rec, bound=rec_c, generation=rec_b, release=rec_a)
panel.show()                    # fidelity and utility against the real-sample range, the bound, the cost
panel.write_markdown("results/RESULTS.md")
```

`evaluate()` draws five real samples by default (`reference_draws`), each with a shuffled copy. The
two reference rows are their means. A second table, `reference spread`, gives the lowest and highest
value of each measure across them. That range is what real data of this size spans on this holdout.
`record_results()` turns the stage records into the results panel that `cortec run` prints last. The
panel shows the marginal fidelity and the downstream utility. It reads each synthetic number against
that range: within it, beyond every real sample or outside it by a stated amount. For each student
it also gives the share of real-sample utility beside its gap. That share is (synthetic − shuffled)
/ (real sample − shuffled). The panel's headline is a reading at one draw, not a pass or a fail.
Nothing in the package compares a number with a threshold.

**What to expect.** On UCI Adult at ε = 2 and n = 300, every student trained on this package's
output is within 0.025 AUC of the same student trained on a real sample of the same size. That holds
on an open model (Qwen3-14B: 0.807, 0.862 and 0.833 against the sample's 0.830, 0.870 and 0.840) and
on a served one (Gemini: 0.805, 0.863 and 0.838). The seen-conditional error is a fifth of the
sample's. Two things are worth knowing before you read your own panel. First, most of that result
comes from the release. Decoded with no language model at all, using the maximum-entropy table that
fits its counts, the release trains students within 0.012 AUC of the package's output. What the
language model adds comes from the column names. When every name is replaced by a neutral token, its
advantage on pairs and on utility disappears. On a schema the model does not know,
expect the release's own result. Second, the reference AIM (fitted the way its authors fit it)
fits pairs of columns more closely. At n = 1,000 it also scores higher on utility. The package is
level with it on
single columns, and on Adult it carries the conditional rates the release names more closely. On a
clinical survey (NHANES) the package's students were 0.07 to 0.10 AUC below a real sample's. The
model-free decoding of the same release scored higher than they did. Do not assume parity. Measure
it on your own data against a real sample of the same size. Read the conditional measures beside the
aggregate ones. The design notes carry the full measurements.

## 9. Where the trust boundary sits

No private record is ever sent to the model. The prompt carries only released statistics: noised
histograms, class balances and a conditional table. Generation is post-processing of a DP release.
By post-processing immunity, whoever runs the model learns nothing beyond what the release already
discloses. The ε guarantee does not depend on where that computation happens. The decoder may be a
tenant-isolated enterprise endpoint under a BAA or an air-gapped model on your own hardware. The
argument is the same for both. What differs between them is capability, not trust.

The guarantee says nothing about the model's pretraining corpus. If a private record was in it, that
happened before this tool ran. The sharper version of the concern is that conditioning on true
marginals for a narrow stratum resembles a prompt-based extraction attack. It tells the model which
region to sample from. DP does not exclude this. In the research behind this tool, four
membership-inference attacks were each validated on a positive control. None found an advantage
above chance, and there were zero exact matches on any dataset. That is evidence about the published
output, not a proof about the corpus.

## 10. Troubleshooting

Each row below is a refusal. The tool stops on purpose, before spending budget or writing output it
cannot stand behind. In a run script you can print refusals in the standard layout instead of a
traceback. Add one line at the top, or wrap the run:

```python
from cortec import install_guard
install_guard()          # a refusal prints as a boxed block and exits with status 2

from cortec import guard
with guard():            # the same, for one block (works in notebooks too)
    ...
```

```
── CoRTeC 1.2.5 · Refused · Data Validation ────────────────────────────────────────────
  refused by  DataValidationError
  reason
    column 'income' is missing from the data. NO privacy budget was spent.

  The tool stopped on purpose. This is a refusal, not a fault in the tool. Change the input
  or the setting the reason names and run again.
```

Only the classes in this table and those of `cortec-hybrid` are treated as refusals. Any other
exception is a fault and keeps its traceback.

| What you see | Why | What to do |
|---|---|---|
| `DataValidationError` before any query | a column is missing, the target has an undeclared label or one class only, or a numeric column has no usable values | fix the data or the schema. No budget was spent |
| `SchemaError: stratify bands ... must tile its declared domain` | a gap or an overlap between bands | make the bands contiguous from `lo` to `hi` |
| `ReleaseError: n_min=... is below the safe floor of 50` | a rate over fewer records is dominated by its noise | raise `n_min`, or coarsen the conditional levels |
| `ReleaseError: no cohort reached n_min` or `the conditional target table is empty` | too few records per cohort or cell | coarsen the stratification or the levels, lower `n_min`, or supply more records |
| `ReleaseError: REFUSED: max_rows_per_person=...` | the per-person ε would exceed 10 | cap or aggregate rows per person, lower `epsilon_total`, or acknowledge a row-level guarantee |
| a printed warning that part of the conditional budget was not spent | some declared levels released no cell | use fewer levels, lower `n_min`, or supply more records |
| a printed warning that a level's noise is at or above the spread of its rates | the table is noise-dominated | raise ε, coarsen to fewer cells, or accept that conditioning will not help on this data |
| `ModelCapabilityError` | the model is unmeasured, sweep-only or insufficient | use a recommended model, or opt in with the flag the message names and verify on your own data |
| `PromptIntegrityError` | a prompt template was modified in memory, or the conditional table is empty | do not edit the templates. Register a variant under a new name if you must |
| `ContextTruncationError` | the prompt plus the output budget exceeds a local model's context | raise `num_ctx` or lower `rows_per_call` |
| `EmptyContentError` | a reasoning model spent its output budget before writing any CSV | raise the output budget. `ModelRefusalError` is the different case of a refusal, which needs a coarser cell or a different generator |
| `FatalAPIError` | billing, quota, credentials, or a model the surface does not serve | fix the account or the model name. Nothing was retried and the release is unchanged |
| the run waits and `rate_limit_waits` rises | the vendor asked it to wait | nothing. The waits back off from 15 s to 4 min before an error surfaces |
| a "budget cap reached" warning and fewer rows than requested | the spend cap was reached | not an error: the output holds the rows completed within the budget. Raise `budget_usd` or use a cheaper validated model for the full request |
| `GenerationError` naming a yield or parse rate | most returned rows were dropped, or too few calls parsed | a genuine fault, so the run stops before spending more. Check the schema against the rows being dropped |
| `select_to_release` refuses the pool | a run that stopped early without a budget cap (a lost connection) left cohorts short of the rows they owe | a budget cap is handled automatically, returning the partial pool with a warning. For other truncations pass `allow_short_pool=True` |
| `generate_by_cell` refuses the release | a conditioning column has under 90% of its mass in released bands | lower `n_min` or use a coarser level, or generate cohort-wise |

## 11. Output and exports

Every stage prints one layout and exports one record through `cortec.report`. The layout is a stage
header, a block of named values, one or more tables (each with the question it answers under its
name), the warnings and a verdict. A run opens with a banner of what is about to run. It closes with
the results panel and a list of every file written. On a terminal, the run's own numbers are printed
in blue and the reference rows (a real sample and a shuffled floor) in grey. Warnings appear in
amber. A verdict against the output or a refusal appears in orange. Colour is used only on a
terminal. `NO_COLOR` or `CORTEC_COLOR=0` turns it off, and `CORTEC_COLOR=1` forces it on. The
exported files never carry colour.

```python
from cortec import show, record

rec = show(release, n_private_rows=len(private_df))   # Stage A
rec = show(gen, n_rows=len(synthetic))                 # Stage B, after generate()
rec = show(report, schema=schema.name)                 # Stage C
files = rec.save("results/run-01", "stage_c")          # {name: path}
```

`show()` prints and returns a `RunRecord`, and `record()` builds one without printing. `save()`
writes the record as `<stem>.json` (format `cortec-result/1`) and the same content as Markdown in
`<stem>.md`. It also writes the named values to `<stem>.csv`, one CSV per table, and `<stem>.png`
when matplotlib is installed. `RunRecord.from_json()` reads a record back. Tables follow the paper's
conventions: a label column on the left, numbers to three decimal places and reference rows marked.
Figures use the paper's own plotting parameters and checked palette. Your own scripts can use them
through `cortec.plots.paper_rcparams()`. [`result-format.md`](result-format.md) is the contract for
the record and lists the tables each stage writes.

**The words in the output.** `cortec terms` prints every term the records use, in the order a run
prints them. `cortec <term>` (for example `cortec discriminating`) prints one. The ones a first run
meets in Stage C and the evaluation:

| Term | Meaning |
|---|---|
| tolerance | the largest per-cell gap you accept. By default derived from the release (the noise half-width plus half the largest gap between a released cell rate and the released base rate, so the permuted floor cannot pass at any sample size), or a number you pass. A table is within tolerance when its worst bound is at or below it |
| ceiling | a real sample of the synthetic table's size, scored the same way: the best any method can do at that size |
| floor | the same real sample with its target permuted: data with no conditional signal |
| discriminating | the test had power: the ceiling is within tolerance and the floor is not. Otherwise the verdict on the synthetic table is not reported |
| covered, thin | a released cell with synthetic rows in it. Thin when fewer than 20. An uncovered cell scores the trivial bound 1.0 |
| mean bound, worst bound | the average and the largest per-cell bound. The verdict uses the worst |
| 1-way TV | the total-variation distance between the table's and the holdout's histogram, per column on the public bins, averaged. Lower is better |
| TSTR-LR, RF, GBM | train on synthetic, test on real: three classifiers trained on the table, scored by AUC on the holdout. Higher is better, read against the ceiling and floor rows |
| yield | rows kept divided by rows requested from the model. Rows outside the schema are dropped and counted |
| support | the number of private records in a cell or cohort, with noise added |
| reference spread | the lowest and highest value of each measure over the real samples drawn as references: the range real data of this size spans on this holdout |
| reading | where a synthetic number lies against that range: within it, beyond every real sample, or outside it by how much |
| share of real-sample utility | (synthetic − permuted) / (real sample − permuted) for one student: the part of a real sample's utility above the no-information floor that the synthetic table reaches |

## Licence

Apache-2.0.
