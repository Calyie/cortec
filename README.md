# CoRTeC

**Cohort-conditioned differentially private synthetic tabular data from a frozen language model.**

CoRTeC spends a differential privacy budget on *released statistics* rather than on training a model.
Cohorts come from a public stratification rule (ε = 0). Each cohort releases one DP histogram per
attribute, L1 sensitivity 1 regardless of bin count, so an entire distributional shape costs what a
single counting query costs. A conditional target table over a disjoint cell partition costs one
query's worth of ε however many cells it holds. A **frozen, un-finetuned** language model then
generates records from those statistics alone.

Because the generator never sees a private record, generation is **post-processing**: it adds no
privacy cost, and unlimited synthetic datasets may be drawn from a single release.

📄 **[The paper](paper/CoRTeC_arxiv.pdf)** (the arXiv version: the final mechanism, its algorithms,
privacy proof and results) · 📚 **[The technical report](paper/CoRTeC.pdf)** (every intermediate
experiment, the per-dataset walkthroughs and the defect catalogue; the audited source every number
in the paper is checked against)

---

## The result, and its boundary

At matched ε = 2.0 and matched sample size on UCI Adult **at n = 300**, models trained on CoRTeC's
synthetic records are statistically indistinguishable from models trained on a real sample of the same
size: +0.007, −0.003 and +0.012 AUC across logistic regression, random forest and gradient boosting,
with the gap bounded inside ±0.027 and every p > 0.18.

**That equivalence was first claimed at one dataset and one sample size, and we measured where it
stopped.** On the finance dataset the pooled release reaches TSTR-LR 0.652 against a matched real
sample's 0.695, a *significant* shortfall; releasing one histogram per outcome at the same ε takes
the two tree students to the real-sample floor (§7.11), and the configuration the tools now ship
(class-conditional release with a fifth of the budget on the conditional table, batches told the
exact counts they owe, and the output selected from a threefold pool of generated rows) brings all
three students within 0.011 AUC of the floor on both datasets, with 1-way marginal error within 0.005 of
MST's and below a real sample of the same size (§7.12). That is a result at n = 300 under one
generator family. Over the same Adult data AIM and MST train models 0.14–0.17 AUC lower
(Holm-adjusted p ≤ 0.0024), which we report as calibration rather than as a verdict: those methods
optimise a different objective and, at adequate n on their own 3-way workloads, AIM still attains it
best. CoRTeC pays per generated record, three times over under the shipped configuration, where the
marginal methods pay once at fit time.

Two findings hold independently of whether CoRTeC is adopted:

- **Both standard acceptance criteria are saturated.** Real data with its target column permuted
  records 95.7% 1-way similarity and passes a 90% bar; an unconditioned prompt with no access to the
  private data reaches 98% of a real sample's downstream AUC.
- **Whether a private conditional relationship survives is a property of the mechanism family, not of
  holding a DP guarantee.** At identical ε = 2.0 on identical data, MST transmits a forced
  relationship at slope 0.998 and PATE-CTGAN at 0.054.

---

## Install

The reference implementations live in the companion repository
**[cortec-framework](https://github.com/Calyie/cortec-framework)**, two Apache-2.0 packages, `cortec` and
`cortec-hybrid`, in which every guarantee this paper establishes is enforced in code. This repository
holds the papers, the research harness and the figure and build scripts; it does not depend on the
packages to run.

```bash
git clone https://github.com/Calyie/cortec-framework   # beside this repository
pip install ./cortec-framework/cortec[anthropic]           # Claude, recommended on AWS Bedrock
pip install ./cortec-framework/cortec[openai]              # GPT, recommended on Azure OpenAI
pip install ./cortec-framework/cortec[gemini]              # Gemini, recommended on Google Vertex AI
pip install ./cortec-framework/cortec-hybrid               # no model server in the dependency tree
```

**On where this runs.** CoRTeC is proposed for deployment on **enterprise-hosted, tenant-isolated**
model surfaces inside your own cloud account, with private networking, data residency, contractual
exclusion of training on inputs, and a BAA where HIPAA applies. **We do not recommend the vendors'
public developer APIs for regulated data**, even though the model weights are the same: what differs
is the contractual envelope, and that is what a compliance review assesses. The paper's own
measurements were made against those public APIs, and §6.2 says so and says exactly which results that
does and does not affect.

## Quickstart

The API below is the `cortec` package's; the schema declares only public knowledge (column list,
bounds, bin edges, target).

```python
from cortec import Schema, Band, release_statistics, Generator

schema = Schema(
    name="encounters",
    numerical={"age": (18, 95), "length_of_stay": (1, 30)},
    categorical={"admission_type": ["Emergency", "Elective", "Urgent"]},
    target="readmitted_30d", positive="YES", negative="NO",
    bins={"age": [18, 40, 55, 70, 95], "length_of_stay": [1, 3, 6, 10, 30]},
)                                                    # no `conditional`: the hierarchy is derived

release = release_statistics(schema, private_df, epsilon_total=2.0, n_min=150,
                             max_rows_per_person=1)  # Stage A: the only code that sees private_df
release.to_json("release.json")                      # contains no private record

gen = Generator(schema, backend="anthropic", model="claude-fable-5", reasoning="on")
synthetic = gen.generate_selected(release, n_rows=5000, pool_factor=3)   # Stage B: post-processing, repeat freely
```

The conditional hierarchy is derived from the declared schema plus one DP-noised selection step, and
the release carries one histogram block per cohort and outcome class by default, with a fifth of the
budget on the conditional table. Generation asks each batch for the exact counts it owes and keeps,
from a threefold pool, the rows whose cell counts match the release (paper §3.3, §7.12). `release.json` is
the controlled artifact: store it, hash it, and reuse it by file. Every further draw is free in ε.

---

## Repository layout

| Path | What it holds |
|---|---|
| `paper/` | The arXiv paper (`CoRTeC_arxiv.md/.pdf`) and the technical report (`CoRTeC.md/.pdf`), in Markdown and PDF, the figures, and the scripts that regenerate every figure and build both PDFs |
| *(companion repo)* | **[cortec-framework](https://github.com/Calyie/cortec-framework)**: the two reference implementations, `cortec` and `cortec-hybrid` |
| `src/` | The research harness: dataset specifications, Stage A release, prompt construction, generation backends, auto-configuration |
| `*.py` (root) | Pipeline drivers, baselines, evaluation and statistics. Appendix B of the technical report maps each one |

## Reproducing an arm

```bash
pip install -r requirements.txt
python3 run_dataset.py --dataset adult --stage generate --backend gemini --surface vertex \
    --model gemini-3.5-flash --n-synthetic 300 --n-out 300 --draws 3 --rows-per-call 25 \
    --epsilon-total 2.0 --cond-frac 0.2 --n-min 150 --pool-factor 3 --seed 42 \
    --skip-header-only --no-by-cell --outdir results/adult_v2_gemini
python3 evaluate_generic.py --dataset adult --spec-json <arms.json> --seed 42
python3 paper/build_pdf.py paper/CoRTeC_arxiv.md paper/CoRTeC_arxiv.pdf      # rebuild the paper
```

Every CoRTeC arm in the papers is one invocation of `run_dataset.py`; the technical report records
the exact arm specification behind each table. A release already present in the output directory is
reused rather than redrawn, so further draws cost no privacy budget. Datasets are fetched from public
sources on first use and cached under `data/` (not committed).

The per-draw result records every published number is re-derived from, and the audit suite that
recomputes them (a number audit, a paper-to-tools parity audit, a defect-pattern scan and a check that
every number in the arXiv paper appears in the technical report), are kept in the project's internal
tree and are available from the authors on request.

## Where the guarantee does not hold

Stated here because a release is only as good as its documented gaps (NIST SP 800-226).

- **ε protects one ROW, not one person.** Under group privacy an individual contributing `k` rows
  receives `k · ε`. CoRTeC as specified is a mechanism for one-row-per-person data. On encounter-level
  data, cap each person's contribution before Stage A; `certify.py` requires the per-person row bound
  to be declared and the library refuses a vacuous one.
- **Floating-point Laplace.** The library mechanism is vulnerable to the Mironov (2012) attack on the
  low bits of the sampled noise. This is a genuine deployment blocker and we name it as one.
- **Pretraining provenance.** The guarantee says nothing about the generator's pretraining corpus.
- **Endpoint.** The measurements used public developer APIs, not the enterprise surfaces recommended
  above.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Apache-2.0, see [`LICENSE`](LICENSE). Both packages carry the same licence.
