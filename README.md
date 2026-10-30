# CoRTeC

**A differentially private release for synthetic tabular data, decodable by any generator.**

![The pipeline](paper/figures/fig01_pipeline.png)

CoRTeC's contribution is a release: it spends a differential privacy budget once, on a small release of statistics built for a
downstream classifier, shipped as an auditable artefact of counts that anyone can decode at no further privacy cost: inside cohorts formed by a public rule, one Laplace histogram per attribute
and outcome class, a noisy class balance and cohort size, and a table of outcome rates over disjoint
cells. A frozen language model then reads that release, and nothing else, and writes rows that carry
exactly the per-bin counts it is asked for; a pool of such rows is raked and refined to the released
cells. The model never sees a private record, so every draw is post-processing of the release: it
spends no budget, can be repeated without limit, and can run on any model, including one reached
through an interface the institution does not control.

## Documents

| Document | What it is |
|---|---|
| [The paper](paper/CoRTeC_arxiv.pdf) | The mechanism, its privacy analysis, and the measurements with their controls (12 pages of main text, appendices A to G) |
| [The deployment guide](paper/CoRTeC_deployment_guide.pdf) | The architecture, the parameters, the full prompt, a checklist, the controls mapped to standards, the failure modes with their guards, and the cost (7 pages) |

## What the paper finds

- **The release carries the result.** Decoded with no language model at all, by the maximum-entropy
  table that fits its counts, the release trains models as well as CoRTeC does: on UCI Adult at
  ε = 2 and n = 1,000, 1-way total variation 0.041 and downstream AUC of 0.83 to 0.87 against
  CoRTeC's 0.036 and 0.83 to 0.86, an ordering that holds on three seeds.
- **The language model adds plausibility driven by the column names.** It repairs the logical pairs
  it recognises, invents dependences between near-independent columns, and loses its edge on pairs
  and on utility when every name is replaced by a neutral token. The selection step applied to the
  decoder's own pool brings single-column and seen-conditional error below a real sample's with no
  model, over three splits, and without the release the same backbone recites the public file into
  most of its one-shot output.
- **The reference AIM is ahead on pairwise fidelity at every seed and, against the fixed-rule release, on utility at n = 1,000.**
  The release is level with it on single columns and ahead on the conditionals it budgets for (0.035
  against 0.061 at n = 300), and with the selection step its single-column error is below a real
  sample's (0.027 on Gemini and 0.033 with no model, against 0.043). The release the auto-configurator
  derives on Adult, decoded with no model and the selection step (`cortec decode`), is level with or ahead
  of AIM on utility at both sizes over three splits and ahead on single columns; AIM keeps pairs at n = 1,000.
- **What survives every control is the privacy architecture.** One release serves unlimited draws;
  the backbone can be replaced at no privacy cost; the strongest membership attack on any arm reaches
  an advantage of 0.02 to 0.05 against a permitted 0.762; no synthetic row copies a training row;
  and the served backbone's memorised head of the Adult file never reaches a draw.

## Install

Python 3.10 or later. The package is not on PyPI; install it from a clone of this repository. One
install gives the two packages, `cortec` (Stages A, B and C) and `cortec_hybrid` (the model-free
correction of a marginal synthesiser's output), and the two commands, `cortec` and `cortec-hybrid`.

```bash
git clone https://github.com/Calyie/cortec
cd cortec
pip install '.[gemini]'        # Gemini (also the `vertex` extra for Google Vertex AI)
pip install '.[openai]'        # GPT, Azure OpenAI, and any server with an OpenAI-compatible interface (vLLM)
pip install '.[anthropic]'     # Claude (also the `bedrock` extra for AWS Bedrock)
pip install '.[pgm]'           # decode a release with no language model (Private-PGM and JAX)
pip install '.[dev]'           # pytest and scikit-learn, for the tests and the evaluator
python3 -m pytest tests -q     # offline; every test is named after the defect it prevents
```

## Quick start

The schema declares public knowledge only: the columns, their bounds, the bin edges and the target.

```python
from cortec import Schema, release_statistics, Generator

schema = Schema(
    name="encounters",
    numerical={"age": (18, 95), "length_of_stay": (1, 30)},
    categorical={"admission_type": ["Emergency", "Elective", "Urgent"]},
    target="readmitted_30d", positive="YES", negative="NO",
    bins={"age": [18, 40, 55, 70, 95], "length_of_stay": [1, 3, 6, 10, 30]},
)                                                     # no `conditional`: the hierarchy is derived

release = release_statistics(schema, private_df, epsilon_total=2.0, n_min=150,
                             max_rows_per_person=1)   # Stage A: the only code that sees private_df
release.to_json("release.json")                       # contains no private record

gen = Generator(schema, backend="gemini", model="gemini-3.5-flash", reasoning="on")
synthetic = gen.generate_selected(release, n_rows=1000, pool_factor=3)   # Stage B: post-processing
```

`cortec run --data private.csv --schema my_schema.py --backend gemini` runs the three stages on a
table from the command line; `cortec decode --schema my_schema.py --release results/<name>/release.json`
decodes the same release with no language model (Private-PGM plus the selection step, the paper's
decoder-with-selection control; needs the `pgm` extra); and `cortec --help` lists what can be run. The setup guide,
[`docs/setup.md`](docs/setup.md), goes through each step: where the model runs and the credentials,
the data, the schema, the release and its audit, generation and selection, the utility bound, the
evaluation against a real sample and a no-information floor, and the troubleshooting of every
refusal. A self-hosted backbone behind an OpenAI-compatible server (the paper's Qwen3 arms ran that
way through vLLM) is reached through `backend="openai"` with `OPENAI_BASE_URL` set.

**Where this should run.** For regulated data, the enterprise-hosted, tenant-isolated surface in
your own cloud account (Vertex AI, Azure OpenAI or Bedrock) or a model on your own hardware. The
vendors' public developer APIs are supported and are not recommended for regulated data: the weights
are the same, the contract and the network path are not. The privacy argument does not depend on
the endpoint, because the request carries only the release; the compliance argument does.

## Reproducing a CoRTeC arm of the paper

The research pipeline that produced every CoRTeC arm is under [`research/`](research/). Datasets are
fetched from public sources on first use and cached under `research/data/`, which is not committed.

```bash
pip install -r research/requirements.txt
cd research
python3 run_dataset.py --dataset adult --stage generate --backend gemini --model gemini-3.5-flash \
    --n-synthetic 300 --draws 3 --rows-per-call 25 --epsilon-total 2.0 --cond-frac 0.2 --n-min 150 \
    --quota --pool-factor 3 --seed 42 --skip-header-only --no-by-cell --outdir results/adult_gemini
python3 evaluate_generic.py --dataset adult --spec-json <arms.json> --seed 42
```

A release already present in the output directory is reused rather than redrawn, so further draws
cost no privacy budget. Released statistics cannot be regenerated bit for bit, because the noise
source ignores any seed; the frozen auto-configurations under `research/results/autoconfig/` are
reused by file for the same reason. The reference-AIM wrapper, the maximum-entropy decoder, the decoder-with-selection and one-shot
controls, the anonymisation of releases and the scorers are under `research/controls/`; the synthetic draws, per-draw score files and
stored releases of every arm are deposited with the paper's Zenodo record, DOI 10.5281/zenodo.22945315.

## Repository layout

| Path | Contents |
|---|---|
| `src/cortec/`, `src/cortec_hybrid/` | The two packages, Apache-2.0 |
| `tests/` | Their tests, offline |
| `docs/` | The setup guide, the hybrid's guide, the design notes (the measurements behind the defaults) and the result format |
| `examples/example.py` | A runnable tour of the three stages with no API key |
| `paper/` | The paper and the deployment guide as PDFs, with the figures |
| `research/` | The pipeline driver, the evaluator, the Stage C script, the membership-inference script, the dataset specifications and the frozen auto-configurations |
| `research/controls/` | The paper's controls: the reference AIM and MST wrapper with its corrected sampler, the maximum-entropy decoder of a release alone and with the selection step, the one-shot control, the anonymisation of a release and its draws, the per-pair scorer and the three paper-protocol scorers |

## Where the guarantee does not hold

- **ε protects one row, not one person.** A person contributing `k` rows receives `k · ε`. Cap each
  person's rows before Stage A; the library requires the per-person row bound and refuses a vacuous
  one.
- **Floating-point Laplace.** The library mechanism is vulnerable to the Mironov (2012) attack on
  the low bits of the sampled noise.
- **Pretraining provenance.** The guarantee says nothing about the model's training corpus; the
  paper's memorisation tests and membership attacks are the measured evidence.
- **Endpoint.** The paper's measurements used the public developer APIs and self-hosted backbones,
  not the enterprise surfaces recommended above.

## AI use

Language models appear in this work as the objects of study: the backbones named in the paper's tables. Language-model assistants were also used in drafting and editing the text and the code in this repository. Every experiment was designed, run and checked by the author, and every number in the documents was produced by the described pipeline from stored result files.

## Citation and licence

See [`CITATION.cff`](CITATION.cff). The licence is Apache-2.0, [`LICENSE`](LICENSE).
