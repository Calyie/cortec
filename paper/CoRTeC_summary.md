<!-- math: on -->
# CoRTeC: A Differentially Private Release for Synthetic Tabular Data, Decodable With or Without a Language Model

<div class="authors">
<p><b>Calister Nnona</b></p>
<p>calisternnona@gmail.com</p>
<p>Summary of the paper, 28 October 2026</p>
<p>DOI: <a href="https://doi.org/10.5281/zenodo.22945315">10.5281/zenodo.22945315</a></p>
</div>

<div class="abstract">
<p><b>Summary.</b> CoRTeC's contribution is a differentially private release built for a downstream classifier, shipped as an auditable artefact of counts and decodable by anyone at no further privacy cost, with Private-PGM or with a frozen language model; because the model is a swappable, budget-free decoder, the paper can also measure what it adds beyond the statistics it is given. CoRTeC spends a differential privacy budget once, on a small release of class-conditional histograms and a conditional outcome table inside publicly defined cohorts, and lets a frozen language model decode that release under exact per-bin counts and a selection step that returns the generated rows to the released cells, never seeing a private record. The paper asks what each part of the design contributes, with controls: a maximum-entropy decoder of the same release, alone and with the selection step, a renamed schema, one-shot rows with no release, and the reference AIM and MST, beside DP-2Stage, DP-LLMTGen and Tab-PE under their own protocols. The release carries the fidelity and the utility; the model adds plausibility driven by the column names; the reference AIM is ahead on pairs and, for the fixed-rule release at larger sizes, on utility; what survives every control is the privacy architecture.</p>
</div>

## 1. The problem, and what CoRTeC does

An institution that holds sensitive tabular data wants to train models outside its perimeter and hand realistic samples to partners. The marginal methods, MST and AIM, run on the private rows inside the perimeter and release what their measured marginals carry; the language-model methods fine-tune on the private rows under DP-SGD, which puts those rows on the training hardware, or spend budget on every round of candidates a frozen model proposes. Neither family fixes its workload before the data is seen and ships the counts for any decoder to read again when the model changes. CoRTeC spends the budget once, on a release built for the classifier, and treats every decoder as post-processing: Private-PGM at no cost, or a frozen model that reads the release and nothing else.

Stage A is the only code that reads the private data. Inside cohorts formed by a public rule it releases a noisy cohort size, a noisy class balance, one Laplace histogram per attribute and outcome class, and a table of noisy outcome rates over disjoint cells; every suppression decision reads a noised, charged count. The release is pure ε-differentially private by a composition of counting queries. Stage B reads the release and nothing else: a frozen model writes rows carrying exactly the counts each cohort owes, a threefold pool is raked to the released cells, and values below bin resolution are redrawn inside their bins. Stage C, optional and separately budgeted, bounds how far the synthetic conditional rates lie from the private ones. Only the release crosses the trust boundary, so every draw is post-processing.

## 2. How it was measured

Every table reports fidelity (1-way and 2-way total variation against the private table, conditional error over the groups the release names and over held-out groups) and utility (train-on-synthetic, test-on-real AUC under three students) side by side, against a real sample of the same size, the same rows with the target permuted, and models trained on a 4,000-row real sample (the whole training split on NHANES). The central cells are replicated over three splits with fresh releases, with paired differences and sign counts. The reference AIM and MST are the authors' private-pgm implementation on the same split and public bins.

**Table 1.** Adult at ε = 2, seed 42, three draws per arm; total variation (lower is better) and the three students (higher is better); the decoder rows have no language model.

| $n$ | condition | 1-way TV | 2-way TV | TSTR-LR | TSTR-RF | TSTR-GBM |
|---|---|---|---|---|---|---|
| 300 | real sample (floor) | 0.043 | 0.107 | 0.830 | 0.870 | 0.840 |
| 300 | CoRTeC, Qwen3-14B | 0.046 | 0.139 | 0.807 | 0.862 | 0.833 |
| 300 | CoRTeC, Gemini 3.6 Flash | 0.027 | 0.110 | 0.805 | 0.863 | 0.838 |
| 300 | maximum-entropy decoder of the release | 0.058 | 0.146 | 0.818 | 0.856 | 0.847 |
| 300 | maximum-entropy decoder with the selection step | 0.033 | 0.118 | 0.824 | 0.866 | 0.862 |
| 300 | auto-configured release, no model, with selection | 0.019 | 0.108 | 0.826 | 0.873 | 0.860 |
| 300 | reference AIM | 0.056 | 0.122 | 0.845 | 0.869 | 0.845 |
| 300 | DP fine-tuned Qwen3-14B | 0.106 | 0.190 | 0.809 | 0.832 | 0.815 |
| 1,000 | real sample (floor) | 0.026 | 0.061 | 0.855 | 0.892 | 0.883 |
| 1,000 | CoRTeC, Qwen3-14B | 0.036 | 0.108 | 0.827 | 0.860 | 0.859 |
| 1,000 | maximum-entropy decoder of the release | 0.041 | 0.106 | 0.826 | 0.870 | 0.863 |
| 1,000 | auto-configured release, no model, with selection | 0.016 | 0.081 | 0.841 | 0.884 | 0.884 |
| 1,000 | reference AIM | 0.029 | 0.072 | 0.854 | 0.888 | 0.883 |

## 3. What was found

- **The release carries the result.** Decoded with no language model, the release is within 0.01 of CoRTeC on single columns and utility at every seed, ahead of it on NHANES pairs and utility, and at the real sample's floor on single columns on bank marketing.
- **The language model adds plausibility driven by the column names.** It repairs the logical pairs it recognises and invents dependences between near-independent columns; its aggregate effect is neutral on Adult and negative on NHANES and bank. Under a renamed schema the Gemini backbone's pair error rises to the decoder's level and its students fall below the decoder's; its single-column edge survives because it is the selection step, which takes the decoder's own pool below a real sample on single columns and seen conditionals with no model, over three splits. Without the release the same backbone recites: about 70 % of its one-shot rows copy the public file's first 25 rows, yet those rows train students within 0.025 AUC of the private methods', so on this public benchmark the fidelity columns carry the comparison.
- **The reference AIM is ahead on pairs at every seed on both datasets and on utility at Adult $n$ = 1,000 against the fixed-rule release**; CoRTeC is level with it on single columns and ahead on the conditionals the release names. The auto-configured release, decoded with no model and the selection step, is level with or ahead of AIM on utility at both sizes over three splits and ahead on single columns; AIM keeps pairs at $n$ = 1,000. The DP fine-tuning family and Tab-PE are behind on fidelity at the same budgets; DP-2Stage is behind on both axes.
- **The privacy architecture survives every control.** No private row reaches the model; four backbones decode the same release; the strongest of four membership attacks reaches an advantage of 0.02 to 0.05 on the CoRTeC arms against a permitted 0.762; no synthetic row copies a training row, and the served backbone's memorised head of the Adult file never reaches a draw.
- **Cost.** A 300-row draw costs about 1.5 dollars through the Gemini interface or 50 minutes on one rented GPU; the release takes seconds.

## 4. Limitations, in brief

The claim is not that CoRTeC beats the reference AIM but that the release spends the budget on a different structure, the seen conditionals, at a cost on pairs: the release is a fixed workload decoded by the estimator AIM uses, and AIM's adaptive selection gives the pairs while the fixed workload gives the named conditionals, an inspectable artefact and seconds of compute. The model's contribution depends on the column names, so a schema the model does not know should expect the decoder's result. The backbone ladder, the renamed-schema control, the one-shot control, the bank dataset and the paper-protocol rows are single-split; the Gemini arms are $n$ = 300 only. No claim is made about high-order correlations. Every dataset is a public benchmark. For a deployment: decode the release with the maximum-entropy model and the selection step first, at no cost; add a language model where the schema is one it knows and per-pair plausibility is the deliverable.

**Artefacts.** The paper (12 pages of main text; appendices A to G from page 16), the deployment guide, the Apache-2.0 reference implementation, the research pipeline with its controls and scorers, and the frozen auto-configurations are in one repository, [github.com/Calyie/cortec](https://github.com/Calyie/cortec); the synthetic draws, score files and releases of every arm are deposited with the Zenodo record (10.5281/zenodo.22945315).
