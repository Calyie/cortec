<!-- math: on -->
# CoRTeC: Cohort-Conditioned Differentially Private Synthetic Tabular Data from a Frozen Language Model

<div class="authors">
<p><b>Calister Nnona</b></p>
<p>calisternnona@gmail.com</p>
<p>Summary of the paper, 23 September 2026</p>
</div>

<div class="abstract">
<p><b>Summary.</b> Differentially private (DP) synthetic tabular data lets an institution train on, and share, records it cannot release. Its value depends on how much accuracy a model loses when it is trained on the synthetic data instead of the real data. The deployed mechanisms, MST and AIM, are selected on marginal fidelity and do not report that loss. CoRTeC spends the privacy budget once, on a statistics release built for a downstream model, and lets a frozen language model decode the release. At ε = 2 and n = 300 on census, credit and clinical data, tree models trained on its output match models trained on a real sample of the same size, its 1-way marginal error is within 0.005 of MST's or below it, and four membership-inference attacks reach an advantage of at most 0.048 against a permitted 0.762. The full paper, the technical report and two Apache-2.0 implementations are public.</p>
</div>

## 1. The problem

An institution that holds sensitive tabular data wants to train a model outside its own perimeter, to give a realistic sample to a partner, and to let researchers work on the problem without working on the people. Synthetic data under differential privacy is the accepted way to do all three. Whether it is worth doing depends on one question: if I train on the synthetic data instead of the real data, how much accuracy do I lose?

The mechanisms an institution can deploy today do not answer that question in their own papers. MST and AIM are evaluated on marginal workload error, which they attain, and they were built for published tables rather than for training models. The generative models trained under DP-SGD or PATE are evaluated on downstream utility but, at the budgets and schema sizes we test, do not preserve the target's base rate. The methods that do reach real-data utility fine-tune a language model on the records, which either carries no guarantee or places the private data on the training hardware.

## 2. What CoRTeC does

CoRTeC separates two things every other method does together: spending the privacy budget and producing records. Stage A spends the budget once, on a release. Within cohorts formed by a public stratification rule it publishes one DP histogram per attribute and per outcome class, a DP class balance and cohort size, and a DP table of target rates over the cells of a conditional hierarchy. Stage B gives a frozen, un-finetuned language model that release and nothing else. Each batch is told the exact number of rows it must produce per bin, per category and per outcome; a threefold pool is filtered to the rows whose cell counts match the release; and values below bin resolution are redrawn inside their released bins. An optional Stage C releases a DP bound on the gap between private and synthetic conditional rates.

Four consequences follow, and they are the reasons to use it.

- *The privacy argument is small.* The generator never sees a private record, so its work is post-processing of a DP release, and the privacy analysis reduces to a composition of counting queries.
- *Generation costs no privacy and can run anywhere.* Any number of datasets can be drawn from one release, and the model can be a tenant-isolated enterprise endpoint or an air-gapped local model with the same guarantee.
- *The budget goes to the structure a classifier needs, and it goes there cheaply.* Histograms have sensitivity 1 regardless of resolution, and disjoint cohorts and cells compose in parallel, so the release carries 6.7× more budget per statistic than a natural implementation.
- *No statistician is needed.* An auto-configurator chooses the cohorts and the hierarchy from the schema and the budget, charging the one step that reads private data.

## 3. What we find

Every comparison is at ε = 2 and n = 300, against a real sample of the same size (the floor no synthetic method should be expected to exceed) and against the same data with its target permuted (the score of data that carries no usable information). Fidelity and utility are always reported together, because a method can score well on one and poorly on the other.

**Table 1:** 1-way total variation (lower is better) and train-on-synthetic, test-on-real (TSTR) AUC under three students, logistic regression (LR), random forest (RF) and gradient boosting (GBM), higher is better, on the three head-to-head datasets. CoRTeC is the shipped configuration with Gemini 3.5 Flash, three draws per dataset. AIM completes on finance only on a 12-column schema, so its finance row is scored on those columns.

| dataset | condition | 1-way TV | TSTR-LR | TSTR-RF | TSTR-GBM |
|---|---|---|---|---|---|
| Adult | real sample | 0.043 | 0.830 | 0.870 | 0.840 |
| Adult | CoRTeC | 0.029 | 0.842 | 0.876 | 0.849 |
| Adult | MST | 0.024 | 0.690 | 0.728 | 0.681 |
| Adult | AIM | 0.037 | 0.675 | 0.700 | 0.691 |
| finance | real sample | 0.041 | 0.695 | 0.727 | 0.717 |
| finance | CoRTeC | 0.029 | 0.680 | 0.731 | 0.712 |
| finance | MST | 0.028 | 0.580 | 0.610 | 0.604 |
| finance | AIM (12 columns) | 0.054 | 0.578 | 0.652 | 0.623 |
| NHANES | real sample | 0.034 | 0.773 | 0.728 | 0.716 |
| NHANES | CoRTeC, auto-configured | 0.015 | 0.752 | 0.732 | 0.711 |
| NHANES | MST | 0.024 | 0.687 | 0.646 | 0.664 |
| NHANES | AIM | 0.024 | 0.596 | 0.650 | 0.650 |

Tree models trained on CoRTeC's output do not separate from the real-sample floor on any of the three datasets; the linear model matches on Adult and is within 0.021 elsewhere. Models trained on AIM's and MST's output score 0.05 to 0.18 AUC lower. CoRTeC's 1-way error is within 0.005 of MST's on Adult and finance, below it on NHANES, and below a real sample's on all three. Of 35 Holm-corrected comparisons against AIM and MST on Adult and finance, 28 favour CoRTeC, one favours a baseline, MST's 0.005 advantage on 1-way error on Adult, and six do not separate. AIM keeps the lowest error on its own 3-way workloads at adequate sample size, which its design predicts. A powered equivalence test on Adult, six draws against five, gives differences of +0.007, −0.003 and +0.012 AUC with every p > 0.18 and every interval inside ±0.027.

Four further results reach beyond the mechanism.

- **Both standard acceptance criteria are saturated.** Real data with its target permuted records 95.7% 1-way similarity and passes a 90% bar. On Adult, an ungrounded language model given only the column names reaches TSTR-LR 0.824 against a real sample's 0.834, AIM's 0.675 and MST's 0.690. On a public benchmark, then, neither marginal similarity nor aggregate utility can tell a mechanism that reads the release from one that recites its prior, and that is why the paper's inversion test exists (Section 6.3): with one released relationship inverted, a matched control with the released arrays deleted emits the inverted relationship at 0.931 where CoRTeC emits 0.000, and only conditional and subgroup measures separate the two.
- **Whether a forced private relationship survives synthesis depends on the mechanism family, not on the guarantee.** MST transmits it with slope 0.998; PATE-CTGAN, at the same ε on the same data, with slope 0.054. What the marginal methods lose is instead a 2.2 to 2.8× inflation of pairwise feature dependence relative to real data; CoRTeC inflates it by 1.24×.
- **Enabling reasoning in the generator changes conditional error 3.8×** (0.173 to 0.045) while downstream AUC cannot detect the change at any feasible draw count.
- **Four membership-inference attacks**, validated on a positive control of real records passed off as synthetic, reach a strongest advantage of 0.048 against a permitted 0.762, with zero exact matches on any dataset.

## 4. Limitations, in brief

The head-to-head is at n = 300 under one generator family, with a one-draw cross-vendor check. AIM enters the finance comparison only on a reduced schema and is absent from a fourth, hospital-encounter dataset (Diabetes 130) used in the transmission experiments, because its fit exceeded three hours. ε protects a row; on encounter-level data the per-person guarantee is weaker by the maximum contribution count, and the mitigation is a contribution cap taken before the release. Every real dataset is a public benchmark the generator has very likely seen; a constructed registry with no public presence rules out gross recitation but not recall of the marginals. CoRTeC pays per generated record, three times over under the shipped configuration, where the marginal methods pay once and sample freely. The coverage guard that prevents cell-wise generation from silently dropping subpopulations is a heuristic, and when a private conditional table can be conveyed to a language model without silent loss is open. The technical report catalogues twenty-seven measurement artefacts that each produced a plausible but wrong conclusion before it was caught.

## 5. Artefacts

The paper (46 pages) and the technical report (93 pages, every intermediate experiment and the defect catalogue) are in the research repository, [github.com/Calyie/cortec](https://github.com/Calyie/cortec), with the pipeline that produced every table. The two reference implementations, in which every guarantee is enforced in code and every regression test is named after the defect it prevents, are in [github.com/Calyie/cortec-framework](https://github.com/Calyie/cortec-framework). Every headline number is re-derived from stored per-draw records by an audit suite, and every number in the paper is checked against the report.
