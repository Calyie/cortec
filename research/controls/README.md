# The paper's controls

These are the scripts behind the controls and baselines of the paper, as run. They read the research folder one level
up (`research/`: the dataset specifications, the evaluator, the frozen auto-configurations) and private-pgm's `mbi`
package (version 1.1) with JAX. The reference baselines fit on the training split exactly as the paper describes. The
decoders and the anonymisation read only a release and public facts.

| file | what it is |
|---|---|
| `pgm_baselines.py` | The reference AIM and MST wrapper: the authors' mechanisms at their defaults, the stable message-passing oracle (`PGM_ORACLE=stable`), the consistency gate (`PGM_CONSISTENCY_GATE`, 0.08 in the paper), the corrected sampler and a diagnostics file per fit (paper, Appendix D) |
| `privatepgm_mechanisms/` | The mechanism files of private-pgm (`aim.py`, `mst.py` and `mechanism.py`, Apache-2.0, by Ryan McKenna and co-authors) and the cdp-to-adp conversion (`cdp2adp.py`, Apache-2.0, IBM), carried as used |
| `mrf_sampler.py` | The corrected sampler: each column drawn from its fitted conditional with the axes aligned to the parents already sampled (Appendix D) |
| `aim_on_release.py` | The maximum-entropy decoder without the selection step (the plain decoder): Private-PGM fed exactly a CoRTeC release, sampled at the requested size (Section 4.2) |
| `decoder_select.py` | CoRTeC with the maximum-entropy decoder: the decoder sampled at three times the requested size, each pool passed through CoRTeC-Select (Algorithm 2) with the integer rule, scored beside the plain decoder (Table 3), the path the package ships as `cortec decode` |
| `oneshot_control.py` | The one-shot control: rows written by a frozen model from the column names alone, no release and no private data (Table 3, Appendix F.3), which costs API calls under a hard spend cap |
| `anonymise_release.py` | The anonymised-schema control: maps a release onto the neutral schema `adult_anon` (`research/src/datasets_anon.py`) and maps draws back (Appendix C) |
| `score_draws.py` | Scores draw files with the evaluator (`research/evaluate_generic.py`) beside the real-sample, permuted and ceiling references, the way every table was produced |
| `pair_tv_generic.py` | The per-pair reading of Appendix E.3 |
| `llmtgen_metrics.py`, `tabpe_metrics.py` | The two paper protocols' metric sets as implemented for Appendix E.6 (the DP-2Stage protocol uses the authors' own metric suite) |

For example, the reference AIM at ε = 2 on the seed-42 Adult split, then the plain decoder of a release. The reference
MST rows use the library's default oracle and `--draws 5` (paper, Appendix D):

```
cd research
PGM_ORACLE=stable PGM_CONSISTENCY_GATE=0.08 python3 controls/pgm_baselines.py --train <train.csv> --dataset adult \
    --epsilon 2 --n-out 300 --draws 3 --seed 42 --methods aim --outdir results/aim_ref
python3 controls/aim_on_release.py --dataset adult --release <cohort_statistics.json> --n-out 300 --draws 3 \
    --seed 42 --outdir results/decoder
```
