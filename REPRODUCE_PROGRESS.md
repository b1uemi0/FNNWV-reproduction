# FNNWV Reproduction Progress

## Implemented algorithms

- MV: CEKA 1.0.1 built-in implementation.
- IWMV: implemented in `code/IWMV.java` from Algorithm 1 of "Error Rate Bounds and Iterative Weighted Majority Voting for Crowdsourcing"; max iterations = 50, matching the FNNWV paper.
- PLAT: CEKA 1.0.1 built-in implementation.
- MNLDP: implemented in `code/MNLDP.java` from the IJCAI 2019 paper; k = 5, eta = 0.5, iterations = 20.
- LAWMV: implemented in `code/LAWMV.java` from the Information Sciences 2022 paper; b = 0.5.
- FNNWV: original `code/FNNWV.java` from this repository.

## Literature files

- Local FNNWV paper text: `references/fnnwv_paper/FNNWV_local.txt`
- IWMV paper: `references/related_methods/IWMV_arxiv_1411.4086.pdf`
- MNLDP paper: `references/related_methods/MNLDP_IJCAI_2019_0204.pdf`
- Related-method search notes: `references/related_methods/SEARCH_SUMMARY.md`

## Current reproduction commands

```powershell
cd "D:\pycode\crowd sourcing\FNNWV-main"
.\scripts\compile_java.ps1
.\scripts\run_java_simulation_all.ps1 .\datasets\simulation .\results\FNNWV_simulation_uniform_mv_iwmv_plat_fnnwv.tsv 10 5 uniform
.\scripts\run_java_ner.ps1 .\datasets\real-world .\results\FNNWV_realworld_ner_mv_iwmv_plat_fnnwv.tsv
```

## Current results

### Simulation, uniform worker quality in [0.5, 0.8], 10 runs

Paper averages:

```text
MV     57.74
IWMV   57.97
PLAT   57.30
MNLDP  72.23
LAWMV  68.10
FNNWV  76.11
```

Current reproduced averages:

```text
MV     58.11
IWMV   58.48
PLAT   57.91
FNNWV  75.55
```

### Real-world NER

Paper:

```text
MV     73.29
IWMV   76.36
PLAT   58.04
MNLDP  61.62
LAWMV  78.27
FNNWV  79.41
```

Current reproduced:

```text
MV     72.92
IWMV   77.17
PLAT   58.04
FNNWV  79.68
```

## Remaining work

- MNLDP and LAWMV have now been implemented from the papers.
- NER ablation, worker-count, and imbalance-ratio experiments have been added.
- FNNWV alpha sensitivity and unsupervised adaptive-alpha experiments have been added with seed-fixed 10-run output, per-run records, summaries, trigger rates, standard deviations, and Wilcoxon analysis.
- Current alpha finding: adaptive Q10 has the best average F1 and a stable 10% trigger rate, but it wins only 9/22 datasets against original FNNWV and is not Wilcoxon-significant versus original alpha=0.10. It is significant versus LAWMV in the current run.
- Remaining rigor work: investigate why Q10 gains heavily on `page-blocks0` but drops on datasets such as `haberman`, then consider a more robust adaptive trigger before running a longer 30-run experiment.
