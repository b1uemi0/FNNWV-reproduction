# FNNWV Reproduction

This workspace contains a CEKA-based reproduction of the FNNWV paper experiments.

## Directory Layout

- `code/`: Java implementations and experiment runners.
- `datasets/`: Simulation and real-world NER datasets from the repository.
- `references/`: Paper PDFs, extracted text, and source notes.
- `results/`: Generated experiment results.
- `scripts/`: PowerShell build and run scripts.
- `vendor/`: CEKA 1.0.1 source and dependencies.
- `versions/`: Saved snapshots of implementation states.

## Saved Versions

- `versions/v1_baseline_mv_iwmv_plat_fnnwv`
  - MV: CEKA built-in.
  - IWMV: implemented from Algorithm 1 of the IWMV paper.
  - PLAT: CEKA built-in.
  - FNNWV: original repository implementation.

- `versions/v2_with_lawmv_mnldp`
  - Adds LAWMV and MNLDP implementations.
  - Adds all-six-algorithm result files.

- `versions/v3_checks_and_extra_experiments`
  - Adds copy-dataset identity checks.
  - Adds neighbor self-inclusion checks.
  - Adds NER ablation, worker-count, and imbalance-ratio experiment runners/results.

## Implemented Algorithms

- `ceka.consensus.MajorityVote`: CEKA built-in MV.
- `ceka.IWMV.IWMV`: iterative weighted majority voting, 50 iterations.
- `ceka.consensus.plat.PLAT`: CEKA built-in PLAT.
- `ceka.MNLDP.MNLDP`: MNLDP with k = 5, eta = 0.5, 20 propagation iterations.
- `ceka.LAWMV.LAWMV`: LAWMV with b = 0.5.
- `ceka.FNNWV.FNNWV`: original FNNWV implementation.

## Run Commands

Compile:

```powershell
cd "D:\pycode\crowd sourcing\FNNWV-main"
.\scripts\compile_java.ps1
```

Simulation experiment, uniform worker quality in `[0.5, 0.8]`, 10 runs:

```powershell
.\scripts\run_java_simulation_all.ps1 .\datasets\simulation .\results\FNNWV_simulation_v2_all_algorithms_uniform.tsv 10 5 uniform
```

Real-world NER:

```powershell
.\scripts\run_java_ner.ps1 .\datasets\real-world .\results\FNNWV_realworld_ner_v2_all_algorithms.tsv
```

Copy semantics check:

```powershell
.\scripts\run_copy_dataset_check.ps1 .\datasets\simulation\iris0.arff
```

Neighbor self-inclusion check:

```powershell
.\scripts\run_neighbor_self_check.ps1 .\datasets\simulation\iris0.arff
.\scripts\run_neighbor_self_check.ps1 .\datasets\simulation\pima.arff
```

NER ablation experiment:

```powershell
.\scripts\run_java_ner_ablation.ps1 .\datasets\real-world .\results\FNNWV_ner_ablation.tsv
```

Worker-count experiment:

```powershell
.\scripts\run_java_workers_experiment.ps1 .\datasets\simulation .\results\FNNWV_workers_experiment.tsv 3 "3,5,7,9,11"
```

Imbalance-ratio experiment:

```powershell
.\scripts\run_java_imbalance_experiment.ps1 .\datasets\simulation .\results\FNNWV_imbalance_ratio_experiment.tsv 3
```

FNNWV alpha sensitivity and unsupervised adaptive-alpha experiment:

```powershell
.\scripts\run_java_alpha_experiment.ps1 .\datasets\simulation .\results\FNNWV_alpha_experiment_runs.tsv .\results\FNNWV_alpha_experiment_summary.tsv 10 5 uniform 20260508
python .\scripts\analyze_alpha_experiment.py .\results\FNNWV_alpha_experiment_summary.tsv .\results\FNNWV_alpha_experiment_analysis.md
```

## Current Results

### Simulation Average F1

Paper Table 2:

```text
MV     57.74
IWMV   57.97
PLAT   57.30
MNLDP  72.23
LAWMV  68.10
FNNWV  76.11
```

Current v2 reproduction:

```text
MV     58.40
IWMV   58.64
PLAT   57.99
MNLDP  74.28
LAWMV  67.19
FNNWV  75.48
```

Result file:

- `results/FNNWV_simulation_v2_all_algorithms_uniform.tsv`

### Real-world NER F1

Paper Figure 4(a):

```text
MV     73.29
IWMV   76.36
PLAT   58.04
MNLDP  61.62
LAWMV  78.27
FNNWV  79.41
```

Current v2 reproduction:

```text
MV     72.59
IWMV   77.17
PLAT   58.04
MNLDP  76.88
LAWMV  78.27
FNNWV  79.68
```

Result file:

- `results/FNNWV_realworld_ner_v2_all_algorithms.tsv`

## Notes

- Most legacy simulation scripts use CEKA's random label generation and are not seed-fixed, so exact values can vary between runs. The alpha experiment runner now uses a local seed-fixed simulation path.
- LAWMV matches the NER paper value exactly in this run.
- MNLDP is implemented from the paper, but without the original authors' `MyQP` solver. The local LLE weights are solved directly and projected to nonnegative normalized weights. This likely explains the NER MNLDP discrepancy.
- The original repository's `Test_S.java` and `Test_R.java` referenced author-side classes not shipped in the repository. Compatibility packages/classes have now been added for IWMV, LAWMV, MNLDP, and MyQP, but the maintained runner is `RunSimulationAll` / `RunRealWorldNer`.

## Checks Added After v2

### copyDataset identity

Command:

```powershell
.\scripts\run_copy_dataset_check.ps1 .\datasets\simulation\iris0.arff
```

Observed:

```text
Example: false
Worker: true
Category: true
```

Interpretation: CEKA copies `Example` objects through `Dataset.addExample`, but the current helper shares `Worker` and top-level `Category` references.

### Neighbor self-inclusion

FNNWV and LAWMV include the inferred instance itself in the nearest-neighbor set. This matches the FNNWV/LAWMV paper wording and the original `FNNWV.java`, where the self distance is zero and `Utils.sort(distance)` places self first.

Observed checks:

```text
iris0:
nearest_is_self: true
FNNWV_with_self_F1: 100.0
FNNWV_without_self_F1: 100.0

pima:
nearest_is_self: true
FNNWV_with_self_F1: 72.6577
FNNWV_without_self_F1: 73.0769
```

Interpretation: excluding self can affect results, though the quick checks showed only small changes. The reproduction keeps self included because that matches the original code and paper language.

### NER ablation

Paper Figure 5(a):

```text
FNNWV-0  73.29
FNNWV-1  78.27
FNNWV-2  33.47
FNNWV    79.41
```

Current:

```text
FNNWV-0  73.30
FNNWV-1  78.27
FNNWV-2  33.47
FNNWV    79.68
```

Result file:

- `results/FNNWV_ner_ablation.tsv`

### Worker-count experiment

Current quick run uses 3 repetitions per worker count:

```text
workers MV    FNNWV
3       50.83 67.61
5       57.54 74.45
7       62.63 76.95
9       67.49 77.04
11      68.53 76.08
```

Result file:

- `results/FNNWV_workers_experiment.tsv`

### Imbalance-ratio experiment

Current quick run uses 3 repetitions per dataset and writes each dataset's imbalance ratio with MV/FNNWV F1:

- `results/FNNWV_imbalance_ratio_experiment.tsv`

### Alpha sensitivity and adaptive alpha

Current seed-fixed run uses 10 repetitions per dataset. Fixed-alpha variants use:

```text
0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30
```

Adaptive variants choose alpha from the unsupervised nearest-neighbor margin distribution:

```text
Q10, Q20, Q30
```

Average F1 and trigger rates from the current 10-run seed-fixed simulation:

```text
method                mean F1   trigger rate
MV                    58.2746
LAWMV                 67.0426
original alpha=0.10   74.8792
fixed alpha=0.01      68.6886   0.0174
fixed alpha=0.03      71.4222   0.0518
fixed alpha=0.05      73.3382   0.0872
fixed alpha=0.10      74.8792   0.1816
fixed alpha=0.15      73.6889   0.2869
fixed alpha=0.20      71.3483   0.4035
fixed alpha=0.30      67.5997   0.6650
adaptive Q10          76.4377   0.0984
adaptive Q20          75.9494   0.1981
adaptive Q30          73.3322   0.2975
```

Result files:

- `results/FNNWV_alpha_experiment_runs.tsv`
- `results/FNNWV_alpha_experiment_summary.tsv`
- `results/FNNWV_alpha_experiment_analysis.md`

Interpretation: per-dataset best fixed alpha is useful as sensitivity analysis only, because selecting it by F1 uses gold labels. The adaptive quantile variants do not use gold labels. Q10 has the strongest average F1 and keeps the trigger rate tightly controlled near 10%, while fixed alpha=0.10 triggers about 18% on average with much larger cross-dataset variation.

However, Q10 is not yet a statistically established replacement for original FNNWV: it wins on 9 datasets and loses on 13 datasets against original alpha=0.10, with a Wilcoxon p-value of 0.974622. Its average gain is heavily influenced by a large improvement on `page-blocks0`. Q10 is significantly better than LAWMV in this run (Wilcoxon p-value 0.000504), but the more cautious conclusion is that adaptive triggering remains promising and needs either a more robust adaptive rule or further dataset-specific analysis before being claimed as a stable FNNWV improvement.
