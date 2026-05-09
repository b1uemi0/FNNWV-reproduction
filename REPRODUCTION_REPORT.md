# FNNWV 论文复现工作总结

## 1. 项目目标

本项目围绕论文 **FNNWV: farthest-nearest neighbor-based weighted voting for class-imbalanced crowdsourcing** 做复现实验。我们的主要目标不是重新设计一个新的系统，而是在原始代码和 CEKA 1.0.1 平台基础上，尽量复现论文中的主要实验结果，并记录复现过程中遇到的问题、实现补充和结果差异。

当前工作主要包括：

- 整理原始 FNNWV 代码和 CEKA 依赖。
- 补齐论文对比方法中原仓库未直接提供的部分实现。
- 构建可重复运行的实验脚本。
- 复现实验表格和图中的核心结果。
- 对若干实现细节做检查，例如数据复制语义、邻居集合是否包含自身。
- 对 FNNWV 中固定阈值 `alpha=0.1` 做了有限的参数敏感性分析和自适应触发探索，作为复现后的补充观察。

需要说明的是，本文档中的结果应理解为复现工作记录。对于 α 相关实验，我们只将其作为补充分析，不将其表述为已经稳定优于原论文方法的新算法结论。

## 2. 项目结构

整理后的核心目录如下：

```text
code/       Java 实现和实验 runner
datasets/   模拟数据集和真实 NER 数据集
references/ 论文和相关方法资料
results/    已生成的主要实验结果
scripts/    编译、运行和分析脚本
vendor/     CEKA 1.0.1 源码和依赖
```

历史探索过程中保留过 `versions/`、`build/`、若干 smoke test 结果和早期中间结果。提交整理时不需要放入这些历史版本，只保留最新实现、核心脚本、必要数据、CEKA 依赖和主要结果即可。

## 3. 复现环境

项目使用：

- Java 编译和运行。
- CEKA 1.0.1 作为 crowdsourcing 实验平台。
- PowerShell 脚本组织编译和实验。
- Python 脚本用于 α 实验的统计分析。

编译命令：

```powershell
.\scripts\compile_java.ps1
```

该脚本会先编译 `vendor/Ceka-v1.0.1/src` 中的 CEKA 源码，再编译 `code/` 中维护的复现代码和实验 runner。

## 4. 实现与补充内容

### 4.1 原始 FNNWV

原仓库中提供了 `FNNWV.java`、`Test_S.java` 和 `Test_R.java`。其中：

- `FNNWV.java` 是论文提出方法的核心实现。
- `Test_S.java` 和 `Test_R.java` 是原作者用于模拟数据集和真实数据集实验的入口。

在当前复现中，我们保留 `FNNWV.java` 的原始逻辑作为主要复现对象。后续实验 runner 使用新的入口类组织实验，而不直接依赖原始 `Test_S.java` 和 `Test_R.java`。

### 4.2 对比方法

当前已整理或实现的对比方法包括：

| 方法 | 当前来源或实现方式 |
|---|---|
| MV | CEKA 1.0.1 内置 MajorityVote |
| IWMV | 根据 IWMV 论文 Algorithm 1 实现，最大迭代次数 50 |
| PLAT | CEKA 1.0.1 内置 PLAT |
| MNLDP | 根据 IJCAI 2019 MNLDP 论文实现 |
| LAWMV | 根据 Information Sciences 2022 LAWMV 论文实现 |
| FNNWV | 原仓库 `FNNWV.java` |

其中 MNLDP 的实现存在一个需要谨慎说明的地方：原论文或作者实现中涉及 `MyQP` 求解器，但原始仓库未提供完整求解器。当前实现中使用了本地 LLE 权重求解和非负归一化近似，这可能是 MNLDP 在真实 NER 数据集上与论文值差异较大的原因之一。

### 4.3 维护的实验入口

当前主要 runner 为：

| 文件 | 作用 |
|---|---|
| `RunSimulationAll.java` | 模拟数据集上运行 MV、IWMV、PLAT、MNLDP、LAWMV、FNNWV |
| `RunRealWorldNer.java` | 真实 NER 数据集实验 |
| `RunNerAblation.java` | FNNWV 消融实验 |
| `RunWorkersExperiment.java` | worker 数量实验 |
| `RunImbalanceRatioExperiment.java` | 类别不平衡比例实验 |
| `RunCopyDatasetCheck.java` | 检查数据复制语义 |
| `RunNeighborSelfCheck.java` | 检查最近邻集合是否包含自身 |
| `RunAlphaExperiment.java` | α 参数敏感性和自适应触发分析 |

## 5. 主要复现实验结果

### 5.1 模拟数据集实验

运行命令：

```powershell
.\scripts\run_java_simulation_all.ps1 .\datasets\simulation .\results\FNNWV_simulation_v2_all_algorithms_uniform.tsv 10 5 uniform
```

论文 Table 2 中的平均 F1：

| 方法 | 论文结果 |
|---|---:|
| MV | 57.74 |
| IWMV | 57.97 |
| PLAT | 57.30 |
| MNLDP | 72.23 |
| LAWMV | 68.10 |
| FNNWV | 76.11 |

当前复现结果：

| 方法 | 当前结果 |
|---|---:|
| MV | 58.40 |
| IWMV | 58.64 |
| PLAT | 57.99 |
| MNLDP | 74.28 |
| LAWMV | 67.19 |
| FNNWV | 75.48 |

整体来看，模拟数据集上多数方法与论文结果较接近。由于该实验中 worker label simulation 存在随机性，未固定随机种子的旧脚本多次运行会有一定波动。

结果文件：

```text
results/FNNWV_simulation_v2_all_algorithms_uniform.tsv
```

### 5.2 真实 NER 数据集实验

运行命令：

```powershell
.\scripts\run_java_ner.ps1 .\datasets\real-world .\results\FNNWV_realworld_ner_v2_all_algorithms.tsv
```

论文 Figure 4(a) 中的 F1：

| 方法 | 论文结果 |
|---|---:|
| MV | 73.29 |
| IWMV | 76.36 |
| PLAT | 58.04 |
| MNLDP | 61.62 |
| LAWMV | 78.27 |
| FNNWV | 79.41 |

当前复现结果：

| 方法 | 当前结果 |
|---|---:|
| MV | 72.59 |
| IWMV | 77.17 |
| PLAT | 58.04 |
| MNLDP | 76.88 |
| LAWMV | 78.27 |
| FNNWV | 79.68 |

NER 实验中，PLAT、LAWMV、FNNWV 与论文值接近，LAWMV 在该次运行中与论文值一致。MNLDP 与论文差异较大，当前判断更可能来自实现细节或 `MyQP` 求解器缺失，而不是数据读取本身。

结果文件：

```text
results/FNNWV_realworld_ner_v2_all_algorithms.tsv
```

## 6. 复现过程中遇到的问题和处理

### 6.1 原始测试入口依赖缺失

原始 `Test_S.java` 和 `Test_R.java` 引用了部分作者侧类或包，在仓库中并未完整提供。为保证实验能够统一运行，我们补充了 IWMV、LAWMV、MNLDP、MyQP 等兼容类，并将主要实验入口迁移到维护的 runner：

```text
RunSimulationAll.java
RunRealWorldNer.java
```

这样做主要是为了提高复现实验的可运行性和可维护性。

### 6.2 MNLDP 中 QP 求解器缺失

MNLDP 论文方法涉及局部线性嵌入权重求解。当前实现没有原作者的 `MyQP` 求解器，因此采用本地求解和非负归一化处理。这个处理可以让方法参与比较，但也可能解释了 NER 数据集上 MNLDP 结果高于论文报告值的现象。

因此，在最终复现结论中，MNLDP 应单独说明为“近似实现”，不宜把它视为与原作者实现完全一致。

### 6.3 CEKA 数据复制语义

在实验中需要对同一份带噪标签数据分别运行多个算法。我们检查了当前 `copyDataset` 的对象共享情况：

```text
Example: false
Worker: true
Category: true
```

含义是：CEKA 在 `Dataset.addExample` 时会复制 `Example` 对象，但当前辅助函数会共享 `Worker` 和顶层 `Category` 引用。由于各算法主要写入的是样本的 integrated label，这一行为在当前实验中可以运行，但它提醒我们在扩展实验时需要注意对象共享带来的潜在副作用。

相关检查：

```powershell
.\scripts\run_copy_dataset_check.ps1 .\datasets\simulation\iris0.arff
```

### 6.4 最近邻集合是否包含自身

FNNWV 和 LAWMV 在计算近邻时会把目标实例自身纳入距离排序。由于自身距离为 0，它通常会出现在最近邻集合中。我们检查了该行为，并确认这与当前 `FNNWV.java` 的实现以及论文描述保持一致。

示例检查结果：

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

该结果说明是否排除自身可能会影响结果，但当前复现保持原始代码行为，即包含自身。

相关命令：

```powershell
.\scripts\run_neighbor_self_check.ps1 .\datasets\simulation\iris0.arff
.\scripts\run_neighbor_self_check.ps1 .\datasets\simulation\pima.arff
```

### 6.5 随机性与 seed 控制

CEKA 原有模拟标注流程中使用了基于时间和 `Math.random()` 的随机过程。早期模拟实验没有固定随机种子，因此重复运行时结果可能略有变化。

在 α 分析中，我们补充了本地 seed-fixed worker label simulation，使同一 dataset/run 下所有方法共享同一份模拟标签。该改动主要用于更公平地比较 α 策略，不代表所有旧实验脚本都已经完全 seed-fixed。

## 7. 补充实验

### 7.1 FNNWV 消融实验

论文 Figure 5(a)：

| 方法 | 论文结果 |
|---|---:|
| FNNWV-0 | 73.29 |
| FNNWV-1 | 78.27 |
| FNNWV-2 | 33.47 |
| FNNWV | 79.41 |

当前结果：

| 方法 | 当前结果 |
|---|---:|
| FNNWV-0 | 73.30 |
| FNNWV-1 | 78.27 |
| FNNWV-2 | 33.47 |
| FNNWV | 79.68 |

该部分结果与论文较接近。

结果文件：

```text
results/FNNWV_ner_ablation.tsv
```

### 7.2 worker 数量实验

当前 quick run 使用每个 worker 数量 3 次重复：

| workers | MV | FNNWV |
|---:|---:|---:|
| 3 | 50.83 | 67.61 |
| 5 | 57.54 | 74.45 |
| 7 | 62.63 | 76.95 |
| 9 | 67.49 | 77.04 |
| 11 | 68.53 | 76.08 |

结果文件：

```text
results/FNNWV_workers_experiment.tsv
```

### 7.3 类别不平衡比例实验

当前实现了按数据集统计 imbalance ratio，并输出 MV 与 FNNWV 的 F1。该部分主要用于观察 FNNWV 在不同不平衡程度数据集上的表现趋势。

结果文件：

```text
results/FNNWV_imbalance_ratio_experiment.tsv
```

## 8. α 参数敏感性与自适应触发分析

FNNWV 中存在一个固定阈值：

```text
若最近邻投票 margin < alpha，则引入最远邻修正。
原始实现中 alpha = 0.1。
```

我们对该阈值做了补充分析，目的主要是理解原方法对该参数是否敏感，而不是宣称已经提出新的优越算法。

### 8.1 固定 α 网格

测试的固定 α 包括：

```text
0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30
```

### 8.2 自适应分位数触发

另外测试了不依赖 gold label 的分位数触发：

```text
Q10, Q20, Q30
```

含义是根据最近邻投票 margin 的分布，只对最低 10%、20%、30% 的低 margin 实例引入最远邻修正。

### 8.3 10-run seed-fixed 结果

运行命令：

```powershell
.\scripts\run_java_alpha_experiment.ps1 .\datasets\simulation .\results\FNNWV_alpha_experiment_runs.tsv .\results\FNNWV_alpha_experiment_summary.tsv 10 5 uniform 20260508
python .\scripts\analyze_alpha_experiment.py .\results\FNNWV_alpha_experiment_summary.tsv .\results\FNNWV_alpha_experiment_analysis.md
```

平均结果：

| 方法 | mean F1 | mean trigger rate |
|---|---:|---:|
| MV | 58.2746 |  |
| LAWMV | 67.0426 |  |
| original alpha=0.10 | 74.8792 |  |
| fixed alpha=0.01 | 68.6886 | 0.0174 |
| fixed alpha=0.03 | 71.4222 | 0.0518 |
| fixed alpha=0.05 | 73.3382 | 0.0872 |
| fixed alpha=0.10 | 74.8792 | 0.1816 |
| fixed alpha=0.15 | 73.6889 | 0.2869 |
| fixed alpha=0.20 | 71.3483 | 0.4035 |
| fixed alpha=0.30 | 67.5997 | 0.6650 |
| adaptive Q10 | 76.4377 | 0.0984 |
| adaptive Q20 | 75.9494 | 0.1981 |
| adaptive Q30 | 73.3322 | 0.2975 |

从平均值看，Q10 在这次实验中最高，并且触发率稳定接近 10%。不过进一步的逐数据集分析显示，Q10 相比原始 FNNWV 是 9 个数据集提升、13 个数据集下降，Wilcoxon signed-rank test 的 p-value 为 0.974622，并不支持“Q10 稳定优于原始 FNNWV”的结论。

因此，当前更合适的表述是：

- FNNWV 对 α 较敏感，过小或过大的触发范围都可能影响效果。
- Q10 是一个值得继续研究的触发策略，因为它能稳定控制触发比例。
- 但 Q10 的平均提升受到 `page-blocks0` 等个别数据集较大提升影响，在 `haberman`、`vehicle1`、`yeast1` 等数据集上存在下降。
- 当前证据不足以将 Q10 作为稳定改进结论，只能作为复现后的补充观察和后续研究方向。

相关结果：

```text
results/FNNWV_alpha_experiment_runs.tsv
results/FNNWV_alpha_experiment_summary.tsv
results/FNNWV_alpha_experiment_analysis.md
```

## 9. 当前局限

本复现工作仍有以下局限：

1. 部分旧模拟实验仍依赖 CEKA 原有随机流程，未全部固定 seed。
2. MNLDP 缺少原作者 QP 求解器，当前实现与原论文实现可能不完全一致。
3. worker-count 和 imbalance-ratio 实验目前为 quick run，重复次数较少。
4. α 分析虽然已经做了 seed-fixed 10-run 和 Wilcoxon 检验，但结论仍偏探索性质。
5. 当前尚未对 `page-blocks0` 上 Q10 大幅提升、`haberman` 上 Q10 明显下降的原因做深入定位。

## 10. 建议提交内容

建议提交时保留以下内容：

```text
REPRODUCTION_REPORT.md
README_REPRODUCTION.md
REPRODUCE_PROGRESS.md
code/
scripts/
datasets/
vendor/
references/
results/
```

建议不提交以下内容：

```text
build/
versions/
tmp 或 smoke test 结果
早期中间结果
原始探索过程中的临时脚本
```

如果提交大小受限，可以优先保留：

- `code/`
- `scripts/`
- `results/`
- `REPRODUCTION_REPORT.md`
- `README_REPRODUCTION.md`

并在文档中说明 CEKA 1.0.1、数据集和论文资料需要另行放置到对应目录。

## 11. 总结

总体而言，本项目完成了 FNNWV 论文主要实验的可运行复现，并补齐了若干原仓库缺失的对比方法实现。模拟数据集和真实 NER 数据集上的核心结果大体接近论文报告值，其中 FNNWV、LAWMV、PLAT 等结果较为接近。MNLDP 由于求解器缺失和实现差异，需要谨慎解释。

复现过程中，我们也检查并记录了 CEKA 数据复制语义、邻居集合是否包含自身、随机性控制等实现细节。这些记录有助于后续复查结果来源。

α 分析提供了一个有参考价值的现象：最远邻修正的触发比例会明显影响 FNNWV 表现，自适应 Q10 在平均值上表现较好，但目前并不能被表述为稳定优于原始 FNNWV。后续如果继续推进，更适合从解释不同数据集上的涨跌原因入手，而不是直接扩大改进结论。
