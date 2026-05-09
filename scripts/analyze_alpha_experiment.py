import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


def method_key(row):
    return row["mode"], row["parameter"]


def method_label(key):
    mode, parameter = key
    if mode in {"MV", "LAWMV"}:
        return mode
    if mode == "original":
        return "original alpha=0.10"
    if mode == "fixed_alpha":
        return f"fixed alpha={float(parameter):.2f}"
    if mode == "adaptive_quantile":
        return f"Q{int(round(float(parameter) * 100))}"
    return f"{mode} {parameter}"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def rank_abs(values):
    indexed = sorted((abs(value), index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank(left, right):
    diffs = [l - r for l, r in zip(left, right) if abs(l - r) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0, 0
    ranks = rank_abs(diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, diffs) if diff > 0)
    w_minus = sum(rank for rank, diff in zip(ranks, diffs) if diff < 0)
    statistic = min(w_plus, w_minus)

    # Exact two-sided p-value for integer ranks without ties; normal approximation otherwise.
    has_ties = len({abs(diff) for diff in diffs}) != n
    if n <= 25 and not has_ties:
        rank_ints = [int(rank) for rank in ranks]
        total = 1 << n
        count = 0
        observed = statistic
        total_rank = sum(rank_ints)
        for mask in range(total):
            s = 0
            for i, rank in enumerate(rank_ints):
                if mask & (1 << i):
                    s += rank
            if min(s, total_rank - s) <= observed + 1e-12:
                count += 1
        p_value = count / total
    else:
        mean_w = n * (n + 1) / 4.0
        var_w = n * (n + 1) * (2 * n + 1) / 24.0
        z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return statistic, p_value, n


def load_summary(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["dataset"] != "average":
                rows.append(row)
    return rows


def main():
    summary_path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/FNNWV_alpha_experiment_summary.tsv")
    report_path = Path(sys.argv[2] if len(sys.argv) > 2 else "results/FNNWV_alpha_experiment_analysis.md")
    rows = load_summary(summary_path)

    by_method = defaultdict(list)
    by_dataset = defaultdict(dict)
    for row in rows:
        key = method_key(row)
        by_method[key].append(row)
        by_dataset[row["dataset"]][key] = row

    q10 = ("adaptive_quantile", "0.1000")
    original = ("original", "0.1000")
    lawmv = ("LAWMV", "0.0000")
    fixed_010 = ("fixed_alpha", "0.1000")

    lines = []
    lines.append("# FNNWV Alpha Experiment Analysis")
    lines.append("")
    lines.append(f"Source: `{summary_path}`")
    lines.append("")

    lines.append("## Average F1 and Trigger Rate")
    lines.append("")
    lines.append("| method | mean F1 | mean run std | std across datasets | mean trigger rate | trigger run std | trigger std across datasets |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(by_method, key=lambda item: (item[0], float(item[1]))):
        method_rows = by_method[key]
        f1_values = [float(row["mean_F1"]) for row in method_rows]
        f1_run_stds = [float(row["std_F1"]) for row in method_rows if row["std_F1"]]
        trigger_values = [float(row["mean_trigger_rate"]) for row in method_rows if row["mean_trigger_rate"]]
        trigger_run_stds = [float(row["std_trigger_rate"]) for row in method_rows if row["std_trigger_rate"]]
        trigger_mean = f"{mean(trigger_values):.4f}" if trigger_values else ""
        trigger_run_std = f"{mean(trigger_run_stds):.4f}" if trigger_run_stds else ""
        trigger_dataset_std = f"{std(trigger_values):.4f}" if trigger_values else ""
        lines.append(
            f"| {method_label(key)} | {mean(f1_values):.4f} | {mean(f1_run_stds):.4f} | "
            f"{std(f1_values):.4f} | {trigger_mean} | {trigger_run_std} | {trigger_dataset_std} |"
        )

    lines.append("")
    lines.append("## Q10 vs Original by Dataset")
    lines.append("")
    wins = losses = ties = 0
    diffs = []
    for dataset in sorted(by_dataset):
        if q10 not in by_dataset[dataset] or original not in by_dataset[dataset]:
            continue
        q10_f1 = float(by_dataset[dataset][q10]["mean_F1"])
        original_f1 = float(by_dataset[dataset][original]["mean_F1"])
        diff = q10_f1 - original_f1
        diffs.append((dataset, diff, q10_f1, original_f1))
        if diff > 1e-9:
            wins += 1
        elif diff < -1e-9:
            losses += 1
        else:
            ties += 1
    lines.append(f"Q10 wins/ties/losses vs original: **{wins}/{ties}/{losses}**")
    lines.append("")
    lines.append("| dataset | Q10 F1 | original F1 | diff |")
    lines.append("|---|---:|---:|---:|")
    for dataset, diff, q10_f1, original_f1 in sorted(diffs, key=lambda item: item[1], reverse=True):
        lines.append(f"| {dataset} | {q10_f1:.4f} | {original_f1:.4f} | {diff:+.4f} |")

    lines.append("")
    lines.append("## Largest Changes")
    lines.append("")
    for dataset, diff, q10_f1, original_f1 in sorted(diffs, key=lambda item: item[1], reverse=True)[:5]:
        lines.append(f"- Gain: {dataset}, {diff:+.4f} ({original_f1:.4f} -> {q10_f1:.4f})")
    for dataset, diff, q10_f1, original_f1 in sorted(diffs, key=lambda item: item[1])[:5]:
        lines.append(f"- Drop: {dataset}, {diff:+.4f} ({original_f1:.4f} -> {q10_f1:.4f})")

    lines.append("")
    lines.append("## Wilcoxon Signed-Rank Tests")
    lines.append("")
    lines.append("| comparison | n | statistic | p-value |")
    lines.append("|---|---:|---:|---:|")
    comparisons = [
        ("Q10 vs original", q10, original),
        ("Q10 vs LAWMV", q10, lawmv),
        ("Q10 vs fixed alpha=0.10", q10, fixed_010),
    ]
    for label, left_key, right_key in comparisons:
        left = []
        right = []
        for dataset in sorted(by_dataset):
            if left_key in by_dataset[dataset] and right_key in by_dataset[dataset]:
                left.append(float(by_dataset[dataset][left_key]["mean_F1"]))
                right.append(float(by_dataset[dataset][right_key]["mean_F1"]))
        statistic, p_value, n = wilcoxon_signed_rank(left, right)
        lines.append(f"| {label} | {n} | {statistic:.4f} | {p_value:.6f} |")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis to {report_path}")


if __name__ == "__main__":
    main()
