import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = "minresults_karman_results_parallel_main.npy"
TARGET_T = 5.0
OUTPUT_PATH = "strong_scaling_T5_runtime.png"
SPEEDUP_OUTPUT_PATH = "strong_scaling_T5_speedup.png"
EFFICIENCY_OUTPUT_PATH = "strong_scaling_T5_efficiency.png"


def combo_label(result):
    return (
        f"{result['solver_1']}/{result['precond1']} | "
        f"{result['solver_2']}/{result['precond2']} | "
        f"{result['solver_3']}/{result['precond3']}"
    )


def main():
    data = np.load(DATA_PATH, allow_pickle=True).tolist()
    filtered = [
        row
        for row in data
        if row["T"] == TARGET_T
        and "ml" not in (row["precond1"], row["precond2"], row["precond3"])
    ]

    if not filtered:
        raise ValueError(f"No entries found for T = {TARGET_T} in {DATA_PATH}.")

    grouped = {}
    for row in filtered:
        label = combo_label(row)
        grouped.setdefault(label, []).append(
            (row["number of ranks"], row["total_runtime"])
        )

    fig, ax = plt.subplots(figsize=(11, 7))

    for label, values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        ranks = [rank for rank, _ in values]
        runtimes = [runtime for _, runtime in values]
        ax.plot(ranks, runtimes, marker="o", linewidth=2, label=label)

    ax.set_title(f"Strong Scaling for Karman Run at T = {TARGET_T}")
    ax.set_xlabel("Number of ranks")
    ax.set_ylabel("Total runtime [s]")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({row["number of ranks"] for row in filtered}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")

    fig_speedup, ax_speedup = plt.subplots(figsize=(11, 7))

    for label, values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        ranks = [rank for rank, _ in values]
        runtimes = [runtime for _, runtime in values]
        baseline_runtime = runtimes[0]
        speedup = [baseline_runtime / runtime for runtime in runtimes]
        ax_speedup.plot(ranks, speedup, marker="o", linewidth=2, label=label)

    unique_ranks = sorted({row["number of ranks"] for row in filtered})
    ax_speedup.plot(unique_ranks, unique_ranks, linestyle="--", color="black", alpha=0.6, label="ideal")
    ax_speedup.set_title(f"Speedup for Karman Run at T = {TARGET_T}")
    ax_speedup.set_xlabel("Number of ranks")
    ax_speedup.set_ylabel("Speedup")
    ax_speedup.set_xscale("log", base=2)
    ax_speedup.set_xticks(unique_ranks)
    ax_speedup.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_speedup.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_speedup.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig_speedup.tight_layout()
    fig_speedup.savefig(SPEEDUP_OUTPUT_PATH, dpi=200, bbox_inches="tight")

    fig_efficiency, ax_efficiency = plt.subplots(figsize=(11, 7))

    for label, values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        ranks = [rank for rank, _ in values]
        runtimes = [runtime for _, runtime in values]
        baseline_runtime = runtimes[0]
        efficiency = [baseline_runtime / (rank * runtime) for rank, runtime in zip(ranks, runtimes)]
        ax_efficiency.plot(ranks, efficiency, marker="o", linewidth=2, label=label)

    ax_efficiency.axhline(1.0, linestyle="--", color="black", alpha=0.6, label="ideal")
    ax_efficiency.set_title(f"Parallel Efficiency for Karman Run at T = {TARGET_T}")
    ax_efficiency.set_xlabel("Number of ranks")
    ax_efficiency.set_ylabel("Efficiency")
    ax_efficiency.set_xscale("log", base=2)
    ax_efficiency.set_xticks(unique_ranks)
    ax_efficiency.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_efficiency.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_efficiency.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig_efficiency.tight_layout()
    fig_efficiency.savefig(EFFICIENCY_OUTPUT_PATH, dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
