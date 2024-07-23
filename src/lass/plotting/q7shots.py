import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import lass.plotting.shared as shared


@dataclass
class Args:
    path: Optional[Path]
    version: Optional[Literal["v1"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path)
    parser.add_argument("--version", type=str, choices=["v1"], default="v1")
    args_raw = parser.parse_args()
    run(Args(**vars(args_raw)))


def run(args: Args):
    base = Path("./artifacts/csv-results-new/")
    default = base / "q7shots" / "deberta-base_bs32_instance-split-07192221"  # fmt: skip
    path = args.path or default
    assert path.exists(), f"Path does not exist: {path}"

    dfs = {}
    long_seqs = []
    for truncation_side in ["left", "right"]:
        for shots in [0, 1, 2, 3]:
            model_dir = path / truncation_side / str(shots)
            metrics = shared.load_metrics(model_dir, load=["total"])
            long_seqs.append(metrics["n_long_sequences"].mean())
            dfs[(truncation_side, shots)] = metrics
    long_seqs = long_seqs[0:4]

    # Concat all dataframes
    results = pd.concat(dfs.values(), keys=dfs.keys())

    plot_path = base / ".." / "plots" / "q7shots"

    auroc, auroc_data = line_plot(results["test_roc_auc"].rename("metric"), "Assessor AUROC", long_seqs)  # fmt: skip
    shared.save_to(
        auroc,
        auroc_data,
        plot_path / f"q7scaling_auroc_{args.version}.pdf",
    )

    brier, brier_data = line_plot(results["test_bs_mcb"].rename("metric"), "Assessor Brier Miscalibration", long_seqs)  # fmt: skip
    shared.save_to(
        brier,
        brier_data,
        plot_path / f"q7scaling_brier_{args.version}.pdf",
    )


def line_plot(
    results: pd.Series, y_label: str, long_seqs: list[int]
) -> Tuple[Figure, pd.DataFrame]:
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    data = results.reset_index(level=[0, 1])
    data = data.reset_index(drop=True)
    data = data.rename(columns={"level_0": "truncation_side", "level_1": "shots"})

    # # Sort
    # order_truncation = {"left": 0, "right": 1}
    # order_subject = {"2m": 0, "16m": 1, "53m": 2, "125m": 3, "244m": 4, "422m": 5, "1b": 6, "2b": 7, "4b": 8, "8b": 9, "27b": 10, "128b": 11}  # fmt: skip
    # data = data.sort_values(
    #     by=["truncation_side", "subject_size"],
    #     key=(
    #         lambda series: series.map(order_subject)
    #         if series.name == "subject_size"
    #         else series.map(order_truncation)
    #     ),
    #     ascending=[True, True],
    # )

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.lineplot(
        data=data,
        x="shots",
        y="metric",
        hue="truncation_side",
        ax=ax,
    )
    # Add 1/30th of the range to the y-axis limits
    y_min, y_max = ax.get_ylim()
    y_min = y_min - (y_max / 30) if y_min > 0.05 else 0.0
    y_max = max(y_max + (y_max / 30), 0.02)
    ax.set_ylim(y_min, y_max)

    # Set x-ticks
    ax.set_xticks(data.shots.unique())

    # Add percentage of long seqs
    ax2 = plt.twinx()
    sns.barplot(
        x=[0, 1, 2, 3],
        y=[-v * 100 for v in long_seqs],
        color="pink",
        alpha=1,
        ax=ax2,
        width=0.8,
    )
    ax2.set_ylim(-100, 0)
    plt.yticks([])

    for container in ax2.containers:  # type: ignore
        ax2.bar_label(
            container, labels=[f"{v*100:.0f}%" for v in long_seqs], label_type="center"
        )

    ax.set_ylabel(y_label)
    ax.set_xlabel("Shots")
    ax.get_legend().set_title("Truncation Side")  # type: ignore
    ax.legend(loc="lower left")

    return fig, data


if __name__ == "__main__":
    main()
