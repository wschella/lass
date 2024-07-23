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
    shots: int
    version: Optional[Literal["v1"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path)
    parser.add_argument("--version", type=str, choices=["v1"], default="v1")
    parser.add_argument("--shots", type=int, default=3, choices=[0, 3])
    args_raw = parser.parse_args()
    run(Args(**vars(args_raw)))


def run(args: Args):
    base = Path("./artifacts/csv-results-new/")
    default = base / "q6scaling" / "deberta_bs32_3sh_instance-split-07191732"  # fmt: skip
    if args.shots == 0:
        default = base / "q6scaling" / "deberta_bs32_0sh_instance-split-07201303"
    path = args.path or default
    assert path.exists(), f"Path does not exist: {path}"

    # Note, we don't have 244m per-task data for 0-shot due to some error
    dfs = {}
    for assessor_size in ["small", "base", "large"]:
        for model_dir in (path / assessor_size).iterdir():
            subject_size = model_dir.name
            metrics = shared.load_metrics(model_dir, load=["total"])
            dfs[(assessor_size, subject_size)] = metrics

    # Concat all dataframes
    results = pd.concat(dfs.values(), keys=dfs.keys())

    plot_path = base / ".." / "plots" / "q6scaling"

    auroc, auroc_data = line_plot(results["test_roc_auc"].rename("metric"), "Assessor AUROC")  # fmt: skip
    shared.save_to(
        auroc,
        auroc_data,
        plot_path / f"q6scaling_auroc_{args.version}_{args.shots}sh.pdf",
    )

    brier, brier_data = line_plot(results["test_bs_mcb"].rename("metric"), "Assessor Brier Miscalibration")  # fmt: skip
    shared.save_to(
        brier,
        brier_data,
        plot_path / f"q6scaling_brier_{args.version}_{args.shots}sh.pdf",
    )


def line_plot(results: pd.Series, y_label: str) -> Tuple[Figure, pd.DataFrame]:
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    data = results.reset_index(level=[0, 1])
    data = data.reset_index(drop=True)
    data = data.rename(columns={"level_0": "assessor_size", "level_1": "subject_size"})

    # Sort
    order_assessor = {"small": 0, "base": 1, "large": 2}
    order_subject = {"2m": 0, "16m": 1, "53m": 2, "125m": 3, "244m": 4, "422m": 5, "1b": 6, "2b": 7, "4b": 8, "8b": 9, "27b": 10, "128b": 11}  # fmt: skip
    data = data.sort_values(
        by=["assessor_size", "subject_size"],
        key=(
            lambda series: series.map(order_subject)
            if series.name == "subject_size"
            else series.map(order_assessor)
        ),
        ascending=[True, True],
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(
        data=data,
        x="subject_size",
        y="metric",
        hue="assessor_size",
        ax=ax,
    )
    # Add 1/30th of the range to the y-axis limits
    y_min, y_max = ax.get_ylim()
    y_min = y_min - (y_max / 30) if y_min > 0.05 else 0.0
    y_max = max(y_max + (y_max / 30), 0.02)
    ax.set_ylim(y_min, y_max)

    ax.set_ylabel(y_label)
    ax.set_xlabel("BIG-G Size")
    ax.get_legend().set_title("Assessor Size")  # type: ignore

    return fig, data


if __name__ == "__main__":
    main()
