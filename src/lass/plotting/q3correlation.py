import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import lass.plotting.shared as shared


@dataclass
class Args:
    path_assessors: Optional[Path]
    path_subjects: Optional[Path]
    shots: Literal[0, 3]
    version: Optional[Literal["v1"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-assessors", type=Path)
    parser.add_argument("--path-subjects", type=Path)
    parser.add_argument("--shots", type=int, choices=[0, 3], default=3)
    parser.add_argument("--version", type=str, choices=["v1"], default="v1")
    args_raw = parser.parse_args()
    run(Args(**vars(args_raw)))


def run(args: Args):
    base = Path("./artifacts/csv-results-new/")
    if args.shots == 3:
        default_subjects = base / "q3correlation" / "deberta-base_bs32_3sh_instance-split-07231721"  # fmt: skip
        default_assessors = base / "q4multitask" / "deberta-base_bs32_3sh_instance-split-07231725"  # fmt: skip
    if args.shots == 0:
        default_subjects = base / "q3correlation" / "deberta-base_bs32_0sh_instance-split-07231724"  # fmt: skip
        default_assessors = base / "q4multitask" / "deberta-base_bs32_0sh_instance-split-07231725"  # fmt: skip

    path_subjects = args.path_subjects or default_subjects
    path_assessors = args.path_assessors or default_assessors
    assert path_subjects.exists(), f"Path does not exist: {path_subjects}"
    assert path_assessors.exists(), f"Path does not exist: {path_assessors}"

    results_subjects = shared.load_metrics(path_subjects, load=["task"])
    results_assessors = shared.load_metrics(path_assessors, save=False, load=["task"])

    plot_path = base / ".." / "plots" / "q3correlation"
    if args.version == "v1":
        plot, plot_data = plotv1(results_subjects, results_assessors)
    else:
        raise ValueError(f"Invalid version: {args.version}")
    shared.save_to(plot, plot_data, plot_path / f"q3correlation_{args.version}_{args.shots}sh.pdf")  # fmt: skip


def plotv1(
    results_subjects: pd.DataFrame, results_assessors: pd.DataFrame
) -> Tuple[Figure, pd.DataFrame]:
    results_assessors.set_index("task", inplace=True)
    results_assessors = results_assessors[results_assessors.instance_count >= 100]

    results_subjects.set_index("task", inplace=True)
    results_subjects = results_subjects[results_subjects.instance_count >= 100]

    print(
        f"Total number of tasks: {len(results_assessors)}, with {len(results_subjects)} having finetuned BERT performance available."
    )

    # Select columns we care about
    data = pd.concat(
        [results_assessors["test_roc_auc"], results_subjects["test_roc_auc"]],
        keys=["assessors", "subjects"],
        axis=1,
    )
    data = data.dropna()
    print(data)

    plot = correlation_plot(data)
    return plot, data


def correlation_plot(data: pd.DataFrame) -> Figure:
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    fig = plt.figure(figsize=(8, 8))

    plt.scatter(data.subjects, data.assessors)
    plt.xlabel(f"Multiclass AUROC of subject DeBERTas finetuned on individual tasks")
    plt.ylabel(f"AUROC of Assessor for BIG-G 128b")

    min_val = min(min(data.subjects), min(data.assessors))
    plt.xlim(min_val - 0.1 * min_val, 1.015)
    plt.ylim(min_val - 0.1 * min_val, 1.015)

    # Plot diagonal line
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c="k")

    # Calculate correlation
    corr = data.corr().loc["subjects", "assessors"]
    plt.title(f"Correlation: {corr:.3f}")

    return fig


if __name__ == "__main__":
    main()
