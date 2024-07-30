import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from matplotlib.figure import Figure
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import lass.plotting.shared as shared


@dataclass
class Args:
    path_assessors: Optional[Path]
    path_subjects: Optional[Path]
    shots: Literal[0, 3, -1]
    version: Optional[Literal["v1"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-assessors", type=Path)
    parser.add_argument("--path-subjects", type=Path)
    parser.add_argument("--shots", type=int, choices=[0, 3, -1], default=-1)
    parser.add_argument("--version", type=str, choices=["v1"], default="v1")
    args_raw = parser.parse_args()
    run(Args(**vars(args_raw)))


def run(args: Args):
    base = Path("./artifacts/csv-results-new/")

    def load(default: Path, args_path: Path | None, save: bool = True) -> pd.DataFrame:
        path = args_path or default
        assert path.exists(), f"Path does not exist: {path}"
        return shared.load_metrics(path, save=save, load=["task"])

    results_subjects3 = load(base / "q3correlation" / "deberta-small_bs32_3sh_instance-split-07241130", args.path_subjects)  # fmt: skip
    results_assessors3 = load(base / "q4multitask" / "deberta-base_bs32_3sh_instance-split-07231725", args.path_assessors, save=False)  # fmt: skip
    results_subjects0 = load(base / "q3correlation" / "deberta-small_bs32_0sh_instance-split-07241130", args.path_subjects)  # fmt: skip
    results_assessors0 = load(base / "q4multitask" / "deberta-base_bs32_0sh_instance-split-07231725", args.path_assessors, save=False)  # fmt: skip

    if args.shots == 3:
        results_subjects = results_subjects3
        results_assessors = results_assessors3
    elif args.shots == 0:
        results_subjects = results_subjects0
        results_assessors = results_assessors0
    elif args.shots == -1:
        results_subjects3["task"] = results_subjects3["task"] + "_3sh"
        results_subjects0["task"] = results_subjects0["task"] + "_0sh"
        results_assessors3["task"] = results_assessors3["task"] + "_3sh"
        results_assessors0["task"] = results_assessors0["task"] + "_0sh"
        results_subjects = pd.concat([results_subjects3, results_subjects0])
        results_assessors = pd.concat([results_assessors3, results_assessors0])

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
    # results_assessors = results_assessors[results_assessors.instance_count >= 100]
    results_assessors = results_assessors[results_assessors.instance_count >= 30]

    results_subjects.set_index("task", inplace=True)
    # results_subjects = results_subjects[results_subjects.instance_count >= 100]
    results_subjects = results_subjects[results_subjects.instance_count >= 30]

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
    max_val = max(max(data.subjects), max(data.assessors))
    plt.xlim(min_val - 0.1 * min_val, max_val + 0.1 * max_val)
    plt.ylim(min_val - 0.1 * min_val, max_val + 0.1 * max_val)

    # Plot diagonal line
    # plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c="k")

    # Calculate correlation
    # corr = data.corr("pearson").loc["subjects", "assessors"]
    corr, p = pearsonr(data.subjects, data.assessors)
    plt.title(f"Pearson Correlation: {corr:.2f} ({p:.2f})")

    return fig


if __name__ == "__main__":
    main()
