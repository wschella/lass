import json
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

FIG_WIDTH1 = 3.5
FIG_HEIGHT_MULT1 = 0.25


def plot_difference(
    sys1: pd.Series,
    sys2: pd.Series,
    sys1_threshold: Optional[float] = None,
    sys2_threshold: Optional[float] = None,
    sort: Literal["name", "sys1", "sys2", "diff"] = "name",
    sort_direction: Literal["asc", "desc"] = "asc",
    xlabel: Optional[str] = None,
) -> Tuple[Figure, pd.DataFrame]:
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Verify that the index is called 'task'
    assert sys1.index.name == "task"
    assert sys2.index.name == "task"

    # Combine the two systems into one dataframe
    data = pd.concat([sys1, sys2], axis=1, keys=["sys1", "sys2"])
    data = data.reset_index()  # Make task just a regular column
    data["diff"] = data["sys2"] - data["sys1"]

    # Filter tasks that are not in both systems
    data = data.dropna()

    # Filter out tasks that don't meet the threshold
    if sys1_threshold is not None:
        data = data[data["sys1"] > sys1_threshold]
    if sys2_threshold is not None:
        data = data[data["sys2"] > sys2_threshold]

    # Sort the data
    if sort == "name":
        data = data.sort_values("task", ascending=sort_direction == "asc")
    elif sort == "sys1":
        data = data.sort_values("sys1", ascending=sort_direction == "asc")
    elif sort == "sys2":
        data = data.sort_values("sys2", ascending=sort_direction == "asc")
    elif sort == "diff":
        data = data.sort_values("diff", ascending=sort_direction == "asc")
    else:
        raise ValueError(f"Invalid sort value: {sort}")

    # Put total at the end
    total = data[data.task == "total"]
    data = data[data.task != "total"]
    data = pd.concat([data, total])

    y = data.apply(lambda x: f"{shorten(x.task, length=22)} ({x.sys1:.2f})", axis=1)

    height = len(data) * FIG_HEIGHT_MULT1
    fig = plt.figure(figsize=(FIG_WIDTH1, height))
    p = sns.barplot(x="diff", y=y, data=data, orient="h")

    # Set the bar labels
    for i in p.containers:  # type: ignore
        p.bar_label(i, fmt="%+.2f", label_type="edge", padding=2)
    p.set_xlim(p.get_xlim()[0] * 1.5, p.get_xlim()[1] * 1.4)
    p.set_ylim(p.get_ylim()[0] * 1.0, p.get_ylim()[1] * 1.3)

    if xlabel is not None:
        p.set_xlabel(xlabel)

    return fig, data


def plot_absolute(
    results: pd.Series,
    reference: pd.Series,
    results_threshold: Optional[float] = None,
    reference_threshold: Optional[float] = None,
    sort: Literal["name", "results", "reference", "diff"] = "name",
    sort_direction: Literal["asc", "desc"] = "asc",
    xlabel: Optional[str] = None,
    marker_line: Optional[float] = None,
) -> Tuple[Figure, pd.DataFrame]:
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    # Verify that the index is called 'task'
    assert results.index.name == "task"
    assert reference.index.name == "task"

    # Combine the two systems into one dataframe
    data = pd.concat([results, reference], axis=1, keys=["results", "reference"])
    data = data.reset_index()  # Make task just a regular column
    data["diff"] = data.results - data.reference

    # Filter tasks that are not in both systems
    data = data.dropna()

    # Filter out tasks that don't meet the threshold
    if results_threshold is not None:
        data = data[data["results"] > results_threshold]
    if reference_threshold is not None:
        data = data[data["reference"] > reference_threshold]

    # Sort the data
    if sort == "name":
        data = data.sort_values("task", ascending=sort_direction == "asc")
    elif sort == "results":
        data = data.sort_values("results", ascending=sort_direction == "asc")
    elif sort == "reference":
        data = data.sort_values("reference", ascending=sort_direction == "asc")
    elif sort == "diff":
        data = data.sort_values("diff", ascending=sort_direction == "asc")
    else:
        raise ValueError(f"Invalid sort value: {sort}")

    # Put total at the end
    total = data[data.task == "total"]
    data = data[data.task != "total"]
    data = pd.concat([data, total])

    y = data.apply(
        lambda x: f"{shorten(x.task, length=22)} ({x.reference:.2f})", axis=1
    )

    height = len(data) * FIG_HEIGHT_MULT1
    fig = plt.figure(figsize=(FIG_WIDTH1, height))
    p = sns.barplot(x="results", y=y, data=data, orient="h")

    # Set the bar labels
    for i in p.containers:  # type: ignore
        labels = [
            f"{m:.2f} ({diff:+.2f})"
            for diff, m in zip(data["diff"].values, data.results.values)
        ]
        p.bar_label(i, labels=labels, label_type="edge", padding=2)
    p.set_xlim(p.get_xlim()[0] * 1.2, p.get_xlim()[1] * 3.8)
    p.set_ylim(p.get_ylim()[0] * 1.0, p.get_ylim()[1] * 1.5)
    if xlabel is not None:
        p.set_xlabel(xlabel)

    # # Format x-tick labels only 2 decimal places
    p.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))  # type: ignore

    # Set x limit
    max = data["results"].max()
    p.set_xlim(0.0, 1.1)
    if max < 0.45:
        p.set_xlim(0.0, 0.5)
    if max < 0.20:
        p.set_xlim(0.0, 0.25)
    if max < 0.15:
        p.set_xlim(0.0, 0.20)

    if marker_line is not None:
        p.axvline(marker_line, color="red", linestyle="--", alpha=0.5)

    return fig, data


def save_to(fig: Figure, data: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    data.to_csv(path.with_suffix(".csv"), index=False)


def shorten(x, length=15):
    """
    Shorten a string to a specific length, adding ellipsis to the end.
    """
    if len(x) > length:
        return x[: length - 3] + "..."
    else:
        return x


def load_metrics(
    dir: Path, save: bool = True, load: list = ["total", "task"]
) -> pd.DataFrame:
    """
    Load total and per task metrics from result directory.
    Also saves the metrics to a csv file in the same directory.
    """

    # Load task based metrics
    results = pd.DataFrame()
    if "task" in load:
        results = pd.read_json(dir / "metrics_per_task.json")
        results = results.T
        results.index.names = ["task"]
        results = results.reset_index()

        # Filter out these tasks with (temporary) contamination issues.
        results = results[
            ~results.task.isin(
                [
                    "unit_conversion",
                    "geometric_shapes",
                    "minute_mysteries_qa",
                    "strategyqa",
                    "penguins_in_a_table",
                    "similarities_abstraction",
                    "kanji_ascii",
                    "emoji_movie",
                    "color",
                    "checkmate_in_one",
                ]
            )
        ]

    # Load total dataset metrics
    total = pd.DataFrame()
    if "total" in load:
        with open(dir / "metrics.json") as f:
            total = pd.json_normalize(json.load(f))
            total["task"] = "total"

    results = pd.concat([results, total], ignore_index=True)

    # Put all columns starting with test_conf_ last
    results = results[sorted(results.columns, key=lambda x: x.startswith("test_conf_"))]

    # Save to csv for easier sharing with journal and co-authors
    if save:
        results.to_csv(dir / "metrics.csv", index=False)

    return results


Baseline = Literal["normalized", "absolute", "distribution"]
