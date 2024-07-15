import logging

import pandas as pd

import lass.utils
import lass.metrics


def distribution_baseline(df: pd.DataFrame) -> pd.Series:
    """
    Compute a baseline prediction that simply predicts the score distribution,
    but disaggregated by task and model.
    """
    return df.groupby(["model_family", "model_name", "task"])["correct"].transform(
        "mean"
    )


def get_baselines(df: pd.DataFrame, metrics: list[str]) -> dict[str, dict[str, float]]:
    # Check if multiple models present
    if df[["model_name", "model_family"]].nunique().max() > 1:
        logging.warning(
            "Data contains multiple models. Baselines for all models together, i.e. as if there was only one model."
        )

    # The label is for the 'failure prediction task', i.e. whether the model was correct or not.
    labels = df["correct"]
    distribution = lass.metrics.metrics.compute_metrics(
        lass.metrics.baseline.distribution_baseline(df),
        labels,
        metrics,
    )
    conf_normalized = lass.metrics.metrics.compute_metrics(
        df["conf_normalized"],
        labels,
        metrics,
    )
    conf_absolute = lass.metrics.metrics.compute_metrics(
        df["conf_absolute"],
        labels,
        metrics,
    )
    task_accuracy = df.groupby("task")["correct"].mean()

    return (
        lass.utils.prefix_keys(distribution, "conf_distribution_")
        | lass.utils.prefix_keys(conf_normalized, "conf_normalized_")
        | lass.utils.prefix_keys(conf_absolute, "conf_absolute_")
        | {"task_accuracy": task_accuracy.mean()}
    )
