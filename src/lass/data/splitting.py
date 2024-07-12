from typing import Tuple, Literal

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SplitType = Literal["instance", "task", "task_DS"]


def split(
    df: pd.DataFrame, split: SplitType, test_fraction: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and test, with multiple split types possible,
    e.g. at a task level or instance level.
    """
    if split == "instance":
        if "model_name" in df.columns and df.model_name.nunique() > 1:
            train, test = split_instance_level_per_model(df, seed, test_fraction)
        else:
            train, test = split_instance_level(df, seed, test_fraction)
    elif split == "task":
        train, test = split_task_level(df, seed, test_fraction)
    elif split == "task_DS":
        train, test = split_task_level_distribution_shift(df, seed)
    else:
        raise ValueError(f"Unknown split {split}")

    return train, test


def split_task_level(
    df: pd.DataFrame, seed: int, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Note that test_ratio is in number of _tasks_, not instances.
    """
    train_tasks, test_tasks = train_test_split(
        df["task"].unique(), test_size=test_fraction, random_state=seed
    )
    train = df[df["task"].isin(train_tasks)]
    test = df[df["task"].isin(test_tasks)]
    return train, test


def split_task_level_distribution_shift(
    df: pd.DataFrame, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    accs = (
        df.groupby("task", as_index=False)
        .agg(acc=("correct", "mean"))  # type: ignore
        .sort_values("acc", ascending=False)
    )
    n_train_tasks = int(len(accs["task"]) * (1 - test_fraction))
    train_tasks, test_tasks = np.split(accs["task"], [n_train_tasks])

    train = df[df["task"].isin(train_tasks)]
    test = df[df["task"].isin(test_tasks)]
    return train, test


def split_instance_level(
    df: pd.DataFrame, seed: int, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.groupby(["task"]).sample(frac=(1 - test_fraction), random_state=seed)
    test = df.drop(train.index)
    return train, test


def split_instance_level_per_model(
    df: pd.DataFrame, seed: int, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For population experiments.
    NOTE: Dangerous assumptions here that might easily no longer hold in updates.
    - We assume that the instances are the same for each model, and they are ordered the same. We have tested this.
    - We assume that df.sample does not work by index, but but just by position.
       - We have verified this in df.sample: https://github.com/pandas-dev/pandas/blob/v1.5.3/pandas/core/generic.py#L5773
       - And in df.groupby.sample: https://github.com/pandas-dev/pandas/blob/main/pandas/core/groupby/groupby.py#L4133-L4262
    """
    print(
        "WARNING: Dangerous assumptions here that might easily no longer hold in updates."
    )
    train = df.groupby(["task", "model_name", "model_family"], group_keys=False).apply(
        lambda frame: frame.sort_values("input").sample(
            frac=(1 - test_fraction), random_state=seed
        )
    )
    test = df.drop(train.index)
    return train, test
