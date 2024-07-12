import dataclasses
from typing import List, Tuple, Literal

from sklearn.model_selection import train_test_split
import bigbench.api.results as bb
import pandas as pd
import numpy as np

from datasets.dataset_dict import DatasetDict

from lass.log_handling import LogLoader
import lass.pipeline


def to_dataframe(loader: LogLoader) -> pd.DataFrame:
    """
    Columns in the output dataframe:
    - input
    - targets
    - scores
    - target_values
    - correct
    - absolute_scores
    - normalized_scores
    - metrics (only for MPC)
    - output (only for gen/scoring)
    - task
    - shots
    - model_name
    - model_family
    """
    tasks: List[bb.ResultsFileData] = list(loader.load_per_model())
    dfs: List[pd.DataFrame] = []
    for task in tasks:
        queries_to_merge = []
        for query in task.queries or []:
            query_type = get_query_type(query)

            if query_type == "multiple_choice":
                df = pd.DataFrame(query.samples)
                df["model_name"] = task.model.model_name
                df["model_family"] = task.model.model_family
                df["task"] = task.task.task_name
                df["shots"] = query.shots
                df["query_type"] = query_type
                dfs.append(df)
            # If scoring/generative query, we need to merge them
            # because confidence scores are in the scoring query,
            # and exact_str_match and outputs are in the generative query
            else:
                queries_to_merge.append(query)

        # Merge scoring and generative queries
        # We assume they are interleaved generative - scoring - generative - scoring ...
        # Iterate per 2 queries + metadata
        for gen_q, score_q in zip(queries_to_merge[0::2], queries_to_merge[1::2]):
            if (
                get_query_type(gen_q) != "generative"
                or get_query_type(score_q) != "scoring"
            ):
                break  # Ignore the non-MPC queries if they are not interleaved gen - scoring pairs

            assert len(gen_q.samples) == len(score_q.samples)

            samples = [
                merge_scoring_generative_sample(gen_s, score_s)
                for gen_s, score_s in zip(gen_q.samples, score_q.samples)
            ]
            df = pd.DataFrame(samples)
            df["model_name"] = task.model.model_name
            df["model_family"] = task.model.model_family
            df["task"] = task.task.task_name
            df["shots"] = gen_q.shots
            df["query_type"] = "scoring_generative"
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f"No data found.")

    return pd.concat(dfs, ignore_index=True)


def merge_scoring_generative_sample(gen_sample, scoring_sample):
    assert scoring_sample.input == gen_sample.input
    # assert len(gen_sample.targets) == len(scoring_sample.targets) # Some weird data-bug with repeated targets in only the scoring sample causes this to not hold
    assert len(gen_sample.targets) >= 1
    sample = dataclasses.asdict(scoring_sample)
    sample["output"] = gen_sample.output
    sample["target_values"] = {target: 1 for target in scoring_sample.targets}
    sample["correct"] = max(
        [target["exact_str_match"] for target in gen_sample.targets.values()]
    )
    return sample


SplitType = Literal["instance", "task", "task_DS"]


def split(
    split: SplitType, df: pd.DataFrame, test_fraction: float, seed: int
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


def huggingfaceify_splits(train: pd.DataFrame, test: pd.DataFrame) -> DatasetDict:
    ds = DatasetDict()
    ds["train"] = lass.pipeline.huggingfaceify(train)
    ds["test"] = lass.pipeline.huggingfaceify(test)
    return ds


def get_query_type(query) -> Literal["multiple_choice", "generative", "scoring"]:
    if query.__class__ == bb.MultipleChoiceQuery:
        return "multiple_choice"
    elif query.__class__ == bb.GenerativeQuery:
        return "generative"
    elif query.__class__ == bb.ScoringQuery:
        return "scoring"
    else:
        raise ValueError(f"Unknown query class: {query.__class__}")
