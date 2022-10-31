from typing import *

from sklearn.model_selection import train_test_split
import bigbench.api.results as bb
import pandas as pd
import numpy as np

from datasets.splits import Split
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset

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
    - metrics
    - task
    - shots
    - model_name
    - model_family
    """
    tasks: List[bb.ResultsFileData] = list(loader.load_per_model())
    dfs: List[pd.DataFrame] = []
    for task in tasks:
        for query in (task.queries or []):
            df = pd.DataFrame(query.samples)
            df['model_name'] = task.model.model_name
            df['model_family'] = task.model.model_family
            df['task'] = task.task.task_name
            df['shots'] = query.shots
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f'No data found.')

    return pd.concat(dfs, ignore_index=True)


SplitType = Literal['instance', 'task', 'task_DS']


def split(
    split: SplitType,
    df: pd.DataFrame,
    test_fraction: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and test, with multiple split types possible,
    e.g. at a task level or instance level.
    """
    if split == 'instance':
        train, test = split_instance_level(df, seed, test_fraction)
    elif split == 'task':
        train, test = split_task_level(df, seed, test_fraction)
    elif split == 'task_DS':
        train, test = split_task_level_distribution_shift(df, seed)
    else:
        raise ValueError(f'Unknown split {split}')

    return train, test


def split_task_level(
    df: pd.DataFrame,
    seed: int,
    test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Note that test_ratio is in number of _tasks_, not instances.
    """
    train_tasks, test_tasks = train_test_split(
        df['task'].unique(), test_size=test_fraction, random_state=seed)
    train = df[df['task'].isin(train_tasks)]
    test = df[df['task'].isin(test_tasks)]
    return train, test


def split_task_level_distribution_shift(
    df: pd.DataFrame,
    test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    accs = (df
            .groupby('task', as_index=False).agg(acc=('correct', 'mean'))  # type: ignore
            .sort_values('acc', ascending=False))
    n_train_tasks = int(len(accs['task']) * (1 - test_fraction))
    train_tasks, test_tasks = np.split(accs['task'], [n_train_tasks])

    train = df[df['task'].isin(train_tasks)]
    test = df[df['task'].isin(test_tasks)]
    return train, test


def split_instance_level(
    df: pd.DataFrame, seed: int, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.groupby('task').sample(frac=(1 - test_fraction), random_state=seed)
    test = df.drop(train.index)
    return train, test


def huggingfaceify_splits(train: pd.DataFrame, test: pd.DataFrame) -> DatasetDict:
    ds = DatasetDict()
    ds['train'] = lass.pipeline.huggingfaceify(train)
    ds['test'] = lass.pipeline.huggingfaceify(test)
    return ds
