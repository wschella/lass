from typing import *
import math

import bigbench.api.results as bb
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
import numpy as np

from datasets.splits import Split
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset

from lass.log_handling import LogLoader


def to_dataframe(loader: LogLoader) -> pd.DataFrame:
    tasks: List[bb.ResultsFileData] = list(loader.load_per_model())
    dfs: List[pd.DataFrame] = []
    for task in tasks:
        for query in (task.queries or []):
            df = pd.DataFrame(query.samples)
            df['task'] = task.task.task_name
            df['shots'] = query.shots
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def split_task_level(
    loader: LogLoader,
    seed: int,
    test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    """
    # TODO: Deal with multiple models!
    # TODO: Note that test_ratio is in _number_ of tasks, not necessarily in instances
    tasks: List[bb.ResultsFileData] = list(loader.load_per_model())

    train_tasks, test_tasks = train_test_split(tasks, test_size=test_fraction, random_state=seed)

    def to_dataframe(tasks: List[bb.ResultsFileData]) -> pd.DataFrame:
        dfs: List[pd.DataFrame] = []
        for task in tasks:
            for query in (task.queries or []):
                df = pd.DataFrame(query.samples)
                df['task'] = task.task.task_name
                df['shots'] = query.shots
                dfs.append(df)

        return pd.concat(dfs)

    df_train = to_dataframe(train_tasks)
    df_test = to_dataframe(test_tasks)

    return df_train, df_test


def analyse(df: pd.DataFrame) -> Dict[str, Any]:
    df_original = df
    df = df[df['correct'].isin([0.0, 1.0])]

    conf_normalized = df.apply(lambda row: math.exp(np.max(row['normalized_scores'])), axis=1)
    conf_absolute = df.apply(lambda row: math.exp(np.max(row['absolute_scores'])), axis=1)

    return {
        'stats': {
            'n_tasks': len(df['task'].unique()),
            'n_instances': len(df),
            'n_instances_nonbinary': len(df_original) - len(df),
        },
        'metrics': {
            'task-acc': df['correct'].mean(),
            'conf-normalized': {
                'roc_auc': metrics.roc_auc_score(df['correct'], conf_normalized),
            },
            'conf-absolute': {
                'roc_auc': metrics.roc_auc_score(df['correct'], conf_absolute),
            }
        },
    }


def merge(a: Dict[str, Any], b: Dict[str, Any], a_name: str, b_name: str) -> Dict[str, Any]:
    """
    For each leaf in a dict returned by analyse(), merge the stats of a en b
    into a new dict with key the name of the overal dict.

    Example result:
    {
        'task_names': {'a': [..tasks..], 'b': [..tasks..]},
    }
    """
    d: Dict[str, Any] = {}
    for (ka, va), (kb, vb) in zip(a.items(), b.items()):
        assert ka == kb, f"Keys of dicts don't match {ka} != {kb}"
        assert isinstance(va, dict) == isinstance(
            vb, dict), f"Types of  dicts don't match {va} != {vb}"

        if isinstance(va, dict):
            d[ka] = merge(va, vb, a_name, b_name)
        else:
            d[ka] = {a_name: va, b_name: vb}

    return d


def split_instance_level(
    loader: LogLoader, seed: int, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = to_dataframe(loader)
    df_train: pd.DataFrame = df.sample(
        frac=(1 - test_fraction), random_state=seed)  # type: ignore
    df_test = df.drop(df_train.index)
    return df_train, df_test


def huggingfaceify(train: pd.DataFrame, test: pd.DataFrame) -> DatasetDict:
    def huggingfaceify_(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a dataframe of BigBench samples for use with HuggingFace transformers.
        """
        df_hf = df.copy()

        # Take only the columns we need, and rename them appropriately
        df_hf = df[['input', 'correct']].rename(columns={'input': 'text', 'correct': 'label'})

        # Drop all samples that do not have binary correctness
        df_hf = df_hf[df_hf['label'].isin([0.0, 1.0])]

        # and convert the labels to ints afterwards
        df_hf[['label']] = df_hf[['label']].astype(int)
        return df_hf

    hf_train = huggingfaceify_(train)
    hf_test = huggingfaceify_(test)

    ds = DatasetDict()
    ds['train'] = Dataset.from_pandas(hf_train, split=Split.TRAIN, preserve_index=False)
    ds['test'] = Dataset.from_pandas(hf_test, split=Split.TEST, preserve_index=False)
    return ds
