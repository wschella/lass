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
import lass.metrics.brier


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

    return pd.concat(dfs, ignore_index=True)


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


def analyse(df: pd.DataFrame) -> Dict[str, Any]:
    df_original = df
    df = df[df['correct'].isin([0.0, 1.0])]

    conf_normalized = df.apply(lambda row: math.exp(np.max(row['normalized_scores'])), axis=1)
    conf_absolute = df.apply(lambda row: math.exp(np.max(row['absolute_scores'])), axis=1)

    def bs(target, confs):
        total, mcb, dsc, unc = lass.metrics.brier.brier_score(target, confs)
        return {"bs": total, "bs_mcb": mcb, "bs_dcr": dsc, "bs_unc": unc}

    return {
        'stats': {
            'n_tasks': len(df['task'].unique()),
            'n_instances': len(df),
            'n_instances_nonbinary': len(df_original) - len(df),
        },
        'metrics': {
            'task-acc': df['correct'].mean(),
            'conf-normalized': {
                'acc': metrics.accuracy_score(df['correct'], conf_normalized > 0.5),
                'balanced_acc': metrics.balanced_accuracy_score(df['correct'], conf_normalized > 0.5),
                'roc_auc': metrics.roc_auc_score(df['correct'], conf_normalized),
                **bs(df['correct'], conf_normalized)
            },
            'conf-absolute': {
                'acc': metrics.accuracy_score(df['correct'], conf_absolute > 0.5),
                'balanced_acc': metrics.balanced_accuracy_score(df['correct'], conf_absolute > 0.5),
                'roc_auc': metrics.roc_auc_score(df['correct'], conf_absolute),
                **bs(df['correct'], conf_absolute)
            },
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
