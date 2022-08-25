from typing import *
import math

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics

import lass.metrics.brier


def baseline(df: pd.DataFrame) -> pd.Series:
    """
    Compute a baseline prediction that simply predicts the score distribution, 
    but disaggregated by task and model.
    """
    return df.groupby(['model_family', 'model_name', 'task'])['correct'].transform('mean')


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
                'acc': sk_metrics.accuracy_score(df['correct'], conf_normalized > 0.5),
                'balanced_acc': sk_metrics.balanced_accuracy_score(df['correct'], conf_normalized > 0.5),
                'roc_auc': sk_metrics.roc_auc_score(df['correct'], conf_normalized),
                **bs(df['correct'], conf_normalized)
            },
            'conf-absolute': {
                'acc': sk_metrics.accuracy_score(df['correct'], conf_absolute > 0.5),
                'balanced_acc': sk_metrics.balanced_accuracy_score(df['correct'], conf_absolute > 0.5),
                'roc_auc': sk_metrics.roc_auc_score(df['correct'], conf_absolute),
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
