from typing import *
import math

import pandas as pd
import numpy as np
from sklearn import metrics as sk_metrics


def LM_confidence_metrics(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    df = df[df['correct'].isin([0.0, 1.0])]
    conf_normalized = df.apply(lambda row: math.exp(np.max(row['normalized_scores'])), axis=1)
    conf_absolute = df.apply(lambda row: math.exp(np.max(row['absolute_scores'])), axis=1)
    return (
        confidence_metrics_(conf_normalized, df['correct']),  # type: ignore
        confidence_metrics_(conf_absolute, df['correct']),  # type: ignore
    )


def confidence_metrics(predictions: pd.Series, target: pd.Series) -> Dict[str, Any]:
    return {
        "roc_auc": sk_metrics.roc_auc_score(target, predictions),
        "brier_score": sk_metrics.brier_score_loss(target, predictions),
        "accuracy": accuracy(predictions > 0.5, target),
    }


def accuracy(predictions: pd.Series, references: pd.Series):
    return (predictions == references).mean()
