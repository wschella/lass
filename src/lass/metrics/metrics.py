import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.special as sc_special
import sklearn.metrics as sk_metrics
import wandb.plot

from datasets.load import load_metric
from transformers.trainer_utils import EvalPrediction

import lass.utils
import lass.metrics.brier
import lass.metrics.stats

accuracy_ = load_metric("accuracy")
accuracy = lambda predictions, references, probs: accuracy_.compute(
    predictions=predictions, references=references
)

precision_ = load_metric("precision")
precision = lambda predictions, references, probs: precision_.compute(
    predictions=predictions, references=references
)

recall_ = load_metric("recall")
recall = lambda predictions, references, probs: recall_.compute(
    predictions=predictions, references=references
)

f1_ = load_metric("f1")
f1 = lambda predictions, references, probs: f1_.compute(
    predictions=predictions, references=references
)

roc_auc_ = load_metric("roc_auc")


def roc_auc(predictions, references, probs) -> dict[str, float]:
    if is_unrocable(predictions, references, probs):
        return {"roc_auc": math.nan}

    return roc_auc_.compute(
        prediction_scores=probs,
        references=references,
    ) or {"roc_auc": math.nan}


roc_auc_multiclass_ = load_metric("roc_auc", config_name="multiclass")


def roc_auc_multiclass(predictions, references, probs) -> dict[str, float]:
    # We use micro, as we often work with small test sets, and we don't know much about the class importances.
    if is_unrocable(predictions, references, probs):
        return {"roc_auc": math.nan}

    return roc_auc_multiclass_.compute(
        prediction_scores=probs,
        references=references,
        average="weighted",
        multi_class="ovr",
    ) or {"roc_auc": math.nan}


def is_unrocable(predictions, references, probs) -> bool:
    n_classes = probs.shape[1] if (probs.ndim > 1 and probs.shape[1] > 2) else 2
    # ROC AUC is not defined when not all classes are present
    return len(np.unique(references)) != n_classes


balanced_accuracy = lambda predictions, references, probs: {
    "balanced_accuracy": sk_metrics.balanced_accuracy_score(references, predictions)
}


def brier_score(predictions, references, probs) -> dict[str, Any]:
    total, mcb, dsc, unc = lass.metrics.brier.brier_score(references, probs)
    return {
        "bs": total,
        "bs_mcb": mcb,
        "bs_dsc": dsc,
        "bs_unc": unc,
    }


def wandb_conf_matrix(predictions, references, probs) -> dict[str, float]:
    class_probs = np.c_[probs, 1 - probs]
    cm = wandb.plot.confusion_matrix(class_probs, references, class_names=["0", "1"])
    return {"wandb_conf_matrix": cm}


METRICS = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "roc_auc_multiclass": roc_auc_multiclass,
    "brier_score": brier_score,
    "balanced_accuracy": balanced_accuracy,
    "wandb_conf_matrix": wandb_conf_matrix,
}


def compute_metrics(
    predictions: pd.Series | np.ndarray,
    labels: pd.Series | np.ndarray,
    metrics: list,
) -> dict[str, float]:
    metrics = [METRICS[metric] for metric in metrics]
    scores = {}
    for metric in metrics:
        # Most metrics use the probabilities, but some (like accuracy) need a decision (e.g. conf > 0.5)
        score = metric(predictions > 0.5, labels, predictions)
        assert score is not None
    return scores


def compute_metrics_trainer(
    predictions: EvalPrediction, metrics: list[str]
) -> dict[str, float]:
    logits = predictions.predictions
    labels = predictions.label_ids
    if isinstance(logits, tuple) or isinstance(labels, tuple):
        raise NotImplementedError("This function does not support tuple predictions")

    probs = sc_special.softmax(logits, axis=-1)
    if logits.shape[1] == 2:  # binary classification, only keep prob of label 1
        probs = probs[:, -1]

    return compute_metrics(probs, labels, metrics)
