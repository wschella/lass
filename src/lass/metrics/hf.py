from typing import *
from collections import ChainMap

import numpy as np
import pandas as pd
import scipy.special as sc_special
import sklearn.metrics as sk_metrics
import wandb.plot

from datasets.load import load_metric
from transformers.trainer_callback import TrainerCallback

import lass.metrics.brier

accuracy_ = load_metric("accuracy")
accuracy = lambda predictions, references, probs: \
    accuracy_.compute(predictions=predictions, references=references)

precision_ = load_metric("precision")
precision = lambda predictions, references, probs: \
    precision_.compute(predictions=predictions, references=references)

recall_ = load_metric("recall")
recall = lambda predictions, references, probs: \
    recall_.compute(predictions=predictions, references=references)

f1_ = load_metric("f1")
f1 = lambda predictions, references, probs: \
    f1_.compute(predictions=predictions, references=references)

roc_auc_ = load_metric("roc_auc")
roc_auc = lambda predictions, references, probs: \
    {"roc_auc": 0} if is_unrocable(predictions, references, probs) else \
    roc_auc_.compute(
        prediction_scores=probs,
        references=references,
    )

# We use micro, as we often work with small test sets, and we don't know much about the class importances.
roc_auc_multiclass_ = load_metric("roc_auc", config_name="multiclass")
roc_auc_multiclass = lambda predictions, references, probs: \
    {"roc_auc": 0} if is_unrocable(predictions, references, probs) else \
    roc_auc_multiclass_.compute(
        prediction_scores=probs,
        references=references,
        average="weighted", multi_class="ovr"
    )


def is_unrocable(predictions, references, probs) -> bool:
    n_classes = probs.shape[1] if (probs.ndim > 1 and probs.shape[1] > 2) else 2
    # ROC AUC is not defined when not all classes are present
    return len(np.unique(references)) != n_classes


balanced_accuracy = lambda predictions, references, probs: \
    {"balanced_accuracy": sk_metrics.balanced_accuracy_score(references, predictions)}


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
    "accuracy": accuracy, "precision": precision,
    "recall": recall, "f1": f1, "roc_auc": roc_auc,
    "roc_auc_multiclass": roc_auc_multiclass,
    "brier_score": brier_score, "balanced_accuracy": balanced_accuracy,
    "wandb_conf_matrix": wandb_conf_matrix,
}


def get_baseline_metrics(labels, metrics: list, baseline: pd.Series, prefix: str = ""):
    metrics = [METRICS[metric] for metric in metrics]
    scores = {}
    prefixer = lambda d: {f"{prefix}{k}": v for k, v in d.items()}
    for metric in metrics:
        score = metric(baseline > 0.5, labels, baseline.values)
        assert score is not None
        scores.update(prefixer(score))
    return lambda eval_pred: scores


def get_metric_computer(metrics: list):
    """
    Args:
    -----
        metrics: list of metric names

    Returns:
    --------
        A function that takes a batch of predictions+labels and returns
        a dict with calculated metrics.
    """
    metrics = [METRICS[metric] for metric in metrics]

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        probs = sc_special.softmax(logits, axis=-1)
        if logits.shape[1] == 2:  # binary classification, only keep prob of label 1
            probs = probs[:, -1]

        scores: dict[str, float] = {}
        for metric in metrics:
            score = metric(predictions, labels, probs)
            assert score is not None
            scores.update(score)

        return scores

    return compute_metrics


def join_metrics(*computers):
    return lambda eval_pred: dict(ChainMap(*[computer(eval_pred) for computer in computers]))
