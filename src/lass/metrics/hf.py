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
accuracy = lambda predictions, references, confs: \
    accuracy_.compute(predictions=predictions, references=references)

precision_ = load_metric("precision")
precision = lambda predictions, references, confs: \
    precision_.compute(predictions=predictions, references=references)

recall_ = load_metric("recall")
recall = lambda predictions, references, confs: \
    recall_.compute(predictions=predictions, references=references)

f1_ = load_metric("f1")
f1 = lambda predictions, references, confs: \
    f1_.compute(predictions=predictions, references=references)

roc_auc_ = load_metric("roc_auc")
roc_auc = lambda predictions, references, confs: \
    roc_auc_.compute(prediction_scores=confs, references=references)

balanced_accuracy = lambda predictions, references, confs: \
    {"balanced_accuracy": sk_metrics.balanced_accuracy_score(references, predictions)}


def brier_score(predictions, references, confs):
    total, mcb, dsc, unc = lass.metrics.brier.brier_score(references, confs)
    return {
        "bs": total,
        "bs_mcb": mcb,
        "bs_dsc": dsc,
        "bs_unc": unc,
    }


def wandb_conf_matrix(predictions, references, confs):
    class_probs = np.c_[confs, 1 - confs]
    cm = wandb.plot.confusion_matrix(class_probs, references, class_names=["0", "1"])
    return {"wandb_conf_matrix": cm}


METRICS = {
    "accuracy": accuracy, "precision": precision,
    "recall": recall, "f1": f1, "roc_auc": roc_auc,
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
    metrics = [METRICS[metric] for metric in metrics]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        confs = sc_special.softmax(logits, axis=-1)[:, -1]

        scores = {}
        for metric in metrics:
            score = metric(predictions, labels, confs)
            assert score is not None
            scores.update(score)

        return scores

    return compute_metrics


def join_metrics(*computers):
    return lambda eval_pred: dict(ChainMap(*[computer(eval_pred) for computer in computers]))
