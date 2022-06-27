from typing import *

from datasets.load import load_metric
import numpy as np
import scipy.special as sc_special
import sklearn.metrics as sk_metrics


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

brier_score = lambda predictions, references, confs: \
    {"brier_score": sk_metrics.brier_score_loss(references, confs)}

METRICS = {
    "accuracy": accuracy, "precision": precision,
    "recall": recall, "f1": f1, "roc_auc": roc_auc, "brier_score": brier_score
}


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
