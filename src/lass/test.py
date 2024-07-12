import os
import logging
from pathlib import Path
from typing import Union
from pprint import pprint

import pandas as pd
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from torch.nn.modules.module import Module

import lass.data.wrangling
import lass.metrics
import lass.metrics.baseline
from lass.metrics.baseline import analyse, merge


def test(
    test_data: pd.DataFrame,
    model_loc: Union[Path, Module],  # Can be location, or the actual model
    model_name: str,
    max_sequence_length: int = 512,
):
    if type(model_loc) in [str, Path, bytes]:
        assert Path(model_loc).exists()  # type: ignore

    # Log some stats & examples
    stats = merge({}, analyse(test_data), "train", "test")
    pprint(stats)
    print(test_data.head(1))

    if type(model_loc) in [str, Path, bytes]:
        model: Module = AutoModelForSequenceClassification.from_pretrained(
            model_loc, num_labels=2
        )
    else:
        model: Module = model_loc  # type: ignore

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "balanced_accuracy",
    ]
    metrics_assessor = lass.metrics.hf.get_metric_computer(metrics)

    # Add baseline metrics as well so we can merge the plots in wandb
    labels = test_data["correct"]
    dist_baseline = lass.metrics.baseline.baseline(test_data)
    get_baseline = lass.metrics.hf.get_baseline_metrics
    compute_metrics_plus = lass.metrics.hf.join_metrics(
        metrics_assessor,
        get_baseline(
            labels, metrics, test_data["conf_normalized"], prefix="conf_normalized_"
        ),
        get_baseline(
            labels, metrics, test_data["conf_absolute"], prefix="conf_absolute_"
        ),
        get_baseline(labels, metrics, dist_baseline, prefix="conf_distribution_"),
    )

    # Dummy Trainer for easy batched predictions
    dummy_args = TrainingArguments(output_dir="tmp_trainer")  # To silence warning
    dummy_trainer = Trainer(
        model=model, args=dummy_args, compute_metrics=compute_metrics_plus
    )

    # Tokenize dataset
    logging.info("Starting tokenization")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    test_hf = lass.data.wrangling.huggingfaceify(test_data)
    print(test_hf[0])
    test_hf_tokenized = lass.data.wrangling.tokenize(
        test_hf, model_name, max_sequence_length
    )

    # Get Results
    logits, labels, metrics = dummy_trainer.predict(test_hf_tokenized)

    return {
        "data": test_data,
        "logits": logits,
        "labels": labels,
        "metrics": metrics,
    }


def test_per_task(
    test_data: pd.DataFrame,
    model_loc: Union[Path, Module],  # Can be location, or the actual model
    model_name: str,
    max_sequence_length: int = 512,
):
    tasks = {}
    for task_name in test_data.task.unique():
        task = test_data.query(f"task == '{task_name}'")
        try:
            tasks[task_name] = test(task, model_loc, model_name, max_sequence_length)
        except Exception as e:
            print(f"Error in task {task_name}: {e}")
            raise e
    return tasks
