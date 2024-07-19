from dataclasses import dataclass
import os
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from torch.nn.modules.module import Module

import lass.data.wrangling
import lass.metrics
import lass.metrics.stats


@dataclass
class TestResults:
    data: pd.DataFrame
    logits: np.ndarray
    labels: np.ndarray
    metrics: dict[str, Any]


def test(
    test_data: pd.DataFrame,
    model_loc: Union[Path, Module],  # Can be location, or the actual model
    model_name: str,
    max_sequence_length: int = 512,
) -> TestResults:
    if type(model_loc) in [str, Path, bytes]:
        assert Path(model_loc).exists()  # type: ignore

    # Log some stats & examples
    # stats = merge({}, analyse(test_data), "train", "test")
    # pprint(stats)
    # print(test_data.head(1))

    if isinstance(model_loc, (Path, str, bytes)):
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

    # Dummy Trainer for easy batched predictions
    baselines = lass.metrics.baseline.get_baselines(test_data, metrics)
    dummy_args = TrainingArguments(output_dir="tmp_trainer")  # To silence warning
    dummy_trainer = Trainer(
        model=model,
        args=dummy_args,
        compute_metrics=lambda predictions: (
            lass.metrics.compute_metrics_trainer(predictions, metrics) | baselines
        ),
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
    logits, labels, metrics_ = dummy_trainer.predict(test_hf_tokenized)
    assert isinstance(logits, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert metrics_ is not None

    # Add instance count to metrics
    metrics_["instance_count"] = len(test_data)

    return TestResults(test_data, logits, labels, metrics_)


def test_per_task(
    test_data: pd.DataFrame,
    model_loc: Union[Path, Module],  # Can be location, or the actual model
    model_name: str,
    max_sequence_length: int = 512,
) -> dict[str, TestResults]:
    tasks = {}
    for task_name in test_data.task.unique():
        task = test_data.query(f"task == '{task_name}'")
        try:
            tasks[task_name] = test(task, model_loc, model_name, max_sequence_length)
        except Exception as e:
            print(f"Error in task {task_name}: {e}")
            raise e
    return tasks
