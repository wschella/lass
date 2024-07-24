from dataclasses import dataclass
import os
import logging
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import pandas as pd
from transformers.models.auto.modeling_auto import AutoModelForMultipleChoice
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
    truncation_side: Union[Literal["left"], Literal["right"]] = "right",
    original_task: bool = False,
    batch_size: int = 1,
) -> TestResults:
    if type(model_loc) in [str, Path, bytes]:
        assert Path(model_loc).exists()  # type: ignore

    # Log some stats & examples
    # stats = merge({}, analyse(test_data), "train", "test")
    # pprint(stats)
    # print(test_data.head(1))

    model = get_model(model_loc, original_task)

    # Tokenize dataset
    logging.info("Starting tokenization")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if original_task:
        n_targets = test_data["n_targets"].iloc[0]
        test_hf = lass.data.wrangling.huggingfaceify_original(test_data)
        test_hf_tokenized = lass.data.wrangling.tokenize_mpc(
            test_hf, model_name, max_sequence_length, n_targets, truncation_side
        )
        pass
    else:
        test_hf = lass.data.wrangling.huggingfaceify(test_data)
        test_hf_tokenized = lass.data.wrangling.tokenize(
            test_hf, model_name, max_sequence_length, truncation_side
        )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "balanced_accuracy",
    ]

    if original_task:
        metrics = ["accuracy"]
        metrics += ["roc_auc_multiclass"] if n_targets > 2 else ["roc_auc"]

    if original_task:
        baselines = {}
    else:
        baselines = lass.metrics.baseline.get_baselines(test_data, metrics)

    # Dummy Trainer for easy batched predictions
    # TODO: Actually batch predictions
    dummy_args = TrainingArguments(
        output_dir="tmp_trainer",  # To silence warning
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
    )
    dummy_trainer = Trainer(
        model=model,
        args=dummy_args,
        compute_metrics=lambda predictions: (
            lass.metrics.compute_metrics_trainer(predictions, metrics) | baselines
        ),
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
    truncation_side: Union[Literal["left"], Literal["right"]] = "right",
    original_task: bool = False,
    batch_size: int = 1,
) -> dict[str, TestResults]:
    tasks = {}

    # Load the model only once
    model = get_model(model_loc, original_task)

    for task_name in test_data.task.unique():
        task = test_data.query(f"task == '{task_name}'")
        try:
            tasks[task_name] = test(
                task,
                model,
                model_name,
                max_sequence_length,
                truncation_side,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"Error in task {task_name}: {e}")
            raise e
    return tasks


def get_model(model_loc: Union[Path, Module], original_task: bool) -> Module:
    if isinstance(model_loc, (Path, str, bytes)):
        if original_task:
            model = AutoModelForMultipleChoice.from_pretrained(model_loc)
        else:
            model: Module = AutoModelForSequenceClassification.from_pretrained(
                model_loc, num_labels=2
            )
    else:
        model: Module = model_loc  # type: ignore
    return model
