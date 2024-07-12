import os
import logging
from pathlib import Path
from typing import Union, Literal
from pprint import pprint


from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from torch.nn.modules.module import Module

import lass.pipeline
import lass.datasets
import lass.metrics
import lass.metrics.baseline
from lass.metrics.baseline import analyse, merge
from lass.log_handling import LogLoader, LogLoaderArgs


def test(
    data_args: LogLoaderArgs,
    split: Union[Literal["instance"], Literal["task"], Literal["task_DS"]],
    model_loc: Union[str, Module],  # Can be location, or the actual model
    model_name: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    max_sequence_length: int = 512,
    include_model_in_input: bool = False,
    include_n_targets_in_input: bool = False,
    per_task: bool = False,
    # is_test_run: bool = False,
    # model_name_short: Optional[str] = None,
    # output_dir: Optional[Union[Path, str]] = None
):
    if type(model_loc) in [str, Path, bytes]:
        assert Path(model_loc).exists()  # type: ignore

    logging.info("Starting data loading")
    loader = LogLoader(data_args)
    data = lass.datasets.to_dataframe(loader)
    logging.info("Loaded data.")

    data = lass.pipeline.binarize(data)
    data = lass.pipeline.augment(data)
    data = lass.pipeline.clean(data)

    data = lass.pipeline.prepend_extra_features(
        data,
        include_model=include_model_in_input,
        include_n_targets=include_n_targets_in_input,
    )

    # In case of population-split, split will order by input. Make sure prepend_extra_features can not change the order.
    assert (
        not include_n_targets_in_input or not data.model_name.nunique() > 1
    ), "Population split not supported with include_n_targets_in_input"

    train, test = lass.datasets.split(
        split, data, test_fraction=test_fraction, seed=seed
    )

    # NOTE: Temporary hack to only test on 128b for population experiment
    # test = test.loc[test.model_name == '128b']

    # Log some stats & examples
    stats = merge(analyse(train), analyse(test), "train", "test")
    pprint(stats)
    print(train.head(1))

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

    # Evaluate per task
    tasks = {}
    if per_task:
        for task_name in test.task.unique():
            test_task = test.loc[test.task == task_name]
            labels = test_task["correct"]
            dist_baseline = lass.metrics.baseline.baseline(test_task)
            get_baseline = lass.metrics.hf.get_baseline_metrics
            compute_metrics_plus = lass.metrics.hf.join_metrics(
                metrics_assessor,
                get_baseline(
                    labels,
                    metrics,
                    test_task["conf_normalized"],
                    prefix="conf_normalized_",
                ),
                get_baseline(
                    labels, metrics, test_task["conf_absolute"], prefix="conf_absolute_"
                ),
                get_baseline(
                    labels, metrics, dist_baseline, prefix="conf_distribution_"
                ),
            )

            dummy_args = TrainingArguments(
                output_dir="tmp_trainer"
            )  # To silence warning
            dummy_trainer = Trainer(
                model=model, args=dummy_args, compute_metrics=compute_metrics_plus
            )

            test_task_hf = lass.pipeline.huggingfaceify(test_task)
            test_task_hf_tokenized = lass.pipeline.tokenize(
                test_task_hf, model_name, max_sequence_length
            )
            try:
                logits, labels, results = dummy_trainer.predict(test_task_hf_tokenized)  # type: ignore
                tasks[task_name] = {
                    "test": test_task,
                    "logits": logits,
                    "labels": labels,
                    "metrics": results,  # called something else to not clash with metric names
                }
            except Exception as e:
                print(f"Error in task {task_name}: {e}")
                raise e

    # Add baseline metrics as well so we can merge the plots in wandb
    labels = test["correct"]
    dist_baseline = lass.metrics.baseline.baseline(test)
    get_baseline = lass.metrics.hf.get_baseline_metrics
    compute_metrics_plus = lass.metrics.hf.join_metrics(
        metrics_assessor,
        get_baseline(
            labels, metrics, test["conf_normalized"], prefix="conf_normalized_"
        ),
        get_baseline(labels, metrics, test["conf_absolute"], prefix="conf_absolute_"),
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
    test_hf = lass.pipeline.huggingfaceify(test)
    print(test_hf[0])
    test_hf_tokenized = lass.pipeline.tokenize(test_hf, model_name, max_sequence_length)

    logits, labels, metrics = dummy_trainer.predict(test_hf_tokenized)  # type: ignore

    return (
        {
            "data": data,
            "train": train,
            "test": test,
            "logits": logits,
            "labels": labels,
            "metrics": metrics,
        }
        | {"tasks": tasks}
        if per_task
        else {}
    )
