from gc import callbacks
import os
import logging
from typing import *
from pathlib import Path
from dataclasses import dataclass
from pprint import pprint
from datetime import datetime

import wandb
import pandas as pd

from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset

import lass.datasets
import lass.pipeline
import lass.metrics
import lass.metrics.baseline
from lass.metrics.baseline import analyse, merge
from lass.log_handling import LogLoader, LogLoaderArgs


def train(
    data_args: LogLoaderArgs,
    split: Union[Literal['instance'], Literal['task'], Literal['task_DS']],
    model_name: str,
    batch_size: int,
    group: str,
    test_fraction: float = 0.2,
    test_max_instances: Optional[int] = 20000,
    train_max_instances: Optional[int] = None,
    model_name_short: Optional[str] = None,
    seed: int = 42,
    n_epochs: int = 6,
    max_sequence_length: int = 512,
    gradient_accumulation_steps: int = 1,
    use_wandb: bool = True,
    is_test_run: bool = False,
    include_model_in_input: bool = True,
    include_n_targets_in_input: bool = True,
    extra_training_args: Dict[str, Any] = {},
    output_dir: Optional[Union[Path, str]] = None,
    truncation_side: Union[Literal['left'], Literal['right']] = "right"
):

    if is_test_run:
        print("Running in test mode")
        n_epochs = 1
        data_args.tasks = 'pipeline-test'
        extra_training_args['eval_steps'] = 5
        extra_training_args['save_steps'] = 5
        extra_training_args['logging_steps'] = 5
        train_max_instances = 200
        test_max_instances = 200
        print(f"Tasks: {data_args.tasks}\n n_epochs: {n_epochs}")

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
        include_n_targets=include_n_targets_in_input)

    # In case of population-split, split will order by input. Make sure prepend_extra_features can not change the order.
    assert not include_n_targets_in_input or not data.model_name.nunique(
    ) > 1, "Population split not supported with include_n_targets_in_input"

    train, test = lass.datasets.split(split, data, test_fraction=test_fraction, seed=seed)

    # Sometimes we just want a little smaller datasets for speed
    if train_max_instances is not None and len(train) > train_max_instances:
        train: pd.DataFrame = train.sample(  # type: ignore
            n=train_max_instances, random_state=seed)
    if test_max_instances is not None and len(test) > test_max_instances:
        test: pd.DataFrame = test.sample(  # type: ignore
            n=test_max_instances, random_state=seed)  # type: ignore

    # Log some stats & examples
    stats = merge(analyse(train), analyse(test), 'train', 'test')

    dataset = lass.datasets.huggingfaceify_splits(train, test)
    print(dataset['train'][0])

    # Tokenize dataset
    logging.info("Starting tokenization")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tokenized_datasets: DatasetDict = lass.pipeline.tokenize(
        dataset, model_name, max_sequence_length, truncation_side=truncation_side)

    train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
    eval_dataset = tokenized_datasets["test"]

    # Setup tagging and paths
    uses_pop_data = (
        len(data_args.model_families or []) != 1 or
        len(data_args.model_sizes or []) != 1)
    model_name_short = model_name_short or model_name
    shot_str = ','.join([str(s) for s in loader.shots or []]) if data_args.shots else "all"
    bs = batch_size if gradient_accumulation_steps == 1 else f"{batch_size}*{gradient_accumulation_steps}"
    name = ""\
        + (f"test-" if is_test_run else "")\
        + (f"{model_name_short}")\
        + (f"-bs{bs}")\
        + (f"-{shot_str}sh")\
        + (f'-pop' if uses_pop_data else '')\
        + (f"-{split}-split")

    # Setup wandb
    if use_wandb:
        if isinstance(data_args.tasks, str):
            wandb_tasks = str(data_args.tasks)
        elif isinstance(data_args.tasks, list):
            if len(data_args.tasks) > 5:
                wandb_tasks = f"uknown-set-len-{len(data_args.tasks)}"
            else:
                wandb_tasks = ','.join(data_args.tasks)
        else:
            wandb_tasks = str(data_args.tasks)

        os.environ['WANDB_LOG_MODEL'] = "false"
        wandb.login()
        wandb.init(
            project="lass",
            dir=f"{output_dir or '.'}/wandb",
            group=group,
            # group=group or f"{split}-split{'pop-' if uses_pop_data else ''}",
            name=name,
            mode="disabled" if is_test_run else "online",
            tags=[
                f"split:{split}-split",
                f"assr:{model_name_short}",
                f"tasks:{wandb_tasks}",
                f"pop:{'yes' if uses_pop_data else 'no'}",
                f"shots:{shot_str}",
            ]
        )

        wandb.config.seed = seed
        wandb.config.is_test_run = is_test_run
        wandb.config.stats = stats
        wandb.config.include_model_in_input = include_model_in_input
        wandb.config.include_n_targets_in_input = include_n_targets_in_input
        wandb.config.data = {
            'query_types': ",".join(data_args.query_types or []),
            'tasks': wandb_tasks,
            'test_fraction': test_fraction,
            'split': split,
            'shots': shot_str,
            'pop_model_family': data_args.model_families,
            'pop_model_size': data_args.model_sizes,
        }
        wandb.config.extra_training_args = extra_training_args

    # Setup trainer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if model_name == "gpt2":
        model.config.pad_token_id = model.config.eos_token_id

    default_args: Dict[str, Any] = {
        "output_dir": f"{output_dir or '.'}/{name}-{datetime.now().strftime('%m%d%H%M')}",
        "optim": "adamw_torch",
        "evaluation_strategy": "steps",
        "report_to": "wandb" if wandb else "none",
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": n_epochs,
        # This combination saves models immediately, but only keeps the best and the last.
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "seed": seed,
    }
    training_args = TrainingArguments(**(default_args | extra_training_args))

    metrics = ["accuracy", "precision", "recall", "f1",
               "roc_auc", "brier_score", "balanced_accuracy"]
    metrics_assessor = lass.metrics.hf.get_metric_computer(metrics)

    # Add baseline metrics as well so we can merge the plots in wandb
    labels = test['correct']
    dist_baseline = lass.metrics.baseline.baseline(test)
    get_baseline = lass.metrics.hf.get_baseline_metrics
    compute_metrics_plus = lass.metrics.hf.join_metrics(
        metrics_assessor,
        get_baseline(labels, metrics, test['conf_normalized'], prefix="conf_normalized_"),
        get_baseline(labels, metrics, test['conf_absolute'], prefix="conf_absolute_"),
        get_baseline(labels, metrics, dist_baseline, prefix="conf_distribution_"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        compute_metrics=compute_metrics_plus,
    )

    trainer.train()

    if use_wandb:
        wandb.finish()

    return model
