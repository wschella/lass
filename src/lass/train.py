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
import lass.metrics
import lass.pipeline
from lass.metrics.baseline import analyse, merge
from lass.log_handling import LogLoader, LoaderArgs


def train(
    data_args: LoaderArgs,
    split: Union[Literal['instance'], Literal['task'], Literal['task_DS']],
    model_name: str,
    batch_size: int,
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
    output_dir: Optional[Union[Path, str]] = None
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
    loader = LogLoader.from_args(data_args)
    data = lass.datasets.to_dataframe(loader)
    data = lass.pipeline.augment(data)
    logging.info("Loaded data.")

    # Prepending of extra info for assessor to input
    uses_pop_data = (
        len(data_args.model_families or []) != 1 or
        len(data_args.model_sizes or []) != 1)

    data = lass.pipeline.prepend_extra_features(
        data,
        include_model=include_model_in_input,
        include_n_targets=include_n_targets_in_input)

    # TODO: We should binarize before the split (before augmentation)
    train, test = lass.datasets.split(split, data, test_fraction=test_fraction, seed=seed)
    train, test = lass.pipeline.binarize(train), lass.pipeline.binarize(test)

    # Sometimes we just want a little smaller datasets for speed
    if train_max_instances is not None and len(train) > train_max_instances:
        train: pd.DataFrame = train.sample(  # type: ignore
            n=train_max_instances, random_state=seed)
    if test_max_instances is not None and len(test) > test_max_instances:
        test: pd.DataFrame = test.sample(  # type: ignore
            n=test_max_instances, random_state=seed)  # type: ignore

    # Log some stats & examples
    stats = merge(analyse(train), analyse(test), 'train', 'test')
    pprint(stats)
    print(train.head(1))

    dataset = lass.datasets.huggingfaceify_splits(train, test)
    print(dataset['train'][0])

    # Tokenize dataset
    logging.info("Starting tokenization")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tokenized_datasets: DatasetDict = lass.pipeline.tokenize(
        dataset, model_name, max_sequence_length)

    train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
    eval_dataset = tokenized_datasets["test"]

    # Setup tagging and paths
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
        os.environ['WANDB_LOG_MODEL'] = "false"
        wandb.login()
        wandb.init(
            project="lass",
            dir=f"{output_dir or '.'}/wandb",
            group=f"{split}-split{'pop-' if uses_pop_data else ''}",
            name=name,
            mode="disabled" if is_test_run else "online",
            tags=[
                f"split:{split}-split",
                f"assr:{model_name_short}",
                f"tasks:{data_args.tasks}",
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
            'tasks': data_args.tasks,
            'test_fraction': test_fraction,
            'split': split,
            'shots': shot_str,
            'pop_model_family': data_args.model_families,
            'pop_model_size': data_args.model_sizes,
        }
        wandb.config.extra_training_args = extra_training_args

    # Setup trainer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f"{output_dir or '.'}/{name}-{datetime.now().strftime('%m%d%H%M')}",
        optim="adamw_torch",  # type: ignore
        evaluation_strategy="steps",  # type: ignore
        report_to="wandb" if use_wandb else "none",  # type: ignore
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_total_limit=1,
        load_best_model_at_end=True,
        num_train_epochs=n_epochs,
        **extra_training_args
    )

    compute_metrics = lass.metrics.hf.get_metric_computer([
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "balanced_accuracy",
    ])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if use_wandb:
        wandb.finish()
