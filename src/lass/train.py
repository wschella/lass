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
from transformers.models.auto.tokenization_auto import AutoTokenizer

import lass.datasets
import lass.metrics
from lass.datasets import analyse, merge
from lass.log_handling import LogLoader, QueryType, QueryFunction, TaskList


@dataclass
class DataArgs():
    logdir: Union[str, Path]
    tasks: TaskList
    model_families: Optional[List[str]] = None
    model_sizes: Optional[List[str]] = None
    shots: Optional[List[int]] = None
    query_types: Optional[List[QueryType]] = None
    query_functions: Optional[List[QueryFunction]] = None
    exclude_faulty_tasks: bool = True
    include_unknown_shots: bool = False


def train(
    data_args: DataArgs,
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
    loader = LogLoader(
        logdir=data_args.logdir,
        tasks=data_args.tasks,
        model_families=data_args.model_families,
        model_sizes=data_args.model_sizes,
        shots=data_args.shots,
        query_types=data_args.query_types,
        exclude_faulty_tasks=data_args.exclude_faulty_tasks,
        include_unknown_shots=data_args.include_unknown_shots
    )
    data = lass.datasets.to_dataframe(loader)
    logging.info("Loaded data.")

    # Population related data wrangling
    uses_pop_data = (
        len(data_args.model_families or []) != 1 or
        len(data_args.model_sizes or []) != 1)
    if uses_pop_data:
        # First population approach: just add the model name prepended to the text
        prepender = lambda r: f"{r['model_family']} {r['model_name']}. {r['input']}"
        data['input'] = data.apply(prepender, axis=1)

    # Split into train and test
    if split == 'instance':
        train, test = lass.datasets.split_instance_level(data, seed, test_fraction)
    elif split == 'task':
        train, test = lass.datasets.split_task_level(data, seed, test_fraction)
    elif split == 'task_DS':
        train, test = lass.datasets.split_task_level_distribution_shift(data, seed)

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

    # Huggingfaceify
    dataset = lass.datasets.huggingfaceify(train, test)
    print(dataset['train'][0])

    # Tokenize dataset
    logging.info("Starting tokenization")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="np"
        )
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)  # .select(range(50))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)  # .select(range(50))
    len(train_dataset), len(eval_dataset)

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
    ])  # + ["wandb_conf_matrix"] if use_wandb else [])

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
