from asyncio import tasks
import os
from typing import *
from pathlib import Path
from dataclasses import dataclass
from pprint import pprint

import numpy as np
import torch
import wandb


from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.models.auto.tokenization_auto import AutoTokenizer

import lass.datasets
import lass.metrics
from lass.datasets import analyse, merge
from lass.log_handling import LogLoader, QueryType, QueryFunction


@dataclass
class DataArgs():
    logdir: Union[str, Path]
    tasks: Optional[List[str]]
    model_families: Optional[List[str]]
    model_sizes: Optional[List[str]]
    shots: Optional[List[int]]
    query_types: Optional[List[QueryType]]
    query_functions: Optional[List[QueryFunction]]
    exclude_faulty_tasks: bool = True
    include_unknown_shots: bool = False


def train(
    data_args: DataArgs,
    split: Union[Literal['instance'], Literal['task'], Literal['task_DS']],
    model_name: str,
    batch_size: int,
    test_fraction: float = 0.2,
    gpus: Optional[List[int]] = None,
    model_name_short: Optional[str] = None,
    seed: int = 42,
    n_epochs: int = 6,
    use_wandb: bool = True,
    is_test_run: bool = False,
    extra_training_args: Dict[str, Any] = {}
):
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

    # Split into train and test
    if split == 'instance':
        train, test = lass.datasets.split_instance_level(loader, seed, test_fraction)
    elif split == 'task':
        train, test = lass.datasets.split_task_level(loader, seed, test_fraction)
    elif split == 'task_DS':
        data = lass.datasets.to_dataframe(loader)
        accs = (data
                .groupby('task', as_index=False).agg(acc=('correct', 'mean'))  # type: ignore
                .sort_values('acc', ascending=False))
        n_train_tasks = int(len(accs['task']) * (1 - test_fraction))
        train_tasks, test_tasks = np.split(accs['task'], [n_train_tasks])
        train = data[data['task'].isin(train_tasks)]
        test = data[data['task'].isin(test_tasks)]

    # Log some stats & examples
    stats = merge(analyse(train), analyse(test), 'train', 'test')
    pprint(stats)
    print(train.head(1))

    # Set GPUs to be used
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

    # Huggingfaceify
    dataset = lass.datasets.huggingfaceify(train, test)
    print(dataset['train'][0])

    # Tokenize dataset
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="np")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42)  # .select(range(50))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42)  # .select(range(50))
    len(train_dataset), len(eval_dataset)

    # Setup tagging and paths
    is_single_system = (
        len(data_args.model_families or []) == 1 and
        len(data_args.model_sizes or []) == 1)
    shot_str = {','.join([str(s) for s in loader.shots or []])} if data_args.shots else "all"
    name = f"{model_name_short}-bs{batch_size}-{shot_str}sh-{split}-split"
    if not model_name_short:
        model_name_short = model_name

    # Setup wandb
    if use_wandb:
        os.environ['WANDB_LOG_MODEL'] = "true"
        wandb.login()
        wandb.init(
            project="lass",
            group=f"{split}-split",
            name=name,
            tags=[
                f"split:{split}-split",
                f"assr:{model_name_short}",
                f"tasks:{data_args.tasks}"
                f"pop:{'single_system' if is_single_system else 'multi_system'}",
                f"shots:{shot_str}"
            ]
        )

        wandb.config.seed = seed
        wandb.config.is_test_run = is_test_run
        wandb.config.data_query_types = ",".join(data_args.query_types or [])
        wandb.config.data_tasks = ",".join(data_args.tasks or [])
        wandb.config.data_test_fraction = test_fraction
        wandb.config.data.split = split
        wandb.config.data.shots = shot_str
        wandb.config.pop_model_family = ",".join(data_args.model_families or [])
        wandb.config.pop_model_size = ",".join(data_args.model_sizes or [])

    # Setup trainer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="bert-bs8-0sh-task-split-wd2-warmup",
        optim="adamw_torch",  # type: ignore
        evaluation_strategy="steps",  # type: ignore
        report_to="wandb" if use_wandb else None,  # type: ignore
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
