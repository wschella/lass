import dataclasses
import os
import logging
from typing import Any, Dict, Literal, Optional, Tuple, Union

import wandb
import pandas as pd

from torch.nn.modules.module import Module
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets.dataset_dict import DatasetDict

import lass.utils
import lass.metrics
import lass.metrics.stats
import lass.data.splitting
import lass.data.wrangling
import lass.config as cfg
from lass.metrics.stats import analyse, merge


def train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    model_name: str,
    log_info: cfg.LogInfo,
    config: cfg.Config,  # Need the config for various logging purposes
    hypers: cfg.HyperParams,
    # ---------------
    is_test_run: bool = False,
    train_max_instances: Optional[int] = None,
    val_max_instances: Optional[int] = 20000,
    max_sequence_length: int = 512,
    truncation_side: Union[Literal["left"], Literal["right"]] = "right",
    finetune: Optional[Module] = None,
) -> Module:
    if is_test_run:
        print("Running in test mode")
        train_max_instances = 200
        val_max_instances = 200
        hypers.n_epochs = 1
        print(f"Tasks: {config.data_spec.tasks}\n n_epochs: {hypers.n_epochs}")

    # Sometimes we just want a little smaller datasets for speed
    if train_max_instances is not None and len(train_data) > train_max_instances:
        train_data = train_data.sample(n=train_max_instances, random_state=config.seed)
    if val_max_instances is not None and len(val_data) > val_max_instances:
        val_data = val_data.sample(n=val_max_instances, random_state=config.seed)

    # Log some stats & examples
    stats = merge(analyse(train_data), analyse(val_data), "train", "val")
    hfify = lass.data.wrangling.huggingfaceify
    dataset = DatasetDict({"train": hfify(train_data), "val": hfify(val_data)})
    # print(dataset["train"][0])

    # Tokenize dataset
    logging.info("Starting tokenization")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenized_datasets: DatasetDict = lass.data.wrangling.tokenize(
        dataset, model_name, max_sequence_length, truncation_side=truncation_side
    )

    train_dataset = tokenized_datasets["train"].shuffle(seed=config.seed)
    val_dataset = tokenized_datasets["val"]

    name, tags = make_model_id(config)

    # Setup wandb
    if isinstance(config.data_spec.tasks, list):
        if len(config.data_spec.tasks) > 5:
            wandb_tasks = f"unknown-set-len-{len(config.data_spec.tasks)}"
        else:
            wandb_tasks = ",".join(config.data_spec.tasks)
    else:
        wandb_tasks = str(config.data_spec.tasks)

    if log_info.use_wandb:
        os.environ["WANDB_LOG_MODEL"] = "false"
        wandb.login()
        wandb.init(
            mode="disabled" if is_test_run else "online",
            project="lass",
            dir=log_info.output_dir,
            group=log_info.log_group,
            name=name,
            config=dataclasses.asdict(config),
            tags=[
                f"split:{config.split_type}-split",
                f"assr:{tags['model_alias']}",
                f"tasks:{wandb_tasks}",
                f"pop:{'yes' if tags['uses_pop_data'] else 'no'}",
                f"shots:{tags['shots']}",
            ],
        )
        wandb.config.stats = stats

    # Setup trainer
    if finetune is not None:
        model = finetune
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    if model_name == "gpt2":
        model.config.pad_token_id = model.config.eos_token_id  # type: ignore

    # Make sure to do saving, logging, eval at least once
    # -> Should be at least once every epoch, to prevent only logging overfitted models
    # 500 is the default value in the Trainer
    # Should always match or undershoot the number of steps
    # Instances at the not fitting the batch might be dropped?
    total_batch_size = hypers.batch_size * hypers.gradient_accumulation_steps
    # n_opt_steps = hypers.n_epochs * (len(train_data) // total_batch_size)
    n_opt_steps = len(train_data) // total_batch_size
    x_every_steps = 500 if n_opt_steps > 500 else n_opt_steps
    x_every_steps = max(x_every_steps, 1)

    default_args: Dict[str, Any] = {
        "output_dir": log_info.output_dir,
        "optim": "adamw_torch",
        "evaluation_strategy": "steps",
        "report_to": "wandb" if log_info.use_wandb else "none",
        "per_device_train_batch_size": hypers.batch_size,
        "per_device_eval_batch_size": hypers.batch_size,
        "gradient_accumulation_steps": hypers.gradient_accumulation_steps,
        "num_train_epochs": hypers.n_epochs,
        "warmup_steps": hypers.warmup_steps,
        "learning_rate": hypers.learning_rate,
        # This combination saves models immediately, but only keeps the best and the last.
        "load_best_model_at_end": True,
        "save_total_limit": 1,
        "save_steps": x_every_steps,
        "eval_steps": x_every_steps,
        "logging_steps": x_every_steps,
        "seed": config.seed,
    }
    training_args = TrainingArguments(**(default_args | hypers.extra))

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "balanced_accuracy",
    ]

    baselines = lass.metrics.baseline.get_baselines(val_data, metrics)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=val_dataset,  # type: ignore
        # Log baseline metrics use them as a reference in wandb
        compute_metrics=lambda predictions: (
            lass.metrics.compute_metrics_trainer(predictions, metrics) | baselines
        ),
    )

    trainer.train()

    if log_info.use_wandb:
        wandb.finish()

    return model


def make_model_id(config) -> Tuple[str, Dict[str, Any]]:
    uses_pop_data = (
        len(config.data_spec.model_families or []) != 1
        or len(config.data_spec.model_sizes or []) != 1
    )
    model_alias = config.log_info.model_alias or config.model_name
    shot_str = (
        ",".join([str(s) for s in config.data_spec.shots])
        if config.data_spec.shots
        else "all"
    )
    batch_size = (
        f"{config.hypers.batch_size * config.hypers.gradient_accumulation_steps}"
    )

    tags = [
        f"test" if config.is_test_run else None,
        f"{model_alias}",
        f"bs{batch_size}",
        f"{shot_str}sh",
        f"pop" if uses_pop_data else None,
        f"{config.split_type}-split",
    ]
    name = "_".join([t for t in tags if t is not None])
    return (
        name,
        {
            "uses_pop_data": uses_pop_data,
            "batch_size": batch_size,
            "model_alias": model_alias,
            "shots": shot_str,
        },
    )
