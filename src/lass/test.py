import os
import logging
from typing import *
from pprint import pprint


from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer
from transformers.models.auto.tokenization_auto import AutoTokenizer

import lass.datasets
import lass.metrics
from lass.metrics.baseline import analyse, merge
from lass.log_handling import LogLoader, LoaderArgs


def test(
    data_args: LoaderArgs,
    split: Union[Literal['instance'], Literal['task'], Literal['task_DS']],
    model_loc: str,
    model_name: str,
    test_fraction: float = 0.2,
    seed: int = 42,
    max_sequence_length: int = 512,
    include_model_in_input: bool = True,
    include_n_targets_in_input: bool = True,
    # is_test_run: bool = False,
    # model_name_short: Optional[str] = None,
    # output_dir: Optional[Union[Path, str]] = None
):
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
    data['n_targets'] = data['targets'].map(lambda x: len(x))
    logging.info("Loaded data.")

    # Prepending of extra info for assessor to input
    uses_pop_data = (
        len(data_args.model_families or []) != 1 or
        len(data_args.model_sizes or []) != 1)

    model_prepender = lambda r: ""
    if include_model_in_input and uses_pop_data:
        model_prepender = lambda r: f"FAM: {r['model_family']} SIZE:{r['model_size']} "

    n_targets_prepender = lambda r: ""
    if include_n_targets_in_input:
        n_targets_prepender = lambda r: f"N_TARGETS: {r['n_targets']} "

    prepender = lambda r: f"{model_prepender(r)} {n_targets_prepender(r)} {r.input}"
    data['input'] = data.apply(prepender, axis=1)

    # Split into train and test
    if split == 'instance':
        train, test = lass.datasets.split_instance_level(data, seed, test_fraction)
    elif split == 'task':
        train, test = lass.datasets.split_task_level(data, seed, test_fraction)
    elif split == 'task_DS':
        train, test = lass.datasets.split_task_level_distribution_shift(data, seed)

    # Log some stats & examples
    stats = merge(analyse(train), analyse(test), 'train', 'test')
    pprint(stats)
    print(train.head(1))

    # Huggingfaceify
    dataset = lass.datasets.huggingfaceify_splits(train, test)
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

    train_dataset = tokenized_datasets["train"].shuffle(seed=seed)
    eval_dataset = tokenized_datasets["test"]
    len(train_dataset), len(eval_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(model_loc, num_labels=2)

    compute_metrics = lass.metrics.hf.get_metric_computer([
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "balanced_accuracy",
    ])

    # Dummy Trainer for easy batched predictions
    dummy_trainer = Trainer(model=model, compute_metrics=compute_metrics)

    logits, labels, metrics = dummy_trainer.predict(tokenized_datasets['test'])  # type: ignore

    return {
        'data': data,
        'train': train,
        'test': test,
        'logits': logits,
        'labels': labels,
        'metrics': metrics,
    }
