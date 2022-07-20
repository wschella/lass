import os
import logging
from typing import *
from pprint import pprint


from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer

import lass.pipeline
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

    # Log some stats & examples
    stats = merge(analyse(train), analyse(test), 'train', 'test')
    pprint(stats)
    print(train.head(1))

    dataset = lass.datasets.huggingfaceify_splits(train, test)
    print(dataset['train'][0])

    # Tokenize dataset
    logging.info("Starting tokenization")
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tokenized_datasets = lass.pipeline.tokenize(dataset, model_name, max_sequence_length)

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
