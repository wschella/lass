import os
import logging
from pathlib import Path
from typing import *
from pprint import pprint


from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.trainer import Trainer

import lass.pipeline
import lass.datasets
import lass.metrics
import lass.metrics.baseline
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
    include_model_in_input: bool = False,
    include_n_targets_in_input: bool = False,
    # is_test_run: bool = False,
    # model_name_short: Optional[str] = None,
    # output_dir: Optional[Union[Path, str]] = None
):
    assert Path(model_loc).exists()

    logging.info("Starting data loading")
    loader = LogLoader.from_args(data_args)
    data = lass.datasets.to_dataframe(loader)
    logging.info("Loaded data.")

    data = lass.pipeline.binarize(data)
    data = lass.pipeline.augment(data)
    data = lass.pipeline.clean(data)

    data = lass.pipeline.prepend_extra_features(
        data,
        include_model=include_model_in_input,
        include_n_targets=include_n_targets_in_input)

    train, test = lass.datasets.split(split, data, test_fraction=test_fraction, seed=seed)

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

    # Dummy Trainer for easy batched predictions
    dummy_trainer = Trainer(model=model, compute_metrics=compute_metrics_plus)

    logits, labels, metrics = dummy_trainer.predict(tokenized_datasets['test'])  # type: ignore

    return {
        'data': data,
        'train': train,
        'test': test,
        'logits': logits,
        'labels': labels,
        'metrics': metrics,
    }
