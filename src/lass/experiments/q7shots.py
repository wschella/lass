# Note: load and filter all data at the same time, so all shots have the same data (since  task filtering is performance based)
import argparse
from datetime import datetime
import logging
import platform
from dataclasses import dataclass, replace

# autopep8: off
import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# autopep8: on

import lass.data
import lass.train
import lass.test
import lass.config as cfg
import lass.experiments.shared as shared
from lass.log_handling import LogLoaderArgs


@dataclass
class Args:
    seed: int
    time: str
    truncation_side: Literal["left", "right"]
    shots: int
    is_test_run: bool
    test_with: Path
    epochs: Optional[int]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(module)s/%(funcName)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time", type=str)
    parser.add_argument(
        "--truncation-side", type=str, choices=["left", "right"], required=True
    )
    parser.add_argument("--shots", type=int, default=None)

    parser.add_argument(
        "--test-run", action="store_true", default=False, dest="is_test_run"
    )
    parser.add_argument(
        "--test-with",
        type=Path,
    )
    parser.add_argument("--epochs", type=int, default=None)
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))
    run(args)


def run(args: Args):
    """
    This script needs to be run for every (truncation side, shot) pair.
    We organize it like this so it can be more easily parallelized on an HPC,
    and the rest of the code is more similar to the other experiments.
    """
    print(
        f"Running {args.truncation_side} for {args.shots} on {platform.node()} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logging.warning(
        "This experiment has different data depending on each shot, not all the same tasks are present, and performance will differ in any case."
    )

    assert args.shots in [0, 1, 2, 3]
    assert args.truncation_side in ["left", "right"]

    artifacts = Path("./artifacts")
    config = cfg.Config(
        seed=args.seed,
        is_test_run=args.is_test_run,
        data_spec=LogLoaderArgs(
            logdir="artifacts/logs",
            tasks="paper-full",
            model_families=["BIG-G T=0"],
            model_sizes=["128b"],
            shots=[args.shots],
            query_types=["multiple_choice", "scoring", "generative"],
        ),
        model=f"microsoft/deberta-v3-base",
        split_type="instance",
        test_fraction=0.2,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        filter_bad_tasks=True,
        hypers=cfg.HYPER_DEFAULT.reduce_mem(16).with_fields(n_epochs=args.epochs),
        log_info=cfg.LogInfo(
            output_dir=str(artifacts / "assessors" / "q7shots"),
            model_alias="deberta-base",
            log_group="q7shots" if not args.is_test_run else "pipeline-test",
            use_wandb=True if not args.is_test_run else False,
        ),
        extra={"truncation_side": args.truncation_side},
    )

    if args.test_with:
        config = shared.load_config(args.test_with)

    # shared.assert_GPU_available()

    data = lass.data.loading.load(config.data_spec, config.is_test_run)
    data = lass.data.wrangling.wrangle(
        data,
        include_model_in_input=config.include_model_in_input,
        include_n_targets_in_input=config.include_n_targets_in_input,
        filter_bad_tasks=config.filter_bad_tasks,
    )
    train, test = lass.data.splitting.split(
        data, config.split_type, config.test_fraction, config.seed
    )

    # Just load existing model
    if args.test_with:
        model_id_timed = args.test_with.name
        model_output_dir = args.test_with / args.truncation_side / str(args.shots)
        model_output_dir = shared.earliest_checkpoint(model_output_dir)
    # Actually train a new model
    else:
        # Can't use default model_id_timed, as this script will be called multiple times, and we'd like files to be grouped
        model_id, _model_id_timed = shared.make_model_id(config)
        model_id_timed = f"{model_id}-{args.time}"
        model_output_dir = (
            Path(config.log_info.output_dir)
            / model_id_timed
            / args.truncation_side
            / str(args.shots)
        )
        shared.save_config(config, model_output_dir)

        model = lass.train.train(
            train,
            test,
            config.model,
            hypers=config.hypers,
            log_info=replace(config.log_info, output_dir=str(model_output_dir)),
            config=config,
            truncation_side=args.truncation_side,
            is_test_run=args.is_test_run,
        )

    # Copy config to CSV output dir
    csv_output_dir = (
        artifacts
        / "csv-results-new"
        / config.log_info.log_group
        / model_id_timed
        / args.truncation_side
        / str(args.shots)
    )
    shared.save_config(config, csv_output_dir)

    # Gather test predictions
    results = lass.test.test(test, model, config.model)
    results.metrics["n_long_sequences"] = percentage_long_sequences(config, data, 512)
    shared.dump_results(results, csv_output_dir)
    results_per_task = lass.test.test_per_task(test, model, config.model)
    shared.dump_results_per_task(results_per_task, csv_output_dir)

    print(results.metrics)
    print("Done!")


def percentage_long_sequences(
    config: cfg.Config, data: pd.DataFrame, max_seq_length: int
) -> float:
    """
    Count the number of sequences with more than n tokens.
    """
    hfdata = lass.data.wrangling.huggingfaceify(data)
    tokenizer = lass.data.wrangling._get_tokenizer(
        config.model, truncation_side=config.extra["truncation_side"]
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=False,
            max_length=None,
            return_tensors="np",
        )

    tokenized = hfdata.map(tokenize_function, batched=True)
    lengths = [len(x) for x in tokenized["input_ids"]]
    n_long_sequences = sum(1 for x in lengths if x > max_seq_length)
    return n_long_sequences / len(lengths)


if __name__ == "__main__":
    main()
