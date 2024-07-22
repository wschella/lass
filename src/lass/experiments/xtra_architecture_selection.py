# Note: load and filter all data at the same time, so all shots have the same data (since  task filtering is performance based)
import argparse
from datetime import datetime
import logging
import platform
from dataclasses import dataclass, replace

# autopep8: off
import os
from pathlib import Path
from typing import Optional

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
    model: str
    is_test_run: bool
    test_with: Path
    shots: Optional[list[int]]
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
    parser.add_argument("--model", type=str)

    parser.add_argument(
        "--test-run", action="store_true", default=False, dest="is_test_run"
    )
    parser.add_argument(
        "--test-with",
        type=Path,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--shots", type=int, nargs="+", default=None)
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))
    run(args)


def run(args: Args):
    """
    This script needs to be run for every model.
    We organize it like this so it can be more easily parallelized on an HPC,
    and the rest of the code is more similar to the other experiments.
    """
    print(
        f"Running {args.model} on {platform.node()} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    artifacts = Path("./artifacts")
    config = cfg.Config(
        seed=args.seed,
        is_test_run=args.is_test_run,
        data_spec=LogLoaderArgs(
            logdir="artifacts/logs",
            tasks="paper-full",
            model_families=["BIG-G T=0"],
            model_sizes=["128b"],
            shots=[3] if args.shots is None else args.shots,
            query_types=["multiple_choice", "scoring", "generative"],
        ),
        model=args.model,
        split_type="instance",
        test_fraction=0.2,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        filter_bad_tasks=True,
        hypers=(
            cfg.HYPER_DEFAULT_REDUCED_MEM
            if args.epochs is None
            else replace(cfg.HYPER_DEFAULT_REDUCED_MEM, epochs=args.epochs)
        ),
        log_info=cfg.LogInfo(
            output_dir=str(artifacts / "assessors" / "xtra_architecture_selection"),
            model_alias=shared.get_alias(args.model),
            log_group=(
                "xtra_architecture_selection"
                if not args.is_test_run
                else "pipeline-test"
            ),
            use_wandb=True if not args.is_test_run else False,
        ),
    )

    if args.test_with:
        config = shared.load_config(args.test_with)

    shared.assert_GPU_available()

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
        model_output_dir = args.test_with.parent
    # Actually train a new model
    else:
        model_output_dir = (
            Path(config.log_info.output_dir)
            / args.time
            / str(config.log_info.model_alias)
        )
        shared.save_config(config, model_output_dir)

        model = lass.train.train(
            train,
            test,
            config.model,
            hypers=config.hypers,
            log_info=replace(config.log_info, output_dir=str(model_output_dir)),
            config=config,
            is_test_run=args.is_test_run,
        )

    # Copy config to CSV output dir
    csv_output_dir = (
        artifacts
        / "csv-results-new"
        / config.log_info.log_group
        / args.time
        / str(config.log_info.model_alias)
    )
    shared.save_config(config, csv_output_dir)

    # Gather test predictions
    results = lass.test.test(test, model, config.model)
    shared.dump_results(results, csv_output_dir)
    results_per_task = lass.test.test_per_task(test, model, config.model)
    shared.dump_results_per_task(results_per_task, csv_output_dir)

    print(results.metrics)
    print("Done!")


if __name__ == "__main__":
    main()