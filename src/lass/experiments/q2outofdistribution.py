import argparse
import logging
from dataclasses import dataclass, replace

# autopep8: off
import os
from pathlib import Path
import shutil
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
    parser.add_argument(
        "--test-run", action="store_true", default=False, dest="is_test_run"
    )
    parser.add_argument(
        "--test-with",
        type=Path,
    )
    parser.add_argument("--shots", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))
    run(args)


def run(args: Args):
    artifacts = Path("./artifacts")

    config = cfg.Config(
        seed=args.seed,
        is_test_run=args.is_test_run,
        data_spec=LogLoaderArgs(
            logdir="artifacts/logs",
            tasks="paper-full",
            # tasks=[
            #     "cause_and_effect",  # An MPC task
            #     "mult_data_wrangling",  # A generative task with exact_string_match
            # ],
            model_families=["BIG-G T=0"],
            model_sizes=["128b"],
            shots=[3] if args.shots is None else args.shots,
            query_types=["multiple_choice", "scoring", "generative"],
        ),
        model="microsoft/deberta-v3-base",
        split_type="task",
        test_fraction=0.2,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        filter_bad_tasks=True,
        hypers=cfg.HYPER_DEFAULT_REDUCED_MEM
        if args.epochs is None
        else replace(cfg.HYPER_DEFAULT_REDUCED_MEM, epochs=args.epochs),
        log_info=cfg.LogInfo(
            output_dir=str(artifacts / "assessors" / "q2outofdistribution"),
            model_alias="deberta-base",
            log_group="q2outofdistribution"
            if not args.is_test_run
            else "pipeline-test",
            use_wandb=True,
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

    model_id, model_id_timed = shared.make_model_id(config)

    # Just load existing model
    if args.test_with:
        model = args.test_with
    # Actually train a new model
    else:
        shared.save_config(config, Path(config.log_info.output_dir) / model_id_timed)
        model = lass.train.train(
            train,
            test,
            config.model,
            hypers=config.hypers,
            log_info=config.log_info,
            config=config,
            is_test_run=args.is_test_run,
        )

    # Copy config to CSV output dir
    csv_output_dir = (
        artifacts / "csv-results-new" / config.log_info.log_group / model_id_timed
    )
    shared.save_config(config, csv_output_dir)

    # Gather test predictions
    results = lass.test.test(test, model, config.model)
    shared.dump_results(results, csv_output_dir)

    # Save predictions and metrics per task
    results_per_task = lass.test.test_per_task(test, model, config.model)
    shared.dump_results_per_task(results_per_task, csv_output_dir)

    # Copy this dir to "latest" dir as well
    latest_dir = csv_output_dir.parent / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(csv_output_dir, latest_dir)

    print(results.metrics)


if __name__ == "__main__":
    main()
