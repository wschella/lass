import argparse
from copy import deepcopy
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
    """
    We only need to train the task specific assessors here.
    The multi-task model will be trained in Q1
    """
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
        model="microsoft/deberta-v3-base",
        split_type="instance",
        test_fraction=0.2,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        filter_bad_tasks=True,
        hypers=cfg.HYPER_DEFAULT_REDUCED_MEM
        if args.epochs is None
        else replace(cfg.HYPER_DEFAULT_REDUCED_MEM, epochs=args.epochs),
        log_info=cfg.LogInfo(
            output_dir=str(artifacts / "assessors" / "q4multitask"),
            model_alias="deberta-base",
            log_group="q4multitask" if not args.is_test_run else "pipeline-test",
            use_wandb=False if args.is_test_run else True,
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
    print(sorted(list(data["task"].unique())))
    train, test = lass.data.splitting.split(
        data, config.split_type, config.test_fraction, config.seed
    )

    tasks: list[str] = sorted(list(data["task"].unique()))
    print(tasks)
    models = {}

    # Just load existing models
    if args.test_with:
        # We can't use specific checkpoints in this case, so just pick the latest
        model_output_dir = args.test_with
        model_id_timed = model_output_dir.name
        for task in tasks:
            models[task] = shared.latest_checkpoint(model_output_dir / task)
    # Actually train a new model
    else:
        model_id, model_id_timed = shared.make_model_id(config)
        model_output_dir = Path(config.log_info.output_dir) / model_id_timed
        shared.save_config(config, model_output_dir)

        for task in tasks:
            cfg_task = deepcopy(config)
            cfg_task.data_spec.tasks = [task]
            cfg_task.log_info.use_wandb = (
                False if task != tasks[-1] else config.log_info.use_wandb
            )

            # Restrict data to the task
            train_task = train[train["task"] == task]
            test_task = test[test["task"] == task]

            # Have a separate output dir for each task
            model_task_output_dir = model_output_dir / task
            cfg_task.log_info.output_dir = str(model_task_output_dir)
            shared.save_config(config, model_task_output_dir)

            # We can't keep the model in memory, otherwise we run out.
            _model = lass.train.train(
                train_task,
                test_task,
                cfg_task.model,
                hypers=config.hypers,
                log_info=cfg_task.log_info,
                config=cfg_task,
                is_test_run=args.is_test_run,
            )
            models[task] = shared.latest_checkpoint(model_task_output_dir)

    # Copy config to CSV output dir
    csv_output_dir = (
        artifacts / "csv-results-new" / config.log_info.log_group / model_id_timed
    )
    shared.save_config(config, csv_output_dir)

    # Gather test predictions
    results = {}
    print(tasks)
    for task in tasks:
        model = models[task]
        test_task = test[test["task"] == task]
        results[task] = lass.test.test(test_task, model, config.model)
    shared.dump_results_per_task(results, csv_output_dir)

    print({task: results[task].metrics for task in tasks})


if __name__ == "__main__":
    main()
