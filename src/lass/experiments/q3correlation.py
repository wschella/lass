import argparse
from copy import deepcopy
import logging
from dataclasses import dataclass

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
    We only really need to train deberta's as subject models.
    The task-specific assessors model results can come from Q4.
    """
    artifacts = Path("./artifacts")

    config = cfg.Config(
        seed=args.seed,
        is_test_run=args.is_test_run,
        data_spec=LogLoaderArgs(
            logdir="artifacts/logs",
            tasks="paper-full",
            model_families=["BIG-G T=0"],
            # Should not matter, since only use the prompt here.
            model_sizes=["128b"],
            shots=[3] if args.shots is None else args.shots,
            # No generative tasks, deberta can't handle it.
            query_types=["multiple_choice"],
        ),
        model="microsoft/deberta-v3-small",
        split_type="instance",
        test_fraction=0.2,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        filter_bad_tasks=False,
        # Reduce to batch size 1 due to data collation and padding issues
        hypers=cfg.HYPER_SMALL_DATA.reduce_mem(1).with_fields(n_epochs=args.epochs),
        log_info=cfg.LogInfo(
            output_dir=str(artifacts / "assessors" / "q3correlation"),
            model_alias="deberta-small",
            log_group="q3correlation" if not args.is_test_run else "pipeline-test",
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

    tasks: list[str] = sorted(list(data["task"].unique()))
    print(tasks)
    models = {}

    # Just load existing models
    if args.test_with:
        # We can't use specific checkpoints in this case, so just pick the latest
        model_output_dir = args.test_with
        model_id_timed = model_output_dir.name
        for task in tasks:
            models[task] = shared.earliest_checkpoint(model_output_dir / task)
    # Actually train a new model
    else:
        model_id, model_id_timed = shared.make_model_id(config)
        model_output_dir = Path(config.log_info.output_dir) / model_id_timed
        shared.save_config(config, model_output_dir)

        faulty_tasks = {}

        for task in tasks:
            print(f"Training on task {task}")
            # Still giving OOMs, can't make it much smaller
            if task in ["dyck_languages", "intersect_geometry"]:
                faulty_tasks[task] = "OOM"
                continue

            cfg_task = deepcopy(config)
            cfg_task.data_spec.tasks = [task]
            cfg_task.log_info.use_wandb = (  # Only log the last task in wandb to avoid spamming
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
            try:
                _model = lass.train.train(
                    train_task,
                    test_task,
                    cfg_task.model,
                    hypers=config.hypers,
                    log_info=cfg_task.log_info,
                    config=cfg_task,
                    is_test_run=args.is_test_run,
                    original_task=True,
                )
                models[task] = shared.earliest_checkpoint(model_task_output_dir)
            except AssertionError as e:
                if "Task has variable number of targets. Unsupported." in str(e):
                    faulty_tasks[task] = "Variable number of targets"
                else:
                    raise e

        shared.dump_as_json(faulty_tasks, model_output_dir / "faulty_tasks.json")

    # Copy config to CSV output dir
    csv_output_dir = (
        artifacts / "csv-results-new" / config.log_info.log_group / model_id_timed
    )
    shared.save_config(config, csv_output_dir)

    # Gather test predictions
    results = {}
    print(tasks)
    for task in tasks:
        if task in faulty_tasks:
            continue
        model = models[task]
        test_task = test[test["task"] == task]
        results[task] = lass.test.test(
            test_task, model, config.model, original_task=True
        )
    shared.dump_results_per_task(results, csv_output_dir)

    print({task: results[task].metrics for task in tasks if task not in faulty_tasks})
    print("Done!")


if __name__ == "__main__":
    main()
