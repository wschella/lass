import argparse
from copy import deepcopy
import logging
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
            model_families=["BIG-G T=0"],
            model_sizes=["4b", "8b", "27b", "128b"],
            shots=[3] if args.shots is None else args.shots,
            query_types=["multiple_choice", "scoring", "generative"],
        ),
        model="microsoft/deberta-v3-base",
        split_type="instance",
        test_fraction=0.2,
        include_model_in_input=True,  #! Important, such that assessors can differentiate
        include_n_targets_in_input=False,
        filter_bad_tasks=True,
        hypers=cfg.HYPER_DEFAULT.reduce_mem(16).with_fields(n_epochs=args.epochs),
        log_info=cfg.LogInfo(
            output_dir=str(artifacts / "assessors" / "q5population"),
            model_alias="deberta-base",
            log_group="q5population" if not args.is_test_run else "pipeline-test",
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

    # Note: Big assumption
    largest_model = (config.data_spec.model_sizes or ["128b"])[-1]
    train_largest = train[train.model_name == largest_model]
    test_largest = test[test.model_name == largest_model]

    # Just load existing model
    if args.test_with:
        model = args.test_with
        model_id_timed = model.parent.name
        model_finetuned = shared.earliest_checkpoint(args.test_with / "finetuned")
    # Actually train a new model
    else:
        _, model_id_timed = shared.make_model_id(config)
        model_output_dir = Path(config.log_info.output_dir) / model_id_timed
        shared.save_config(config, model_output_dir)
        model = lass.train.train(
            train,
            test,
            config.model,
            hypers=config.hypers,
            log_info=replace(config.log_info, output_dir=model_output_dir),
            config=config,
            is_test_run=args.is_test_run,
        )

        # Fine tune on largest model
        model_output_dir_finetuned = model_output_dir / "finetuned"
        config_finetuned = deepcopy(config)
        config_finetuned.hypers.n_epochs = 2
        config_finetuned.log_info.output_dir = str(model_output_dir_finetuned)
        config_finetuned.log_info.log_group = config.log_info.log_group + "-finetuned"

        model_finetuned = lass.train.train(
            train_largest,
            test_largest,
            config_finetuned.model,
            finetune=deepcopy(model),
            hypers=config_finetuned.hypers,
            log_info=config_finetuned.log_info,
            config=config_finetuned,
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
    results_per_task = lass.test.test_per_task(test, model, config.model)
    shared.dump_results_per_task(results_per_task, csv_output_dir)

    # Gather test predictions for largest subject only
    results_largest = lass.test.test(test_largest, model, config.model)
    shared.dump_results(results_largest, csv_output_dir / "largest")
    results_per_task_largest = lass.test.test_per_task(test_largest, model, config.model)  # fmt: skip
    shared.dump_results_per_task(results_per_task_largest, csv_output_dir / "largest")

    # Gather test predictions for finetuned model
    shared.save_config(config_finetuned, csv_output_dir / "finetuned")
    results_ft = lass.test.test(test_largest, model_finetuned, config.model)
    shared.dump_results(results_ft, csv_output_dir / "finetuned")
    results_per_task_ft = lass.test.test_per_task(test_largest, model_finetuned, config.model)  # fmt: skip
    shared.dump_results_per_task(results_per_task_ft, csv_output_dir / "finetuned")

    print(results_ft.metrics)
    print("Done!")


if __name__ == "__main__":
    main()
