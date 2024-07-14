import argparse
import dataclasses
import logging
import json
from dataclasses import dataclass
from datetime import datetime

# autopep8: off
import os
from pathlib import Path
import shutil
from typing import Optional

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# autopep8: on

import lass.data
import lass.train
import lass.test
import lass.config as cfg
from lass.log_handling import LogLoaderArgs


@dataclass
class Args:
    seed: int
    is_test_run: bool
    test_with: Path
    shots: Optional[list[int]]


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
        split_type="instance",
        split_fraction=0.8,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        hypers=cfg.HYPER_DEFAULT_REDUCED_MEM,
        log_info=cfg.LogInfo(
            output_dir=str(artifacts / "assessors" / "q1indistribution"),
            model_alias="deberta-base",
            log_group="q1indistribution" if not args.is_test_run else "pipeline-test",
            use_wandb=True,
        ),
    )

    if args.test_with:
        # This would look something like
        # "./artifact/assessors/q1indistribution/deberta-base_20210901120000/checkpoint-4000"
        config_path = args.test_with.parent / "config.json"
        with open(config_path, "r") as f:
            config = cfg.Config.from_dict(json.load(f))

    data = lass.data.loading.load(config.data_spec, config.is_test_run)
    data = lass.data.wrangling.wrangle(data)
    train, test = lass.data.splitting.split(
        data, config.split_type, config.split_fraction, config.seed
    )

    # Set specific model_id
    model_id, _ = lass.train.make_model_id(config)
    model_id_timed = f"{model_id}-{datetime.now().strftime('%m%d%H%M')}"

    # Just load existing model
    if args.test_with:
        model = args.test_with

    # Actually train
    else:
        # Save config
        output_dir = Path(config.log_info.output_dir) / model_id_timed
        output_dir.mkdir(parents=True, exist_ok=True)
        config.log_info.output_dir = str(output_dir)
        with open(output_dir / "config.json", "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)

        # Train
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
    csv_output_dir = artifacts / "csv-results-new" / "q1indistribution" / model_id_timed
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_output_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    # Gather test predictions
    results = lass.test.test(test, model, config.model)

    # Save predictions and metrics
    dumpable = lambda r: {"logits": r.logits.tolist(), "labels": r.labels.tolist()}
    with open(csv_output_dir / "metrics.json", "w") as f:
        json.dump(results.metrics, f, indent=2)
    with open(csv_output_dir / "predictions.json", "w") as f:
        json.dump(dumpable(results), f)

    # Save predictions and metrics per task
    results_per_task = lass.test.test_per_task(test, model, config.model)
    with open(csv_output_dir / "metrics_per_task.json", "w") as f:
        metrics = {k: v.metrics for k, v in results_per_task.items()}
        json.dump(metrics, f, indent=2)
    with open(csv_output_dir / "predictions_per_task.json", "w") as f:
        predictions = {k: dumpable(v) for k, v in results_per_task.items()}
        json.dump(predictions, f)

    # Copy this dir to "latest" dir as well
    latest_dir = csv_output_dir.parent / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(csv_output_dir, latest_dir)

    print(results.metrics)


if __name__ == "__main__":
    main()
