import argparse
import logging
from dataclasses import dataclass

# autopep8: off
import os
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    test_with: Path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(module)s/%(funcName)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_with",
        type=Path,
    )
    args_raw = parser.parse_args()
    args = Args(**vars(args_raw))
    run(args)


def run(args: Args):
    is_test_run = True
    config = cfg.Config(
        seed=args.seed,
        is_test_run=is_test_run,
        data_spec=LogLoaderArgs(
            logdir="artifacts/logs",
            tasks="paper-full",
            # tasks=[
            #     "cause_and_effect",  # An MPC task
            #     "mult_data_wrangling",  # A generative task with exact_string_match
            # ],
            model_families=["BIG-G T=0"],
            model_sizes=["128b"],
            shots=[3],
            query_types=["multiple_choice", "scoring", "generative"],
        ),
        model="microsoft/deberta-v3-base",
        split_type="instance",
        split_fraction=0.8,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        hypers=cfg.HYPER_DEFAULT_REDUCED_MEM,
        log_info=cfg.LogInfo(
            output_dir="./artifacts/assessors/q1indistribution/",
            model_alias="deberta-base",
            log_group="q1indistribution" if not is_test_run else "pipeline-test",
            use_wandb=False,
        ),
    )

    data = lass.data.loading.load(config.data_spec, config.is_test_run)
    data = lass.data.wrangling.wrangle(data)
    train, test = lass.data.splitting.split(
        data, config.split_type, config.split_fraction, config.seed
    )

    if not args.test_with:
        model = lass.train.train(
            train,
            test,
            config.model,
            hypers=config.hypers,
            log_info=config.log_info,
            config=config,
            is_test_run=is_test_run,
        )
    else:
        model = args.test_with

    # TODO: Print relevant CSV, do plotting?
    # TODO: Make sure model_name actually matches model?
    results = lass.test.test(test, model, config.model)
    results_per_task = lass.test.test_per_task(test, model, config.model)
    print(results)


if __name__ == "__main__":
    main()
