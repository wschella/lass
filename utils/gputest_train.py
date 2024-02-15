from pathlib import Path

import logging
from lass.train import train
from lass.log_handling import LogLoaderArgs

ROOT = Path(".")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(module)s/%(funcName)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train(
        data_args=LogLoaderArgs(
            logdir=ROOT / "artifacts/logs",
            tasks=["strategyqa"],
            model_families=["BIG-G T=0"],
            model_sizes=["128b"],
            shots=[0],
            query_types=["multiple_choice"],
        ),
        group="test",
        split="instance",
        model_name="microsoft/deberta-v3-xsmall",
        model_name_short="deberta-reference-xsmall",
        batch_size=32,
        gradient_accumulation_steps=1,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        extra_training_args={
            #     "eval_steps": 2000,
            #     "save_steps": 2000,
            "warmup_steps": 3000,
            "learning_rate": 2e-5,
            #     "weight_decay": 0.01,
            #     "label_smoothing_factor": 0.25,
        },
        output_dir=ROOT / "artifacts/tmp",
        is_test_run=True,
    )


if __name__ == "__main__":
    main()
