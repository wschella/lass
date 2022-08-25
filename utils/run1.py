import logging

# autopep8: off
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from lass.train import train
from lass.log_handling import LoaderArgs
# autopep8: on


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(module)s/%(funcName)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    train(
        data_args=LoaderArgs(
            logdir="artifacts/logs",
            tasks="paper-full",
            model_families=["BIG-G T=0"],
            model_sizes=["128b"],
            shots=[0],
            query_types=["multiple_choice"],
        ),
        group="test",
        split="instance",
        # model_name="albert-base-v2",
        # model_name_short="albert",
        # batch_size=32,
        model_name="microsoft/deberta-v3-base",
        model_name_short="deberta-cleaned-data",
        batch_size=16,
        gradient_accumulation_steps=2,
        include_model_in_input=False,
        include_n_targets_in_input=False,
        output_dir="notebooks",
        extra_training_args={
            #     "eval_steps": 2000,
            #     "save_steps": 2000,
            "warmup_steps": 3000,
            "learning_rate": 2e-5,
            #     "weight_decay": 0.01,
            #     "label_smoothing_factor": 0.25,
        },
        # is_test_run=True,
    )


if __name__ == "__main__":
    main()
