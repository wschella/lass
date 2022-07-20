import logging

# autopep8: off
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from lass.train import train, DataArgs
# autopep8: on


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(module)s/%(funcName)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    train(
        data_args=DataArgs(
            logdir="artifacts/logs",
            tasks=["hyperbaton"],
            model_families=["BIG-G T=0"],
            model_sizes=["128b"],
            shots=[0],
            query_types=["multiple_choice"],
        ),
        split="instance",
        # model_name="albert-base-v2",
        # model_name_short="albert",
        # batch_size=32,
        model_name="microsoft/deberta-v3-base",
        model_name_short="deberta-hyperbaton",
        batch_size=16,
        gradient_accumulation_steps=2,
        output_dir="notebooks",
        extra_training_args={
            "eval_steps": 25,
            "save_steps": 25,
            "logging_steps": 25,
            # "warmup_steps": 5000,
            # "learning_rate": 5e-6,
            #     "weight_decay": 0.01,
            #     "label_smoothing_factor": 0.25,
        },
        # is_test_run=True,
    )


if __name__ == "__main__":
    main()
