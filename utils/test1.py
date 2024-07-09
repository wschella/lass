import logging

# autopep8: off
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from lass.test import test
from lass.log_handling import LogLoaderArgs
# autopep8: on


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(module)s/%(funcName)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    test(
        data_args=LogLoaderArgs(
            logdir="artifacts/logs",
            # tasks="paper-full",
            tasks=["hyperbaton"],
            model_families=["BIG-G T=0"],
            model_sizes=["128b"],
            # model_sizes=["2m"],
            shots=[0],
            query_types=["multiple_choice"],
        ),
        model_name="microsoft/deberta-v3-base",
        split="instance",
        model_loc="./artifacts/assessors/reference-models/deberta-reference-bs16*2-0sh-instance-split-10241537/checkpoint-4000",
        per_task=True,
    )


if __name__ == "__main__":
    main()
