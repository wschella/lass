import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import lass.plotting.shared as shared
from lass.plotting.shared import Baseline, shorten


@dataclass
class Args:
    pass


def main():
    parser = argparse.ArgumentParser()
    args_raw = parser.parse_args()
    run(Args(**vars(args_raw)))


def run(args: Args):
    base = Path("../../../artifacts/csv-results-new/")
    # path = base / "q1indistribution/deberta-base_bs32_3sh_instance-split-07122152/"
    path = base / "reproduce_rev1_q1/latest"

    results = shared.load_metrics(path)
    plot(results)


def plot(df: pd.DataFrame, baseline: Baseline = "normalized", metric: str = "roc_auc"):
    pass


if __name__ == "__main__":
    main()
