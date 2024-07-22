import json
from pathlib import Path
from typing import Literal

import pandas as pd


def shorten(x, length=15):
    """
    Shorten a string to a specific length, adding ellipsis to the end.
    """
    if len(x) > length:
        return x[: length - 3] + "..."
    else:
        return x


def load_metrics(dir: Path, save: bool = True) -> pd.DataFrame:
    # Load task based metrics
    results = pd.read_json(dir / "metrics_per_task.json")
    results = results.T
    results.index.names = ["task"]
    results = results.reset_index()

    # Load total dataset metrics
    with open(dir / "metrics.json") as f:
        total = pd.json_normalize(json.load(f))
        total["task"] = "total"
    results = pd.concat([results, total], ignore_index=True)

    # Put all columns starting with test_conf_ last
    results = results[sorted(results.columns, key=lambda x: x.startswith("test_conf_"))]

    # Save to csv for easier sharing with journal and co-authors
    if save:
        results.to_csv(dir / "metrics.csv", index=False)

    return results


Baseline = Literal["normalized", "absolute", "distribution"]
