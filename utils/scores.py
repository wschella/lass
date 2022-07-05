from typing import *
from pathlib import Path
from pprint import pprint

import bigbench.api.results as bb
import numpy as np

from lass.log_handling import LogLoader
from lass.datasets import split_instance_level, split_task_level, to_dataframe, analyse, merge

shots = [
    # [0],
    # [3],
    [0, 1, 2, 3]
]


def main():
    get_loader = lambda shots: (LogLoader(logdir=Path('artifacts/logs'))
                                .with_tasks('paper-full')
                                .with_model_families(['BIG-G T=0'])
                                .with_model_sizes(['128b'])
                                .with_shots(shots)
                                .with_query_types([bb.MultipleChoiceQuery]))

    for shot in shots:
        loader = get_loader(shot)
        # Instance split
        train, test = split_instance_level(loader, seed=42, test_fraction=0.2)
        rprint("Instance split", shot, train, test)
        del train, test

        # Task split
        train, test = split_task_level(loader, seed=42, test_fraction=0.2)
        rprint("Task split", shot, train, test)
        del train, test

        # Task distribution shift
        data = to_dataframe(loader)
        accs = (data
                .groupby('task', as_index=False).agg(acc=('correct', 'mean'))  # type: ignore
                .sort_values('acc', ascending=False))

        train_tasks, test_tasks = np.split(accs['task'], [int(len(accs) * 0.8)])
        train = data[data['task'].isin(train_tasks)]
        test = data[data['task'].isin(test_tasks)]
        rprint("Task distribution shift", shot, train, test)
        del train, test


def rprint(name, shots, train, test):
    print(f"-----{name} ({shots} shots)")
    r = merge(analyse(train), analyse(test), 'train', 'test')
    pprint(r)


if __name__ == '__main__':
    main()
