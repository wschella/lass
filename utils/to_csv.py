#! .env/bin/python

import csv
from pathlib import Path
from typing import *

import bigbench.api.results as bb

from lass.log_handling import LogLoader


def main():
    loader = (LogLoader(logdir=Path('artifacts/logs'))
              .with_tasks('paper-full')
              .with_model_families(['BIG-G T=0'])
              .with_model_sizes(['128b'])
              .with_shots([0])
              .with_query_types([bb.MultipleChoiceQuery])
              )
    folder = Path('artifacts/csv-datasets')
    folder.mkdir(exist_ok=True)
    with open(folder / 'paper-full-128b-0sh.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['text', 'target'])

        for sample in loader.load_per_sample():
            sample: bb.SampleType

            csvwriter.writerow([sample.input, int(sample.correct)])  # type: ignore


if __name__ == '__main__':
    main()
