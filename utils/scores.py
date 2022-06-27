from typing import *
from pathlib import Path

import bigbench.api.results as bb

from lass.log_handling import LogLoader


def main():
    loader = (LogLoader(logdir=Path('../artifacts/logs'))
              .with_tasks('paper-full')
              .with_model_families(['BIG-G T=0'])
              .with_model_sizes(['128b'])
              #   .with_shots([])
              .with_query_types([bb.MultipleChoiceQuery]))

    pass


def scores(samples: Iterator[bb.SampleType]) -> Dict[str, float]:
    return {}


if __name__ == '__main__':
    main()
