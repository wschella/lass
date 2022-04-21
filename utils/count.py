#! .env/bin/python
from typing import *
import textwrap

from lmasss.log_handling import LogLoader
import bigbench.api.results as bb


def main():
    # TODO: Just try removing ARC, because no targets
    loader = LogLoader()\
        .with_tasks('paper-lite')\
        .with_model_families(['BIG-G T=0'])\
        .with_model_sizes(['2m'])\
        .with_query_types([bb.GenerativeQuery])\
        # .with_shots([], include_unknown=True)\

    total_queries = 0
    total_samples = 0
    for log in loader.load():
        log = cast(bb.ResultsFileData, log)
        if not log.queries:
            print("\n{}\nNo (matching) queries.\n".format(log.task.task_name))
            continue

        n_queries = len(log.queries)
        n_samples = sum(len(q.samples) for q in log.queries)
        total_queries += n_queries
        total_samples += n_samples
        print(textwrap.dedent(
            f"""
            {log.task.task_name}
            #queries: {n_queries}
            sample sizes: {[len(q.samples) for q in log.queries]}
            #samples: {n_samples}
            """
        ))

    print(f"\nTotal: {total_queries} queries, {total_samples} samples")


if __name__ == '__main__':
    main()
