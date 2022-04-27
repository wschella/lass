#! .env/bin/python
"""
Test assumptions about the BIG-bench evaluation logs.
- Are all models evaluated on all tasks?
- Are all models evaluated with the same number of queries?
- Are all models evaluated with the same number of samples?
- Are all samples the same?
- Are all samples in the same order?
- Are queries in the same order?
"""
from typing import *
from collections import defaultdict
from bigbench.api.results import QueryType

from tqdm import tqdm

from lmasss.log_handling import LogLoader, TaskLogs


def test_wether_samples_in_same_order_model_wise():
    """
    For a given tasks, are samples (and queries) in the same order (in the logs)
    for all models?
    """

    loader = (LogLoader("./artifacts/logs")
              .with_tasks('paper-full')
              .with_output_unit('task'))

    results: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    print("Testing whether all samples are in the same order...")
    for unit in tqdm(loader.load(), total=len(loader.tasks)):
        task: TaskLogs = cast(TaskLogs, unit)
        reference = list(task.values())[0]
        task_name = reference.task.task_name
        assert reference.queries, task_name

        for model, logfile in list(task.items())[1:]:
            if not logfile.queries:
                print(f"{model} has no queries for {task_name}")
                results[task_name].append(model)
                continue

            if not len(logfile.queries) == len(reference.queries):
                print(f"{model} has different number of queries for {task_name}")
                results[task_name].append(model)

            for qidx, query in enumerate(logfile.queries):
                if not query_meta_equal(query, reference.queries[qidx]):
                    print(f"{model} has different query {qidx} for {task_name}")
                    results[task_name].append(model)
                    continue

                for sidx, sample in enumerate(query.samples):
                    ref_input = reference.queries[qidx].samples[sidx].input
                    if not sample.input == ref_input:
                        print(f"{model} has different sample order for {task_name} in query {qidx}")
                        results[task_name].append(model)
                        break
    if results:
        print("\n\n")
        print("The following models have samples in different order:")
        for task_name, models in results.items():
            print(f"{task_name}: {models}")
    else:
        print("All models have the samples in the same order.")


def query_meta_equal(query: QueryType, reference: QueryType) -> bool:
    # TODO: This is a proxy for actual equality, there's more fields in each query type
    return (query.function == reference.function
            and len(query.samples) == len(reference.samples)
            and query.shots == reference.shots)


if __name__ == '__main__':
    test_wether_samples_in_same_order_model_wise()
