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
from bigbench.api.results import QueryType, ResultsFileData
import bigbench.api.results as bb

from tqdm import tqdm

from lass.log_handling import LogLoader, TaskLogs


def test_whether_samples_in_same_order_model_wise():
    """
    For a given tasks, are samples (and queries) in the same order (in the logs)
    for all models?
    """

    loader = (LogLoader("./artifacts/logs")
              .with_tasks('paper-full')
              .with_output_unit('task'))

    faulty: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    print("Testing whether all samples are in the same order model-wise...")
    for unit in tqdm(loader.load(), total=len(loader.tasks)):
        task: TaskLogs = cast(TaskLogs, unit)
        reference = list(task.values())[0]
        task_name = reference.task.task_name
        assert reference.queries, task_name

        for model, logfile in list(task.items())[1:]:
            if not logfile.queries:
                print(f"{model} has no queries for {task_name}")
                faulty[task_name].append(model)
                continue

            if not len(logfile.queries) == len(reference.queries):
                print(f"{model} has different number of queries for {task_name}")
                faulty[task_name].append(model)

            for qidx, query in enumerate(logfile.queries):
                if not query_meta_equal(query, reference.queries[qidx]):
                    print(f"{model} has different query {qidx} for {task_name}")
                    faulty[task_name].append(model)
                    continue

                for sidx, sample in enumerate(query.samples):
                    ref_input = reference.queries[qidx].samples[sidx].input
                    if not sample.input == ref_input:
                        print(f"{model} has different sample order for {task_name} in query {qidx}")
                        faulty[task_name].append(model)
                        break
    if faulty:
        print("\n\n")
        print("The following models have samples in different order:")
        for task_name, models in faulty.items():
            print(f"{task_name}: {models}")
    else:
        print("All models have the samples in the same order.")


def query_meta_equal(query: QueryType, reference: QueryType) -> bool:
    # TODO: This is a proxy for actual equality, there's more fields in each query type
    return (query.function == reference.function
            and len(query.samples) == len(reference.samples)
            and query.shots == reference.shots)


def test_whether_samples_in_same_order_shot_wise():
    """
    For a given tasks and model, are samples in the same order across shots?
    """

    loader = (LogLoader("./artifacts/logs")
              .with_tasks('paper-full')
              #   .with_tasks(['social_iqa'])
              .with_output_unit('results-file'))

    faulty: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    previous_task_name = loader.tasks[0]

    print("Testing whether all samples are in the same order shot-wise...")
    pbar = tqdm(total=len(loader.tasks))
    for unit in loader.load():
        # Set up
        log: ResultsFileData = cast(ResultsFileData, unit)
        task_name = log.task.task_name
        model = (log.model.model_family, log.model.model_name)
        assert log.queries, f"{log.task.task_name} has no queries for {model}"
        if task_name != previous_task_name:
            previous_task_name = task_name
            pbar.update(1)

        # Group queries across shots
        # We assume all queries that differ only in shots are after one another,
        # and that shots are incremented by 1.
        groups: List[List[Tuple[Union[int, None], QueryType]]] = [[(0, log.queries[0])]]
        for qidx, query in enumerate(log.queries[1:], 1):
            previous_query = groups[-1][-1][1]
            # Interpret None as 0 (for previous query only)
            previous_shots = previous_query.shots or 0

            # Query types differ -> new group
            if query.__class__ != previous_query.__class__:
                groups.append([(qidx, query)])
                continue

            # No shots -> new group
            if query.shots is None:
                groups.append([(qidx, query)])
                continue

            # Successive shots -> add to group
            if previous_shots + 1 == query.shots:
                groups[-1].append((qidx, query))

            # Shots are not consecutive -> new group
            else:
                groups.append([(qidx, query)])

        # print([len(group) for group in groups])

        # Check consistency across shots for each query type
        for group in groups:
            ref_idx, reference = group[0]
            for (qidx, query) in group[1:]:
                for sidx, sample in enumerate(query.samples):
                    ref_input = reference.samples[sidx].input
                    if not is_same_instance(sample.input, ref_input):
                        # print(f"{model} has different sample order for {task_name} in query {qidx}")
                        faulty[log.task.task_name].append(model)
                        # input(f"{log.task.task_name} Press enter to continue...")
                        break

                # Breaking out of lested loops
                # https://stackoverflow.com/questions/653509/breaking-out-of-nested-loops
                else:
                    continue
                break
            else:
                continue
            break

    pbar.close()
    if faulty:
        print("\n\n")
        print("The following models have samples in different order:")
        for task_name, models in faulty.items():
            print(f"{task_name}: {models}")
        print(f"{len(faulty)} tasks have different sample order.")
    else:
        print("All models have the samples in the same order.")


def is_same_instance(input: str, ref: str):
    """
    The same instance across multiple shots won't have the same input text,
    since for higher shots, there will be examples prepended.
    Assuming the reference is 0-shot, we could check if 'input' ends on 'ref'.
    There might also be a shared prefix before the examples describing the task.
    """
    # Find longest prefix
    first_prefix_diff = 0
    for i in range(len(ref)):
        if ref[i] != input[i]:
            break
        first_prefix_diff = i

    ref_instance = ref[first_prefix_diff:]
    # print(ref_instance, input)
    return input.endswith(ref_instance)


def test_is_same_instance_1():
    # Task: social_iqa
    ref = "\nQ: Taylor got married but decided to keep her identity and last name. How would you describe Taylor? \n  choice: Independent\n  choice: Like a feminist\n  choice: Shy and reticent\nA: "
    input = "\nQ: Skylar lent attention to the view even though she hated the view. Why did Skylar do this? \n  choice: Wanted to act interested\n  choice: Wanted to express her hatred\n  choice: Wanted to show she was unimpressed\nA: Wanted to act interested\n\nQ: Taylor got married but decided to keep her identity and last name. How would you describe Taylor? \n  choice: Independent\n  choice: Like a feminist\n  choice: Shy and reticent\nA: "
    assert is_same_instance(input, ref)


def test_is_same_instance_2():
    # Task: abstract_narrative_understanding
    ref = "In what follows, we provide short narratives, each of which illustrates a common proverb. \nNarrative: Carla was having trouble juggling college, working at the diner and being a mother. She never had a day off and was burnt out. Her friend told her that her hard work would pay off one day and to keep going! Carla's friend was right; after a few more years of hard work, Carla graduated school and was able to get a better job. She was able to take a vacation and become overall successful.\nThis narrative is a good illustration of the following proverb: "
    input = "In what follows, we provide short narratives, each of which illustrates a common proverb. \nNarrative: Margie got along with everyone, which is unusual.  During a workshop on interpersonal interactions, her coworkers asked her how she could be so nice to nasty people.  She said she had found \"killing them with kindness\" to be effective.  Not to get angry, but to be very nice instead.\nThis narrative is a good illustration of the following proverb: A soft answer turneth away wrath\n\nNarrative: Carla was having trouble juggling college, working at the diner and being a mother. She never had a day off and was burnt out. Her friend told her that her hard work would pay off one day and to keep going! Carla's friend was right; after a few more years of hard work, Carla graduated school and was able to get a better job. She was able to take a vacation and become overall successful.\nThis narrative is a good illustration of the following proverb: "
    assert is_same_instance(input, ref)


if __name__ == '__main__':
    # test_whether_samples_in_same_order_model_wise()
    test_is_same_instance_1()
    test_is_same_instance_2()
    test_whether_samples_in_same_order_shot_wise()
