import dataclasses
import logging
from typing import List, Literal, Tuple


import bigbench.api.results as bb
import pandas as pd

from lass.log_handling import LogLoader
from lass.log_handling import LogLoaderArgs


def load(data_spec: LogLoaderArgs, is_test_run: bool = False) -> pd.DataFrame:
    if is_test_run:
        data_spec = dataclasses.replace(data_spec, tasks="pipeline-test")
    loader = LogLoader(data_spec)
    df = to_dataframe(loader)
    df = remove_duplicate_queries(df)
    return df


def to_dataframe(loader: LogLoader) -> pd.DataFrame:
    """
    Columns in the output dataframe:
    - input
    - targets
    - scores
    - target_values
    - correct
    - absolute_scores
    - normalized_scores
    - metrics (only for MPC)
    - output (only for gen/scoring)
    - task
    - query_type
    - shots
    - model_name
    - model_family
    """
    tasks: List[bb.ResultsFileData] = list(loader.load_per_model())
    assert len(tasks) > 0, "No tasks found."
    dfs: List[pd.DataFrame] = []
    for task in tasks:
        queries_to_merge = []
        for query in task.queries or []:
            query_type = get_query_type(query)

            if query_type == "multiple_choice":
                df = pd.DataFrame(query.samples)
                df["model_name"] = task.model.model_name
                df["model_family"] = task.model.model_family
                df["task"] = task.task.task_name
                df["shots"] = query.shots
                df["query_type"] = query_type
                dfs.append(df)
            # If scoring/generative query, we need to merge them
            # because confidence scores are in the scoring query,
            # and exact_str_match and outputs are in the generative query
            else:
                queries_to_merge.append(query)

        # Merge scoring and generative queries
        # We assume they are interleaved generative - scoring - generative - scoring ...
        # Iterate per 2 queries + metadata
        for gen_q, score_q in zip(queries_to_merge[0::2], queries_to_merge[1::2]):
            if (
                get_query_type(gen_q) != "generative"
                or get_query_type(score_q) != "scoring"
            ):
                break  # Ignore the non-MPC queries if they are not interleaved gen - scoring pairs

            assert len(gen_q.samples) == len(score_q.samples)

            samples = [
                merge_scoring_generative_sample(gen_s, score_s)
                for gen_s, score_s in zip(gen_q.samples, score_q.samples)
            ]
            df = pd.DataFrame(samples)
            df["model_name"] = task.model.model_name
            df["model_family"] = task.model.model_family
            df["task"] = task.task.task_name
            df["shots"] = gen_q.shots
            df["query_type"] = "scoring_generative"
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f"No data found.")

    return pd.concat(dfs, ignore_index=True)


def merge_scoring_generative_sample(gen_sample, scoring_sample):
    assert scoring_sample.input == gen_sample.input
    # assert len(gen_sample.targets) == len(scoring_sample.targets) # Some weird data-bug with repeated targets in only the scoring sample causes this to not hold
    assert len(gen_sample.targets) >= 1
    sample = dataclasses.asdict(scoring_sample)
    sample["output"] = gen_sample.output
    sample["target_values"] = {target: 1 for target in scoring_sample.targets}
    sample["correct"] = max(
        [target["exact_str_match"] for target in gen_sample.targets.values()]
    )
    return sample


def get_query_type(query) -> Literal["multiple_choice", "generative", "scoring"]:
    if query.__class__ == bb.MultipleChoiceQuery:
        return "multiple_choice"
    elif query.__class__ == bb.GenerativeQuery:
        return "generative"
    elif query.__class__ == bb.ScoringQuery:
        return "scoring"
    else:
        raise ValueError(f"Unknown query class: {query.__class__}")


def remove_duplicate_queries(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the tasks that have both a multiple_choice and a scoring_generative query_type,
    remove one of them.
    """
    if df[["model_name", "model_family"]].nunique().max() > 1:
        logging.warning(
            "Data contains multiple models. Dropping duplicate queries based on performance of best model."
        )

    if "128b" not in df.model_name.unique():
        raise ValueError(
            "Data does not contain 128b model. This is unexpected. Are you sure?"
        )

    # First select the runs for the best performing (overal!) model
    best: Tuple[str, str] = (
        df.groupby(["model_name", "model_family"]).correct.mean().idxmax()
    )  # type: ignore
    df_best = df.query(f"model_name == '{best[0]}' and model_family == '{best[1]}'")

    # Find the tasks that have both a multiple_choice and a scoring_generative query_type
    tasks_with_both = df_best.groupby(["task", "query_type"]).query_type.nunique()
    tasks_with_both = tasks_with_both[tasks_with_both == 2].index

    # Iterate over the tasks and decide which query_type to drop (get a list of pairs (task, query_type))
    queries_to_drop = [
        (task, decide_which_query_to_drop(df_best.query(f"task == '{task}'")))
        for task in tasks_with_both
    ]

    # Drop the queries
    for task, query_type in queries_to_drop:
        df = df.drop(
            df.query(f"task == '{task}' and query_type == '{query_type}'").index
        )

    return df


def decide_which_query_to_drop(df, threshold=0.05):
    return (
        "multiple_choice"
        if decide_which_query_to_keep(df, threshold) == "scoring_generative"
        else "scoring_generative"
    )


def decide_which_query_to_keep(df, threshold=0.05):
    """
    This receives a DataFrame with both queries for a specific task, and returns the query_type to keep.

    Basically select scoring_generative if it is above some threshold, otherwise select multiple_choice.
    """
    # Calculate scores and separate them by query_type
    scores = (
        df.groupby(["query_type"])
        .correct.describe()
        .reset_index()
        .sort_values("query_type", ascending=True)
    )
    mpc = scores.iloc[0]
    sg = scores.iloc[1]

    assert (
        mpc.query_type == "multiple_choice"
    ), "The first query_type should be multiple_choice"

    assert (
        sg.query_type == "scoring_generative"
    ), "The second query_type should be scoring_generative"

    if mpc["mean"] < threshold and sg["mean"] < threshold:
        return "scoring_generative" if sg["mean"] > mpc["mean"] else "multiple_choice"

    if sg["mean"] < threshold:
        return "multiple_choice"

    if mpc["mean"] < threshold:
        return "scoring_generative"

    return "scoring_generative"
