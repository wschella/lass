import dataclasses
import logging
from typing import Any, Optional, Tuple, Union, Literal, cast

import pandas as pd
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.generic import PaddingStrategy


def wrangle(
    df: pd.DataFrame,
    include_model_in_input: bool,
    include_n_targets_in_input: bool,
    filter_bad_tasks: bool,
) -> pd.DataFrame:
    df = binarize(df)
    df = augment(df)
    df = clean(df)
    if filter_bad_tasks:
        df = remove_bad_tasks(df)

    df = prepend_extra_features(
        df,
        include_model=include_model_in_input,
        include_n_targets=include_n_targets_in_input,
    )

    return df


def augment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the dataframe with additional columns.
    """
    df["n_targets"] = df["targets"].map(lambda x: len(x))
    df["conf_normalized"] = np.exp(df["normalized_scores"].map(lambda s: max(s)))
    df["conf_absolute"] = np.exp(df["absolute_scores"].map(lambda s: max(s)))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe of samples that are not correct.
    """
    single_target_mpc = df.query("query_type == 'mpc' and n_targets == 1").index
    df = df.drop(single_target_mpc)
    logging.info(f"Dropped {len(single_target_mpc)} single-target MPC tasks")
    return df


def binarize(df: pd.DataFrame) -> pd.DataFrame:
    # Drop all samples that do not have binary correctness
    # We could also round instead of drop here
    is_non_binary = ~(df["correct"].isin([0.0, 1.0]))
    df = df.drop(df[is_non_binary].index)
    logging.info(f"Dropped {is_non_binary.sum()} samples with non-binary correctness")

    # and convert the labels to ints afterwards
    df.loc[:, "correct"] = df["correct"].astype(int)

    return df


def remove_bad_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove tasks with aggregate performance that is too low
    Note: This aggregates across models.
    Note: Data will look different for different models.
    """
    if df[["model_name", "model_family"]].nunique().max() > 1:
        logging.warning(
            "Data contains multiple models. Dropping tasks  based on performance of best model."
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

    # For MPC tasks, the threshold is random chance + 0.05
    # with random chance being 1/n_targets
    mpc_perf = df_best.query("query_type == 'multiple_choice'").groupby("task").mean()
    bad_mpc_tasks = mpc_perf[
        mpc_perf["correct"] < 1 / mpc_perf["n_targets"] + 0.05
    ].index
    df = df[~df["task"].isin(bad_mpc_tasks)]
    logging.info(f"Removed {len(bad_mpc_tasks)} MPC tasks with low performance")

    # For generative tasks, the threshold is 0.05
    gen_perf = (
        df_best.query("query_type == 'scoring_generative'")
        .groupby("task")["correct"]
        .mean()
    )
    bad_gen_tasks = gen_perf[gen_perf < 0.05].index
    df = df[~df["task"].isin(bad_gen_tasks)]
    logging.info(f"Removed {len(bad_gen_tasks)} generative tasks with low performance")
    return df


def prepend_extra_features(
    df: pd.DataFrame, include_model: bool, include_n_targets: bool
) -> pd.DataFrame:
    # In case of population-split, split will order by input. Make sure prepend_extra_features can not change the order.
    assert (
        not include_n_targets or not df.model_name.nunique() > 1
    ), "Population split not supported with include_n_targets_in_input"

    # No-op if we don't need to include any extra features
    if not include_model and not include_n_targets:
        return df

    # By default do nothing
    model_formatter = lambda r: ""
    n_targets_formatter = lambda r: ""

    # Prepend extra features if needed
    if include_model:
        # E.g. FAM: BIG-G T=0 SIZE: 128b
        model_formatter = lambda r: f"FAM: {r['model_family']} SIZE:{r['model_name']} "
    if include_n_targets:
        n_targets_formatter = lambda r: f"N_TARGETS: {r['n_targets']} "
    prepender = lambda r: f"{model_formatter(r)} {n_targets_formatter(r)} {r.input}"

    df["input"] = df.apply(prepender, axis=1)
    return df


def huggingfaceify(df: pd.DataFrame) -> Dataset:
    """
    Prepare a dataframe of BigBench samples for use with HuggingFace transformers.
    """
    # Take only the columns we need, and rename them appropriately
    df_hf = df[["input", "correct"]].rename(
        columns={"input": "text", "correct": "label"}
    )
    return Dataset.from_pandas(df_hf, preserve_index=False)


def huggingfaceify_original(df: pd.DataFrame) -> Dataset:
    """
    Prepare a dataframe of BigBench samples for use with HuggingFace transformers.
    Solve the original task, instead of failure prediction.
    """
    find_label = lambda row: max(
        ((idx, row.target_values[target]) for idx, target in enumerate(row.targets)),
        key=lambda x: x[1],
    )[0]
    df_hf = pd.DataFrame()
    df_hf["text"] = df.input
    df_hf["options"] = df.targets
    df_hf["label"] = df.apply(find_label, axis=1)
    # Watch out for those with multiple correct labels.
    return Dataset.from_pandas(df_hf, preserve_index=False)


def huggingfaceify_splits(train: pd.DataFrame, test: pd.DataFrame) -> DatasetDict:
    ds = DatasetDict()
    ds["train"] = huggingfaceify(train)
    ds["test"] = huggingfaceify(test)
    return ds


def tokenize(
    ds: Union[Dataset, DatasetDict],
    model_name: str,
    max_sequence_length: int,
    truncation_side: Union[Literal["left"], Literal["right"]] = "right",
) -> Any:
    tokenizer = _get_tokenizer(model_name, truncation_side=truncation_side)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="np",
        )

    return ds.map(tokenize_function, batched=True)


def tokenize_mpc(
    ds: Union[Dataset, DatasetDict],
    model_name: str,
    max_sequence_length: int,
    n_targets: int,
    truncation_side: Union[Literal["left"], Literal["right"]] = "right",
) -> Any:
    tokenizer = _get_tokenizer(model_name, truncation_side=truncation_side)

    def preprocess_function(examples):
        # Create a 'text' context for each option
        texts = [[text] * n_targets for text in examples["text"]]
        options = examples["options"]

        # Flatten the lists
        texts = sum(texts, [])
        options = sum(options, [])

        # Tokenize the flattened representations
        tokenized_examples = tokenizer(
            texts,
            options,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length,
        )

        # Un-flatten the lists
        values = {
            k: [v[i : i + n_targets] for i in range(0, len(v), n_targets)]
            for k, v in tokenized_examples.items()
        }
        return examples | values

    return ds.map(preprocess_function, batched=True)


def _get_tokenizer(
    model_name: str, truncation_side: Union[Literal["left"], Literal["right"]] = "right"
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, truncation_side=truncation_side
    )

    if model_name == "openai-community/gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
