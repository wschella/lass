import logging
from typing import Any, Union, Literal, cast

import pandas as pd
import numpy as np
from transformers.models.auto.tokenization_auto import AutoTokenizer
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict


def wrangle(
    df: pd.DataFrame,
    include_model_in_input: bool = False,
    include_n_targets_in_input: bool = True,
) -> pd.DataFrame:
    df = binarize(df)
    df = augment(df)
    # df = clean(df)  # TODO: Still needed?
    # df = remove_bad_tasks(df) Note: Will make the data be different for different models

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
    single_target_mpc = df.query("task_type == 'mpc' and n_targets == 1").index
    df = df.drop(single_target_mpc)
    logging.info(f"Dropped {len(single_target_mpc)} single-target MPC tasks")
    return df


def binarize(df: pd.DataFrame) -> pd.DataFrame:
    # Drop all samples that do not have binary correctness
    # We could also round instead of drop here
    is_non_binary = (~df["correct"].isin([0.0, 1.0])).index
    df = df.drop(is_non_binary)
    logging.info(f"Dropped {len(is_non_binary)} samples with non-binary correctness")

    # and convert the labels to ints afterwards
    df.loc[:, "correct"] = df["correct"].astype(int)

    return df


def remove_bad_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove tasks with aggregate performance that is too low
    Note: This aggregates across models.
    Note: Data will look different for different models.
    """
    # For MPC tasks, the threshold is random chance + 0.05
    # with random chance being 1/n_targets
    mpc_perf = df.query("task_type == 'mpc'").groupby("task_name").mean()
    bad_mpc_tasks = mpc_perf[
        mpc_perf["correct"] < 1 / mpc_perf["n_targets"] + 0.05
    ].index
    df = df[~df["task_name"].isin(bad_mpc_tasks)]

    # For generative tasks, the threshold is 0.05
    gen_tasks = (
        df.query("task_type == 'generative'").groupby("task_name")["correct"].mean()
    )
    bad_gen_tasks = gen_tasks[gen_tasks < 0.05].index
    df = df[~df["task_name"].isin(bad_gen_tasks)]
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


def _get_tokenizer(
    model_name: str, truncation_side: Union[Literal["left"], Literal["right"]] = "right"
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, truncation_side=truncation_side
    )

    if model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def truncate(input: pd.Series, model_name: str, max_sequence_length: int) -> pd.Series:
    ds = Dataset.from_pandas(input.to_frame("text"))
    truncated_ds = truncate_(ds, model_name, max_sequence_length)

    return cast(pd.DataFrame, truncated_ds.to_pandas())["text"]


def truncate_(input: Dataset, model_name: str, max_sequence_length: int):
    """
    Encoding can be a destructive process [1], so we work with offset_mapping to determine
    the character indexes of the original string mapping to the last token.

    [1] https://github.com/huggingface/tokenizers/issues/826#issuecomment-966082496
    """

    tokenizer = _get_tokenizer(model_name)

    def truncate(batch):
        tokens = tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            return_offsets_mapping=True,
            max_length=max_sequence_length,
        )
        # Expected shape here is [batch size, sequence length (in tokens), 2]
        # but first dimension is a list, second a list, and third a tuple
        # We take the previous to last for each offset mapping, and take the
        # end off the span tuple.
        lengths = [l[-2][1] for l in tokens["offset_mapping"]]

        # We use `amax` as a trick to find the last non-zero offset mapping.
        # Array dimensions are [batch size, sequence length (in tokens), 2],
        # where the last dimension is [start, end] of the token (referring to index in the string).
        # With [:,:,-1], we make it [batch size, sequence length], taking only the end offset of the token.
        # Then we take the max of each sequence, producing batch_size numbers.
        # print(np.array(offset_mapping))
        # lengths = np.amax(offset_mapping[:,:,-1], axis=1) # type: ignore

        assert len(lengths) == len(batch["text"])

        # Now we cut all the strings
        texts = [text[:end] for text, end in zip(batch["text"], lengths)]
        return {"text": texts}

    return input.map(truncate, batched=True)
