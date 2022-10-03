import logging
from typing import *

import pandas as pd
import numpy as np

from transformers.models.auto.tokenization_auto import AutoTokenizer
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict


def augment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the dataframe with additional columns.
    """
    df['n_targets'] = df['targets'].map(lambda x: len(x))
    df['conf_normalized'] = np.exp(df['normalized_scores'].map(lambda s: max(s)))
    df['conf_absolute'] = np.exp(df['absolute_scores'].map(lambda s: max(s)))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe of samples that are not correct.
    """
    faulty = df[df['n_targets'] == 1].index
    df = df.drop(faulty)  # type: ignore
    logging.info(f"Dropped {len(faulty)} samples with faulty targets")
    return df


def binarize(df: pd.DataFrame) -> pd.DataFrame:
    # Drop all samples that do not have binary correctness
    # We could also round instead of drop here
    is_binary = df['correct'].isin([0.0, 1.0])
    df = df[is_binary]  # TODO: Use df.drop(~is_binary)
    logging.info(f"Dropped {len(is_binary) - len(df)} samples with non-binary correctness")

    # and convert the labels to ints afterwards
    df.loc[:, 'correct'] = df['correct'].astype(int)

    return df


def prepend_extra_features(df: pd.DataFrame, include_model: bool, include_n_targets: bool) -> pd.DataFrame:
    # No-op if we don't need to include any extra features
    if not include_model and not include_n_targets:
        return df

    # By default do nothing
    model_formatter = lambda r: ""
    n_targets_formatter = lambda r: ""

    # Prepend extra features if needed
    if include_model:
        model_formatter = lambda r: \
            f"FAM: {r['model_family']} SIZE:{r['model_name']} "
    if include_n_targets:
        n_targets_formatter = lambda r: \
            f"N_TARGETS: {r['n_targets']} "
    prepender = lambda r: f"{model_formatter(r)} {n_targets_formatter(r)} {r.input}"

    df['input'] = df.apply(prepender, axis=1)
    return df


def huggingfaceify(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a dataframe of BigBench samples for use with HuggingFace transformers.
    """
    # Take only the columns we need, and rename them appropriately
    df_hf = df[['input', 'correct']].rename(columns={'input': 'text', 'correct': 'label'})
    return df_hf


def get_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize(ds: Union[Dataset, DatasetDict], model_name: str, max_sequence_length) -> Any:
    tokenizer = get_tokenizer(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="np"
        )
    return ds.map(tokenize_function, batched=True)


def truncate(input: pd.Series, model_name: str, max_sequence_length: int) -> pd.Series:
    ds = Dataset.from_pandas(input.to_frame('text'))
    truncated_ds = truncate_(ds, model_name, max_sequence_length)

    return cast(pd.DataFrame, truncated_ds.to_pandas())['text']


def truncate_(input: Dataset, model_name: str, max_sequence_length: int):
    """
    Encoding is a destructive process [1], so we work with offset_mapping to determine 
    the character indexes of the original string mapping to the last token.

    [1] https://github.com/huggingface/tokenizers/issues/826#issuecomment-966082496
    """

    tokenizer = get_tokenizer(model_name)

    def truncate(batch):
        tokens = tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            return_offsets_mapping=True,
            max_length=max_sequence_length
        )
        # Expected shape here is [batch size, sequence length (in tokens), 2]
        # but first dimension is a list, second a list, and third a tuple
        # We take the previous to last for each offset mapping, and take the
        # end off the span tuple.
        lengths = [l[-2][1] for l in tokens['offset_mapping']]  # type: ignore

        # We use `amax` as a trick to find the last non-zero offset mapping.
        # Array dimensions are [batch size, sequence length (in tokens), 2],
        # where the last dimension is [start, end] of the token (referring to index in the string).
        # With [:,:,-1], we make it [batch size, sequence length], taking only the end offset of the token.
        # Then we take the max of each sequence, producing batch_size numbers.
        # print(np.array(offset_mapping))
        # lengths = np.amax(offset_mapping[:,:,-1], axis=1) # type: ignore

        assert len(lengths) == len(batch['text'])

        # Now we cut all the strings
        texts = [text[:end] for text, end in zip(batch["text"], lengths)]
        return {'text': texts}

    return input.map(truncate, batched=True)
