from typing import cast
import polars as pl
from datasets import DatasetDict, Dataset, load_from_disk
import os
import torch

from options import Config


def dataset_to_df(dataset: DatasetDict) -> pl.DataFrame:
    """Convert a DatasetDict to a Polars DataFrame while preserving the split information."""
    return pl.concat(
        [
            dataset.to_polars().with_columns([pl.lit(split).alias("split")])
            for split, dataset in dataset.items()
        ]
    )


def df_to_dataset(df: pl.DataFrame) -> DatasetDict:
    """Convert a Dataframe back to a DatasetDict with the right split."""
    return DatasetDict(
        {
            "train": Dataset.from_polars(df.filter(pl.col("split") == "train")),
            "validation": Dataset.from_polars(
                df.filter(pl.col("split") == "validation")
            ),
            "test": Dataset.from_polars(df.filter(pl.col("split") == "test")),
        }
    )


def preprocess_encoder_decoder(dataset: DatasetDict):
    """
    Preprocess encoder-decoder model

    The dataset will be converted so there wont be three distractors per row, but a
    single distractor. To keep all distractors, the original row will be split into
    three rows. For the encoder decoder model we believe this to be beneficial, so it
    can focus on generating small sequences of tokens that don't vary too much in length.
    Those lengths may vary more when predicting multiple answer options at once.

    So if the current dataset looks like this: `A B C1 C2 C3`, it will become like this:

    ```
    A B C1
    A B C2
    A B C3
    ```
    """
    df = dataset_to_df(dataset)

    df = df.unpivot(
        index=["question", "correct_answer", "support", "split"],
        on=["distractor1", "distractor2", "distractor3"],
        variable_name="x",
        value_name="distractor",
    )

    df = df.drop(["x"])

    return df_to_dataset(df)


def preprocess_decoder(dataset: DatasetDict) -> DatasetDict:
    """Preprocess decoder model"""
    df = dataset_to_df(dataset)

    # We are creating two new columns: the text column with the input for the model such as the
    # the support, question and answer. The target for the model will be the three distractors.
    # With these transformations we can easily tokenize the data inside the tokenize function.
    df = df.with_columns(
        [
            pl.format(
                "Question: {}\nCorrect Answer: {}\nSupport: {}",
                pl.col("question"),
                pl.col("correct_answer"),
                pl.col("support"),
            ).alias("text"),
            pl.format(
                "Distractor 1: {}\nDistractor 2: {}\nDistractor 3: {}",
                pl.col("distractor1"),
                pl.col("distractor2"),
                pl.col("distractor3"),
            ).alias("target"),
        ]
    )

    # the following columns are no longer needed since they are used in the creation of the new columns
    df = df.drop(
        [
            "distractor1",
            "distractor2",
            "distractor3",
            "question",
            "correct_answer",
            "support",
        ]
    )

    return df_to_dataset(df)


def get_dataset_name(config: Config) -> str:
    return os.path.join(config.data_directory, config.model_type.value)


def save_dataset(dataset: DatasetDict, config: Config) -> None:
    """Save dataset to disk"""
    print(dataset)
    dataset.save_to_disk(get_dataset_name(config))


def load_dataset_from_disk(config: Config) -> DatasetDict:
    print("Loading dataset from disk...")
    dataset = cast(DatasetDict, load_from_disk(get_dataset_name(config)))
    print("Dataset loaded")
    return dataset


def get_device() -> torch.device:
    device = None

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available!")
    else:
        device = torch.device("cpu")
        print("MPS and CUDA are not available. Using CPU.")

    return device


def get_results_directory_name(config: Config) -> str:
    return os.path.join(config.results_directory, config.model.value)


def get_model_directory_name(config: Config):
    return os.path.join(config.model_directory, config.model.value)
