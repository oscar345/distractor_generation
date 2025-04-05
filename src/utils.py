from typing import cast, List
import polars as pl
from datasets import DatasetDict, Dataset, load_from_disk
import os
import torch
from dataclasses import asdict, dataclass

from options import Config


@dataclass
class Distractors:
    distractor1: str
    distractor2: str
    distractor3: str


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


def get_predictions_directory_name(config: Config):
    return os.path.join(config.predictions_directory, config.model.value)


def replace_distractors_in_dataset(
    dataset: Dataset, distractors: List[Distractors]
) -> Dataset:
    df_dataset = cast(pl.DataFrame, dataset.to_polars())
    df_tokens = pl.DataFrame([asdict(object) for object in distractors])

    df = df_dataset.update(df_tokens, include_nulls=True)

    return Dataset.from_polars(df)
