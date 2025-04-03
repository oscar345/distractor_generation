import enum
from dataclasses import dataclass, field
from typing import Callable, cast
from datasets.load import DatasetDict
from polars import Date
from simple_parsing import ArgumentParser
from datasets import load_dataset
import utils


class ModelType(enum.Enum):
    encoder_decoder = "encoder_decoder"
    decoder = "decoder"


class Model(enum.Enum):
    baseline = "baseline"
    bert_decoder = "bert_decoder"
    llama = "llama"
    mistrall = "mistrall"


class Mode(enum.Enum):
    train = "train"
    eval = "eval"
    predict = "predict"
    preprocess = "preprocess"


@dataclass
class Options:
    """Options for the models."""

    model: Model = field()  # Choose what model you want to use
    mode: Mode = field()  # You can use different modes to run the part you want: train, eval, predict, preprocess


@dataclass
class Config(Options):
    """Configuration for the models. This includes all config options that are needed to run the models."""

    model_type: ModelType = field()

    data_directory: str = field(default="data")

    dataset_name: str = field(default="allenai/sciq")
    llama_model_name: str = field(default="meta-llama/Llama-3.2-1B")
    mistrall_model_name: str = field(default="")
    bert_model_name: str = field(default="bert-base-uncased")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    return parser.parse_args()


def preprocess(
    preprocess_func: Callable[[DatasetDict], DatasetDict],
    config: Config,
):
    dataset = cast(DatasetDict, load_dataset(config.dataset_name))
    print("Dataset loaded")
    dataset = preprocess_func(dataset)
    utils.save_dataset(dataset, config)
    print("Dataset preprocessed")


def run(config: Config):
    print("Running...")

    match (config.model, config.model_type, config.mode):
        case (_, ModelType.encoder_decoder, Mode.preprocess):
            preprocess(utils.preprocess_encoder_decoder, config)
        case (_, ModelType.decoder, Mode.preprocess):
            preprocess(utils.preprocess_decoder, config)
        case _:
            raise ValueError(
                f"Unsupported model type {config.model_type} for mode {config.mode}"
            )


def main():
    arguments = parse_arguments()

    config = Config(
        **vars(arguments.options),
        model_type=(
            ModelType.encoder_decoder
            if arguments.options.model == Model.bert_decoder
            else ModelType.decoder
        ),
    )

    run(config)


if __name__ == "__main__":
    main()
