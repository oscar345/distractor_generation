from options import Options, Model, ModelType, Config, Mode
from typing import Callable, cast
from datasets.load import DatasetDict
from simple_parsing import ArgumentParser
from datasets import load_dataset
from models import baseline, bert, llama, mistral
import utils
import preprocessing


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

    # - predict: all models
    # - preprocess: all models
    # - train: bert, (baseline)
    # - evaluation: ...

    match (config.model, config.model_type, config.mode):
        case (_, ModelType.encoder_decoder, Mode.preprocess):
            preprocess(preprocessing.preprocess_encoder_decoder, config)
        case (_, ModelType.decoder, Mode.preprocess):
            preprocess(preprocessing.preprocess_decoder, config)
        case (Model.bert_decoder, _, Mode.train):
            bert.train(config)
        case (Model.bert_decoder, _, Mode.predict):
            bert.predict(config)
        case (Model.llama, _, Mode.predict):
            llama.predict(config)
        case (Model.baseline, _, Mode.predict):
            baseline.predict(config)
        case (Model.mistral, _, Mode.predict):
            mistral.predict(config)
        case _:
            raise ValueError(
                # this can happen for the "baseline" model, which is a regular llama model (not fine-tuned),
                # and you use the mode "train"
                f"This combination of model type {config.model_type} and mode {config.mode} is not supported"
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
