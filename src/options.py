import enum
from dataclasses import dataclass, field


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
    results_directory: str = field(default="results")
    model_directory: str = field(default="models")

    dataset_name: str = field(default="allenai/sciq")
    llama_model_name: str = field(default="meta-llama/Llama-3.2-1B")
    mistrall_model_name: str = field(default="")
    bert_model_name: str = field(default="bert-base-uncased")
