from transformers import AutoModelForCausalLM
from models import decoder
from options import Config
from typing import cast
from datasets import Dataset, load_dataset
import utils
import torch


def predict(config: Config):
    print("Loading dataset...")
    original_dataset = cast(Dataset, load_dataset(config.dataset_name, split="test"))
    test_dataset = cast(Dataset, utils.load_dataset_from_disk(config).get("test"))
    print("Dataset loaded")

    device = utils.get_device()
    tokenizer = decoder.load_tokenizer(config.llama_model_name)

    dataset = test_dataset.map(
        lambda examples: decoder.tokenize_function(
            tokenizer, examples, with_labels=False, text_column="baseline"
        ),
        batched=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.llama_model_name,
        load_in_4bit=decoder.load_in_4bit(device),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded")
    model = model.to(device)

    distractors = decoder.generate_predictions(
        model, dataset, tokenizer, device, is_baseline=True
    )
    print(distractors)
    predictions_dataset = utils.replace_distractors_in_dataset(
        original_dataset, distractors
    )
    predictions_dataset.save_to_disk(utils.get_predictions_directory_name(config))
