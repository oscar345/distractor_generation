# A lot of the code between the three decoder models is similar. An example
# is the tokenizer function. Since that code is shared, it is stored in a
# separate file.

from typing import Dict, List, Any
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
import re

from utils import Distractors


def tokenize_function(tokenizer: AutoTokenizer, examples: Dict, with_labels=True):
    """"""
    text = (
        [text + target for text, target in zip(examples["text"], examples["target"])]
        if with_labels
        else examples["text"]
    )

    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=768 + 64,
        return_tensors="pt",
    )  # type: ignore

    if with_labels:
        return {
            "labels": tokenized["input_ids"].clone(),
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def load_in_4bit(device: torch.device) -> bool:
    return device == "cuda"


def safe_get(list: List[Any], index: int) -> Any:
    l = len(list)

    if index < l:
        return list[index]
    else:
        return None


def generate_predictions(
    model: AutoModelForCausalLM,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> List[Distractors]:
    dataloader = DataLoader(
        dataset,  # type:ignore
        batch_size=4,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    model.eval()  # type:ignore set the model in evaluation mode

    outputs = []

    for batch in dataloader:
        # set input to the correct device
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            generated = model.generate(  # type: ignore
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=64,
                do_sample=True,
                top_k=50,
            )

        text = tokenizer.batch_decode(generated, skip_special_tokens=True)  # type:ignore
        outputs.extend(text)

    pattern = r"Distractor \d+: (.+)"
    distractors = []

    for output in outputs:
        answers = re.findall(pattern, output)

        distractor = Distractors(
            distractor1=safe_get(answers, 0),
            distractor2=safe_get(answers, 1),
            distractor3=safe_get(answers, 2),
        )

        distractors.append(distractor)

    return distractors
