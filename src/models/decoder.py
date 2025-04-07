# A lot of the code between the three decoder models is similar. An example
# is the tokenizer function. Since that code is shared, it is stored in a
# separate file.


from typing import Dict, List, Any
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from torch.utils.data import DataLoader
from tqdm import tqdm
import re


from utils import Distractors


def tokenize_function(
    tokenizer: AutoTokenizer, examples: Dict, with_labels=True, text_column="text"
):
    """
    The batched examples will be tokenized. The labels can be provided for training,
    but for predicting the test dataset they can be excluded by setting `with_labels`
    to `False`. For the baseline model the prompt is a little different to instruct
    in which format we want the results. To use the text with the format prompt,
    you can change the `text_column` value from "text" to "baseline".
    """
    text = (
        [
            text + target
            for text, target in zip(examples[text_column], examples["target"])
        ]
        if with_labels
        else examples[text_column]
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
    """
    The list with distractors may not have an item at the given index. Since
    we do not want the program to crash, we return None instead of the value.

    The returned value is the same as the inner value in the list:

    `[T](list: List[T], index: int) -> Optional[T]`
    """
    length = len(list)

    if index < length:
        return list[index]
    else:
        return None


def generate_predictions(
    model: AutoModelForCausalLM,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    device: torch.device,
    is_baseline=False,
) -> List[Distractors]:
    """
    In batches of 8, the model will run on the preprocessed and tokenized dataset
    without the labels. The model uses top k sampling to generate (random) text.

    The generated text should include the distractors in the following format:
    Distractor 1: <answer>
    Distractor 2: <answer>
    Distractor 3: <answer>

    So with regex we retrieve the generated answers from the text. For the baseline
    model the index of the regex list of findings is increased with 3, because the
    prompt of the baseline includes text with how to format the answers. So for
    predicting with the baseline, the `is_baseline` parameter should be set to `True`
    """

    dataloader = DataLoader(
        dataset,  # type:ignore
        batch_size=8,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    model.eval()  # type:ignore set the model in evaluation mode

    outputs = []

    for batch in tqdm(dataloader):
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
            # the index needs to be increased by 3 if it is the baseline,
            # since in the prompt for the baseline model contains the pattern
            # already three times
            distractor1=safe_get(answers, 3 if is_baseline else 0),
            distractor2=safe_get(answers, 4 if is_baseline else 1),
            distractor3=safe_get(answers, 5 if is_baseline else 2),
        )

        distractors.append(distractor)

    return distractors


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
