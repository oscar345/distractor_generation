# A lot of the code between the three decoder models is similar. An example
# is the tokenizer function. Since that code is shared, it is stored in a
# separate file.

from typing import Dict
from transformers import AutoTokenizer


def tokenize(tokenizer: AutoTokenizer, examples: Dict, training: bool):
    """Tokenize the input text and target text when training. When not training, only
    tokenize the input text, since the target text is not available."""

    text = (
        [
            text + tokenizer.eos_token + target + tokenizer.eos_token  # type: ignore
            for text, target in zip(examples["text"], examples["target"])
        ]
        if training
        else [
            text + tokenizer.eos_token  # type: ignore
            for text in examples["text"]
        ]
    )

    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=768 + 64 + 2,
        return_tensors="pt",
    )  # type: ignore

    if not training:
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    sep_token_id = tokenizer.eos_token_id  # type: ignore
    input_ids = tokenized["input_ids"]

    # Find the second EOS (start of the target)
    labels = input_ids.clone()

    for i, ids in enumerate(input_ids):
        eos_indices = (ids == sep_token_id).nonzero(as_tuple=True)[0]

        if len(eos_indices) < 2:
            # If there's no second EOS, mask everything
            labels[i, :] = -100
        else:
            # Mask everything before the second EOS (including the second EOS itself).
            # Because of this the loss will only be calculated based on the other part
            # of the text the model generated (which is the target, and thus the
            # distractors).
            labels[i, : eos_indices[1] + 1] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }
