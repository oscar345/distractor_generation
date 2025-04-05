from typing import cast
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from models import decoder
from options import Config
import utils


def train(config: Config):
    dataset = utils.load_dataset_from_disk(config)
    tokenizer = decoder.load_tokenizer(config.llama_model_name)
    device = utils.get_device()

    dataset = dataset.map(
        lambda examples: decoder.tokenize_function(tokenizer, examples), batched=True
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.llama_model_name,
        load_in_4bit=decoder.load_in_4bit(device),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded")
    model = get_peft_model(model, peft_config)
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=utils.get_results_directory_name(config),
        eval_strategy="epoch",
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        bf16=True,
        save_strategy="steps",
        save_total_limit=2,
        save_steps=350,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["evaluation"],
    )

    print("Training the model")
    trainer.train()

    print("Saving the model...")
    model.save_pretrained(utils.get_model_directory_name(config))
    print("Model saved")


def predict(config: Config):
    print("Loading dataset...")
    original_dataset = cast(Dataset, load_dataset(config.dataset_name, split="test"))
    test_dataset = cast(Dataset, utils.load_dataset_from_disk(config).get("test"))
    print("Dataset loaded")

    device = utils.get_device()
    tokenizer = decoder.load_tokenizer(config.llama_model_name)

    dataset = test_dataset.map(
        lambda examples: decoder.tokenize_function(
            tokenizer, examples, with_labels=False
        ),
        batched=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        utils.get_model_directory_name(config),
        load_in_4bit=decoder.load_in_4bit(device),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded")
    model = model.to(device)

    distractors = decoder.generate_predictions(model, dataset, tokenizer, device)
    predictions_dataset = utils.replace_distractors_in_dataset(
        original_dataset, distractors
    )
    predictions_dataset.save_to_disk(utils.get_predictions_directory_name(config))
