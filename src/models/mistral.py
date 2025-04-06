from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from models import decoder
from options import Config
from datasets import Dataset, load_dataset
from typing import cast
import utils
import torch


def train(config: Config):
    dataset = utils.load_dataset_from_disk(config)
    tokenizer = decoder.load_tokenizer(config.mistral_model_name)
    device = utils.get_device()

    dataset = dataset.map(
        lambda examples: decoder.tokenize_function(tokenizer, examples), batched=True
    )

    # Lora can help with reducing the amount of parameters that are adjusted during
    # finetuning. This increases the speed with which we can train a large model such
    # as Llama (even though we use the smallest, it takes a long time on habrok to
    # finetune).
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.mistral_model_name,
        load_in_4bit=decoder.load_in_4bit(device),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("Model loaded")
    model = get_peft_model(model, peft_config)
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir="./results_mistral",
        eval_strategy="epoch",
        per_device_eval_batch_size=2,  # a lower batch size to reduce memory
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,  # to make the training more similar to the llama model, added a gradient accumulation of 2 (2 * 2 = 4)
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=True,
        save_strategy="steps",
        save_total_limit=2,
        save_steps=350,
        optim="adamw_bnb_8bit",  # Uses 8-bit AdamW optimizer which uses less memory
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,  # type:ignore
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
    tokenizer = decoder.load_tokenizer(config.mistral_model_name)

    dataset = test_dataset.map(
        lambda examples: decoder.tokenize_function(
            tokenizer, examples, with_labels=False
        ),
        batched=True,
    )

    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.mistral_model_name,
        load_in_4bit=decoder.load_in_4bit(device),
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(
        base_model, utils.get_model_directory_name(config)
    )
    print("Model loaded")

    model = model.to(device)

    distractors = decoder.generate_predictions(
        model,  # type:ignore
        dataset.select(range(8)),
        tokenizer,
        device,
    )
    predictions_dataset = utils.replace_distractors_in_dataset(
        original_dataset, distractors
    )
    predictions_dataset.save_to_disk(utils.get_predictions_directory_name(config))
