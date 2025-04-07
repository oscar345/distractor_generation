import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Training Llama for false answer generation""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Importing dependencies""")
    return


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset, DatasetDict, Dataset
    from transformers import (
        Trainer,
        TrainingArguments,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    import polars as pl
    import torch
    from peft import LoraConfig, get_peft_model

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        Dataset,
        DatasetDict,
        LoraConfig,
        Trainer,
        TrainingArguments,
        get_peft_model,
        load_dataset,
        mo,
        pl,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""## Setting global constants""")
    return


@app.cell
def _():
    DATASET_NAME = "allenai/sciq"
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    return DATASET_NAME, MODEL_NAME


@app.cell
def _(mo):
    mo.md(r"""## Importing data and transforming it""")
    return


@app.cell
def _(DATASET_NAME, DatasetDict, load_dataset):
    dataset_dict: DatasetDict = load_dataset(DATASET_NAME)
    return (dataset_dict,)


@app.cell
def _(dataset_dict, pl):
    df = pl.concat(
        [
            dataset.to_polars().with_columns([pl.lit(split).alias("split")])
            for split, dataset in dataset_dict.items()
        ]
    )

    # We are creating two new columns: the text column with the input for the model such as the
    # the support, question and answer. The target for the model will be the three distractors.
    # With these transformations we can easily tokenize the data inside the tokenize function.
    df = df.with_columns(
        [
            pl.format(
                "Question: {}\nCorrect Answer: {}\nSupport: {}\nGenerate 3 false answer options:\n",
                pl.col("question"),
                pl.col("correct_answer"),
                pl.col("support"),
            ).alias("text"),
            pl.format(
                "Distractor 1: {}\nDistractor 2: {}\nDistractor 3: {}",
                pl.col("distractor1"),
                pl.col("distractor2"),
                pl.col("distractor3"),
            ).alias("target"),
        ]
    )

    # the following columns are no longer needed since they are used in the creation of the new columns
    df = df.drop(
        [
            "distractor1",
            "distractor2",
            "distractor3",
            "question",
            "correct_answer",
            "support",
        ]
    )

    df
    return (df,)


@app.cell
def _(Dataset, DatasetDict, df, pl):
    # Converting the dataframe back into a datasetdict so it work with the other huggingface
    # libraries.
    dataset_ = DatasetDict(
        {
            "train": Dataset.from_polars(df.filter(pl.col("split") == "train")),
            "validation": Dataset.from_polars(
                df.filter(pl.col("split") == "validation")
            ),
            "test": Dataset.from_polars(df.filter(pl.col("split") == "test")),
        }
    )
    return (dataset_,)


@app.cell
def _(mo):
    mo.md(r"""## Tokenize the dataset""")
    return


@app.cell
def _(AutoTokenizer, MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    # since we need padding but there is no padding for the Llama model for some reason
    # we can use the EOS token to replace it.
    tokenizer.pad_token = tokenizer.eos_token
    return (tokenizer,)


@app.cell
def _(tokenizer):
    def tokenize_function(examples, with_labels=True):
        """"""
        text = (
            [
                text + target
                for text, target in zip(examples["text"], examples["target"])
            ]
            if with_labels
            else examples["text"]
        )

        tokenized = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=768 + 64,
            return_tensors="pt",
        )

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

    return (tokenize_function,)


@app.cell
def _(DatasetDict, dataset_, tokenize_function):
    dataset = DatasetDict(
        {
            "train": dataset_["train"].map(tokenize_function, batched=True),
            "validation": dataset_["validation"].map(tokenize_function, batched=True),
            "test": dataset_["test"].map(
                lambda examples: tokenize_function(examples, with_labels=False),
                batched=True,
            ),
        }
    )
    return (dataset,)


@app.cell
def _(mo):
    mo.md(r"""## Setting the device""")
    return


@app.cell
def _(torch):
    device = None

    if torch.backends.mps.is_available():
        # Habrok's interactive GPU's are great, but it can be nice to develop the code locally
        # if Habrok is acting weird. For example: loading the Llama model with the interactive
        # did not seem possible. MPS makes it possible to train with models on macbooks.
        device = torch.device("mps")
        print("MPS is available!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available!")
    else:
        device = torch.device("cpu")
        print("MPS and CUDA are not available. Using CPU.")
    return (device,)


@app.cell
def _(mo):
    mo.md(r"""## Create model and training arguments""")
    return


@app.cell
def _(LoraConfig):
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
    )
    return (peft_config,)


@app.cell
def _(AutoModelForCausalLM, MODEL_NAME, device, torch):
    load_in_4bit = device == "cuda"

    # adding additional arguments, so we can load the model with a smaller memory footprint
    model_ = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=load_in_4bit,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return load_in_4bit, model_


@app.cell
def _(device, get_peft_model, model_, peft_config):
    # The model needs to be converted to work with the lora and the device we are training
    # with: cuda and mps are both faster to train with than the cpu.
    model = get_peft_model(model_, peft_config)
    model = model.to(device)
    return (model,)


@app.cell
def _(TrainingArguments):
    # Setting the training arguments. We use bf16, so the computations the GPU has to
    # perform are less expensive.
    args = TrainingArguments(
        output_dir="./results_llama_2",
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
    return (args,)


@app.cell
def _(mo):
    mo.md(r"""## Training the model""")
    return


@app.cell
def _(Trainer, args, dataset, model, tokenizer):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _():
    # model_trained = AutoModelForCausalLM.from_pretrained(
    #     "./results_llama_2/checkpoint-189",
    #     load_in_4bit=load_in_4bit,
    #     device_map="auto",
    #     torch_dtype=torch.float16,
    # )
    return


@app.cell
def _():
    # from torch.utils.data import DataLoader
    # from transformers import default_data_collator

    # def test():
    #     dataloader = DataLoader(
    #         dataset["test"].select(range(10)),
    #         batch_size=4,
    #         shuffle=False,
    #         collate_fn=default_data_collator  # pads input_ids & attention_mask
    #     )

    #     # Step 2: Run generation
    #     model.eval()
    #     generated_texts = []

    #     for batch in dataloader:
    #         # Move batch to GPU if available
    #         batch = {k: v.to(model.device) for k, v in batch.items()}

    #         with torch.no_grad():
    #             outputs = model.generate(
    #                 input_ids=batch["input_ids"],
    #                 attention_mask=batch["attention_mask"],
    #                 max_new_tokens=64,
    #                 do_sample=True,
    #                 top_k=50
    #             )

    #         # Step 3: Decode the output tokens
    #         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #         generated_texts.extend(decoded)

    #     # Step 4: Print results
    #     for i, text in enumerate(generated_texts[:5]):
    #         print(f"[{i}] Generated: {text}")
    return


@app.cell
def _():
    # test()
    return


if __name__ == "__main__":
    app.run()
