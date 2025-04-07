import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Training Bert with decoder for false answer generation""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Importing dependencies""")
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        BertModel,
        BertTokenizer,
        PreTrainedModel,
        PretrainedConfig,
        Trainer,
        TrainingArguments,
        AutoModel,
    )
    from datasets import load_dataset, DatasetDict, Dataset
    import polars as pl
    from safetensors.torch import load_file

    return (
        AutoModel,
        BertModel,
        BertTokenizer,
        Dataset,
        DatasetDict,
        F,
        PreTrainedModel,
        PretrainedConfig,
        Trainer,
        TrainingArguments,
        load_dataset,
        load_file,
        mo,
        nn,
        pl,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""## Setting global constants""")
    return


@app.cell
def _():
    DECODER_N_HEADS = 8
    DECODER_NUM_LAYERS = 6
    DECODER_MAX_GEN_LENGTH = 8

    DATASET_NAME = "allenai/sciq"
    MODEL_NAME = "bert-base-uncased"
    return (
        DATASET_NAME,
        DECODER_MAX_GEN_LENGTH,
        DECODER_NUM_LAYERS,
        DECODER_N_HEADS,
        MODEL_NAME,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Importing the dataset and transforming it

        The dataset will be converted so there wont be three distractors per row, but a single distractor. To keep all distractors, the original row will be split into three rows. For the encoder decoder model we believe this to be beneficial, so it can focus on generating small sequences of tokens that don't vary too much in length. Those lengths may vary more when predicting multiple answer options at once.

        So if the current dataset looks like this: `A B C1 C2 C3`, it will become like this:

        ```
        A B C1
        A B C2
        A B C3
        ```
        """
    )
    return


@app.cell
def _(DATASET_NAME, DatasetDict, load_dataset):
    dataset_dict: DatasetDict = load_dataset(DATASET_NAME)
    return (dataset_dict,)


@app.cell
def _(dataset_dict, pl):
    df = pl.concat(
        [
            pl.DataFrame(dataset.to_pandas()).with_columns(
                [pl.lit(split).alias("split")]
            )
            for split, dataset in dataset_dict.items()
        ]
    )

    df = df.unpivot(
        index=["question", "correct_answer", "support", "split"],
        on=["distractor1", "distractor2", "distractor3"],
        variable_name="x",
        value_name="distractor",
    )

    df = df.drop(["x"])

    df
    return (df,)


@app.cell
def _(Dataset, DatasetDict, df, pl):
    # Converting the polars dataframe back to a DatasetDict from huggingface
    dataset = DatasetDict(
        {
            "train": Dataset.from_polars(df.filter(pl.col("split") == "train")),
            "validation": Dataset.from_polars(
                df.filter(pl.col("split") == "validation")
            ),
            "test": Dataset.from_polars(df.filter(pl.col("split") == "test")),
        }
    )
    return (dataset,)


@app.cell
def _(mo):
    mo.md(r"""## Tokenize dataset""")
    return


@app.cell
def _(BertTokenizer, MODEL_NAME):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    return (tokenizer,)


@app.cell
def _(DECODER_MAX_GEN_LENGTH, tokenizer):
    def tokenize_function_normal(examples):
        text = [
            f"[CLS] context: {c} [SEP] question: {q} [SEP] correct answer: {a} [SEP]"
            for c, q, a in zip(
                examples["support"],
                examples["question"],
                examples["correct_answer"],
            )
        ]

        encoder_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        decoder_input = tokenizer(
            examples["distractor"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=DECODER_MAX_GEN_LENGTH,
        )

        return {
            "input_ids": encoder_input["input_ids"],
            "attention_mask": encoder_input["attention_mask"],
            "labels": decoder_input["input_ids"],
        }

    def tokenize_function_pretrain(examples):
        text = [
            f"[CLS] {c} [SEP] {q} [SEP] {a} [SEP]"
            for c, q, a in zip(
                examples["support"],
                examples["question"],
                examples["correct_answer"],
            )
        ]

        input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        return {"input_ids": input["input_ids"], "labels": input["input_ids"].clone()}

    return tokenize_function_normal, tokenize_function_pretrain


@app.cell
def _(dataset, tokenize_function_normal, tokenize_function_pretrain):
    dataset_normal = dataset.map(tokenize_function_normal, batched=True)
    dataset_pretrain = dataset.map(tokenize_function_pretrain, batched=True)
    return dataset_normal, dataset_pretrain


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Defining the custom BERT head

        We will create a custom BERT head that has a `TransformerDecoder` layer so that from the inputs from the BERT model, we can create a sequence of text for one of the answer options.
        """
    )
    return


@app.cell
def _(PretrainedConfig):
    class DecoderBertConfig(PretrainedConfig):
        def __init__(self, max_generation_length: int = 16, **kwargs):
            super().__init__(**kwargs)
            self.bert_model_name = "bert-base-uncased"
            self.hidden_dim = 768
            self.vocab_size = 30522
            self.max_generation_length = max_generation_length
            self.__name__ = "decoder-bert"

    return (DecoderBertConfig,)


@app.cell
def _():
    # class Decoder(PreTrainedModel):
    #     def __init__(self):
    return


@app.cell
def _(
    BertModel,
    DECODER_NUM_LAYERS,
    DECODER_N_HEADS,
    DecoderBertConfig,
    MODEL_NAME,
    PreTrainedModel,
    load_file,
    nn,
    torch,
):
    class DecoderBert(PreTrainedModel):
        def __init__(self, config, tokenizer):
            super().__init__(config)

            self.is_pretraining = False
            self.tokenizer = tokenizer

            self.bert = BertModel.from_pretrained(MODEL_NAME)

            self.decoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.bert.config.hidden_size,
                    nhead=DECODER_N_HEADS,
                    batch_first=True,
                ),
                num_layers=DECODER_NUM_LAYERS,
            )

            self.output = nn.Linear(
                self.bert.config.hidden_size, self.bert.config.vocab_size
            )

            self.query_embeddings = nn.Parameter(
                torch.randn(
                    self.config.max_generation_length, self.bert.config.hidden_size
                )
            )

        @classmethod
        def from_pretrained(cls, model_path, tokenizer):
            """Custom method to load model with tokenizer"""
            config = DecoderBertConfig.from_pretrained(model_path)
            model = cls(config, tokenizer)  # Instantiate the model

            state_dict = load_file(f"{model_path}/model.safetensors")
            model.load_state_dict(state_dict)

            return model

        def set_is_pretraining(self, value):
            self.is_pretraining = value

        def forward(self, input_ids, attention_mask=None, labels=None):
            input_ids = input_ids.to(self.device)
            if self.is_pretraining:
                logits = self._forward_pretrain(input_ids, labels)
            else:
                logits = self._forward(input_ids, attention_mask, labels)
            return self._generate_return_values(logits, labels)

        def _forward_pretrain(self, input_ids, labels):
            embeddings = self.bert.embeddings(input_ids)

            sequence_length = input_ids.size(1)
            tgt_mask = self._generate_mask(sequence_length)

            decoder_output = self.decoder(
                tgt=embeddings, memory=embeddings, tgt_mask=tgt_mask
            )

            logits = self.output(decoder_output)

            return logits

        def _forward(self, input_ids, attention_mask, labels):
            attention_mask = attention_mask.to(self.device)
            memory = self.bert(input_ids, attention_mask).last_hidden_state

            batch_size = input_ids.size(0)
            tgt = (
                self.query_embeddings.unsqueeze(1)
                .expand(-1, batch_size, -1)
                .permute(1, 0, 2)
            )

            tgt_mask = self._generate_mask(self.config.max_generation_length)

            output_decoder = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

            logits = self.output(output_decoder)

            return logits

        def _generate_return_values(self, logits, labels):
            if labels is not None:
                loss_func = nn.CrossEntropyLoss(
                    ignore_index=self.tokenizer.pad_token_id
                )
                loss = loss_func(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )

                return {"logits": logits, "loss": loss}

            batch_size, seq_len, vocab_size = logits.size()
            sampled_sequences = []

            for _ in range(3):
                # Get top-k logits and indices
                top_k_logits, top_k_indices = torch.topk(
                    logits, 50, dim=-1
                )  # [batch, seq_len, k]
                top_k_probs = torch.softmax(top_k_logits, dim=-1)

                # Sample from top-k for each token position
                sampled = torch.distributions.Categorical(
                    top_k_probs
                ).sample()  # [batch, seq_len]

                # Gather actual token ids using sampled indices
                sampled_tokens = torch.gather(
                    top_k_indices, dim=-1, index=sampled.unsqueeze(-1)
                ).squeeze(-1)
                sampled_sequences.append(sampled_tokens)

            # Shape: [batch, num_samples, seq_len]
            token_ids = torch.stack(sampled_sequences, dim=1)

            return {"token_ids": token_ids}

        def _generate_mask(self, size):
            return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1).to(
                self.device
            )

    return (DecoderBert,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Setting training arguments

        Here settings such as config of the custom BERT and the device for training are set.
        """
    )
    return


@app.cell
def _(torch):
    device = None

    if torch.backends.mps.is_available():
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
def _(
    DECODER_MAX_GEN_LENGTH,
    DecoderBert,
    DecoderBertConfig,
    TrainingArguments,
    device,
    tokenizer,
):
    config = DecoderBertConfig(max_generation_length=DECODER_MAX_GEN_LENGTH)
    model = DecoderBert(config, tokenizer)
    model.to(device)

    print(f"\n\nDevice of model is: {model.device}\n\n")

    args = TrainingArguments(
        output_dir="./results_mine_pretrained",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        optim="adamw_torch_fused",
        bf16=True,
        save_strategy="steps",
        save_total_limit=2,
        save_steps=1000,
    )
    return args, config, model


@app.cell
def _(mo):
    mo.md(r"""## Train the model""")
    return


@app.cell
def _(Trainer, args, dataset_pretrain, model):
    model.set_is_pretraining(True)

    trainer_pretrain = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_pretrain["train"],
        eval_dataset=dataset_pretrain["validation"],
    )

    trainer_pretrain.train()
    return (trainer_pretrain,)


@app.cell
def _(Trainer, args, dataset_normal, model):
    model.set_is_pretraining(False)

    trainer_normal = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_normal["train"],
        eval_dataset=dataset_normal["validation"],
    )

    trainer_normal.train()
    return (trainer_normal,)


@app.cell
def _(mo):
    mo.md(r"""## Making predictions""")
    return


@app.cell
def _(dataset_normal, trainer_normal):
    output = trainer_normal.predict(dataset_normal["test"].remove_columns(["labels"]))
    predictions, label_ids, metrics = output
    return label_ids, metrics, output, predictions


@app.cell
def _(dataset, metrics, predictions, tokenizer):
    print(predictions, metrics)
    d_predictions = [
        tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions
    ]
    for ii, text_ in enumerate(d_predictions):  # Show first 5 predictions
        print(dataset["test"][ii])
        print(f"Predicted: {text_}")
    return d_predictions, ii, text_


@app.cell
def _(mo):
    mo.md(r"""## Making multiple predictions""")
    return


@app.cell
def _(DecoderBert, tokenizer):
    model_trained = DecoderBert.from_pretrained(
        "./results_mine_pretrained/checkpoint-10950", tokenizer=tokenizer
    )
    return (model_trained,)


@app.cell
def _(Trainer, model_trained):
    trainer_trained = Trainer(model=model_trained)
    return (trainer_trained,)


@app.cell
def _(dataset_normal, trainer_trained):
    output_trained = trainer_trained.predict(
        dataset_normal["test"].remove_columns(["labels"])
    )
    return (output_trained,)


@app.cell
def _(dataset, output_trained, tokenizer):
    predictions_trained, _, _ = output_trained

    d_predictions_trained = [
        (
            tokenizer.decode(p1, skip_special_tokens=True),
            tokenizer.decode(p2, skip_special_tokens=True),
            tokenizer.decode(p3, skip_special_tokens=True),
        )
        for p1, p2, p3 in predictions_trained
    ]
    for iii, text__ in enumerate(d_predictions_trained):  # Show first 5 predictions
        print(dataset["test"][iii])
        print(f"Predicted: {text__}")
    return d_predictions_trained, iii, predictions_trained, text__


if __name__ == "__main__":
    app.run()
