from datasets import Dataset, load_dataset
from transformers import (
    BertModel,
    BertTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)
import torch.nn as nn
import torch
from typing import Dict, cast
from options import Config
import utils
from utils import Distractors
from safetensors.torch import load_file

DECODER_MAX_GEN_LENGTH = 8
DECODER_N_HEADS = 8
DECODER_NUM_LAYERS = 6


class DecoderBertConfig(PretrainedConfig):
    def __init__(
        self,
        max_generation_length: int = 16,
        bert_model_name: str = "bert-base-uncased",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bert_model_name = bert_model_name
        self.hidden_dim = 768
        self.vocab_size = 30522
        self.max_generation_length = max_generation_length
        self.__name__ = "decoder-bert"


class DecoderBert(PreTrainedModel):
    def __init__(self, config, tokenizer):
        super().__init__(config)

        self.is_pretraining = False
        self.tokenizer = tokenizer

        self.bert = BertModel.from_pretrained(config.bert_model_name)

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
            torch.randn(self.config.max_generation_length, self.bert.config.hidden_size)
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
            loss_func = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_func(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

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
        # maybe add a better way to generate the tokens, since now they are not random
        return {"token_ids": token_ids}

    def _generate_mask(self, size):
        return torch.triu(torch.ones(size, size) * float("-inf"), diagonal=1).to(
            self.device
        )


def tokenize_function_normal(
    tokenizer: BertTokenizer, examples: Dict, with_labels=True
):
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

    if with_labels:
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

    return {
        "input_ids": encoder_input["input_ids"],
        "attention_mask": encoder_input["attention_mask"],
    }


def tokenize_function_pretrain(tokenizer: BertTokenizer, examples: Dict):
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

    return {"input_ids": input["input_ids"], "labels": input["input_ids"].clone()}  # type: ignore


def train(config: Config):
    dataset = utils.load_dataset_from_disk(config)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    device = utils.get_device()

    dataset_normal = dataset.map(
        lambda examples: tokenize_function_normal(tokenizer, examples), batched=True
    )
    dataset_pretrain = dataset.map(
        lambda examples: tokenize_function_pretrain(tokenizer, examples), batched=True
    )

    bert_config = DecoderBertConfig(
        max_generation_length=DECODER_MAX_GEN_LENGTH,
        bert_model_name=config.bert_model_name,
    )

    print("Loading model...")
    model = DecoderBert(bert_config, tokenizer)
    print("Model loaded")
    model.to(device)  # type: ignore

    print(f"Device of model is: {model.device}")

    training_args = TrainingArguments(
        output_dir=utils.get_results_directory_name(config),
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

    model.set_is_pretraining(True)

    trainer_pretrain = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_pretrain["train"],
        eval_dataset=dataset_pretrain["evaluation"],
    )

    print("Training pretraining model...")
    trainer_pretrain.train()

    model.set_is_pretraining(False)

    trainer_normal = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_normal["train"],
        eval_dataset=dataset_normal["evaluation"],
    )

    print("Training normal model...")
    trainer_normal.train()

    print("Saving model...")
    model.save_pretrained(utils.get_model_directory_name(config))
    print("Model saved")


def predict(config: Config):
    print("Loading dataset...")
    test_dataset = cast(Dataset, load_dataset(config.dataset_name, split="test"))
    print("Dataset loaded")

    device = utils.get_device()
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)

    dataset = test_dataset.map(
        lambda examples: tokenize_function_normal(
            tokenizer, examples, with_labels=False
        ),
        batched=True,
    )

    print("Loading model...")
    model = DecoderBert.from_pretrained(
        utils.get_model_directory_name(config), tokenizer=tokenizer
    )
    print("Model loaded")
    model = model.to(device)  # type: ignore

    print(f"\n\nDevice of model is: {model.device}\n\n")

    trainer = Trainer(model=model)
    predictions, _, _ = trainer.predict(dataset)  # type: ignore

    tokens = [
        Distractors(
            distractor1=tokenizer.decode(t1, skip_special_tokens=True),
            distractor2=tokenizer.decode(t2, skip_special_tokens=True),
            distractor3=tokenizer.decode(t3, skip_special_tokens=True),
        )
        for t1, t2, t3 in predictions
    ]

    predictions_dataset = utils.replace_distractors_in_dataset(test_dataset, tokens)
    predictions_dataset.save_to_disk(utils.get_predictions_directory_name(config))
