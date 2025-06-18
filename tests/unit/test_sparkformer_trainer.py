import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from sparkformers.trainer import SparkformerTrainer


def test_sparkformer_trainer(spark_context):
    batch_size = 5
    epochs = 1

    dataset = load_dataset("gfigueroa/wikitext_processed")
    x = dataset["train"]["text"][:60]

    x_train, x_test = train_test_split(x, test_size=0.2)

    model_name = "hf-internal-testing/tiny-random-gptj"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_kwargs = {
        "max_length": 15,
        "padding": True,
        "truncation": True,
        "padding_side": "left",
    }
    data_collater = DataCollatorForLanguageModeling
    trainer = SparkformerTrainer(
        model,
        tokenizer,
        data_collater=data_collater,
        optimizer_fn=lambda params: torch.optim.AdamW(params, lr=5e-5),
        loader=AutoModelForCausalLM,
        tokenizer_kwargs=tokenizer_kwargs,
        num_workers=2,
    )
    trainer.train(
        x_train,
        x_train,
        epochs=epochs,
        training_args=dict(per_device_train_batch_size=batch_size),
    )
