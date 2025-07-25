import numpy as np
import pytest
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
)
import torch

from sparkformers.sparkformer import Sparkformer


@pytest.mark.parametrize("num_workers", [1, 2])
def test_sequence_classification(spark_context, num_workers):
    batch_size = 5
    epochs = 1

    dataset = load_dataset("ag_news")
    x = dataset["train"]["text"][:50]
    y = dataset["train"]["label"][:50]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.5)

    model_name = "prajjwal1/bert-tiny"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(np.unique(y_encoded)),
        problem_type="single_label_classification",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}

    sparkformer_model = Sparkformer(
        model=model,
        tokenizer=tokenizer,
        loader=AutoModelForSequenceClassification,
        optimizer_fn=lambda params: torch.optim.AdamW(params, lr=5e-5),
        tokenizer_kwargs=tokenizer_kwargs,
        num_workers=num_workers,
    )

    sparkformer_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Inference
    predictions = sparkformer_model.predict(x_test)
    sparkformer_model._master_network.eval()
    inputs = tokenizer(x_test, **tokenizer_kwargs)
    with torch.no_grad():
        expected = (
            sparkformer_model._master_network(**{k: v for k, v in inputs.items()})
            .logits.cpu()
            .numpy()
        )
    for pred, exp in zip(predictions, expected):
        assert np.allclose(pred, exp, atol=0.1)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_generation(spark_context, num_workers):
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

    sparkformer_model = Sparkformer(
        model=model,
        tokenizer=tokenizer,
        loader=AutoModelForCausalLM,
        optimizer_fn=lambda params: torch.optim.AdamW(params, lr=5e-5),
        tokenizer_kwargs=tokenizer_kwargs,
        num_workers=num_workers,
    )

    sparkformer_model.train(x_train, epochs=epochs, batch_size=batch_size)

    generations = sparkformer_model.generate(
        x_test, max_new_tokens=10, num_return_sequences=1
    )
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in generations
    ]

    # Reference output
    sparkformer_model._master_network.eval()
    inputs = tokenizer(x_test, **tokenizer_kwargs)
    with torch.no_grad():
        expected_outputs = sparkformer_model._master_network.generate(
            **inputs, max_new_tokens=10, num_return_sequences=1
        )
    expected_texts = [
        tokenizer.decode(out, skip_special_tokens=True) for out in expected_outputs
    ]

    assert len(generated_texts) == len(expected_texts)


@pytest.mark.parametrize("num_workers", [1, 2])
def test_training_huggingface_token_classification(spark_context, num_workers: int):
    batch_size = 5
    epochs = 2
    model_name = "hf-internal-testing/tiny-bert-for-token-classification"

    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = load_dataset("conll2003", split="train[:5%]", trust_remote_code=True)
    dataset = dataset.map(tokenize_and_align_labels, batched=True)

    x = dataset["tokens"]
    y = dataset["labels"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "is_split_into_words": True,
    }

    sparkformer_model = Sparkformer(
        model=model,
        tokenizer=tokenizer,
        loader=AutoModelForTokenClassification,
        optimizer_fn=lambda params: torch.optim.AdamW(params, lr=5e-5),
        tokenizer_kwargs=tokenizer_kwargs,
        num_workers=num_workers,
    )

    sparkformer_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

    inputs = tokenizer(x_test, **tokenizer_kwargs)
    distributed_preds = sparkformer_model(**inputs)
    sparkformer_model._master_network.eval()
    with torch.no_grad():
        outputs = sparkformer_model._master_network(**{k: v for k, v in inputs.items()})
    reference_preds = outputs.logits.detach().cpu().numpy()

    for dpred, rpred in zip(distributed_preds, reference_preds):
        assert np.allclose(dpred, rpred, atol=0.1)
