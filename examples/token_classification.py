from sklearn.model_selection import train_test_split
from sparkformers.sparkformer import SparkFormer
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from datasets import load_dataset
import numpy as np
import torch

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

sparkformer_model = SparkFormer(
    model=model,
    tokenizer=tokenizer,
    loader=AutoModelForTokenClassification,
    optimizer_fn=lambda params: torch.optim.AdamW(params, lr=5e-5),
    loss_fn=lambda: torch.nn.CrossEntropyLoss(),
    tokenizer_kwargs=tokenizer_kwargs,
    num_workers=2,
)

sparkformer_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

inputs = tokenizer(x_test, **tokenizer_kwargs, return_tensors="pt")
distributed_preds = sparkformer_model(**inputs)
print([int(np.argmax(x)) for x in np.squeeze(distributed_preds)])
