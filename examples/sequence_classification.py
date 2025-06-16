from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch import softmax

from sparkformers.sparkformer import SparkFormer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np
import torch

batch_size = 16
epochs = 20


dataset = load_dataset("ag_news")
x = dataset["train"]["text"][:2000]  # ty: ignore[possibly-unbound-implicit-call]
y = dataset["train"]["label"][:2000]  # ty: ignore[possibly-unbound-implicit-call]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

model_name = "prajjwal1/bert-tiny"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(np.unique(y)),
    problem_type="single_label_classification",
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}

sparkformer_model = SparkFormer(
    model=model,
    tokenizer=tokenizer,
    loader=AutoModelForSequenceClassification,
    optimizer_fn=lambda params: torch.optim.AdamW(params, lr=2e-4),
    tokenizer_kwargs=tokenizer_kwargs,
    num_workers=2,
)

# perform distributed training
sparkformer_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

# perform distributed inference
predictions = sparkformer_model.predict(x_train)
for i, pred in enumerate(predictions[:10]):
    probs = softmax(torch.tensor(pred), dim=-1)
    print(f"Example {i}: probs={probs.numpy()}, predicted={probs.argmax().item()}")

# review the predicted labels
print([int(np.argmax(pred)) for pred in predictions])
