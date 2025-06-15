from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sparkformers.sparkformer import SparkFormer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np
import torch

batch_size = 20
epochs = 10

newsgroups = fetch_20newsgroups(subset="train")
x = newsgroups.data[:50]
y = newsgroups.target[:50]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.5)

model_name = "albert-base-v2"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(np.unique(y_encoded))
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_kwargs = {"padding": True, "truncation": True}

sparkformer_model = SparkFormer(
    model=model,
    tokenizer=tokenizer,
    loader=AutoModelForSequenceClassification,
    optimizer_fn=lambda params: torch.optim.AdamW(params, lr=5e-5),
    loss_fn=lambda: torch.nn.CrossEntropyLoss(),
    tokenizer_kwargs=tokenizer_kwargs,
    num_workers=2,
)

# perform distributed training
sparkformer_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

# perform distributed prediction
predictions = sparkformer_model.predict(x_test)

# review the predicted labels
print([np.argmax(pred) for pred in predictions])
