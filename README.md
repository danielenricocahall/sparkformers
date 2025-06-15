# Overview
Distributed training of [Transformers](https://github.com/huggingface/transformers) models on Spark!

# Motivation
Derived from [Elephas](https://github.com/danielenricocahall/elephas), however with [HuggingFace removing support for Tensorflow](https://www.linkedin.com/posts/leonidboytsov_wow-the-huggingface-library-is-dropping-activity-7339003651773915137-mmrV#:~:text=I%20have%20bittersweet%20news%20to,even%20if%20outside%20of%20PyTorch.), I decided to spin some of the logic off into its own separate project, and also rework the paradigm to support the [Torch](https://pytorch.org/) backend!

# Example

## Autoregressive Language Model Training
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sparkformers.sparkformer import SparkFormer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
batch_size = 20
epochs = 10

newsgroups = fetch_20newsgroups(subset="train")
x = newsgroups.data

x_train, x_test = train_test_split(x, test_size=0.2)

model_name = "sshleifer/tiny-gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer_kwargs = {
    "max_length": 15,
    "padding": True,
    "truncation": True,
    "padding_side": "left",
}

sparkformer_model = SparkFormer(
    model=model,
    tokenizer=tokenizer,
    loader=AutoModelForCausalLM,
    optimizer_fn=lambda params: torch.optim.AdamW(params, lr=5e-5),
    loss_fn=lambda: torch.nn.CrossEntropyLoss(),
    tokenizer_kwargs=tokenizer_kwargs
)

# perform distributed training
sparkformer_model.train(x_train, epochs=epochs, batch_size=batch_size)

# perform distributed generation
generations = sparkformer_model.generate(
    x_test, max_new_tokens=10, num_return_sequences=1
)
# decode the generated texts
generated_texts = [
    tokenizer.decode(output, skip_special_tokens=True) for output in generations
]

for i, text in enumerate(generated_texts):
    print(f"Original text {i}: {x_test[i]}")
    print(f"Generated text {i}: {text}")
```

## Sequence Classification
```python
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
```

> 💡 Interested in contributing? Check out the [Local Development & Contributions Guide](https://github.com/danielenricocahall/sparkformers/blob/main/CONTRIBUTING.md).