# Overview
![img.png](https://raw.githubusercontent.com/danielenricocahall/sparkformers/main/logo.png)
Welcome to Sparkformers, where we offer distributed training of [Transformers](https://github.com/huggingface/transformers) models on [Spark](https://spark.apache.org/)!

# Motivation / Purpose
Derived from [Elephas](https://github.com/danielenricocahall/elephas), however with [HuggingFace removing support for Tensorflow](https://www.linkedin.com/posts/leonidboytsov_wow-the-huggingface-library-is-dropping-activity-7339003651773915137-mmrV#:~:text=I%20have%20bittersweet%20news%20to,even%20if%20outside%20of%20PyTorch.), I decided to spin some of the logic off into its own separate project, and also rework the paradigm to support the [Torch](https://pytorch.org/) backend! The purpose of this project is to serve as an experimental backend for distributed training that may be more developergonomic compared to other solutions such as [Ray](https://docs.ray.io/en/latest/train/train.html). Additionally, `Sparkformers` offers the capability for distributed prediction, model calling, and generation (for causal/autoregressive models). 

The project is currently in a beta/experimental state. While not yet production ready, I invite you to experiment, provide feedback, and/or even contribute!

# Approach

**Training**: The current architecture utilizes [federated averaging (FedAvg)](https://en.wikipedia.org/wiki/Federated_learning), meaning that each executor is trained on a subset of data, and the model weights are averaged across all executors after each epoch. The original model is then updated with the averaged weights, and then the process is repeated for the next epoch.

**Inference**:  The input data is distributed across the executors, and each executor performs the inference on its subset of data. The results are then collected and returned to the driver.

**Generation**: Same as above, but with the `generate` method of the model.


# Installation
To install, you can simply run:
```bash
pip install sparkformers
````

(or `uv add`, `poetry add`, etc. with whichever project dependency management tool you may use).

# Examples
Note that all examples are also available in the [examples directory](https://github.com/danielenricocahall/sparkformers/tree/main/examples).

## Autoregressive (Causal) Language Model Training and Inference
```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sparkformers.sparkformer import SparkFormer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

batch_size = 16
epochs = 30

dataset = load_dataset("ag_news")
x = dataset["train"]["text"][:2000] 

x_train, x_test = train_test_split(x, test_size=0.1)

model_name = "sshleifer/tiny-gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer_kwargs = {
    "max_length": 50,
    "padding": True,
    "truncation": True,
    "padding_side": "left",
}

sparkformer_model = SparkFormer(
    model=model,
    tokenizer=tokenizer,
    loader=AutoModelForCausalLM,
    optimizer_fn=lambda params: torch.optim.AdamW(params, lr=1e-3),
    tokenizer_kwargs=tokenizer_kwargs,
    num_workers=2,
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
x = dataset["train"]["text"][:2000]
y = dataset["train"]["label"][:2000]

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
predictions = sparkformer_model.predict(x_test)
for i, pred in enumerate(predictions[:10]):
    probs = softmax(torch.tensor(pred), dim=-1)
    print(f"Example {i}: probs={probs.numpy()}, predicted={probs.argmax().item()}")

# review the predicted labels
print([int(np.argmax(pred)) for pred in predictions])
```

## Token Classification (NER)
```python
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
epochs = 1
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

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
    tokenizer_kwargs=tokenizer_kwargs,
    num_workers=2,
)

sparkformer_model.train(x_train, y_train, epochs=epochs, batch_size=batch_size)

inputs = tokenizer(x_test, **tokenizer_kwargs)
distributed_preds = sparkformer_model(**inputs)
print([int(np.argmax(x)) for x in np.squeeze(distributed_preds)])

 ```

# TODO
- [ ] Add support for distributed training of other model types (e.g., image classification, object detection, etc.)
- [ ] Potentially add asynchronous support for training
- [ ] Expose more configuration options
- [ ] Consider simplifying the API further (e.g; builder pattern, providing the model string and push loader logic inside the `SparkFormer` class, etc.)
- [ ] Support Tensorflow/Keras models for completeness (potentially taking similar approach as `transformers` where each class is prefixed with `TF` - it would essentially be copying the old logic from Elephas)
> 💡 Interested in contributing? Check out the [Local Development & Contributions Guide](https://github.com/danielenricocahall/sparkformers/blob/main/CONTRIBUTING.md).