from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sparkformers.sparkformer import SparkFormer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

batch_size = 20
epochs = 10

dataset = load_dataset("ag_news")
x = dataset["train"]["text"]  # ty: ignore[possibly-unbound-implicit-call]


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
    tokenizer_kwargs=tokenizer_kwargs,
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
