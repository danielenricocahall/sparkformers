import numpy as np
import torch
from pyspark import SparkFiles
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorMixin

from sparkformers.utils.model_utils import save_and_broadcast_model
from sparkformers.utils.rdd_utils import to_simple_rdd


class SparkformerTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        loader,
        optimizer_fn,
        tokenizer_kwargs=None,
        metrics=None,
        num_workers=None,
    ):
        self._master_network = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.master_optimizer = optimizer_fn
        self.master_metrics = metrics or {}
        self.num_workers = num_workers
        self.training_histories = []
        self.tokenizer_kwargs |= {"return_tensors": "pt"}

    def train(self, data: np.ndarray, labels: np.ndarray | None = None, **kwargs):
        rdd = to_simple_rdd(data, labels)
        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)
        batch_size = kwargs.get("batch_size", 32)
        epochs = kwargs.get("epochs", 1)

        for epoch in range(epochs):
            with save_and_broadcast_model(
                self._master_network, rdd.context
            ) as broadcast_dir:
                _ = TrainerWorker(
                    tokenizer=self.tokenizer,
                    args=TrainingArguments(
                        output_dir=broadcast_dir,
                        per_device_train_batch_size=batch_size,
                        num_train_epochs=1,
                        logging_steps=10,
                        evaluation_strategy="epoch",
                    ),
                    data_collater=DataCollatorForLanguageModeling,
                )


class TrainerWorker:
    def __init__(
        self,
        tokenizer,
        tokenizer_kwargs,
        args,
        data_collater: DataCollatorMixin,
        temp_dir,
        loader,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer.tokenizer_kwargs
        self.args = args
        self.data_collater = data_collater(tokenizer)
        self.temp_dir = temp_dir
        self.loader = loader
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, data_iterator):
        temp_dir = self.temp_dir.value
        model_path = SparkFiles.get(temp_dir)
        model = self.loader.from_pretrained(model_path).to(self.device)
        x_train, y_train = zip(*data_iterator)
        tokenized = self.tokenizer(list(x_train), **self.tokenizer_kwargs).to(
            self.device
        )
        trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            args=self.args,
            data_collator=self.data_collater,
            train_dataset=tokenized,
        )
        trainer.train()
        new_model = trainer.model
        print(new_model)
