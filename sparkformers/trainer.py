import logging

import numpy as np
from datasets import Dataset
from pyspark import SparkFiles
from transformers import Trainer, TrainingArguments
from sparkformers.utils.model_utils import save_and_broadcast_model
from sparkformers.utils.rdd_utils import (
    to_simple_rdd,
    accumulate_model_parameters_and_history,
)
from sparkformers.utils.torch_utils import divide_by

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SparkformerTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        loader,
        optimizer_fn,
        tokenizer_kwargs=None,
        data_collater=None,
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
        self.data_collater = data_collater
        # self.tokenizer_kwargs |= {"return_tensors": "pt"}

    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray | None = None,
        training_args=None,
        **kwargs,
    ):
        rdd = to_simple_rdd(data, labels)
        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)
        epochs = kwargs.get("epochs", 1)

        for epoch in range(epochs):
            with save_and_broadcast_model(
                self._master_network, rdd.context
            ) as broadcast_dir:
                worker = TrainerWorker(
                    tokenizer=self.tokenizer,
                    args=training_args,
                    data_collater=self.data_collater,
                    loader=self.loader,
                    temp_dir=broadcast_dir,
                    tokenizer_kwargs=self.tokenizer_kwargs,
                )
                aggregated_params, history = rdd.mapPartitions(worker.train).reduce(
                    accumulate_model_parameters_and_history
                )
                averaged_params = divide_by(aggregated_params, self.num_workers)
                averaged_history = {k: v / self.num_workers for k, v in history.items()}
                self._master_network.load_state_dict(averaged_params)
                self.training_histories.append(averaged_history)
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {averaged_history['loss']:.4f}"
                )


class TrainerWorker:
    def __init__(
        self,
        tokenizer,
        tokenizer_kwargs,
        args,
        data_collater,
        temp_dir,
        loader,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.args = args
        self.data_collater = data_collater(tokenizer, mlm=False)
        self.temp_dir = temp_dir
        self.loader = loader
        self.device = "cpu"

    def train(self, data_iterator):
        temp_dir = self.temp_dir.value
        model_path = SparkFiles.get(temp_dir)
        model = self.loader.from_pretrained(model_path)
        x_train, y_train = zip(*data_iterator)
        tokenized = self.tokenizer(x_train, **self.tokenizer_kwargs)
        input_ids = tokenized["input_ids"]
        labels = [
            [
                -100 if token_id == self.tokenizer.pad_token_id else token_id
                for token_id in sequence
            ]
            for sequence in input_ids
        ]
        tokenized["labels"] = labels

        dataset = Dataset.from_dict(tokenized)

        trainer = Trainer(
            model=model,
            processing_class=self.tokenizer,
            args=TrainingArguments(**self.args, output_dir="./tmp", num_train_epochs=1),
            data_collator=self.data_collater,
            train_dataset=dataset,
        )
        print("TRAINER IS", trainer)
        trainer.train()
        print("Training completed.")
        new_model = trainer.model
        yield new_model.state_dict(), {"loss": trainer.state.log_history}
