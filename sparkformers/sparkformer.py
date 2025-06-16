import shutil
import tempfile
import logging
import uuid
from functools import partial
from pathlib import Path
from typing import List, Callable, Iterable

import numpy as np
import torch
from pyspark.core.rdd import RDD
from pyspark.core.files import SparkFiles
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
)

from sparkformers.utils.rdd_utils import to_simple_rdd
from sparkformers.utils.torch_utils import add_params, average_states
from sparkformers.utils.hf_utils import pad_labels

logger = logging.getLogger(__name__)


class SparkFormer:
    def __init__(
        self,
        model,
        tokenizer,
        loader,
        optimizer_fn,
        tokenizer_kwargs=None,
        metrics=None,
        custom_objects=None,
        num_workers=None,
    ):
        self._master_network = model
        self.tokenizer = tokenizer
        self.loader = loader
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.master_optimizer = optimizer_fn
        self.master_metrics = metrics or {}
        self.custom_objects = custom_objects or {}
        self.num_workers = num_workers
        self.training_histories = []
        self.tokenizer_kwargs |= {"return_tensors": "pt"}

    def train(self, data: np.ndarray, labels: np.ndarray | None = None, **kwargs):
        rdd = to_simple_rdd(data, labels)
        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)
        optimizer_fn = self.master_optimizer
        metrics = self.master_metrics
        batch_size = kwargs.get("batch_size", 32)
        epochs = kwargs.get("epochs", 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            self._master_network.save_pretrained(temp_dir)
            rdd.context.addFile(temp_dir, recursive=True)
            broadcast_dir = rdd.context.broadcast(temp_dir)

            worker = SparkFormerWorker(
                optimizer_fn,
                metrics,
                temp_dir=broadcast_dir,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                loader=self.loader,
                batch_size=batch_size,
                epochs=epochs,
            )
            training_outcomes = rdd.mapPartitions(worker.train).collect()
            self.training_histories = [history for _, history in training_outcomes]
            model_states = [
                add_params(self._master_network.state_dict(), delta)
                for delta, _ in training_outcomes
            ]
            new_state = average_states(model_states)
            self._master_network.load_state_dict(new_state)

    def predict(self, data: Iterable) -> List[np.ndarray]:
        rdd = to_simple_rdd(data)
        tokenizer = self.tokenizer
        loader = self.loader
        tokenizer_kwargs = self.tokenizer_kwargs

        with tempfile.TemporaryDirectory() as temp_dir:
            self._master_network.save_pretrained(temp_dir)
            rdd.context.addFile(temp_dir, recursive=True)
            broadcast_dir = rdd.context.broadcast(temp_dir)

            def _predict(partition):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                temp_dir_ = broadcast_dir.value
                model_path = SparkFiles.get(temp_dir_)
                model = loader.from_pretrained(model_path).to(device)
                model.eval()
                predictions = []
                with torch.no_grad():
                    for batch in partition:
                        inputs = tokenizer(batch, **tokenizer_kwargs)
                        outputs = model(**inputs)
                        predictions.extend(outputs.logits.detach().cpu().numpy())
                return predictions

            def _predict_with_indices(partition):
                data, indices = zip(*partition)
                predictions = _predict(data)
                return zip(predictions, indices)

            return self._call_and_collect(rdd, _predict, _predict_with_indices)

    def generate(self, data: Iterable, **kwargs) -> List[np.ndarray]:
        if self.loader.__name__ == AutoModelForSequenceClassification.__name__:
            raise ValueError(
                "This method is only for causal language models, not classification models."
            )
        rdd = to_simple_rdd(data)
        tokenizer = self.tokenizer
        loader = self.loader
        tokenizer_kwargs = self.tokenizer_kwargs

        with tempfile.TemporaryDirectory() as temp_dir:
            self._master_network.save_pretrained(temp_dir)
            rdd.context.addFile(temp_dir, recursive=True)
            broadcast_dir = rdd.context.broadcast(temp_dir)

            def _generate(partition):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                temp_dir_ = broadcast_dir.value
                model_path = SparkFiles.get(temp_dir_)
                model = loader.from_pretrained(model_path).to(device)
                generations = []

                for batch in partition:
                    inputs = tokenizer(batch, **tokenizer_kwargs)
                    outputs = model.generate(**inputs, **kwargs)
                    generations.extend(outputs.cpu().numpy())
                return generations

            def _generate_with_indices(partition):
                data, indices = zip(*partition)
                generations = _generate(data)
                return zip(generations, indices)

            return self._call_and_collect(rdd, _generate, _generate_with_indices)

    def _call_and_collect(
        self, rdd: RDD, predict_func: Callable, predict_with_indices_func: Callable
    ) -> List[np.ndarray]:
        if self.num_workers and self.num_workers > 1:
            rdd = rdd.zipWithIndex().repartition(self.num_workers)
            predictions_and_indices = rdd.mapPartitions(
                partial(predict_with_indices_func)
            )
            predictions_sorted_by_index = predictions_and_indices.sortBy(lambda x: x[1])
            return predictions_sorted_by_index.map(lambda x: x[0]).collect()
        else:
            rdd = rdd.coalesce(1)
            return rdd.mapPartitions(partial(predict_func)).collect()

    def save(self, dir_path: str, overwrite: bool = False):
        path = Path(dir_path)
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"{dir_path} exists and is not a directory.")
            if overwrite:
                shutil.rmtree(path)
            else:
                raise FileExistsError(
                    f"{dir_path} already exists. Use `overwrite=True` to replace it."
                )

        self._master_network.save_pretrained(dir_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(dir_path)

    def __call__(self, **kwargs):
        batched_inputs = {k: v for k, v in kwargs.items()}
        rdd = to_simple_rdd(batched_inputs)
        loader = self.loader

        with tempfile.TemporaryDirectory() as temp_base:
            # Use a unique subdirectory to avoid Spark caching collisions
            unique_model_dir = Path(temp_base) / f"model_{uuid.uuid4().hex}"
            unique_model_dir.mkdir()
            self._master_network.save_pretrained(unique_model_dir)
            rdd.context.addFile(str(unique_model_dir), recursive=True)
            broadcast_dir = rdd.context.broadcast(unique_model_dir.name)

            def _predict_tokenized_partition(partition):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_path = SparkFiles.get(broadcast_dir.value)
                model = loader.from_pretrained(model_path).to(device)
                model.eval()
                predictions = []

                with torch.no_grad():
                    for batch in partition:
                        inputs = {
                            k: torch.tensor(v).unsqueeze(0)
                            if not torch.is_tensor(v)
                            else v.unsqueeze(0)
                            for k, v in batch.items()
                        }
                        outputs = model(**inputs)
                        predictions.extend(outputs.logits.detach().cpu().numpy())

                return predictions

            def _predict_tokenized_partition_with_indices(partition):
                data, indices = zip(*partition)
                preds = _predict_tokenized_partition(data)
                return zip(preds, indices)

            return self._call_and_collect(
                rdd,
                _predict_tokenized_partition,
                _predict_tokenized_partition_with_indices,
            )


class SparkFormerWorker:
    def __init__(
        self,
        master_optimizer,
        master_metrics,
        temp_dir,
        tokenizer,
        tokenizer_kwargs,
        loader,
        batch_size,
        epochs,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.temp_dir = temp_dir
        self.loader = loader
        self.master_optimizer = master_optimizer
        self.master_metrics = master_metrics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, data_iterator):
        temp_dir = self.temp_dir.value
        model_path = SparkFiles.get(temp_dir)
        model = self.loader.from_pretrained(model_path).to(self.device)
        model.train()

        optimizer = self.master_optimizer(model.parameters())

        if self.loader.__name__ == AutoModelForSequenceClassification.__name__:
            x_train, y_train = zip(*data_iterator)
            tokenized = self.tokenizer(list(x_train), **self.tokenizer_kwargs).to(
                self.device
            )
            y_train = torch.tensor(y_train).to(self.device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            history = self.run_classification_training_loop(
                model, input_ids, attention_mask, y_train, optimizer
            )

        elif self.loader.__name__ == AutoModelForTokenClassification.__name__:
            x_train, y_train = zip(*data_iterator)

            tokenized = self.tokenizer(list(x_train), **self.tokenizer_kwargs)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            max_len = input_ids.shape[1]

            y_train_padded = pad_labels(y_train, max_len, -100)
            labels = torch.tensor(y_train_padded)
            history = self.run_classification_training_loop(
                model, input_ids, attention_mask, labels, optimizer
            )
        elif self.loader.__name__ == AutoModelForCausalLM.__name__:
            x_train = list(data_iterator)
            tokenized = self.tokenizer(x_train, **self.tokenizer_kwargs)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.clone()
            history = self.run_classification_training_loop(
                model, input_ids, attention_mask, labels, optimizer
            )
        else:
            raise ValueError(f"Unsupported loader: {self.loader.__name__}")
        updated_state = model.state_dict()
        yield [updated_state, history]

    def run_classification_training_loop(
        self, model, input_ids, attention_mask, y_train, optimizer
    ):
        dataset = TensorDataset(input_ids, attention_mask, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        total_loss = 0.0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                input_ids_batch, attn_mask_batch, labels_batch = [
                    t.to(self.device) for t in batch
                ]

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids_batch,
                    attention_mask=attn_mask_batch,
                    labels=labels_batch,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            total_loss += epoch_loss
            print(f"Epoch {epoch + 1} - Loss: {epoch_loss / len(dataloader):.4f}")

        history = {"loss": total_loss / len(dataloader)}
        return history


def save_and_zip_model(model, temp_dir):
    model.save_pretrained(temp_dir)
    zip_path = shutil.make_archive(temp_dir, "zip", temp_dir)
    return zip_path
