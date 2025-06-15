import shutil
import tempfile
import logging
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
from sparkformers.utils.torch_utils import divide_by, subtract_params, get_param_diff
from sparkformers.utils.hf_utils import pad_labels, load_model_from_zip

logger = logging.getLogger(__name__)


class SparkFormer:
    def __init__(
        self,
        model,
        tokenizer,
        loader,
        optimizer_fn,
        loss_fn,
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
        self.master_loss = loss_fn
        self.master_metrics = metrics or {}
        self.custom_objects = custom_objects or {}
        self.num_workers = num_workers
        self.training_histories = []

    def train(self, data: np.ndarray, labels: np.ndarray | None = None, **kwargs):
        rdd = to_simple_rdd(data, labels)
        if self.num_workers > 1:
            rdd = rdd.repartition(self.num_workers)
        optimizer_fn = self.master_optimizer
        loss_fn = self.master_loss
        metrics = self.master_metrics
        batch_size = kwargs.get("batch_size", 32)
        epochs = kwargs.get("epochs", 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            self._master_network.save_pretrained(temp_dir)
            rdd.context.addFile(temp_dir, recursive=True)
            broadcast_dir = rdd.context.broadcast(temp_dir)

            worker = SparkFormerWorker(
                optimizer_fn,
                loss_fn,
                metrics,
                temp_dir=broadcast_dir,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                loader=self.loader,
                batch_size=batch_size,
                epochs=epochs,
            )

            training_outcomes = rdd.mapPartitions(worker.train).collect()
            new_state = self._master_network.state_dict()
            number_of_sub_models = len(training_outcomes)

            for delta, history in training_outcomes:
                self.training_histories.append(history)
                weighted_grad = divide_by(delta, number_of_sub_models)
                new_state = subtract_params(new_state, weighted_grad)

            self._master_network.load_state_dict(new_state)

    def predict(self, data: Iterable) -> List[np.ndarray]:
        rdd = to_simple_rdd(data)
        tokenizer = self.tokenizer
        loader = self.loader
        tokenizer_kwargs = self.tokenizer_kwargs

        with tempfile.TemporaryDirectory() as temp_dir:
            _zip_path = save_and_zip_model(self._master_network, temp_dir)
            rdd.context.addFile(_zip_path)

            def _predict(partition):
                model, model_dir = load_model_from_zip(
                    SparkFiles.get(_zip_path), loader
                )
                predictions = []
                for batch in partition:
                    inputs = tokenizer(batch, **tokenizer_kwargs, return_tensors="pt")
                    outputs = model(**inputs)
                    predictions.extend(outputs.logits.detach().cpu().numpy())
                shutil.rmtree(model_dir)
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
            _zip_path = save_and_zip_model(self._master_network, temp_dir)
            rdd.context.addFile(_zip_path)

            def _generate(partition):
                model, model_dir = load_model_from_zip(
                    SparkFiles.get(_zip_path), loader
                )
                generations = []

                for batch in partition:
                    inputs = tokenizer(batch, **tokenizer_kwargs, return_tensors="pt")
                    outputs = model.generate(**inputs, **kwargs)
                    generations.extend(outputs.cpu().numpy())
                shutil.rmtree(model_dir)
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

    def __call__(self, *args, **kwargs):
        from pyspark.sql import SparkSession

        sc = (
            SparkSession.builder.getOrCreate().sparkContext  # ty: ignore[possibly-unbound-attribute]
        )
        inputs_list = [
            {key: kwargs[key][i] for key in kwargs}
            for i in range(len(next(iter(kwargs.values()))))
        ]
        rdd = sc.parallelize(inputs_list)
        loader = self.loader

        with tempfile.TemporaryDirectory() as temp_dir:
            _zip_path = save_and_zip_model(self._master_network, temp_dir)
            rdd.context.addFile(_zip_path)

            def _call(partition):
                model, model_dir = load_model_from_zip(
                    SparkFiles.get(_zip_path), loader
                )
                tokenizer = self.tokenizer
                tokenizer_kwargs = self.tokenizer_kwargs
                outputs = process_partition(
                    partition, tokenizer, model, tokenizer_kwargs
                )
                shutil.rmtree(model_dir)
                return outputs

            def _call_with_indices(partition):
                data, indices = zip(*partition)
                outputs = _call(data)
                return zip(outputs, indices)

            def process_partition(data, tokenizer, model, tokenizer_kwargs):
                outputs_list = []
                for sample in data:
                    # Tokenize if raw input
                    if isinstance(sample, str) or isinstance(sample, list):
                        inputs = tokenizer(
                            sample, return_tensors="pt", **tokenizer_kwargs
                        )
                    elif isinstance(sample, dict):
                        # If it's already tokenized, assume it's numeric data
                        inputs = {
                            k: torch.as_tensor(v).unsqueeze(0)
                            if not torch.is_tensor(v)
                            else v.unsqueeze(0)
                            for k, v in sample.items()
                        }
                    else:
                        raise ValueError(f"Unexpected sample type: {type(sample)}")

                    with torch.no_grad():
                        outputs = model(**inputs)

                    if hasattr(outputs, "logits"):
                        outputs_list.append(outputs.logits.detach().cpu().numpy())
                    elif hasattr(outputs, "sequences"):
                        outputs_list.append(outputs.sequences.detach().cpu().numpy())
                    else:
                        outputs_list.append(outputs.detach().cpu().numpy())

                return outputs_list

        return self._call_and_collect(rdd, _call, _call_with_indices)


class SparkFormerWorker:
    def __init__(
        self,
        master_optimizer,
        master_loss,
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
        self.master_loss = master_loss
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
            tokenized = self.tokenizer(
                list(x_train), **self.tokenizer_kwargs, return_tensors="pt"
            ).to(self.device)
            y_train = torch.tensor(y_train).to(self.device)
            loss_fn = self.master_loss()
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
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
                        input_ids=input_ids_batch, attention_mask=attn_mask_batch
                    )
                    loss = loss_fn(outputs.logits, labels_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                total_loss += epoch_loss
                logger.info(
                    f"Epoch {epoch + 1} - Loss: {epoch_loss / len(dataloader):.4f}"
                )

            history = {"loss": total_loss / len(dataloader)}

        elif self.loader.__name__ == AutoModelForTokenClassification.__name__:
            x_train, y_train = zip(*data_iterator)

            tokenized = self.tokenizer(
                list(x_train), **self.tokenizer_kwargs, return_tensors="pt"
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            max_len = input_ids.shape[1]

            y_train_padded = pad_labels(
                y_train, max_len, -100
            )  # -100 is the default padding label for transformers
            labels = torch.tensor(y_train_padded)

            dataset = TensorDataset(input_ids, attention_mask, labels)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            loss_fn = self.master_loss()
            total_loss = 0.0
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch in dataloader:
                    input_ids_batch, attn_mask_batch, label_batch = [
                        b.to(self.device) for b in batch
                    ]

                    optimizer.zero_grad()
                    outputs = model(
                        input_ids=input_ids_batch, attention_mask=attn_mask_batch
                    )
                    loss = loss_fn(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        label_batch.view(-1),
                    )
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                total_loss += epoch_loss
                logger.info(
                    f"Epoch {epoch + 1} - Loss: {epoch_loss / len(dataloader):.4f}"
                )

            history = {"loss": total_loss / len(dataloader)}
        elif self.loader.__name__ == AutoModelForCausalLM.__name__:
            x_train = list(data_iterator)
            tokenized = self.tokenizer(
                x_train, **self.tokenizer_kwargs, return_tensors="pt"
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            labels = input_ids.clone()
            dataset = TensorDataset(input_ids, attention_mask, labels)

            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            total_loss = 0.0
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for batch in dataloader:
                    input_ids_batch, attn_mask_batch, labels_batch = [
                        b.to(self.device) for b in batch
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

                logger.info(
                    f"Epoch {epoch + 1} - Loss: {epoch_loss / len(dataloader):.4f}"
                )

            history = {"loss": total_loss / len(dataloader)}

        else:
            raise ValueError(f"Unsupported loader: {self.loader.__name__}")

        delta = get_param_diff(model)
        yield [delta, history]


def save_and_zip_model(model, temp_dir):
    model.save_pretrained(temp_dir)
    zip_path = shutil.make_archive(temp_dir, "zip", temp_dir)
    return zip_path
