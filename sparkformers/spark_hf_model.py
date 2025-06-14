import tempfile


class SparkHFModel:
    def __init__(self, model, num_workers=None, batch_size=32, tokenizer=None,
                 tokenizer_kwargs=None, loader=None,
                 *args,
                 **kwargs):
        """
        SparkHFModel
        Class for distributed training of Hugging Face models on RDDs.
        :param model_name_or_path: path to pre-trained model or model identifier from huggingface.co/models
        :param tokenizer_name_or_path: path to pre-trained tokenizer or tokenizer identifier (if different from model)
        :param mode: String, choose from 'asynchronous', 'synchronous' and 'hogwild'
        :param frequency: String, either 'epoch' or 'batch'
        :param parameter_server_mode: String, either 'http' or 'socket'
        :param num_workers: int, number of workers used for training
        :param batch_size: batch size used for training and inference
        :param port: port used in case of 'http' parameter server mode
        """
        super().__init__(model, mode=mode, frequency=frequency, parameter_server_mode=parameter_server_mode,
                         num_workers=num_workers, batch_size=batch_size, port=port, *args, **kwargs)
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.tf_loader = loader
        self.tokenizer_kwargs = tokenizer_kwargs or {}

    def _fit(self, rdd: RDD, **kwargs):
        """
        Train a Hugging Face model on an RDD.
        """
        optimizer = self.master_optimizer
        loss = self.master_loss
        metrics = self.master_metrics
        custom = self.custom_objects
        train_config = kwargs
        serialized_optimizer = serialize_optimizer(optimizer)

        with tempfile.TemporaryDirectory() as temp_dir:
            self._master_network.save_pretrained(temp_dir)
            rdd.context.addFile(temp_dir, recursive=True)
            temp_dir = rdd.context.broadcast(temp_dir)

            worker = SparkHFWorker(None, None, train_config,
                                   serialized_optimizer, loss, metrics, custom, temp_dir=temp_dir, tokenizer=self.tokenizer,
                                   tokenizer_kwargs=self.tokenizer_kwargs,
                                   loader=self.tf_loader)
            training_outcomes = rdd.mapPartitions(worker.train).collect()
            new_parameters = self._master_network.get_weights()
            number_of_sub_models = len(training_outcomes)
            for training_outcome in training_outcomes:
                grad, history = training_outcome
                self.training_histories.append(history)
                weighted_grad = divide_by(grad, number_of_sub_models)
                new_parameters = subtract_params(new_parameters, weighted_grad)
            self._master_network.set_weights(new_parameters)

    def _predict(self, rdd: RDD) -> List[np.ndarray]:
        """
        Perform distributed inference with the Hugging Face model.
        """
        tokenizer = self.tokenizer
        loader = self.tf_loader
        tokenizer_kwargs = self.tokenizer_kwargs
        with tempfile.TemporaryDirectory() as temp_dir:
            _zip_path = save_and_zip_model(self._master_network, temp_dir)
            rdd.context.addFile(_zip_path)

            def _predict(partition):
                zip_path = SparkFiles.get(_zip_path)
                model_dir = tempfile.mkdtemp()

                shutil.unpack_archive(zip_path, model_dir)

                model = loader.from_pretrained(model_dir, local_files_only=True)

                predictions = []
                for batch in partition:
                    inputs = tokenizer(batch, **tokenizer_kwargs, return_tensors="tf")
                    outputs = model(**inputs)
                    predictions.extend(outputs.logits.numpy())
                shutil.rmtree(model_dir)
                return predictions

            def _predict_with_indices(partition):
                data, indices = zip(*partition)
                zip_path = SparkFiles.get(_zip_path)
                model_dir = tempfile.mkdtemp()

                shutil.unpack_archive(zip_path, model_dir)

                model = loader.from_pretrained(model_dir, local_files_only=True)
                predictions = []
                for batch in data:
                    inputs = tokenizer(batch, **tokenizer_kwargs, return_tensors="tf")
                    outputs = model(**inputs)
                    predictions.extend(outputs.logits.numpy())
                return zip(predictions, indices)

            return self._call_and_collect(rdd, _predict, _predict_with_indices)

    def _evaluate(self, rdd: RDD, **kwargs) -> List[np.ndarray]:
        """
        Perform distributed evaluation with the Hugging Face model.
        """
        # TODO: forgive me for violating Liskov's substitution principle here, but I'm not sure how to implement this
        logging.info(f"We're not currently supporting distributed evaluation for Hugging Face models, as the logic "
                     f"gets a bit more gnarly than for Keras models since the model can be predictive or generative, "
                     f"and there isn't a native `evaluate` method on HuggingFace models."
                     f" Maybe in a future release.")
        return []

    def generate(self, data: Union[RDD, np.array], **kwargs) -> List[np.ndarray]:
        """Perform distributed generation with the model"""
        if isinstance(data, (np.ndarray,)):
            from pyspark.sql import SparkSession
            sc = SparkSession.builder.getOrCreate().sparkContext
            data = sc.parallelize(data)
        return self._generate(data, **kwargs)

    def _generate(self, rdd: RDD, **kwargs) -> List[np.ndarray]:
        """
        Perform distributed generation with the Hugging Face model, for generative models
        """
        from transformers import TFAutoModelForSequenceClassification
        if self.tf_loader.__name__ == TFAutoModelForSequenceClassification.__name__:
            raise ValueError("This method is only for causal language models, not classification models.")
        tokenizer = self.tokenizer
        loader = self.tf_loader
        tokenizer_kwargs = self.tokenizer_kwargs
        with tempfile.TemporaryDirectory() as temp_dir:
            _zip_path = save_and_zip_model(self._master_network, temp_dir)
            rdd.context.addFile(_zip_path)

            def _generate(partition):
                zip_path = SparkFiles.get(_zip_path)
                model_dir = tempfile.mkdtemp()

                shutil.unpack_archive(zip_path, model_dir)

                model = loader.from_pretrained(model_dir, local_files_only=True)

                generations = []
                for batch in partition:
                    inputs = tokenizer(batch, **tokenizer_kwargs, return_tensors="tf")
                    outputs = model.generate(**inputs, **kwargs)
                    generations.extend(outputs)
                shutil.rmtree(model_dir)
                return generations

            def _generate_with_indices(partition):
                data, indices = zip(*partition)
                zip_path = SparkFiles.get(_zip_path)
                model_dir = tempfile.mkdtemp()

                shutil.unpack_archive(zip_path, model_dir)

                model = loader.from_pretrained(model_dir, local_files_only=True)
                generations = []
                for batch in data:
                    inputs = tokenizer(batch, **tokenizer_kwargs, return_tensors="tf")
                    outputs = model.generate(**inputs, **kwargs)
                    generations.extend(outputs)
                return zip(generations, indices)

            return self._call_and_collect(rdd, _generate, _generate_with_indices)

    def save(self, file_name: str, overwrite: bool = False, to_hadoop: bool = False):
        """
        Save a Hugging Face model.
        """
        if not file_name.endswith(".h5"):
            raise ValueError("File name must end with '.h5'")

        if overwrite and not to_hadoop and Path(file_name).exists():
            Path(file_name).unlink()

        self._master_network.save_pretrained(file_name)

        if to_hadoop:
            # Logic for saving to Hadoop (not implemented)
            raise NotImplementedError("Saving to Hadoop needs to be implemented.")

    def __call__(self, *args, **kwargs):
        from pyspark.sql import SparkSession
        import numpy as np
        sc = SparkSession.builder.getOrCreate().sparkContext

        inputs_list = [{key: kwargs[key][i] for key in kwargs} for i in
                       range(len(next(iter(kwargs.values()))))]
        rdd = sc.parallelize(inputs_list)

        loader = self.tf_loader
        with tempfile.TemporaryDirectory() as temp_dir:
            _zip_path = save_and_zip_model(self._master_network, temp_dir)
            rdd.context.addFile(_zip_path)

            def _call(partition):
                zip_path = SparkFiles.get(_zip_path)
                model_dir = tempfile.mkdtemp()
                shutil.unpack_archive(zip_path, model_dir)
                model = loader.from_pretrained(model_dir, local_files_only=True)
                partition_results = __call(partition, model)
                shutil.rmtree(model_dir)
                return partition_results

            def _call_with_indices(partition):
                zip_path = SparkFiles.get(_zip_path)
                model_dir = tempfile.mkdtemp()
                shutil.unpack_archive(zip_path, model_dir)
                model = loader.from_pretrained(model_dir, local_files_only=True)
                data, indices = zip(*partition)
                partition_results = __call(data, model)
                return zip(partition_results, indices)

        def __call(partition, model):
            results = []
            for sample in partition:
                sample_dict = {key: np.array([value]) for key, value in sample.items()}
                outputs = model(**sample_dict)
                if hasattr(outputs, "logits"):
                    results.append(outputs.logits.numpy())
                elif hasattr(outputs, "sequences"):
                    results.append(outputs.sequences.numpy())
                else:
                    results.append(outputs.numpy())
            return results

        return self._call_and_collect(rdd, _call, _call_with_indices)




class SparkHFWorker:
    def __init__(self, json, parameters, train_config, master_optimizer,
                 master_loss, master_metrics, custom_objects, temp_dir, tokenizer, tokenizer_kwargs, loader):
        super().__init__(json, parameters, train_config, master_optimizer,
                         master_loss, master_metrics, custom_objects)
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.temp_dir = temp_dir
        self.loader = loader

    def train(self, data_iterator):
        """Train a Huggingface model on a worker
        """
        from transformers import TFAutoModelForSequenceClassification, TFAutoModelForCausalLM, TFAutoModelForTokenClassification

        temp_dir = self.temp_dir.value
        config = SparkFiles.get(temp_dir)

        self.model = self.loader.from_pretrained(config, local_files_only=True)
        optimizer = deserialize_optimizer(self.master_optimizer)
        self.model.compile(optimizer=optimizer,
                           loss=self.master_loss, metrics=self.master_metrics)
        weights_before_training = self.model.get_weights()
        if self.loader.__name__ == TFAutoModelForSequenceClassification.__name__:
            x_train, y_train = zip(*data_iterator)
            x_train = self.tokenizer(list(x_train), **self.tokenizer_kwargs, return_tensors="tf")
            y_train = np.array(y_train)
            history = self.model.fit(dict(x_train), y_train, **self.train_config)
        elif self.loader.__name__ == TFAutoModelForTokenClassification.__name__:
            x_train, y_train = zip(*data_iterator)
            x_train = self.tokenizer(list(x_train), **self.tokenizer_kwargs, return_tensors="tf")
            max_length = max(len(seq) for seq in x_train['input_ids'])
            y_train_padded = pad_labels(y_train, max_length, -100)
            y_train = np.array(y_train_padded)
            history = self.model.fit(dict(x_train), y_train, **self.train_config)
        elif self.loader.__name__ == TFAutoModelForCausalLM.__name__:
            x_train = self.tokenizer(list(data_iterator), **self.tokenizer_kwargs, return_tensors="tf")
            x_train, y_train = x_train['input_ids'][:, :-1], x_train['input_ids'][:, 1:]
            history = self.model.fit(x_train, y_train, **self.train_config)
        else:
            raise ValueError(f"Unsupported loader type: {self.loader.__name__}")

        weights_after_training = self.model.get_weights()
        deltas = subtract_params(
            weights_before_training, weights_after_training)
        if history:
            yield [deltas, history.history]
        else:
            yield [deltas, None]