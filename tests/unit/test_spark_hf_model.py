from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModelForCausalLM, \
    TFAutoModelForTokenClassification
def test_training_huggingface_classification(spark_context):
    batch_size = 5
    epochs = 1
    num_workers = 2

    newsgroups = fetch_20newsgroups(subset='train')
    x = newsgroups.data[:50]  # Limit the data size for the test
    y = newsgroups.target[:50]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.5)

    model_name = 'albert-base-v2'  # use the smallest classification model for testing

    rdd = to_simple_rdd(spark_context, x_train, y_train)

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(np.unique(y_encoded)))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_kwargs = {'padding': True, 'truncation': True}

    model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    spark_model = SparkHFModel(model, num_workers=num_workers, mode=Mode.SYNCHRONOUS, tokenizer=tokenizer,
                               tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForSequenceClassification)

    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)

    # Run inference on trained Spark model
    predictions = spark_model.predict(spark_context.parallelize(x_test))
    samples = tokenizer(x_test, padding=True, truncation=True, return_tensors="tf")
    # Evaluate results
    assert all(np.isclose(x, y, 0.01).all() for x, y in zip(predictions, spark_model.master_network(**samples)[0]))


def test_training_huggingface_generation(spark_context):
    batch_size = 5
    epochs = 1
    num_workers = 2

    newsgroups = fetch_20newsgroups(subset='train')
    x = newsgroups.data[:60]

    x_train, x_test = train_test_split(x, test_size=0.2)

    model_name = 'sshleifer/tiny-gpt2'  # use the smaller generative model for testing

    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_kwargs = {'max_length': 15, 'padding': True, 'truncation': True}

    model.compile(optimizer=SGD(), metrics=['accuracy'], loss='sparse_categorical_crossentropy')

    spark_model = SparkHFModel(model, num_workers=num_workers, mode=Mode.SYNCHRONOUS, tokenizer=tokenizer,
                               tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForCausalLM)
    rdd = spark_context.parallelize(x_train)
    rdd_test = spark_context.parallelize(x_test)
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)
    generations = spark_model.generate(rdd_test, max_length=20, num_return_sequences=1)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generations]
    assert generated_texts == [tokenizer.decode(output, skip_special_tokens=True) for output in
                               spark_model.master_network.generate(
                                   **tokenizer(x_test, max_length=15, padding=True, truncation=True,
                                               return_tensors="tf"), num_return_sequences=1)]


def test_training_huggingface_token_classification(spark_context):
    batch_size = 5
    epochs = 2
    num_workers = 2
    model_name = 'hf-internal-testing/tiny-bert-for-token-classification'  # use the smallest classification model for testing

    model = TFAutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = load_dataset("conll2003", split='train[:5%]', trust_remote_code=True)
    dataset = dataset.map(tokenize_and_align_labels, batched=True)

    x = dataset['tokens']
    y = dataset['labels']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    rdd = to_simple_rdd(spark_context, x_train, y_train)

    tokenizer_kwargs = {'padding': True, 'truncation': True, 'is_split_into_words': True}

    model.compile(optimizer=Adam(learning_rate=5e-5), metrics=['accuracy'])
    spark_model = SparkHFModel(model, num_workers=num_workers, mode=Mode.SYNCHRONOUS, tokenizer=tokenizer,
                               tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForTokenClassification)

    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)

    # Run inference on trained Spark model
    samples = tokenizer(x_test, **tokenizer_kwargs, return_tensors="tf")
    distributed_predictions = spark_model(**samples)
    regular_predictions = spark_model.master_network(**samples)
    # Evaluate results
    assert all(np.isclose(x, y, 0.01).all() for x, y in zip(distributed_predictions[0], regular_predictions[0]))