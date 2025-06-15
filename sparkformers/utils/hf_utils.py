import shutil
import tempfile

import numpy as np


def pad_labels(labels: list[list[int]], max_length: int, pad_token_label_id: int):
    processed_labels = []
    for label_seq in labels:
        padded_seq = label_seq + [pad_token_label_id] * (max_length - len(label_seq))
        processed_labels.append(padded_seq)
    return np.array(processed_labels)


def load_model_from_zip(zip_path, loader):
    model_dir = tempfile.mkdtemp()
    shutil.unpack_archive(zip_path, model_dir)
    model = loader.from_pretrained(model_dir).eval()
    return model, model_dir
