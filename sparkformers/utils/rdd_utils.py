from typing import Iterable

import torch
from pyspark.core.rdd import RDD
from pyspark.core.context import SparkContext
import numpy as np

from sparkformers.utils.torch_utils import add_params

ModelState = dict[str, torch.Tensor]
History = dict[str, float]
StateAndHistory = tuple[ModelState, History]


def to_simple_rdd(
    features: np.ndarray | Iterable,
    labels: np.ndarray | Iterable | None = None,
    sc: SparkContext | None = None,
) -> RDD:
    """Convert numpy arrays of features and labels into
    an RDD of pairs.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :return: Spark RDD with feature-label pairs
    """
    if sc is None:
        sc = SparkContext.getOrCreate()
    if isinstance(features, dict):
        features = [dict(zip(features, t)) for t in zip(*features.values())]
    if labels is None:
        return sc.parallelize(features)
    pairs = [(x, y) for x, y in zip(features, labels)]
    return sc.parallelize(pairs)


def accumulate_model_parameters_and_history(
    x: StateAndHistory, y: StateAndHistory
) -> StateAndHistory:
    state_dict, history = x
    other_state_dict, other_history = y
    updated_state = add_params(state_dict, other_state_dict)
    combined_history = {k: v + other_history[k] for k, v in history.items()}
    return updated_state, combined_history
