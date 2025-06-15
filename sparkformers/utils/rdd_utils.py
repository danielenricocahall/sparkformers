from pyspark.core.rdd import RDD
from pyspark.core.context import SparkContext
import numpy as np


def to_simple_rdd(
    features: np.ndarray, labels: np.ndarray, sc: SparkContext | None = None
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
    pairs = [(x, y) for x, y in zip(features, labels)]
    return sc.parallelize(pairs)
