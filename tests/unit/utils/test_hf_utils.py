import numpy as np
import pytest
from sparkformers.utils.hf_utils import pad_labels


@pytest.mark.parametrize(
    "labels,max_length,pad_id,expected",
    [
        ([[1, 2], [3]], 4, -100, [[1, 2, -100, -100], [3, -100, -100, -100]]),
        ([[0, 1, 2, 3]], 4, -1, [[0, 1, 2, 3]]),
        ([[]], 3, -1, [[-1, -1, -1]]),
        ([[5], [6, 7, 8]], 4, 0, [[5, 0, 0, 0], [6, 7, 8, 0]]),
        ([[1, 2]], 5, 999, [[1, 2, 999, 999, 999]]),
    ],
    ids=[
        "basic_padding",
        "no_padding",
        "empty_sequences",
        "mixed_lengths",
        "large_pad_id",
    ],
)
def test_pad_labels(labels, max_length, pad_id, expected):
    result = pad_labels(labels, max_length, pad_id)
    np.testing.assert_array_equal(result, np.array(expected))
