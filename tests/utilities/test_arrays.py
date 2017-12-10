
import numpy as np

from fastats.utilities import is_square


def test_is_square():
    a = np.array([
        [1, 2],
        [3, 4]
    ])

    assert is_square(a)

    b = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    assert not is_square(b)
