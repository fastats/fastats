
import numpy as np
from pytest import approx

from fastats import single_pass
from fastats.maths import logistic


def test_basic_sanity():
    data = np.arange(-2, 2, dtype='float32')

    result = single_pass(data, value=logistic)

    assert result[0] == approx(0.11920292)
    assert result[1] == approx(0.26894143)
    assert result[2] == approx(0.5)
    assert result[3] == approx(0.7310586)


if __name__ == '__main__':
    import pytest
    pytest.main()