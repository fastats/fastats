
from fastats.maths.ewma import ewma
import pandas as pd
import numpy as np


def test_ewma_basic_sanity():
    random_data = np.random.random((100, 100))
    df = pd.DataFrame(random_data)

    pandas_result = df.ewm(halflife=10).mean()

    ewma_result = ewma(random_data, halflife=10)
    fast_result = pd.DataFrame(ewma_result)

    pd.testing.assert_frame_equal(pandas_result, fast_result)

if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
