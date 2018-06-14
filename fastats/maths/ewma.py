
import numpy as np


def ewma(x, halflife):
    """
    Exponential weighted moving average
    """
    factor = np.exp(np.log(0.5) / halflife)
    out = np.empty_like(x, dtype=np.float64)

    for i in range(out.shape[0]):
        if i == 0:
            out[i, :] = x[i, :]
            sum_prior = 1
        else:
            sum_i = sum_prior + np.power(factor, i)
            out[i, :] = (factor * out[i - 1, :] * sum_prior + x[i, :]) / sum_i
            sum_prior = sum_i

    return out

if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
