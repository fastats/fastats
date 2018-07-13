
import numpy as np


def ewma(x, halflife):
    """
    Exponentially Weighted Moving Average

    >>> ewma(np.arange(5), halflife=2)
    array([ 0.        ,  0.58578644,  1.22654092,  1.91911977,  2.65947261])
    """
    decay_coefficient = np.exp(np.log(0.5) / halflife)
    out = np.empty_like(x, dtype=np.float64)

    for i in range(out.shape[0]):
        if i == 0:
            out[i] = x[i]
            sum_prior = 1
        else:
            sum_i = sum_prior + np.power(decay_coefficient, i)
            out[i] = (decay_coefficient * out[i - 1] * sum_prior + x[i]) / sum_i
            sum_prior = sum_i

    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
