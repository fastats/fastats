
import numpy as np


def ewma(x, halflife):
    """
    Exponentially Weighted Moving Average
    It is expected that the numbers passed as x will be finite, halflife is
    expected to be a finite, positive number.
    >>> ewma(np.arange(5), halflife=2)
    array([0.        , 0.58578644, 1.22654092, 1.91911977, 2.65947261])
    """
    assert np.isfinite(halflife) and halflife > 0

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


def ewma_2d(x, halflife):
    """
    Exponentially Weighted Moving Average,
    optimised for 2D data sets.
    """
    assert x.ndim == 2
    assert np.isfinite(halflife) and halflife > 0

    decay_coefficient = np.exp(np.log(0.5) / halflife)
    out = np.empty_like(x, dtype=np.float64)

    for i in range(out.shape[0]):
        if i == 0:
            out[i, :] = x[i, :]
            sum_prior = 1
            first_weight = 1
        else:
            first_weight *= decay_coefficient
            sum_i = sum_prior + first_weight

            for j in range(x.shape[1]):
                out[i, j] = (decay_coefficient * out[i - 1, j] * sum_prior + x[i, j]) / sum_i

            sum_prior = sum_i

    return out


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
