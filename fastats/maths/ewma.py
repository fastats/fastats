
import numpy as np

from numba import jit
from numba import float64


@jit((float64[:], float64), nopython=True, nogil=True)
def ewma(arr_in, halflife):
    r"""Exponentialy weighted moving average specified by a decay ``halflife``
    to provide better adjustments for small windows via:

        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    halflife : float64
        Specify decay in terms of half-life,
        :math:`\alpha = 1 - exp(log(0.5) / halflife),\text{ for } halflife > 0`

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(halflife=2, adjust=True).mean()
    >>> np.isclose(ewma(a, 2), exp.values.ravel())  # isclose for floating point err
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    decay_coefficient = 1 - np.exp(np.log(0.5) / halflife)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1 - decay_coefficient)**i
        ewma_old = ewma_old * (1 - decay_coefficient) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma


@jit((float64[:], float64), nopython=True, nogil=True)
def _ewma_infinite_hist(arr_in, halflife):
    r"""Exponentialy weighted moving average specified by a decay ``halflife``
    assuming infinite history via the recursive form:

        (2) (i)  y[0] = x[0]; and
            (ii) y[t] = a*x[t] + (1-a)*y[t-1] for t>0.

    This method is less accurate that ``_ewma`` but
    much faster:

        In [1]: import numpy as np, bars
           ...: arr = np.random.random(100000)
           ...: %timeit bars._ewma(arr, 2)
           ...: %timeit bars._ewma_infinite_hist(arr, 2)
        3.74 ms ± 60.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        262 µs ± 1.54 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    Parameters
    ----------
    arr_in : np.ndarray, float64
        A single dimenisional numpy array
    halflife : float64
        Specify decay in terms of half-life,
        :math:`\alpha = 1 - exp(log(0.5) / halflife),\text{ for } halflife > 0`

    Returns
    -------
    np.ndarray
        The EWMA vector, same length / shape as ``arr_in``

    Examples
    --------
    >>> import pandas as pd
    >>> a = np.arange(5, dtype=float)
    >>> exp = pd.DataFrame(a).ewm(halflife=2, adjust=False).mean()
    >>> np.array_equal(_ewma_infinite_hist(a, 2), exp.values.ravel())
    True
    """
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=float64)
    decay_coefficient = 1 - np.exp(np.log(0.5) / halflife)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = arr_in[i] * decay_coefficient + ewma[i - 1] * (1 - decay_coefficient)
    return ewma


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
