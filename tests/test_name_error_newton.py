from unittest import skip

import numpy as np

from fastats import single_pass
from fastats.optimise.root_finding import newton_raphson

data = np.array([1, 2, 3])


def my_func(x):
    return x**3 - x - 1


@skip('Waiting on being able to pass multiple args to kwarg substitution')
def test_single_pass_with_transform():
    """ This requires multiple args passed to newton_raphson.
        Scheduled for 2017.2 / 2018.1

        This tests multiple previous issues - by default,
        newton_raphson has an @fs decorator, which means it
        tries to parse all functions and ends up with free
        vars errors.
    """
    data = np.array([0.9, 1.0, 1.1])
    func = single_pass(data, 1e-6,
                       value=newton_raphson,
                       root=my_func,
                       return_callable=True)

    result = func(data)


if __name__ == '__main__':
    import pytest
    pytest.main()
