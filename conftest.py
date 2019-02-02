
import numba
import numpy
import pkg_resources
import pytest
import scipy


@pytest.fixture(autouse=True)
def add_preconfigured_np(doctest_namespace):
    """
    Fixture executed for every doctest.

    Injects pre-configured numpy into each test's namespace.

    Note that even with this, doctests might fail due to the lack of full
    compatibility when using ``numpy.set_printoptions(legacy='1.13')``.

    Some of the whitespace issues can be fixed by ``NORMALIZE_WHITESPACE``
    doctest option, which is currently set in ``pytest.ini``.

    See: https://github.com/numpy/numpy/issues/10383
    """
    current_version = pkg_resources.parse_version(numpy.__version__)
    doctest_namespace['np'] = numpy


def pytest_report_header(config):
    return 'Testing fastats using: Numba {}, NumPy {}, SciPy {}'.format(
        numba.__version__, numpy.__version__, scipy.__version__,
    )

