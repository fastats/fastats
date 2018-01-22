import numpy
import pkg_resources
import pytest


# The first version of numpy that broke backwards compat and improved printing.
#
# We set the printing format to legacy to maintain our doctests' compatibility
# with both newer and older versions.
#
# See: https://docs.scipy.org/doc/numpy/release.html#many-changes-to-array-printing-disableable-with-the-new-legacy-printing-mode
#
NUMPY_PRINT_ALTERING_VERSION = pkg_resources.parse_version('1.14.0')


@pytest.fixture(autouse=True)
def add_preconfigured_np(doctest_namespace):
    """
    Fixture executed for every doctest.

    Injects pre-configured numpy into each test's namespace.

    Note that even with this, doctests might fail due to the lack of full
    compatibility when using ``numpy.set_printoptions(legacy='1.13')``.

    Some of the whitespace issues can be fixed by NORMALIZE_WHITESPACE
    doctest option, which is currently set in ``pytest.ini``.

    See: https://github.com/numpy/numpy/issues/10383
    """
    current_version = pkg_resources.parse_version(numpy.__version__)

    if current_version >= NUMPY_PRINT_ALTERING_VERSION:
        numpy.set_printoptions(legacy='1.13')

    doctest_namespace['np'] = numpy
